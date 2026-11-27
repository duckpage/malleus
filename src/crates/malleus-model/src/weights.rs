//! Weight formats (bf16 and the MLS2/3/4/8 codebooks), the linear a load builds from them, and the
//! stacked/fused copies the prefill runs on.

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::{DeviceBuffer, Stream};

pub(crate) const WMMA_MIN_TOKENS: usize = 8;
/// Up to this many tokens the decode weights are read once by a GEMV for all tokens (speculative verify).
pub const GEMV_MAX_T: usize = 6;
/// MLA prefill is absorbed and projected in tiles of this many tokens (the qt/lat buffers are sized for it).
pub(crate) const MLA_PREFILL_TILE: usize = 2048;

/// Context length of the fp16 KV cache rows (V is stored transposed, padded so 32-key tiles stay in bounds).
pub(crate) fn pad128(n: usize) -> usize {
    n.next_multiple_of(128)
}

/// Context length of the fp16 KV cache rows (V is stored transposed, padded so 32-key tiles stay in bounds).
pub(crate) fn kv_ctx(max_ctx: usize) -> usize {
    max_ctx.next_multiple_of(32)
}

/// int8 GEMM variant for t tokens: (kernel, token tile, column tile).
pub(crate) fn i8_kernel(t: usize, glu: bool) -> (&'static str, usize, usize) {
    match (t <= 128, glu) {
        (true, false) => ("gemm_i8_m128", 128, 128),
        (false, false) => ("gemm_i8", 256, 64),
        (true, true) => ("gemm_i8_glu_m128", 128, 128),
        (false, true) => ("gemm_i8_glu", 256, 64),
    }
}

pub enum Weights {
    Bf16(DeviceBuffer),
    /// One sign per weight with an e4m3 scale per 32 (1.25 bpw): the format that makes a 763B backbone resident.
    Mls1 {
        q: DeviceBuffer,
        sc: DeviceBuffer,
    },
    /// 2-bit codes with an fp16 scale per group of 128: the routed experts of Qwen3.8-Flash-Next (MLS2).
    Mls2 {
        q: DeviceBuffer,
        sc: DeviceBuffer,
    },
    Mls4 {
        q: DeviceBuffer,
        sc: DeviceBuffer,
    },
    /// 3-bit codes with an fp16 scale per group of 128 (Ling's routed gate/up, docs/model-ling3-flash.md).
    Mls3 {
        q: DeviceBuffer,
        sc: DeviceBuffer,
    },
    Mls8 {
        q: DeviceBuffer,
        sc: DeviceBuffer,
    },
}

pub struct Linear {
    pub w: Weights,
    /// Optional MLS8 copy used for prefill (T > 1).
    pub prefill: Option<(DeviceBuffer, DeviceBuffer)>,
    pub n: usize,
    pub k: usize,
}

/// Requantizes an MLS8 matrix (same rotation) to MLS4 with round-to-nearest group scales.
pub(crate) fn mls8_to_mls4(q8: &DeviceBuffer, s8: &DeviceBuffer, n: usize, k: usize) -> Result<Linear> {
    let q: Vec<i8> = q8.download(n * k)?;
    let s: Vec<f32> = s8.download(n)?;
    let g = k / malleus_gguf::quant::BLOCK;
    let mut q4 = vec![0u8; n * k / 2];
    let mut s4 = vec![half::f16::ZERO; n * g];
    let threads = std::thread::available_parallelism().map_or(4, |v| v.get());
    let rows = n.div_ceil(threads);
    std::thread::scope(|sc| {
        for ((qc, sc4), (i, q8c)) in q4
            .chunks_mut(rows * k / 2)
            .zip(s4.chunks_mut(rows * g))
            .zip(q.chunks(rows * k).enumerate())
        {
            let s = &s;
            sc.spawn(move || {
                for r in 0..q8c.len() / k {
                    let row_s = s[i * rows + r];
                    for j in 0..g {
                        let grp = &q8c[r * k + j * 128..r * k + (j + 1) * 128];
                        let amax = grp.iter().fold(0i32, |m, &v| m.max((v as i32).abs())) as f32 * row_s;
                        let gs = half::f16::from_f32(amax / 7.0);
                        sc4[r * g + j] = gs;
                        let inv = if gs.to_f32() > 0.0 { row_s / gs.to_f32() } else { 0.0 };
                        for e in 0..64 {
                            let v = |x: i8| ((x as f32 * inv).round().clamp(-8.0, 7.0) as i8 as u8) & 0xF;
                            qc[r * k / 2 + j * 64 + e] = v(grp[2 * e]) | (v(grp[2 * e + 1]) << 4);
                        }
                    }
                }
            });
        }
    });
    Ok(Linear {
        w: Weights::Mls4 {
            q: DeviceBuffer::from_slice(&q4)?,
            sc: DeviceBuffer::from_slice(&s4)?,
        },
        prefill: None,
        n,
        k,
    })
}

/// The rows `ids` of an MLS4 linear as a linear of their own: same layout `mls8_to_mls4` produces ([n, k / 2]
/// nibbles, [n, k / 128] fp16 scales), gathered rather than requantized. This is what the draft head needs when
/// lm_head itself is already MLS4, or there would be no draft head at all.
pub(crate) fn mls4_rows(q4: &DeviceBuffer, s4: &DeviceBuffer, ids: &[u32], n_src: usize, k: usize) -> Result<Linear> {
    let g = k / malleus_gguf::quant::BLOCK;
    let qb: Vec<u8> = q4.download(n_src * k / 2)?;
    let sb: Vec<u16> = s4.download(n_src * g)?;
    let mut q = Vec::with_capacity(ids.len() * k / 2);
    let mut s = Vec::with_capacity(ids.len() * g);
    for &i in ids {
        let r = i as usize;
        ensure!(r < n_src, "draft head row {r} outside lm_head ({n_src} rows)");
        q.extend_from_slice(&qb[r * k / 2..(r + 1) * k / 2]);
        s.extend_from_slice(&sb[r * g..(r + 1) * g]);
    }
    Ok(Linear {
        w: Weights::Mls4 {
            q: DeviceBuffer::from_slice(&q)?,
            sc: DeviceBuffer::from_slice(&s)?,
        },
        prefill: None,
        n: ids.len(),
        k,
    })
}

/// The int8 weights of a linear to run a prefill GEMM on: the tensor itself when it is MLS8, otherwise the separate
/// MLS8 prefill copy. The hyper-connection mixing has no int4 kernel, so a file converted with `--hc-mls4` runs
/// both of its paths on the copy until one lands.
pub(crate) fn mls8_of(l: &Linear) -> Result<(&DeviceBuffer, &DeviceBuffer)> {
    match (&l.w, &l.prefill) {
        (Weights::Mls1 { .. }, _) => bail!("MLS1 weights have no int8 form"),
        (Weights::Mls2 { .. }, Some((q, sc))) => Ok((q, sc)),
        (Weights::Mls8 { q, sc }, _) => Ok((q, sc)),
        (_, Some((q, sc))) => Ok((q, sc)),
        (Weights::Mls4 { .. }, None) => bail!("MLS4 weights and no MLS8 prefill copy"),
        (Weights::Mls3 { .. }, None) => bail!("MLS3 weights have no MLS8 form"),
        (Weights::Mls2 { .. }, None) => bail!("MLS2 weights have no MLS8 form"),
        (Weights::Bf16(_), None) => bail!("bf16 weights have no MLS8 form"),
    }
}

/// MLS4 projections sharing an input stacked by rows for decode: one GEMV writing up to 4 output buffers.
pub(crate) struct Merged {
    pub(crate) q: DeviceBuffer,
    pub(crate) sc: DeviceBuffer,
    pub(crate) n: usize,
    pub(crate) k: usize,
    /// First row of segments 1..3 (N for absent segments).
    pub(crate) bounds: [usize; 3],
}

pub(crate) fn merge_mls4(parts: &[&Linear]) -> Result<Option<Merged>> {
    let k = parts[0].k;
    if parts.len() > 4
        || !k.is_multiple_of(32)
        || parts.iter().any(|p| p.k != k || !matches!(p.w, Weights::Mls4 { .. }))
    {
        return Ok(None);
    }
    let n: usize = parts.iter().map(|p| p.n).sum();
    let g = k / 128;
    let st = Stream::null();
    let mut q = DeviceBuffer::alloc(n * k / 2)?;
    let mut sc = DeviceBuffer::alloc(n * g * 2)?;
    let mut bounds = [n; 3];
    let mut row = 0;
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            bounds[i - 1] = row;
        }
        let Weights::Mls4 { q: pq, sc: ps } = &p.w else {
            unreachable!()
        };
        q.copy_from(row * k / 2, pq, 0, p.n * k / 2, &st)?;
        sc.copy_from(row * g * 2, ps, 0, p.n * g * 2, &st)?;
        row += p.n;
    }
    st.synchronize()?;
    Ok(Some(Merged { q, sc, n, k, bounds }))
}

pub(crate) struct Fused {
    pub(crate) q: DeviceBuffer,
    pub(crate) sc: DeviceBuffer,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) segs: [usize; 2],
    pub(crate) glu: bool,
}

/// Rotated per-row int8 copy of bf16 tensors stacked by rows and zero-padded to `rows` (a prefill-only Linear).
pub(crate) fn int8_rows(
    src: &dyn TensorSource,
    names: &[String],
    n_each: usize,
    k: usize,
    rows: usize,
    seed: u64,
) -> Result<Linear> {
    use malleus_gguf::quant;
    ensure!(names.len() * n_each <= rows, "too many rows");
    let signs = quant::hadamard_signs(seed);
    let mut q = vec![0i8; rows * k];
    let mut sc = vec![0f32; rows];
    for (i, name) in names.iter().enumerate() {
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.dtype == DType::BF16 && info.size == n_each * k * 2,
            "{name}: expected bf16 [{n_each}, {k}]"
        );
        let raw = src.read_all(name)?;
        for r in 0..n_each {
            let mut row: Vec<f32> = raw[r * k * 2..(r + 1) * k * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::bf16::from_le_bytes(*b).to_f32())
                .collect();
            quant::rotate_row(&mut row, &signs);
            let o = i * n_each + r;
            sc[o] = quant::quantize_row_int8(&row, &mut q[o * k..(o + 1) * k]);
        }
    }
    let qb = DeviceBuffer::from_slice(&q)?;
    let sb = DeviceBuffer::from_slice(&sc)?;
    Ok(Linear {
        w: Weights::Mls8 {
            q: DeviceBuffer::from_slice(&q)?,
            sc: DeviceBuffer::from_slice(&sc)?,
        },
        prefill: Some((qb, sb)),
        n: rows,
        k,
    })
}

/// Moves the MLS8 prefill copies of `parts` into one fused weight (None if any part has no copy).
pub(crate) fn fuse(parts: &mut [&mut Linear], glu: bool) -> Result<Option<Fused>> {
    // Without a merged int8 weight the caller falls back to one GEMM per projection, which is what the dense W4A4
    // path needs. The per-tensor MLS8 copies stay, so the GEMMs whose input transform only rot_quant_row can do
    // keep using them.
    if dense_w4a4() || parts.iter().any(|p| p.prefill.is_none()) {
        return Ok(None);
    }
    let k = parts[0].k;
    ensure!(parts.iter().all(|p| p.k == k), "fused projections must share K");
    let n: usize = parts.iter().map(|p| p.n).sum();
    let st = Stream::null();
    let mut q = DeviceBuffer::alloc(n * k)?;
    let mut sc = DeviceBuffer::alloc(n * 4)?;
    if glu {
        ensure!(
            parts.len() == 2 && parts[0].n == parts[1].n && parts[0].n.is_multiple_of(16),
            "bad gate/up"
        );
        for j in 0..parts[0].n / 16 {
            for (h, p) in parts.iter().enumerate() {
                let (pq, ps) = p.prefill.as_ref().unwrap();
                let dst = 32 * j + 16 * h;
                q.copy_from(dst * k, pq, 16 * j * k, 16 * k, &st)?;
                sc.copy_from(dst * 4, ps, 16 * j * 4, 64, &st)?;
            }
        }
    } else {
        let mut row = 0;
        for p in parts.iter() {
            ensure!(p.n.is_multiple_of(128), "fused segments must be multiples of 128 rows");
            let (pq, ps) = p.prefill.as_ref().unwrap();
            q.copy_from(row * k, pq, 0, p.n * k, &st)?;
            sc.copy_from(row * 4, ps, 0, p.n * 4, &st)?;
            row += p.n;
        }
    }
    st.synchronize()?;
    for p in parts.iter_mut() {
        p.prefill = None;
    }
    let segs = if glu {
        [0, 0]
    } else {
        [parts[0].n, parts.get(1).map_or(0, |p| p.n)]
    };
    Ok(Some(Fused { q, sc, n, k, segs, glu }))
}

/// Dense prefill GEMMs on the MLS4 weights with int4 activations (MALLEUS_DENSE_W4A4=1): the int4 WMMA is double
/// rate on gfx11 and the 4-bit weights are already there, but the projections must stop being merged, because
/// `fuse` stacks the MLS8 copies and the merged kernel has no int4 form.
///
/// Opt-in, and the measurement that keeps it that way is on GPTQ-calibrated weights, so it is not the quality
/// argument that ruled it out before. Same model, same command, flag on against flag off: prefill +0.6/+1.9/+2.4 %
/// at 512/4290/17034 tokens, all-position argmax a tie (94.34 against 94.28 %), but last-token top-1 21/21 ->
/// 19/21 and worst KL 0.282 -> 0.435. Two first generated tokens out of 21 are not worth 2 % of prefill while
/// prefill is 20-30 % behind anyway. On round-to-nearest weights it is worse still (93.30 %, KL 1.751).
pub(crate) static ENV_DENSE_W4A4: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var("MALLEUS_DENSE_W4A4").is_ok_and(|v| v == "1" || v == "acts" || v == "2" || v == "w4a8" || v == "3")
});

/// Whether the dense prefill GEMMs run W4A4. Read by the CLI to report the mode it chose.
pub fn dense_w4a4() -> bool {
    *ENV_DENSE_W4A4
}

/// `MALLEUS_DENSE_W4A4=acts` (or `2`): the same, extended to the projections whose input needs a transform
/// (`Act::Norm`, `GatedNorm`, `AttnGate` — on Flash-Next exactly `o_proj` and `out_proj`), which then fold it in
/// `rot_quant_i4_row` instead of staying on int8 and on the MLS8 copy of their weights. Measured on Flash-Next at
/// p32768, same binary: prefill **+5,6 %** at 24576 tokens and nothing at 8192, but the gate costs
/// **-1,58 punti** di argmax su tutte le posizioni (94,09 -> 92,51 %) e la KL peggiore raddoppia (0,465 -> 1,246).
/// That is the positive of the "twelve `o_proj` tensors to 8 bits" finding, on this model: without calibration these
/// two tensors are the sensitive ones. Opt-in, as the substrate for a calibrated conversion.
pub(crate) static ENV_DENSE_W4A4_ACTS: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("MALLEUS_DENSE_W4A4").is_ok_and(|v| v == "acts" || v == "2"));

pub fn dense_w4a4_acts() -> bool {
    *ENV_DENSE_W4A4_ACTS
}

/// `MALLEUS_DENSE_W4A4=w4a8` (or `3`): the same two projections on 4-bit weights, but with **int8** activations per
/// group folded in `rot_quant_i8_row`. What the MLS8 copies cost is the weight bytes, and those go away all the same;
/// the activation format is where the quality of the int4 variant was lost (measured: -1,58 punti di argmax).
pub(crate) static ENV_DENSE_W4A8_ACTS: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("MALLEUS_DENSE_W4A8_ACTS").map_or(true, |v| v != "0"));

pub fn dense_w4a8_acts() -> bool {
    *ENV_DENSE_W4A8_ACTS
}
