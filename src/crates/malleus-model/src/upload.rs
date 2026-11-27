//! The pinned uploader: one tensor at a time from the file to the device, plus the layout of a
//! `Linear` (decode weights and, when the file carries it, the MLS8 prefill copy).

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::{DeviceBuffer, PinnedBuffer};

use crate::weights::{Linear, Weights};

pub(crate) struct Uploader {
    pub(crate) pinned: PinnedBuffer,
    pub(crate) bytes: usize,
    pub(crate) use_prefill: bool,
    pub(crate) prefill_families: Vec<String>,
    /// Decode on the MLS8 copy when the file has one (MTP head).
    pub(crate) decode_int8: bool,
    /// Whether any `.prefill` tensor was actually read: a single-copy file has none, and then the prefill runs on
    /// the MLS4 nibbles (W4A8) whatever the flags say. Reported at startup and used to word it correctly.
    pub(crate) prefill_loaded: bool,
}

impl Uploader {
    /// Whether the MLS8 prefill copy of `name` is loaded: all of them when the list is unset, none with `none`, else
    /// the ones whose tensor name contains one of the entries (`MALLEUS_PREFILL_INT8_FAMILIES`).
    pub(crate) fn want_prefill(&self, name: &str) -> bool {
        match self.prefill_families.first().map(String::as_str) {
            Some("none") => false,
            None => true,
            _ => self.prefill_families.iter().any(|f| name.contains(f.as_str())),
        }
    }

    pub(crate) const CHUNK: usize = 64 << 20;

    pub(crate) fn new() -> Result<Self> {
        Ok(Uploader {
            pinned: PinnedBuffer::alloc(Self::CHUNK)?,
            bytes: 0,
            use_prefill: true,
            prefill_families: Vec::new(),
            decode_int8: false,
            prefill_loaded: false,
        })
    }

    pub(crate) fn tensor(&mut self, src: &dyn TensorSource, name: &str, shape: &[usize]) -> Result<DeviceBuffer> {
        self.typed(src, name, DType::BF16, shape)
    }

    pub(crate) fn typed(
        &mut self,
        src: &dyn TensorSource,
        name: &str,
        dtype: DType,
        shape: &[usize],
    ) -> Result<DeviceBuffer> {
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(info.dtype == dtype, "{name}: expected {dtype:?}, got {:?}", info.dtype);
        let flat: Vec<usize> = info.shape.iter().copied().filter(|&d| d != 1).collect();
        let want: Vec<usize> = shape.iter().copied().filter(|&d| d != 1).collect();
        ensure!(flat == want, "{name}: shape {:?}, expected {:?}", info.shape, shape);
        let mut buf = DeviceBuffer::alloc(info.size)?;
        self.copy(src, name, info.size, &mut buf, 0)?;
        Ok(buf)
    }

    /// Uploads a tensor into `buf` at byte offset `at`; `shape` is checked against the tensor flattened to the same
    /// number of elements and, for matrices, last dimension (expert stacks [E, N, K] load as [E * N, K]).
    pub(crate) fn into(
        &mut self,
        src: &dyn TensorSource,
        name: &str,
        dtype: DType,
        shape: &[usize],
        buf: &mut DeviceBuffer,
        at: usize,
    ) -> Result<()> {
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(info.dtype == dtype, "{name}: expected {dtype:?}, got {:?}", info.dtype);
        ensure!(
            info.shape.iter().product::<usize>() == shape.iter().product::<usize>()
                && (shape.len() < 2 || info.shape.last() == shape.last()),
            "{name}: shape {:?}, expected {:?}",
            info.shape,
            shape
        );
        ensure!(at + info.size <= buf.len(), "{name}: does not fit its buffer");
        self.copy(src, name, info.size, buf, at)
    }

    pub(crate) fn copy(
        &mut self,
        src: &dyn TensorSource,
        name: &str,
        size: usize,
        buf: &mut DeviceBuffer,
        at: usize,
    ) -> Result<()> {
        self.copy_range(src, name, 0, size, buf, at)
    }

    /// The same from byte `from` of the tensor: the rows of a quantized matrix are contiguous, so a window of it
    /// (the blocks of a grouped projection) is a byte range.
    pub(crate) fn copy_range(
        &mut self,
        src: &dyn TensorSource,
        name: &str,
        from: usize,
        size: usize,
        buf: &mut DeviceBuffer,
        at: usize,
    ) -> Result<()> {
        let mut off = 0;
        while off < size {
            let n = Self::CHUNK.min(size - off);
            src.read(name, from + off, &mut self.pinned.as_mut_slice()[..n])?;
            self.pinned.copy_to(n, buf, at + off)?;
            off += n;
        }
        self.bytes += size;
        Ok(())
    }

    pub(crate) fn linear(&mut self, src: &dyn TensorSource, name: &str, n: usize, k: usize) -> Result<Linear> {
        let wn = format!("{name}.weight");
        let info = src.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
        let g = k / malleus_gguf::quant::BLOCK;
        let pn = format!("{wn}.prefill");
        let w = match info.dtype {
            _ if self.decode_int8 && src.info(&pn).is_some() => Weights::Mls8 {
                q: self.typed(src, &pn, DType::Mls8, &[n, k])?,
                sc: self.typed(src, &format!("{pn}.scales"), DType::F32, &[n])?,
            },
            DType::BF16 => Weights::Bf16(self.tensor(src, &wn, &[n, k])?),
            DType::Mls1 => Weights::Mls1 {
                q: self.typed(src, &wn, DType::Mls1, &[n, k])?,
                sc: self.typed(src, &format!("{wn}.scales"), DType::F8E4M3, &[n, k / 32])?,
            },
            DType::Mls2 => Weights::Mls2 {
                q: self.typed(src, &wn, DType::Mls2, &[n, k])?,
                sc: self.typed(src, &format!("{wn}.scales"), DType::F16, &[n, g])?,
            },
            DType::Mls4 => Weights::Mls4 {
                q: self.typed(src, &wn, DType::Mls4, &[n, k])?,
                sc: self.typed(src, &format!("{wn}.scales"), DType::F16, &[n, g])?,
            },
            DType::Mls3 => Weights::Mls3 {
                q: self.typed(src, &wn, DType::Mls3, &[n, k])?,
                sc: self.typed(src, &format!("{wn}.scales"), DType::F16, &[n, g])?,
            },
            DType::Mls8 => Weights::Mls8 {
                q: self.typed(src, &wn, DType::Mls8, &[n, k])?,
                sc: self.typed(src, &format!("{wn}.scales"), DType::F32, &[n])?,
            },
            // A release's blocked fp8 — DeepSeek-V4.1's indexer query, eight tensors of five million weights. There is
            // no device path for that block format, and a kernel for it would serve nothing else: the host
            // dequantises once and the weights go up as bf16, which every GEMV here reads.
            DType::F8E4M3 => {
                let w = crate::deepseek_text::load_matrix(src, &wn, n, k)?;
                let b: Vec<half::bf16> = w.iter().map(|v| half::bf16::from_f32(*v)).collect();
                self.bytes += b.len() * 2;
                Weights::Bf16(DeviceBuffer::from_slice(&b)?)
            }
            d => bail!("{wn}: unsupported dtype {d:?}"),
        };
        let family_ok = self.want_prefill(name);
        let prefill = if self.use_prefill && family_ok && src.info(&pn).is_some() {
            self.prefill_loaded = true;
            Some((
                self.typed(src, &pn, DType::Mls8, &[n, k])?,
                self.typed(src, &format!("{pn}.scales"), DType::F32, &[n])?,
            ))
        } else {
            None
        };
        Ok(Linear { w, prefill, n, k })
    }

    /// Rows `[row0, row0 + n)` of an `[n_total, k]` matrix as a `Linear` of their own. DeepSeek-V4.1's output
    /// projection is one tensor of `o_groups` blocks that each read a different slice of the same input, so the
    /// device path needs the blocks apart: a GEMV of `n` rows reads `n` rows of the input. The prefill copy is not
    /// loaded (a decode step is what these are for).
    pub(crate) fn linear_rows(
        &mut self,
        src: &dyn TensorSource,
        name: &str,
        n_total: usize,
        k: usize,
        row0: usize,
        n: usize,
    ) -> Result<Linear> {
        let wn = format!("{name}.weight");
        let info = src.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
        ensure!(
            info.shape == [n_total, k],
            "{wn}: shape {:?}, expected [{n_total}, {k}]",
            info.shape
        );
        ensure!(row0 + n <= n_total, "{wn}: rows {row0}..{} past {n_total}", row0 + n);
        let groups = k / malleus_gguf::quant::BLOCK;
        // Bytes one row of the codes takes, and one row of the scales (None: a plain fp16 matrix).
        let (row_bytes, scale_row) = match info.dtype {
            DType::BF16 => (2 * k, None),
            DType::Mls1 => (k / 8, Some(k / 32)),
            DType::Mls2 => (k / 4, Some(2 * groups)),
            DType::Mls3 => (k / 8 * 3, Some(2 * groups)),
            DType::Mls4 => (k / 2, Some(2 * groups)),
            DType::Mls8 => (k, Some(4)),
            d => bail!("{wn}: a window of rows of a {d:?} matrix is not supported"),
        };
        let mut codes = DeviceBuffer::alloc(n * row_bytes)?;
        self.copy_range(src, &wn, row0 * row_bytes, n * row_bytes, &mut codes, 0)?;
        let w = match (info.dtype, scale_row) {
            (DType::BF16, _) => Weights::Bf16(codes),
            (dt, Some(sr)) => {
                let mut s = DeviceBuffer::alloc(n * sr)?;
                let sn = format!("{wn}.scales");
                self.copy_range(src, &sn, row0 * sr, n * sr, &mut s, 0)?;
                match dt {
                    DType::Mls1 => Weights::Mls1 { q: codes, sc: s },
                    DType::Mls2 => Weights::Mls2 { q: codes, sc: s },
                    DType::Mls3 => Weights::Mls3 { q: codes, sc: s },
                    DType::Mls8 => Weights::Mls8 { q: codes, sc: s },
                    _ => Weights::Mls4 { q: codes, sc: s },
                }
            }
            (dt, None) => bail!("{wn}: a window of rows of a {dt:?} matrix is not supported"),
        };
        Ok(Linear { w, prefill: None, n, k })
    }

    /// A norm weight as f32, for the kernels that multiply by it directly instead of going through `rmsnorm`
    /// (which adds the one and takes bf16): `mla_cache_write` reads `const float* norm_w`.
    pub(crate) fn norm_f32(&mut self, src: &dyn TensorSource, name: &str, n: usize) -> Result<DeviceBuffer> {
        let raw = src.read_all(name).with_context(|| format!("missing tensor {name}"))?;
        let v: Vec<f32> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b).to_f32())
            .collect();
        ensure!(v.len() == n, "{name}: expected {n} weights, got {}", v.len());
        self.bytes += v.len() * 4;
        Ok(DeviceBuffer::from_slice(&v)?)
    }

    /// An RMSNorm weight. Ling stores `w` where the kernels expect `1 + w` (Qwen's form): folding the one at load
    /// keeps every norm kernel single-flavoured, which is the whole point of ADR 0005. The tensors are a few KB, so
    /// the fold happens on the host while uploading.
    /// An MLS4 tensor read back as plain fp32 [n, k]: dequantize every 128 group and undo the Hadamard rotation.
    /// This is the one place the quantized form has to become a matrix again, and it is here because MLA decode
    /// absorbs `kv_b_proj` into the query — a product against the transpose, which no GEMV can express.
    pub(crate) fn linear_f32(
        &mut self,
        src: &dyn TensorSource,
        name: &str,
        n: usize,
        k: usize,
        seed: u64,
    ) -> Result<DeviceBuffer> {
        use malleus_gguf::quant::{self, BLOCK};
        let wn = format!("{name}.weight");
        let info = src.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
        ensure!(info.dtype == DType::Mls4, "{wn}: expected MLS4, got {:?}", info.dtype);
        let q = src.read_all(&wn)?;
        let sc = src.read_all(&format!("{wn}.scales"))?;
        let g = k / BLOCK;
        let signs = quant::hadamard_signs(seed);
        let mut out = vec![0f32; n * k];
        for r in 0..n {
            for j in 0..g {
                // One f16 scale per (row, 128-group): the byte offset is `(r * g + j) * 2`, not `r * g + j * 2`.
                let at = (r * g + j) * 2;
                let s = half::f16::from_le_bytes([sc[at], sc[at + 1]]).to_f32();
                let mut blk = [0f32; BLOCK];
                quant::dequant_int4(&q[r * k / 2 + j * BLOCK / 2..][..BLOCK / 2], s, &mut blk);
                quant::unrotate_block(&mut blk, &signs);
                out[r * k + j * BLOCK..][..BLOCK].copy_from_slice(&blk);
            }
        }
        self.bytes += n * k * 4;
        Ok(DeviceBuffer::from_slice(&out)?)
    }

    /// The same dequantization as `linear_f32`, kept in bf16: half the bytes for a matrix a GEMM reads as tiles, and
    /// the reference keeps these weights in bf16 anyway (Hy4's `kv_b_proj`, which the absorbed MLA reads directly).
    pub(crate) fn linear_bf16(
        &mut self,
        src: &dyn TensorSource,
        name: &str,
        n: usize,
        k: usize,
        seed: u64,
    ) -> Result<DeviceBuffer> {
        use malleus_gguf::quant::{self, BLOCK};
        let wn = format!("{name}.weight");
        let info = src.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
        ensure!(info.dtype == DType::Mls4, "{wn}: expected MLS4, got {:?}", info.dtype);
        let q = src.read_all(&wn)?;
        let sc = src.read_all(&format!("{wn}.scales"))?;
        let g = k / BLOCK;
        let signs = quant::hadamard_signs(seed);
        let mut out = vec![0u16; n * k];
        for r in 0..n {
            for j in 0..g {
                let at = (r * g + j) * 2;
                let s = half::f16::from_le_bytes([sc[at], sc[at + 1]]).to_f32();
                let mut blk = [0f32; BLOCK];
                quant::dequant_int4(&q[r * k / 2 + j * BLOCK / 2..][..BLOCK / 2], s, &mut blk);
                quant::unrotate_block(&mut blk, &signs);
                for (i, v) in blk.iter().enumerate() {
                    out[r * k + j * BLOCK + i] = half::bf16::from_f32(*v).to_bits();
                }
            }
        }
        self.bytes += n * k * 2;
        Ok(DeviceBuffer::from_slice(&out)?)
    }

    /// A norm weight, as the kernels that read it want it.
    ///
    /// Qwen's RMSNorm is `(1 + w) * x / rms(x)` and the kernels add the one, so the tensor goes up as it is
    /// (`raw`). Ling's is `w * x / rms(x)`: rather than teaching every kernel the difference — the one is added
    /// in `rmsnorm`, in the fused GEMM epilogues of `wmma.hip` and in the hyper-connection passes — the one is
    /// taken away here and they put it back. Storing `w - 1` in bf16 is also *more* accurate in absolute terms
    /// than storing `w`, because the rounding is relative to the stored value.
    ///
    /// The gated norms (KDA `o_norm`, MLA `kv_a_layernorm`) are fla's `FusedRMSNormGated`, which uses `w`
    /// directly in a kernel of its own that never adds one: those are `raw` for both families.
    pub(crate) fn norm(&mut self, src: &dyn TensorSource, name: &str, n: usize, raw: bool) -> Result<DeviceBuffer> {
        if raw {
            return self.tensor(src, name, &[n]);
        }
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.dtype == DType::BF16 && info.size == n * 2,
            "{name}: expected bf16 [{n}]"
        );
        let bytes = src.read_all(name)?;
        let w: Vec<u16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b).to_f32() - 1.0)
            .map(|x| half::bf16::from_f32(x).to_le_bytes())
            .map(u16::from_le_bytes)
            .collect();
        self.bytes += w.len() * 2;
        Ok(DeviceBuffer::from_slice(&w)?)
    }
}
