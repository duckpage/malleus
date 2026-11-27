//! Qwen3.8-Flash-Next: the gated hyper-connection streams, the PLE n-gram embedding, and the QSA token
//! indexer. Everything else is the Qwen3.5 graph (`crate::qwen35`).

use anyhow::{Context, Result, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::{Arg, DeviceBuffer, Stream};
use serde_json::Value;

use crate::config::{Config, HcConfig, IndexerConfig, PleConfig};
use crate::engine::{Act, Model};
use crate::session::{ATTN_WMMA_MIN, IdxStates, Scratch, Session, f32s};
use crate::upload::Uploader;
use crate::weights::{GEMV_MAX_T, Linear, Weights, mls8_of};

/// Qwen3.8-Flash-Next spells `model_type: qwen4_exp_text` under `text_config`; the other families leave the
/// three shapes below absent.
fn qwen4(t: &Value) -> bool {
    t["model_type"].as_str() == Some("qwen4_exp_text")
}

/// The gated hyper-connection streams: how many, and the rank of the mixing.
pub(crate) fn hc(t: &Value, u: &dyn Fn(&str) -> Result<usize>) -> Result<Option<HcConfig>> {
    match t["hc_count"].as_u64() {
        Some(n) if qwen4(t) => Ok(Some(HcConfig {
            count: n as usize,
            rank: u("hc_lowrank")?,
        })),
        _ => Ok(None),
    }
}

/// The PLE layer: one layer whose input receives the hashed n-gram embedding.
pub(crate) fn ple(t: &Value, u: &dyn Fn(&str) -> Result<usize>) -> Result<Option<PleConfig>> {
    let ids = match t["ple_layer_ids"].as_array() {
        Some(ids) if qwen4(t) && !ids.is_empty() => ids,
        _ => return Ok(None),
    };
    ensure!(ids.len() == 1, "only one PLE layer is supported");
    let eos = match &t["eos_token_id"] {
        Value::Array(a) => a.first().and_then(|v| v.as_u64()),
        v => v.as_u64(),
    }
    .context("config: eos_token_id")?;
    Ok(Some(PleConfig {
        layer: ids[0].as_u64().context("ple_layer_ids")? as usize - 1,
        ngram: u("ngram_size")?,
        heads_per_ngram: u("heads_per_ngram")?,
        dim: t["ple_embed_dim"].as_u64().map_or(u("hidden_size")?, |v| v as usize),
        conv_k: u("ple_conv_kernel_size")?,
        eos: eos as u32,
    }))
}

/// The QSA token indexer: how many heads score the blocks, their width, the selection budget and the
/// compression ratio of a block.
pub(crate) fn indexer(t: &Value, u: &dyn Fn(&str) -> Result<usize>) -> Result<Option<IndexerConfig>> {
    match t["indexer_n_heads"].as_u64() {
        Some(h) if qwen4(t) => Ok(Some(IndexerConfig {
            heads: h as usize,
            dim: u("indexer_head_dim")?,
            budget: u("indexer_budget")?,
            ratio: u("indexer_compress_ratio")?,
        })),
        _ => Ok(None),
    }
}

/// QSA token indexer: q (heads x dim) | k (1 x dim) projection and their RMSNorms.
pub(crate) struct Indexer {
    pub(crate) qk: Linear,
    pub(crate) q_norm: DeviceBuffer,
    pub(crate) k_norm: DeviceBuffer,
}

/// Gated hyper-connection: per-stream RMSNorm, [down; inject] (rows padded to 128) and up (K padded to 128). MLS8
/// `up` rows are reordered at load to j * count + s (stream s of output j), so one GEMM tile holds every stream of
/// its outputs and the mix runs in its epilogue.
pub(crate) struct Hc {
    pub(crate) norm: DeviceBuffer,
    pub(crate) down: Linear,
    pub(crate) up: Linear,
    pub(crate) up_inter: bool,
}

/// PLE layer: key (one per stream) and value projections of the n-gram embedding, norms and the dilated conv.
pub(crate) struct Ple {
    pub(crate) key: Linear,
    pub(crate) value: Linear,
    pub(crate) norm_key: DeviceBuffer,
    pub(crate) norm_query: DeviceBuffer,
    pub(crate) norm_conv: DeviceBuffer,
    pub(crate) conv: DeviceBuffer,
}

/// Steps of at most this many tokens read their n-gram rows on the calling thread (`NgramTable::embed`).
pub(crate) const NGRAM_INLINE_TOKENS: usize = 16;

/// Hashed n-gram embedding table read from the model file on the host (int8 rows + f32 scale, ngram_table).
pub(crate) struct NgramTable {
    file: std::sync::Arc<malleus_gguf::TensorFile>,
    /// Row reads are page-cache or disk reads: many in flight (decode reads 16 rows per token).
    pool: rayon::ThreadPool,
    rows: usize,
    /// Width of one head's row.
    dim: usize,
    mult: Vec<i64>,
    sizes: Vec<i64>,
    offsets: Vec<i64>,
    cfg: PleConfig,
}

impl NgramTable {
    pub(crate) fn load(src: &dyn TensorSource, base: &str, cfg: &PleConfig) -> Result<Self> {
        let name = format!("{base}ngram_table");
        let info = src.info(&name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.dtype == DType::I8 && info.shape.len() == 2,
            "{name}: expected I8 [rows, dim + 4]"
        );
        let (rows, dim) = (info.shape[0], info.shape[1] - 4);
        ensure!(
            dim * cfg.heads() == cfg.dim,
            "{name}: row width {dim} for {} heads",
            cfg.heads()
        );
        let i64s = |n: &str, len: usize| -> Result<Vec<i64>> {
            let t = format!("{base}{n}");
            let info = src.info(&t).with_context(|| format!("missing tensor {t}"))?;
            ensure!(
                info.dtype == DType::I64 && info.numel() == len,
                "{t}: expected I64 [{len}]"
            );
            Ok(src
                .read_all(&t)?
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect())
        };
        // Opened again so that the hint does not reach the loader's sequential reads: without it every 4 KB row
        // read brings in a readahead window that fills the page cache with rows nobody asked for.
        let file = std::sync::Arc::new(
            src.open_tensor(&name)
                .context("n-gram table: no file handle")?
                .reopened(),
        );
        if let Some((f, _, _)) = file.range(0, 0) {
            malleus_hip::file_hint::random(f);
        }
        if std::env::var("MALLEUS_NGRAM_WARM").is_ok_and(|v| v == "1") {
            // Read the table once in the background so the first lookups hit the page cache. Off by default: with the
            // weights in GTT the table (52 GB) does not fit in the remaining page cache, and streaming it evicts the
            // rows of frequent n-grams that plain lookups keep cached.
            let f = file.clone();
            std::thread::Builder::new().name("ngram-warm".into()).spawn(move || {
                let mut buf = vec![0u8; 8 << 20];
                let mut off = 0;
                while off < f.size {
                    let n = buf.len().min(f.size - off);
                    if f.read_at(off, &mut buf[..n]).is_err() {
                        break;
                    }
                    off += n;
                }
            })?;
        }
        let t = NgramTable {
            file,
            pool: rayon::ThreadPoolBuilder::new()
                // One blocking pread per thread, so this is the queue depth the table is read at: the rows are
                // scattered over 52 GB and the drive, not the CPU, sets the pace. The 94 400 row reads of a 5900
                // token prompt take 253 ms at 32 threads, 152 at 96 and 43 at 256. The stacks stay at rayon's
                // default: 64 KiB overflows inside the work-stealing and error paths.
                .num_threads(
                    std::env::var("MALLEUS_NGRAM_THREADS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|&n| n > 0)
                        .unwrap_or(256),
                )
                .thread_name(|i| format!("ngram-{i}"))
                .build()?,
            rows,
            dim,
            mult: i64s("layer_multipliers", cfg.ngram)?,
            sizes: i64s("ngram_heads_vocab_sizes", cfg.heads())?,
            offsets: i64s("ngram_heads_offsets", cfg.heads())?,
            cfg: cfg.clone(),
        };
        ensure!(
            t.sizes
                .iter()
                .zip(&t.offsets)
                .all(|(s, o)| *s > 0 && *o >= 0 && (o + s) as usize <= rows),
            "{name}: head ranges out of the table"
        );
        Ok(t)
    }

    /// Table row of every head for each token, given the `ngram - 1` tokens before them (`ctx`).
    pub(crate) fn row_ids(&self, ctx: &[u32], tokens: &[u32]) -> Vec<usize> {
        let c = &self.cfg;
        let hist: Vec<u32> = ctx.iter().chain(tokens).copied().collect();
        let mut ids = Vec::with_capacity(tokens.len() * c.heads());
        for p in ctx.len()..hist.len() {
            // Token s positions back, or eos when that crosses an eos (n-grams stay inside one document).
            let mut shifted = vec![hist[p] as i64; c.ngram];
            let mut crossed = false;
            for s in 1..c.ngram {
                crossed |= p < s || hist[p - s] == c.eos;
                shifted[s] = if crossed { c.eos as i64 } else { hist[p - s] as i64 };
            }
            for n in 2..=c.ngram {
                let mut mixed = shifted[0].wrapping_mul(self.mult[0]);
                for (j, sh) in shifted.iter().enumerate().take(n).skip(1) {
                    mixed ^= sh.wrapping_mul(self.mult[j]);
                }
                for h in 0..c.heads_per_ngram {
                    let head = (n - 2) * c.heads_per_ngram + h;
                    ids.push((mixed.rem_euclid(self.sizes[head]) + self.offsets[head]) as usize);
                }
            }
        }
        ids
    }

    /// Starts reading the rows of a decode step: called before the layers that precede the PLE are launched, so
    /// the drive works while the GPU does and `embed` finds the rows cached. A prefill is left to `embed`'s pool.
    pub(crate) fn prefetch(&self, ctx: &[u32], tokens: &[u32]) {
        if tokens.len() > NGRAM_INLINE_TOKENS {
            return;
        }
        let rb = self.dim + 4;
        for id in self.row_ids(ctx, tokens) {
            if let Some((f, offset, len)) = self.file.range(id * rb, rb) {
                malleus_hip::file_hint::will_need(f, offset, len);
            }
        }
    }

    /// Dequantized embeddings [tokens, heads * dim] (f32).
    pub(crate) fn embed(&self, ctx: &[u32], tokens: &[u32]) -> Result<Vec<f32>> {
        use rayon::prelude::*;
        let ids = self.row_ids(ctx, tokens);
        let (dim, rb) = (self.dim, self.dim + 4);
        let mut out = vec![0f32; ids.len() * dim];
        let file = &self.file;
        let rows = self.rows;
        if tokens.len() <= NGRAM_INLINE_TOKENS {
            // A decode step: its few rows were asked for by `prefetch`, and waking a pool of blocking readers
            // costs more than the reads (measured: 1-2 ms of idle GPU per token at the PLE layer).
            let mut buf = vec![0u8; rb];
            for (row, &id) in out.chunks_mut(dim).zip(&ids) {
                ensure!(id < rows, "n-gram row {id} out of range");
                file.read_at(id * rb, &mut buf)?;
                let s = f32::from_le_bytes([buf[dim], buf[dim + 1], buf[dim + 2], buf[dim + 3]]);
                for (d, &q) in row.iter_mut().zip(&buf[..dim]) {
                    *d = q as i8 as f32 * s;
                }
            }
            return Ok(out);
        }
        self.pool.install(|| {
            out.par_chunks_mut(dim)
                .zip(ids.par_iter())
                .with_min_len(4)
                .try_for_each_init(
                    || vec![0u8; rb],
                    |buf, (row, &id)| -> Result<()> {
                        ensure!(id < rows, "n-gram row {id} out of range");
                        file.read_at(id * rb, buf)?;
                        let s = f32::from_le_bytes([buf[dim], buf[dim + 1], buf[dim + 2], buf[dim + 3]]);
                        for (d, &q) in row.iter_mut().zip(&buf[..dim]) {
                            *d = q as i8 as f32 * s;
                        }
                        Ok(())
                    },
                )
        })?;
        Ok(out)
    }
}

/// Queries scored per pass of the indexer (bounds the score buffer).
pub(crate) const IDX_SUB: usize = 256;

/// Permutes `rows` weights from stream-major (`s * hidden + j`) to the interleaved (`j * cnt + s`) order the
/// hyper-connection up GEMM reads. Weight and scale rows move as opaque bytes, so this serves MLS8 (`qw` = K int8,
/// `sw` = 4 for one f32) and MLS4 (`qw` = K / 2 nibble bytes, `sw` = 2 * K / 128 for the f16 group scales) alike.
pub(crate) fn permute_hc_rows(
    q: &mut DeviceBuffer,
    sc: &mut DeviceBuffer,
    rows: usize,
    qw: usize,
    sw: usize,
    cnt: usize,
    hidden: usize,
) -> Result<()> {
    let qv: Vec<u8> = q.download(rows * qw)?;
    let sv: Vec<u8> = sc.download(rows * sw)?;
    let (mut q2, mut s2) = (vec![0u8; rows * qw], vec![0u8; rows * sw]);
    for s in 0..cnt {
        for j in 0..hidden {
            let (from, to) = (s * hidden + j, j * cnt + s);
            q2[to * qw..(to + 1) * qw].copy_from_slice(&qv[from * qw..(from + 1) * qw]);
            s2[to * sw..(to + 1) * sw].copy_from_slice(&sv[from * sw..(from + 1) * sw]);
        }
    }
    q.upload(&q2)?;
    sc.upload(&s2)?;
    Ok(())
}

/// Permutes every representation of the hyper-connection `up` weights that the file carries: the decode weights
/// and, when `--hc-mls4` stored them apart, the MLS8 prefill copy. A path that read one and not the other would
/// mix rows of different streams.
pub(crate) fn permute_hc_up(l: &mut Linear, cnt: usize, hidden: usize) -> Result<bool> {
    let k = l.k;
    match &mut l.w {
        Weights::Mls1 { .. } => {}
        Weights::Mls8 { q, sc } => permute_hc_rows(q, sc, l.n, k, 4, cnt, hidden)?,
        Weights::Mls4 { q, sc } => {
            permute_hc_rows(q, sc, l.n, k / 2, 2 * (k / malleus_gguf::quant::BLOCK), cnt, hidden)?
        }
        // Hyper-connections only exist in Qwen4Exp, whose mixing is MLS4 or MLS8.
        Weights::Mls2 { .. } | Weights::Mls3 { .. } | Weights::Bf16(_) => return Ok(false),
    }
    if let Some((q, sc)) = &mut l.prefill {
        permute_hc_rows(q, sc, l.n, k, 4, cnt, hidden)?;
    }
    Ok(true)
}

/// The int8 weights of a linear to run a prefill GEMM on: the tensor itself when it is MLS8, otherwise the separate
/// MLS8 prefill copy. The hyper-connection mixing has no int4 kernel, so a file converted with `--hc-mls4` runs
/// both of its paths on the copy until one lands.
pub(crate) fn load_hc(up: &mut Uploader, src: &dyn TensorSource, c: &Config, pre: &str) -> Result<Hc> {
    let sd = c.stream_dim();
    let rows = |n: &str| -> Result<usize> {
        let info = src.info(n).with_context(|| format!("missing tensor {n}"))?;
        Ok(info.shape[0])
    };
    let (down, upn) = (format!("{pre}mix_down"), format!("{pre}mix_up"));
    let nd = rows(&format!("{down}.weight"))?;
    let rp = src.info(&format!("{upn}.weight")).map_or(0, |i| i.shape[1]);
    let mut h = Hc {
        norm: up.tensor(src, &format!("{pre}hc_norm.weight"), &[sd])?,
        down: up.linear(src, &down, nd, sd)?,
        up: up.linear(src, &upn, sd, rp)?,
        up_inter: false,
    };
    // MALLEUS_HC_W4=0 drops the MLS4 weights and keeps decode on the MLS8 copy. A file converted with --hc-mls4
    // carries both, so the 4-bit mixing can be measured against today's behaviour on one file; converting twice
    // would instead also change the dense calibration, because calibrate.py propagates through the weights it has
    // just quantized.
    if std::env::var("MALLEUS_HC_W4").is_ok_and(|v| v == "0") {
        for l in [&mut h.down, &mut h.up] {
            if matches!(l.w, Weights::Mls4 { .. })
                && let Some((q, sc)) = l.prefill.take()
            {
                l.w = Weights::Mls8 { q, sc };
            }
        }
    }
    h.up_inter = permute_hc_up(&mut h.up, c.hc.as_ref().map_or(1, |x| x.count), c.hidden)?;
    Ok(h)
}

impl Model {
    /// Hyper-connection mix of the streams x [t, C * H] into out [t, H]; with `inj_out` also the injection weights of
    /// the paired combine. `inject` (block output, its injection weights) is first added to the streams.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_mix(
        &self,
        st: &Stream,
        sc: &Scratch,
        h: &Hc,
        x: &DeviceBuffer,
        out: &DeviceBuffer,
        inj_out: Option<&DeviceBuffer>,
        t: usize,
        inject: Option<(&DeviceBuffer, &DeviceBuffer)>,
    ) -> Result<()> {
        let c = &self.cfg;
        let hcfg = c.hc.as_ref().context("no hyper-connections")?;
        let hs = sc.hc.as_ref().context("no hyper-connection scratch")?;
        let (cnt, hd) = (hcfg.count, c.hidden);
        let rp = h.down.n;
        let sd = cnt * hd;
        const SPLIT: usize = 4;
        if let (Weights::Mls8 { q: dq, sc: ds }, Weights::Mls8 { q: uq, sc: us }) = (&h.down.w, &h.up.w)
            && t <= GEMV_MAX_T
            && cnt <= 4
            && sd.is_multiple_of(SPLIT * 512)
            && rp.is_multiple_of(128)
        {
            // Single token only: with more tokens every block of the fused down kernel normalizes all of them.
            if t == 1 && hd <= 2560 && rp <= 384 && std::env::var("MALLEUS_HC_FUSED").map_or(true, |v| v != "0") {
                // Two kernels: inject + norm + rotation + split-K down per stream; gate + up + mix + state update.
                let (o, ij) = match inject {
                    Some((o, ij)) => (o.into(), ij.into()),
                    None => (Arg::U64(0), Arg::U64(0)),
                };
                self.k.get("hc_down_fused").launch(
                    (rp.div_ceil(8) as u32, cnt as u32, 1),
                    (256, 1, 1),
                    (t * hd * 2) as u32,
                    st,
                    &[
                        (&hs.hu).into(),
                        (&hs.inv).into(),
                        x.into(),
                        o,
                        ij,
                        (&h.norm).into(),
                        dq.into(),
                        ds.into(),
                        (&self.signs).into(),
                        t.into(),
                        rp.into(),
                        cnt.into(),
                        hd.into(),
                        c.eps.into(),
                    ],
                )?;
                ensure!(h.up_inter, "hyper-connection up rows not reordered");
                // Gate once (rotated a in hc.ha, injection weights), read by every block of the up kernel.
                self.k.get("hc_gate_dec").launch(
                    ((t * rp / 128) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &[
                        (&hs.ha).into(),
                        inj_out.map_or(Arg::U64(0), |b| b.into()),
                        (&hs.hu).into(),
                        t.into(),
                        hcfg.rank.into(),
                        rp.into(),
                        cnt.into(),
                        cnt.into(),
                        (&self.signs).into(),
                    ],
                )?;
                self.k.get("hc_up_fused").launch(
                    (hd.div_ceil(8) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        out.into(),
                        x.into(),
                        o,
                        ij,
                        Arg::U64(0),
                        (&hs.hu).into(),
                        uq.into(),
                        us.into(),
                        (&h.norm).into(),
                        (&hs.inv).into(),
                        (&self.signs).into(),
                        t.into(),
                        hcfg.rank.into(),
                        rp.into(),
                        cnt.into(),
                        hd.into(),
                        (&hs.ha).into(),
                    ],
                )?;
                return Ok(());
            }
            if let Some((o, ij)) = inject {
                self.hc_inject(st, x, o, ij, t)?;
            }
            // Few tokens: norm + rotation, split-K down, gate + rotation, up with the mix in its epilogue.
            self.k.get("hc_pre_dec").launch(
                ((t * sd / 128) as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &[
                    (&sc.xr).into(),
                    (&hs.hcn).into(),
                    x.into(),
                    (&h.norm).into(),
                    (&self.signs).into(),
                    cnt.into(),
                    hd.into(),
                    c.eps.into(),
                ],
            )?;
            self.k.get("hc_down_dec").launch(
                (rp.div_ceil(8) as u32, SPLIT as u32, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&hs.hu).into(),
                    (&sc.xr).into(),
                    dq.into(),
                    ds.into(),
                    t.into(),
                    rp.into(),
                    sd.into(),
                    (sd / SPLIT).into(),
                ],
            )?;
            let inj_arg: Arg = inj_out.map_or(Arg::U64(0), |b| b.into());
            self.k.get("hc_gate_dec").launch(
                ((t * rp / 128) as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &[
                    (&hs.ha).into(),
                    inj_arg,
                    (&hs.hu).into(),
                    t.into(),
                    hcfg.rank.into(),
                    rp.into(),
                    cnt.into(),
                    SPLIT.into(),
                    (&self.signs).into(),
                ],
            )?;
            self.k.get("hc_up_mix_dec").launch(
                (hd.div_ceil(8) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    out.into(),
                    (&hs.hcn).into(),
                    (&hs.ha).into(),
                    uq.into(),
                    us.into(),
                    t.into(),
                    cnt.into(),
                    hd.into(),
                    rp.into(),
                ],
            )?;
            self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        if self.row8(&h.down, t) && self.row8(&h.up, t) && cnt <= 4 && sd <= 8 * 128 * 10 {
            // Many tokens: inject + per-stream norm + rotation + int8 in one pass over the streams, the int8 GEMMs,
            // and the mix normalizing the streams again from the saved 1 / rms.
            let (o, ij) = match inject {
                Some((o, ij)) => (o.into(), ij.into()),
                None => (Arg::U64(0), Arg::U64(0)),
            };
            self.k.get("hc_quant_row").launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&sc.xh).into(),
                    (&sc.xsc).into(),
                    (&hs.inv).into(),
                    x.into(),
                    o,
                    ij,
                    (&h.norm).into(),
                    (&self.signs).into(),
                    cnt.into(),
                    hd.into(),
                    c.eps.into(),
                ],
            )?;
            // mix_down has few rows: split K over 4 slices (partials in hc.hu, summed by hc_gate).
            let (dq, ds) = mls8_of(&h.down)?;
            let (uq, us) = mls8_of(&h.up)?;
            const SK: usize = 4;
            self.k.get("gemm_i8_sk").launch(
                ((rp.div_ceil(128) * t.div_ceil(256)) as u32, SK as u32, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&hs.hu).into(),
                    (&sc.xh).into(),
                    (&sc.xsc).into(),
                    dq.into(),
                    ds.into(),
                    t.into(),
                    rp.into(),
                    (sd / SK).into(),
                    sd.into(),
                ],
            )?;
            let inj_arg: Arg = inj_out.map_or(Arg::U64(0), |b| b.into());
            self.ew(
                st,
                "hc_gate",
                t * rp,
                &[
                    (&hs.ha).into(),
                    inj_arg,
                    (&hs.hu).into(),
                    t.into(),
                    hcfg.rank.into(),
                    rp.into(),
                    cnt.into(),
                    SK.into(),
                ],
            )?;
            self.quant_row(st, &sc.xh, &sc.xsc, Act::X(&hs.ha), rp, t)?;
            ensure!(h.up_inter, "hyper-connection up rows not reordered");
            self.k.get("gemm_i8_hcmix").launch(
                ((sd.div_ceil(128) * t.div_ceil(128)) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    out.into(),
                    (&sc.xh).into(),
                    (&sc.xsc).into(),
                    uq.into(),
                    us.into(),
                    t.into(),
                    sd.into(),
                    rp.into(),
                    x.into(),
                    (&hs.inv).into(),
                    (&h.norm).into(),
                    cnt.into(),
                    hd.into(),
                ],
            )?;
            self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        if let Some((o, ij)) = inject {
            self.k.get("hc_inject_norm").launch(
                ((t * cnt) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&hs.hcn).into(),
                    x.into(),
                    o.into(),
                    ij.into(),
                    (&h.norm).into(),
                    cnt.into(),
                    hd.into(),
                    c.eps.into(),
                ],
            )?;
        } else {
            self.k.get("hc_norm").launch(
                ((t * cnt) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&hs.hcn).into(),
                    x.into(),
                    (&h.norm).into(),
                    cnt.into(),
                    hd.into(),
                    c.eps.into(),
                ],
            )?;
        }
        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &hs.hd, Act::X(&hs.hcn), &h.down, t, false)?;
        let inj_arg: Arg = inj_out.map_or(Arg::U64(0), |b| b.into());
        self.ew(
            st,
            "hc_gate",
            t * rp,
            &[
                (&hs.ha).into(),
                inj_arg,
                (&hs.hd).into(),
                t.into(),
                hcfg.rank.into(),
                rp.into(),
                cnt.into(),
                1usize.into(),
            ],
        )?;
        ensure!(h.up.k == rp, "hyper-connection up K {} != down rows {rp}", h.up.k);
        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &hs.hu, Act::X(&hs.ha), &h.up, t, false)?;
        self.ew(
            st,
            "hc_mix",
            t * hd,
            &[
                out.into(),
                (&hs.hu).into(),
                (&hs.hcn).into(),
                t.into(),
                cnt.into(),
                hd.into(),
                (h.up_inter as i32).into(),
            ],
        )?;
        self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// x[t, s] += o[t] * inj[t, s].
    pub(crate) fn hc_inject(
        &self,
        st: &Stream,
        x: &DeviceBuffer,
        o: &DeviceBuffer,
        inj: &DeviceBuffer,
        t: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let cnt = c.hc.as_ref().context("no hyper-connections")?.count;
        self.ew(
            st,
            "hc_inject",
            t * cnt * c.hidden,
            &[x.into(), o.into(), inj.into(), t.into(), cnt.into(), c.hidden.into()],
        )
    }

    /// PLE: streams += gv + silu(conv(norm(gv))), gv = sigmoid(gate(key_s(e), stream_s)) * value(e), with the n-gram
    /// embedding e in sc.hc.ple_e.
    pub(crate) fn ple(
        &self,
        st: &Stream,
        sc: &Scratch,
        p: &Ple,
        state: &DeviceBuffer,
        snap: Option<&DeviceBuffer>,
        t: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let (hcfg, pc) = (c.hc.as_ref().context("no hc")?, c.ple.as_ref().context("no ple")?);
        let hs = sc.hc.as_ref().context("no hyper-connection scratch")?;
        let (cnt, hd, sd) = (hcfg.count, c.hidden, c.stream_dim());
        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &hs.hu, Act::X(&hs.ple_e), &p.key, t, false)?;
        self.gemm(
            st,
            &sc.xr,
            &sc.xh,
            &sc.xsc,
            &hs.mo,
            Act::Same(&hs.ple_e),
            &p.value,
            t,
            false,
        )?;
        self.k.get("ple_gate_norm").launch(
            ((t * cnt) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&hs.hcn).into(),
                (&hs.hu).into(),
                (&hs.hu).into(),
                (&hs.hc).into(),
                (&hs.mo).into(),
                (&p.norm_key).into(),
                (&p.norm_query).into(),
                (&p.norm_conv).into(),
                cnt.into(),
                hd.into(),
                c.eps.into(),
            ],
        )?;
        self.ew(
            st,
            "ple_conv",
            t * sd,
            &[
                (&hs.hc).into(),
                (&hs.hcn).into(),
                (&hs.hu).into(),
                state.into(),
                (&p.conv).into(),
                t.into(),
                sd.into(),
                pc.conv_k.into(),
                pc.ngram.into(),
            ],
        )?;
        if let Some(snap) = snap {
            // Verify: keep the state before the step and the step's normalized rows for `rollback`.
            let sl = pc.conv_state() * sd;
            ensure!(snap.len() >= (sl + t * sd) * 4, "PLE snapshot too small");
            self.ew(st, "copy_f32", sl, &[snap.into(), state.into(), sl.into()])?;
            let tail = Arg::U64(snap.addr() + sl as u64 * 4);
            self.ew(st, "copy_f32", t * sd, &[tail, (&hs.hu).into(), (t * sd).into()])?;
        }
        self.ew(
            st,
            "ple_state_save",
            sd,
            &[
                state.into(),
                (&hs.hu).into(),
                t.into(),
                sd.into(),
                pc.conv_state().into(),
            ],
        )?;
        self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Words of a block-selection bitset for a session.
    pub(crate) fn idx_words(&self, max_ctx: usize) -> usize {
        self.cfg
            .indexer
            .as_ref()
            .map_or(0, |ic| (max_ctx / ic.ratio + 1).div_ceil(32))
    }

    /// Whether an input of t tokens after `len` has a query with more complete blocks than the indexer budget.
    pub(crate) fn idx_sparse(&self, len: usize, t: usize) -> bool {
        self.cfg
            .indexer
            .as_ref()
            .is_some_and(|ic| (len + t) / ic.ratio > ic.budget / ic.ratio)
    }

    /// Grows the indexer score and selection buffers of the session for an input of t tokens after `len`.
    pub(crate) fn idx_grow_len(&self, sess: &mut Session, len: usize, t: usize) -> Result<()> {
        if !self.idx_sparse(len, t) {
            return Ok(());
        }
        let w = self.idx_words(sess.max_ctx);
        if sess.idx_scores.is_none() {
            sess.idx_scores = Some(f32s(IDX_SUB * w * 32)?);
        }
        if let Some(ic) = &self.cfg.indexer
            && sess.idx_list.as_ref().is_none_or(|l| l.1 < t)
        {
            let rows = t.max(ATTN_WMMA_MIN);
            sess.idx_list = Some((DeviceBuffer::alloc(rows * (ic.budget / ic.ratio + 1) * 4)?, rows));
        }
        if sess.idx_sel.as_ref().is_none_or(|s| s.2 < t) {
            let rows = t.max(64);
            sess.idx_sel = Some((
                DeviceBuffer::alloc(rows * w * 4)?,
                DeviceBuffer::alloc(rows.div_ceil(16) * w * 4)?,
                rows,
            ));
        }
        Ok(())
    }

    /// The selection buffers for an input of t tokens after `len`, when it needs sparse attention.
    #[allow(clippy::type_complexity)]
    pub(crate) fn idx_sel_bufs<'a>(
        &self,
        sess: &'a Session,
        len: usize,
        t: usize,
    ) -> Result<Option<(&'a DeviceBuffer, &'a DeviceBuffer, &'a DeviceBuffer, usize)>> {
        if !self.idx_sparse(len, t) {
            return Ok(None);
        }
        let (sel, uni, _) = sess.idx_sel.as_ref().context("indexer buffers")?;
        Ok(Some((
            sess.idx_scores.as_ref().context("indexer buffers")?,
            sel,
            uni,
            self.idx_words(sess.max_ctx),
        )))
    }

    pub(crate) fn idx_states<'a>(&self, sess: &'a Session, t: usize, reuse: bool) -> Result<Option<IdxStates<'a>>> {
        if self.cfg.indexer.is_none() {
            return Ok(None);
        }
        Ok(Some(IdxStates {
            kb: &sess.idx_kb,
            tk: &sess.idx_tk,
            tp: &sess.idx_tp,
            sel: self.idx_sel_bufs(sess, sess.len, t)?,
            list: sess.idx_list.as_ref().map(|(b, r)| (b, *r)),
            cache: Some(&sess.idx_sel_cache),
            reuse,
        }))
    }

    /// QSA indexer of attention layer `li` for t tokens after `len` (input in sc.hn): pooled keys of the completed
    /// blocks, the tail of the last incomplete block and, when sparse, the block selection of every query.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn indexer(
        &self,
        st: &Stream,
        sc: &Scratch,
        ix: &Indexer,
        is: &IdxStates,
        li: usize,
        len: usize,
        t: usize,
        proj_ready: bool,
    ) -> Result<()> {
        let c = &self.cfg;
        let ic = c.indexer.as_ref().context("no indexer config")?;
        let [proj, qi, kr] = sc.idx.as_ref().context("no indexer scratch")?;
        let r = ic.ratio;
        let [_, sh, sw] = c.mrope_section;
        if !proj_ready {
            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, proj, Act::X(&sc.hn), &ix.qk, t, false)?;
        }
        self.k.get("idx_prep").launch(
            (t as u32, (ic.heads + 1) as u32, 1),
            (128, 1, 1),
            0,
            st,
            &[
                qi.into(),
                kr.into(),
                proj.into(),
                (&ix.q_norm).into(),
                (&sc.pos).into(),
                ic.heads.into(),
                c.eps.into(),
                c.rope_theta.into(),
                c.rot_dim.into(),
                sh.into(),
                sw.into(),
                self.rope_invf.as_ref().map_or(Arg::U64(0), |b| Arg::U64(b.addr())),
            ],
        )?;
        // Pooling the keys of every block that completes this step. The kernel reads len from the device buffer so the
        // node params stay constant for a future graph capture; the grid still follows the host-side new-block count
        // (making it capture-constant belongs with the capture wiring, since a fixed max grid is a pre-capture cost).
        let nb_new = (len + t) / r - len / r;
        if nb_new > 0 {
            self.k.get("idx_pool").launch(
                (nb_new as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &[
                    (&is.kb[li]).into(),
                    kr.into(),
                    (&is.tk[li]).into(),
                    (&is.tp[li]).into(),
                    (&sc.pos).into(),
                    (&ix.k_norm).into(),
                    (&sc.len_dev).into(),
                    r.into(),
                    c.eps.into(),
                    c.rope_theta.into(),
                    c.rot_dim.into(),
                    sh.into(),
                    sw.into(),
                    self.rope_invf.as_ref().map_or(Arg::U64(0), |b| Arg::U64(b.addr())),
                ],
            )?;
        }
        self.k.get("idx_tail").launch(
            (1, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[
                (&is.tk[li]).into(),
                (&is.tp[li]).into(),
                kr.into(),
                (&sc.pos).into(),
                (&sc.len_dev).into(),
                t.into(),
                r.into(),
            ],
        )?;
        if let Some((scores, sel, uni, w)) = is.sel {
            let ld = w * 32;
            let mut reused = false;
            if is.reuse
                && let Some(cache) = is.cache.and_then(|c| c.get(li))
            {
                sel.copy_from_shared(0, cache, 0, w * 4, st)?;
                reused = true;
            }
            if !reused {
                for q0 in (0..t).step_by(IDX_SUB) {
                    let nq = IDX_SUB.min(t - q0);
                    let maxb = (len + q0 + nq) / r;
                    let sel12 = std::env::var("MALLEUS_IDX_SELECT")
                        .map_or(self.k.arch == "gfx1151" && maxb >= 12288, |v| v == "12");
                    // One score/selection per group of four query rows: the representative is the last row, and
                    // the gather masks the future keys it may name for the earlier rows. At 256K this is 232 -> 185 s
                    // and makes the 32K-256K curve flat (1 453 -> 1 422 tok/s); MALLEUS_IDX_GROUP=1 restores
                    // per-query selection.
                    let want = std::env::var("MALLEUS_IDX_GROUP")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&v| (1..=16).contains(&v))
                        .unwrap_or(if self.k.arch == "gfx1151" { 4 } else { 1 });
                    // Decode and verify have nq = 1..6: there is no group to share, and the score tile's
                    // representative rows would not even be written. The group is a prefill-only shape and must
                    // divide the sub-batch exactly, so the partial last sub-batch also falls back to per-query.
                    let group = if sel12 && want > 1 && nq >= want && nq.is_multiple_of(want) {
                        want
                    } else {
                        1
                    };
                    let tiled = std::env::var("MALLEUS_IDX_SCORE_TILE").map_or(self.k.arch == "gfx1151", |v| v == "1")
                        && ic.heads == 4;
                    if tiled {
                        // 16 query rows read each 64-key tile once: 7.95 -> 5.27 s at 98K tokens, scores bit-identical
                        // to `idx_score` (same (c, e) order per score). Prefill reuses the q tile over 8 key tiles;
                        // decode needs the key parallelism more than the q reuse, so one key tile per block there.
                        let kt = std::env::var("MALLEUS_IDX_KT")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                            .filter(|&v| v > 0)
                            .unwrap_or(if nq >= 64 { 8 } else { 1 });
                        self.k.get("idx_score_tile").launch(
                            (maxb.div_ceil(64 * kt) as u32, nq.div_ceil(16) as u32, 1),
                            (1024, 1, 1),
                            0,
                            st,
                            &[
                                scores.into(),
                                qi.into(),
                                (&is.kb[li]).into(),
                                (&sc.len_dev).into(),
                                q0.into(),
                                r.into(),
                                ld.into(),
                                nq.into(),
                                kt.into(),
                                group.into(),
                            ],
                        )?;
                    } else {
                        // Tiled over queries (IDX_TQ rows share one read of the pooled keys): see `idx_score`.
                        const IDX_TQ: usize = 4;
                        self.k.get("idx_score").launch(
                            (maxb.div_ceil(256) as u32, nq.div_ceil(IDX_TQ) as u32, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                scores.into(),
                                qi.into(),
                                (&is.kb[li]).into(),
                                (&sc.len_dev).into(),
                                q0.into(),
                                ic.heads.into(),
                                r.into(),
                                ld.into(),
                                nq.into(),
                            ],
                        )?;
                    }
                    // The 4096-bin radix pays 16 KB of LDS and three histogram sweeps; it wins from ~64K blocks
                    // (a 64K-token context) and loses below, where the 256-bin four-pass kernel is cheaper. At 98K
                    // it is 3.67 -> 2.51 s with the same bitset.
                    if sel12 {
                        // One query (decode/verify) is one block on one CU: 1024 threads hide the row's latency;
                        // prefill keeps the 256-thread tile.
                        let (name, threads) = if nq == 1 {
                            ("idx_select12_1024", 1024)
                        } else {
                            ("idx_select12", 256)
                        };
                        self.k.get(name).launch(
                            (nq.div_ceil(group) as u32, 1, 1),
                            (threads, 1, 1),
                            0,
                            st,
                            &[
                                sel.into(),
                                scores.into(),
                                (&sc.len_dev).into(),
                                q0.into(),
                                r.into(),
                                (ic.budget / r).into(),
                                ld.into(),
                                w.into(),
                                group.into(),
                                nq.into(),
                            ],
                        )?;
                    } else {
                        self.k.get("idx_select").launch(
                            (nq as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                sel.into(),
                                scores.into(),
                                (&sc.len_dev).into(),
                                q0.into(),
                                r.into(),
                                (ic.budget / r).into(),
                                ld.into(),
                                w.into(),
                            ],
                        )?;
                    }
                    if let Ok(dir) = std::env::var("MALLEUS_IDX_DUMP_SEL")
                        && q0 + nq == t
                    {
                        // Debug for the recall gate: the selection bitsets of the last sub-batch, one file per
                        // indexer layer. Rows are q0..t of the current chunk.
                        st.synchronize()?;
                        let v: Vec<u32> = sel.download_at(q0 * w * 4, nq * w)?;
                        std::fs::create_dir_all(&dir)?;
                        let mut out = format!("{len} {q0} {nq} {w}\n");
                        for i in 0..nq {
                            for j in 0..w {
                                out.push_str(&format!("{:08x} ", v[i * w + j]));
                            }
                            out.push('\n');
                        }
                        std::fs::write(format!("{dir}/L{li}.txt"), out)?;
                    }
                }
            }
            if !reused
                && t == 1
                && let Some(cache) = is.cache.and_then(|c| c.get(li))
            {
                cache.copy_from_shared(0, sel, 0, w * 4, st)?;
            }
            if !reused && std::env::var("MALLEUS_IDX_STATS").is_ok_and(|v| v == "all" || li == 0) {
                // Debug: density of the selections and of their unions over 16 and 64 queries.
                st.synchronize()?;
                let v: Vec<u32> = sel.download(t * w)?;
                let pop = |r: &[u32]| r.iter().map(|x| x.count_ones() as usize).sum::<usize>();
                let ncb = (len + t) / r;
                let uni = |g: usize| -> f64 {
                    let mut tot = 0;
                    for q0 in (0..t).step_by(g) {
                        let mut u = vec![0u32; w];
                        for q in q0..(q0 + g).min(t) {
                            for (a, b) in u.iter_mut().zip(&v[q * w..(q + 1) * w]) {
                                *a |= b;
                            }
                        }
                        tot += pop(&u);
                    }
                    tot as f64 / t.div_ceil(g) as f64
                };
                eprintln!(
                    "idx stats layer {li} len {len} t {t} blocks {ncb}: per query {:.0}, union4 {:.0}, union16 {:.0}, \
                     union64 {:.0}",
                    pop(&v) as f64 / t as f64,
                    uni(4),
                    uni(16),
                    uni(64)
                );
            }
            if t >= ATTN_WMMA_MIN {
                self.k.get("idx_union").launch(
                    (w.div_ceil(256) as u32, t.div_ceil(16) as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[uni.into(), sel.into(), t.into(), w.into()],
                )?;
            }
        }
        self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}
