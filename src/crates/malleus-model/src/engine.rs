//! The shared graph: one `Model` that loads a checkpoint of any family and runs it.

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::TensorSource;
use malleus_hip::{Arg, DeviceBuffer, Stream};

use crate::config::*;
use crate::kernels::Kernels;
use crate::layer::*;
use crate::qwen4exp::{Hc, NgramTable, load_hc};
use crate::qwen35::rope_invf;
use crate::session::*;
use crate::upload::Uploader;
use crate::weights::*;

pub struct Model {
    pub cfg: Config,
    pub k: Kernels,
    pub(crate) embed: DeviceBuffer,
    pub(crate) layers: Vec<Layer>,
    /// Final norm, or the final hyper-connection mixer of the streams.
    pub(crate) norm: Option<DeviceBuffer>,
    pub(crate) hc_final: Option<Hc>,
    pub(crate) ngram: Option<NgramTable>,
    pub(crate) lm_head: Linear,
    pub(crate) mtp: Option<Mtp>,
    /// Feed the MTP head the final-normed hidden state instead of the last layer output.
    pub(crate) mtp_post_norm: bool,
    /// Chained drafts start from the MTP output after its final norm.
    pub(crate) mtp_chain_post: bool,
    /// 4-bit copy of the MLS8 lm_head scoring the MTP drafts (half the bytes per draft).
    pub(crate) draft_head: Option<Linear>,
    /// Token id of every draft_head row when drafts are scored on a token list (MALLEUS_DRAFT_VOCAB_FILE).
    pub(crate) draft_map: Option<Vec<u32>>,
    pub(crate) signs: DeviceBuffer,
    /// Inverse RoPE frequencies blended by YaRN (`MALLEUS_ROPE_YARN` > 1): `rot_dim / 2` fp32, or None for the plain
    /// `1 / theta^(2j/rot)` the kernels compute inline. The blend depends on the wavelength of every index, so it is
    /// a table rather than a closed form.
    pub(crate) rope_invf: Option<DeviceBuffer>,
    pub(crate) scratch_cache: std::sync::Mutex<Option<Scratch>>,
    /// The stream every `forward` runs on.
    pub(crate) stream: Stream,
    /// Format of the activations last quantized into the scratch: 0 none, 1 per-row int8, 2 per-group int8.
    pub(crate) quant_mode: std::sync::atomic::AtomicU8,
    /// Prefill GEMMs on int8 activations (W4A8 / W8A8) instead of fp16 activations.
    pub prefill_a8: bool,
    /// Inputs up to this many tokens use the 4-bit weights (W4A8): half the bytes of the int8 copy.
    pub w4_max_t: usize,
    /// Prefill MoE with int4 activations (W4A4): the int4 WMMA is double rate on gfx11 and the expert nibbles
    /// reach it unmodified. MALLEUS_MOE_W4A4=0 keeps W4A8, `glu` limits int4 to the gate/up product.
    pub moe_w4a4: u8,
    /// `y` of the grouped MoE prefill (the expert output of every (token, slot) pair) in fp16 instead of f32,
    /// MALLEUS_MOE_Y_HALF=1: half the bytes of the widest intermediate of the prefill, which the down GEMM writes
    /// and `moe_combine` reads back (755 MB a layer at pp8192, and that kernel measured 72 % of the memory wall).
    pub moe_y_half: bool,
    pub weight_bytes: usize,
    /// `MALLEUS_EXPERT_CACHE`: one full layer of routed experts, streamed from the file at every prefill visit
    /// (a grouped GEMM needs all of them at once). Shared by every MoE layer, allocated once.
    pub(crate) expert_stage: Option<std::sync::Mutex<crate::expert_cache::Stage>>,
    /// The file carries MLS8 copies of the weights for the prefill (`--prefill-int8` at conversion). A single-copy
    /// file runs the prompt on the MLS4 nibbles: faster, ~35 GB lighter on a 35B MoE, and measurably lower quality.
    pub prefill_copies: bool,
    /// Diagnostics `MALLEUS_ATTN_MASS`: (buffer, sampling stride, tiles per row). The attention kernel writes into it,
    /// for one query row every `mstride`, the (max, sum of exp) of every tile of 32 keys, from which the attention
    /// mass per block is recovered. It does not touch the hot path: without the variable it is None and the kernel
    /// receives a null pointer.
    pub(crate) attn_mass: Option<(DeviceBuffer, usize, usize)>,
    pub vision: Option<crate::vision::VisionModel>,
    /// Hy4-flash weights: its own graph (`crate::hy4`), not the `Layer`/`Mixer` shape of the carried models.
    pub(crate) hy4: Option<Box<crate::hy4::Weights>>,
    /// Hy4-flash's vision tower (`vit.*`), when the file carries one.
    pub(crate) hy4_vision: Option<Box<crate::hy4::Vision>>,
    /// Inkling-2's weights: its own graph (`crate::inkling2`), read by a standalone forward until the
    /// engine's session plumbing carries the loop.
    pub(crate) inkling2: Option<Box<crate::inkling2::Weights>>,
    /// DeepSeek-V4.1's host graph (`crate::deepseek_text`): a streamed backbone and a decode cache, both on the
    /// CPU and both owned here, so a served model outlives the loader that opened the files. Behind a mutex
    /// because a step borrows the streamer mutably and the engine holds the model behind `&self`.
    pub(crate) deepseek: Option<std::sync::Mutex<Deepseek>>,
}

/// DeepSeek-V4.1's host graph, as a served model holds it: the streamer (the backbone, the Engram tables, their
/// addressing) and the sequence's KV cache. Nothing of it lives on the device.
pub struct Deepseek {
    pub text: crate::deepseek_text::Streamer,
    pub cache: crate::deepseek_text::Cache,
}

/// The DeepSeek-V4.1 files the host graph reads: the backbone, the companion file with the two Engram tables, and
/// the prefix of the Engram state (`<prefix>-state.json`, `<prefix>-token-map.i32`).
#[derive(Clone, Debug)]
pub struct DeepseekFiles {
    pub backbone: std::path::PathBuf,
    pub engram: std::path::PathBuf,
    pub state: std::path::PathBuf,
}

pub struct LoadOptions {
    pub prefix: String,
    pub prefill_int8: bool,
    /// Load the MLS8 prefill copy only for linears whose name contains one of these (empty = all).
    pub prefill_families: Vec<String>,
    pub hadamard_seed: u64,
    /// Load the multi-token prediction head (speculative decoding) when the file has it.
    pub mtp: bool,
    /// DeepSeek-V4.1: its graph owns its sources, so the loader is given the paths instead of an opened backbone.
    pub deepseek: Option<DeepseekFiles>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            prefix: "model.language_model.".into(),
            prefill_int8: true,
            prefill_families: Vec::new(),
            hadamard_seed: malleus_gguf::quant::HADAMARD_SEED,
            mtp: true,
            deepseek: None,
        }
    }
}

/// Rows per tile of the grouped MoE GEMMs (moe_i8*).
pub(crate) const MOE_BM: usize = 64;

pub(crate) static MOE_BM128_ROWS_ENV: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("MALLEUS_MOE_BM128")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48)
});

pub enum Act<'a> {
    X(&'a DeviceBuffer),
    Same(&'a DeviceBuffer),
    SiluMul(&'a DeviceBuffer, &'a DeviceBuffer),
    /// RMSNorm(x) with weight (1 + w), applied while quantizing (int8 row path only, as the variants below).
    Norm(&'a DeviceBuffer, &'a DeviceBuffer),
    /// Per-head RMSNorm(x) * w * silu(z) (DeltaNet output).
    GatedNorm(&'a DeviceBuffer, &'a DeviceBuffer, &'a DeviceBuffer),
    /// x * sigmoid(gate), gate interleaved in the q projection output.
    AttnGate(&'a DeviceBuffer, &'a DeviceBuffer),
}

impl Model {
    pub fn load(
        cfg: Config,
        src: &dyn TensorSource,
        opts: &LoadOptions,
        kernels: Kernels,
        progress: &dyn Fn(usize, usize),
    ) -> Result<Self> {
        let c = &cfg;
        if c.family == Family::Hy4 {
            return Self::load_hy4(cfg, src, opts, kernels, progress);
        }
        // DeepSeek-V4.1 has its own graph (`crate::deepseek_text`), which streams the backbone from the file: it
        // owns its sources, so it opens them itself and none of the device loader below is involved.
        if c.family == Family::Deepseek41 {
            return Self::load_deepseek(cfg, opts, kernels);
        }
        // Inkling-2 has its own graph (`crate::inkling2`): check the file against the map, load the weights, and
        // hand the model to the standalone forward.
        if c.family == Family::Inkling2 {
            for line in crate::inkling2::check(src, c)? {
                eprintln!("{line}");
            }
            let t0 = std::time::Instant::now();
            let weights = crate::inkling2::Weights::load(src, c, progress)?;
            let n = weights.layers.len();
            let gb = weights.bytes as f64 / (1u64 << 30) as f64;
            eprintln!(
                "inkling2: {n} layers loaded, {gb:.1} GiB on the device in {:.0}s",
                t0.elapsed().as_secs_f64()
            );
            let mut m = Self::host_only(cfg, kernels, weights.bytes, opts.hadamard_seed)?;
            m.inkling2 = Some(Box::new(weights));
            return Ok(m);
        }
        // Read before `kernels` moves into the model: the int4 default for the MoE prefill depends on the GPU.
        let arch = kernels.arch.clone();
        // Ling stores its weights under `model.` and calls the embedding `word_embeddings`; Qwen nests them
        // under `model.language_model.`. Everything else in the loader goes through `p`.
        let ling = c.family == Family::Ling3;
        let prefix = if ling { "model." } else { opts.prefix.as_str() };
        let mut up = Uploader::new()?;
        up.use_prefill = opts.prefill_int8;
        up.prefill_families = opts.prefill_families.clone();
        let signs = DeviceBuffer::from_slice(&malleus_gguf::quant::hadamard_signs(opts.hadamard_seed))?;
        let rope_invf = rope_invf(c)?.map(|t| DeviceBuffer::from_slice(&t)).transpose()?;
        let p = |s: &str| format!("{prefix}{s}");
        let embed = up.tensor(
            src,
            &p(if ling {
                "word_embeddings.weight"
            } else {
                "embed_tokens.weight"
            }),
            &[c.vocab, c.hidden],
        )?;
        // Decode on the MLS8 prefill copies of the non-expert linears (quality over speed).
        up.decode_int8 = std::env::var("MALLEUS_DECODE_INT8").is_ok_and(|v| v == "1");
        // `MALLEUS_EXPERT_CACHE=<GiB>` (issue #14): the routed experts stay on disk, one pool per MoE layer.
        // Models whose shared expert is stacked into the same matrix (Qwen's `shared_expert_gate`) are not
        // supported yet: their `experts + 1` rows are not all in the four tensors the cache reads, and guessing
        // wrong there would silently drop the shared expert.
        let cache_gib = crate::expert_cache::gib();
        let n_moe = c
            .n_layers
            .saturating_sub(c.dense_ffn.min(c.n_layers))
            .saturating_add(usize::from(opts.mtp))
            .max(1);
        let cache_budget = if cache_gib > 0.0 {
            (cache_gib * (1u64 << 30) as f64 / n_moe as f64) as usize
        } else {
            0
        };
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            progress(i, c.n_layers);
            layers.push(load_layer(
                &mut up,
                src,
                c,
                &p(&format!("layers.{i}.")),
                c.kind[i],
                i,
                opts.hadamard_seed,
                c.ple.as_ref().is_some_and(|pc| pc.layer == i),
                cache_budget,
            )?);
        }
        let (norm, hc_final) = match &c.hc {
            Some(_) => (None, Some(load_hc(&mut up, src, c, &p("hyper_connection_mixer."))?)),
            None => (
                Some(up.norm(src, &p("norm.weight"), c.hidden, c.norm_one_plus_w)?),
                None,
            ),
        };
        let ngram = match &c.ple {
            Some(pc) => Some(NgramTable::load(
                src,
                &p(&format!("layers.{}.ple.ple_embedding.", pc.layer)),
                pc,
            )?),
            None => None,
        };
        let mut lm_head = up.linear(src, "lm_head", c.vocab, c.hidden)?;
        if std::env::var("MALLEUS_LMHEAD_W4").is_ok_and(|v| v == "1")
            && let Weights::Mls8 { q, sc } = &lm_head.w
        {
            lm_head = mls8_to_mls4(q, sc, c.vocab, c.hidden)?;
        }
        // The draft head reads its 8-bit prefill copy by default: on code, JSON and lists the draft chain proposes
        // two to three times more tokens per round and three quarters of them are accepted, where with the 4-bit
        // weights it stops at the second draft (measured, bench/baseline.md). `MALLEUS_MTP_INT8=0` keeps 4 bits.
        up.decode_int8 = std::env::var("MALLEUS_MTP_INT8").map_or(true, |v| v != "0");
        let mtp = if opts.mtp && src.info("mtp.fc.weight").is_some() {
            Some(Mtp {
                fc: Some(up.linear(src, "mtp.fc", c.hidden, 2 * c.hidden)?),
                fc_e: None,
                fc_h: None,
                pre_e: up.tensor(src, "mtp.pre_fc_norm_embedding.weight", &[c.hidden])?,
                pre_h: up.tensor(src, "mtp.pre_fc_norm_hidden.weight", &[c.hidden])?,
                norm: Some(up.tensor(src, "mtp.norm.weight", &[c.hidden])?),
                mixer: None,
                layer: load_layer(
                    &mut up,
                    src,
                    c,
                    "mtp.layers.0.",
                    MixerKind::Attn,
                    usize::MAX,
                    opts.hadamard_seed,
                    false,
                    cache_budget,
                )?,
            })
        } else if opts.mtp && c.hc.is_some() && src.info("mtp.fc_embedding.weight").is_some() {
            Some(Mtp {
                fc: None,
                fc_e: Some(up.linear(src, "mtp.fc_embedding", c.hidden, c.hidden)?),
                fc_h: Some(up.linear(src, "mtp.fc_hidden", c.hidden, c.hidden)?),
                pre_e: up.tensor(src, "mtp.pre_fc_norm_embedding.weight", &[c.hidden])?,
                pre_h: up.tensor(src, "mtp.pre_fc_norm_hidden.weight", &[c.stream_dim()])?,
                norm: None,
                mixer: Some(load_hc(&mut up, src, c, "mtp.hyper_connection_mixer.")?),
                layer: load_layer(
                    &mut up,
                    src,
                    c,
                    "mtp.layers.0.",
                    MixerKind::Attn,
                    usize::MAX,
                    opts.hadamard_seed,
                    false,
                    cache_budget,
                )?,
            })
        } else {
            None
        };
        // Drafts are scored on the first `MALLEUS_DRAFT_VOCAB` tokens (ids follow the BPE merge order, so the most
        // frequent ones come first): a fraction of the lm_head bytes per draft.
        let draft_vocab = std::env::var("MALLEUS_DRAFT_VOCAB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(c.vocab)
            .clamp(1024, c.vocab);
        // A token list (one id per line, most frequent first) restricts the drafts to those rows of lm_head.
        let draft_list: Option<Vec<u32>> = match std::env::var("MALLEUS_DRAFT_VOCAB_FILE") {
            Ok(f) if mtp.is_some() => {
                let mut ids: Vec<u32> = std::fs::read_to_string(&f)
                    .with_context(|| f.clone())?
                    .lines()
                    .filter_map(|l| l.trim().parse().ok())
                    .filter(|&i: &u32| (i as usize) < c.vocab)
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                Some(ids)
            }
            _ => None,
        };
        let (draft_head, draft_map) = match (&mtp, &lm_head.w) {
            (Some(_), Weights::Mls8 { q, sc }) if std::env::var("MALLEUS_DRAFT_HEAD").map_or(true, |v| v != "int8") => {
                match draft_list {
                    Some(ids) => {
                        let k = c.hidden;
                        let (q8, s8): (Vec<i8>, Vec<f32>) = (q.download(c.vocab * k)?, sc.download(c.vocab)?);
                        let mut qg = Vec::with_capacity(ids.len() * k);
                        let mut sg = Vec::with_capacity(ids.len());
                        for &i in &ids {
                            qg.extend_from_slice(&q8[i as usize * k..(i as usize + 1) * k]);
                            sg.push(s8[i as usize]);
                        }
                        let (qb, sb) = (DeviceBuffer::from_slice(&qg)?, DeviceBuffer::from_slice(&sg)?);
                        (Some(mls8_to_mls4(&qb, &sb, ids.len(), k)?), Some(ids))
                    }
                    None => (Some(mls8_to_mls4(q, sc, draft_vocab, c.hidden)?), None),
                }
            }
            // lm_head is already MLS4 (MALLEUS_LMHEAD_W4): gather only the rows that would make the draft head
            // differ from it. With no token list and a full draft vocabulary it would be a copy of lm_head, and
            // `None` already scores the drafts on lm_head itself.
            (Some(_), Weights::Mls4 { q, sc })
                if std::env::var("MALLEUS_DRAFT_HEAD").map_or(true, |v| v != "int8")
                    && (draft_list.is_some() || draft_vocab < c.vocab) =>
            {
                let ids = draft_list.clone().unwrap_or_else(|| (0..draft_vocab as u32).collect());
                (Some(mls4_rows(q, sc, &ids, c.vocab, c.hidden)?), draft_list)
            }
            _ => (None, None),
        };
        // The cache's shared staging (one full layer) and the slot-id buffer of the decode path.
        let expert_stage = max_cache_geom(&layers, mtp.as_ref())
            .map(crate::expert_cache::Stage::alloc)
            .transpose()?
            .map(std::sync::Mutex::new);
        if let Some(g) = expert_stage.as_ref().map(|s| crate::expert_cache::lock(s).bytes()) {
            let pool: usize = layers
                .iter()
                .chain(mtp.as_ref().map(|m| &m.layer))
                .map(|l| match &l.ffn {
                    Ffn::Moe(m) => m.cache.as_ref().map_or(0, |c| c.bytes()),
                    _ => 0,
                })
                .sum();
            eprintln!(
                "expert cache: {:.1} GiB of pools + {:.1} GiB of staging instead of the routed experts",
                pool as f64 / (1u64 << 30) as f64,
                g as f64 / (1u64 << 30) as f64
            );
        }
        let cache_bytes: usize = layers
            .iter()
            .chain(mtp.as_ref().map(|m| &m.layer))
            .map(|l| match &l.ffn {
                Ffn::Moe(m) => m.cache.as_ref().map_or(0, |c| c.bytes()),
                _ => 0,
            })
            .sum::<usize>()
            + expert_stage
                .as_ref()
                .map_or(0, |s| crate::expert_cache::lock(s).bytes());
        progress(c.n_layers, c.n_layers);
        // `y` in fp16 needs one width for the whole process, and the int8 grouped path (the one a checkpoint with
        // MLS8 expert copies takes) has no fp16 variant: there the flag is ignored rather than half applied. Which
        // path a checkpoint takes is a property of its experts, not of the prompt length.
        let moe_y_half = std::env::var("MALLEUS_MOE_Y_HALF").is_ok_and(|v| v != "0")
            && !layers.iter().any(|l| matches!(&l.ffn, Ffn::Moe(m) if m.pre.is_some()));
        // Diagnostics of the attention concentration (#10): sized for 262144 tokens of context, one sampled row every
        // 4096, a tile of 32 keys. ~4 MB.
        let attn_mass = match std::env::var_os("MALLEUS_ATTN_MASS") {
            Some(_) => {
                let num = |k: &str, d: usize| {
                    std::env::var(k)
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|v| *v > 0)
                        .unwrap_or(d)
                };
                let (mstride, ctx_max) = (
                    num("MALLEUS_ATTN_MASS_STRIDE", 4096),
                    num("MALLEUS_ATTN_MASS_MAX", 262_144),
                );
                let mtiles = ctx_max / 32 + 2;
                let rows = ctx_max / mstride + 2;
                Some((DeviceBuffer::zeroed(rows * mtiles * 8)?, mstride, mtiles))
            }
            None => None,
        };
        Ok(Model {
            weight_bytes: up.bytes + cache_bytes,
            expert_stage,
            attn_mass,
            prefill_copies: up.prefill_loaded,
            cfg,
            k: kernels,
            embed,
            layers,
            norm,
            hc_final,
            ngram,
            lm_head,
            mtp,
            mtp_post_norm: std::env::var("MALLEUS_MTP_HIDDEN").is_ok_and(|v| v == "post"),
            mtp_chain_post: std::env::var("MALLEUS_MTP_CHAIN").map_or(true, |v| v != "pre"),
            draft_head,
            draft_map,
            signs,
            rope_invf,
            scratch_cache: std::sync::Mutex::new(None),
            stream: Stream::new()?,
            quant_mode: std::sync::atomic::AtomicU8::new(0),
            prefill_a8: std::env::var("MALLEUS_PREFILL_GEMM").map_or(true, |v| v != "f16"),
            w4_max_t: std::env::var("MALLEUS_W4_MAX_T")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            // int4 activations for the MoE prefill are double rate on gfx1151 and about twice as slow as int8 on
            // gfx1150 for this model's shapes (measured: docs, "Portare il lavoro di gfx1151 su gfx1150"), so the
            // default follows the GPU: a file without the int8 expert copies (W4A8) then runs the fast path here
            // without needing the environment.
            moe_w4a4: match std::env::var("MALLEUS_MOE_W4A4").unwrap_or_default().as_str() {
                "0" => 0,
                "glu" => 2,
                // An empty variable counts as unset: a container that forwards an optional setting passes it as
                // an empty string, and that must not silently pick the int4 path (measured 662 against 1062 tok/s
                // of prefill at pp2048 on gfx1150).
                "" if arch == "gfx1150" => 0,
                _ => 1,
            },
            moe_y_half,
            vision: None,
            hy4: None,
            hy4_vision: None,
            inkling2: None,
            deepseek: None,
        })
    }

    /// A model whose graph is not the device stack: every weight field is empty and only the config, the stream and
    /// the small device buffers the engine expects are real. Inkling-2 and DeepSeek-V4.1 both fill their own field
    /// afterwards.
    pub(crate) fn host_only(cfg: Config, kernels: Kernels, weight_bytes: usize, hadamard_seed: u64) -> Result<Self> {
        Ok(Model {
            cfg,
            k: kernels,
            embed: DeviceBuffer::zeroed(4)?,
            layers: Vec::new(),
            norm: None,
            hc_final: None,
            ngram: None,
            lm_head: Linear {
                w: crate::weights::Weights::Bf16(DeviceBuffer::zeroed(4)?),
                prefill: None,
                n: 1,
                k: 1,
            },
            mtp: None,
            mtp_post_norm: false,
            mtp_chain_post: false,
            draft_head: None,
            draft_map: None,
            signs: DeviceBuffer::from_slice(&malleus_gguf::quant::hadamard_signs(hadamard_seed))?,
            rope_invf: None,
            scratch_cache: std::sync::Mutex::new(None),
            stream: Stream::new()?,
            quant_mode: std::sync::atomic::AtomicU8::new(0),
            prefill_a8: false,
            w4_max_t: 0,
            moe_w4a4: 0,
            moe_y_half: false,
            weight_bytes,
            expert_stage: None,
            prefill_copies: false,
            attn_mass: None,
            vision: None,
            hy4: None,
            hy4_vision: None,
            inkling2: None,
            deepseek: None,
        })
    }

    /// DeepSeek-V4.1's host graph (issue #83): the backbone and the Engram tables are owned by the streamer, the
    /// decode cache is allocated once, and every block's non-expert weights are held resident because a served
    /// step would otherwise read 40 blocks of attention back from the file per token. `MALLEUS_DEEPSEEK_EXPERTS`
    /// is how many routed experts of one block the cache holds across positions.
    fn load_deepseek(cfg: Config, opts: &LoadOptions, kernels: Kernels) -> Result<Self> {
        let files = opts.deepseek.as_ref().context(
            "deepseek_v41: the host graph reads its own files; the loader needs the backbone, the Engram tables \
             (MALLEUS_DEEPSEEK_ENGRAM) and the Engram state (MALLEUS_DEEPSEEK_ENGRAM_STATE)",
        )?;
        let t0 = std::time::Instant::now();
        let backbone = open_source(&files.backbone)?;
        let engram: crate::deepseek_text::Source = Box::new(
            malleus_gguf::gguf::Gguf::open(&files.engram).with_context(|| files.engram.display().to_string())?,
        );
        let state = crate::deepseek_engram::EngramState::from_json(
            &std::fs::read_to_string(format!("{}-state.json", files.state.display()))
                .with_context(|| format!("{}-state.json", files.state.display()))?,
            &std::fs::read(format!("{}-token-map.i32", files.state.display()))
                .with_context(|| format!("{}-token-map.i32", files.state.display()))?,
        )?;
        let hold = std::env::var("MALLEUS_DEEPSEEK_EXPERTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let text = crate::deepseek_text::Streamer::open(backbone, &cfg)?
            .with_engram(engram, state)
            .with_experts_held(hold)
            .with_blocks()?;
        let gb = text.resident_bytes() as f64 / 1e9;
        let cache = text.cache(hold);
        eprintln!(
            "deepseek_v41: host graph, {gb:.1} GB of weights resident (the blocks included), up to {hold} experts of \
             one block held per layer, ready in {:.0}s",
            t0.elapsed().as_secs_f64()
        );
        let mut m = Self::host_only(cfg, kernels, text.resident_bytes(), opts.hadamard_seed)?;
        m.deepseek = Some(std::sync::Mutex::new(Deepseek { text, cache }));
        Ok(m)
    }

    /// Hy4-flash: the whole model file is the text graph (plus the vision tower and the DFly2 draft, which this
    /// increment does not load). The loader proves the tensor map and the footprint before the forward exists.
    fn load_hy4(
        cfg: Config,
        src: &dyn TensorSource,
        opts: &LoadOptions,
        kernels: Kernels,
        progress: &dyn Fn(usize, usize),
    ) -> Result<Self> {
        let c = &cfg;
        let mut up = Uploader::new()?;
        // A file converted with `--prefill-int8` carries the MLS8 copies of the dense linears (the routed experts
        // never do: they are ~75 GB, see `convert`). `MALLEUS_PREFILL_WEIGHTS=none` leaves them on the host and
        // gives the 5.9 GiB back to a 1M session that would rather spend it on state.
        up.use_prefill = opts.prefill_int8;
        up.prefill_families = opts.prefill_families.clone();
        let embed = up.tensor(src, "model.embed_tokens.weight", &[c.vocab, c.hidden])?;
        let lm_head = up.linear(src, "lm_head", c.vocab, c.hidden)?;
        let weights = crate::hy4::Weights::load(src, c, &mut up, opts.hadamard_seed, progress)?;
        let vision = match &c.hy4.as_ref().context("no hy4 config")?.vision {
            Some(v) if src.info("vit.embeddings.patch_embedding.weight").is_some() => {
                Some(Box::new(crate::hy4::Vision::load(src, &mut up, v)?))
            }
            _ => None,
        };
        progress(c.n_layers, c.n_layers);
        Ok(Model {
            weight_bytes: up.bytes,
            expert_stage: None,
            attn_mass: None,
            prefill_copies: up.prefill_loaded,
            cfg,
            k: kernels,
            embed,
            layers: Vec::new(),
            norm: None,
            hc_final: None,
            ngram: None,
            lm_head,
            mtp: None,
            mtp_post_norm: false,
            mtp_chain_post: false,
            draft_head: None,
            draft_map: None,
            signs: DeviceBuffer::from_slice(&malleus_gguf::quant::hadamard_signs(opts.hadamard_seed))?,
            rope_invf: None,
            scratch_cache: std::sync::Mutex::new(None),
            stream: Stream::new()?,
            quant_mode: std::sync::atomic::AtomicU8::new(0),
            prefill_a8: false,
            w4_max_t: 0,
            moe_w4a4: 0,
            moe_y_half: false,
            vision: None,
            hy4: Some(Box::new(weights)),
            hy4_vision: vision,
            inkling2: None,
            deepseek: None,
        })
    }

    pub fn new_session(&self, max_ctx: usize) -> Result<Session> {
        if self.hy4.is_some() {
            return crate::hy4::new_session(self, max_ctx);
        }
        // The DeepSeek-V4.1 host graph keeps its own cache and serves one request at a time: there is no session.
        if self.deepseek.is_some() {
            bail!("deepseek_v41: the host graph serves without sessions");
        }
        // Inkling-2 keeps its keys in its own cache, so the generic per-layer state below is not allocated for it:
        // 20 sliding layers of `kv` that the looped graph never read, at 80 KiB a position (issue #110).
        if self.inkling2.is_some() {
            return crate::inkling2::new_session(self, max_ctx);
        }
        let c = &self.cfg;
        let mut s = Session {
            max_ctx,
            len: 0,
            next_pos: 0,
            kv: vec![],
            conv: vec![],
            rec: vec![],
            attn_part: f32s((ATTN_WMMA_MIN - 1) * c.n_heads * kv_ctx(max_ctx).div_ceil(ATTN_SPLIT) * (c.head_dim + 2))?,
            mla_part: match &c.mla {
                Some(m) => Some(f32s(
                    GEMV_MAX_T * c.n_heads * kv_ctx(max_ctx).div_ceil(ATTN_SPLIT) * (m.kv_lora + 2),
                )?),
                None => None,
            },
            conv_snap: vec![],
            dn_rec: vec![],
            dn_np: 0,
            dn_par: 0,
            snap_t: 0,
            mtp_kv: vec![],
            mtp_len: 0,
            mtp_h: None,
            mtp_pos: vec![],
            mtp_out: None,
            mtp_out_pos: [0; 3],
            mtp_chain: 0,
            ple_state: match &c.ple {
                Some(p) => Some(DeviceBuffer::zeroed(p.conv_state() * c.stream_dim() * 4)?),
                None => None,
            },
            ngram_ctx: c.ple.as_ref().map_or(vec![], |p| vec![p.eos; p.ngram - 1]),
            idx_kb: vec![],
            idx_tk: vec![],
            idx_tp: vec![],
            idx_scores: None,
            idx_sel: None,
            idx_sel_cache: vec![],
            idx_cache_len: 0,
            mtp_idx: None,
            idx_list: None,
            ple_snap: None,
            ngram_prev: vec![],
            inkling2: None,
            hy4_spec: None,
            hy4_draft: Default::default(),
            ix_tk: vec![],
            ix_gk: vec![],
            ix_pooled: vec![],
            ix_pos: None,
        };
        if let Some(ic) = &c.indexer {
            let idx_bufs = || -> Result<[DeviceBuffer; 3]> {
                Ok([
                    DeviceBuffer::zeroed((max_ctx / ic.ratio + 1) * ic.dim * 2)?,
                    DeviceBuffer::zeroed(ic.ratio * ic.dim * 4)?,
                    DeviceBuffer::zeroed(ic.ratio * 3 * 4)?,
                ])
            };
            let idx_w = self.idx_words(max_ctx);
            for _ in c.full_attn.iter().filter(|f| **f) {
                let [kb, tk, tp] = idx_bufs()?;
                s.idx_kb.push(kb);
                s.idx_tk.push(tk);
                s.idx_tp.push(tp);
                s.idx_sel_cache.push(DeviceBuffer::zeroed(idx_w * 4)?);
            }
            if self.mtp.is_some() {
                s.mtp_idx = Some(idx_bufs()?);
            }
            s.idx_list = Some((
                DeviceBuffer::alloc(ATTN_WMMA_MIN * (ic.budget / ic.ratio + 1) * 4)?,
                ATTN_WMMA_MIN,
            ));
        }
        if let (Some(p), Some(_)) = (&c.ple, &self.mtp) {
            s.ple_snap = Some(DeviceBuffer::alloc((p.conv_state() + GEMV_MAX_T) * c.stream_dim() * 4)?);
        }
        let kv_bytes = kv_ctx(max_ctx) * c.n_kv * c.head_dim * 2 + 64;
        // With the indexer, the V buffer also holds V row-major after V^T (gathered by the sparse attention).
        let v_bytes = if c.indexer.is_some() { 2 * kv_bytes } else { kv_bytes };
        let (conv_b, rec_b) = (c.lin_qkv_dim() * (c.conv_k - 1) * 4, c.lin_nv * c.lin_dk * c.lin_dv * 4);
        // MLA stores the latent and the rotated dims apart: 512 + 64 fp16 per token instead of the 20 KiB per
        // layer and token the HF code would spend on K and V (docs/model-ling3-flash.md, § MLA).
        let mla_kv = c.mla.map(|m| kv_ctx(max_ctx) * m.kv_lora * 2).unwrap_or(0);
        let mla_rot = c.mla.map(|m| kv_ctx(max_ctx) * m.rope * 2).unwrap_or(0);
        for &kind in &c.kind {
            if kind == MixerKind::Attn || kind == MixerKind::Mla {
                let (kb, vb) = if kind == MixerKind::Attn {
                    (kv_bytes, v_bytes)
                } else {
                    (mla_kv, mla_rot)
                };
                s.kv.push((DeviceBuffer::zeroed(kb)?, DeviceBuffer::zeroed(vb)?));
            } else {
                s.conv.push(DeviceBuffer::zeroed(conv_b)?);
                s.rec.push(DeviceBuffer::zeroed(rec_b)?);
                if self.mtp.is_some() {
                    s.conv_snap.push(DeviceBuffer::alloc((GEMV_MAX_T - 1) * conv_b)?);
                    let rec = (c.lin_nk * 128 + c.lin_nv * 129) * 4 * GEMV_MAX_T;
                    s.dn_rec.push([DeviceBuffer::alloc(rec)?, DeviceBuffer::alloc(rec)?]);
                }
            }
        }
        if self.mtp.is_some() {
            s.mtp_kv
                .push((DeviceBuffer::zeroed(kv_bytes)?, DeviceBuffer::zeroed(v_bytes)?));
            s.mtp_h = Some(f32s(GEMV_MAX_T * c.stream_dim())?);
            s.mtp_out = Some(f32s(c.stream_dim())?);
        }
        Ok(s)
    }

    /// An empty snapshot with one buffer per state of this session, to be refilled by `snapshot_into`. The
    /// scheduler pools these and recycles them (`Model::snapshot_into`) because allocating them per request
    /// dominates the cost of saving a prompt.
    pub fn new_snapshot(&self, s: &Session) -> Result<Snapshot> {
        let a = |b: &DeviceBuffer| -> Result<DeviceBuffer> { Ok(DeviceBuffer::alloc(b.len())?) };
        Ok(Snapshot {
            len: 0,
            next_pos: 0,
            conv: s.conv.iter().map(a).collect::<Result<_>>()?,
            rec: s.rec.iter().map(a).collect::<Result<_>>()?,
            ple_state: s.ple_state.as_ref().map(a).transpose()?,
            ngram_ctx: Vec::new(),
            idx_tk: s.idx_tk.iter().map(a).collect::<Result<_>>()?,
            idx_tp: s.idx_tp.iter().map(a).collect::<Result<_>>()?,
            mtp_len: 0,
            mtp_h: s.mtp_h.as_ref().map(a).transpose()?,
            mtp_pos: Vec::new(),
            mtp_idx: match &s.mtp_idx {
                Some([_, tk, tp]) => Some([a(tk)?, a(tp)?]),
                None => None,
            },
        })
    }

    /// Copies the states of `s` into a snapshot made by `new_snapshot` on a session of the same model, reusing
    /// the buffers it already holds.
    pub fn snapshot_into(&self, s: &Session, snap: &mut Snapshot) -> Result<()> {
        ensure!(s.dn_np == 0 && s.snap_t == 0, "snapshot during a speculative step");
        let st = Stream::null();
        copy_states(&mut snap.conv, &s.conv, &st)?;
        copy_states(&mut snap.rec, &s.rec, &st)?;
        copy_states(&mut snap.idx_tk, &s.idx_tk, &st)?;
        copy_states(&mut snap.idx_tp, &s.idx_tp, &st)?;
        copy_state_opt(&mut snap.ple_state, &s.ple_state, &st)?;
        copy_state_opt(&mut snap.mtp_h, &s.mtp_h, &st)?;
        match &s.mtp_idx {
            Some([_, tk, tp]) => {
                let fresh = snap
                    .mtp_idx
                    .as_ref()
                    .is_none_or(|[a, b]| a.len() != tk.len() || b.len() != tp.len());
                if fresh {
                    snap.mtp_idx = Some([DeviceBuffer::alloc(tk.len())?, DeviceBuffer::alloc(tp.len())?]);
                }
                if let Some([a, b]) = snap.mtp_idx.as_mut() {
                    a.copy_from(0, tk, 0, tk.len(), &st)?;
                    b.copy_from(0, tp, 0, tp.len(), &st)?;
                }
            }
            None => snap.mtp_idx = None,
        }
        snap.len = s.len;
        snap.next_pos = s.next_pos;
        snap.ngram_ctx = s.ngram_ctx.clone();
        snap.mtp_len = s.mtp_len - s.mtp_chain;
        snap.mtp_pos = s.mtp_pos.clone();
        st.synchronize()?;
        Ok(())
    }

    /// Copies the recurrent and auxiliary states of a session (everything the KV caches do not hold) so a later
    /// input sharing its first `len` tokens can restart from here. Not during a speculative step. A caller that
    /// snapshots on every request wants `new_snapshot` + `snapshot_into` on a pooled buffer instead.
    pub fn snapshot(&self, s: &Session) -> Result<Snapshot> {
        let mut snap = self.new_snapshot(s)?;
        self.snapshot_into(s, &mut snap)?;
        Ok(snap)
    }

    /// Whether `snap` can still be restored on `s`: a model whose sliding layers keep a ring of positions has
    /// overwritten the window behind the snapshot once the session moved far enough past it.
    pub fn can_restore(&self, s: &Session, snap: &Snapshot) -> bool {
        self.hy4.is_none() || crate::hy4::can_restore(&self.cfg, s, snap.len)
    }

    /// Restarts a session from a snapshot taken on it; its KV rows [0, len) must still hold the same tokens.
    pub fn restore(&self, s: &mut Session, snap: &Snapshot) -> Result<()> {
        let st = Stream::null();
        let cp = |d: &mut DeviceBuffer, b: &DeviceBuffer| d.copy_from(0, b, 0, b.len(), &st);
        ensure!(
            s.conv.len() == snap.conv.len() && s.rec.len() == snap.rec.len(),
            "snapshot of another model"
        );
        for (d, b) in s.conv.iter_mut().zip(&snap.conv) {
            cp(d, b)?;
        }
        for (d, b) in s.rec.iter_mut().zip(&snap.rec) {
            cp(d, b)?;
        }
        if let (Some(d), Some(b)) = (s.ple_state.as_mut(), &snap.ple_state) {
            cp(d, b)?;
        }
        for (d, b) in s.idx_tk.iter_mut().zip(&snap.idx_tk) {
            cp(d, b)?;
        }
        for (d, b) in s.idx_tp.iter_mut().zip(&snap.idx_tp) {
            cp(d, b)?;
        }
        if let (Some(d), Some(b)) = (s.mtp_h.as_mut(), &snap.mtp_h) {
            cp(d, b)?;
        }
        if let (Some([_, tk, tp]), Some([btk, btp])) = (s.mtp_idx.as_mut(), &snap.mtp_idx) {
            cp(tk, btk)?;
            cp(tp, btp)?;
        }
        st.synchronize()?;
        s.len = snap.len;
        s.next_pos = snap.next_pos;
        s.ngram_ctx = snap.ngram_ctx.clone();
        s.ngram_prev.clear();
        if self.hy4.is_some() {
            crate::hy4::restored(self, s);
        }
        s.mtp_len = snap.mtp_len;
        s.mtp_chain = 0;
        s.mtp_pos = snap.mtp_pos.clone();
        s.snap_t = 0;
        s.dn_np = 0;
        s.idx_cache_len = 0;
        if let Some(c) = s.inkling2.as_mut() {
            c.reset()?;
        }
        Ok(())
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        let st = Stream::null();
        for b in s.conv.iter().chain(&s.rec).chain(&s.ple_state) {
            b.memset(0, &st)?;
        }
        match &self.cfg.ple {
            Some(p) => s.ngram_ctx = vec![p.eos; p.ngram - 1],
            None => s.ngram_ctx.clear(),
        }
        st.synchronize()?;
        s.len = 0;
        s.next_pos = 0;
        s.snap_t = 0;
        s.dn_np = 0;
        s.mtp_len = 0;
        s.mtp_pos.clear();
        s.mtp_chain = 0;
        s.idx_cache_len = 0;
        if let Some(c) = s.inkling2.as_mut() {
            c.reset()?;
        }
        Ok(())
    }

    pub(crate) fn take_scratch(&self, t: usize) -> Result<Scratch> {
        let cached = self.scratch_cache.lock().unwrap().take();
        if let Some(sc) = cached {
            if sc.cap >= t {
                return Ok(sc);
            }
            // The old buffers go before the new ones are asked for: on these APUs a request that grows the scratch
            // used to need the pages of both at once (the `match` kept the old one alive until after the
            // allocation), which is what turned a long prompt into one oversized allocation instead of two.
            drop(sc);
        }
        self.scratch(t.max(16))
    }

    /// The working set of a chunk of `cap` tokens. `memplan::working_set_bytes` mirrors these sizes for the
    /// pre-flight: a buffer added here has to be added there.
    pub(crate) fn scratch(&self, cap: usize) -> Result<Scratch> {
        let c = &self.cfg;
        let big = c.inter.max(c.n_heads * c.head_dim * 2).max(c.lin_qkv_dim());
        let small = (c.n_heads * c.head_dim).max(c.lin_nv * c.lin_dv);
        // Widest GEMM input: also the hyper-connection streams and the PLE embedding.
        // Hy4's sliding layers project to `heads * (nope + v)` (16384), wider than the Qwen widths: the GEMM input
        // buffer must hold `t * k`.
        let hy4_k = c.hy4.as_ref().map_or(0, |h| {
            (h.heads * (h.nope + h.v))
                .max(h.swa_heads * (h.swa_nope + h.swa_v))
                .max(h.hc_mult * c.hidden)
                .max(h.latent)
        });
        let kmax = c
            .inter
            .max(c.hidden)
            .max(hy4_k)
            .max(c.n_heads * c.head_dim)
            .max(c.stream_dim())
            .max(c.ple.as_ref().map_or(0, |p| p.dim));
        let mla_prefill = std::env::var("MALLEUS_MLA_PREFILL").is_ok_and(|v| v == "1");
        let hdr_bytes = (STEP_HDR + 4 * cap) * 4;
        let [hdr, len_dev, ids, pos]: [DeviceBuffer; 4] = DeviceBuffer::alloc(hdr_bytes)?
            .into_views(&[
                (0, hdr_bytes),
                (0, 4),
                (STEP_HDR * 4, cap * 4),
                ((STEP_HDR + cap) * 4, cap * 12),
            ])
            .try_into()
            .map_err(|_| anyhow::anyhow!("step header views"))?;
        Ok(Scratch {
            cap,
            x: f32s(cap * c.hidden)?,
            hn: f32s(cap * c.hidden)?,
            big0: f32s(cap * big)?,
            big1: f32s(cap * big)?,
            big2: f32s(cap * big)?,
            small0: f32s(cap * small)?,
            small1: f32s(cap * small)?,
            small2: f32s(cap * small)?,
            small3: f32s(cap * small)?,
            kda: if c.kind.contains(&MixerKind::Kda) {
                Some([f32s(cap * c.lin_nv * c.lin_dv)?, f32s(cap * c.lin_nv * c.lin_dv)?])
            } else {
                None
            },
            mla_qt: match &c.mla {
                Some(m) => Some(f32s(
                    (if mla_prefill { MLA_PREFILL_TILE } else { GEMV_MAX_T }) * c.n_heads * m.kv_lora,
                )?),
                None => None,
            },
            mla_lat: if mla_prefill {
                match &c.mla {
                    Some(m) => Some(f32s(MLA_PREFILL_TILE * c.n_heads * m.kv_lora)?),
                    None => None,
                }
            } else {
                None
            },
            xr: f32s(cap * kmax)?,
            xh: DeviceBuffer::alloc(cap * kmax * 2)?,
            xsc: DeviceBuffer::alloc(cap * kmax / 128 * 4)?,
            hdr,
            ids,
            pos,
            len_dev,
            mh: f32s(cap * c.stream_dim())?,
            cat: f32s(cap * 2 * c.hidden)?,
            dn: {
                let blocks = cap.div_ceil(DN_CHUNK) * c.lin_nv;
                let (cc, d) = (DN_CHUNK, c.lin_dv);
                [
                    DeviceBuffer::alloc(blocks * cc * d * 2)?,
                    DeviceBuffer::alloc(blocks * cc * d * 4)?,
                    DeviceBuffer::alloc(blocks * cc * d * 2)?,
                    DeviceBuffer::alloc(blocks * cc * cc * 2)?,
                    DeviceBuffer::alloc(blocks * d * cc * 2)?,
                    DeviceBuffer::alloc(blocks * 4)?,
                ]
            },
            // The grouped MoE of the other families hinges on `c.moe`; Hy4's latent MoE has its own geometry
            // (`hy4.rs`: the experts run in the latent width, the shared expert on the full one), so it gets the
            // same scratch sized from `hy4` instead.
            moe: match (&c.moe, &c.hy4) {
                (None, Some(h)) if h.sparse.iter().any(|s| *s) => {
                    let (p, e1) = (cap * h.top_k, h.experts + 1);
                    let (latent, inter) = (h.latent, h.moe_inter);
                    Some(MoeScratch {
                        logits: f32s(cap * e1)?,
                        ids: DeviceBuffer::alloc(p * 4)?,
                        slot_ids: std::sync::Mutex::new(DeviceBuffer::alloc(4)?),
                        w: f32s(p)?,
                        gu: f32s(p * 2 * inter)?,
                        act: f32s(p * inter)?,
                        offs: DeviceBuffer::alloc((e1 + 1) * 4)?,
                        perm: DeviceBuffer::alloc(p * 4)?,
                        pos: DeviceBuffer::alloc(p * 4)?,
                        tiles: DeviceBuffer::alloc((p.div_ceil(MOE_BM) + 3 * e1 + 1) * 8)?,
                        ntiles: DeviceBuffer::alloc(8)?,
                        xg: DeviceBuffer::alloc(p * latent)?,
                        xgs: f32s(p * latent.div_ceil(128))?,
                        aq: DeviceBuffer::alloc(p * inter)?,
                        aqs: f32s(p * inter.div_ceil(128))?,
                        y: DeviceBuffer::alloc(p * latent * 4)?,
                    })
                }
                (Some(m), _) => {
                    let (p, e1) = (cap * m.slots(), m.experts + 1);
                    Some(MoeScratch {
                        logits: f32s(cap * e1)?,
                        ids: DeviceBuffer::alloc(p * 4)?,
                        slot_ids: std::sync::Mutex::new(DeviceBuffer::alloc(if self.expert_stage.is_some() {
                            p * 4
                        } else {
                            4
                        })?),
                        w: f32s(p)?,
                        gu: f32s(p * 2 * m.inter)?,
                        act: f32s(p * m.inter)?,
                        offs: DeviceBuffer::alloc((e1 + 1) * 4)?,
                        perm: DeviceBuffer::alloc(p * 4)?,
                        pos: DeviceBuffer::alloc(p * 4)?,
                        tiles: DeviceBuffer::alloc((p.div_ceil(MOE_BM) + 3 * e1 + 1) * 8)?,
                        ntiles: DeviceBuffer::alloc(8)?,
                        xg: DeviceBuffer::alloc(p * c.hidden)?,
                        xgs: f32s(p * c.hidden.div_ceil(128))?,
                        aq: DeviceBuffer::alloc(p * m.inter)?,
                        aqs: f32s(p * m.inter.div_ceil(128))?,
                        y: DeviceBuffer::alloc(p * c.hidden * if self.moe_y_half { 2 } else { 4 })?,
                    })
                }
                _ => None,
            },
            idx: match &c.indexer {
                Some(ic) => Some([
                    f32s(cap * (ic.heads + 1) * ic.dim)?,
                    f32s(cap * ic.heads * ic.dim)?,
                    f32s(cap * ic.dim)?,
                ]),
                None => None,
            },
            hc: match &c.hc {
                Some(h) => {
                    let (sd, rp) = (c.stream_dim(), pad128(h.rank + h.count));
                    Some(HcScratch {
                        hc: f32s(cap * sd)?,
                        hcn: f32s(cap * sd)?,
                        hd: f32s(cap * rp)?,
                        ha: f32s(cap * rp)?,
                        hu: f32s(cap * sd)?,
                        inj: [f32s(cap * h.count)?, f32s(cap * h.count)?],
                        mo: f32s(cap * c.hidden)?,
                        ple_e: f32s(cap * c.ple.as_ref().map_or(1, |p| p.dim))?,
                        inv: f32s(cap * h.count)?,
                    })
                }
                None => None,
            },
        })
    }

    /// Rotation + per-row int8 quantization of the GEMM input into (xh, xsc), skipped when `act` is `Same` and the
    /// previous GEMM already quantized it this way.
    pub(crate) fn quant_row(
        &self,
        s: &Stream,
        xh: &DeviceBuffer,
        xsc: &DeviceBuffer,
        act: Act,
        k: usize,
        t: usize,
    ) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        if matches!(act, Act::Same(_)) && self.quant_mode.load(Relaxed) == 1 {
            return Ok(());
        }
        ensure!(k <= 8 * 128 * 17, "rot_quant_row supports K up to 17408");
        let (x, u, mode, nw) = match act {
            Act::X(x) | Act::Same(x) => (x, x, 0i32, Arg::U64(0)),
            Act::SiluMul(g, u) => (g, u, 1, Arg::U64(0)),
            Act::Norm(x, w) => (x, x, 0, w.into()),
            Act::GatedNorm(x, z, w) => (x, z, if self.cfg.gate_sigmoid { 4 } else { 2 }, w.into()),
            Act::AttnGate(x, g) => (x, g, 3, Arg::U64(0)),
        };
        let qa = [
            xh.into(),
            xsc.into(),
            x.into(),
            u.into(),
            (&self.signs).into(),
            k.into(),
            mode.into(),
            nw,
            self.cfg.eps.into(),
        ];
        self.k
            .get("rot_quant_row")
            .launch((t as u32, 1, 1), (256, 1, 1), 0, s, &qa)?;
        self.quant_mode.store(1, Relaxed);
        Ok(())
    }

    /// Rotation + quantization of the GEMM input into (xh, xsc) with the input transform folded in, one scale per 128
    /// group: the counterpart of `quant_row` for the projections whose input needs a transform (`Act::Norm`,
    /// `GatedNorm`, `AttnGate`) and that would otherwise stay on int8 and on the MLS8 weight copy. `bits` = 4 (int4,
    /// `MALLEUS_DENSE_W4A4=acts`) or 8 (int8, `MALLEUS_DENSE_W4A4=w4a8`).
    pub(crate) fn quant_row_g(
        &self,
        s: &Stream,
        xh: &DeviceBuffer,
        xsc: &DeviceBuffer,
        act: Act,
        k: usize,
        t: usize,
        bits: u8,
    ) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        ensure!(k <= 8 * 128 * 17, "rot_quant_*_row supports K up to 17408");
        let (x, u, mode, nw) = match act {
            Act::X(x) | Act::Same(x) => (x, x, 0i32, Arg::U64(0)),
            Act::SiluMul(g, u) => (g, u, 1, Arg::U64(0)),
            Act::Norm(x, w) => (x, x, 0, w.into()),
            Act::GatedNorm(x, z, w) => (x, z, if self.cfg.gate_sigmoid { 4 } else { 2 }, w.into()),
            Act::AttnGate(x, g) => (x, g, 3, Arg::U64(0)),
        };
        let qa = [
            xh.into(),
            xsc.into(),
            x.into(),
            u.into(),
            (&self.signs).into(),
            k.into(),
            mode.into(),
            nw,
            self.cfg.eps.into(),
        ];
        self.k
            .get(if bits == 4 {
                "rot_quant_i4_row"
            } else {
                "rot_quant_i8_row"
            })
            .launch((t as u32, 1, 1), (256, 1, 1), 0, s, &qa)?;
        self.quant_mode.store(if bits == 4 { 3 } else { 2 }, Relaxed);
        Ok(())
    }

    /// Whether `w` runs on the per-row int8 activation path for t tokens.
    pub(crate) fn row8(&self, w: &Linear, t: usize) -> bool {
        self.fused_ok(t) && (w.prefill.is_some() || matches!(w.w, Weights::Mls8 { .. }))
    }

    pub(crate) fn fused_ok(&self, t: usize) -> bool {
        t >= WMMA_MIN_TOKENS && self.prefill_a8 && t > self.w4_max_t
    }

    /// One int8 GEMM for projections sharing an input: outputs split by segment, or silu(gate) * up for `glu`.
    pub(crate) fn gemm_fused(
        &self,
        s: &Stream,
        sc: &Scratch,
        outs: [&DeviceBuffer; 3],
        act: Act,
        f: &Fused,
        t: usize,
    ) -> Result<()> {
        self.quant_row(s, &sc.xh, &sc.xsc, act, f.k, t)?;
        let (name, bm, bn) = i8_kernel(t, f.glu);
        let grid = ((f.n.div_ceil(bn) * t.div_ceil(bm)) as u32, 1, 1);
        let args = [
            outs[0].into(),
            outs[1].into(),
            outs[2].into(),
            f.segs[0].into(),
            f.segs[1].into(),
            (&sc.xh).into(),
            (&sc.xsc).into(),
            (&f.q).into(),
            (&f.sc).into(),
            t.into(),
            f.n.into(),
            f.k.into(),
            0i32.into(),
        ];
        self.k.get(name).launch(grid, (256, 1, 1), 0, s, &args)?;
        Ok(())
    }

    pub(crate) fn gemm(
        &self,
        s: &Stream,
        xr: &DeviceBuffer,
        xh: &DeviceBuffer,
        xsc: &DeviceBuffer,
        out: &DeviceBuffer,
        act: Act,
        w: &Linear,
        t: usize,
        acc: bool,
    ) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        // The scratch holds the GEMM input in three forms (f32 rotated, fp16 rotated, int8 scales): a mismatch here
        // used to surface as a memory fault inside a kernel.
        ensure!(
            t * w.k * 4 <= xr.len() && t * w.k * 2 <= xh.len() && t * w.k.div_ceil(128) * 4 <= xsc.len(),
            "gemm scratch too small: t={t} k={} n={} xr={} xh={} xsc={}",
            w.k,
            w.n,
            xr.len(),
            xh.len(),
            xsc.len()
        );
        let a8 = t >= WMMA_MIN_TOKENS && self.prefill_a8 && !matches!(w.w, Weights::Bf16(_));
        // int4 activations. The plain inputs go through the per-chunk quantizer (`rot_quant_i4_w`); an input that needs
        // a transform folds it in `rot_quant_i4_row`, the int4 twin of the int8 row path, so o_proj and out_proj no
        // longer have to stay on int8 and on the MLS8 copy of their weights.
        let plain = matches!(act, Act::X(_) | Act::Same(_) | Act::SiluMul(..));
        let a4 =
            dense_w4a4() && matches!(w.w, Weights::Mls4 { .. }) && (plain || (dense_w4a4_acts() && self.fused_ok(t)));
        // Weights at 4 bits with int8 activations for a transform input: only the weight bytes change, and those are
        // what the MLS8 copy costs.
        let a8g = !a4
            && !plain
            && dense_w4a4()
            && dense_w4a8_acts()
            && matches!(w.w, Weights::Mls4 { .. })
            && self.fused_ok(t);
        let row8 = self.row8(w, t) && !a4 && !a8g;
        let x = match act {
            Act::X(x) | Act::Same(x) => x,
            Act::Norm(..) | Act::GatedNorm(..) | Act::AttnGate(..) if !row8 && !a4 && !a8g => {
                bail!("this input transform needs the row quantizer (int8 or int4)")
            }
            Act::Norm(x, _) | Act::GatedNorm(x, ..) | Act::AttnGate(x, _) => x,
            Act::SiluMul(g, u) => {
                if !row8 {
                    let n = t * w.k;
                    self.ew(s, "silu_mul", n, &[g.into(), g.into(), u.into(), n.into()])?;
                }
                g
            }
        };
        let reuse = |mode: u8| matches!(act, Act::Same(_)) && self.quant_mode.load(Relaxed) == mode;
        let tail = |q: &DeviceBuffer, sc: &DeviceBuffer| -> Vec<Arg> {
            vec![
                out.into(),
                xr.into(),
                q.into(),
                sc.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
            ]
        };
        if let Weights::Bf16(b) = &w.w {
            let grid = (w.n.div_ceil(8) as u32, t.div_ceil(8) as u32, 1);
            let args = [
                out.into(),
                x.into(),
                b.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
            ];
            self.k.get("gemm_bf16").launch(grid, (256, 1, 1), 0, s, &args)?;
            return Ok(());
        }
        let chunks = (t * w.k / 128) as u32;
        if a8 {
            let (q, sc, w4) = match (&w.w, &w.prefill) {
                (Weights::Mls4 { q, sc }, _) if t <= self.w4_max_t || a4 || a8g => (q, sc, true),
                (_, Some((q, sc))) => (q, sc, false),
                (Weights::Mls4 { q, sc }, None) => (q, sc, true),
                (Weights::Mls8 { q, sc }, None) => (q, sc, false),
                (Weights::Mls3 { .. }, None) => {
                    bail!("MLS3 prefill (W3A8) is not wired yet: docs/ling3-engine.md, increment I7")
                }
                (Weights::Mls1 { .. }, _) => bail!("MLS1 has no int8 prefill path (W1A8): the decode GEMV only"),
                (Weights::Mls2 { .. }, None) => bail!("MLS2 is a routed-expert format, not a dense prefill weight"),
                (Weights::Bf16(_), _) => unreachable!(),
            };
            debug_assert_eq!(w4, !row8);
            let (name, bm) = if w4 {
                // quant_mode 2 is int8 activations and 3 is int4: the same input quantized the other way is not
                // reusable, so the two paths get their own mode.
                let mode = u8::from(a4) + 2;
                if !reuse(mode) {
                    if a4 && !plain {
                        self.quant_row_g(s, xh, xsc, act, w.k, t, 4)?;
                    } else if a8g {
                        self.quant_row_g(s, xh, xsc, act, w.k, t, 8)?;
                    } else {
                        let qa = [xh.into(), xsc.into(), x.into(), (&self.signs).into(), w.k.into()];
                        self.k
                            .get(if a4 { "rot_quant_i4_w" } else { "rot_quant_i8_w" })
                            .launch((chunks, 1, 1), (32, 1, 1), 0, s, &qa)?;
                        self.quant_mode.store(mode, Relaxed);
                    }
                }
                match (t, a4) {
                    (0..=32, true) => ("gemm_w4a4_m32", 32),
                    (33..=64, true) => ("gemm_w4a4_m64", 64),
                    (_, true) => ("gemm_w4a4", 128),
                    (0..=32, false) => ("gemm_w4a8_m32", 32),
                    (33..=64, false) => ("gemm_w4a8_m64", 64),
                    (_, false) => ("gemm_w4a8", 128),
                }
            } else {
                self.quant_row(s, xh, xsc, act, w.k, t)?;
                // Few rows and a long K (o_proj, out_proj): 128 x 128 tiles are ~25% faster.
                let (name, bm, bn) = if w.n <= 4096 && w.k >= 4096 {
                    ("gemm_i8_m128", 128, 128)
                } else {
                    i8_kernel(t, false)
                };
                let grid = ((w.n.div_ceil(bn) * t.div_ceil(bm)) as u32, 1, 1);
                let args = [
                    out.into(),
                    out.into(),
                    out.into(),
                    w.n.into(),
                    0usize.into(),
                    xh.into(),
                    xsc.into(),
                    q.into(),
                    sc.into(),
                    t.into(),
                    w.n.into(),
                    w.k.into(),
                    (acc as i32).into(),
                ];
                self.k.get(name).launch(grid, (256, 1, 1), 0, s, &args)?;
                return Ok(());
            };
            let grid = ((w.n.div_ceil(128) * t.div_ceil(bm)) as u32, 1, 1);
            let args = [
                out.into(),
                xh.into(),
                xsc.into(),
                q.into(),
                sc.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
            ];
            self.k.get(name).launch(grid, (256, 1, 1), 0, s, &args)?;
            return Ok(());
        }
        if t >= WMMA_MIN_TOKENS {
            let (q, sc, int8) = match (&w.w, &w.prefill) {
                (Weights::Mls1 { .. }, _) => bail!("MLS1 has no WMMA path yet: it runs on the decode GEMV"),
                (_, Some((q, sc))) => (q, sc, 1i32),
                (Weights::Mls4 { q, sc }, None) => (q, sc, 0),
                (Weights::Mls8 { q, sc }, None) => (q, sc, 1),
                (Weights::Mls3 { .. }, None) => {
                    bail!("MLS3 prefill (W3A8) is not wired yet: docs/ling3-engine.md, increment I7")
                }
                (Weights::Mls2 { .. }, None) => bail!("MLS2 is a routed-expert format, not a dense prefill weight"),
                (Weights::Bf16(_), _) => unreachable!(),
            };
            let rot = [xh.into(), x.into(), (&self.signs).into(), w.k.into()];
            self.k
                .get("hadamard128_h")
                .launch((chunks, 1, 1), (128, 1, 1), 0, s, &rot)?;
            let (name, bm) = match t {
                0..=32 => ("gemm_wmma_m32", 32),
                33..=64 => ("gemm_wmma_m64", 64),
                _ => ("gemm_wmma", 128),
            };
            let grid = (w.n.div_ceil(128) as u32, t.div_ceil(bm) as u32, 1);
            let args = [
                out.into(),
                xh.into(),
                q.into(),
                sc.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
                int8.into(),
            ];
            self.k.get(name).launch(grid, (256, 1, 1), 0, s, &args)?;
            return Ok(());
        }
        // MLS4 GEMVs of a few tokens rotate their input while staging it.
        let fused_rot =
            matches!(&w.w, Weights::Mls4 { .. }) && (1..=GEMV_MAX_T).contains(&t) && w.k.is_multiple_of(128);
        if !fused_rot {
            let rot = [xr.into(), x.into(), (&self.signs).into(), w.k.into()];
            self.k
                .get("hadamard128")
                .launch((chunks, 1, 1), (128, 1, 1), 0, s, &rot)?;
        }
        let tail_rot = |q: &DeviceBuffer, sc: &DeviceBuffer| -> Vec<Arg> {
            vec![
                out.into(),
                x.into(),
                q.into(),
                sc.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
                (&self.signs).into(),
            ]
        };
        const L: [&str; GEMV_MAX_T] = [
            "gemv_mls4_l1",
            "gemv_mls4_l2",
            "gemv_mls4_l3",
            "gemv_mls4_l4",
            "gemv_mls4_l5",
            "gemv_mls4_l6",
        ];
        const L8: [&str; GEMV_MAX_T] = [
            "gemv_mls8",
            "gemv_mls8_t2",
            "gemv_mls8_t3",
            "gemv_mls8_t4",
            "gemv_mls8_t5",
            "gemv_mls8_t6",
        ];
        let (name, q, sc) = match (&w.w, &w.prefill, t) {
            (Weights::Mls4 { q, sc }, _, 1..=GEMV_MAX_T) if fused_rot => (L[t - 1], q, sc),
            (Weights::Mls4 { q, sc }, _, 1) => ("gemv_mls4", q, sc),
            // The MLS3 GEMV/GEMM take already-rotated activations and the same argument list as MLS4, but there
            // is no fused-rotation variant of them yet (docs/model-ling3-flash.md, § MLS3: da fare).
            (Weights::Mls3 { q, sc }, _, 1) => ("gemv_mls3", q, sc),
            (Weights::Mls3 { q, sc }, _, _) => ("gemm_mls3", q, sc),
            (Weights::Mls1 { q, sc }, _, 1) => ("gemv_mls1", q, sc),
            (Weights::Mls1 { q, sc }, _, _) => ("gemm_mls1", q, sc),
            (Weights::Mls2 { q, sc }, _, 1) => ("gemv_mls2", q, sc),
            (Weights::Mls2 { q, sc }, _, _) => ("gemm_mls2", q, sc),
            (Weights::Mls8 { q, sc }, _, 1..=GEMV_MAX_T) => (L8[t - 1], q, sc),
            (_, Some((q, sc)), _) => ("gemm_mls8", q, sc),
            (Weights::Mls4 { q, sc }, _, _) => ("gemm_mls4", q, sc),
            (Weights::Mls8 { q, sc }, _, _) => ("gemm_mls8", q, sc),
            (Weights::Bf16(_), _, _) => unreachable!(),
        };
        let grid = if t <= GEMV_MAX_T {
            (w.n.div_ceil(32) as u32, 1, 1)
        } else {
            (w.n.div_ceil(16) as u32, t.div_ceil(8) as u32, 1)
        };
        // Hy4: the fused rotation redoes the Hadamard of all t inputs in every block (for o_proj, K = 16384, it cost as
        // much as decoding the weights, and a verify of six paid it six times). Rotate once, then the same GEMV
        // without it: r1 / l1 for a decode step, L[t - 1] for every width of a verify, one row's arithmetic in all,
        // so a verify row is the serial step bit for bit.
        if name.starts_with("gemv_mls4_l") && self.hy4.is_some() {
            let rot = [xr.into(), x.into(), (&self.signs).into(), w.k.into()];
            self.k
                .get("hadamard128")
                .launch((chunks, 1, 1), (128, 1, 1), 0, s, &rot)?;
            let args = [
                out.into(),
                xr.into(),
                q.into(),
                sc.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
                Arg::U64(0),
            ];
            const R: [&str; GEMV_MAX_T] = [
                "gemv_mls4_r1",
                "gemv_mls4_r2",
                "gemv_mls4_r3",
                "gemv_mls4_r4",
                "gemv_mls4_r5",
                "gemv_mls4_r6",
            ];
            let (kname, g) = if w.n <= 4096 {
                (R[t - 1], w.n.div_ceil(if t == 1 { 8 } else { 16 }))
            } else {
                (name, w.n.div_ceil(32))
            };
            self.k.get(kname).launch((g as u32, 1, 1), (256, 1, 1), 0, s, &args)?;
            return Ok(());
        }
        if name == "gemv_mls4_l1" && w.n <= 4096 {
            // Few rows (o_proj, out_proj): one row per warp gives the GPU 4x the blocks.
            let g = (w.n.div_ceil(8) as u32, 1, 1);
            self.k
                .get("gemv_mls4_r1")
                .launch(g, (256, 1, 1), 0, s, &tail_rot(q, sc))?;
            return Ok(());
        }
        let args = if fused_rot { tail_rot(q, sc) } else { tail(q, sc) };
        self.k.get(name).launch(grid, (256, 1, 1), 0, s, &args)?;
        Ok(())
    }

    pub(crate) fn rmsnorm(
        &self,
        s: &Stream,
        out: &DeviceBuffer,
        x: &DeviceBuffer,
        w: &DeviceBuffer,
        rows: usize,
        n: usize,
    ) -> Result<()> {
        self.k.get("rmsnorm").launch(
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                out.into(),
                x.into(),
                w.into(),
                n.into(),
                self.cfg.eps.into(),
                1i32.into(),
            ],
        )?;
        Ok(())
    }

    /// Decode GEMV of merged projections on sc.hn (rotated into sc.xr).
    pub(crate) fn gemv_merged(
        &self,
        st: &Stream,
        sc: &Scratch,
        m: &Merged,
        outs: [&DeviceBuffer; 4],
        t: usize,
    ) -> Result<()> {
        const M: [&str; GEMV_MAX_T] = [
            "gemv_mls4_m1",
            "gemv_mls4_m2",
            "gemv_mls4_m3",
            "gemv_mls4_m4",
            "gemv_mls4_m5",
            "gemv_mls4_m6",
        ];
        self.k.get(M[t - 1]).launch(
            (m.n.div_ceil(32) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                outs[0].into(),
                outs[1].into(),
                outs[2].into(),
                outs[3].into(),
                m.bounds[0].into(),
                m.bounds[1].into(),
                m.bounds[2].into(),
                (&sc.hn).into(),
                (&m.q).into(),
                (&m.sc).into(),
                t.into(),
                m.n.into(),
                m.k.into(),
                (&self.signs).into(),
            ],
        )?;
        self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn ew(&self, s: &Stream, name: &str, n: usize, args: &[Arg]) -> Result<()> {
        self.k
            .get(name)
            .launch((n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, s, args)?;
        Ok(())
    }

    /// Whether the file is the looped Inkling-2 graph, whose prefill goes through in one forward.
    pub fn is_inkling2(&self) -> bool {
        self.inkling2.is_some()
    }

    /// Inkling-2's vision tower, when the file carries one: its own graph (`crate::inkling2::Vision`), not the
    /// `vision` field of the carried families.
    pub fn inkling2_vision(&self) -> Option<&crate::inkling2::Vision> {
        self.inkling2.as_ref().and_then(|i| i.vision.as_ref())
    }

    /// Inkling-2's standalone generation: the prompt in one forward, then a token at a time against a cache of its
    /// own. Returns the generated ids after `tokens`.
    pub fn inkling2_generate(&self, tokens: &[u32], n: usize) -> Result<Vec<u32>> {
        crate::inkling2::generate(self, tokens, n)
    }

    /// Whether the file is the DeepSeek-V4.1 host graph, which serves without a `Session`.
    pub fn is_deepseek(&self) -> bool {
        self.deepseek.is_some()
    }

    /// Greedy generation on the DeepSeek-V4.1 host graph: the prompt in one prefill into the cache, then one
    /// position per token. `on_token(step, out, row)` gets each token with the logits row it was chosen from and
    /// returns false to stop (a stop token, a cancel). The cache is reset first and dropped with the request: a
    /// host prefill is minutes long, so no prompt is ever reused.
    pub fn deepseek_generate(
        &self,
        ids: &[u32],
        tokens: usize,
        mut on_token: impl FnMut(usize, &[u32], &[f32]) -> bool,
    ) -> Result<Vec<u32>> {
        let lock = self.deepseek.as_ref().context("not a deepseek model")?;
        let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let d = &mut *guard;
        let (text, cache) = (&mut d.text, &mut d.cache);
        let vocab = self.cfg.vocab;
        let batched = ids.to_vec();
        let mut seq = batched.clone();
        let prefill = text.prefill_cached(cache, &seq)?;
        let mut row = prefill[(seq.len() - 1) * vocab..].to_vec();
        let mut out: Vec<u32> = Vec::with_capacity(tokens);
        for step in 0..tokens {
            let mut best = 0;
            for (i, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = i;
                }
            }
            let next = best as u32;
            out.push(next);
            if !on_token(step, &out, &row) || step + 1 == tokens {
                break;
            }
            seq = batched.clone();
            seq.extend_from_slice(&out);
            row = text.step(cache, &seq)?;
        }
        cache.reset();
        Ok(out)
    }

    /// Runs `tokens` through the model, appending to the session. Text positions are assigned automatically
    /// unless `positions` (T, H, W per token) is given. Returns logits for the last `n_logits` tokens.
    pub fn forward(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        positions: Option<&[[i32; 3]]>,
        n_logits: usize,
    ) -> Result<Vec<f32>> {
        self.forward_mm(sess, tokens, positions, n_logits, &[])
    }

    /// Like `forward`, with rows `[start, start + n)` of the input embeddings replaced by `buf` ([n, hidden] f32).
    pub fn forward_mm(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        positions: Option<&[[i32; 3]]>,
        n_logits: usize,
        inserts: &[(usize, &DeviceBuffer)],
    ) -> Result<Vec<f32>> {
        // Inkling-2's release graph is the looped forward in `crate::inkling2`, not the generic stack: the
        // generic path's layer/state bookkeeping does not describe its 48 KV slots.
        if self.inkling2.is_some() {
            let cache = sess.inkling2.as_deref_mut().context("inkling2 session")?;
            let all = crate::inkling2::logits_mm(self, Some(cache), tokens, n_logits, inserts)?;
            let vocab = self.cfg.vocab;
            let start = all.len().saturating_sub(n_logits * vocab);
            return Ok(all[start..].to_vec());
        }
        self.forward_impl(sess, tokens, positions, n_logits, inserts, false, false)
    }

    /// One-token greedy step: the argmax is computed on the device and only the token id comes back. The
    /// logits never cross the bus; `forward` keeps the full row for sampling and verification.
    pub fn forward_argmax(&self, sess: &mut Session, tokens: &[u32]) -> Result<u32> {
        if self.inkling2.is_some() {
            // The one-token path's argmax: the cache holds the prompt, so the row does not cross the bus twice.
            let all = self.forward(sess, tokens, None, 1)?;
            let i = self.cfg.inkling2.as_ref().context("inkling2 config")?;
            let mut bi = 0u32;
            let mut bv = f32::NEG_INFINITY;
            for (j, &x) in all.iter().take(i.unpadded_vocab).enumerate() {
                if x > bv {
                    bv = x;
                    bi = j as u32;
                }
            }
            return Ok(bi);
        }
        let v = self.forward_impl(sess, tokens, None, 1, &[], false, true)?;
        Ok(v[0] as u32)
    }

    /// Whether this model can run one token for several sessions in one pass over the weights: a flat model (no
    /// MoE, hyper-connections, PLE, indexer, KDA or MLA) and no MTP head, whose chain is per session.
    pub fn can_batch(&self) -> bool {
        let c = &self.cfg;
        c.moe.is_none()
            && c.hc.is_none()
            && c.ple.is_none()
            && c.indexer.is_none()
            && c.mla.is_none()
            && self.mtp.is_none()
            && c.kind.iter().all(|k| matches!(k, MixerKind::Attn | MixerKind::Delta))
    }

    /// One decode step for several sessions in one pass over the weights: the projections and the LM head read the
    /// weights once for all the rows, while the attention and the DeltaNet recurrence run per session on that
    /// session's own KV and length. Returns the logits row of every session, in the order given.
    pub fn decode_batch(&self, runs: &mut [(&mut Session, u32)]) -> Result<Vec<Vec<f32>>> {
        let c = &self.cfg;
        let n = runs.len();
        ensure!(n >= 1 && self.can_batch(), "this model does not batch");
        ensure!(runs.iter().all(|(s, _)| s.len < s.max_ctx), "context full");
        let st = &self.stream;
        let mut sc = self.scratch(n.max(16))?;
        let ids: Vec<i32> = runs.iter().map(|(_, t)| *t as i32).collect();
        let pos: Vec<i32> = runs.iter().flat_map(|(s, _)| [s.next_pos as i32; 3]).collect();
        sc.upload_step(0, &ids, &pos)?;
        self.k.get("embed_bf16").launch(
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&sc.x).into(), (&self.embed).into(), (&sc.ids).into(), c.hidden.into()],
        )?;
        let lens: Vec<DeviceBuffer> = runs
            .iter()
            .map(|(s, _)| -> Result<DeviceBuffer> { Ok(DeviceBuffer::from_slice(&[s.len as i32])?) })
            .collect::<Result<_>>()?;
        let states: Vec<LayerStates> = runs
            .iter()
            .map(|(s, _)| LayerStates {
                kv: &s.kv,
                conv: &s.conv,
                rec: &s.rec,
                len: s.len,
                max_ctx: s.max_ctx,
                attn_part: &s.attn_part,
                mla_part: None,
                conv_snaps: None,
                dn: (!s.dn_rec.is_empty()).then_some((&s.dn_rec[..], s.dn_np, s.dn_par, false)),
                ple_state: None,
                ple_snap: None,
                idx: None,
            })
            .collect();
        let batch = Batch {
            runs: states
                .iter()
                .zip(&lens)
                .enumerate()
                .map(|(i, (ls, len))| Run {
                    ls,
                    off: i,
                    rows: 1,
                    len_dev: len.addr(),
                })
                .collect(),
        };
        let empty: [Scratch; 0] = [];
        self.run_layers_from(
            &self.layers,
            &states[0],
            Some((&batch, &empty)),
            &sc,
            n,
            st,
            0,
            (0, 0),
            true,
        )?;
        let norm = self.norm.as_ref().context("no final norm")?;
        let normed = f32s(n * c.hidden)?;
        let mut last = f32s(n * c.hidden)?;
        last.copy_from(0, &sc.x, 0, n * c.hidden * 4, st)?;
        self.rmsnorm(st, &normed, &last, norm, n, c.hidden)?;
        let logits = f32s(n * c.vocab)?;
        self.gemm(
            st,
            &sc.xr,
            &sc.xh,
            &sc.xsc,
            &logits,
            Act::X(&normed),
            &self.lm_head,
            n,
            false,
        )?;
        st.synchronize()?;
        let flat: Vec<f32> = logits.download(n * c.vocab)?;
        for (s, _) in runs.iter_mut() {
            s.len += 1;
            s.next_pos += 1;
            s.dn_np = 0;
            s.snap_t = 0;
        }
        Ok((0..n).map(|i| flat[i * c.vocab..(i + 1) * c.vocab].to_vec()).collect())
    }

    /// Speculative verify: runs 2..=GEMV_MAX_T tokens on the decode weights, keeping the DeltaNet state after each
    /// token so `rollback` can drop the rejected tail. Returns the logits of every token.
    pub fn forward_verify(&self, sess: &mut Session, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(
            // A block drafter may propose nothing (its drafts were not paying): then the verify is one step.
            ((2..=GEMV_MAX_T).contains(&tokens.len()) || (tokens.len() == 1 && self.has_block_draft()))
                && self.can_verify(),
            "bad verify step"
        );
        // The looped graph verifies in one chunk that continues the session's cache: the rows read the keys before
        // them exactly as they would one step at a time, which is what `spec-exact` checks.
        if self.inkling2.is_some() {
            let cache = sess.inkling2.as_deref_mut().context("inkling2 session")?;
            let all = crate::inkling2::logits(self, Some(cache), tokens, tokens.len())?;
            return Ok(all);
        }
        self.forward_impl(sess, tokens, None, tokens.len(), &[], true, false)
    }

    /// Writes to `path` the attention mass collected with `MALLEUS_ATTN_MASS` (#10): a header
    /// `mstride tile_k mtiles` and then one `row tile max sum` line per non-empty tile. The exact mass is recovered
    /// afterwards: `lse = max + ln(sum)`, then `exp(lse - logsumexp(lse))`.
    pub fn dump_attn_mass(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write as _;
        let Some((buf, mstride, mtiles)) = &self.attn_mass else {
            return Ok(());
        };
        self.stream.synchronize()?;
        let rows = buf.len() / (mtiles * 8);
        let data: Vec<f32> = buf.download::<f32>(buf.len() / 4)?;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "{mstride} 32 {mtiles}")?;
        for r in 0..rows {
            for t in 0..*mtiles {
                let (mx, sum) = (data[(r * mtiles + t) * 2], data[(r * mtiles + t) * 2 + 1]);
                if sum > 0.0 && mx.is_finite() {
                    writeln!(f, "{r} {t} {mx} {sum}")?;
                }
            }
        }
        Ok(())
    }

    /// Drops the last `n` tokens of the preceding `forward_verify`.
    pub fn rollback(&self, sess: &mut Session, n: usize) -> Result<()> {
        if self.hy4.is_some() {
            return crate::hy4::rollback(self, sess, n);
        }
        if let Some(k) = sess.inkling2.as_deref_mut() {
            return crate::inkling2::rollback(self, k, n);
        }
        if n == 0 {
            return Ok(());
        }
        ensure!(
            n < sess.snap_t,
            "rollback of {n} tokens after a {}-token verify",
            sess.snap_t
        );
        let keep = sess.snap_t - n;
        let st = Stream::null();
        let c = &self.cfg;
        if let (Some(p), Some(snap), Some(state)) = (&c.ple, &sess.ple_snap, &mut sess.ple_state) {
            let sd = c.stream_dim();
            state.copy_from(0, snap, keep * sd * 4, p.conv_state() * sd * 4, &st)?;
            let h = &sess.ngram_prev;
            ensure!(h.len() >= n + p.ngram - 1, "n-gram history of the verify step missing");
            sess.ngram_ctx = h[h.len() - n - (p.ngram - 1)..h.len() - n].to_vec();
        }
        let conv_b = c.lin_qkv_dim() * (c.conv_k - 1) * 4;
        for (dst, src) in sess.conv.iter_mut().zip(&sess.conv_snap) {
            dst.copy_from(0, src, (keep - 1) * conv_b, conv_b, &st)?;
        }
        st.synchronize()?;
        sess.dn_np = keep;
        sess.len -= n;
        sess.next_pos -= n as i64;
        sess.mtp_pos.truncate(keep);
        sess.snap_t = 0;
        Ok(())
    }

    /// Whether the model drafts and verifies: an MTP head, or Hy4's DFly2 block drafter.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some() || self.has_block_draft()
    }

    /// Whether the model can check several tokens in one pass. Not every model that can has a draft head: the
    /// looped graph verifies a chunk because that is the same code path as chunked prefill.
    pub fn can_verify(&self) -> bool {
        self.has_mtp() || self.inkling2.is_some()
    }

    /// Hy4's DFly2 and Inkling-2's MTP head: the drafts of a block come in one call (`block_draft`), not one step at
    /// a time — the head is a block of proposals, and the block is what the verify then checks in one pass.
    pub fn has_block_draft(&self) -> bool {
        crate::hy4::has_draft(self) || (self.inkling2.is_some() && crate::inkling2::draft_enabled())
    }

    /// Up to `n` greedy drafts after `bonus`, the token the last call produced (DFly2, or the MTP block).
    pub fn block_draft(&self, sess: &mut Session, bonus: u32, n: usize) -> Result<Vec<u32>> {
        if let Some(k) = sess.inkling2.as_deref_mut() {
            return crate::inkling2::draft(self, k, bonus, n);
        }
        crate::hy4::draft(self, sess, bonus, n)
    }

    /// Hidden rows waiting for their next token before the MTP head can use them.
    pub fn mtp_pending(&self, sess: &Session) -> usize {
        sess.mtp_pos.len()
    }

    /// Feeds the pending hidden rows with their next tokens to the MTP head; returns its greedy draft for the token
    /// after `next.last()` and the draft's probability.
    pub fn mtp_draft(&self, sess: &mut Session, next: &[u32]) -> Result<(u32, f32)> {
        let k = next.len();
        ensure!(
            k > 0 && k == sess.mtp_pos.len(),
            "MTP: {k} tokens for {} pending rows",
            sess.mtp_pos.len()
        );
        let c = &self.cfg;
        let st = Stream::null();
        let mut sc = self.take_scratch(k * c.stream_dim() / c.hidden)?;
        sc.mh.copy_from(
            0,
            sess.mtp_h.as_ref().context("no MTP state")?,
            0,
            k * c.stream_dim() * 4,
            &st,
        )?;
        let pos: Vec<i32> = sess.mtp_pos.drain(..).flatten().collect();
        let logits = self.mtp_rows(sess, &mut sc, next, &pos, Some(false), &st)?;
        *self.scratch_cache.lock().unwrap() = Some(sc);
        logits.context("MTP logits")
    }

    /// Next draft (and its probability) from the MTP head applied to its own last output and the previous draft.
    pub fn mtp_chain(&self, sess: &mut Session, draft: u32) -> Result<(u32, f32)> {
        let c = &self.cfg;
        let st = Stream::null();
        let mut sc = self.take_scratch(c.stream_dim() / c.hidden)?;
        sc.mh.copy_from(
            0,
            sess.mtp_out.as_ref().context("no MTP state")?,
            0,
            c.stream_dim() * 4,
            &st,
        )?;
        let pos = sess.mtp_out_pos.map(|p| p + 1);
        let logits = self.mtp_rows(sess, &mut sc, &[draft], &pos, Some(true), &st)?;
        *self.scratch_cache.lock().unwrap() = Some(sc);
        logits.context("MTP logits")
    }

    /// Runs the MTP head on the rows of `sc.mh` paired with `ids` and appends them to its KV cache.
    pub(crate) fn mtp_rows(
        &self,
        sess: &mut Session,
        sc: &mut Scratch,
        ids: &[u32],
        pos: &[i32],
        logits: Option<bool>,
        st: &Stream,
    ) -> Result<Option<(u32, f32)>> {
        let c = &self.cfg;
        let m = self.mtp.as_ref().context("no MTP head")?;
        let r = ids.len();
        let chain = logits == Some(true);
        if !chain {
            sess.mtp_len -= sess.mtp_chain;
            sess.mtp_chain = 0;
        }
        if sess.mtp_len + r > sess.max_ctx {
            return Ok(None);
        }
        let idv: Vec<i32> = ids.iter().map(|&v| v as i32).collect();
        // The MTP head runs through run_layers like the main model, so its attention kernels read the
        // KV offset from sc.len_dev too — here the head's own cache position, not the main model's.
        sc.upload_step(sess.mtp_len as i32, &idv, pos)?;
        self.k.get("embed_bf16").launch(
            (r as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&sc.hn).into(), (&self.embed).into(), (&sc.ids).into(), c.hidden.into()],
        )?;
        let normed = f32s(c.hidden)?;
        if let (Some(fc_e), Some(fc_h), Some(mixer), Some(hs)) = (&m.fc_e, &m.fc_h, &m.mixer, &sc.hc) {
            // Streams of the main model in sc.mh [r, C * H]: full-vector RMSNorm, fc_hidden on each stream, plus
            // fc_embedding(RMSNorm(embed(next))) on every stream.
            let sd = c.stream_dim();
            let cnt = sd / c.hidden;
            ensure!(sc.cap >= r * cnt, "MTP scratch too small");
            self.rmsnorm(st, &sc.small0, &sc.hn, &m.pre_e, r, c.hidden)?;
            self.gemm(
                st,
                &sc.xr,
                &sc.xh,
                &sc.xsc,
                &sc.small1,
                Act::X(&sc.small0),
                fc_e,
                r,
                false,
            )?;
            self.rmsnorm(st, &hs.hcn, &sc.mh, &m.pre_h, r, sd)?;
            self.gemm(
                st,
                &sc.xr,
                &sc.xh,
                &sc.xsc,
                &hs.hc,
                Act::X(&hs.hcn),
                fc_h,
                r * cnt,
                false,
            )?;
            self.ew(
                st,
                "hc_bcast_add",
                r * sd,
                &[
                    (&hs.hc).into(),
                    (&sc.small1).into(),
                    r.into(),
                    cnt.into(),
                    c.hidden.into(),
                ],
            )?;
            self.idx_grow_len(sess, sess.mtp_len, r)?;
            let mi = sess.mtp_idx.as_ref();
            let ls = LayerStates {
                kv: &sess.mtp_kv,
                conv: &[],
                rec: &[],
                len: sess.mtp_len,
                max_ctx: sess.max_ctx,
                attn_part: &sess.attn_part,
                mla_part: sess.mla_part.as_ref(),
                conv_snaps: None,
                dn: None,
                ple_state: None,
                ple_snap: None,
                idx: match mi {
                    Some([kb, tk, tp]) => Some(IdxStates {
                        kb: std::slice::from_ref(kb),
                        tk: std::slice::from_ref(tk),
                        tp: std::slice::from_ref(tp),
                        sel: self.idx_sel_bufs(sess, sess.mtp_len, r)?,
                        list: sess.idx_list.as_ref().map(|(b, r)| (b, *r)),
                        cache: None,
                        reuse: false,
                    }),
                    None => None,
                },
            };
            self.run_layers(std::slice::from_ref(&m.layer), &ls, sc, r, st, c.n_layers)?;
            sess.mtp_len += r;
            if chain {
                sess.mtp_chain += r;
            }
            if logits.is_none() {
                return Ok(None);
            }
            let out_h = sess.mtp_out.as_mut().context("no MTP state")?;
            out_h.copy_from(0, &hs.hc, (r - 1) * sd * 4, sd * 4, st)?;
            sess.mtp_out_pos = [pos[3 * (r - 1)], pos[3 * (r - 1) + 1], pos[3 * (r - 1) + 2]];
            let out_ro: &DeviceBuffer = out_h;
            self.hc_mix(st, sc, mixer, out_ro, &normed, None, 1, None)?;
        } else {
            if self.mtp_post_norm {
                let norm = self.norm.as_ref().context("no final norm")?;
                self.rmsnorm(st, &sc.small0, &sc.mh, norm, r, c.hidden)?;
                sc.mh.copy_from(0, &sc.small0, 0, r * c.hidden * 4, st)?;
            }
            self.k.get("mtp_cat").launch(
                (r as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&sc.cat).into(),
                    (&sc.hn).into(),
                    (&sc.mh).into(),
                    (&m.pre_e).into(),
                    (&m.pre_h).into(),
                    c.hidden.into(),
                    c.eps.into(),
                ],
            )?;
            self.gemm(
                st,
                &sc.xr,
                &sc.xh,
                &sc.xsc,
                &sc.x,
                Act::X(&sc.cat),
                m.fc.as_ref().context("no MTP fc")?,
                r,
                false,
            )?;
            let ls = LayerStates {
                kv: &sess.mtp_kv,
                conv: &[],
                rec: &[],
                len: sess.mtp_len,
                max_ctx: sess.max_ctx,
                attn_part: &sess.attn_part,
                mla_part: sess.mla_part.as_ref(),
                conv_snaps: None,
                dn: None,
                ple_state: None,
                ple_snap: None,
                idx: None,
            };
            self.run_layers(std::slice::from_ref(&m.layer), &ls, sc, r, st, c.n_layers)?;
            sess.mtp_len += r;
            if chain {
                sess.mtp_chain += r;
            }
            if logits.is_none() {
                return Ok(None);
            }
            let mut last = f32s(c.hidden)?;
            last.copy_from(0, &sc.x, (r - 1) * c.hidden * 4, c.hidden * 4, st)?;
            self.rmsnorm(st, &normed, &last, m.norm.as_ref().context("no MTP norm")?, 1, c.hidden)?;
            let out_h = sess.mtp_out.as_mut().context("no MTP state")?;
            out_h.copy_from(
                0,
                if self.mtp_chain_post { &normed } else { &last },
                0,
                c.hidden * 4,
                st,
            )?;
            sess.mtp_out_pos = [pos[3 * (r - 1)], pos[3 * (r - 1) + 1], pos[3 * (r - 1) + 2]];
        }
        let head = self.draft_head.as_ref().unwrap_or(&self.lm_head);
        let out = f32s(head.n)?;
        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &out, Act::X(&normed), head, 1, false)?;
        let top = f32s(2)?;
        self.k.get("logits_top1").launch(
            (1, 1, 1),
            (1024, 1, 1),
            0,
            st,
            &[(&top).into(), (&out).into(), head.n.into()],
        )?;
        st.synchronize()?;
        let t: Vec<f32> = top.download(2)?;
        let idx = t[0].to_bits();
        let tok = self.draft_map.as_ref().map_or(idx, |m| m[idx as usize]);
        Ok(Some((tok, t[1])))
    }

    pub(crate) fn forward_impl(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        positions: Option<&[[i32; 3]]>,
        n_logits: usize,
        inserts: &[(usize, &DeviceBuffer)],
        verify: bool,
        argmax_only: bool,
    ) -> Result<Vec<f32>> {
        if self.hy4.is_some() {
            let logits = crate::hy4::forward(self, sess, tokens, positions, n_logits, inserts)?;
            // `argmax_only` callers read the token id from the first element, not a row of logits.
            if argmax_only {
                let v = &logits[logits.len() - self.cfg.vocab..];
                let best = (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
                return Ok(vec![best as f32]);
            }
            return Ok(logits);
        }
        let c = &self.cfg;
        let t = tokens.len();
        ensure!(t > 0, "empty input");
        let len0 = sess.len;
        // Decode selection reuse: one query, no complete block has appeared since the previous token. The cache
        // holds the previous token's bitset; the gather already masks keys after the current position.
        let reuse_enabled = std::env::var("MALLEUS_IDX_DECODE_REUSE").map_or(true, |v| v != "0");
        let idx_reuse = reuse_enabled
            && c.indexer.as_ref().is_some_and(|ic| {
                t == 1
                    && sess.idx_sel.is_some()
                    && sess.idx_cache_len == len0
                    && (len0 + 1) / ic.ratio == len0 / ic.ratio
            });
        ensure!(
            sess.len + t <= sess.max_ctx,
            "context full ({} + {} > {})",
            sess.len,
            t,
            sess.max_ctx
        );
        ensure!(n_logits >= 1 && n_logits <= t, "bad n_logits");
        // The model's own stream, made once: creating and destroying one per call cost about a millisecond of
        // idle GPU per decode step. It is a blocking stream, so it still orders against the synchronous host
        // copies and the null-stream sites. Results are read back only after `st.synchronize()` below.
        let st = &self.stream;
        if let Some(ng) = &self.ngram {
            ng.prefetch(&sess.ngram_ctx, tokens);
        }
        let mut sc = self.take_scratch(t)?;
        let ids: Vec<i32> = tokens.iter().map(|&v| v as i32).collect();
        let pos: Vec<i32> = match positions {
            Some(p) => {
                ensure!(p.len() == t, "positions length mismatch");
                p.iter().flatten().copied().collect()
            }
            None => (0..t).flat_map(|i| [(sess.next_pos + i as i64) as i32; 3]).collect(),
        };
        let max_pos = pos.iter().copied().max().unwrap_or(0) as i64;
        // The KV position this forward starts at is device-resident, so the attention kernels read it from
        // memory instead of a per-step launch immediate.
        sc.upload_step(sess.len as i32, &ids, &pos)?;
        self.k.get("embed_bf16").launch(
            (t as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&sc.x).into(), (&self.embed).into(), (&sc.ids).into(), c.hidden.into()],
        )?;
        for (start, buf) in inserts {
            let rows = buf.len() / (c.hidden * 4);
            ensure!(start + rows <= t, "embedding insert out of range");
            sc.x.copy_from(start * c.hidden * 4, buf, 0, rows * c.hidden * 4, st)?;
        }
        // The n-gram context before this input (its rows are read while the GPU runs the layers before PLE).
        let ng_ctx = sess.ngram_ctx.clone();
        if self.ngram.is_some() {
            let mut hist = std::mem::take(&mut sess.ngram_ctx);
            let keep = hist.len();
            hist.extend_from_slice(tokens);
            sess.ngram_ctx = hist[hist.len() - keep..].to_vec();
            sess.ngram_prev = if verify { hist } else { vec![] };
        }
        self.idx_grow_len(sess, sess.len, t)?;
        if let (Some(hcfg), Some(hs)) = (&c.hc, &sc.hc) {
            self.ew(
                st,
                "hc_expand",
                t * c.stream_dim(),
                &[
                    (&hs.hc).into(),
                    (&sc.x).into(),
                    t.into(),
                    hcfg.count.into(),
                    c.hidden.into(),
                ],
            )?;
        }
        let ls = LayerStates {
            kv: &sess.kv,
            conv: &sess.conv,
            rec: &sess.rec,
            len: sess.len,
            max_ctx: sess.max_ctx,
            attn_part: &sess.attn_part,
            mla_part: sess.mla_part.as_ref(),
            conv_snaps: verify.then_some(&sess.conv_snap[..]),
            dn: (!sess.dn_rec.is_empty()).then_some((&sess.dn_rec[..], sess.dn_np, sess.dn_par, verify)),
            ple_state: sess.ple_state.as_ref(),
            ple_snap: if verify { sess.ple_snap.as_ref() } else { None },
            idx: self.idx_states(sess, t, idx_reuse)?,
        };
        match (&self.ngram, &c.ple) {
            (Some(ng), Some(pc)) => {
                let split = pc.layer;
                let a0 = c.full_attn[..split].iter().filter(|f| **f).count();
                self.run_layers_from(&self.layers[..split], &ls, None, &sc, t, st, 0, (0, 0), false)?;
                let te = std::time::Instant::now();
                let e = ng.embed(&ng_ctx, tokens)?;
                if std::env::var_os("MALLEUS_TIMING").is_some() {
                    eprintln!("ngram embed {t} tokens: {:.1} ms", te.elapsed().as_secs_f64() * 1e3);
                }
                sc.hc
                    .as_mut()
                    .context("no hyper-connection scratch")?
                    .ple_e
                    .upload(&e)?;
                self.run_layers_from(
                    &self.layers[split..],
                    &ls,
                    None,
                    &sc,
                    t,
                    st,
                    split,
                    (a0, split - a0),
                    true,
                )?;
            }
            _ => self.run_layers(&self.layers, &ls, &sc, t, st, 0)?,
        }
        let first = t - n_logits;
        let normed = f32s(n_logits * c.hidden)?;
        match (&self.norm, &self.hc_final, &sc.hc) {
            (Some(norm), _, _) => {
                let mut last = f32s(n_logits * c.hidden)?;
                last.copy_from(0, &sc.x, first * c.hidden * 4, n_logits * c.hidden * 4, st)?;
                self.rmsnorm(st, &normed, &last, norm, n_logits, c.hidden)?;
            }
            (None, Some(hf), Some(hs)) => {
                let sd = c.stream_dim();
                let mut last = f32s(n_logits * sd)?;
                last.copy_from(0, &hs.hc, first * sd * 4, n_logits * sd * 4, st)?;
                self.hc_mix(st, &sc, hf, &last, &normed, None, n_logits, None)?;
            }
            _ => bail!("no final norm"),
        }
        let logits = f32s(n_logits * c.vocab)?;
        self.gemm(
            st,
            &sc.xr,
            &sc.xh,
            &sc.xsc,
            &logits,
            Act::X(&normed),
            &self.lm_head,
            n_logits,
            false,
        )?;
        if self.mtp.is_some() {
            self.mtp_after_forward(sess, &mut sc, tokens, &pos, verify, st)?;
        }
        st.synchronize()?;
        sess.len += t;
        if c.indexer.is_some() {
            sess.idx_cache_len = if t == 1 && self.idx_sparse(len0, 1) {
                len0 + 1
            } else {
                0
            };
        }
        sess.next_pos = max_pos + 1;
        sess.snap_t = if verify { t } else { 0 };
        if verify {
            sess.dn_par ^= 1;
            sess.dn_np = t;
        } else {
            sess.dn_np = 0;
        }
        *self.scratch_cache.lock().unwrap() = Some(sc);
        if argmax_only {
            ensure!(n_logits == 1, "forward_argmax needs one logits row");
            let out = DeviceBuffer::alloc(4)?;
            self.k.get("logits_argmax").launch(
                (1, 1, 1),
                (1024, 1, 1),
                0,
                st,
                &[(&out).into(), (&logits).into(), c.vocab.into()],
            )?;
            st.synchronize()?;
            let id: Vec<i32> = out.download(1)?;
            return Ok(vec![id[0] as f32]);
        }
        Ok(logits.download(n_logits * c.vocab)?)
    }

    /// After a forward (hidden rows in sc.x): a verify keeps all rows pending; otherwise the rows whose next token
    /// is known (the previous pending row and all but the last new one) go through the MTP head, the last waits.
    pub(crate) fn mtp_after_forward(
        &self,
        sess: &mut Session,
        sc: &mut Scratch,
        tokens: &[u32],
        pos: &[i32],
        verify: bool,
        st: &Stream,
    ) -> Result<()> {
        let c = &self.cfg;
        let t = tokens.len();
        let row = c.stream_dim() * 4;
        let rows_pos = |a: usize, b: usize| (a..b).map(|i| [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]);
        let mut mtp_h = sess.mtp_h.take().context("no MTP state")?;
        // MTP rows per call: the hyper-connection variant runs fc_hidden on C rows per token.
        let per = sc.cap / (c.stream_dim() / c.hidden);
        let res = (|| -> Result<()> {
            // The main model's output rows: its streams with hyper-connections, else the residual.
            let out_rows: &DeviceBuffer = sc.hc.as_ref().map_or(&sc.x, |hs| &hs.hc);
            if verify {
                mtp_h.copy_from(0, out_rows, 0, t * row, st)?;
                sess.mtp_pos = rows_pos(0, t).collect();
                return Ok(());
            }
            let pend = usize::from(sess.mtp_pos.len() == 1);
            let mut p: Vec<i32> = if pend == 1 { sess.mtp_pos[0].to_vec() } else { vec![] };
            p.extend(rows_pos(0, t - 1).flatten());
            let ids = &tokens[1 - pend..];
            let n = ids.len();
            // MTP input row j: the pending row, then the new rows but the last. The MTP layer overwrites the
            // streams, so with several calls the rows are copied first.
            let main = if n > per {
                let mut b = DeviceBuffer::alloc(t * row)?;
                b.copy_from(0, out_rows, 0, t * row, st)?;
                Some(b)
            } else {
                None
            };
            let j1 = per.min(n);
            {
                let src_rows: &DeviceBuffer = match &main {
                    Some(b) => b,
                    None => sc.hc.as_ref().map_or(&sc.x, |hs| &hs.hc),
                };
                if pend == 1 {
                    sc.mh.copy_from(0, &mtp_h, 0, row, st)?;
                }
                if j1 > pend {
                    sc.mh.copy_from(pend * row, src_rows, 0, (j1 - pend) * row, st)?;
                }
                mtp_h.copy_from(0, src_rows, (t - 1) * row, row, st)?;
            }
            sess.mtp_pos = rows_pos(t - 1, t).collect();
            if n == 0 {
                return Ok(());
            }
            self.mtp_rows(sess, sc, &ids[..j1], &p[..3 * j1], None, st)?;
            let mut j0 = j1;
            while j0 < n {
                let j1 = (j0 + per).min(n);
                let b = main.as_ref().context("MTP rows")?;
                sc.mh.copy_from(0, b, (j0 - pend) * row, (j1 - j0) * row, st)?;
                self.mtp_rows(sess, sc, &ids[j0..j1], &p[3 * j0..3 * j1], None, st)?;
                j0 = j1;
            }
            Ok(())
        })();
        sess.mtp_h = Some(mtp_h);
        res
    }

    /// MoE FFN: out += Σ_slots w · expert(x), shared expert in the last slot; x is RMSNorm(sc.x) with `norm`,
    /// otherwise the input already in sc.hn.
    pub(crate) fn moe(
        &self,
        st: &Stream,
        sc: &Scratch,
        norm: Option<&DeviceBuffer>,
        m: &Moe,
        layer: usize,
        t: usize,
        out: &DeviceBuffer,
        accumulate: bool,
    ) -> Result<()> {
        let c = &self.cfg;
        let mc = c.moe.as_ref().context("no MoE config")?;
        let ms = sc.moe.as_ref().context("no MoE scratch")?;
        let (h, i, slots, e1) = (c.hidden, mc.inter, mc.slots(), mc.experts + 1);
        let p = t * slots;
        if let Some(norm) = norm {
            self.rmsnorm(st, &sc.hn, &sc.x, norm, t, h)?;
        }
        if t >= 16 && h.is_multiple_of(32) {
            self.k.get("gemm_bf16_wmma").launch(
                (e1.div_ceil(64) as u32, t.div_ceil(64) as u32, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&ms.logits).into(),
                    (&sc.hn).into(),
                    (&m.router).into(),
                    t.into(),
                    e1.into(),
                    h.into(),
                    h.into(),
                    e1.into(),
                    0i32.into(),
                    0i32.into(),
                ],
            )?;
        } else {
            self.k.get("gemm_bf16").launch(
                (e1.div_ceil(8) as u32, t.div_ceil(8) as u32, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&ms.logits).into(),
                    (&sc.hn).into(),
                    (&m.router).into(),
                    t.into(),
                    e1.into(),
                    h.into(),
                    0i32.into(),
                ],
            )?;
        }
        if mc.group_routing() {
            // Ling's `noaux_tc`: groups are chosen on sigmoid + expert_bias, experts within the chosen groups
            // are chosen on the same, and the weights come from the sigmoid without the bias (ADR:
            // docs/model-ling3-flash.md, § MoE). The bias is a separate fp32 tensor, never quantized.
            let bias = m.bias.as_ref().context("group routing without expert_bias")?;
            self.k.get("moe_route_g").launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&ms.ids).into(),
                    (&ms.w).into(),
                    (&ms.logits).into(),
                    bias.into(),
                    mc.experts.into(),
                    mc.top_k.into(),
                    mc.n_group.into(),
                    mc.topk_group.into(),
                    mc.scaling.into(),
                    (mc.experts + 1).into(),
                    1i32.into(),
                ],
            )?;
        } else {
            let (route, rt) = if mc.experts <= 1024 && mc.top_k <= 32 {
                ("moe_route_w", 32)
            } else {
                ("moe_route", 256)
            };
            self.k.get(route).launch(
                (t as u32, 1, 1),
                (rt, 1, 1),
                0,
                st,
                &[
                    (&ms.ids).into(),
                    (&ms.w).into(),
                    (&ms.logits).into(),
                    mc.experts.into(),
                    mc.top_k.into(),
                ],
            )?;
        }
        if ENV_TRACE_ROUTE.is_some() {
            trace_routes(st, ms, layer, t, slots)?;
        }
        // Where the kernels read the routed experts from: the resident buffers, this layer's pool (decode) or the
        // shared staging (prefill, which needs all the experts of the layer at once). The weights are the same
        // bytes in all three, so the output is bit-identical; with the cache the decode path is handed slot
        // indices in place of the expert ids, and the kernels index the pool with them (`q + e * N * K / 2`).
        let cache = m.cache.as_deref();
        // The staging is needed when the kernels will read the layer by *expert id*: the grouped GEMMs sort the
        // tokens per expert and index the stacked matrix by id, and the per-pair path has to as well when this
        // call's pairs do not fit the pool (a slot reused inside one call would invalidate an earlier pair).
        // Otherwise the pool is read with slot indices.
        let pool_fits = cache.is_none_or(|c| p <= c.expert_slots());
        let staged = match cache {
            Some(_) if !pool_fits || (self.fused_ok(t) && !m.gu_mls3 && !m.down_mls3 && !m.gu_mls2 && !m.down_mls2) => {
                let cell = self
                    .expert_stage
                    .as_ref()
                    .context("expert cache without a staging buffer")?;
                cache.unwrap().stage_into(&mut crate::expert_cache::lock(cell))?;
                // The layer is on the device anyway: fill the pool with what this chunk routed most, so the decode
                // does not pay the working set from the disk a second time.
                st.synchronize()?;
                let routed: Vec<i32> = ms.ids.download(p)?;
                cache
                    .unwrap()
                    .warm_from_stage(&routed, &crate::expert_cache::lock(cell), st)?;
                Some(crate::expert_cache::lock(cell))
            }
            _ => None,
        };
        let (gu_q, gu_s, dn_q, dn_s) = match (&m.gu, &m.down, &staged) {
            (Some(g), Some(d), _) => ((&g.0).into(), (&g.1).into(), (&d.0).into(), (&d.1).into()),
            (None, None, Some(s)) => (
                Arg::Ptr(s.gu_q.addr()),
                Arg::Ptr(s.gu_s.addr()),
                Arg::Ptr(s.down_q.addr()),
                Arg::Ptr(s.down_s.addr()),
            ),
            _ => {
                let c = cache.context("routed experts neither resident nor cached")?;
                let (gq, gs) = c.gu();
                let (dq, ds) = c.down();
                (Arg::Ptr(gq), Arg::Ptr(gs), Arg::Ptr(dq), Arg::Ptr(ds))
            }
        };
        let ids: Arg = match (cache, &staged) {
            (Some(c), None) => {
                // The router chose on the device; the cache needs the choice on the host to know what to read.
                // One synchronization per layer, which is the price of the design (see issue #14).
                st.synchronize()?;
                let routed: Vec<i32> = ms.ids.download(p)?;
                let mut out = crate::expert_cache::lock(&ms.slot_ids);
                c.prepare(&routed, &mut out)?;
                Arg::Ptr(out.addr())
            }
            _ => (&ms.ids).into(),
        };
        match &m.pre {
            Some([(guq, gus), (dq, ds)])
                if self.fused_ok(t) && !m.gu_mls3 && !m.down_mls3 && !m.gu_mls2 && !m.down_mls2 =>
            {
                self.quant_row(st, &sc.xh, &sc.xsc, Act::X(&sc.hn), h, t)?;
                self.k.get("moe_sort").launch(
                    (1, 1, 1),
                    (1024, 1, 1),
                    0,
                    st,
                    &[
                        (&ms.offs).into(),
                        (&ms.perm).into(),
                        (&ms.pos).into(),
                        (&ms.tiles).into(),
                        (&ms.ntiles).into(),
                        (&ms.ids).into(),
                        p.into(),
                        e1.into(),
                        MOE_BM.into(),
                        // HIP has no default arguments: BM2 and OFF2 (the remainder tiles of the split path, added
                        // for the W4A8 kernels) must be passed here too. Omitting them made hipModuleLaunchKernel
                        // read past the argument list and crash on every MoE model with int8 expert copies.
                        0usize.into(),
                        0usize.into(),
                    ],
                )?;
                self.k.get("moe_gather").launch(
                    (p as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &[
                        (&ms.xg).into(),
                        (&ms.xgs).into(),
                        (&sc.xh).into(),
                        (&sc.xsc).into(),
                        (&ms.perm).into(),
                        slots.into(),
                        h.into(),
                    ],
                )?;
                let max_tiles = p.div_ceil(MOE_BM) + e1;
                let grouped = |name: &str,
                               out: &DeviceBuffer,
                               x: &DeviceBuffer,
                               xs: &DeviceBuffer,
                               q: &DeviceBuffer,
                               s: &DeviceBuffer,
                               n: usize,
                               k: usize| {
                    self.k.get(name).launch(
                        ((max_tiles * n.div_ceil(128)) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &[
                            out.into(),
                            x.into(),
                            xs.into(),
                            q.into(),
                            s.into(),
                            (&ms.offs).into(),
                            (&ms.tiles).into(),
                            (&ms.ntiles).into(),
                            n.into(),
                            k.into(),
                        ],
                    )
                };
                grouped("moe_i8_glu", &ms.act, &ms.xg, &ms.xgs, guq, gus, 2 * i, h)?;
                self.quant_row(st, &ms.aq, &ms.aqs, Act::X(&ms.act), i, p)?;
                // f32: `moe_y_half` is off for every checkpoint that reaches this branch (see `Model::load`).
                grouped("moe_i8", &ms.y, &ms.aq, &ms.aqs, dq, ds, h, i)?;
                self.k.get("moe_combine").launch(
                    (t as u32, h.div_ceil(1024) as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        out.into(),
                        (&ms.y).into(),
                        (&ms.w).into(),
                        (&ms.pos).into(),
                        slots.into(),
                        h.into(),
                        (accumulate as i32).into(),
                    ],
                )?;
            }
            // MLS3 experts have no grouped prefill kernel yet (no W3A8: docs/ling3-engine.md, I7),
            // so a long prompt walks the per-pair GEMV path. Correct and slow.
            None if self.fused_ok(t) && !m.gu_mls3 && !m.down_mls3 => {
                // MLS4 experts without an int8 copy: grouped GEMMs on activations quantized per 128-group, int8
                // (W4A8) or int4 (W4A4, double-rate WMMA and no widening of the expert nibbles). MLS2 experts use
                // the same grouped GEMM with their codes widened to int8 in LDS (W2A8).
                // gate/up averages over K = hidden (20 groups of 128), down over K = inter (5). `glu` keeps the
                // shorter product on int8: about a third of the speed-up and of the argmax cost of doing both.
                let (w2g, w2d) = (m.gu_mls2, m.down_mls2);
                let (a4g, a4d) = (!w2g && self.moe_w4a4 >= 1, !w2d && self.moe_w4a4 == 1);
                let rq = |out: &DeviceBuffer, scl: &DeviceBuffer, x: &DeviceBuffer, rows: usize, k: usize, i4| {
                    self.k.get(if i4 { "rot_quant_i4_w" } else { "rot_quant_i8_w" }).launch(
                        ((rows * k / 128) as u32, 1, 1),
                        (32, 1, 1),
                        0,
                        st,
                        &[out.into(), scl.into(), x.into(), (&self.signs).into(), k.into()],
                    )
                };
                rq(&sc.xh, &sc.xsc, &sc.hn, t, h, a4g)?;
                // Row tiles of 128 once experts average more than ~48 rows (weights read once per tile).
                let bm = if p / mc.experts >= *MOE_BM128_ROWS_ENV {
                    128
                } else {
                    MOE_BM
                };
                // With 128-row tiles and few rows per expert (padding up to ~1/3 of the work at 120 rows), the
                // remainder of each expert (< 128 rows) goes to the 64-row kernel.
                // Likewise 64-row tiles with remainders on the 32-row kernel.
                let split =
                    (bm == 64 || p / mc.experts < 200) && std::env::var("MALLEUS_MOE_SPLIT").map_or(true, |v| v != "0");
                let off2 = p.div_ceil(bm) + 1;
                self.k.get("moe_sort").launch(
                    (1, 1, 1),
                    (1024, 1, 1),
                    0,
                    st,
                    &[
                        (&ms.offs).into(),
                        (&ms.perm).into(),
                        (&ms.pos).into(),
                        (&ms.tiles).into(),
                        (&ms.ntiles).into(),
                        (&ms.ids).into(),
                        p.into(),
                        e1.into(),
                        bm.into(),
                        (if split { bm / 2 } else { 0 }).into(),
                        off2.into(),
                    ],
                )?;
                self.k.get(if a4g { "moe_gather_g4" } else { "moe_gather_g" }).launch(
                    (p as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &[
                        (&ms.xg).into(),
                        (&ms.xgs).into(),
                        (&sc.xh).into(),
                        (&sc.xsc).into(),
                        (&ms.perm).into(),
                        slots.into(),
                        h.into(),
                    ],
                )?;
                let max_tiles = p.div_ceil(bm) + e1;
                let max_tiles = if split { p / bm } else { max_tiles };
                // `MALLEUS_DEBUG_MOE=1`: how many tiles each of the two lists carries, and how many rows they cover.
                // The copy synchronizes, so it is a diagnostic and not a path to leave on (issue #9).
                if std::env::var_os("MALLEUS_DEBUG_MOE").is_some() {
                    st.synchronize()?;
                    let nt: Vec<i32> = ms.ntiles.download(2)?;
                    let (n0, n1) = (nt[0].max(0) as usize, nt[1].max(0) as usize);
                    eprintln!(
                        "moe layer {layer}: t={t} p={p} bm={bm} split={split} \
                         tiles[big]={n0} ({:5} righe) tiles[small]={n1} ({:5} righe)",
                        n0 * bm,
                        n1 * (bm / 2)
                    );
                }
                let grouped = |name: &str,
                               out: &DeviceBuffer,
                               x: &DeviceBuffer,
                               xs: &DeviceBuffer,
                               w: (Arg, Arg),
                               n: usize,
                               k: usize,
                               second: bool| {
                    let (tiles, cnt, nt) = if second {
                        (ms.tiles.addr() + 8 * off2 as u64, ms.ntiles.addr() + 4, 2 * e1)
                    } else {
                        (ms.tiles.addr(), ms.ntiles.addr(), max_tiles)
                    };
                    if nt == 0 {
                        return Ok(());
                    }
                    self.k.get(name).launch(
                        ((nt * n.div_ceil(128)) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &[
                            out.into(),
                            x.into(),
                            xs.into(),
                            w.0,
                            w.1,
                            (&ms.offs).into(),
                            Arg::U64(tiles),
                            Arg::U64(cnt),
                            n.into(),
                            k.into(),
                        ],
                    )
                };
                let (glu, _) = if w2g {
                    match bm == 128 {
                        true => ("moe_w2a8_glu_m128", "moe_w2a8_m128"),
                        false => ("moe_w2a8_glu", "moe_w2a8"),
                    }
                } else {
                    match (bm == 128, a4g) {
                        (true, true) => ("moe_w4a4_glu_m128", "moe_w4a4_m128"),
                        (true, false) => ("moe_w4a8_glu_m128", "moe_w4a8_m128"),
                        (false, true) => ("moe_w4a4_glu", "moe_w4a4"),
                        (false, false) => ("moe_w4a8_glu", "moe_w4a8"),
                    }
                };
                grouped(glu, &ms.act, &ms.xg, &ms.xgs, (gu_q, gu_s), 2 * i, h, false)?;
                let (glu2, _) = if w2g {
                    match bm == 128 {
                        true => ("moe_w2a8_glu", "moe_w2a8"),
                        false => ("moe_w2a8_glu_m32", "moe_w2a8_m32"),
                    }
                } else {
                    match (bm == 128, a4g) {
                        (true, true) => ("moe_w4a4_glu", "moe_w4a4"),
                        (true, false) => ("moe_w4a8_glu", "moe_w4a8"),
                        (false, true) => ("moe_w4a4_glu_m32", "moe_w4a4_m32"),
                        (false, false) => ("moe_w4a8_glu_m32", "moe_w4a8_m32"),
                    }
                };
                if split {
                    grouped(glu2, &ms.act, &ms.xg, &ms.xgs, (gu_q, gu_s), 2 * i, h, true)?;
                }
                rq(&ms.aq, &ms.aqs, &ms.act, p, i, a4d)?;
                let (_, dn) = if w2d {
                    match bm == 128 {
                        true => ("", "moe_w2a8_m128"),
                        false => ("", "moe_w2a8"),
                    }
                } else {
                    match (bm == 128, a4d) {
                        (true, true) => ("", "moe_w4a4_m128"),
                        (true, false) => ("", "moe_w4a8_m128"),
                        (false, true) => ("", "moe_w4a4"),
                        (false, false) => ("", "moe_w4a8"),
                    }
                };
                let (yh, dn_h) = (self.moe_y_half, format!("{dn}_yh"));
                grouped(
                    if yh { dn_h.as_str() } else { dn },
                    &ms.y,
                    &ms.aq,
                    &ms.aqs,
                    (dn_q, dn_s),
                    h,
                    i,
                    false,
                )?;
                if split {
                    let (_, dn2) = if w2d {
                        match bm == 128 {
                            true => ("", "moe_w2a8"),
                            false => ("", "moe_w2a8_m32"),
                        }
                    } else {
                        match (bm == 128, a4d) {
                            (true, true) => ("", "moe_w4a4"),
                            (true, false) => ("", "moe_w4a8"),
                            (false, true) => ("", "moe_w4a4_m32"),
                            (false, false) => ("", "moe_w4a8_m32"),
                        }
                    };
                    let dn2_h = format!("{dn2}_yh");
                    grouped(
                        if yh { dn2_h.as_str() } else { dn2 },
                        &ms.y,
                        &ms.aq,
                        &ms.aqs,
                        (dn_q, dn_s),
                        h,
                        i,
                        true,
                    )?;
                }
                self.k.get(if yh { "moe_combine_yh" } else { "moe_combine" }).launch(
                    (t as u32, h.div_ceil(1024) as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        out.into(),
                        (&ms.y).into(),
                        (&ms.w).into(),
                        (&ms.pos).into(),
                        slots.into(),
                        h.into(),
                        (accumulate as i32).into(),
                    ],
                )?;
            }
            _ => {
                // MLS3 experts want their input rotated once, in a buffer of their own: `gemv4_hl` folds the
                // rotation into its staging and `mls3_core` does not, and `sc.hn` is still needed unrotated by
                // the shared expert of the same layer.
                // MLS2 and MLS3 take already-rotated activations; MLS4 folds the rotation into its staging and
                // takes the signs instead. `sc.xr` is free here: `sc.hn` is still needed unrotated by the shared
                // expert of the same layer.
                let gate_in = if m.gu_mls3 || m.gu_mls2 {
                    self.k.get("hadamard128").launch(
                        ((t * h / 128) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        st,
                        &[(&sc.xr).into(), (&sc.hn).into(), (&self.signs).into(), h.into()],
                    )?;
                    &sc.xr
                } else {
                    &sc.hn
                };
                let mut args = vec![
                    (&ms.gu).into(),
                    (gate_in).into(),
                    gu_q,
                    gu_s,
                    ids,
                    slots.into(),
                    (2 * i).into(),
                    h.into(),
                ];
                if !m.gu_mls3 && !m.gu_mls2 {
                    args.push((&self.signs).into());
                }
                let gate_kernel = if m.gu_mls2 {
                    "moe_gate_up2"
                } else if m.gu_mls3 {
                    "moe_gate_up3"
                } else {
                    "moe_gate_up"
                };
                self.k.get(gate_kernel).launch(
                    ((2 * i).div_ceil(32) as u32, p as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &args,
                )?;
                self.k.get("moe_act").launch(
                    ((p * i / 128) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &[
                        (&ms.act).into(),
                        (&ms.gu).into(),
                        (&self.signs).into(),
                        i.into(),
                        m.swiglu.0.into(),
                    ],
                )?;
                let down_kernel = if m.down_mls2 {
                    "moe_down2"
                } else if m.down_mls3 {
                    "moe_down3"
                } else {
                    "moe_down"
                };
                self.k.get(down_kernel).launch(
                    (h.div_ceil(32) as u32, t as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        out.into(),
                        (&ms.act).into(),
                        dn_q,
                        dn_s,
                        ids,
                        (&ms.w).into(),
                        slots.into(),
                        h.into(),
                        i.into(),
                        (accumulate as i32).into(),
                    ],
                )?;
            }
        }
        // Ling's shared expert: a dense FFN on the same normalized input, added with weight 1 and its own clamp
        // (docs/model-ling3-flash.md, § MoE — it is not in the renormalization of the routed weights).
        if let Some((gate, up, down)) = &m.shared {
            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big0, Act::X(&sc.hn), gate, t, false)?;
            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big1, Act::Same(&sc.hn), up, t, false)?;
            let act = if m.swiglu.1 > 0.0 {
                // The clamp is part of the activation, not of the GEMM epilogue: one pass over the two halves,
                // then the product is the input of `down`.
                self.ew(
                    st,
                    "swiglu_clamp",
                    t * i,
                    &[
                        (&sc.big0).into(),
                        (&sc.big0).into(),
                        (&sc.big1).into(),
                        (t * i).into(),
                        m.swiglu.1.into(),
                    ],
                )?;
                Act::X(&sc.big0)
            } else {
                Act::SiluMul(&sc.big0, &sc.big1)
            };
            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, out, act, down, t, accumulate)?;
        }
        self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn run_layers(
        &self,
        layers: &[Layer],
        ls: &LayerStates,
        sc: &Scratch,
        t: usize,
        st: &Stream,
        base: usize,
    ) -> Result<()> {
        self.run_layers_from(layers, ls, None, sc, t, st, base, (0, 0), true)
    }

    /// Runs `layers`, whose first attention / DeltaNet layers use states `first` of `ls`; `all` when they reach the
    /// last layer (the state counts are then checked).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_layers_from(
        &self,
        layers: &[Layer],
        ls: &LayerStates,
        batch: Option<(&Batch, &[Scratch])>,
        sc: &Scratch,
        t: usize,
        st: &Stream,
        base: usize,
        first: (usize, usize),
        all: bool,
    ) -> Result<()> {
        let c = &self.cfg;
        let (mut ai, mut di) = first;
        ensure!(sc.cap >= t, "scratch too small");
        // The MoE output of the previous layer not yet added to the streams (done by the next mix).
        let mut pending = false;
        // Injection weights buffer the next mix writes (the other holds the last ones written).
        let mut ib = 0usize;
        let dump = std::env::var_os("MALLEUS_DUMP_DIR");
        // Debug: writes a buffer of `n` floats as `{tag}.f32` when MALLEUS_DUMP_DIR is set, for the layer-by-layer
        // comparison against `ling_ref.py layers`.
        let dump_buf = |tag: &str, buf: &DeviceBuffer, n: usize| -> Result<()> {
            match &dump {
                Some(dir) => {
                    st.synchronize()?;
                    let v: Vec<f32> = buf.download(n)?;
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(std::path::Path::new(dir).join(format!("{tag}.f32")), bytes)?;
                    Ok(())
                }
                None => Ok(()),
            }
        };
        for (local, layer) in layers.iter().enumerate() {
            let layer_index = base + local;
            let fused = self.fused_ok(t) && layer.fin.is_some();
            // Mixer and FFN outputs: added to the residual, or written for the hyper-connection injection.
            let (res, res_acc) = match &sc.hc {
                Some(hs) if layer.hc.is_some() => (&hs.mo, false),
                _ => (&sc.x, true),
            };
            let x_in = match (&layer.hc, &layer.in_norm) {
                (Some(hcw), _) => {
                    let hs = sc.hc.as_ref().context("no hyper-connection scratch")?;
                    if pending && layer.ple.is_some() {
                        self.hc_inject(st, &hs.hc, &hs.mo, &hs.inj[ib ^ 1], t)?;
                        pending = false;
                    }
                    if let Some(p) = &layer.ple {
                        self.ple(st, sc, p, ls.ple_state.context("no PLE state")?, ls.ple_snap, t)?;
                    }
                    let inject = pending.then_some((&hs.mo, &hs.inj[ib ^ 1]));
                    pending = false;
                    self.hc_mix(
                        st,
                        sc,
                        hcw.first().unwrap(),
                        &hs.hc,
                        &sc.hn,
                        Some(&hs.inj[ib]),
                        t,
                        inject,
                    )?;
                    ib ^= 1;
                    // Debug: the mixer input, the mixer output, the FFN input and the FFN output of every layer
                    // (`X<layer>_{hn,mo,hn2,ffn}_t<tokens>`), named by the layer's own index so that the MTP head's
                    // layer does not overwrite layer 0. `malleus spec-exact` with MALLEUS_DUMP_DIR set leaves the
                    // serial step and row 0 of the verify side by side: the first file that differs is the kernel.
                    dump_buf(&format!("X{layer_index:02}_hn_t{t}"), &sc.hn, t * c.hidden)?;
                    Act::X(&sc.hn)
                }
                (None, Some(w)) => {
                    if !fused {
                        self.rmsnorm(st, &sc.hn, &sc.x, w, t, c.hidden)?;
                    }
                    Act::Norm(&sc.x, w)
                }
                (None, None) => bail!("layer without pre-norm"),
            };
            match &layer.mixer {
                Mixer::Kda { .. } => self.mixer_kda(layer, ls, sc, t, st, res, res_acc, &mut di)?,
                Mixer::Mla { .. } => self.mixer_mla(layer, ls, sc, t, st, res, res_acc, &mut ai)?,
                Mixer::Attn { .. } => self.mixer_attn(layer, ls, batch, sc, t, st, x_in, res, res_acc, &mut ai)?,
                Mixer::Delta { .. } => self.mixer_delta(layer, ls, batch, sc, t, st, x_in, res, res_acc, &mut di)?,
            }
            if dump.is_some() {
                // The residual with the mixer added and the FFN not yet: the delta to the dump at the end of the
                // layer is the FFN output, which is what the reference's `moe` has to match.
                dump_buf(&format!("L{:02}_preffn_t{}", ai + di - 1, t), res, t * c.hidden)?;
                dump_buf(&format!("X{layer_index:02}_mo_t{t}"), res, t * c.hidden)?;
            }
            if let (Some(hcw), Some(hs)) = (&layer.hc, &sc.hc) {
                let attn_out = Some((&hs.mo, &hs.inj[ib ^ 1]));
                self.hc_mix(st, sc, &hcw[1], &hs.hc, &sc.hn, Some(&hs.inj[ib]), t, attn_out)?;
                ib ^= 1;
                dump_buf(&format!("X{layer_index:02}_hn2_t{t}"), &sc.hn, t * c.hidden)?;
                let Ffn::Moe(m) = &layer.ffn else {
                    bail!("hyper-connections need the MoE FFN")
                };
                self.moe(st, sc, None, m, layer_index, t, &hs.mo, false)?;
                dump_buf(&format!("X{layer_index:02}_ffn_t{t}"), &hs.mo, t * c.hidden)?;
                pending = true;
                if let Some(dir) = &dump {
                    self.hc_inject(st, &hs.hc, &hs.mo, &hs.inj[ib ^ 1], t)?;
                    pending = false;
                    // Debug: the stream state after every layer, as scripts/ref_layerwise.py --dump writes it.
                    st.synchronize()?;
                    let v: Vec<f32> = hs.hc.download(t * c.stream_dim())?;
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(std::path::Path::new(dir).join(format!("L{}.f32", ai + di - 1)), bytes)?;
                }
                continue;
            }
            let post_norm = layer.post_norm.as_ref().context("layer without post-norm")?;
            match &layer.ffn {
                Ffn::Moe(m) => self.moe(st, sc, Some(post_norm), m, layer_index, t, &sc.x, true)?,
                Ffn::Dense { fgu: Some(f), down, .. } if self.fused_ok(t) => {
                    let x_in = Act::Norm(&sc.x, post_norm);
                    self.gemm_fused(st, sc, [&sc.big0, &sc.big0, &sc.big0], x_in, f, t)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.x, Act::X(&sc.big0), down, t, true)?;
                }
                Ffn::Dense { gate, up, down, .. } => {
                    self.rmsnorm(st, &sc.hn, &sc.x, post_norm, t, c.hidden)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big0, Act::X(&sc.hn), gate, t, false)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big1, Act::Same(&sc.hn), up, t, false)?;
                    let act = Act::SiluMul(&sc.big0, &sc.big1);
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.x, act, down, t, true)?;
                }
            }
            if let Some(dir) = &dump {
                // Debug: the residual stream after every layer, for the layer-by-layer comparison the reference
                // writes with `ling_ref.py layers` (MIGRATE_MODELS.md §4.3). One file per layer, [t, hidden] f32.
                st.synchronize()?;
                let v: Vec<f32> = sc.x.download(t * c.hidden)?;
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(
                    std::path::Path::new(dir).join(format!("L{:02}_t{}.f32", ai + di - 1, t)),
                    bytes,
                )?;
            }
        }
        if pending && let Some(hs) = &sc.hc {
            self.hc_inject(st, &hs.hc, &hs.mo, &hs.inj[ib ^ 1], t)?;
        }
        if all && (ai != ls.kv.len() || di != ls.conv.len()) {
            bail!("layer/state mismatch");
        }
        Ok(())
    }
}

/// A tensor source opened from a path: a `.malleus` file, or a safetensors directory (the gates' reference
/// checkpoints are directories). The DeepSeek-V4.1 graph opens its own files because it owns them for the whole
/// life of the model.
fn open_source(path: &std::path::Path) -> Result<crate::deepseek_text::Source> {
    if path.is_dir() {
        Ok(Box::new(malleus_gguf::safetensors::SafeTensors::open_dir(path)?))
    } else {
        Ok(Box::new(
            malleus_gguf::gguf::Gguf::open(path).with_context(|| path.display().to_string())?,
        ))
    }
}
