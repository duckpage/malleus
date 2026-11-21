//! Hy4-flash (`hy_v4_flash`, issue #40): the config, the weights as the file stores them, and the Engram
//! memories read by row from the file.
//!
//! The graph is not the Qwen/Ling one: iHC streams instead of a residual, a gated MLA with a learned sink, two
//! attention shapes with two RoPE bases, a compressed indexer, and a latent MoE. So the weights live in the
//! structs below rather than in `Layer`, and `Model::forward` hands over to this module once it lands.

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::{Arg, DeviceBuffer};
use serde_json::Value;

use crate::config::{Config, Family, Hy4Config, Hy4Engram, MixerKind};
use crate::upload::Uploader;
use crate::weights::Linear;

/// The `text_config` of a Hy4-flash checkpoint, as `Config` plus the Hy4 geometry.
pub(crate) fn config(cfg: &Value, generation_config: Option<&str>) -> Result<Config> {
    let t = &cfg["text_config"];
    let u = |k: &str| -> Result<usize> {
        t[k].as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("config: {k}"))
    };
    let f = |k: &str| -> Result<f32> { t[k].as_f64().map(|v| v as f32).with_context(|| format!("config: {k}")) };
    let n_layers = u("num_hidden_layers")?;
    // Three sliding layers for every indexed one, starting with two indexed ones; the checkpoint spells the
    // global kind `deepseek_sparse_attention` and the canonical `indexed_attention` is the same layer.
    let kind: Vec<MixerKind> = t["layer_types"]
        .as_array()
        .context("config: layer_types")?
        .iter()
        .map(|v| match v.as_str() {
            Some("sliding_attention") => Ok(MixerKind::Hy4Sliding),
            Some("deepseek_sparse_attention") | Some("indexed_attention") => Ok(MixerKind::Hy4Indexed),
            other => bail!("config: unknown hy4 layer type {other:?}"),
        })
        .collect::<Result<_>>()?;
    ensure!(kind.len() == n_layers, "config: layer_types length");
    let sparse: Vec<bool> = t["mlp_layer_types"]
        .as_array()
        .context("config: mlp_layer_types")?
        .iter()
        .map(|v| v.as_str() == Some("sparse"))
        .collect();
    ensure!(sparse.len() == n_layers, "config: mlp_layer_types length");

    // The published checkpoint keys the rope parameters by its legacy layer-type names.
    let rope = &t["rope_parameters"];
    let rope_of = |name: &str| -> &Value {
        if rope[name].is_object() {
            &rope[name]
        } else {
            &Value::Null
        }
    };
    let indexed = rope_of("deepseek_sparse_attention");
    let sliding = rope_of("sliding_attention");
    let theta = |q: &Value, k: &str, d: f32| q[k].as_f64().map(|v| v as f32).unwrap_or(d);
    let sec = |q: &Value| -> [usize; 3] {
        let v: Vec<usize> = q["mrope_section"]
            .as_array()
            .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default();
        if v.len() == 3 { [v[0], v[1], v[2]] } else { [12, 10, 10] }
    };
    let mrope_section = sec(indexed);

    let vision = match cfg["vision_config"]["hidden_size"].as_u64() {
        Some(hidden) => {
            let v = &cfg["vision_config"];
            Some(crate::config::Hy4Vision {
                hidden: hidden as usize,
                inter: v["intermediate_size"]
                    .as_u64()
                    .context("vision_config: intermediate_size")? as usize,
                heads: v["num_attention_heads"].as_u64().context("vision_config: heads")? as usize,
                patch: v["patch_size"].as_u64().context("vision_config: patch_size")? as usize,
                edge: (v["max_image_size"].as_u64().unwrap_or(2048) / v["patch_size"].as_u64().unwrap_or(16)) as usize,
                merge: v["spatial_merge_size"].as_u64().unwrap_or(2) as usize,
                out: v["text_hidden_size"]
                    .as_u64()
                    .context("vision_config: text_hidden_size")? as usize,
            })
        }
        None => None,
    };
    let engram = match t["engram_layer_ids"].as_array() {
        Some(ids) if !ids.is_empty() => {
            let layers: Vec<usize> = ids
                .iter()
                .map(|v| Ok(v.as_u64().context("config: engram_layer_ids")? as usize - 1))
                .collect::<Result<_>>()?;
            let sizes = &t["engram_layer_head_vocab_sizes"];
            let mut head_vocab_sizes = Vec::with_capacity(layers.len());
            for (i, &l) in layers.iter().enumerate() {
                let one_based = (l + 1).to_string();
                let orders = sizes[&one_based]
                    .as_array()
                    .with_context(|| format!("config: engram_layer_head_vocab_sizes[{one_based}]"))?;
                let mut per_order = Vec::with_capacity(orders.len());
                for order in orders {
                    per_order.push(
                        order
                            .as_array()
                            .context("config: engram head vocab sizes")?
                            .iter()
                            .map(|v| v.as_u64().unwrap_or(0) as usize)
                            .collect::<Vec<_>>(),
                    );
                }
                ensure!(
                    ids.len() > i && per_order.len() == u("engram_max_ngram_size")? - 1,
                    "config: engram orders for layer {one_based}"
                );
                head_vocab_sizes.push(per_order);
            }
            Some(Hy4Engram {
                layers,
                heads: u("engram_n_head_per_ngram")?,
                embed: u("engram_n_embed_per_ngram")?,
                max_ngram: u("engram_max_ngram_size")?,
                pad: u("engram_pad_id")? as u32,
                // The class defaults it to the tokenizer's vocabulary size, and the checkpoint leaves it out.
                lookup: t["engram_lookup_table_size"].as_u64().unwrap_or(120818) as usize,
                seed: t["engram_seed"].as_u64().unwrap_or(0),
                fp8: t["engram_fp8_embedding"].as_bool().unwrap_or(true),
                head_vocab_sizes,
            })
        }
        _ => None,
    };

    let mut eos = vec![];
    if let Some(g) = generation_config {
        let g: Value = serde_json::from_str(g)?;
        match &g["eos_token_id"] {
            Value::Array(a) => eos.extend(a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32)),
            Value::Number(n) => eos.extend(n.as_u64().map(|v| v as u32)),
            _ => {}
        }
    }
    if eos.is_empty()
        && let Some(v) = t["eos_token_id"].as_u64()
    {
        eos.push(v as u32);
    }

    let heads = u("num_attention_heads")?;
    let hy4 = Hy4Config {
        hc_mult: u("hc_mult")?,
        hc_magnitude: f("hc_magnitude")?,
        hc_eps: f("hc_eps")?,
        sink: t["learnable_sink"].as_bool().unwrap_or(false),
        q_lora: u("q_lora_rank")?,
        kv_lora: u("kv_lora_rank")?,
        heads,
        nope: u("qk_nope_head_dim")?,
        rope: u("qk_rope_head_dim")?,
        v: u("v_head_dim")?,
        swa_heads: u("swa_num_attention_heads")?,
        swa_nope: u("swa_qk_nope_head_dim")?,
        swa_rope: u("swa_qk_rope_head_dim")?,
        swa_v: u("swa_v_head_dim")?,
        window: u("sliding_window")?,
        theta_indexed: theta(indexed, "rope_theta", 1e7),
        theta_sliding: theta(sliding, "rope_theta", 1e4),
        mrope_section,
        index_pool: u("compressed_indexer_compress_ratio")?,
        index_topk: u("index_topk")?,
        index_heads: u("index_n_heads")?,
        index_dim: u("index_head_dim")?,
        latent: u("routed_expert_hidden_size")?,
        experts: u("n_routed_experts")?,
        top_k: u("num_experts_per_tok")?,
        moe_inter: u("moe_intermediate_size")?,
        swiglu_limit: f("swiglu_limit")?,
        routed_scaling: f("routed_scaling_factor")?,
        sparse,
        engram,
        vision,
        image_token: cfg["image_token_id"].as_u64().unwrap_or(0) as u32,
        dfly: match cfg["dfly_num_hidden_layers"].as_u64() {
            Some(n) => {
                let list = |k: &str| -> Vec<usize> {
                    cfg[k].as_array().map_or(vec![], |a| {
                        a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()
                    })
                };
                Some(crate::config::Hy4DFly {
                    layers: n as usize,
                    mask: cfg["dfly_mask_token_id"].as_u64().context("dfly_mask_token_id")? as u32,
                    kv_layers: list("dfly_target_kv_layers"),
                    index_layers: list("dfly_target_index_layer"),
                })
            }
            None => None,
        },
    };
    let c = Config {
        hidden: u("hidden_size")?,
        n_layers,
        // The dense FFN of the first layer; the MoE width is `moe_intermediate_size` and the latent one.
        inter: u("intermediate_size")?,
        vocab: u("vocab_size")?,
        n_heads: heads,
        n_kv: heads,
        // Hy4 expands the latent to `nope + v` per head; there is no single per-head KV width.
        head_dim: hy4.nope + hy4.rope,
        lin_nk: 0,
        lin_nv: 0,
        lin_dk: 0,
        lin_dv: 0,
        conv_k: 0,
        eps: f("rms_norm_eps")?,
        rope_theta: hy4.theta_indexed,
        rot_dim: hy4.rope,
        mrope_section,
        full_attn: kind.iter().map(|k| *k == MixerKind::Hy4Indexed).collect(),
        eos,
        moe: None,
        hc: None,
        ple: None,
        indexer: None,
        gate_sigmoid: false,
        family: Family::Hy4,
        kind,
        mla: None,
        kda_lower_bound: 0.0,
        // `HYV4RMSNorm` multiplies by the weight, as Ling's does; the loader folds the one.
        norm_one_plus_w: false,
        mrope: true,
        dense_ffn: 1,
        swiglu: None,
        hy4: Some(hy4),
        inkling2: None,
    };
    c.validate()?;
    Ok(c)
}

/// The geometry the kernels will require of a Hy4 checkpoint.
pub(crate) fn validate(c: &Config) -> Result<()> {
    let h = c.hy4.as_ref().context("config: hy4 geometry")?;
    ensure!(c.family == Family::Hy4 && c.n_layers == c.kind.len(), "bad hy4 layers");
    ensure!(
        (2..=8).contains(&h.hc_mult),
        "hy4 needs 2..=8 streams (ihc_pre keeps 2c mixes in registers)"
    );
    ensure!(h.rope == 64 && h.swa_rope == h.rope, "hy4 rotates 64 dims per head");
    ensure!(
        h.mrope_section.iter().sum::<usize>() * 2 == h.rope,
        "hy4 M-RoPE sections must cover half the rotary width"
    );
    ensure!(
        h.nope + h.rope == c.head_dim && h.swa_nope + h.swa_rope == c.head_dim,
        "hy4 head geometry: both layer types carry the same qk width"
    );
    ensure!(
        h.q_lora > 0 && h.kv_lora.is_multiple_of(64) && h.latent.is_multiple_of(128),
        "hy4 latent widths"
    );
    ensure!(
        h.index_pool > 0 && h.index_topk % h.index_pool == 0 && h.index_heads > 0,
        "hy4 compressed indexer geometry"
    );
    // The pooling kernel is one block of `index_dim` threads (its reduction is 128 wide) and holds one slot per
    // token of the group in registers.
    ensure!(
        h.index_dim == 128 && h.index_pool <= 8,
        "hy4 compressed indexer: a 128-wide key and at most 8 slots per group"
    );
    ensure!(h.sparse.len() == c.n_layers, "hy4 mlp_layer_types length");
    // `moe_route_g` (the device router of the latent MoE) keeps the expert scores in 512 shared floats and the chosen
    // ids in 32.
    ensure!(
        h.experts <= 512 && h.top_k <= 32,
        "hy4 MoE routing: at most 512 experts, 32 per token"
    );
    ensure!(
        c.kind.contains(&MixerKind::Hy4Indexed) && c.kind.contains(&MixerKind::Hy4Sliding),
        "hy4 needs both indexed and sliding layers"
    );
    if let Some(e) = &h.engram {
        ensure!(
            e.layers.iter().all(|&l| l < c.n_layers)
                && e.head_vocab_sizes.len() == e.layers.len()
                && e.max_ngram >= 2
                && e.embed.is_multiple_of(e.heads),
            "hy4 engram geometry"
        );
        ensure!(e.pad < e.lookup as u32, "hy4 engram pad id must be inside the map");
    }
    Ok(())
}

/// The iHC mixing of one block: `[2 * streams, streams * hidden]` fp32, its bias and its two scales.
#[allow(dead_code)] // read by the text forward, which lands after this loader (issue #40)
pub(crate) struct Ihc {
    pub(crate) f: DeviceBuffer,
    pub(crate) base: DeviceBuffer,
    pub(crate) scale: DeviceBuffer,
}

/// The compressed indexer of a global layer: the query projection reads the q_a latent, the token keys the
/// hidden state, and `head_weight` (16 rows) weights the heads into the score.
#[allow(dead_code)]
pub(crate) struct Indexer {
    pub(crate) query: Linear,
    pub(crate) token_key: Linear,
    pub(crate) token_key_norm: DeviceBuffer,
    /// fp32 `[index_dim, index_dim]`: the pooled key's value projection, applied on the host layout the kernel
    /// wants.
    pub(crate) value: DeviceBuffer,
    pub(crate) gate: DeviceBuffer,
    pub(crate) key_norm: DeviceBuffer,
    pub(crate) head_weight: Linear,
    /// fp32 `[index_dim, pool]`: the position embedding inside one pooled group.
    pub(crate) intra_ape: DeviceBuffer,
}

/// One gated-MLA layer. `heads`/`nope`/`rope`/`v` are the shape this layer type uses (the sliding layers are
/// twice as wide), so the loader resolves them from `MixerKind`.
#[allow(dead_code)]
pub(crate) struct Attn {
    pub(crate) heads: usize,
    pub(crate) nope: usize,
    pub(crate) rope: usize,
    pub(crate) v: usize,
    pub(crate) q_a: Linear,
    pub(crate) q_b: Linear,
    pub(crate) kv_a: Linear,
    /// bf16 `[heads * (nope + v), kv_lora]`, dequantized at load: the absorbed kernels read it as a matrix (the
    /// reference keeps it in bf16, and the V half goes through the WMMA GEMM on bf16 anyway).
    pub(crate) kv_b: DeviceBuffer,
    pub(crate) o: Linear,
    pub(crate) gate: Linear,
    pub(crate) q_a_norm: DeviceBuffer,
    pub(crate) kv_a_norm: DeviceBuffer,
    pub(crate) post_norm: DeviceBuffer,
    /// fp32 `[heads]`: the learned sink, an extra logit per head.
    pub(crate) sink: Option<DeviceBuffer>,
    pub(crate) indexer: Option<Indexer>,
}

/// The latent MoE: `pre` maps the hidden state into the latent, the routed experts run there, `up` maps back, and
/// the shared expert sees the full width. `combine_output_norm` closes the block.
#[allow(dead_code)]
pub(crate) struct Moe {
    /// bf16 `[experts, hidden]`: the router, as a `Linear` so the GEMM dispatch can read it.
    pub(crate) router: Linear,
    /// fp32 `[experts]`: the selection bias (`e_score_correction_bias`).
    pub(crate) bias: DeviceBuffer,
    pub(crate) pre: Linear,
    pub(crate) up: Linear,
    /// MLS4 `[experts, 2 * inter, latent]` and its group scales.
    pub(crate) gu: (DeviceBuffer, DeviceBuffer),
    /// MLS4 `[experts, latent, inter]` and its group scales.
    pub(crate) down: (DeviceBuffer, DeviceBuffer),
    pub(crate) shared: (Linear, Linear, Linear),
    pub(crate) norm: DeviceBuffer,
}

#[allow(dead_code, clippy::large_enum_variant)]
pub(crate) enum Ffn {
    /// The first layer's dense FFN, with its own sandwich norm (`mlp.post_fc2_norm`).
    Dense {
        gate: Linear,
        up: Linear,
        down: Linear,
        norm: DeviceBuffer,
    },
    Latent(Moe),
}

/// The Engram memory of one layer. The 32 GB table stays in the file and is read by row (`Table`); everything
/// else is a projection on the device.
#[allow(dead_code)]
pub(crate) struct Engram {
    pub(crate) key: Vec<Linear>,
    pub(crate) value: Linear,
    pub(crate) norm_key: Vec<Vec<f32>>,
    pub(crate) norm_query: Vec<Vec<f32>>,
    /// `[heads * dim]`: the per-hash-head norm weights, row-major by head.
    pub(crate) head_norms: Vec<f32>,
    pub(crate) pre_gamma: Vec<f32>,
    /// Per stream, the branch scale applied to the memory's value.
    pub(crate) scales: Vec<Vec<f32>>,
    /// The projections carry a bias, which `Linear` does not hold: added on the host after the GEMM.
    pub(crate) value_bias: Vec<f32>,
    pub(crate) key_bias: Vec<Vec<f32>>,
    pub(crate) head_vocab: Vec<usize>,
    /// Hash heads per n-gram order, the width of one head's row, the pad id and whether the rows round-trip
    /// through fp8.
    pub(crate) heads: usize,
    pub(crate) dim: usize,
    pub(crate) pad: u32,
    pub(crate) fp8: bool,
    pub(crate) table: Table,
    /// int32 `[lookup]`: the frozen normalized-vocabulary map the hashes go through.
    pub(crate) lookup: Vec<i32>,
    /// The odd multipliers `hash_ngrams` mixes the n-gram with, one per order: drawn by the checkpoint's seed
    /// (`engram_seed + 10007 * layer_id`, numpy PCG64) from the loaded map, so they are a property of the file.
    pub(crate) hash_mult: [i64; 3],
    /// f32 on the device for `hy4_engram_gate`: `[value_bias | pre_gamma | per stream (key_bias, norm_query,
    /// norm_key, scale)]`, each `hidden` wide.
    pub(crate) gate_params: DeviceBuffer,
}

/// One Engram row table as the file stores it: `rows` rows of `k` 4-bit codes (bias 8) with an f16 scale each,
/// read on the host like the n-gram table of Qwen3.8-Flash-Next.
pub(crate) struct Table {
    file: std::sync::Arc<malleus_gguf::TensorFile>,
    scales: std::sync::Arc<malleus_gguf::TensorFile>,
    pub(crate) rows: usize,
    pub(crate) k: usize,
}

#[allow(dead_code)]
impl Table {
    fn open(src: &dyn TensorSource, name: &str) -> Result<Self> {
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.dtype == DType::RowI4 && info.shape.len() == 2,
            "{name}: expected RowI4 [rows, k]"
        );
        let file = src
            .open_tensor(name)
            .context("engram table: no file handle")?
            .reopened();
        // One f16 scale per row, in its own tensor.
        let scales = src
            .open_tensor(&format!("{name}.scales"))
            .context("engram table: no scale handle")?
            .reopened();
        if let Some((f, _, _)) = file.range(0, 0) {
            malleus_hip::file_hint::random(f);
        }
        Ok(Table {
            file: std::sync::Arc::new(file),
            scales: std::sync::Arc::new(scales),
            rows: info.shape[0],
            k: info.shape[1],
        })
    }

    /// Dequantized rows `ids` as `[ids, k]` f32: the 4-bit codes (bias 8) times the f16 scale of the row.
    pub(crate) fn rows(&self, ids: &[usize]) -> Result<Vec<f32>> {
        use rayon::prelude::*;
        let rb = self.k / 2;
        let mut out = vec![0f32; ids.len() * self.k];
        // The rows are hashed, so a prompt reads them at random from a table mostly on NVMe: one read after the
        // other waited ~80 us each, 2.2 s per Engram layer for a 2048-token prompt. In parallel the drive overlaps
        // them; a decode step (16 rows) stays on one thread.
        let read = |(id, dst): (&usize, &mut [f32])| -> Result<()> {
            let id = *id;
            ensure!(id < self.rows, "engram row {id} out of range");
            let mut buf = vec![0u8; rb];
            let mut sb = [0u8; 2];
            self.file.read_at(id * rb, &mut buf)?;
            self.scales.read_at(id * 2, &mut sb)?;
            let s = half::f16::from_le_bytes(sb).to_f32();
            for (i, &b) in buf.iter().enumerate() {
                dst[2 * i] = ((b & 0xF) as f32 - 8.0) * s;
                dst[2 * i + 1] = ((b >> 4) as f32 - 8.0) * s;
            }
            Ok(())
        };
        if ids.len() > 64 {
            ids.par_iter().zip(out.par_chunks_mut(self.k)).try_for_each(read)?;
        } else {
            ids.iter().zip(out.chunks_mut(self.k)).try_for_each(read)?;
        }
        Ok(out)
    }

    /// The f16 scale of every row, as the file stores it next to the codes? No: the scales are a tensor of their
    /// own, loaded on the device by [`Weights::load`] when the forward needs them; here only the codes.
    fn bytes(&self) -> usize {
        self.rows * self.k / 2
    }
}

#[allow(dead_code)]
pub(crate) struct Layer {
    pub(crate) attn_hc: Ihc,
    pub(crate) ffn_hc: Ihc,
    pub(crate) in_norm: DeviceBuffer,
    pub(crate) post_norm: DeviceBuffer,
    pub(crate) attn: Attn,
    pub(crate) ffn: Ffn,
    pub(crate) engram: Option<Engram>,
}

/// Everything the text graph reads, loaded from the converted file. The final norm and the head (`hc_head`) are
/// loaded with the layers so that `forward` has one thing to take.
#[allow(dead_code)]
pub(crate) struct Weights {
    pub(crate) layers: Vec<Layer>,
    pub(crate) engram_scales: Vec<DeviceBuffer>,
    pub(crate) norm: DeviceBuffer,
    pub(crate) head: Ihc,
    pub(crate) dfly: Option<DFly>,
    pub(crate) bytes: usize,
}

impl Weights {
    pub(crate) fn load(
        src: &dyn TensorSource,
        c: &Config,
        up: &mut Uploader,
        seed: u64,
        progress: &dyn Fn(usize, usize),
    ) -> Result<Self> {
        let h = c.hy4.as_ref().context("no hy4 config")?;
        let before = up.bytes;
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            progress(i, c.n_layers);
            layers.push(load_layer(up, src, c, h, i, seed)?);
        }
        let mut engram_scales = Vec::new();
        for (li, l) in layers.iter().enumerate() {
            if let Some(e) = &l.engram {
                // One f16 scale per row of the 4-bit table.
                engram_scales.push(up.typed(
                    src,
                    &format!("model.layers.{li}.engram_layer.multi_head_embedding.embedding.weight.scales"),
                    DType::F16,
                    &[e.table.rows],
                )?);
            }
        }
        let norm = up.norm(src, "model.norm.weight", c.hidden, false)?;
        let head = Ihc {
            f: up.typed(
                src,
                "model.hc_head.hc_head_fn",
                DType::F32,
                &[h.hc_mult, h.hc_mult * c.hidden],
            )?,
            base: up.typed(src, "model.hc_head.hc_head_base", DType::F32, &[h.hc_mult])?,
            scale: up.typed(src, "model.hc_head.hc_head_scale", DType::F32, &[1])?,
        };
        let dfly = DFly::load(src, c, up, seed)?;
        Ok(Weights {
            layers,
            engram_scales,
            norm,
            head,
            dfly,
            bytes: up.bytes - before,
        })
    }
}

fn load_layer(
    up: &mut Uploader,
    src: &dyn TensorSource,
    c: &Config,
    h: &Hy4Config,
    i: usize,
    seed: u64,
) -> Result<Layer> {
    let p = |s: &str| format!("model.layers.{i}.{s}");
    let hc = |up: &mut Uploader, which: &str| -> Result<Ihc> {
        Ok(Ihc {
            f: up.typed(
                src,
                &p(&format!("{which}.hc_fn")),
                DType::F32,
                &[2 * h.hc_mult, h.hc_mult * c.hidden],
            )?,
            base: up.typed(src, &p(&format!("{which}.hc_base")), DType::F32, &[2 * h.hc_mult])?,
            scale: up.typed(src, &p(&format!("{which}.hc_scale")), DType::F32, &[2])?,
        })
    };
    let (heads, nope, rope, v) = match c.kind[i] {
        MixerKind::Hy4Sliding => (h.swa_heads, h.swa_nope, h.swa_rope, h.swa_v),
        _ => (h.heads, h.nope, h.rope, h.v),
    };
    let indexer = match c.kind[i] {
        MixerKind::Hy4Indexed => {
            let ix = |s: &str| p(&format!("self_attn.compressed_indexer.{s}"));
            Some(Indexer {
                query: up.linear(src, &ix("query_projection"), h.index_heads * h.index_dim, h.q_lora)?,
                token_key: up.linear(src, &ix("token_key_projection"), h.index_dim, c.hidden)?,
                // The indexer's norms multiply by the weight without the one, and the pooling kernel reads them as f32.
                token_key_norm: up.norm_f32(src, &ix("token_key_norm.weight"), h.index_dim)?,
                value: up.typed(
                    src,
                    &ix("key_value_projection.weight"),
                    DType::F32,
                    &[h.index_dim, h.index_dim],
                )?,
                gate: up.typed(
                    src,
                    &ix("key_gate_projection.weight"),
                    DType::F32,
                    &[h.index_dim, h.index_dim],
                )?,
                key_norm: up.typed(src, &ix("key_norm.weight"), DType::F32, &[h.index_dim])?,
                head_weight: up.linear(src, &ix("head_weight_projection"), h.index_heads, c.hidden)?,
                intra_ape: up.typed(src, &ix("intra_group_ape"), DType::F32, &[h.index_pool, h.index_dim])?,
            })
        }
        _ => None,
    };
    let attn = load_attn(up, src, c, h, &p, (heads, nope, rope, v), indexer, seed)?;
    let ffn = if h.sparse[i] {
        let e = h_experts(src)?;
        let inter = moe_inter_width(src, &p("mlp"))?;
        let (g, s) = stack_experts(src, up, &p("mlp"), e, h.latent, inter)?;
        Ffn::Latent(Moe {
            router: Linear {
                w: crate::weights::Weights::Bf16(up.tensor(src, &p("mlp.gate.weight"), &[e, c.hidden])?),
                prefill: None,
                n: e,
                k: c.hidden,
            },
            bias: up.typed(src, &p("mlp.gate.e_score_correction_bias"), DType::F32, &[e])?,
            pre: up.linear(src, &p("mlp.routed_expert_down_proj"), h.latent, c.hidden)?,
            up: up.linear(src, &p("mlp.routed_expert_up_proj"), c.hidden, h.latent)?,
            gu: g,
            down: s,
            shared: (
                up.linear(src, &p("mlp.shared_experts.gate_proj"), inter, c.hidden)?,
                up.linear(src, &p("mlp.shared_experts.up_proj"), inter, c.hidden)?,
                up.linear(src, &p("mlp.shared_experts.down_proj"), c.hidden, inter)?,
            ),
            // `combine_output_norm` is applied with the weight as stored (no fold): the kernel's one-plus is off.
            norm: up.norm(src, &p("mlp.combine_output_norm.weight"), c.hidden, true)?,
        })
    } else {
        Ffn::Dense {
            gate: up.linear(src, &p("mlp.gate_proj"), c.inter, c.hidden)?,
            up: up.linear(src, &p("mlp.up_proj"), c.inter, c.hidden)?,
            down: up.linear(src, &p("mlp.down_proj"), c.hidden, c.inter)?,
            norm: up.norm(src, &p("mlp.post_fc2_norm.weight"), c.hidden, false)?,
        }
    };
    let engram = match &h.engram {
        Some(e) if e.layers.contains(&i) => {
            let k = e.layers.iter().position(|&l| l == i).unwrap();
            Some(load_engram(up, src, c, h, e, i, &e.head_vocab_sizes[k])?)
        }
        _ => None,
    };
    Ok(Layer {
        attn_hc: hc(up, "hc_attn_layer.hc_pre")?,
        ffn_hc: hc(up, "hc_mlp_layer.hc_pre")?,
        in_norm: up.norm(src, &p("input_layernorm.weight"), c.hidden, false)?,
        post_norm: up.norm(src, &p("post_attention_layernorm.weight"), c.hidden, false)?,
        attn,
        ffn,
        engram,
    })
}

/// A bf16 tensor as a host vector (the Engram norms are applied on the host, where the memory is read).
/// The gated MLA of a layer under `p` (a target layer or a drafter layer: the tensor names are the same).
#[allow(clippy::too_many_arguments)]
fn load_attn(
    up: &mut Uploader,
    src: &dyn TensorSource,
    c: &Config,
    h: &Hy4Config,
    p: &dyn Fn(&str) -> String,
    (heads, nope, rope, v): (usize, usize, usize, usize),
    indexer: Option<Indexer>,
    seed: u64,
) -> Result<Attn> {
    let qd = heads * (nope + rope);
    let kvd = heads * (nope + v);
    let od = heads * v;
    Ok(Attn {
        heads,
        nope,
        rope,
        v,
        q_a: up.linear(src, &p("self_attn.q_a_proj"), h.q_lora, c.hidden)?,
        q_b: up.linear(src, &p("self_attn.q_b_proj"), qd, h.q_lora)?,
        kv_a: up.linear(src, &p("self_attn.kv_a_proj_with_mqa"), h.kv_lora + rope, c.hidden)?,
        kv_b: up.linear_bf16(src, &p("self_attn.kv_b_proj"), kvd, h.kv_lora, seed)?,
        o: up.linear(src, &p("self_attn.o_proj"), c.hidden, od)?,
        gate: up.linear(src, &p("self_attn.linear_gate"), od, c.hidden)?,
        q_a_norm: up.norm(src, &p("self_attn.q_a_layernorm.weight"), h.q_lora, false)?,
        // `mla_cache_write` multiplies by this weight directly and reads it as f32, so it goes up raw in f32.
        kv_a_norm: up.norm_f32(src, &p("self_attn.kv_a_layernorm.weight"), h.kv_lora)?,
        post_norm: up.norm(src, &p("self_attn.post_proj_norm.weight"), c.hidden, false)?,
        sink: match h.sink {
            // f32 in the target's layers, bf16 in the drafter's: the kernels read f32.
            true => Some(match src.info(&p("self_attn.learnable_sink_param")).map(|i| i.dtype) {
                Some(DType::BF16) => {
                    DeviceBuffer::from_slice(&bf16_vec(src, &p("self_attn.learnable_sink_param"), heads)?)?
                }
                _ => up.typed(src, &p("self_attn.learnable_sink_param"), DType::F32, &[heads])?,
            }),
            false => None,
        },
        indexer,
    })
}

/// One DFly2 block: pre-norm residual, the gated MLA of a target-geometry layer, a dense SwiGLU.
#[allow(dead_code)]
pub(crate) struct DLayer {
    pub(crate) in_norm: DeviceBuffer,
    pub(crate) post_norm: DeviceBuffer,
    pub(crate) attn: Attn,
    pub(crate) gate: Linear,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
}

/// The DFly2 drafter (`model.dfly_layers.*`): the slot-0 fusion of the bonus token with the target's last hidden
/// state, the blocks, the final norm and the hidden correction applied before the shared LM head.
#[allow(dead_code)]
pub(crate) struct DFly {
    pub(crate) layers: Vec<DLayer>,
    pub(crate) final_norm: DeviceBuffer,
    pub(crate) eh_proj: Linear,
    pub(crate) enorm: DeviceBuffer,
    pub(crate) hnorm: DeviceBuffer,
    /// f32 `[hidden]`: the per-channel gate of the slot-0 fusion.
    pub(crate) fusion_gate: DeviceBuffer,
    pub(crate) hc_gate: Linear,
    pub(crate) hc_up: Linear,
    pub(crate) hc_down: Linear,
    pub(crate) hc_hidden_norm: DeviceBuffer,
    pub(crate) hc_embed_norm: DeviceBuffer,
}

impl DFly {
    pub(crate) fn load(src: &dyn TensorSource, c: &Config, up: &mut Uploader, seed: u64) -> Result<Option<Self>> {
        let h = c.hy4.as_ref().context("no hy4 config")?;
        let Some(d) = &h.dfly else { return Ok(None) };
        if src.info("model.dfly_layers.final_norm.weight").is_none() {
            return Ok(None);
        }
        ensure!(
            d.kv_layers.len() == d.layers
                && d.index_layers == d.kv_layers
                && d.kv_layers
                    .iter()
                    .all(|&l| l < c.n_layers && c.kind[l] == MixerKind::Hy4Indexed),
            "dfly: every draft layer reads an indexed target layer's cache and selection"
        );
        let hd = c.hidden;
        let mut layers = Vec::with_capacity(d.layers);
        for j in 0..d.layers {
            let p = |s: &str| format!("model.dfly_layers.{j}.{s}");
            let inter = src
                .info(&p("mlp.gate_proj.weight"))
                .context("dfly mlp")?
                .shape
                .iter()
                .copied()
                .find(|&x| x != hd)
                .unwrap_or(hd);
            layers.push(DLayer {
                in_norm: up.norm(src, &p("input_layernorm.weight"), hd, false)?,
                post_norm: up.norm(src, &p("post_attention_layernorm.weight"), hd, false)?,
                attn: load_attn(up, src, c, h, &p, (h.heads, h.nope, h.rope, h.v), None, seed)?,
                gate: up.linear(src, &p("mlp.gate_proj"), inter, hd)?,
                up: up.linear(src, &p("mlp.up_proj"), inter, hd)?,
                down: up.linear(src, &p("mlp.down_proj"), hd, inter)?,
            });
        }
        let q = |s: &str| format!("model.dfly_layers.{s}");
        Ok(Some(DFly {
            layers,
            final_norm: up.norm(src, &q("final_norm.weight"), hd, false)?,
            eh_proj: up.linear(src, &q("mtp_fusion.eh_proj"), hd, 2 * hd)?,
            enorm: up.norm(src, &q("mtp_fusion.enorm.weight"), hd, false)?,
            hnorm: up.norm(src, &q("mtp_fusion.hnorm.weight"), hd, false)?,
            fusion_gate: DeviceBuffer::from_slice(&bf16_vec(src, &q("mtp_fusion.gate"), hd)?)?,
            hc_gate: up.linear(src, &q("hidden_correction.gate_proj"), hd, 2 * hd)?,
            hc_up: up.linear(src, &q("hidden_correction.up_proj"), hd, 2 * hd)?,
            hc_down: up.linear(src, &q("hidden_correction.down_proj"), hd, hd)?,
            hc_hidden_norm: up.norm(src, &q("hidden_correction.hidden_norm.weight"), hd, false)?,
            hc_embed_norm: up.norm(src, &q("hidden_correction.embed_norm.weight"), hd, false)?,
        }))
    }
}

fn bf16_vec(src: &dyn TensorSource, name: &str, n: usize) -> Result<Vec<f32>> {
    let raw = src.read_all(name).with_context(|| format!("missing tensor {name}"))?;
    let v: Vec<f32> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::bf16::from_le_bytes(*b).to_f32())
        .collect();
    ensure!(v.len() == n, "{name}: expected {n} values, got {}", v.len());
    Ok(v)
}

/// The hash multipliers `HYV4FlashEngramLayer.draw_hash_multipliers` draws for each Engram layer:
/// numpy's PCG64 seeded with `engram_seed + 10007 * layer_id` (0), `engram_max_ngram_size` odd draws bounded by
/// `i64::MAX / compressed_vocab / 2`. They are a property of the file (the memory is only readable with them), so
/// they are transcribed instead of depending on numpy: for the layers of this checkpoint, the compressed
/// vocabulary is 92620 and the drawn values are these.
fn hash_multipliers(layer_1based: usize) -> [i64; 3] {
    match layer_1based {
        2 => [81853575799317, 39690085502883, 24229915018165],
        12 => [19457948045719, 44729556510419, 84779352922315],
        22 => [72974230057307, 21530602121221, 55998057243097],
        32 => [29128509562787, 16745155128523, 22474588302733],
        // A checkpoint with Engram on other layers needs its seed drawn again (see the comment above).
        _ => unreachable!("hash multipliers for engram layer {layer_1based}"),
    }
}

/// fp8 (e4m3) round trip of one vector, as the memory was trained to be read: scale by the vector's amax, round to
/// the e4m3 grid, scale back.
fn fp8_round_trip(row: &mut [f32]) {
    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
    let scale = amax / 448.0;
    for v in row.iter_mut() {
        *v = e4m3(*v / scale) * scale;
    }
}

/// Round to the nearest value of the e4m3 grid (1 sign, 4 exponent, 3 mantissa; subnormals below 2^-6).
fn e4m3(x: f32) -> f32 {
    const MAX: f32 = 448.0;
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let a = x.abs().min(MAX);
    if !a.is_finite() {
        return sign * MAX;
    }
    if a < 0.015625 {
        // Subnormal: multiples of 2^-9.
        return sign * ((a / 0.001953125).round() * 0.001953125);
    }
    let e = a.log2().floor();
    let step = 2f32.powf(e - 3.0);
    sign * ((a / step).round() * step).min(MAX)
}

fn h_experts(src: &dyn TensorSource) -> Result<usize> {
    let info = src
        .info("model.layers.1.mlp.experts.down_proj")
        .context("hy4 experts")?;
    Ok(info.shape[0])
}

fn moe_inter_width(src: &dyn TensorSource, base: &str) -> Result<usize> {
    let info = src
        .info(&format!("{base}.shared_experts.gate_proj.weight"))
        .context("hy4 shared expert")?;
    info.shape.first().copied().context("hy4 shared expert shape")
}

/// The routed experts: one MLS4 matrix of `[experts, 2 * inter, latent]` and one of `[experts, latent, inter]`.
fn stack_experts(
    src: &dyn TensorSource,
    up: &mut Uploader,
    base: &str,
    experts: usize,
    latent: usize,
    inter: usize,
) -> Result<((DeviceBuffer, DeviceBuffer), (DeviceBuffer, DeviceBuffer))> {
    let mut stack = |name: &str, n: usize, k: usize| -> Result<(DeviceBuffer, DeviceBuffer)> {
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.dtype == DType::Mls4,
            "{name}: hy4 routed experts must be MLS4, got {:?}",
            info.dtype
        );
        let g = k / malleus_gguf::quant::BLOCK;
        let mut q = DeviceBuffer::alloc(experts * n * k / 2)?;
        let mut sc = DeviceBuffer::alloc(experts * n * g * 2)?;
        up.into(src, name, DType::Mls4, &[experts, n, k], &mut q, 0)?;
        up.into(src, &format!("{name}.scales"), DType::F16, &[experts, n, g], &mut sc, 0)?;
        Ok((q, sc))
    };
    Ok((
        stack(&format!("{base}.experts.gate_up_proj"), 2 * inter, latent)?,
        stack(&format!("{base}.experts.down_proj"), latent, inter)?,
    ))
}

fn load_engram(
    up: &mut Uploader,
    src: &dyn TensorSource,
    c: &Config,
    h: &Hy4Config,
    e: &Hy4Engram,
    layer: usize,
    sizes: &[Vec<usize>],
) -> Result<Engram> {
    let p = |s: &str| format!("model.layers.{layer}.engram_layer.{s}");
    let width = (e.max_ngram - 1) * e.embed;
    ensure!(
        sizes.iter().flatten().count() == (e.max_ngram - 1) * e.heads,
        "engram heads for layer {layer}"
    );
    let mut key = Vec::with_capacity(h.hc_mult);
    let mut key_bias = Vec::with_capacity(h.hc_mult);
    let mut norm_key = Vec::with_capacity(h.hc_mult);
    let mut norm_query = Vec::with_capacity(h.hc_mult);
    for s in 0..h.hc_mult {
        key.push(up.linear(src, &p(&format!("key_projs.{s}")), c.hidden, width)?);
        key_bias.push(bf16_vec(src, &p(&format!("key_projs.{s}.bias")), c.hidden)?);
        norm_key.push(bf16_vec(src, &p(&format!("norm_key.{s}.weight")), c.hidden)?);
        norm_query.push(bf16_vec(src, &p(&format!("norm_query.{s}.weight")), c.hidden)?);
    }
    let nheads = sizes.iter().flatten().count();
    let mut head_norms = Vec::with_capacity(nheads * (e.embed / e.heads));
    for h in 0..nheads {
        head_norms.extend(bf16_vec(
            src,
            &p(&format!("lookup_per_head_norms.{h}.weight")),
            e.embed / e.heads,
        )?);
    }
    let mut scales = Vec::with_capacity(h.hc_mult);
    for s in 0..h.hc_mult {
        scales.push(bf16_vec(src, &p(&format!("unified_per_hc_scales.{s}")), c.hidden)?);
    }
    let lookup: Vec<i32> = up
        .typed(src, &p("lookup_table"), DType::I32, &[e.lookup])?
        .download(e.lookup)?;
    let head_vocab: Vec<usize> = sizes.iter().flatten().copied().collect();
    let pre_gamma = bf16_vec(src, &p("unified_pre_gamma"), c.hidden)?;
    let value_bias = bf16_vec(src, &p("value_proj.bias"), c.hidden)?;
    let mut params = Vec::with_capacity((2 + 4 * key_bias.len()) * c.hidden);
    params.extend_from_slice(&value_bias);
    params.extend_from_slice(&pre_gamma);
    for s in 0..key_bias.len() {
        for v in [&key_bias[s], &norm_query[s], &norm_key[s], &scales[s]] {
            ensure!(v.len() == c.hidden, "engram gate vector width");
            params.extend_from_slice(v);
        }
    }
    Ok(Engram {
        key,
        value: up.linear(src, &p("value_proj"), c.hidden, width)?,
        norm_key,
        norm_query,
        head_norms,
        pre_gamma,
        scales,
        value_bias,
        key_bias,
        gate_params: DeviceBuffer::from_slice(&params)?,
        head_vocab,
        heads: e.heads,
        dim: e.embed / e.heads,
        pad: e.pad,
        fp8: e.fp8,
        hash_mult: hash_multipliers(layer + 1),
        table: Table::open(src, &p("multi_head_embedding.embedding.weight"))?,
        lookup,
    })
}

/// The most tokens one forward call takes: a longer call is split, so that a sliding layer's ring always holds the
/// window behind every query of the call.
/// `MALLEUS_HY4_RING_CHUNK` lowers it (a small ring is how the wrap-around is tested on a short prompt).
pub(crate) fn hy4_max_chunk() -> usize {
    static V: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("MALLEUS_HY4_RING_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32768)
            .max(64)
    });
    *V
}

/// Rows of a layer's KV cache when it is a ring: a sliding layer sees the last `window` positions, so it keeps
/// `window + hy4_max_chunk()` of them (position p in row p % ring). 0 for a layer that keeps the whole context, or
/// when the context is no longer than the ring. At 1M tokens this takes the 30 sliding layers from 35 GB of cache
/// to 1.1 GB.
pub(crate) fn ring_rows(c: &Config, li: usize, max_ctx: usize) -> usize {
    let Some(h) = &c.hy4 else { return 0 };
    if c.kind[li] != crate::config::MixerKind::Hy4Sliding || h.window == 0 {
        return 0;
    }
    let ring = (h.window + hy4_max_chunk()).next_multiple_of(32);
    if crate::weights::kv_ctx(max_ctx) <= ring {
        0
    } else {
        ring
    }
}

/// A snapshot at `len` restored on `s` needs the window behind `len` still in every ring: nothing written more than
/// `ring - window` positions past it (64 more for the rejected drafts of a verify).
pub(crate) fn can_restore(c: &Config, s: &crate::session::Session, len: usize) -> bool {
    let Some(h) = &c.hy4 else { return true };
    (0..c.n_layers).all(|li| match ring_rows(c, li, s.max_ctx) {
        0 => true,
        r => s.len + 64 <= len + r - h.window,
    })
}

/// KV bytes of one session: the latent and the rotated dims per layer, fp16, plus the sliding window's own copy.
pub(crate) fn session_bytes(c: &Config, ctx: usize) -> u64 {
    let Some(h) = &c.hy4 else { return 0 };
    let per_token = h.kv_lora as u64 + h.rope as u64;
    let n = crate::weights::kv_ctx(ctx) as u64;
    let mut t: u64 = (0..c.n_layers)
        .map(|li| match ring_rows(c, li, ctx) {
            0 => n,
            r => r as u64,
        })
        .sum::<u64>()
        * per_token
        * 2;
    // The compressed indexer of every indexed layer keeps its token keys, the groups that closed, and the 3-D
    // position of every slot (shared by the layers).
    let indexed = c
        .kind
        .iter()
        .filter(|k| **k == crate::config::MixerKind::Hy4Indexed)
        .count() as u64;
    let dim = h.index_dim as u64;
    t += indexed * (n * dim * 2 + (n / h.index_pool as u64 + 1) * dim * 2);
    if indexed > 0 {
        t += n * 3 * 4;
    }
    t
}

/// An RMSNorm applied with the weight as stored (`w * x`, no `1 +`): what every Hy4 norm is.
fn rmsnorm_raw(
    model: &crate::engine::Model,
    st: &malleus_hip::Stream,
    out: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    rows: usize,
    n: usize,
) -> Result<()> {
    model.k.get("rmsnorm").launch(
        (rows as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            out.into(),
            x.into(),
            w.into(),
            n.into(),
            model.cfg.eps.into(),
            0i32.into(),
        ],
    )?;
    Ok(())
}

/// The latent MoE: route on the full width, run the chosen experts in the latent space, add the shared expert and
/// close with `combine_output_norm`.
///
/// A chunk of `HY4_MOE_GROUPED_T` tokens or more sorts its (token, slot) pairs per expert and runs two grouped GEMMs
/// over the stacked MLS4 expert matrices (int8 activations); shorter steps walk the pair list with GEMVs, where the
/// grouped tiles would be all padding. The router stays on the host in both: 512 experts per token is a rounding
/// noise next to one expert read.
/// Tokens per step below which the routed experts take the per-pair GEMVs (`MALLEUS_HY4_MOE_GROUPED=0` forces them
/// at any size, for the A/B). The grouped path is faster even at one token — it reads the routing from the scratch
/// instead of the host, which is one synchronization per layer — so the default is always grouped.
const HY4_MOE_GROUPED_T: usize = 1;

/// Up to this many tokens the routed experts run as per-pair GEMVs on the device router's choice: a decode step,
/// and a verify, whose rows then go through the same kernels as the serial step.
const HY4_MOE_GEMV_T: usize = crate::weights::GEMV_MAX_T;

/// The routed experts of a short step: one (token, slot) pair at a time, two GEMVs and the clamped SwiGLU over the
/// pair's slice of the stacked matrices. Exact and launch-bound: it is what a 12-token prompt and a decode step take.
fn pair_experts(
    model: &crate::engine::Model,
    m: &Moe,
    rout_in: &DeviceBuffer,
    ids: &[i32],
    ws: &[f32],
    t: usize,
    routed: &DeviceBuffer,
) -> Result<()> {
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let st = &model.stream;
    let (latent, inter) = (h.latent, h.moe_inter);
    let slots = h.top_k;
    let (gq, gs_) = &m.gu;
    let (dq, ds) = &m.down;
    let gu = crate::session::f32s(t * slots * 2 * inter)?;
    let act = crate::session::f32s(t * slots * inter)?;
    let y = crate::session::f32s(t * slots * latent)?;
    let (gu_row, gu_grp) = (2 * inter, 2 * inter * (latent / 128) * 2);
    let (dn_row, dn_grp) = (latent, latent * (inter / 128) * 2);
    let nexp = if std::env::var("MALLEUS_HY4_MOE_NOEXPERTS").is_ok() {
        0
    } else {
        slots
    };
    let xr = crate::session::f32s(inter.max(latent))?;
    for ti in 0..t {
        for slot in 0..nexp {
            let p = ti * slots + slot;
            let e = ids[p] as usize;
            model.k.get("hadamard128").launch(
                ((latent / 128) as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &[
                    (&xr).into(),
                    Arg::U64(rout_in.addr() + (ti * latent * 4) as u64),
                    (&model.signs).into(),
                    latent.into(),
                ],
            )?;
            model.k.get("hy4_expert_gemv").launch(
                ((2 * inter) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    Arg::U64(gu.addr() + (p * 2 * inter * 4) as u64),
                    (&xr).into(),
                    Arg::U64(gq.addr() + (e * gu_row * latent / 2) as u64),
                    Arg::U64(gs_.addr() + (e * gu_grp) as u64),
                    (2 * inter).into(),
                    latent.into(),
                    0i32.into(),
                ],
            )?;
            model.k.get("hy4_glu").launch(
                (inter.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    Arg::U64(act.addr() + (p * inter * 4) as u64),
                    Arg::U64(gu.addr() + (p * 2 * inter * 4) as u64),
                    inter.into(),
                    h.swiglu_limit.into(),
                ],
            )?;
            model.k.get("hadamard128").launch(
                ((inter / 128) as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &[
                    (&xr).into(),
                    Arg::U64(act.addr() + (p * inter * 4) as u64),
                    (&model.signs).into(),
                    inter.into(),
                ],
            )?;
            model.k.get("hy4_expert_gemv").launch(
                (latent as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    Arg::U64(y.addr() + (p * latent * 4) as u64),
                    (&xr).into(),
                    Arg::U64(dq.addr() + (e * dn_row * inter / 2) as u64),
                    Arg::U64(ds.addr() + (e * dn_grp) as u64),
                    latent.into(),
                    inter.into(),
                    0i32.into(),
                ],
            )?;
            model.k.get("hy4_moe_acc").launch(
                (latent.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    Arg::U64(routed.addr() + (ti * latent * 4) as u64),
                    Arg::U64(y.addr() + (p * latent * 4) as u64),
                    ws[p].into(),
                    latent.into(),
                ],
            )?;
        }
    }
    Ok(())
}

/// The routed experts of a chunk: the (token, slot) pairs sorted by expert, the latent inputs rotated and quantized
/// to int8, one grouped gate|up GEMM over the stacked MLS4 experts, the clamped SwiGLU, one grouped down GEMM, and the
/// weighted sum back into each token's latent row. The weights of an expert are read once per tile either way; what
/// this removes is the launch per pair (5 * slots * t per layer on the pair path).
fn grouped_experts(
    model: &crate::engine::Model,
    m: &Moe,
    rout_in: &DeviceBuffer,
    ids: &[i32],
    ws: &[f32],
    t: usize,
    routed: &DeviceBuffer,
    sc: &mut crate::session::Scratch,
) -> Result<()> {
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let st = &model.stream;
    let (latent, inter, experts, slots) = (h.latent, h.moe_inter, h.experts, h.top_k);
    let ms = sc.moe.as_mut().context("hy4 grouped MoE scratch")?;
    let (p, e1) = (t * slots, experts + 1);
    ms.ids.upload(ids)?;
    ms.w.upload(ws)?;
    // Row tiles: 64 rows, and each expert's remainder (< 64) on 32-row tiles, because a prefill chunk has only a few
    // rows per expert and a 64-row tile of padding is what the grouped GEMM is paid for.
    let bm = if p / experts >= *crate::engine::MOE_BM128_ROWS_ENV {
        128
    } else {
        crate::engine::MOE_BM
    };
    let off2 = p.div_ceil(bm) + 1;
    model.k.get("moe_sort").launch(
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
            (bm / 2).into(),
            off2.into(),
        ],
    )?;
    // The latent rows in the rotated basis the experts are stored in, int8 with one scale per 128-group, then
    // gathered into expert order.
    model.k.get("rot_quant_i8_w").launch(
        ((t * latent / 128) as u32, 1, 1),
        (32, 1, 1),
        0,
        st,
        &[
            (&sc.xh).into(),
            (&sc.xsc).into(),
            rout_in.into(),
            (&model.signs).into(),
            latent.into(),
        ],
    )?;
    model.k.get("moe_gather_g").launch(
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
            latent.into(),
        ],
    )?;
    // Both lists of tiles: the full BM-row tiles, then the remainder of each expert on BM / 2 rows.
    let grouped = |name: &str,
                   out: &DeviceBuffer,
                   x: &DeviceBuffer,
                   xs: &DeviceBuffer,
                   w: (&DeviceBuffer, &DeviceBuffer),
                   n: usize,
                   k: usize,
                   second: bool|
     -> Result<()> {
        let (tiles, cnt, nt) = if second {
            (ms.tiles.addr() + 8 * off2 as u64, ms.ntiles.addr() + 4, 2 * e1)
        } else {
            (ms.tiles.addr(), ms.ntiles.addr(), p / bm)
        };
        model.k.get(name).launch(
            ((nt * n.div_ceil(128)).max(1) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                out.into(),
                x.into(),
                xs.into(),
                w.0.into(),
                w.1.into(),
                (&ms.offs).into(),
                Arg::U64(tiles),
                Arg::U64(cnt),
                n.into(),
                k.into(),
            ],
        )?;
        Ok(())
    };
    let gu = (&m.gu.0, &m.gu.1);
    // The kernels follow the tile height `moe_sort` cut: a 64-row kernel on a 128-row tile computes half of it.
    let (big, small) = if bm == 128 {
        ("moe_w4a8_m128", "moe_w4a8")
    } else {
        ("moe_w4a8", "moe_w4a8_m32")
    };
    grouped(big, &ms.gu, &ms.xg, &ms.xgs, gu, 2 * inter, latent, false)?;
    grouped(small, &ms.gu, &ms.xg, &ms.xgs, gu, 2 * inter, latent, true)?;
    model.k.get("hy4_glu_rows").launch(
        (inter.div_ceil(256) as u32, p as u32, 1),
        (256, 1, 1),
        0,
        st,
        &[(&ms.act).into(), (&ms.gu).into(), inter.into(), h.swiglu_limit.into()],
    )?;
    model.k.get("rot_quant_i8_w").launch(
        ((p * inter / 128) as u32, 1, 1),
        (32, 1, 1),
        0,
        st,
        &[
            (&ms.aq).into(),
            (&ms.aqs).into(),
            (&ms.act).into(),
            (&model.signs).into(),
            inter.into(),
        ],
    )?;
    let dn = (&m.down.0, &m.down.1);
    grouped(big, &ms.y, &ms.aq, &ms.aqs, dn, latent, inter, false)?;
    grouped(small, &ms.y, &ms.aq, &ms.aqs, dn, latent, inter, true)?;
    model.k.get("moe_combine").launch(
        (t as u32, latent.div_ceil(1024) as u32, 1),
        (256, 1, 1),
        0,
        st,
        &[
            routed.into(),
            (&ms.y).into(),
            (&ms.w).into(),
            (&ms.pos).into(),
            slots.into(),
            latent.into(),
            0i32.into(),
        ],
    )?;
    Ok(())
}

fn moe(
    model: &crate::engine::Model,
    m: &Moe,
    hn2: &DeviceBuffer,
    t: usize,
    sc: &mut crate::session::Scratch,
) -> Result<DeviceBuffer> {
    use crate::engine::Act;
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let st = &model.stream;
    let (hidden, latent, inter) = (c.hidden, h.latent, h.moe_inter);
    let (experts, top_k) = (h.experts, h.top_k);
    // The router sees the full-width input; the experts run on its projection into the latent space.
    let logits = crate::session::f32s(t * experts)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &logits, Act::X(hn2), &m.router, t, false)?;
    let rout_in = crate::session::f32s(t * latent)?;
    // `Act::X`, not `Same`: the router GEMM is bf16 and does not quantize its input, so there is nothing to reuse.
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &rout_in, Act::X(hn2), &m.pre, t, false)?;
    // The router runs on the device (`moe_route_g`: one group, sigmoid + `e_score_correction_bias` for the choice,
    // plain sigmoid renormalized and scaled for the weight) — the host top-k used to cost a synchronization and a
    // 512-way sort per layer, which at a 256-token chunk was most of the prefill.
    let ms = sc.moe.as_mut().context("hy4 MoE scratch")?;
    model.k.get("moe_route_g").launch(
        (t as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            (&ms.ids).into(),
            (&ms.w).into(),
            (&logits).into(),
            (&m.bias).into(),
            experts.into(),
            top_k.into(),
            1i32.into(),
            1i32.into(),
            h.routed_scaling.into(),
            experts.into(),
            0i32.into(),
        ],
    )?;
    // The pair path reads the choice in its launch arguments, so it needs it on the host (a short step: a few
    // hundred ints). The grouped path reads it from the scratch and never synchronizes.
    let grouped = match std::env::var("MALLEUS_HY4_MOE_GROUPED") {
        Ok(v) => v != "0",
        Err(_) => t >= HY4_MOE_GROUPED_T,
    };
    let (ids, ws) = if grouped {
        (Vec::new(), Vec::new())
    } else {
        st.synchronize()?;
        (ms.ids.download(t * top_k)?, ms.w.download(t * top_k)?)
    };
    let trace = std::env::var("MALLEUS_HY4_TRACE").is_ok();
    let dump = |name: &str, buf: &DeviceBuffer, n: usize| -> Result<()> {
        if !trace {
            return Ok(());
        }
        st.synchronize()?;
        let v: Vec<f32> = buf.download(n)?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let dir = std::env::var("MALLEUS_DUMP_DIR")?;
        std::fs::write(std::path::Path::new(&dir).join(format!("M_{name}.f32")), bytes)?;
        Ok(())
    };
    dump("rout_in", &rout_in, t * latent)?;
    if trace {
        st.synchronize()?;
        let i4: Vec<i32> = ms.ids.download(t * top_k)?;
        let w4: Vec<f32> = ms.w.download(t * top_k)?;
        eprintln!(
            "routing ids {:?} w {:?}",
            &i4[..i4.len().min(4)],
            &w4[..w4.len().min(4)]
        );
    }
    let routed = crate::session::f32s(t * latent)?;
    if t <= HY4_MOE_GEMV_T && std::env::var("MALLEUS_HY4_MOE_GEMV").map_or(true, |v| v != "0") {
        // A decode step or a verify: the (token, slot) pairs sorted by expert, each used expert's rows in one GEMV
        // (its weights read once however many tokens picked it), the clamped SwiGLU rotated for the down
        // projection, and the slots summed in order. Every row is computed the same way whatever t is.
        let ms = sc.moe.as_ref().context("hy4 MoE scratch")?;
        let p = t * top_k;
        model.k.get("moe_sort").launch(
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
                (experts + 1).into(),
                8i32.into(),
                0i32.into(),
                0i32.into(),
            ],
        )?;
        let grp = |name: &str,
                   out: &DeviceBuffer,
                   x: &DeviceBuffer,
                   w: &(DeviceBuffer, DeviceBuffer),
                   n: usize,
                   k: usize,
                   gu: bool|
         -> Result<()> {
            let mut a: Vec<malleus_hip::Arg> = vec![
                out.into(),
                x.into(),
                (&w.0).into(),
                (&w.1).into(),
                (&ms.offs).into(),
                (&ms.perm).into(),
                (&ms.tiles).into(),
                (&ms.ntiles).into(),
            ];
            if gu {
                a.push(top_k.into());
            }
            a.push(n.into());
            a.push(k.into());
            if gu {
                a.push(malleus_hip::Arg::U64(0));
            }
            model
                .k
                .get(name)
                .launch((n.div_ceil(32) as u32, p as u32, 1), (256, 1, 1), 0, st, &a)?;
            Ok(())
        };
        let one = if t == 1 { "1" } else { "" };
        // The latent rows rotated once, not by every expert's block for every row it reads.
        model.k.get("hadamard128").launch(
            ((t * latent / 128) as u32, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[(&sc.xr).into(), (&rout_in).into(), (&model.signs).into(), latent.into()],
        )?;
        grp(
            &format!("moe_gate_up_grp{one}"),
            &ms.gu,
            &sc.xr,
            &m.gu,
            2 * inter,
            latent,
            true,
        )?;
        model.k.get("moe_act").launch(
            ((p * inter / 128) as u32, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[
                (&ms.act).into(),
                (&ms.gu).into(),
                (&model.signs).into(),
                inter.into(),
                h.swiglu_limit.into(),
            ],
        )?;
        grp(
            &format!("moe_down_grp{one}"),
            &ms.y,
            &ms.act,
            &m.down,
            latent,
            inter,
            false,
        )?;
        model.k.get("moe_combine").launch(
            (t as u32, latent.div_ceil(1024) as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&routed).into(),
                (&ms.y).into(),
                (&ms.w).into(),
                (&ms.pos).into(),
                top_k.into(),
                latent.into(),
                0i32.into(),
            ],
        )?;
    } else {
        routed.memset(0, st)?;
        if grouped {
            grouped_experts(model, m, &rout_in, &ids, &ws, t, &routed, sc)?;
        } else {
            pair_experts(model, m, &rout_in, &ids, &ws, t, &routed)?;
        }
    }

    // Back to the hidden width, then the shared expert on the full-width input added in: plain GEMMs, since the
    // whole chunk shares one input (the pair loop could not, which is why this used to be a GEMV per token).
    dump("routed", &routed, t * latent)?;
    let out = crate::session::f32s(t * hidden)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &out, Act::X(&routed), &m.up, t, false)?;
    let sg = crate::session::f32s(t * inter)?;
    let su = crate::session::f32s(t * inter)?;
    let shared = std::env::var("MALLEUS_HY4_MOE_NOSHARED").is_err();
    if shared {
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sg, Act::X(hn2), &m.shared.0, t, false)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &su, Act::Same(hn2), &m.shared.1, t, false)?;
        model.ew(
            st,
            "silu_mul",
            t * inter,
            &[(&sg).into(), (&sg).into(), (&su).into(), (t * inter).into()],
        )?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &out, Act::X(&sg), &m.shared.2, t, true)?;
    }
    dump("out", &out, t * hidden)?;
    let normed = crate::session::f32s(t * hidden)?;
    rmsnorm_raw(model, st, &normed, &out, &m.norm, t, hidden)?;
    Ok(normed)
}

/// The processor's `smart_resize`: both sides a multiple of `patch * merge`, inside the pixel budget. It rounds
/// halves to even, like Python's `round`.
pub(crate) fn smart_resize(h: usize, w: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    let (hf, wf) = (h as f64, w as f64);
    let f = factor as f64;
    let mut hb = (hf / f).round_ties_even() as usize * factor;
    let mut wb = (wf / f).round_ties_even() as usize * factor;
    if hb * wb > max_pixels {
        let beta = (hf * wf / max_pixels as f64).sqrt();
        hb = ((hf / beta / f).floor().max(1.0) as usize) * factor;
        wb = ((wf / beta / f).floor().max(1.0) as usize) * factor;
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        hb = ((hf * beta / f).ceil() as usize) * factor;
        wb = ((wf * beta / f).ceil() as usize) * factor;
    }
    (hb.max(factor), wb.max(factor))
}

/// The weights of one PIL resample pass: for every output position, the first input index and the filter weights
/// around it (the triangle filter with its support widened by the reduction).
fn resample_weights(in_len: usize, out_len: usize) -> Vec<(usize, Vec<f64>, f64)> {
    let scale = in_len as f64 / out_len as f64;
    let fscale = scale.max(1.0);
    let support = fscale;
    (0..out_len)
        .map(|o| {
            let center = (o as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor() as isize).max(0) as usize;
            let hi = (((center + support + 0.5).floor() as isize).max(0) as usize).min(in_len);
            let mut ws = Vec::with_capacity(hi.saturating_sub(lo));
            let mut total = 0.0f64;
            for i in lo..hi {
                let x = ((i as f64 - center + 0.5) / fscale).abs();
                let w = (1.0 - x).max(0.0);
                total += w;
                ws.push(w);
            }
            (lo, ws, total)
        })
        .collect()
}

/// PIL's bilinear resize of an `u8` image with `c` interleaved channels: horizontal then vertical, each pass
/// rounded back to `u8` the way PIL's 8-bit resampler is.
fn resize_u8(src: &[u8], h: usize, w: usize, nh: usize, nw: usize, c: usize) -> Vec<u8> {
    let wh = resample_weights(w, nw);
    let mut mid = vec![0u8; h * nw * c];
    for y in 0..h {
        for (o, (lo, ws, total)) in wh.iter().enumerate() {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for (k, &wt) in ws.iter().enumerate() {
                    acc += src[(y * w + lo + k) * c + ch] as f64 * wt;
                }
                let v = if *total > 0.0 { acc / total } else { 0.0 };
                mid[(y * nw + o) * c + ch] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    let wv = resample_weights(h, nh);
    let mut out = vec![0u8; nh * nw * c];
    for (o, (lo, ws, total)) in wv.iter().enumerate() {
        for x in 0..nw {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for (k, &wt) in ws.iter().enumerate() {
                    acc += mid[((lo + k) * nw + x) * c + ch] as f64 * wt;
                }
                let v = if *total > 0.0 { acc / total } else { 0.0 };
                out[(o * nw + x) * c + ch] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// The processor: smart resize, bilinear resize, rescale and normalise to `[-1, 1]` (mean and std 0.5), and the
/// row-major patches of `patch x patch` in `(channel, y, x)` order. Returns the flat patch tensor and the patch
/// grid.
pub(crate) fn preprocess(rgb: &[u8], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
    const PATCH: usize = 16;
    const MERGE: usize = 2;
    const MIN_PIXELS: usize = 4096;
    const MAX_PIXELS: usize = 4194304;
    ensure!(rgb.len() >= h * w * 3, "rgb buffer too small");
    let (rh, rw) = smart_resize(h, w, PATCH * MERGE, MIN_PIXELS, MAX_PIXELS);
    let img = resize_u8(rgb, h, w, rh, rw, 3);
    let (gh, gw) = (rh / PATCH, rw / PATCH);
    let mut out = vec![0f32; gh * gw * 3 * PATCH * PATCH];
    for py in 0..gh {
        for px in 0..gw {
            let p = py * gw + px;
            for c in 0..3 {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let v = img[((py * PATCH + y) * rw + px * PATCH + x) * 3 + c];
                        // (v / 255 - 0.5) / 0.5
                        out[p * 3 * PATCH * PATCH + c * PATCH * PATCH + y * PATCH + x] = v as f32 * (2.0 / 255.0) - 1.0;
                    }
                }
            }
        }
    }
    Ok((out, gh, gw))
}

/// The 3-D positions of a prompt that carries images, as `get_rope_index` builds them: text grows by one on all
/// three planes; an image block is `t` frames of `h/merge` rows of `w/merge + 1` tokens (every merged row ends
/// with a newline), all three planes offset by the block's start; the text after it continues at the block's
/// maximum plus one.
pub(crate) fn image_positions(
    tokens: &[u32],
    image_token: u32,
    merge: usize,
    grids: &[(usize, usize, usize)],
) -> Result<Vec<[i32; 3]>> {
    ensure!(merge > 0, "bad merge size");
    let mut out = Vec::with_capacity(tokens.len());
    let mut pos: i32 = 0;
    let mut img = 0usize;
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] != image_token {
            out.push([pos; 3]);
            pos += 1;
            i += 1;
            continue;
        }
        let start = i;
        while i < tokens.len() && tokens[i] == image_token {
            i += 1;
        }
        let (gt, gh, gw) = *grids.get(img).context("more image blocks than images")?;
        img += 1;
        let (hm, wm) = (gh / merge, gw / merge);
        ensure!(hm > 0 && wm > 0, "image grid smaller than the merge size");
        let rows = gt * hm * (wm + 1);
        ensure!(i - start == rows, "image block {} tokens, grid wants {rows}", i - start);
        let block = hm.max(wm + 1) as i32;
        let mut mx = pos;
        for t in 0..gt as i32 {
            let base = pos + t * block;
            for r in 0..hm as i32 {
                for c in 0..wm as i32 + 1 {
                    out.push([base, base + r, base + c]);
                    mx = mx.max(base + r).max(base + c);
                }
            }
        }
        pos = mx + 1;
    }
    ensure!(img == grids.len(), "fewer image blocks than images");
    Ok(out)
}

/// The precision `nn.RMSNorm` uses when its eps is left to the dtype (the Engram norms do that).
const BF16_EPS: f32 = 0.0;

/// The Engram memory of one layer, added to the streams before the attention.
///
/// The hashes and the 4-bit row reads are host work (the table stays in the file); the two projections that map the
/// memory to and from the hidden width are GEMMs on the device. The gates, the per-stream scales and the sum are
/// host arithmetic for this first pass: `t` is a prompt, so it is a few hundred KiB per Engram layer.
fn engram(
    model: &crate::engine::Model,
    e: &Engram,
    streams: &mut DeviceBuffer,
    hist: &[u32],
    tokens: &[u32],
    t: usize,
    sc: &mut crate::session::Scratch,
) -> Result<()> {
    use crate::engine::Act;
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let st = &model.stream;
    let (hidden, hc) = (c.hidden, h.hc_mult);
    let per_order = e.heads;
    let orders = e.head_vocab.len() / per_order;
    ensure!(orders >= 1 && orders + 1 == e.hash_mult.len(), "engram orders");
    let mut offsets = Vec::with_capacity(e.head_vocab.len());
    let mut off = 0usize;
    for &n in &e.head_vocab {
        offsets.push(off);
        off += n;
    }
    // One row of the shared table per (token, hash head). An n-gram reaches back into the tokens of earlier calls
    // (`hist`, the session's last ones): a decode step or a later prefill chunk sees the same n-grams as one prefill.
    let seq: Vec<u32> = hist.iter().chain(tokens).copied().collect();
    let base = hist.len();
    let mut ids = Vec::with_capacity(t * e.head_vocab.len());
    for ti in 0..t {
        let mut mixed = [0i64; 3];
        for (o, m) in mixed.iter_mut().enumerate().take(e.hash_mult.len()) {
            let id = if base + ti >= o { seq[base + ti - o] } else { e.pad };
            let id = if (id as usize) < e.lookup.len() { id } else { e.pad };
            *m = (e.lookup[id as usize] as i64).wrapping_mul(e.hash_mult[o]);
        }
        for k in 0..orders {
            let n = k + 2;
            let mut mix = mixed[0];
            for &m in mixed.iter().take(n).skip(1) {
                mix ^= m;
            }
            for head in 0..per_order {
                let modulus = e.head_vocab[k * per_order + head].max(1) as i64;
                ids.push(offsets[k * per_order + head] + mix.rem_euclid(modulus) as usize);
            }
        }
    }
    if std::env::var("MALLEUS_HY4_TRACE").is_ok() {
        eprintln!("engram ids {:?}", &ids[..ids.len().min(4)]);
    }
    let mut rows = e.table.rows(&ids)?;
    let heads = e.head_vocab.len();
    if std::env::var("MALLEUS_HY4_TRACE").is_ok() {
        eprintln!(
            "raw row0 {:?} row1 {:?} nw1 {:?} heads {heads} dim {}",
            &rows[..4],
            &rows[96..100],
            &e.head_norms[96..100],
            e.dim
        );
    }
    for (r, row) in rows.chunks_mut(e.dim).enumerate() {
        if false && e.fp8 {
            fp8_round_trip(row);
        }
        let w = &e.head_norms[(r % heads) * e.dim..][..e.dim];
        let ss = row.iter().map(|v| v * v).sum::<f32>() / e.dim as f32;
        let inv = 1.0 / (ss + BF16_EPS).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = *v * inv * w[i];
        }
    }
    if std::env::var("MALLEUS_HY4_TRACE").is_ok() {
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        eprintln!(
            "engram ids {:?} row0 {:?} gamma rms {:.5}",
            &ids[..ids.len().min(4)],
            rows[..rows.len().min(4)]
                .iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>(),
            rms(&e.pre_gamma)
        );
        let _ = rms;
        let _ = &rows;
    }
    if std::env::var("MALLEUS_HY4_TRACE").is_ok()
        && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR")
    {
        let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(std::path::Path::new(&dir).join("E_emb_norm.f32"), bytes)?;
    }
    let emb = DeviceBuffer::from_slice(&rows)?;
    let main = crate::session::f32s(t * hidden)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &main, Act::X(&emb), &e.value, t, false)?;
    let mut keys = Vec::with_capacity(hc);
    for k in &e.key {
        let buf = crate::session::f32s(t * hidden)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &buf, Act::X(&emb), k, t, false)?;
        keys.push(buf);
    }
    // The gate per stream on the device: the value and the keys get their bias there, and the streams are updated
    // in place. On the host this was a download and an upload of all the streams per Engram layer.
    for (si, key) in keys.iter().enumerate() {
        model.k.get("hy4_engram_gate").launch(
            (t as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&*streams).into(),
                (&main).into(),
                key.into(),
                (&e.gate_params).into(),
                (&sc.ids).into(),
                (h.image_token as i32).into(),
                hc.into(),
                si.into(),
                hidden.into(),
                BF16_EPS.into(),
            ],
        )?;
    }
    Ok(())
}

/// Hy4's vision tower (the `vit.*` tensors): HunYuan's ViT with Hy4's patch merger.
///
/// LayerNorm with bias, exact GELU, full per-image attention, no RoPE. The patch embed's convolution is a
/// `[1152, 3 * 16 * 16]` GEMM over the patches, and the merger's 2x2 convolution a `[4608, 4 * 1152]` one over the
/// packed 2x2 blocks; the position grid is interpolated on the host (bilinear, align_corners=False, as the module).
#[allow(dead_code)]
pub(crate) struct Vision {
    pub(crate) patch: Linear,
    pub(crate) patch_bias: DeviceBuffer,
    /// The 128x128 learned patch grid (the checkpoint's first row is a class token, never used).
    pub(crate) pos: Vec<f32>,
    pub(crate) blocks: Vec<VBlock>,
    pub(crate) norm: DeviceBuffer,
    pub(crate) proj0: Linear,
    pub(crate) proj0_bias: DeviceBuffer,
    pub(crate) proj2: Linear,
    pub(crate) proj2_bias: DeviceBuffer,
    pub(crate) newline: Vec<f32>,
    pub(crate) hidden: usize,
    pub(crate) inter: usize,
    pub(crate) heads: usize,
    pub(crate) edge: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

#[allow(dead_code)]
pub(crate) struct VBlock {
    pub(crate) ln1: (DeviceBuffer, DeviceBuffer),
    pub(crate) q: Linear,
    pub(crate) k: Linear,
    pub(crate) v: Linear,
    pub(crate) o: Linear,
    pub(crate) qb: DeviceBuffer,
    pub(crate) kb: DeviceBuffer,
    pub(crate) vb: DeviceBuffer,
    pub(crate) ob: DeviceBuffer,
    pub(crate) ln2: (DeviceBuffer, DeviceBuffer),
    pub(crate) fc1: Linear,
    pub(crate) fc2: Linear,
    pub(crate) f1b: DeviceBuffer,
    pub(crate) f2b: DeviceBuffer,
}

impl Vision {
    pub(crate) fn load(src: &dyn TensorSource, up: &mut Uploader, cfg: &crate::config::Hy4Vision) -> Result<Self> {
        let (hidden, inter, heads, edge) = (cfg.hidden, cfg.inter, cfg.heads, cfg.edge);
        let out = cfg.out;
        let lin = |up: &mut Uploader, n: &str, rows: usize, cols: usize| -> Result<Linear> {
            up.linear(src, &format!("vit.{n}"), rows, cols)
        };
        let bias = |up: &mut Uploader, n: &str, len: usize| -> Result<DeviceBuffer> {
            up.tensor(src, &format!("vit.{n}.bias"), &[len])
        };
        // The convolutions are GEMMs: their weight flattening is the im2col order the kernels see.
        let conv = |up: &mut Uploader, n: &str, rows: usize, cols: usize, shape: &[usize]| -> Result<Linear> {
            Ok(Linear {
                w: crate::weights::Weights::Bf16(up.tensor(src, &format!("vit.{n}.weight"), shape)?),
                prefill: None,
                n: rows,
                k: cols,
            })
        };
        let patch = conv(
            up,
            "embeddings.patch_embedding",
            hidden,
            3 * 16 * 16,
            &[hidden, 3, 16, 16],
        )?;
        let patch_bias = bias(up, "embeddings.patch_embedding", hidden)?;
        // The learned grid, without the class token row and as f32 for the host interpolation.
        let pos = bf16_vec(
            src,
            "vit.embeddings.position_embedding.weight",
            (edge * edge + 1) * hidden,
        )?;
        let pos = pos[hidden..].to_vec();
        let n_layers = src
            .names()
            .iter()
            .filter(|n| n.starts_with("vit.layers.") && n.ends_with(".input_layernorm.weight"))
            .count();
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            blocks.push(VBlock {
                ln1: (
                    up.tensor(src, &format!("vit.{}.weight", p("input_layernorm")), &[hidden])?,
                    up.tensor(src, &format!("vit.{}.bias", p("input_layernorm")), &[hidden])?,
                ),
                q: lin(up, &p("self_attn.q_proj"), hidden, hidden)?,
                k: lin(up, &p("self_attn.k_proj"), hidden, hidden)?,
                v: lin(up, &p("self_attn.v_proj"), hidden, hidden)?,
                o: lin(up, &p("self_attn.o_proj"), hidden, hidden)?,
                qb: bias(up, &p("self_attn.q_proj"), hidden)?,
                kb: bias(up, &p("self_attn.k_proj"), hidden)?,
                vb: bias(up, &p("self_attn.v_proj"), hidden)?,
                ob: bias(up, &p("self_attn.o_proj"), hidden)?,
                ln2: (
                    up.tensor(src, &format!("vit.{}.weight", p("post_attention_layernorm")), &[hidden])?,
                    up.tensor(src, &format!("vit.{}.bias", p("post_attention_layernorm")), &[hidden])?,
                ),
                fc1: lin(up, &p("mlp.dense_h_to_4h"), inter, hidden)?,
                fc2: lin(up, &p("mlp.dense_4h_to_h"), hidden, inter)?,
                f1b: bias(up, &p("mlp.dense_h_to_4h"), inter)?,
                f2b: bias(up, &p("mlp.dense_4h_to_h"), hidden)?,
            });
        }
        Ok(Vision {
            patch,
            patch_bias,
            pos,
            blocks,
            // `before_rms`: RMSNorm without bias, applied raw (one-plus off).
            norm: up.norm(src, "vit.perceive.norm.weight", hidden, true)?,
            proj0: conv(
                up,
                "perceive.proj.0",
                4 * hidden,
                4 * hidden,
                &[4 * hidden, hidden, 2, 2],
            )?,
            proj0_bias: bias(up, "perceive.proj.0", 4 * hidden)?,
            proj2: conv(up, "perceive.proj.2", out, 4 * hidden, &[out, 4 * hidden, 1, 1])?,
            proj2_bias: bias(up, "perceive.proj.2", out)?,
            newline: bf16_vec(src, "vit.perceive.image_newline", out)?,
            hidden,
            inter,
            heads,
            edge,
            rows: cfg.merge,
            cols: cfg.merge,
        })
    }
}

/// Bilinear interpolation of the learned patch grid to `(gh, gw)`, as `torch.nn.functional.interpolate` with
/// `align_corners=False` does it.
fn interp_pos(pos: &[f32], edge: usize, gh: usize, gw: usize, hidden: usize) -> Vec<f32> {
    let (sy, sx) = (edge as f32 / gh as f32, edge as f32 / gw as f32);
    let mut out = vec![0f32; gh * gw * hidden];
    for oy in 0..gh {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(edge - 1);
        let y1 = (y0 + 1).min(edge - 1);
        let wy = fy - y0 as f32;
        for ox in 0..gw {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(edge - 1);
            let x1 = (x0 + 1).min(edge - 1);
            let wx = fx - x0 as f32;
            let dst = (oy * gw + ox) * hidden;
            for c in 0..hidden {
                let p = |y: usize, x: usize| pos[(y * edge + x) * hidden + c];
                let top = p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx;
                let bot = p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx;
                out[dst + c] = top * (1.0 - wy) + bot * wy;
            }
        }
    }
    out
}

/// The vision tower on one image: `pixels` are `[n, 3 * 16 * 16]` (the processor's flat patches), the grid is
/// `gh x gw`. Returns the tower's last hidden state `[n, hidden]` and the merged features
/// `[gh/2 * (gw/2 + 1), out_hidden]`, the newline column appended per row.
pub(crate) fn vision_forward(
    model: &crate::engine::Model,
    v: &Vision,
    pixels: &[f32],
    gh: usize,
    gw: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    use crate::engine::Act;
    let st = &model.stream;
    let c = &model.cfg;
    let n = gh * gw;
    let hidden = v.hidden;
    ensure!(pixels.len() >= n * 3 * 16 * 16, "pixel values too small for the grid");
    let sc = model.scratch(n)?;
    let pix = DeviceBuffer::from_slice(&pixels[..n * 3 * 16 * 16])?;
    let mut x = crate::session::f32s(n * hidden)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &x, Act::X(&pix), &v.patch, n, false)?;
    model.k.get("hy4_bias").launch(
        ((n * hidden).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&x).into(), (&v.patch_bias).into(), hidden.into(), (n * hidden).into()],
    )?;
    // The position grid, interpolated and added on the host.
    let pos = interp_pos(&v.pos, v.edge, gh, gw, hidden);
    let mut xh: Vec<f32> = x.download(n * hidden)?;
    for i in 0..n * hidden {
        xh[i] += pos[i];
    }
    x.upload(&xh)?;
    let ln = crate::session::f32s(n * hidden)?;
    let qq = crate::session::f32s(n * hidden)?;
    let kk = crate::session::f32s(n * hidden)?;
    let vv = crate::session::f32s(n * hidden)?;
    let attn = crate::session::f32s(n * hidden)?;
    let tmp = crate::session::f32s(n * hidden.max(v.inter))?;
    let dim = hidden / v.heads;
    let scale = 1.0 / (dim as f32).sqrt();
    for b in &v.blocks {
        model.k.get("hy4_ln").launch(
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&ln).into(),
                (&x).into(),
                (&b.ln1.0).into(),
                (&b.ln1.1).into(),
                hidden.into(),
            ],
        )?;
        for (dst, w, bias) in [(&qq, &b.q, &b.qb), (&kk, &b.k, &b.kb), (&vv, &b.v, &b.vb)] {
            model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, dst, Act::X(&ln), w, n, false)?;
            model.k.get("hy4_bias").launch(
                ((n * hidden).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[dst.into(), bias.into(), hidden.into(), (n * hidden).into()],
            )?;
        }
        model.k.get("hy4_vit_attn").launch(
            (v.heads as u32, n as u32, 1),
            (256, 1, 1),
            (4096 * 4 + 256 * 4) as u32,
            st,
            &[
                (&attn).into(),
                (&qq).into(),
                (&kk).into(),
                (&vv).into(),
                n.into(),
                v.heads.into(),
                dim.into(),
                scale.into(),
            ],
        )?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &tmp, Act::X(&attn), &b.o, n, false)?;
        model.k.get("hy4_bias").launch(
            ((n * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&tmp).into(), (&b.ob).into(), hidden.into(), (n * hidden).into()],
        )?;
        model.k.get("hy4_add").launch(
            ((n * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&x).into(), (&tmp).into(), (n * hidden).into()],
        )?;
        model.k.get("hy4_ln").launch(
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&ln).into(),
                (&x).into(),
                (&b.ln2.0).into(),
                (&b.ln2.1).into(),
                hidden.into(),
            ],
        )?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &tmp, Act::X(&ln), &b.fc1, n, false)?;
        model.k.get("hy4_bias").launch(
            ((n * v.inter).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&tmp).into(), (&b.f1b).into(), v.inter.into(), (n * v.inter).into()],
        )?;
        model.k.get("hy4_gelu").launch(
            ((n * v.inter).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&tmp).into(), (n * v.inter).into()],
        )?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &vv, Act::X(&tmp), &b.fc2, n, false)?;
        model.k.get("hy4_bias").launch(
            ((n * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&vv).into(), (&b.f2b).into(), hidden.into(), (n * hidden).into()],
        )?;
        model.k.get("hy4_add").launch(
            ((n * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&x).into(), (&vv).into(), (n * hidden).into()],
        )?;
    }
    let tower_out: Vec<f32> = {
        st.synchronize()?;
        x.download(n * hidden)?
    };
    // The merger: before_rms, the 2x2 pack, the two convolutions and the newline per merged row.
    rmsnorm_raw(model, st, &ln, &x, &v.norm, n, hidden)?;
    let (mh, mw) = (gh / 2, gw / 2);
    ensure!(mh > 0 && mw > 0, "the grid must be at least 2x2");
    let nm = mh * mw;
    let packed = crate::session::f32s(nm * 4 * hidden)?;
    model.k.get("hy4_merge_pack").launch(
        ((nm * 4 * hidden).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&packed).into(), (&ln).into(), hidden.into(), mh.into(), mw.into()],
    )?;
    let big = crate::session::f32s(nm * 4 * hidden)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &big, Act::X(&packed), &v.proj0, nm, false)?;
    model.k.get("hy4_bias").launch(
        ((nm * 4 * hidden).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            (&big).into(),
            (&v.proj0_bias).into(),
            (4 * hidden).into(),
            (nm * 4 * hidden).into(),
        ],
    )?;
    model.k.get("hy4_gelu").launch(
        ((nm * 4 * hidden).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&big).into(), (nm * 4 * hidden).into()],
    )?;
    let out_w = c.hidden;
    let o = crate::session::f32s(nm * out_w)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &o, Act::X(&big), &v.proj2, nm, false)?;
    model.k.get("hy4_bias").launch(
        ((nm * out_w).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&o).into(), (&v.proj2_bias).into(), out_w.into(), (nm * out_w).into()],
    )?;
    st.synchronize()?;
    let merged_raw: Vec<f32> = o.download(nm * out_w)?;
    // The newline column goes at the end of every merged row.
    let mut merged = Vec::with_capacity(mh * (mw + 1) * out_w);
    for r in 0..mh {
        merged.extend_from_slice(&merged_raw[r * mw * out_w..(r + 1) * mw * out_w]);
        merged.extend_from_slice(&v.newline);
    }
    *model.scratch_cache.lock().unwrap() = Some(sc);
    Ok((tower_out, merged))
}

/// Hy4's session: the latent and the rotated dims per layer (fp16), no conv or recurrent state.
pub(crate) fn new_session(model: &crate::engine::Model, max_ctx: usize) -> Result<crate::session::Session> {
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let mut s = crate::session::Session {
        max_ctx,
        len: 0,
        next_pos: 0,
        kv: Vec::with_capacity(c.n_layers),
        conv: Vec::new(),
        rec: Vec::new(),
        attn_part: crate::session::f32s(1)?,
        mla_part: None,
        conv_snap: Vec::new(),
        dn_rec: Vec::new(),
        dn_np: 0,
        dn_par: 0,
        snap_t: 0,
        mtp_kv: Vec::new(),
        mtp_len: 0,
        mtp_h: None,
        mtp_pos: Vec::new(),
        mtp_out: None,
        mtp_out_pos: [0; 3],
        mtp_chain: 0,
        ple_state: None,
        ngram_ctx: Vec::new(),
        idx_kb: Vec::new(),
        idx_tk: Vec::new(),
        idx_tp: Vec::new(),
        idx_scores: None,
        idx_sel: None,
        idx_sel_cache: Vec::new(),
        idx_cache_len: 0,
        mtp_idx: None,
        idx_list: None,
        ple_snap: None,
        ngram_prev: Vec::new(),
        inkling2: None,
        hy4_spec: None,
        hy4_draft: Default::default(),
        ix_tk: Vec::new(),
        ix_gk: Vec::new(),
        ix_pooled: Vec::new(),
        ix_pos: None,
    };
    let n = crate::weights::kv_ctx(max_ctx);
    let w = model.hy4.as_ref().context("no hy4 weights")?;
    for li in 0..c.n_layers {
        let rows = match ring_rows(c, li, max_ctx) {
            0 => n,
            r => r,
        };
        s.kv.push((
            DeviceBuffer::zeroed(rows * h.kv_lora * 2)?,
            DeviceBuffer::zeroed(rows * h.rope * 2)?,
        ));
        // The compressed indexer caches the token keys of every indexed layer and the groups that closed; the
        // sliding layers have neither, so they hold `None` and the layer index stays the one of `s.kv`.
        match w.layers[li].attn.indexer {
            Some(_) => {
                s.ix_tk.push(Some(DeviceBuffer::zeroed(n * h.index_dim * 2)?));
                s.ix_gk
                    .push(Some(DeviceBuffer::zeroed((n / h.index_pool + 1) * h.index_dim * 2)?));
                s.ix_pooled.push(0);
            }
            None => {
                s.ix_tk.push(None);
                s.ix_gk.push(None);
                s.ix_pooled.push(0);
            }
        }
    }
    if c.kind.contains(&crate::config::MixerKind::Hy4Indexed) {
        s.ix_pos = Some(DeviceBuffer::zeroed(n * 3 * 4)?);
    }
    Ok(s)
}

/// Query rows scored per pass of the compressed indexer: bounds its score buffer to `IX_CHUNK * groups` floats,
/// which at 262k of context is 16 MB instead of the whole prompt's worth.
const IX_CHUNK: usize = 64;

/// One step of the compressed indexer of an indexed layer (issue #40).
///
/// The token keys go into the layer's fp16 cache on every step (the reference caches them too, even below the
/// budget); a group is pooled once, when its last token arrives, and rotated at its first token's position. Below
/// `index_topk` the layer attends densely and nothing else runs (`None`), which is what the reference does: the
/// indexer only pays off once the context outgrows the budget.
///
/// Above it, every query scores the groups that closed before it (`sum_h head_w * relu(q . pooled_key)`), keeps
/// the `index_topk / pool - 1` best plus its own group, and the returned list is what its attention walks.
fn indexer_step(
    model: &crate::engine::Model,
    ix: &Indexer,
    sess: &mut crate::session::Session,
    sc: &crate::session::Scratch,
    li: usize,
    q_resid: &DeviceBuffer,
    block_in: &DeviceBuffer,
    t: usize,
    dump: &dyn Fn(&str, usize, &DeviceBuffer, usize) -> Result<()>,
) -> Result<Option<DeviceBuffer>> {
    use crate::engine::Act;
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let st = &model.stream;
    let (dim, pool, nh, rope) = (h.index_dim, h.index_pool, h.index_heads, h.rope);
    let (len, nkeys) = (sess.len, sess.len + t);
    // Debug A/B (`MALLEUS_HY4_IX_OFF=1`): the indexed layers attend densely, as they did before the indexer.
    if std::env::var("MALLEUS_HY4_IX_OFF").is_ok() {
        return Ok(None);
    }
    let pooled = sess.ix_pooled[li];
    let ncg = nkeys / pool;
    let group_topk = h.index_topk / pool;
    // The indexer's own norms multiply by the weight, so the kernels read them as f32 raw (no plus one).
    {
        let cache = sess.ix_tk[li].as_ref().context("hy4 indexer: token-key cache")?;
        let tk = crate::session::f32s(t * dim)?;
        model.gemm(
            st,
            &sc.xr,
            &sc.xh,
            &sc.xsc,
            &tk,
            Act::X(block_in),
            &ix.token_key,
            t,
            false,
        )?;
        model.k.get("hy4_ix_keys").launch(
            (t as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                cache.into(),
                (&tk).into(),
                (&ix.token_key_norm).into(),
                len.into(),
                t.into(),
                dim.into(),
                c.eps.into(),
            ],
        )?;
        dump("ix_tk", li, &tk, t * dim)?;
        if ncg > pooled {
            let gk = sess.ix_gk[li].as_ref().context("hy4 indexer: group-key cache")?;
            let ng = ncg - pooled;
            model.k.get("hy4_ix_pool").launch(
                (ng as u32, 1, 1),
                (dim as u32, 1, 1),
                ((pool * dim + 128) * 4) as u32,
                st,
                &[
                    gk.into(),
                    cache.into(),
                    (&ix.value).into(),
                    (&ix.gate).into(),
                    (&ix.intra_ape).into(),
                    (&ix.key_norm).into(),
                    pooled.into(),
                    ng.into(),
                    dim.into(),
                    pool.into(),
                    c.eps.into(),
                ],
            )?;
            let pos = sess.ix_pos.as_ref().context("hy4 indexer: positions")?;
            model.k.get("hy4_ix_krope").launch(
                ((ng * (rope / 2)).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    gk.into(),
                    pos.into(),
                    pooled.into(),
                    ng.into(),
                    pool.into(),
                    dim.into(),
                    rope.into(),
                    h.mrope_section[1].into(),
                    h.mrope_section[2].into(),
                    h.theta_indexed.into(),
                ],
            )?;
            // The group keys are fp16 in the cache: the dump goes through the cast the other caches use.
            if std::env::var("MALLEUS_HY4_TRACE").is_ok()
                && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR")
            {
                let f = crate::session::f32s(ncg * dim)?;
                model.ew(
                    st,
                    "hy4_cast_f32",
                    ncg * dim,
                    &[(&f).into(), gk.into(), (ncg * dim).into()],
                )?;
                let v: Vec<f32> = f.download(ncg * dim)?;
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(format!("IX{li}_gk.f32")), bytes)?;
            }
        }
    }
    if ncg > pooled {
        sess.ix_pooled[li] = ncg;
    }
    let force_all = std::env::var("MALLEUS_HY4_IX_ALL").is_ok();
    if nkeys <= h.index_topk && !force_all {
        return Ok(None);
    }
    let gk = sess.ix_gk[li].as_ref().context("hy4 indexer: group-key cache")?;
    // The query: `query_projection` of the q_a latent, rotated at the token's own position (the same M-RoPE).
    let qd = nh * dim;
    let iq = crate::session::f32s(t * qd)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &iq, Act::X(q_resid), &ix.query, t, false)?;
    model.k.get("hy4_ix_qrope").launch(
        ((t * nh * (rope / 2)).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            (&iq).into(),
            (&sc.pos).into(),
            t.into(),
            nh.into(),
            dim.into(),
            rope.into(),
            h.mrope_section[1].into(),
            h.mrope_section[2].into(),
            h.theta_indexed.into(),
        ],
    )?;
    dump("ix_q", li, &iq, t * qd)?;
    let hw = crate::session::f32s(t * nh)?;
    model.gemm(
        st,
        &sc.xr,
        &sc.xh,
        &sc.xsc,
        &hw,
        Act::X(block_in),
        &ix.head_weight,
        t,
        false,
    )?;
    // `MALLEUS_HY4_IX_ALL` recalls every closed group (the list outgrows the budget): the sparse path then has to
    // reproduce the dense one, which is how the kernel is checked against the attention already verified.
    let slots = if force_all { ncg + 1 } else { group_topk };
    let nsel = if force_all { ncg } else { group_topk - 1 };
    let mut list = DeviceBuffer::alloc(t * slots * 4)?;
    let keys = DeviceBuffer::alloc(IX_CHUNK.min(t) * ncg.max(1) * 4)?;
    let scale = 1.0f32 / ((nh * dim) as f32).sqrt();
    for ti0 in (0..t).step_by(IX_CHUNK) {
        let nq = IX_CHUNK.min(t - ti0);
        if force_all {
            let mut flat = vec![-1i32; nq * slots];
            for i in 0..nq {
                let own = (len + ti0 + i) / pool;
                for g in 0..own.min(slots) {
                    flat[i * slots + g] = g as i32;
                }
                if own < slots {
                    flat[i * slots + own] = own as i32;
                }
            }
            list.upload_at(ti0 * slots * 4, &flat)?;
            continue;
        }
        if nh == 16 && dim == 128 && std::env::var("MALLEUS_HY4_IX_SCORE_WMMA").map_or(true, |v| v != "0") {
            model.k.get("hy4_ix_score_wmma").launch(
                (ncg.div_ceil(256) as u32, nq.div_ceil(8) as u32, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&keys).into(),
                    (&iq).into(),
                    gk.into(),
                    (&hw).into(),
                    len.into(),
                    ti0.into(),
                    ncg.into(),
                    nq.into(),
                    pool.into(),
                    scale.into(),
                ],
            )?;
        } else {
            model.k.get("hy4_ix_score").launch(
                (ncg.div_ceil(256) as u32, nq as u32, 1),
                (256, 1, 1),
                ((nh * dim + nh) * 4) as u32,
                st,
                &[
                    (&keys).into(),
                    (&iq).into(),
                    gk.into(),
                    (&hw).into(),
                    len.into(),
                    ti0.into(),
                    ncg.into(),
                    nq.into(),
                    nh.into(),
                    dim.into(),
                    pool.into(),
                    scale.into(),
                ],
            )?;
        }
        // Debug: the order-preserving scores of this chunk, appended in query order, so the selection can be
        // compared against the reference's own logits (the transform is inverted in ref_indexer_replay.py).
        if std::env::var("MALLEUS_HY4_TRACE").is_ok()
            && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR")
        {
            use std::io::Write;
            let v: Vec<u32> = keys.download(nq * ncg)?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let p = std::path::Path::new(&dir).join(format!("IX{li}_keys.bin"));
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p)?;
            f.write_all(&bytes)?;
        }
        model.k.get("hy4_ix_select").launch(
            (nq as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&list).into(),
                (&keys).into(),
                len.into(),
                ti0.into(),
                pool.into(),
                ncg.into(),
                nsel.into(),
                slots.into(),
            ],
        )?;
    }
    if std::env::var("MALLEUS_HY4_TRACE").is_ok()
        && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR")
    {
        dump("ix_hw", li, &hw, t * nh)?;
        let v: Vec<i32> = list.download(t * slots)?;
        let mut text = String::new();
        for ti in 0..t {
            let row: Vec<String> = v[ti * slots..][..slots].iter().map(|x| x.to_string()).collect();
            text.push_str(&row.join(" "));
            text.push('\n');
        }
        std::fs::write(std::path::Path::new(&dir).join(format!("IX{li}_list.txt")), text)?;
        std::fs::write(
            std::path::Path::new(&dir).join(format!("IX{li}_list_meta.txt")),
            format!("{len} {t} {ncg} {nsel} {slots}"),
        )?;
    }
    Ok(Some(list))
}

/// Slices of the `c * hidden` input that `ihc_pre_part` spreads one token over.
const IHC_SLICES: usize = 24;

/// Where a layer's attention gets its sparse key list from: the target runs its own compressed indexer, the
/// drafter brings the lists it derived from the target's.
pub(crate) enum IxSource {
    Target,
    Given(Option<DeviceBuffer>),
}

/// The per-step buffers of the gated MLA, sized by the caller for the wider of the two head geometries.
pub(crate) struct AttnBufs<'a> {
    q: &'a DeviceBuffer,
    qa: &'a DeviceBuffer,
    kva: &'a DeviceBuffer,
    gate: &'a DeviceBuffer,
    attn_out: &'a DeviceBuffer,
    o: &'a DeviceBuffer,
    qt: &'a DeviceBuffer,
    lat_merged: &'a DeviceBuffer,
}

/// One gated MLA with its learned sink on `t` rows: `hn2` is the normed input, the output (after `post_proj_norm`)
/// goes to `hn`, and the keys are the KV cache of layer `li` (rope base `theta`, a sliding `window` or 0, a `ring`
/// of rows or 0). Returns the sparse list the attention walked, if any.
#[allow(clippy::too_many_arguments)]
fn mla_attention(
    model: &crate::engine::Model,
    sess: &mut crate::session::Session,
    sc: &crate::session::Scratch,
    a: &Attn,
    li: usize,
    theta: f32,
    window: usize,
    ring: usize,
    hn: &DeviceBuffer,
    hn2: &DeviceBuffer,
    b: &AttnBufs,
    t: usize,
    ix: IxSource,
    dump: &dyn Fn(&str, usize, &DeviceBuffer, usize) -> Result<()>,
) -> Result<Option<DeviceBuffer>> {
    use crate::engine::Act;
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let st = &model.stream;
    let eps = h.hc_eps;
    let (q, qa, kva, gate, attn_out, o, qt, lat_merged) =
        (b.q, b.qa, b.kva, b.gate, b.attn_out, b.o, b.qt, b.lat_merged);
    let qd = a.heads * (a.nope + a.rope);
    let vd = a.heads * a.v;
    let hidden = c.hidden;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, qa, Act::X(hn2), &a.q_a, t, false)?;
    model.rmsnorm(st, hn, qa, &a.q_a_norm, t, h.q_lora)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, q, Act::X(hn), &a.q_b, t, false)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, kva, Act::X(hn2), &a.kv_a, t, false)?;
    dump("q", li, q, t * qd)?;
    dump("ix_qresid", li, hn, t * h.q_lora)?;
    // The compressed indexer of an indexed layer: `hn` is the q_a latent its query reads and `hn2` the block
    // input its token keys and head weights read. `ix_list` is `Some` once the context outgrows `index_topk`.
    let ix_list = match (ix, &a.indexer) {
        (IxSource::Target, Some(i)) => indexer_step(model, i, sess, sc, li, hn, hn2, t, dump)?,
        (IxSource::Target, None) => None,
        (IxSource::Given(l), _) => l,
    };
    model.k.get("hy4_rope").launch(
        // The layer's own geometry: a sliding layer has twice the heads of an indexed one, so the indexed
        // shape would leave half of its query unrotated and the rest rotated at the wrong token's position.
        ((t * (a.heads + 1) * (a.rope / 2)).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            q.into(),
            kva.into(),
            (&sc.pos).into(),
            t.into(),
            a.heads.into(),
            a.nope.into(),
            h.kv_lora.into(),
            a.rope.into(),
            h.mrope_section[1].into(),
            h.mrope_section[2].into(),
            theta.into(),
        ],
    )?;
    let (ckv, ckrot) = &sess.kv[li];
    model.k.get("hy4_cache_write").launch(
        ((t * (h.kv_lora + h.rope)).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            ckv.into(),
            ckrot.into(),
            kva.into(),
            (&a.kv_a_norm).into(),
            eps.into(),
            sess.len.into(),
            t.into(),
            h.kv_lora.into(),
            h.rope.into(),
            ring.into(),
        ],
    )?;
    // Post-rope query and rotated key dims, for the score reconstruction against the reference replay.
    dump("q_rot", li, q, t * qd)?;
    if std::env::var("MALLEUS_HY4_DUMP_KROT").is_ok() {
        model.ew(
            st,
            "hy4_cast_f32",
            t * h.rope,
            &[(&sc.big1).into(), ckrot.into(), (t * h.rope).into()],
        )?;
        dump("krot", li, &sc.big1, t * h.rope)?;
    }
    // Absorbed MLA: `kv_b`'s QK half is folded into the query once per token and head, the split walks the cached
    // latents (the sliding layers only the last `window` of them), and the combine applies its V half and the gate.
    let nsplit = 8usize;
    if t > 8 {
        // A chunk: one WMMA GEMM per head over all its tokens.
        model.k.get("hy4_absorb_wmma").launch(
            (h.kv_lora.div_ceil(64) as u32, t.div_ceil(64) as u32, a.heads as u32),
            (256, 1, 1),
            0,
            st,
            &[
                qt.into(),
                q.into(),
                (&a.kv_b).into(),
                t.into(),
                h.kv_lora.into(),
                a.nope.into(),
                (a.heads * (a.nope + a.rope)).into(),
                (a.nope + a.rope).into(),
                (a.heads * h.kv_lora).into(),
                ((a.nope + a.v) * h.kv_lora).into(),
            ],
        )?;
    } else {
        model.k.get("hy4_absorb").launch(
            ((t * a.heads) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                qt.into(),
                q.into(),
                (&a.kv_b).into(),
                t.into(),
                a.heads.into(),
                a.nope.into(),
                h.kv_lora.into(),
                a.rope.into(),
                (a.nope + a.v).into(),
            ],
        )?;
    }
    dump("qt", li, qt, t * a.heads * h.kv_lora)?;
    if let Ok(dir) = std::env::var("MALLEUS_HY4_DUMP_W") {
        st.synchronize()?;
        let v: Vec<f32> = a.kv_b.download(a.kv_b.len() / 4)?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(std::path::Path::new(&dir).join(format!("W{li}_kvb.f32")), bytes)?;
        // The gate weight, dequantized like linear_f32 does (MLS4 -> f32), to check the row order.
        if let crate::weights::Weights::Mls4 { q, sc } = &a.gate.w {
            use malleus_gguf::quant::{self, BLOCK};
            let (n, k) = (a.gate.n, a.gate.k);
            let g = k / BLOCK;
            let signs = quant::hadamard_signs(malleus_gguf::quant::HADAMARD_SEED);
            let qw = q.download::<u8>(n * k / 2)?;
            let sw = sc.download::<u8>(n * g * 2)?;
            let mut out = vec![0f32; n * k];
            for r in 0..n {
                for j in 0..g {
                    let at = (r * g + j) * 2;
                    let s = half::f16::from_le_bytes([sw[at], sw[at + 1]]).to_f32();
                    let mut blk = [0f32; BLOCK];
                    quant::dequant_int4(&qw[r * k / 2 + j * BLOCK / 2..][..BLOCK / 2], s, &mut blk);
                    quant::unrotate_block(&mut blk, &signs);
                    out[r * k + j * BLOCK..][..BLOCK].copy_from_slice(&blk);
                }
            }
            let bytes: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join(format!("W{li}_gate.f32")), bytes)?;
        }
    }
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, gate, Act::X(hn2), &a.gate, t, false)?;
    dump("gate", li, gate, t * vd)?;
    let sink = a.sink.as_ref().context("hy4 gated MLA needs the learned sink")?;
    // A chunk on the dense path: one WMMA kernel per 32 heads of a token, which writes the weighted latent itself.
    let wmma_attn = t > 8
        && (ix_list.is_none() || h.index_pool == 4)
        && a.heads.is_multiple_of(32)
        && std::env::var("MALLEUS_HY4_ATTN_WMMA").map_or(true, |v| v != "0");
    if wmma_attn {
        model.k.get("hy4_mla_attn_wmma").launch(
            ((t * a.heads / 32) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                lat_merged.into(),
                qt.into(),
                q.into(),
                ckv.into(),
                ckrot.into(),
                sink.into(),
                sess.len.into(),
                a.heads.into(),
                a.nope.into(),
                h.kv_lora.into(),
                a.rope.into(),
                (1.0f32 / ((a.nope + a.rope) as f32).sqrt()).into(),
                window.into(),
                ix_list.as_ref().map_or(Arg::U64(0), |l| l.into()),
                ix_list.as_ref().map_or(0, |l| l.len() / (t * 4)).into(),
                h.index_pool.into(),
                ring.into(),
            ],
        )?;
    } else {
        let part = crate::session::f32s(t * a.heads * nsplit * (h.kv_lora + 2))?;
        if let Some(list) = &ix_list {
            // The groups this query recalled (`index_topk / pool - 1` plus its own), or every closed one under
            // `MALLEUS_HY4_IX_ALL`; the same split and merge as the dense path, one kernel over the list.
            let slots = list.len() / (t * 4);
            model.k.get("hy4_mla_split_ix").launch(
                ((t * a.heads) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&part).into(),
                    qt.into(),
                    q.into(),
                    ckv.into(),
                    ckrot.into(),
                    list.into(),
                    sink.into(),
                    sess.len.into(),
                    t.into(),
                    a.heads.into(),
                    a.nope.into(),
                    h.kv_lora.into(),
                    a.rope.into(),
                    (1.0f32 / ((a.nope + a.rope) as f32).sqrt()).into(),
                    nsplit.into(),
                    h.index_pool.into(),
                    slots.into(),
                ],
            )?;
        } else {
            model.k.get("hy4_mla_split").launch(
                ((t * a.heads) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&part).into(),
                    qt.into(),
                    q.into(),
                    ckv.into(),
                    ckrot.into(),
                    sink.into(),
                    sess.len.into(),
                    t.into(),
                    a.heads.into(),
                    a.nope.into(),
                    h.kv_lora.into(),
                    a.rope.into(),
                    (1.0f32 / ((a.nope + a.rope) as f32).sqrt()).into(),
                    nsplit.into(),
                    window.into(),
                    ring.into(),
                ],
            )?;
        }
        // The merge into a global buffer, then the V half of kv_b_proj and the gate: two kernels so that the
        // latent the V projection reads is the one written, not a block of shared memory.
        model.k.get("hy4_mla_merge").launch(
            ((t * a.heads) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[lat_merged.into(), (&part).into(), h.kv_lora.into(), nsplit.into()],
        )?;
        if std::env::var("MALLEUS_HY4_DUMP_PART").is_ok() && li == 0 {
            st.synchronize()?;
            let dir = std::env::var("MALLEUS_DUMP_DIR")?;
            let n_lat = crate::weights::kv_ctx(sess.max_ctx) * h.kv_lora;
            let cache: Vec<u16> = ckv.download(n_lat)?;
            let bytes: Vec<u8> = cache.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("T0_ckv.f16"), bytes)?;
            let v: Vec<f32> = part.download(t * a.heads * nsplit * (h.kv_lora + 2))?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("T0_part.f32"), bytes)?;
            let lm: Vec<f32> = lat_merged.download(t * a.heads * h.kv_lora)?;
            let bytes: Vec<u8> = lm.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("T0_latmerged.f32"), bytes)?;
            let gl: Vec<f32> = gate.download(t * vd)?;
            let bytes: Vec<u8> = gl.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("T0_gate2.f32"), bytes)?;
        }
    }
    // The V half of `kv_b_proj`: one bf16 WMMA GEMM per head over the whole chunk (a batch on `blockIdx.z`), then
    // the per-(head, dim) gate. As a per-(token, head) GEMV this read the 256 rows of a head one per thread — 32
    // cache lines per instruction — and was 90 % of the prefill.
    let args = [
        attn_out.into(),
        lat_merged.into(),
        (&a.kv_b).into(),
        t.into(),
        a.v.into(),
        h.kv_lora.into(),
        (a.heads * h.kv_lora).into(),
        (a.heads * a.v).into(),
        ((a.nope + a.v) * h.kv_lora).into(),
        (a.nope * h.kv_lora).into(),
    ];
    // A few rows (decode, a short verify) read each bf16 row once in a GEMV; a chunk goes through the WMMA tiles.
    let (name, grid) = if t <= 8 {
        ("hy4_vhalf_gemv", (a.v.div_ceil(8) as u32, a.heads as u32, 1))
    } else {
        (
            "gemm_bf16_wmma",
            (a.v.div_ceil(64) as u32, t.div_ceil(64) as u32, a.heads as u32),
        )
    };
    model.k.get(name).launch(grid, (256, 1, 1), 0, st, &args)?;
    model.ew(
        st,
        "hy4_gate_mul",
        t * vd,
        &[attn_out.into(), gate.into(), (t * vd).into()],
    )?;
    dump("attn_raw2", li, attn_out, t * vd)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, o, Act::X(attn_out), &a.o, t, false)?;
    model.rmsnorm(st, hn, o, &a.post_norm, t, hidden)?;
    dump("attn", li, hn, t * hidden)?;
    Ok(ix_list)
}

/// The iHC pre-mix of one sublayer: `out` the block input, `post` the residual gates.
fn ihc_pre(
    model: &crate::engine::Model,
    w: &Ihc,
    out: &DeviceBuffer,
    post: &DeviceBuffer,
    streams: &DeviceBuffer,
    part: &DeviceBuffer,
    t: usize,
) -> Result<()> {
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let st = &model.stream;
    let (hc, hidden) = (h.hc_mult, c.hidden);
    model.k.get("ihc_pre_part").launch(
        (t as u32, IHC_SLICES as u32, 1),
        (256, 1, 1),
        0,
        st,
        &[
            part.into(),
            streams.into(),
            (&w.f).into(),
            t.into(),
            hc.into(),
            hidden.into(),
        ],
    )?;
    model.k.get("ihc_pre").launch(
        (t as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            out.into(),
            post.into(),
            streams.into(),
            part.into(),
            (&w.base).into(),
            (&w.scale).into(),
            t.into(),
            hc.into(),
            hidden.into(),
            IHC_SLICES.into(),
            h.hc_magnitude.into(),
            h.hc_eps.into(),
        ],
    )?;
    Ok(())
}

/// One forward of the text graph: iHC, the Engram memories, the gated MLA with its sink (dense below `index_topk`,
/// over the recalled groups above it), the compressed indexer of the indexed layers, the latent MoE and the head.
pub(crate) fn forward(
    model: &crate::engine::Model,
    sess: &mut crate::session::Session,
    tokens: &[u32],
    positions: Option<&[[i32; 3]]>,
    n_logits: usize,
) -> Result<Vec<f32>> {
    use crate::engine::Act;
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let w = model.hy4.as_ref().context("no hy4 weights")?;
    // Debug: run the vision tower alone on the processor's pixels (`MALLEUS_HY4_VISION_PIX` + `..._GRID`), dump
    // the two stages and stop. It is how the tower is compared against `--vision` in the reference tool.
    if let Ok(pix) = std::env::var("MALLEUS_HY4_VISION_PIX") {
        let v = model.hy4_vision.as_ref().context("the file has no vision tower")?;
        let bytes = std::fs::read(&pix)?;
        let pixels: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let grid = std::env::var("MALLEUS_HY4_VISION_GRID").context("MALLEUS_HY4_VISION_GRID")?;
        let (gh, gw) = grid.split_once('x').context("grid as HxW")?;
        let (gh, gw): (usize, usize) = (gh.parse()?, gw.parse()?);
        let (tower_back, merged) = vision_forward(model, v, &pixels, gh, gw)?;
        let dir = std::env::var("MALLEUS_DUMP_DIR").context("MALLEUS_DUMP_DIR")?;
        let put = |name: &str, v: &[f32]| -> Result<()> {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            Ok(std::fs::write(std::path::Path::new(&dir).join(name), bytes)?)
        };
        put("V_tower.f32", &tower_back)?;
        put("V_merged.f32", &merged)?;
        bail!(
            "vision tower: {} patches, {} merged tokens of {} (dumped to MALLEUS_DUMP_DIR)",
            gh * gw,
            merged.len() / c.hidden,
            c.hidden
        );
    }
    let t = tokens.len();
    ensure!(t > 0 && n_logits >= 1 && n_logits <= t, "bad hy4 step");
    // A ring holds the window behind every query only up to `hy4_max_chunk()` tokens a call: a longer text prompt is
    // split (an image prompt carries its own positions and features for the whole call, and is never that long).
    if t > hy4_max_chunk() && positions.is_none() && std::env::var_os("MALLEUS_HY4_IMAGE_PNG").is_none() {
        let (head, tail) = tokens.split_at(t - t % hy4_max_chunk());
        let (head, tail) = if tail.len() >= n_logits {
            (head, tail)
        } else {
            tokens.split_at(t - hy4_max_chunk())
        };
        for part in head.chunks(hy4_max_chunk()) {
            forward(model, sess, part, None, 1)?;
        }
        return forward(model, sess, tail, None, n_logits);
    }
    ensure!(
        t <= hy4_max_chunk() || (0..c.n_layers).all(|li| ring_rows(c, li, sess.max_ctx) == 0),
        "a hy4 call of {t} tokens with positions is longer than the sliding ring allows ({})",
        hy4_max_chunk()
    );
    ensure!(
        sess.len + t <= sess.max_ctx,
        "context full ({} + {} > {})",
        sess.len,
        t,
        sess.max_ctx
    );
    let st = &model.stream;
    // Every GEMM of the step runs over the step's own rows (the absorbed MLA never expands the cache), so the scratch
    // is sized on `t`: sized on `len + t` it grew by one row per decode token and was reallocated at every step.
    let mut sc = model.take_scratch(t)?;
    let ids: Vec<i32> = tokens.iter().map(|&v| v as i32).collect();
    // An image prompt: the merged features replace the placeholder rows and the positions are the 3-D M-RoPE
    // ones (an image block is a rectangle of patches with a newline column, not a run of positions).
    let merge = h.vision.as_ref().map_or(2, |v| v.merge);
    let mut image: Option<(usize, usize, Vec<f32>)> = None;
    let mut grid: Option<(usize, usize, usize)> = None;
    // The whole image path from a PNG: the processor, the tower, then the prompt with the features in place.
    if let Ok(png) = std::env::var("MALLEUS_HY4_IMAGE_PNG") {
        let v = model.hy4_vision.as_ref().context("the file has no vision tower")?;
        let bytes = std::fs::read(&png)?;
        let (rgb, ih, iw) = crate::image_pre::decode(&bytes, &crate::image_pre::PreConfig::default())?;
        let (pixels, gh, gw) = preprocess(&rgb, ih, iw)?;
        let (tower_back, merged) = vision_forward(model, v, &pixels, gh, gw)?;
        if std::env::var("MALLEUS_HY4_TRACE").is_ok()
            && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR")
        {
            let put = |name: &str, v: &[f32]| -> Result<()> {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                Ok(std::fs::write(std::path::Path::new(&dir).join(name), bytes)?)
            };
            put("P_pixels.f32", &pixels)?;
            put("P_tower.f32", &tower_back)?;
            put("P_merged.f32", &merged)?;
            std::fs::write(std::path::Path::new(&dir).join("P_grid.txt"), format!("1x{gh}x{gw}"))?;
        }
        eprintln!(
            "image {}x{} -> {}x{} patches, {} merged",
            iw,
            ih,
            gh,
            gw,
            merged.len() / c.hidden
        );
        let start = tokens
            .iter()
            .position(|&x| x == h.image_token)
            .context("no image placeholder in the prompt")?;
        image = Some((start, merged.len() / c.hidden, merged));
        grid = Some((1, gh, gw));
    } else if let Ok(f) = std::env::var("MALLEUS_HY4_IMAGE_FEATURES") {
        let g = std::env::var("MALLEUS_HY4_IMAGE_GRID").context("MALLEUS_HY4_IMAGE_GRID")?;
        let d: Vec<usize> = g
            .split('x')
            .map(|v| v.parse::<usize>().context("grid as TxHxW"))
            .collect::<Result<_>>()?;
        ensure!(d.len() == 3, "grid as TxHxW");
        let start = tokens
            .iter()
            .position(|&x| x == h.image_token)
            .context("no image placeholder in the prompt")?;
        let rows = d[0] * (d[1] / merge) * (d[2] / merge + 1);
        let bytes = std::fs::read(&f)?;
        let v: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        ensure!(v.len() >= rows * c.hidden, "image features too small");
        image = Some((start, rows, v));
    }
    let pos: Vec<i32> = match (&image, positions) {
        (Some((_, _, _)), _) => {
            let g = match grid {
                Some(g) => g,
                None => {
                    let g = std::env::var("MALLEUS_HY4_IMAGE_GRID").context("MALLEUS_HY4_IMAGE_GRID")?;
                    let d: Vec<usize> = g.split('x').filter_map(|v| v.parse().ok()).collect();
                    (d[0], d[1], d[2])
                }
            };
            let p3 = image_positions(tokens, h.image_token, merge, &[g])?;
            if std::env::var("MALLEUS_HY4_TRACE").is_ok()
                && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR")
            {
                let mut text = String::new();
                for plane in 0..3 {
                    let line: Vec<String> = p3.iter().map(|p| p[plane].to_string()).collect();
                    text.push_str(&line.join(" "));
                    text.push('\n');
                }
                std::fs::write(std::path::Path::new(&dir).join("V_pos.txt"), text)?;
            }
            p3.iter().flatten().copied().collect()
        }
        (None, Some(p)) => p.iter().flatten().copied().collect(),
        (None, None) => (0..t).flat_map(|i| [(sess.next_pos + i as i64) as i32; 3]).collect(),
    };
    let max_pos = pos.iter().copied().max().unwrap_or(0) as i64;
    sc.upload_step(sess.len as i32, &ids, &pos)?;
    // The indexer reads the position of every token slot (its group keys rotate at their first token), and the
    // slots outlive the step, so they are kept beside the KV cache.
    if let Some(ixp) = sess.ix_pos.as_mut() {
        ensure!(
            (sess.len + t) * 3 * 4 <= ixp.len(),
            "hy4 indexer position cache too small ({} + {})",
            sess.len,
            t
        );
        ixp.copy_from(sess.len * 12, &sc.pos, 0, t * 12, st)?;
    }
    model.k.get("embed_bf16").launch(
        (t as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&sc.x).into(), (&model.embed).into(), (&sc.ids).into(), c.hidden.into()],
    )?;
    if let Some((start, rows, feat)) = &image {
        let buf = DeviceBuffer::from_slice(feat)?;
        sc.x.copy_from(start * c.hidden * 4, &buf, 0, rows * c.hidden * 4, st)?;
    }
    // The streams are the embedding repeated `hc_mult` times.
    let sd = h.hc_mult * c.hidden;
    let mut streams = crate::session::f32s(t * sd)?;
    model.ew(
        st,
        "hc_expand",
        t * sd,
        &[
            (&streams).into(),
            (&sc.x).into(),
            t.into(),
            h.hc_mult.into(),
            c.hidden.into(),
        ],
    )?;
    let (hidden, hc, eps) = (c.hidden, h.hc_mult, c.hy4.as_ref().unwrap().hc_eps);
    let ihc_part = crate::session::f32s(t * IHC_SLICES * 17)?;
    let layers = std::env::var("MALLEUS_HY4_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(w.layers.len())
        .min(w.layers.len());
    let trace = std::env::var("MALLEUS_HY4_TRACE").is_ok();
    let dump = |name: &str, li: usize, buf: &DeviceBuffer, n: usize| -> Result<()> {
        if !trace {
            return Ok(());
        }
        st.synchronize()?;
        let v: Vec<f32> = buf.download(n)?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let dir = std::env::var("MALLEUS_DUMP_DIR").context("MALLEUS_HY4_TRACE needs MALLEUS_DUMP_DIR")?;
        std::fs::write(std::path::Path::new(&dir).join(format!("T{li}_{name}.f32")), bytes)?;
        Ok(())
    };
    let post = crate::session::f32s(t * hc)?;
    let mut hn = crate::session::f32s(t * hidden)?;
    let mut hn2 = crate::session::f32s(t * hidden)?;
    // The sliding layers are twice as wide: the per-step buffers are sized for the wider of the two shapes.
    let nh_max = h.heads.max(h.swa_heads);
    let base = h.nope.max(h.swa_nope) + h.rope.max(h.swa_rope);
    let vmax = h.v.max(h.swa_v);
    let qd = nh_max * base;
    let kvd = nh_max * (base + vmax - h.rope.max(h.swa_rope));
    let vd = nh_max * vmax;
    let q = crate::session::f32s(t * qd)?;
    let qa = crate::session::f32s(t * h.q_lora)?;
    let kva = crate::session::f32s(t * (h.kv_lora + h.rope))?;
    let gate = crate::session::f32s(t * vd)?;
    let attn_out = crate::session::f32s(t * vd)?;
    let o = crate::session::f32s(t * hidden)?;
    // The absorbed query and the weighted latent of every layer, sized for the wider (sliding) head count: above the
    // buffer pool's size limit, a per-layer allocation was a hipMalloc (~60 ms per layer at 2048 tokens).
    let qt = crate::session::f32s(t * nh_max * h.kv_lora)?;
    let lat_merged = crate::session::f32s(t * nh_max * h.kv_lora)?;
    let _ = (sess.len + t, kvd);
    // The tokens before this call that the Engram n-grams reach (`engram_max_ngram_size - 1` of them).
    // DFly2 drafts from the indexer lists of its target layers and the final hidden state of this call.
    let spec_on = has_draft(model);
    let dfly_layers: Vec<usize> = h.dfly.as_ref().map_or(vec![], |d| d.kv_layers.clone());
    let mut spec_lists: Vec<(usize, Option<DeviceBuffer>)> = Vec::new();
    let hist: Vec<u32> = if sess.len == 0 {
        Vec::new()
    } else {
        sess.ngram_ctx.clone()
    };
    for (li, layer) in w.layers.iter().take(layers).enumerate() {
        let a = &layer.attn;
        // Debug: take the streams from a file (`MALLEUS_HY4_STREAMS_IN`), so the Engram can be compared against
        // the reference on the reference's own input instead of the layer-0 output the engine happens to have.
        let streams_in = match std::env::var("MALLEUS_HY4_STREAMS_IN_DIR") {
            Ok(d) if li > 0 => Some(format!("{d}/L{}.f32", li - 1)),
            _ => std::env::var("MALLEUS_HY4_STREAMS_IN").ok().filter(|_| li == 1),
        };
        if let Some(f) = streams_in {
            let bytes = std::fs::read(&f)?;
            let v: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            ensure!(v.len() >= t * sd, "streams input file too small");
            streams.upload(&v[..t * sd])?;
        }
        if let Some(e) = &layer.engram {
            engram(model, e, &mut streams, &hist, tokens, t, &mut sc)?;
            dump("engram_out", li, &streams, t * sd)?;
        }
        // iHC pre-mix of the mixer, its norm, and the MLA projections.
        ihc_pre(model, &layer.attn_hc, &hn, &post, &streams, &ihc_part, t)?;
        dump("ihc_pre", li, &hn, t * hidden)?;
        dump("ihc_post", li, &post, t * hc)?;
        dump("ihc_out", li, &hn, t * hidden)?;
        model.rmsnorm(st, &hn2, &hn, &layer.in_norm, t, hidden)?;
        dump("in_norm", li, &hn2, t * hidden)?;
        let theta = match c.kind[li] {
            crate::config::MixerKind::Hy4Sliding => h.theta_sliding,
            _ => h.theta_indexed,
        };
        let window = match c.kind[li] {
            crate::config::MixerKind::Hy4Sliding => h.window,
            _ => 0,
        };
        let ring = ring_rows(c, li, sess.max_ctx);
        let bufs = AttnBufs {
            q: &q,
            qa: &qa,
            kva: &kva,
            gate: &gate,
            attn_out: &attn_out,
            o: &o,
            qt: &qt,
            lat_merged: &lat_merged,
        };
        let ix_list = mla_attention(
            model,
            sess,
            &sc,
            a,
            li,
            theta,
            window,
            ring,
            &hn,
            &hn2,
            &bufs,
            t,
            IxSource::Target,
            &dump,
        )?;
        if spec_on && dfly_layers.contains(&li) {
            spec_lists.push((li, ix_list));
        }
        model.k.get("ihc_res").launch(
            ((t * sd).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&streams).into(),
                (&hn).into(),
                (&post).into(),
                (&streams).into(),
                (t * sd).into(),
                hc.into(),
                hidden.into(),
            ],
        )?;
        dump("streams_attn", li, &streams, t * sd)?;
        // FFN: iHC pre-mix, pre-norm, the dense SwiGLU and its sandwich norm, then the residual.
        ihc_pre(model, &layer.ffn_hc, &hn, &post, &streams, &ihc_part, t)?;
        dump("ffn_pre", li, &hn, t * hidden)?;
        dump("ffn_post", li, &post, t * hc)?;
        model.rmsnorm(st, &hn2, &hn, &layer.post_norm, t, hidden)?;
        dump("ffn_in", li, &hn2, t * hidden)?;
        // Debug: take the FFN's input from a file, to check the MoE on the reference's own input.
        if li == 1
            && let Ok(f) = std::env::var("MALLEUS_HY4_FFN_IN")
        {
            let bytes = std::fs::read(&f)?;
            let v: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            ensure!(v.len() >= t * hidden, "ffn input file too small");
            hn2.upload(&v[..t * hidden])?;
        }
        match &layer.ffn {
            Ffn::Dense {
                gate: g,
                up,
                down,
                norm,
            } => {
                model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big0, Act::X(&hn2), g, t, false)?;
                model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big1, Act::Same(&hn2), up, t, false)?;
                let n = t * c.inter;
                model.ew(
                    st,
                    "silu_mul",
                    n,
                    &[(&sc.big0).into(), (&sc.big0).into(), (&sc.big1).into(), n.into()],
                )?;
                model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &o, Act::X(&sc.big0), down, t, false)?;
                model.rmsnorm(st, &hn, &o, norm, t, hidden)?;
            }
            Ffn::Latent(m) => {
                let y = moe(model, m, &hn2, t, &mut sc)?;
                hn.copy_from(0, &y, 0, t * hidden * 4, st)?;
            }
        }
        dump("ffn_out", li, &hn, t * hidden)?;
        model.k.get("ihc_res").launch(
            ((t * sd).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&streams).into(),
                (&hn).into(),
                (&post).into(),
                (&streams).into(),
                (t * sd).into(),
                hc.into(),
                hidden.into(),
            ],
        )?;
        // The layer's state in the reference's layout (`layers/<id>/L<i>.f32`, [t, streams * hidden] f32).
        if let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR") {
            st.synchronize()?;
            let v: Vec<f32> = streams.download(t * sd)?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join(format!("L{li}_t{t}.f32")), bytes)?;
        }
    }
    sess.len += t;
    sess.next_pos = max_pos + 1;
    // The n-gram reach plus a verify's worth, so that a rollback still leaves the n-grams of the kept tokens.
    let keep = h.engram.as_ref().map_or(0, |e| e.max_ngram.saturating_sub(1)) + SPEC_ROWS;
    let all: Vec<u32> = hist.iter().chain(tokens).copied().collect();
    sess.ngram_ctx = all[all.len().saturating_sub(keep)..].to_vec();
    // The head: the streams mixed down, the final norm and the logits.
    let normed = crate::session::f32s(t * hidden)?;
    model.k.get("ihc_head").launch(
        (t as u32, 1, 1),
        (256, 1, 1),
        (256 * 4 + 64 * 4) as u32,
        st,
        &[
            (&normed).into(),
            (&streams).into(),
            (&w.head.f).into(),
            (&w.head.base).into(),
            (&w.head.scale).into(),
            t.into(),
            hc.into(),
            hidden.into(),
            eps.into(),
        ],
    )?;
    model.rmsnorm(st, &hn, &normed, &w.norm, t, hidden)?;
    if spec_on {
        spec_lists.sort_by_key(|(li, _)| dfly_layers.iter().position(|l| l == li));
        keep_spec(model, sess, &hn, &spec_lists, t)?;
    }
    let first = t - n_logits;
    let mut last = crate::session::f32s(n_logits * hidden)?;
    last.copy_from(0, &hn, first * hidden * 4, n_logits * hidden * 4, st)?;
    let logits = crate::session::f32s(n_logits * c.vocab)?;
    model.gemm(
        st,
        &sc.xr,
        &sc.xh,
        &sc.xsc,
        &logits,
        Act::X(&last),
        &model.lm_head,
        n_logits,
        false,
    )?;
    st.synchronize()?;
    *model.scratch_cache.lock().unwrap() = Some(sc);
    Ok(logits.download(n_logits * c.vocab)?)
}

/// What the next DFly2 block needs from the target's last call: the final-norm hidden state and, for each draft
/// layer, the indexer list of the last `rows` rows; `src` is the row of the last token kept (moved back by a
/// rollback).
pub(crate) struct SpecState {
    hidden: DeviceBuffer,
    lists: Vec<Option<(Vec<i32>, usize)>>,
    src: usize,
}

/// How a session's drafts have fared: a moving average of the drafts accepted per round sets the next block's
/// length, and a session whose drafts keep failing stops drafting (one probe every `PROBE` rounds) so that it never
/// decodes slower than without them.
#[derive(Clone, Copy)]
pub(crate) struct DraftStats {
    accepted: f32,
    last: usize,
    rounds: usize,
}

impl Default for DraftStats {
    fn default() -> Self {
        DraftStats {
            accepted: 2.0,
            last: 0,
            rounds: 0,
        }
    }
}

const PROBE: usize = 16;

/// The drafts the next block proposes, at most `max`: none while the average says a verify would not pay.
pub(crate) fn draft_len(sess: &mut crate::session::Session, max: usize) -> usize {
    let s = &mut sess.hy4_draft;
    s.rounds += 1;
    if s.accepted < 0.6 {
        return if s.rounds.is_multiple_of(PROBE) { 2.min(max) } else { 0 };
    }
    ((s.accepted + 1.0).ceil() as usize).clamp(1, max)
}

/// Rows of the last call the speculative state keeps: a verify is at most this wide.
const SPEC_ROWS: usize = crate::weights::GEMV_MAX_T;

/// Drops the last `n` tokens of the preceding call: the caches are positional (rows past `len` are rewritten), so
/// what moves back is the length, the Engram history, the indexer's count of closed groups and the draft's source.
pub(crate) fn rollback(model: &crate::engine::Model, sess: &mut crate::session::Session, n: usize) -> Result<()> {
    let h = model.cfg.hy4.as_ref().context("no hy4 config")?;
    ensure!(n < sess.len && n <= sess.ngram_ctx.len(), "hy4 rollback of {n} tokens");
    sess.len -= n;
    sess.next_pos -= n as i64;
    let keep = sess.ngram_ctx.len() - n;
    sess.ngram_ctx.truncate(keep);
    let closed = sess.len / h.index_pool;
    for p in sess.ix_pooled.iter_mut() {
        *p = (*p).min(closed);
    }
    if let Some(sp) = sess.hy4_spec.as_mut() {
        ensure!(n <= sp.src, "hy4 rollback past the verify");
        sp.src -= n;
    }
    let d = &mut sess.hy4_draft;
    if d.last > 0 {
        let acc = d.last.saturating_sub(n) as f32;
        d.accepted = 0.7 * d.accepted + 0.3 * acc;
        d.last = 0;
    }
    Ok(())
}

/// After a restore to `len`: nothing of the indexer past it counts, and the draft state is the target's to rebuild.
pub(crate) fn restored(model: &crate::engine::Model, sess: &mut crate::session::Session) {
    if let Some(h) = model.cfg.hy4.as_ref() {
        let closed = sess.len / h.index_pool;
        for p in sess.ix_pooled.iter_mut() {
            *p = (*p).min(closed);
        }
    }
    sess.hy4_spec = None;
}

/// Whether the model carries a DFly2 drafter that drafting may use (`MALLEUS_HY4_DFLY=0` turns it off).
pub(crate) fn has_draft(model: &crate::engine::Model) -> bool {
    model.hy4.as_ref().is_some_and(|w| w.dfly.is_some()) && std::env::var("MALLEUS_HY4_DFLY").map_or(true, |v| v != "0")
}

/// Keeps what the next DFly2 block reads from this call: the last rows of the final-norm hidden state and of the
/// draft layers' indexer lists.
fn keep_spec(
    model: &crate::engine::Model,
    sess: &mut crate::session::Session,
    normed: &DeviceBuffer,
    lists: &[(usize, Option<DeviceBuffer>)],
    t: usize,
) -> Result<()> {
    let c = &model.cfg;
    let st = &model.stream;
    let rows = t.min(SPEC_ROWS);
    let first = t - rows;
    let mut hidden = crate::session::f32s(SPEC_ROWS * c.hidden)?;
    if let Some(old) = sess.hy4_spec.take() {
        hidden = old.hidden;
    }
    hidden.copy_from(0, normed, first * c.hidden * 4, rows * c.hidden * 4, st)?;
    let mut out = Vec::with_capacity(lists.len());
    for (_, l) in lists {
        out.push(match l {
            Some(buf) => {
                let slots = buf.len() / (t * 4);
                st.synchronize()?;
                let all: Vec<i32> = buf.download(t * slots)?;
                Some((all[first * slots..].to_vec(), slots))
            }
            None => None,
        });
    }
    sess.hy4_spec = Some(Box::new(SpecState {
        hidden,
        lists: out,
        src: rows - 1,
    }));
    Ok(())
}

/// One DFly2 block after the target's last call: slot 0 is `bonus` (the token the target just produced) fused with
/// the target's hidden state at the last kept row, slots 1..=n the mask token, at the positions that follow. The
/// blocks run once over the whole block on the KV cache of their target layers (the block writes its own keys in
/// the rows the verify then overwrites); then each slot's hidden state gets the correction from the token drafted
/// before it and goes through the shared LM head, greedily.
pub(crate) fn draft(
    model: &crate::engine::Model,
    sess: &mut crate::session::Session,
    bonus: u32,
    n: usize,
) -> Result<Vec<u32>> {
    use crate::engine::Act;
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let w = model.hy4.as_ref().context("no hy4 weights")?;
    let d = w.dfly.as_ref().context("no DFly2 drafter")?;
    let dc = h.dfly.as_ref().context("no DFly2 config")?;
    let st = &model.stream;
    let hd = c.hidden;
    let n = draft_len(sess, n);
    if n == 0 {
        return Ok(vec![]);
    }
    sess.hy4_draft.last = n;
    let wdt = n + 1;
    ensure!(wdt <= SPEC_ROWS + 2, "dfly block of {n} drafts");
    ensure!(sess.len + wdt <= sess.max_ctx, "context full for a draft block");
    let sp = sess.hy4_spec.take().context("dfly: no target state to draft from")?;
    let mut sc = model.take_scratch(wdt.max(2))?;
    let anchor = sess.len;
    let ids: Vec<i32> = std::iter::once(bonus as i32)
        .chain(std::iter::repeat_n(dc.mask as i32, n))
        .collect();
    let pos: Vec<i32> = (0..wdt).flat_map(|k| [(sess.next_pos + k as i64) as i32; 3]).collect();
    sc.upload_step(anchor as i32, &ids, &pos)?;
    let x = crate::session::f32s(wdt * hd)?;
    model.k.get("embed_bf16").launch(
        (wdt as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&x).into(), (&model.embed).into(), (&sc.ids).into(), hd.into()],
    )?;
    // Slot 0: x0 += gate * eh_proj([enorm(e0) | hnorm(h_src)]).
    let mut cat = crate::session::f32s(2 * hd)?;
    let mut tmp = crate::session::f32s(2 * hd)?;
    model.rmsnorm(st, &cat, &x, &d.enorm, 1, hd)?;
    tmp.copy_from(0, &sp.hidden, sp.src * hd * 4, hd * 4, st)?;
    model.rmsnorm(st, &tmp, &tmp, &d.hnorm, 1, hd)?;
    cat.copy_from(hd * 4, &tmp, 0, hd * 4, st)?;
    let fused = crate::session::f32s(hd)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &fused, Act::X(&cat), &d.eh_proj, 1, false)?;
    model.k.get("hy4_fma_vec").launch(
        (hd.div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&x).into(), (&d.fusion_gate).into(), (&fused).into(), hd.into()],
    )?;
    // The blocks.
    let nh = h.heads;
    let qd = nh * (h.nope + h.rope);
    let vd = nh * h.v;
    let (q, qa, kva) = (
        crate::session::f32s(wdt * qd)?,
        crate::session::f32s(wdt * h.q_lora)?,
        crate::session::f32s(wdt * (h.kv_lora + h.rope))?,
    );
    let (gate, attn_out, o) = (
        crate::session::f32s(wdt * vd)?,
        crate::session::f32s(wdt * vd)?,
        crate::session::f32s(wdt * hd)?,
    );
    let (qt, lat_merged) = (
        crate::session::f32s(wdt * nh * h.kv_lora)?,
        crate::session::f32s(wdt * nh * h.kv_lora)?,
    );
    let bufs = AttnBufs {
        q: &q,
        qa: &qa,
        kva: &kva,
        gate: &gate,
        attn_out: &attn_out,
        o: &o,
        qt: &qt,
        lat_merged: &lat_merged,
    };
    let hn = crate::session::f32s(wdt * hd)?;
    let hn2 = crate::session::f32s(wdt * hd)?;
    let no_dump = |_: &str, _: usize, _: &DeviceBuffer, _: usize| -> Result<()> { Ok(()) };
    for (j, l) in d.layers.iter().enumerate() {
        let li = dc.kv_layers[j];
        // The target's selection of the source row, plus the groups the block itself spans: every row of the block
        // walks the same list, and the causal bound of the attention keeps a row to the keys before it.
        let list = match &sp.lists[j] {
            Some((rows, slots)) => {
                let base = &rows[sp.src * slots..][..*slots];
                let mut row: Vec<i32> = base.iter().copied().filter(|&g| g >= 0).collect();
                let pool = h.index_pool;
                for g in anchor / pool..=(anchor + n) / pool {
                    if !row.contains(&(g as i32)) {
                        row.push(g as i32);
                    }
                }
                let width = slots + 4;
                ensure!(row.len() <= width, "dfly list wider than its slots");
                row.resize(width, -1);
                let all: Vec<i32> = (0..wdt).flat_map(|_| row.iter().copied()).collect();
                Some(DeviceBuffer::from_slice(&all)?)
            }
            None => None,
        };
        model.rmsnorm(st, &hn2, &x, &l.in_norm, wdt, hd)?;
        mla_attention(
            model,
            sess,
            &sc,
            &l.attn,
            li,
            h.theta_indexed,
            0,
            0,
            &hn,
            &hn2,
            &bufs,
            wdt,
            IxSource::Given(list),
            &no_dump,
        )?;
        model.k.get("hy4_add").launch(
            ((wdt * hd).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&x).into(), (&hn).into(), (wdt * hd).into()],
        )?;
        model.rmsnorm(st, &hn2, &x, &l.post_norm, wdt, hd)?;
        let g = crate::session::f32s(wdt * l.gate.n)?;
        let u = crate::session::f32s(wdt * l.up.n)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &g, Act::X(&hn2), &l.gate, wdt, false)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &u, Act::Same(&hn2), &l.up, wdt, false)?;
        model.ew(
            st,
            "silu_mul",
            wdt * l.gate.n,
            &[(&g).into(), (&g).into(), (&u).into(), (wdt * l.gate.n).into()],
        )?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &x, Act::X(&g), &l.down, wdt, true)?;
    }
    let fin = crate::session::f32s(wdt * hd)?;
    model.rmsnorm(st, &fin, &x, &d.final_norm, wdt, hd)?;
    // The chained head: slot k corrected with the token drafted at slot k - 1 (the bonus for slot 1).
    let mut out = Vec::with_capacity(n);
    let mut prev = bonus;
    let mut pid = DeviceBuffer::alloc(4)?;
    let emb = crate::session::f32s(hd)?;
    let mut row = crate::session::f32s(hd)?;
    let hg = crate::session::f32s(d.hc_gate.n)?;
    let hu = crate::session::f32s(d.hc_up.n)?;
    let logits = crate::session::f32s(c.vocab)?;
    let best = DeviceBuffer::alloc(4)?;
    for k in 1..=n {
        pid.upload(&[prev as i32])?;
        model.k.get("embed_bf16").launch(
            (1, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&emb).into(), (&model.embed).into(), (&pid).into(), hd.into()],
        )?;
        row.copy_from(0, &fin, k * hd * 4, hd * 4, st)?;
        model.rmsnorm(st, &cat, &row, &d.hc_hidden_norm, 1, hd)?;
        model.rmsnorm(st, &tmp, &emb, &d.hc_embed_norm, 1, hd)?;
        cat.copy_from(hd * 4, &tmp, 0, hd * 4, st)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &hg, Act::X(&cat), &d.hc_gate, 1, false)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &hu, Act::Same(&cat), &d.hc_up, 1, false)?;
        model.ew(
            st,
            "silu_mul",
            d.hc_gate.n,
            &[(&hg).into(), (&hg).into(), (&hu).into(), d.hc_gate.n.into()],
        )?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &row, Act::X(&hg), &d.hc_down, 1, true)?;
        model.gemm(
            st,
            &sc.xr,
            &sc.xh,
            &sc.xsc,
            &logits,
            Act::X(&row),
            &model.lm_head,
            1,
            false,
        )?;
        model.k.get("logits_argmax").launch(
            (1, 1, 1),
            (1024, 1, 1),
            0,
            st,
            &[(&best).into(), (&logits).into(), c.vocab.into()],
        )?;
        st.synchronize()?;
        let id: Vec<i32> = best.download(1)?;
        prev = id[0] as u32;
        out.push(prev);
    }
    *model.scratch_cache.lock().unwrap() = Some(sc);
    sess.hy4_spec = Some(sp);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape the published checkpoint has, cut down to two layers of each kind and two n-gram heads.
    const HY4: &str = r#"{
      "model_type": "hy_v4_flash_text",
      "hidden_size": 256, "num_hidden_layers": 4, "vocab_size": 1024,
      "intermediate_size": 512, "moe_intermediate_size": 256, "rms_norm_eps": 1e-5,
      "num_attention_heads": 2, "q_lora_rank": 128, "kv_lora_rank": 512,
      "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
      "swa_num_attention_heads": 4, "swa_qk_nope_head_dim": 192,
      "swa_qk_rope_head_dim": 64, "swa_v_head_dim": 256, "sliding_window": 512,
      "layer_types": ["deepseek_sparse_attention", "sliding_attention",
                      "sliding_attention", "deepseek_sparse_attention"],
      "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
      "hc_mult": 4, "hc_magnitude": 2.0, "hc_eps": 1e-6, "learnable_sink": true,
      "routed_expert_hidden_size": 256, "n_routed_experts": 4, "num_experts_per_tok": 2,
      "swiglu_limit": 10.0, "routed_scaling_factor": 4.0,
      "compressed_indexer_compress_ratio": 4, "index_topk": 64,
      "index_n_heads": 2, "index_head_dim": 128,
      "engram_layer_ids": [2], "engram_max_ngram_size": 3, "engram_n_head_per_ngram": 2,
      "engram_n_embed_per_ngram": 64, "engram_pad_id": 7, "engram_lookup_table_size": 999,
      "engram_seed": 0, "engram_fp8_embedding": true,
      "engram_layer_head_vocab_sizes": {"2": [[101, 103], [107, 109]]},
      "rope_parameters": {
        "deepseek_sparse_attention": {"rope_theta": 10000000.0, "mrope_section": [12, 10, 10]},
        "sliding_attention": {"rope_theta": 10000.0, "mrope_section": [12, 10, 10]}
      }
    }"#;

    fn parse(json: &str) -> Result<Config> {
        let text: Value = serde_json::from_str(json)?;
        config(&serde_json::json!({ "text_config": text }), None)
    }

    #[test]
    fn geometry_and_engram() {
        let c = parse(HY4).unwrap();
        assert_eq!(c.family, Family::Hy4);
        assert_eq!((c.hidden, c.n_layers, c.vocab), (256, 4, 1024));
        assert_eq!(c.kind[0], MixerKind::Hy4Indexed);
        assert_eq!(c.kind[1], MixerKind::Hy4Sliding);
        assert_eq!(c.full_attn, vec![true, false, false, true]);
        // Both layer types carry the same qk width; the sliding one is twice as wide in v.
        assert_eq!(c.head_dim, 256);
        let h = c.hy4.as_ref().unwrap();
        assert_eq!((h.heads, h.swa_heads), (2, 4));
        assert_eq!((h.theta_indexed, h.theta_sliding), (1e7, 1e4));
        assert_eq!((h.hc_mult, h.hc_magnitude), (4, 2.0));
        assert!(h.sink);
        assert_eq!(h.sparse, vec![false, true, true, true]);
        let e = h.engram.as_ref().unwrap();
        assert_eq!(e.layers, vec![1]);
        assert_eq!(e.head_vocab_sizes[0], vec![vec![101, 103], vec![107, 109]]);
        assert_eq!((e.max_ngram, e.lookup, e.pad), (3, 999, 7));
    }

    #[test]
    fn a_bad_mrope_section_is_rejected() {
        let bad = HY4.replace(r#""mrope_section": [12, 10, 10]}"#, r#""mrope_section": [1, 1, 1]}"#);
        assert!(parse(&bad).is_err());
    }
}
