//! Inkling-2 (`inkling2_loop`, issue #62): the config and the map of the checkpoint's text tensors.
//!
//! The graph is not the Qwen/Ling one. A block of `attn_res_block` layers feeds an *attention residual*: a
//! learned query attends over the running stack of block states, once for the attention input (three times in the
//! full layers, for q, k and v) and once for the MLP input. Every layer input and output passes a short
//! convolution, the full layers carry a relative-position bias with a per-head state, and the middle 12 layers run
//! `loop_passes` times, mixing cross-pass messages through a low-rank controller per stream.
//!
//! This module holds the geometry and the map; the loader and the forward are the next increment.

use anyhow::{Context, Result, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::DeviceBuffer;
use serde_json::Value;

use crate::config::{Config, Family, Inkling2Config, Inkling2Mtp, Inkling2Vision, MixerKind, MoeConfig};
use crate::upload::Uploader;
use crate::weights::Linear;

/// The `text_config` of an Inkling-2 checkpoint, as `Config` plus the Inkling-2 geometry.
pub(crate) fn config(cfg: &Value, generation_config: Option<&str>) -> Result<Config> {
    let t = &cfg["text_config"];
    let u = |k: &str| -> Result<usize> {
        t[k].as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("config: {k}"))
    };
    let f = |k: &str| -> Result<f32> { t[k].as_f64().map(|v| v as f32).with_context(|| format!("config: {k}")) };
    let n_layers = u("num_hidden_layers")?;
    let heads = u("num_attention_heads")?;
    let kv_heads = u("num_key_value_heads")?;
    let head_dim = u("head_dim")?;
    // The sliding layers are the ones `local_layer_ids` names; the full ones carry the relative-position bias.
    let local: Vec<usize> = t["local_layer_ids"]
        .as_array()
        .context("config: local_layer_ids")?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).context("config: local_layer_ids"))
        .collect::<Result<_>>()?;
    let kind: Vec<MixerKind> = (0..n_layers)
        .map(|i| {
            if local.contains(&i) {
                MixerKind::Attn
            } else {
                MixerKind::AttnFull
            }
        })
        .collect();
    ensure!(local.len() <= n_layers, "config: local_layer_ids longer than the stack");
    let loop_cfg = &t["loop_config"];
    let lu = |k: &str| -> Result<usize> {
        loop_cfg[k]
            .as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("config: loop_config.{k}"))
    };
    let vision = match cfg["vision_config"].as_object() {
        Some(v) => {
            let vu = |k: &str| -> Result<usize> {
                v[k].as_u64()
                    .map(|v| v as usize)
                    .with_context(|| format!("config: vision_config.{k}"))
            };
            Some(Inkling2Vision {
                hidden: vu("hidden_size")?,
                inter: vu("hidden_size")? * 4,
                heads: vu("num_attention_heads")?,
                layers: vu("num_hidden_layers")?,
                patch: vu("native_patch_size")?,
                temporal: vu("temporal_patch_size")?,
                merge: vu("adapter_merge_size")?,
                grid: vu("posemb_grid_size")?,
                proj_hidden: vu("projection_hidden_dim")?,
                eps: v["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
                // The image placeholder is the token the processor writes; the hub config does not carry it.
                image_token: cfg["image_token_id"].as_u64().unwrap_or(200054) as u32,
            })
        }
        None => None,
    };
    let mtp = match cfg["mtp_config"].as_object() {
        Some(v) => {
            let mu = |k: &str| -> Result<usize> {
                v[k].as_u64()
                    .map(|v| v as usize)
                    .with_context(|| format!("config: mtp_config.{k}"))
            };
            Some(Inkling2Mtp {
                layers: mu("num_hidden_layers")?,
                inter: mu("intermediate_size")?,
                heads: mu("num_attention_heads")?,
                kv_heads: mu("num_key_value_heads")?,
                head_dim: mu("head_dim")?,
                rope_theta: v["rope_theta"].as_f64().unwrap_or(1e6) as f32,
                eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
                window: mu("sliding_window_size")?,
                local: v["local_layer_ids"]
                    .as_array()
                    .context("config: mtp_config.local_layer_ids")?
                    .iter()
                    .map(|x| x.as_u64().map(|x| x as usize).context("local_layer_ids"))
                    .collect::<Result<_>>()?,
                block: mu("block_size")?,
                markov_rank: mu("markov_rank")?,
                target_layers: v["target_layer_ids"]
                    .as_array()
                    .context("config: mtp_config.target_layer_ids")?
                    .iter()
                    .map(|x| x.as_u64().map(|x| x as usize).context("target_layer_ids"))
                    .collect::<Result<_>>()?,
            })
        }
        None => None,
    };
    let hidden = u("hidden_size")?;
    let inter = u("intermediate_size")?;
    let experts = u("n_routed_experts")?;
    let top_k = u("num_experts_per_tok")?;
    let inkling2 = Inkling2Config {
        heads,
        kv_heads,
        head_dim,
        rel_extent: u("rel_extent")?,
        d_rel: u("d_rel")?,
        log_floor: u("log_scaling_n_floor")?,
        log_alpha: f("log_scaling_alpha")?,
        qk_clip: f("qk_l2_clip_threshold")?,
        rel_clip: f("rel_l2_clip_threshold")?,
        attn_res_block: u("n_layers_per_attn_res_block")?,
        sconv_k: u("sconv_kernel_size")?,
        window: u("sliding_window_size")?,
        experts,
        top_k,
        moe_inter: inter,
        route_scale: f("route_scale")?,
        gate_softcap: f("mlp_gate_softcap")?,
        up_softcap: f("mlp_up_softcap")?,
        logit_softcap: f("final_logit_softcapping")?,
        unpadded_vocab: u("unpadded_vocab_size")?,
        loop_start: lu("core_start")?,
        loop_end: lu("core_end")?,
        // Debug: `MALLEUS_INKLING2_PASSES=1` runs the base graph (prelude, one core pass, tail) for the
        // layer-by-layer comparison against a reference loaded with `loop_num_passes=1`.
        loop_passes: std::env::var("MALLEUS_INKLING2_PASSES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(lu("num_passes")?),
        loop_rank: lu("gate_rank")?,
        vision,
        mtp,
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
    let c = Config {
        hidden,
        n_layers,
        inter,
        vocab: u("vocab_size")?,
        n_heads: heads,
        n_kv: kv_heads,
        head_dim,
        lin_nk: heads,
        lin_nv: heads,
        lin_dk: head_dim,
        lin_dv: head_dim,
        conv_k: u("sconv_kernel_size")?,
        eps: f("rms_norm_eps")?,
        // The sliding and full layers rotate nothing: the relative bias replaces RoPE inside the attention.
        rope_theta: 0.0,
        rot_dim: 0,
        max_positions: 262144,
        mrope_section: [0, 0, 0],
        full_attn: kind.iter().map(|k| *k == MixerKind::AttnFull).collect(),
        eos,
        moe: Some(MoeConfig {
            experts,
            top_k,
            inter,
            sigmoid: t["scoring_func"].as_str() == Some("sigmoid"),
            n_group: 1,
            topk_group: 1,
            bias: t["use_gate_bias"].as_bool().unwrap_or(false),
            scaling: f("route_scale")?,
            shared_gate: false,
        }),
        hc: None,
        ple: None,
        indexer: None,
        gate_sigmoid: false,
        family: Family::Inkling2,
        kind,
        mla: None,
        kda_lower_bound: 0.0,
        norm_one_plus_w: false,
        mrope: false,
        dense_ffn: 0,
        swiglu: None,
        hy4: None,
        inkling2: Some(inkling2),
        deepseek: None,
    };
    c.validate()?;
    Ok(c)
}

/// The geometry the kernels will require of an Inkling-2 checkpoint.
pub(crate) fn validate(c: &Config) -> Result<()> {
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    ensure!(
        c.n_layers == c.kind.len() && c.n_layers == c.full_attn.len(),
        "layer kind count"
    );
    ensure!(
        i.heads == c.n_heads && i.kv_heads == c.n_kv && i.head_dim == c.head_dim,
        "head geometry"
    );
    ensure!(c.n_heads.is_multiple_of(c.n_kv), "bad head grouping");
    ensure!(
        i.attn_res_block > 0 && c.n_layers.is_multiple_of(i.attn_res_block),
        "attention-residual blocks"
    );
    ensure!(i.sconv_k == 4, "the short convolution is kernel 4");
    ensure!(
        i.loop_start > 0 && i.loop_start < i.loop_end && i.loop_end <= c.n_layers,
        "the loop range must be a non-empty part of the stack"
    );
    ensure!(i.loop_passes >= 1, "a loop needs at least one pass");
    ensure!(
        c.hidden.is_multiple_of(128) && i.moe_inter.is_multiple_of(128),
        "the 4-bit rotation needs widths in 128s"
    );
    ensure!(i.top_k <= i.experts, "more experts per token than experts");
    Ok(())
}

/// One tensor the forward will read: its name in the file, what it holds, and its shape.
pub struct Expected {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// Every text tensor of the graph, in load order. The experts arrive as the fused `gate_up_proj`/`down_proj` the
/// converter writes, at 4 bits; everything else is what the checkpoint stores.
pub fn text_map(c: &Config) -> Result<Vec<Expected>> {
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let mut v = Vec::new();
    let mut push = |name: String, dtype: DType, shape: Vec<usize>| v.push(Expected { name, dtype, shape });
    let (h, qd, kvd) = (c.hidden, i.heads * i.head_dim, i.kv_heads * i.head_dim);
    push("model.llm.embed.weight".into(), DType::BF16, vec![c.vocab, h]);
    push("model.llm.embed_norm.weight".into(), DType::BF16, vec![h]);
    push("model.llm.norm.weight".into(), DType::BF16, vec![h]);
    push("model.llm.unembed.weight".into(), DType::Mls8, vec![c.vocab, h]);
    push("model.llm.unembed.weight.scales".into(), DType::F32, vec![c.vocab]);
    push("model.llm.attn_res.output_query".into(), DType::F32, vec![h]);
    // One attention query per layer after the first; the full layers take three (q, k, v).
    for l in 1..c.n_layers {
        let full = c.kind[l] == MixerKind::AttnFull;
        let shape = if full { vec![3, h] } else { vec![h] };
        push(format!("model.llm.attn_res.attn_queries.{}", l - 1), DType::F32, shape);
    }
    for l in 0..c.n_layers {
        push(format!("model.llm.attn_res.mlp_queries.{l}"), DType::F32, vec![h]);
    }
    let core = i.loop_end - i.loop_start;
    for name in ["middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"] {
        push(format!("model.llm.{name}"), DType::F32, vec![2 * core, h]);
    }
    for l in 0..c.n_layers {
        let p = |s: &str| format!("model.llm.layers.{l}.{s}");
        let full = c.kind[l] == MixerKind::AttnFull;
        push(p("mlp_norm.weight"), DType::BF16, vec![h]);
        push(p("attn_sconv.weight"), DType::BF16, vec![h, 1, i.sconv_k]);
        push(p("mlp_sconv.weight"), DType::BF16, vec![h, 1, i.sconv_k]);
        if full {
            for n in ["q", "k", "v"] {
                push(p(&format!("{n}_input_norm.weight")), DType::BF16, vec![h]);
            }
            for n in ["k", "v"] {
                push(
                    p(&format!("attn.{n}_sconv.static_w")),
                    DType::BF16,
                    vec![i.sconv_k, kvd],
                );
                push(
                    p(&format!("attn.{n}_sconv.weight_proj")),
                    DType::BF16,
                    vec![i.kv_heads * i.sconv_k, h],
                );
            }
        } else {
            push(p("attn_norm.weight"), DType::BF16, vec![h]);
            for n in ["k", "v"] {
                push(
                    p(&format!("attn.{n}_sconv.weight")),
                    DType::BF16,
                    vec![kvd, 1, i.sconv_k],
                );
            }
        }
        for (n, dtype, shape) in [
            ("attn.wq_du.weight", DType::Mls4, vec![qd, h]),
            ("attn.wk_dv.weight", DType::Mls4, vec![kvd, h]),
            ("attn.wv_dv.weight", DType::Mls4, vec![kvd, h]),
            ("attn.wo_ud.weight", DType::Mls4, vec![h, qd]),
            ("attn.wr_du.weight", DType::Mls4, vec![i.heads * i.d_rel, h]),
            ("attn.gate_attn.weight", DType::Mls4, vec![qd, h]),
        ] {
            let k = *shape.last().unwrap();
            let scales = vec![shape[0], k / 128];
            push(p(n), dtype, shape);
            push(p(&format!("{n}.scales")), DType::F16, scales);
        }
        push(p("attn.rel_logits_proj.proj"), DType::BF16, vec![i.d_rel, i.rel_extent]);
        push(p("attn.attn_sink_proj.weight"), DType::BF16, vec![i.heads, h]);
        push(p("attn.q_l2_clip.scale"), DType::BF16, vec![i.heads, i.head_dim]);
        push(p("attn.rel_l2_clip.scale"), DType::BF16, vec![i.heads, i.d_rel]);
        push(p("mlp.gate.weight"), DType::BF16, vec![i.experts, h]);
        push(p("mlp.gate.bias"), DType::F32, vec![i.experts]);
        push(
            p("mlp.experts.gate_up_proj"),
            DType::Mls4,
            vec![i.experts, 2 * i.moe_inter, h],
        );
        push(
            p("mlp.experts.gate_up_proj.scales"),
            DType::F16,
            vec![i.experts, 2 * i.moe_inter, h / 128],
        );
        push(p("mlp.experts.down_proj"), DType::Mls4, vec![i.experts, h, i.moe_inter]);
        push(
            p("mlp.experts.down_proj.scales"),
            DType::F16,
            vec![i.experts, h, i.moe_inter / 128],
        );
        // The loop's cross-pass controller: one low-rank gate per stream, in the layers it repeats.
        if (i.loop_start..i.loop_end).contains(&l) {
            for stream in ["attn", "k", "v", "mlp"] {
                push(
                    p(&format!("controllers.{stream}.down_weight")),
                    DType::BF16,
                    vec![i.loop_rank, 2 * h],
                );
                push(
                    p(&format!("controllers.{stream}.forget_weight")),
                    DType::BF16,
                    vec![h, i.loop_rank],
                );
                push(
                    p(&format!("controllers.{stream}.gain_weight")),
                    DType::BF16,
                    vec![h, i.loop_rank],
                );
            }
        }
    }
    Ok(v)
}

/// The whole file: the text graph, the vision tower and the MTP head. The order is the order the loader reads.
pub fn map(c: &Config) -> Result<Vec<Expected>> {
    let mut v = text_map(c)?;
    v.extend(vision_map(c)?);
    v.extend(mtp_map(c)?);
    Ok(v)
}

/// Checks the file against the map: every tensor the forward reads is there, with the shape and the dtype it
/// expects, and nothing in the file is left unclaimed.
pub fn check(src: &dyn TensorSource, c: &Config) -> Result<Vec<String>> {
    let expected = map(c)?;
    let mut lines = Vec::new();
    // A map entry says the *narrowest* form a linear may take: the file may carry a wider copy of the same tensor
    // (`--int8-attn`, `--keep-int8-layers`, and the MLS8 copies `--prefill-int8` writes), and then the weight is MLS8
    // and its scale is one f32 per row instead of one f16 per 128 weights. The loader reads what it finds
    // (`Uploader::linear`), so the check takes the pair the file actually holds.
    let wide: Vec<&str> = src
        .names()
        .into_iter()
        .filter(|n| n.ends_with(".weight") && src.info(n).is_some_and(|i| i.dtype == DType::Mls8))
        .filter_map(|n| n.strip_suffix(".weight"))
        .collect();
    let mut claimed: Vec<&str> = Vec::with_capacity(expected.len());
    for e in &expected {
        let info = src
            .info(&e.name)
            .with_context(|| format!("{}: not in the file", e.name))?;
        // `.scales` hangs off the weight's name, so both have to reduce to the tensor's own name.
        let base = e
            .name
            .strip_suffix(".scales")
            .or_else(|| e.name.strip_suffix(".weight"))
            .map(|b| b.strip_suffix(".weight").unwrap_or(b));
        if base.is_some_and(|b| wide.contains(&b)) {
            // The weight in its 8-bit form, one f32 scale a row.
            let ok = if e.name.ends_with(".weight") {
                info.dtype == DType::Mls8 && info.shape == e.shape
            } else {
                info.dtype == DType::F32 && info.shape == [e.shape[0]]
            };
            ensure!(
                ok,
                "{}: the file has {:?} {:?} for an 8-bit copy",
                e.name,
                info.dtype,
                info.shape
            );
            claimed.push(&e.name);
            continue;
        }
        ensure!(
            info.dtype == e.dtype && info.shape == e.shape,
            "{}: expected {:?} {:?}, the file has {:?} {:?}",
            e.name,
            e.dtype,
            e.shape,
            info.dtype,
            info.shape
        );
        claimed.push(&e.name);
    }
    let mut unclaimed: Vec<&str> = src
        .names()
        .into_iter()
        .filter(|n| !claimed.iter().any(|c| c == n))
        .collect();
    unclaimed.sort_unstable();
    lines.push(format!(
        "{} of {} tensors mapped ({} text, {} vision, {} mtp)",
        claimed.len(),
        claimed.len() + unclaimed.len(),
        text_map(c)?.len(),
        vision_map(c)?.len(),
        mtp_map(c)?.len()
    ));
    if !unclaimed.is_empty() {
        lines.push(format!(
            "{} unclaimed, first: {:?}",
            unclaimed.len(),
            &unclaimed[..unclaimed.len().min(8)]
        ));
    }
    // What the device will hold: the stored bytes of each part, which is what the plan has to fit.
    let bytes = |v: &[Expected]| -> f64 {
        v.iter()
            .map(|e| e.dtype.size_of_elems(e.shape.iter().product::<usize>()) as f64)
            .sum::<f64>()
            / 1e9
    };
    lines.push(format!(
        "footprint: {:.2} GB text, {:.2} GB vision, {:.2} GB mtp",
        bytes(&text_map(c)?),
        bytes(&vision_map(c)?),
        bytes(&mtp_map(c)?)
    ));
    lines.push(format!(
        "loop {}..{} x{}, attention-residual blocks of {}, window {}, {} experts of {} active",
        c.inkling2.as_ref().unwrap().loop_start,
        c.inkling2.as_ref().unwrap().loop_end,
        c.inkling2.as_ref().unwrap().loop_passes,
        c.inkling2.as_ref().unwrap().attn_res_block,
        c.inkling2.as_ref().unwrap().window,
        c.inkling2.as_ref().unwrap().top_k,
        c.inkling2.as_ref().unwrap().experts
    ));
    if let Some(m) = &c.inkling2.as_ref().unwrap().mtp {
        lines.push(format!(
            "mtp: {} layers, block {}, markov rank {}",
            m.layers, m.block, m.markov_rank
        ));
    }
    Ok(lines)
}

/// The text graph, the vision tower and the MTP head, as the buffers the forward will read.
/// The vision tower as the buffers its graph reads: 23 pre-norm blocks at 1024 wide, 16 heads of 64, a 4096-wide MLP
/// with exact GELU, and the 2x2 merger projecting 4096 lanes to the text width.
pub struct Vision {
    pub proj_in: DeviceBuffer,
    pub proj_out: DeviceBuffer,
    pub ln_pre: (DeviceBuffer, DeviceBuffer),
    pub class_embedding: DeviceBuffer,
    pub cls_position: DeviceBuffer,
    pub patch: DeviceBuffer,
    pub position: DeviceBuffer,
    /// The same table on the host: `grid^2 x hidden` f32, resampled to the patch grid of every image.
    pub pos_f32: Vec<f32>,
    pub layers: Vec<VisLayer>,
    /// Patch side (the patch embedding covers one frame), temporal patch, merge side and the side of the learned
    /// position grid.
    pub patch_side: usize,
    pub temporal: usize,
    pub merge: usize,
    pub grid: usize,
    /// Token id of an image placeholder in the prompt: the processor expands the one the template writes into
    /// `rows * columns` copies, and the forward replaces their rows with the merger's features.
    pub image_token: u32,
}

pub struct VisLayer {
    pub ln_1: (DeviceBuffer, DeviceBuffer),
    pub ln_2: (DeviceBuffer, DeviceBuffer),
    pub ls_1: DeviceBuffer,
    pub ls_2: DeviceBuffer,
    pub fc: (DeviceBuffer, DeviceBuffer),
    pub proj: (DeviceBuffer, DeviceBuffer),
    pub q: (DeviceBuffer, DeviceBuffer),
    pub k: (DeviceBuffer, DeviceBuffer),
    pub v: (DeviceBuffer, DeviceBuffer),
    pub out: (DeviceBuffer, DeviceBuffer),
}

impl Vision {
    fn load(src: &dyn TensorSource, c: &Config, up: &mut Uploader) -> Result<Option<Self>> {
        let v = match c.inkling2.as_ref().and_then(|i| i.vision.as_ref()) {
            Some(v) => v,
            None => return Ok(None),
        };
        let p = |s: &str| format!("model.visual.{s}");
        let b = |up: &mut Uploader, s: &str, shape: &[usize]| up.tensor(src, &p(s), shape);
        let mut layers = Vec::with_capacity(v.layers);
        for l in 0..v.layers {
            let q = |s: &str| format!("backbone.layers.{l}.{s}");
            let bb = |up: &mut Uploader, s: &str, shape: &[usize]| up.tensor(src, &p(&q(s)), shape);
            layers.push(VisLayer {
                ln_1: (bb(up, "ln_1.weight", &[v.hidden])?, bb(up, "ln_1.bias", &[v.hidden])?),
                ln_2: (bb(up, "ln_2.weight", &[v.hidden])?, bb(up, "ln_2.bias", &[v.hidden])?),
                ls_1: {
                    let b = bb(up, "ls_1.gamma", &[v.hidden])?;
                    if l == 0 && std::env::var_os("MALLEUS_INKLING2_VISION_DUMP").is_some() {
                        let raw = src.read_all(&p(&q("ls_1.gamma")))?;
                        let v: Vec<f32> = raw
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .take(4)
                            .map(|b| half::bf16::from_le_bytes(*b).to_f32())
                            .collect();
                        eprintln!("ls_1[0..4] = {v:?} ({} bytes)", raw.len());
                    }
                    b
                },
                ls_2: bb(up, "ls_2.gamma", &[v.hidden])?,
                fc: (
                    bb(up, "mlp.c_fc.weight", &[v.inter, v.hidden])?,
                    bb(up, "mlp.c_fc.bias", &[v.inter])?,
                ),
                proj: (
                    bb(up, "mlp.c_proj.weight", &[v.hidden, v.inter])?,
                    bb(up, "mlp.c_proj.bias", &[v.hidden])?,
                ),
                q: (
                    bb(up, "self_attn.q_proj.weight", &[v.hidden, v.hidden])?,
                    bb(up, "self_attn.q_proj.bias", &[v.hidden])?,
                ),
                k: (
                    bb(up, "self_attn.k_proj.weight", &[v.hidden, v.hidden])?,
                    bb(up, "self_attn.k_proj.bias", &[v.hidden])?,
                ),
                v: (
                    bb(up, "self_attn.v_proj.weight", &[v.hidden, v.hidden])?,
                    bb(up, "self_attn.v_proj.bias", &[v.hidden])?,
                ),
                out: (
                    bb(up, "self_attn.out_proj.weight", &[v.hidden, v.hidden])?,
                    bb(up, "self_attn.out_proj.bias", &[v.hidden])?,
                ),
            });
        }
        Ok(Some(Vision {
            proj_in: b(up, "proj_in.weight", &[v.proj_hidden, v.proj_hidden])?,
            proj_out: b(up, "proj_out.weight", &[c.hidden, v.proj_hidden])?,
            ln_pre: (
                b(up, "backbone.ln_pre.weight", &[v.hidden])?,
                b(up, "backbone.ln_pre.bias", &[v.hidden])?,
            ),
            class_embedding: b(up, "backbone.embeddings.class_embedding", &[v.hidden])?,
            cls_position: b(up, "backbone.embeddings.cls_position_embedding", &[v.hidden])?,
            patch: b(
                up,
                "backbone.embeddings.patch_embedding.weight",
                &[v.hidden, 3 * v.patch * v.patch],
            )?,
            pos_f32: {
                let raw = src
                    .read_all(&p("backbone.embeddings.position_embedding"))
                    .context("vision position table")?;
                let n = v.grid * v.grid * v.hidden;
                ensure!(raw.len() == n * 2, "position table: expected {n} bf16 values");
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| half::bf16::from_le_bytes(*b).to_f32())
                    .collect()
            },
            position: b(
                up,
                "backbone.embeddings.position_embedding",
                &[v.hidden, v.grid * v.grid],
            )?,
            layers,
            patch_side: v.patch,
            temporal: v.temporal,
            merge: v.merge,
            grid: v.grid,
            image_token: v.image_token,
        }))
    }
}

impl Vision {
    /// The tower's forward (issue #74): the patch embedding, the bilinear position table, one class token in front of
    /// the patches, 23 pre-norm blocks with a per-channel layer scale, then the two-layer projector.
    ///
    /// Returns the sequence the tower produced (the class token included, which is what `vit_out` of the reference is)
    /// and the merged features, one row per image token, in the order the prompt's image tokens are in.
    pub fn encode(&self, model: &Model, pre: &crate::image_pre::Preprocessed) -> Result<(DeviceBuffer, DeviceBuffer)> {
        let st = &model.stream;
        let k = &model.k;
        let cs = model.cfg.inkling2.as_ref().context("no inkling2 config")?;
        let v = cs.vision.as_ref().context("no vision config")?;
        let (h, heads, inter, m, ps) = (v.hidden, v.heads, v.inter, self.merge, self.patch_side);
        let hd = h / heads;
        ensure!(pre.grid_t == 1, "video input is not supported");
        let (gh, gw) = (pre.grid_h, pre.grid_w);
        let (ph, pw) = (gh * m, gw * m);
        let (np, rows) = (ph * pw, ph * pw + 1);
        let dim = ps * ps * 3;
        ensure!(
            pre.pixels.len() == gh * gw * m * m * dim,
            "pixel buffer {} does not match a {gh}x{gw} grid of {m}x{m} patches of {ps}",
            pre.pixels.len()
        );
        let f = |n: usize| f32s(n);
        // 1. Tiles to native patches, then the patch embedding (the checkpoint's has no bias).
        let tiles = DeviceBuffer::from_slice(&pre.pixels)?;
        let patches = f(np * dim)?;
        k.get("inkling2_vision_split").launch(
            ((np * dim).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&patches).into(),
                (&tiles).into(),
                np.into(),
                (ps * m).into(),
                ps.into(),
                m.into(),
            ],
        )?;
        let x = f(np * h)?;
        gemm_bf16(k, st, &x, &patches, &self.patch, np, h, dim)?;
        vdump("split", &patches, np * dim)?;
        vdump("patch_emb", &x, np * h)?;
        // 2. The learned position table, resampled to this patch grid, then one class token in front.
        let pos = DeviceBuffer::from_slice(&self.pos_embed(ph, pw, h))?;
        vdump("pos", &pos, np * h)?;
        k.get("add_inplace").launch(
            ((np * h).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&x).into(), (&pos).into(), (np * h).into()],
        )?;
        let seq = f(rows * h)?;
        seq.memset(0, st)?;
        bias_act(k, st, &seq, &self.class_embedding, 1, h, 0)?;
        bias_act(k, st, &seq, &self.cls_position, 1, h, 0)?;
        seq.copy_from_shared(h * 4, &x, 0, np * h * 4, st)?;
        vdump("emb", &seq, rows * h)?;
        // 3. The RoPE positions: the class token sits at (0, 0), the patches at one-based (row, col). The patches are
        // in tile-major order (`merge x merge` per tile), so their positions are too: row-major here would rotate
        // every patch of a tile with the position of its neighbour.
        let mut hw = Vec::with_capacity(rows * 2);
        hw.extend_from_slice(&[0i32, 0]);
        for tr in 0..gh {
            for tc in 0..gw {
                for sy in 0..m {
                    for sx in 0..m {
                        hw.extend_from_slice(&[(tr * m + sy) as i32 + 1, (tc * m + sx) as i32 + 1]);
                    }
                }
            }
        }
        let hw = DeviceBuffer::from_slice(&hw)?;
        let normed = f(rows * h)?;
        layernorm(k, st, &normed, &seq, &self.ln_pre, rows, h, v.eps)?;
        // The tower replaces the sequence with its pre-layernormed form: that is what the blocks carry as the
        // residual, not the embedding.
        seq.copy_from_shared(0, &normed, 0, rows * h * 4, st)?;
        vdump("lnpre", &seq, rows * h)?;
        let (q, kk, vv) = (f(rows * h)?, f(rows * h)?, f(rows * h)?);
        let (att, mlp) = (f(rows * h)?, f(rows * inter)?);
        let scale = 1.0 / (hd as f32).sqrt();
        for (bi, b) in self.layers.iter().enumerate() {
            layernorm(k, st, &normed, &seq, &b.ln_1, rows, h, v.eps)?;
            if bi == 0 {
                vdump("lnb", &normed, rows * h)?;
            }
            for (out, w) in [(&q, &b.q), (&kk, &b.k), (&vv, &b.v)] {
                gemm_bf16(k, st, out, &normed, &w.0, rows, h, h)?;
                bias_act(k, st, out, &w.1, rows, h, 0)?;
            }
            for t in [&q, &kk] {
                k.get("inkling2_vision_rope").launch(
                    (rows as u32, heads as u32, 1),
                    ((hd as u32).next_power_of_two().clamp(32, 128), 1, 1),
                    0,
                    st,
                    &[
                        t.into(),
                        (&hw).into(),
                        rows.into(),
                        heads.into(),
                        hd.into(),
                        10000f32.into(),
                    ],
                )?;
            }
            if bi == 0 {
                vdump("qrope", &q, rows * h)?;
                vdump("krope", &kk, rows * h)?;
            }
            k.get("inkling2_vision_attn").launch(
                (rows as u32, heads as u32, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&att).into(),
                    (&q).into(),
                    (&kk).into(),
                    (&vv).into(),
                    rows.into(),
                    heads.into(),
                    hd.into(),
                    scale.into(),
                ],
            )?;
            gemm_bf16(k, st, &normed, &att, &b.out.0, rows, h, h)?;
            bias_act(k, st, &normed, &b.out.1, rows, h, 0)?;
            if bi == 0 {
                vdump("attn_out", &normed, rows * h)?;
                vdump("v", &vv, rows * h)?;
            }
            k.get("inkling2_vision_scale_add").launch(
                ((rows * h).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&seq).into(),
                    (&normed).into(),
                    (&b.ls_1).into(),
                    (rows * h).into(),
                    h.into(),
                ],
            )?;
            layernorm(k, st, &normed, &seq, &b.ln_2, rows, h, v.eps)?;
            if bi == 0 {
                vdump("lna", &normed, rows * h)?;
            }
            gemm_bf16(k, st, &mlp, &normed, &b.fc.0, rows, inter, h)?;
            bias_act(k, st, &mlp, &b.fc.1, rows, inter, 2)?;
            gemm_bf16(k, st, &normed, &mlp, &b.proj.0, rows, h, inter)?;
            bias_act(k, st, &normed, &b.proj.1, rows, h, 0)?;
            if bi == 0 {
                vdump("mlp_out", &normed, rows * h)?;
                vdump("fc1", &mlp, rows * inter)?;
            }
            k.get("inkling2_vision_scale_add").launch(
                ((rows * h).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&seq).into(),
                    (&normed).into(),
                    (&b.ls_2).into(),
                    (rows * h).into(),
                    h.into(),
                ],
            )?;
            vdump(&format!("l{bi}"), &seq, rows * h)?;
        }
        // 4. The projector: the patch rows only (the class token is dropped), `m * m` of them per output row.
        let nm = np / (m * m);
        let mid = f(nm * v.proj_hidden)?;
        let patch_rows = Arg::U64(seq.addr() + (h * 4) as u64);
        k.get("gemm_bf16").launch(
            (v.proj_hidden.div_ceil(8) as u32, nm.div_ceil(8) as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&mid).into(),
                patch_rows,
                (&self.proj_in).into(),
                nm.into(),
                v.proj_hidden.into(),
                (m * m * h).into(),
                0i32.into(),
            ],
        )?;
        // The projector's activation sits between its two layers (the checkpoint ships no biases here, so the gelu
        // rides in through a zero one).
        let zero = DeviceBuffer::zeroed(v.proj_hidden * 2)?;
        bias_act(k, st, &mid, &zero, nm, v.proj_hidden, 2)?;
        let out = f(nm * model.cfg.hidden)?;
        gemm_bf16(k, st, &out, &mid, &self.proj_out, nm, model.cfg.hidden, v.proj_hidden)?;
        vdump("vit", &seq, rows * h)?;
        vdump("merger", &out, nm * model.cfg.hidden)?;
        Ok((seq, out))
    }

    /// The learned `grid x grid` position table resampled to a `ph x pw` patch grid, `bilinear`, `align_corners=False`,
    /// border padding — the tower's grid is sampled per patch, and the patches are already in tile-major order, so the
    /// table's rows are walked in that order too.
    fn pos_embed(&self, ph: usize, pw: usize, h: usize) -> Vec<f32> {
        let (side, m) = (self.grid, self.merge);
        let taps = |i: usize, n: usize| -> [(usize, f32); 2] {
            let src = if n > 1 {
                (i as f32 + 0.5) * (side as f32) / (n as f32) - 0.5
            } else {
                0.0
            };
            // The two taps are `floor` and `floor + 1` clamped to the table, weights `1 - frac` and `frac`: when the
            // grid is wider than the table (`n > side`) `floor` goes negative and both taps land on row 0, which is
            // what the reference does too — clamping the first tap before stepping to the second would not.
            let f = src.floor();
            let fr = (src - f).clamp(0.0, 1.0);
            let tap = |k: f32| (k.max(0.0) as usize).min(side - 1);
            [(tap(f), 1.0 - fr), (tap(f + 1.0), fr)]
        };
        let table = &self.pos_f32;
        let mut out = Vec::with_capacity(ph * pw * h);
        let mut acc = vec![0f32; h];
        for tr in 0..ph / m {
            for tc in 0..pw / m {
                for sy in 0..m {
                    for sx in 0..m {
                        let (r, c) = (tr * m + sy, tc * m + sx);
                        acc.iter_mut().for_each(|v| *v = 0.0);
                        for (ty, wy) in taps(r, ph) {
                            for (tx, wx) in taps(c, pw) {
                                let wgt = wy * wx;
                                for (i, v) in acc.iter_mut().enumerate() {
                                    *v += wgt * table[(ty * side + tx) * h + i];
                                }
                            }
                        }
                        out.extend_from_slice(&acc);
                    }
                }
            }
        }
        out
    }
}

/// `MALLEUS_INKLING2_VISION_DUMP=<dir>`: write the tower's stage outputs there, for a comparison against the
/// reference's hooks. Debug only: it synchronizes the stream and copies to the host.
fn vdump(name: &str, buf: &DeviceBuffer, n: usize) -> Result<()> {
    let Ok(dir) = std::env::var("MALLEUS_INKLING2_VISION_DUMP") else {
        return Ok(());
    };
    let v = buf.download::<f32>(n)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x: &f32| x.to_le_bytes()).collect();
    std::fs::write(format!("{dir}/eng_{name}.f32"), bytes)?;
    Ok(())
}

/// `y = x @ w^T` with a bf16 weight ([n, k]) and no bias: the tower's own GEMM, which the `gemm_bf16` kernel does.
fn gemm_bf16(
    k: &Kernels,
    st: &Stream,
    y: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    t: usize,
    n: usize,
    kk: usize,
) -> Result<()> {
    k.get("gemm_bf16").launch(
        (n.div_ceil(8) as u32, t.div_ceil(8) as u32, 1),
        (256, 1, 1),
        0,
        st,
        &[y.into(), x.into(), w.into(), t.into(), n.into(), kk.into(), 0i32.into()],
    )?;
    Ok(())
}

/// `y = x + bias` (bf16, broadcast per row) and optionally an activation: 0 none, 2 the exact `gelu`.
fn bias_act(k: &Kernels, st: &Stream, y: &DeviceBuffer, b: &DeviceBuffer, t: usize, n: usize, act: i32) -> Result<()> {
    k.get("bias_act").launch(
        ((t * n).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[y.into(), b.into(), t.into(), n.into(), act.into()],
    )?;
    Ok(())
}

fn layernorm(
    k: &Kernels,
    st: &Stream,
    y: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &(DeviceBuffer, DeviceBuffer),
    rows: usize,
    n: usize,
    eps: f32,
) -> Result<()> {
    k.get("layernorm").launch(
        (rows as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[y.into(), x.into(), (&w.0).into(), (&w.1).into(), n.into(), eps.into()],
    )?;
    Ok(())
}

/// The MTP head: three draft layers at the text width, the fuse of the four target hidden states, and the Markov
/// correction that shapes the proposed logits. The names are the file's; the semantics are the next increment.
pub struct Mtp {
    pub fuse: DeviceBuffer,
    /// `softmax(fuse_weight)` per row, on the host: one row per draft stage and the last one for the Markov head.
    /// The fuse is 4x4, so its softmax is arithmetic on sixteen numbers, not a kernel.
    pub alpha: Vec<f32>,
    pub mask_embed: DeviceBuffer,
    pub norm: DeviceBuffer,
    pub layers: Vec<MtpLayer>,
    pub markov: Markov,
}

pub struct MtpLayer {
    pub dflare_norm: DeviceBuffer,
    pub input_norm: DeviceBuffer,
    pub post_attention_norm: DeviceBuffer,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub k_ctx: DeviceBuffer,
    pub v_ctx: DeviceBuffer,
    pub q_norm: DeviceBuffer,
    pub k_norm: DeviceBuffer,
    pub k_norm_ctx: DeviceBuffer,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

pub struct Markov {
    pub dflare_norm: DeviceBuffer,
    pub hidden_out: (DeviceBuffer, DeviceBuffer),
    pub w1: DeviceBuffer,
    pub rnn_proj: (DeviceBuffer, DeviceBuffer),
}

impl Mtp {
    fn load(src: &dyn TensorSource, c: &Config, up: &mut Uploader) -> Result<Option<Self>> {
        let m = match c.inkling2.as_ref().and_then(|i| i.mtp.as_ref()) {
            Some(m) => m,
            None => return Ok(None),
        };
        let h = c.hidden;
        let mut layers = Vec::with_capacity(m.layers);
        for l in 0..m.layers {
            let p = |s: &str| format!("layers.{l}.{s}");
            let t = |up: &mut Uploader, s: &str, shape: &[usize]| up.tensor(src, &p(s), shape);
            layers.push(MtpLayer {
                dflare_norm: t(up, "dflare_norm.weight", &[h])?,
                input_norm: t(up, "input_layernorm.weight", &[h])?,
                post_attention_norm: t(up, "post_attention_layernorm.weight", &[h])?,
                q: up.linear(src, &p("self_attn.q_proj"), m.heads * m.head_dim, h)?,
                k: up.linear(src, &p("self_attn.k_proj"), m.kv_heads * m.head_dim, h)?,
                v: up.linear(src, &p("self_attn.v_proj"), m.kv_heads * m.head_dim, h)?,
                o: up.linear(src, &p("self_attn.o_proj"), h, m.heads * m.head_dim)?,
                k_ctx: t(up, "self_attn.k_proj_ctx.weight", &[m.kv_heads * m.head_dim, h])?,
                v_ctx: t(up, "self_attn.v_proj_ctx.weight", &[m.kv_heads * m.head_dim, h])?,
                q_norm: t(up, "self_attn.q_norm.weight", &[m.head_dim])?,
                k_norm: t(up, "self_attn.k_norm.weight", &[m.head_dim])?,
                k_norm_ctx: t(up, "self_attn.k_norm_ctx.weight", &[m.head_dim])?,
                gate: up.linear(src, &p("mlp.gate_proj"), m.inter, h)?,
                up: up.linear(src, &p("mlp.up_proj"), m.inter, h)?,
                down: up.linear(src, &p("mlp.down_proj"), h, m.inter)?,
            });
        }
        let fuse = up.tensor(src, "fuse_weight", &[4, 4])?;
        let raw = fuse.download::<u16>(16)?;
        let mut alpha = vec![0f32; 16];
        for r in 0..4 {
            let row: Vec<f32> = (0..4).map(|j| half::bf16::from_bits(raw[r * 4 + j]).to_f32()).collect();
            let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let e: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let sum: f32 = e.iter().sum();
            for j in 0..4 {
                alpha[r * 4 + j] = e[j] / sum;
            }
        }
        Ok(Some(Mtp {
            fuse,
            alpha,
            mask_embed: up.tensor(src, "mask_embed", &[h])?,
            norm: up.tensor(src, "norm.weight", &[h])?,
            layers,
            markov: Markov {
                dflare_norm: up.tensor(src, "markov_head.dflare_norm.weight", &[h])?,
                hidden_out: (
                    up.tensor(src, "markov_head.hidden_out.weight", &[h, m.markov_rank])?,
                    up.tensor(src, "markov_head.hidden_out.bias", &[h])?,
                ),
                w1: up.tensor(src, "markov_head.markov_w1.weight", &[c.vocab, m.markov_rank])?,
                rnn_proj: (
                    up.tensor(src, "markov_head.rnn_proj.weight", &[3 * m.markov_rank, 5120])?,
                    up.tensor(src, "markov_head.rnn_proj.bias", &[3 * m.markov_rank])?,
                ),
            },
        }))
    }
}

pub struct Weights {
    pub embed: DeviceBuffer,
    pub embed_norm: DeviceBuffer,
    pub norm: DeviceBuffer,
    pub unembed: Linear,
    pub output_query: DeviceBuffer,
    /// One per layer after the first: `[hidden]`, or `[3, hidden]` in the full layers (q, k, v).
    pub attn_queries: Vec<DeviceBuffer>,
    pub mlp_queries: Vec<DeviceBuffer>,
    pub cross_pass_q: DeviceBuffer,
    pub cross_pass_kv: DeviceBuffer,
    pub layers: Vec<Layer>,
    pub vision: Option<Vision>,
    pub mtp: Option<Mtp>,
    pub bytes: usize,
}

pub struct Layer {
    pub full: bool,
    pub mlp_norm: DeviceBuffer,
    pub attn_norm: Option<DeviceBuffer>,
    pub attn_sconv: DeviceBuffer,
    pub mlp_sconv: DeviceBuffer,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub wr: Linear,
    pub gate: Linear,
    pub rel_logits: DeviceBuffer,
    pub sink: DeviceBuffer,
    pub q_clip: DeviceBuffer,
    pub rel_clip: DeviceBuffer,
    pub sconv: Sconv,
    pub moe: Moe,
    pub controller: Option<Controller>,
}

/// The short convolution of the keys and values. The sliding layers convolve with a fixed depthwise kernel; the full
/// ones add per-token taps projected from the hidden state to a fixed kernel, one tap set per key/value head.
pub enum Sconv {
    Local {
        k: DeviceBuffer,
        v: DeviceBuffer,
    },
    Dynamic {
        k_static: DeviceBuffer,
        k_proj: DeviceBuffer,
        v_static: DeviceBuffer,
        v_proj: DeviceBuffer,
        q_norm: DeviceBuffer,
        k_norm: DeviceBuffer,
        v_norm: DeviceBuffer,
    },
}

pub struct Moe {
    pub gate: DeviceBuffer,
    pub bias: DeviceBuffer,
    /// MLS4 `[experts, 2 * inter, hidden]` and its group scales.
    pub gate_up: (DeviceBuffer, DeviceBuffer),
    /// MLS4 `[experts, hidden, inter]` and its group scales.
    pub down: (DeviceBuffer, DeviceBuffer),
}

pub struct Gate {
    pub down: DeviceBuffer,
    pub forget: DeviceBuffer,
    pub gain: DeviceBuffer,
}

/// The loop's cross-pass controller: one low-rank gate per stream.
pub struct Controller {
    pub attn: Gate,
    pub k: Gate,
    pub v: Gate,
    pub mlp: Gate,
}

/// A small bf16 tensor read back as f32: the kernels that consume the clip scales and the relative-position
/// table read them as `const float*`.
fn bf16_as_f32(src: &dyn TensorSource, name: &str, n: usize) -> Result<DeviceBuffer> {
    let raw = src.read_all(name).with_context(|| format!("missing tensor {name}"))?;
    ensure!(raw.len() == n * 2, "{name}: expected {n} bf16 values");
    let v: Vec<f32> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::bf16::from_le_bytes(*b).to_f32())
        .collect();
    Ok(DeviceBuffer::from_slice(&v)?)
}

impl Weights {
    pub fn load(src: &dyn TensorSource, c: &Config, progress: &dyn Fn(usize, usize)) -> Result<Self> {
        let i = c.inkling2.as_ref().context("no inkling2 config")?;
        let mut up = Uploader::new()?;
        let p = |s: &str| format!("model.llm.{s}");
        let (h, qd, kvd) = (c.hidden, i.heads * i.head_dim, i.kv_heads * i.head_dim);
        progress(0, c.n_layers);
        let embed = up.tensor(src, &p("embed.weight"), &[c.vocab, h])?;
        let embed_norm = up.tensor(src, &p("embed_norm.weight"), &[h])?;
        let norm = up.tensor(src, &p("norm.weight"), &[h])?;
        let unembed = up.linear(src, &p("unembed"), c.vocab, h)?;
        let output_query = up.typed(src, &p("attn_res.output_query"), DType::F32, &[h])?;
        let mut attn_queries = Vec::with_capacity(c.n_layers - 1);
        for l in 1..c.n_layers {
            let full = c.kind[l] == MixerKind::AttnFull;
            let shape = if full { vec![3, h] } else { vec![h] };
            attn_queries.push(up.typed(src, &p(&format!("attn_res.attn_queries.{}", l - 1)), DType::F32, &shape)?);
        }
        let mut mlp_queries = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            mlp_queries.push(up.typed(src, &p(&format!("attn_res.mlp_queries.{l}")), DType::F32, &[h])?);
        }
        let cross = 2 * (i.loop_end - i.loop_start);
        let cross_pass_q = up.typed(src, &p("middle_cycle_cross_pass_queries"), DType::F32, &[cross, h])?;
        let cross_pass_kv = up.typed(src, &p("middle_cycle_cross_pass_kv_queries"), DType::F32, &[cross, h])?;
        let mut layers = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            progress(l, c.n_layers);
            let p = |s: &str| format!("model.llm.layers.{l}.{s}");
            let full = c.kind[l] == MixerKind::AttnFull;
            let attn_norm = if full {
                None
            } else {
                Some(up.tensor(src, &p("attn_norm.weight"), &[h])?)
            };
            let sconv = if full {
                Sconv::Dynamic {
                    k_static: up.tensor(src, &p("attn.k_sconv.static_w"), &[i.sconv_k, kvd])?,
                    k_proj: up.tensor(src, &p("attn.k_sconv.weight_proj"), &[i.kv_heads * i.sconv_k, h])?,
                    v_static: up.tensor(src, &p("attn.v_sconv.static_w"), &[i.sconv_k, kvd])?,
                    v_proj: up.tensor(src, &p("attn.v_sconv.weight_proj"), &[i.kv_heads * i.sconv_k, h])?,
                    q_norm: up.tensor(src, &p("q_input_norm.weight"), &[h])?,
                    k_norm: up.tensor(src, &p("k_input_norm.weight"), &[h])?,
                    v_norm: up.tensor(src, &p("v_input_norm.weight"), &[h])?,
                }
            } else {
                Sconv::Local {
                    k: up.tensor(src, &p("attn.k_sconv.weight"), &[kvd, 1, i.sconv_k])?,
                    v: up.tensor(src, &p("attn.v_sconv.weight"), &[kvd, 1, i.sconv_k])?,
                }
            };
            let stack = |up: &mut Uploader, name: &str, n: usize, k: usize| -> Result<(DeviceBuffer, DeviceBuffer)> {
                let g = k / malleus_gguf::quant::BLOCK;
                let mut q = DeviceBuffer::alloc(i.experts * n * k / 2)?;
                let mut sc = DeviceBuffer::alloc(i.experts * n * g * 2)?;
                up.into(src, &p(name), DType::Mls4, &[i.experts, n, k], &mut q, 0)?;
                up.into(
                    src,
                    &p(&format!("{name}.scales")),
                    DType::F16,
                    &[i.experts, n, g],
                    &mut sc,
                    0,
                )?;
                Ok((q, sc))
            };
            let controller = if (i.loop_start..i.loop_end).contains(&l) {
                let gate = |up: &mut Uploader, stream: &str| -> Result<Gate> {
                    Ok(Gate {
                        down: up.tensor(
                            src,
                            &p(&format!("controllers.{stream}.down_weight")),
                            &[i.loop_rank, 2 * h],
                        )?,
                        forget: up.tensor(
                            src,
                            &p(&format!("controllers.{stream}.forget_weight")),
                            &[h, i.loop_rank],
                        )?,
                        gain: up.tensor(src, &p(&format!("controllers.{stream}.gain_weight")), &[h, i.loop_rank])?,
                    })
                };
                Some(Controller {
                    attn: gate(&mut up, "attn")?,
                    k: gate(&mut up, "k")?,
                    v: gate(&mut up, "v")?,
                    mlp: gate(&mut up, "mlp")?,
                })
            } else {
                None
            };
            layers.push(Layer {
                full,
                mlp_norm: up.tensor(src, &p("mlp_norm.weight"), &[h])?,
                attn_norm,
                attn_sconv: up.tensor(src, &p("attn_sconv.weight"), &[h, 1, i.sconv_k])?,
                mlp_sconv: up.tensor(src, &p("mlp_sconv.weight"), &[h, 1, i.sconv_k])?,
                wq: up.linear(src, &p("attn.wq_du"), qd, h)?,
                wk: up.linear(src, &p("attn.wk_dv"), kvd, h)?,
                wv: up.linear(src, &p("attn.wv_dv"), kvd, h)?,
                wo: up.linear(src, &p("attn.wo_ud"), h, qd)?,
                wr: up.linear(src, &p("attn.wr_du"), i.heads * i.d_rel, h)?,
                gate: up.linear(src, &p("attn.gate_attn"), qd, h)?,
                rel_logits: bf16_as_f32(src, &p("attn.rel_logits_proj.proj"), i.d_rel * i.rel_extent)?,
                sink: up.tensor(src, &p("attn.attn_sink_proj.weight"), &[i.heads, h])?,
                q_clip: bf16_as_f32(src, &p("attn.q_l2_clip.scale"), i.heads * i.head_dim)?,
                rel_clip: bf16_as_f32(src, &p("attn.rel_l2_clip.scale"), i.heads * i.d_rel)?,
                sconv,
                moe: Moe {
                    gate: up.tensor(src, &p("mlp.gate.weight"), &[i.experts, h])?,
                    bias: up.typed(src, &p("mlp.gate.bias"), DType::F32, &[i.experts])?,
                    gate_up: stack(&mut up, "mlp.experts.gate_up_proj", 2 * i.moe_inter, h)?,
                    down: stack(&mut up, "mlp.experts.down_proj", h, i.moe_inter)?,
                },
                controller,
            });
        }
        progress(c.n_layers, c.n_layers);
        let vision = Vision::load(src, c, &mut up)?;
        let mtp = Mtp::load(src, c, &mut up)?;
        Ok(Weights {
            embed,
            embed_norm,
            norm,
            unembed,
            output_query,
            attn_queries,
            mlp_queries,
            cross_pass_q,
            cross_pass_kv,
            layers,
            vision,
            mtp,
            bytes: up.bytes,
        })
    }
}

/// The vision tower: 23 pre-norm ViT blocks at 1024 wide, 16 heads, 4x MLP, native patches of 20x20 at two frames,
/// merged 2x2 by the adapter, then a 4096-wide projection in and out. Stored at 16 bits, like the checkpoint.
pub fn vision_map(c: &Config) -> Result<Vec<Expected>> {
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let v = match &i.vision {
        Some(v) => v,
        None => return Ok(Vec::new()),
    };
    let mut out = Vec::new();
    let mut push = |name: String, shape: Vec<usize>| {
        out.push(Expected {
            name,
            dtype: DType::BF16,
            shape,
        })
    };
    let h = v.hidden;
    push("model.visual.proj_in.weight".into(), vec![v.proj_hidden, v.proj_hidden]);
    push("model.visual.proj_out.weight".into(), vec![c.hidden, v.proj_hidden]);
    for n in ["weight", "bias"] {
        push(format!("model.visual.backbone.ln_pre.{n}"), vec![h]);
    }
    let e = "model.visual.backbone.embeddings";
    push(format!("{e}.class_embedding"), vec![h]);
    push(format!("{e}.cls_position_embedding"), vec![h]);
    push(format!("{e}.patch_embedding.weight"), vec![h, 3 * v.patch * v.patch]);
    push(format!("{e}.position_embedding"), vec![h, v.grid * v.grid]);
    for l in 0..v.layers {
        let p = |s: &str| format!("model.visual.backbone.layers.{l}.{s}");
        for n in ["ln_1", "ln_2"] {
            for t in ["weight", "bias"] {
                push(p(&format!("{n}.{t}")), vec![h]);
            }
        }
        for n in ["ls_1", "ls_2"] {
            push(p(&format!("{n}.gamma")), vec![h]);
        }
        push(p("mlp.c_fc.weight"), vec![v.inter, h]);
        push(p("mlp.c_fc.bias"), vec![v.inter]);
        push(p("mlp.c_proj.weight"), vec![h, v.inter]);
        push(p("mlp.c_proj.bias"), vec![h]);
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            push(p(&format!("self_attn.{n}.weight")), vec![h, h]);
            push(p(&format!("self_attn.{n}.bias")), vec![h]);
        }
    }
    Ok(out)
}

/// The MTP head: 3 layers at the text width, one sliding and two full, plus the Markov correction of the logits.
pub fn mtp_map(c: &Config) -> Result<Vec<Expected>> {
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let m = match &i.mtp {
        Some(m) => m,
        None => return Ok(Vec::new()),
    };
    let mut out = Vec::new();
    let mut push = |name: String, dtype: DType, shape: Vec<usize>| out.push(Expected { name, dtype, shape });
    let h = c.hidden;
    let qd = m.heads * m.head_dim;
    let kvd = m.kv_heads * m.head_dim;
    push("fuse_weight".into(), DType::BF16, vec![4, 4]);
    push("mask_embed".into(), DType::BF16, vec![h]);
    push("norm.weight".into(), DType::BF16, vec![h]);
    for l in 0..m.layers {
        let p = |s: &str| format!("layers.{l}.{s}");
        for n in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] {
            push(p(&format!("{n}.weight")), DType::BF16, vec![h]);
        }
        for (n, shape) in [
            ("mlp.gate_proj.weight", vec![m.inter, h]),
            ("mlp.up_proj.weight", vec![m.inter, h]),
            ("mlp.down_proj.weight", vec![h, m.inter]),
            ("self_attn.q_proj.weight", vec![qd, h]),
            ("self_attn.k_proj.weight", vec![kvd, h]),
            ("self_attn.v_proj.weight", vec![kvd, h]),
            ("self_attn.o_proj.weight", vec![h, qd]),
        ] {
            let k = *shape.last().unwrap();
            let scales = vec![shape[0], k / 128];
            push(p(n), DType::Mls4, shape);
            push(p(&format!("{n}.scales")), DType::F16, scales);
        }
        for n in ["k_proj_ctx", "v_proj_ctx"] {
            push(p(&format!("self_attn.{n}.weight")), DType::BF16, vec![kvd, h]);
        }
        for n in ["q_norm", "k_norm", "k_norm_ctx"] {
            push(p(&format!("self_attn.{n}.weight")), DType::BF16, vec![m.head_dim]);
        }
    }
    let mk = "markov_head";
    push(format!("{mk}.dflare_norm.weight"), DType::BF16, vec![h]);
    push(format!("{mk}.hidden_out.weight"), DType::BF16, vec![h, m.markov_rank]);
    push(format!("{mk}.hidden_out.bias"), DType::BF16, vec![h]);
    push(
        format!("{mk}.markov_w1.weight"),
        DType::BF16,
        vec![c.vocab, m.markov_rank],
    );
    push(
        format!("{mk}.rnn_proj.weight"),
        DType::BF16,
        vec![3 * m.markov_rank, 5120],
    );
    push(format!("{mk}.rnn_proj.bias"), DType::BF16, vec![3 * m.markov_rank]);
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------
// The text forward: the 24 layers, the loop's extra passes, then the head. The whole prompt runs as one graph with
// `T` rows (`Graph`); one token at a time runs the same layer code with `T = 1` against `Cache`, which carries the
// keys and values of every slot and the three-row histories of the four short convolutions. The prefill writes the
// cache too, so decode starts where the prompt ended.
// ---------------------------------------------------------------------------------------------------------------

use crate::engine::Model;
use crate::kernels::Kernels;
use malleus_hip::{Arg, Stream};

fn f32s(n: usize) -> Result<DeviceBuffer> {
    Ok(DeviceBuffer::alloc(n * 4)?)
}

fn argmax(v: &[f32], n: usize) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().take(n).enumerate() {
        if x > bv {
            bv = x;
            bi = i as u32;
        }
    }
    bi
}

/// One KV stream's state across tokens: the keys and the values it has produced, and the three-row input histories
/// of its short convolutions (`sconv` reads them for its taps, the decode path shifts them in place).
///
/// The release runs the 12 core layers three times, and every pass has its own attention: 24 physical layers plus
/// two extra passes over the core (`n_layers + (passes - 1) * core`), 48 slots for Inkling-2-Mini. Keys and values
/// are fp16, so at 8 KV heads x 128 that is 4 KiB per token per slot: 34.5 KiB a position over the 8 slots of the
/// full layers, which are the ones that keep the whole context.
pub struct Slot {
    /// [`rows`, kv_heads * head_dim] int8, the keys after the convolution, the clip and the L2 norm, with one fp32
    /// scale per (row, KV head) in `ks`: half the bytes of the fp16 cache, and eight bits of mantissa instead of the
    /// four an e4m3 code carries (issue #106, item 2). The convolution histories below stay fp32 — they are inputs,
    /// not the cache.
    k: DeviceBuffer,
    v: DeviceBuffer,
    /// [`rows`, kv_heads] fp32, the scale of each head's 128 values, in the row's own (ring-wrapped) index.
    ks: DeviceBuffer,
    vs: DeviceBuffer,
    /// Rows of `k`/`v`, and the ring they are indexed with: a sliding layer only ever reads the last `window` keys,
    /// so its cache is a ring of `slot_shape` rows and position `p` lives at `p % ring`. A full-attention layer has no
    /// ring (0) and position `p` is row `p`, with `rows == max`.
    rows: usize,
    ring: usize,
    /// [3, kv_heads * head_dim] the convolution input of the last three positions, oldest first.
    past_k: DeviceBuffer,
    past_v: DeviceBuffer,
    /// [3, hidden] the same for the attention output and the MLP output.
    past_y: DeviceBuffer,
    past_moe: DeviceBuffer,
}

/// The session state of the decode path: one [`Slot`] per KV stream, the bias row of the current query, and the
/// log-scaled position factor of the current position (`tau`, 1 below the floor).
pub struct Cache {
    max: usize,
    len: usize,
    slots: Vec<Slot>,
    /// [heads, max_ctx] the relative-bias row of the decode query (the prefill computes it inside the attention).
    bias: DeviceBuffer,
    tau: DeviceBuffer,
    /// The target states the MTP head reads (`mtp_config.target_layer_ids`): the state entering those visits and the
    /// final normalized one, for the rows of the step. `[4, FEAT_ROWS, hidden]`, see `feat_rows`.
    feats: DeviceBuffer,
    /// How many of the `FEAT_ROWS` rows of `feats` this step filled (the last rows of the step).
    feat_rows: usize,
    /// The draft head's context and keys: `None` when the model has no MTP head or `MALLEUS_MTP` is off.
    draft: Option<Draft>,
    /// The convolution histories after every row of the last multi-row forward, so a verify can be rolled back.
    snap: Option<Snap>,
}

/// The MTP head's own state: one K/V row per committed token, per draft layer, fp16 like the target's cache, and the
/// zeroed relative-bias tile the decode attention demands (this head has no bias, no sink and no output gate, so the
/// sink and gate pointers are null and the bias row is read from a zero buffer).
///
/// Rows are positions `base_pos .. base_pos + len`: a prefill chunk that carries more rows than `FEAT_ROWS` can only
/// keep the features of its tail, and then the context restarts there instead of being wrong (`reset`).
pub struct Draft {
    k: Vec<DeviceBuffer>,
    v: Vec<DeviceBuffer>,
    /// [`rows`, kv_heads] fp32, one scale per (row, KV head), as the target's cache.
    ks: Vec<DeviceBuffer>,
    vs: Vec<DeviceBuffer>,
    rows: usize,
    len: usize,
    base_pos: usize,
    /// The zeroed bias tile: allocated and kept with the head (the decode attention takes the pointer), but no
    /// current kernel reads it, so the lint sees it as dead.
    #[expect(dead_code, reason = "zero bias tile: kept for the transport, no kernel reads it yet")]
    bias: DeviceBuffer,
    /// `unembed @ hidden_out` as `[rank, vocab]` bf16 and `unembed @ hidden_out.bias` as `[vocab]` f32, built at the
    /// first draft (118 GFLOP once, 103 MB): the folded Markov bias, read once per draft row.
    fold: Option<(DeviceBuffer, DeviceBuffer)>,
    /// The walk's carrier and its latent state, `[rank]`.
    state: DeviceBuffer,
    latent: DeviceBuffer,
}

/// The state a verify has to put back: `hist[slot][stream]` is the three-row history after every row of the step,
/// `[t, 3, C]`, with the streams in the order the layer writes them (keys, values, attention output, MLP output).
struct Snap {
    rows: usize,
    hist: Vec<[DeviceBuffer; 4]>,
}

/// Rows of target state the MTP head keeps: a verify is at most `GEMV_MAX_T` tokens, plus the bonus one.
pub const DRAFT_ROWS: usize = crate::weights::GEMV_MAX_T + 1;

/// Rows of target state one forward captures: a prefill chunk, so that the draft's context covers the prompt it was
/// fed instead of only its last seven tokens. `[4, FEAT_ROWS, hidden]` f32 is 19 MB at 512.
pub const FEAT_ROWS: usize = PREFILL_CHUNK;

/// Tokens a prefill chunk of this model carries: the scheduler feeds the prompt in pieces of this size, and the
/// sliding cache's ring is sized for one (`window + PREFILL_CHUNK + DRAFT_ROWS + 1`).
pub const PREFILL_CHUNK: usize = 512;

/// Slots the cache of a session holds: one per physical layer, plus one per core layer for each extra pass of the
/// loop (`loop_passes`).
pub fn n_slots(i: &Inkling2Config, n_layers: usize) -> usize {
    n_layers + (i.loop_passes.max(1) - 1) * (i.loop_end - i.loop_start)
}

/// The physical layer a slot's keys belong to: the inverse of `slot_of`, because the extra slots are the core layers
/// of the loop's later passes.
fn slot_layer(i: &Inkling2Config, n_layers: usize, s: usize) -> usize {
    if s < n_layers {
        s
    } else {
        i.loop_start + (s - n_layers) % (i.loop_end - i.loop_start)
    }
}

/// Rows and ring of one slot: `(rows, 0)` for a full-attention layer, which keeps the whole context, and a ring
/// otherwise. A sliding layer reads at most its window back, so the rest of the cache would never be read again; the
/// ring holds a whole prefill chunk on top of the window, because a chunk writes all its keys before it attends and a
/// smaller ring would clobber keys the chunk's own first rows still see.
fn slot_shape(i: &Inkling2Config, full: bool, ctx: usize) -> (usize, usize) {
    if full {
        return (ctx, 0);
    }
    let rows = (i.window + PREFILL_CHUNK + DRAFT_ROWS + 1).min(ctx);
    (rows, if rows == ctx { 0 } else { rows })
}

/// Bytes the cache of a session of `ctx` tokens asks the device for: the int8 keys and values of every slot with
/// their scales, the four three-row convolution histories, the bias row of the decode query, `tau` and the MTP rows.
///
/// This is `Cache::new` priced, not an estimate: both go through `slot_shape`, so the pre-flight and the allocator
/// cannot drift (`memplan::session_bytes` returns this for the family).
pub fn session_bytes(cfg: &Config, ctx: usize) -> u64 {
    let Some(i) = cfg.inkling2.as_ref() else {
        return 0;
    };
    let kvd = (i.kv_heads * i.head_dim) as u64;
    // The histories are fp32: inputs of the short convolutions, not the cache. Keys and values are int8 with one
    // fp32 scale per (row, KV head), which the SSA/DS buffer sizing below mirrors.
    let hist = 2 * (3 * kvd * 4) + 2 * (3 * cfg.hidden as u64 * 4);
    let mut t = 0u64;
    for s in 0..n_slots(i, cfg.n_layers) {
        let (rows, _) = slot_shape(i, cfg.kind[slot_layer(i, cfg.n_layers, s)] == MixerKind::AttnFull, ctx);
        t += rows as u64 * kvd * 2 + rows as u64 * i.kv_heads as u64 * 8 + hist;
    }
    t += (i.heads * i.rel_extent * 4) as u64;
    t += 4;
    let feat_rows = if draft_enabled() { FEAT_ROWS } else { DRAFT_ROWS };
    t += (4 * feat_rows * cfg.hidden * 4) as u64;
    // The draft head's context: one K/V row per committed token per draft layer, plus the block's own rows.
    if let Some(m) = i.mtp.as_ref() {
        let d = (m.kv_heads * m.head_dim) as u64;
        t += m.layers as u64 * 2 * (ctx as u64 + DRAFT_ROWS as u64) * d;
        t += m.layers as u64 * 2 * (ctx as u64 + DRAFT_ROWS as u64) * m.kv_heads as u64 * 4;
        t += (m.heads * (ctx + DRAFT_ROWS) * 4) as u64;
    }
    t
}

impl Cache {
    pub fn new(model: &Model, max_ctx: usize) -> Result<Self> {
        let c = &model.cfg;
        let i = c.inkling2.as_ref().context("no inkling2 config")?;
        let kvd = i.kv_heads * i.head_dim;
        let mut out = Vec::with_capacity(n_slots(i, c.n_layers));
        for s in 0..n_slots(i, c.n_layers) {
            let full = c.kind[slot_layer(i, c.n_layers, s)] == MixerKind::AttnFull;
            let (rows, ring) = slot_shape(i, full, max_ctx);
            out.push(Slot {
                k: DeviceBuffer::alloc(rows * kvd)?,
                v: DeviceBuffer::alloc(rows * kvd)?,
                ks: f32s(rows * i.kv_heads)?,
                vs: f32s(rows * i.kv_heads)?,
                rows,
                ring,
                // Zeroed: the taps before the first token are the reference's left padding.
                past_k: DeviceBuffer::zeroed(3 * kvd * 4)?,
                past_v: DeviceBuffer::zeroed(3 * kvd * 4)?,
                past_y: DeviceBuffer::zeroed(3 * c.hidden * 4)?,
                past_moe: DeviceBuffer::zeroed(3 * c.hidden * 4)?,
            });
        }
        let draft = match i.mtp.as_ref().filter(|_| draft_enabled()) {
            Some(m) => {
                let d = m.kv_heads * m.head_dim;
                let rows = max_ctx + DRAFT_ROWS;
                let mut k = Vec::with_capacity(m.layers);
                let mut v = Vec::with_capacity(m.layers);
                let mut ks = Vec::with_capacity(m.layers);
                let mut vs = Vec::with_capacity(m.layers);
                for _ in 0..m.layers {
                    // int8 codes with one fp32 scale a head: the same cache as the target's, half the bytes.
                    k.push(DeviceBuffer::alloc(rows * d)?);
                    v.push(DeviceBuffer::alloc(rows * d)?);
                    ks.push(f32s(rows * m.kv_heads)?);
                    vs.push(f32s(rows * m.kv_heads)?);
                }
                Some(Draft {
                    k,
                    v,
                    ks,
                    vs,
                    rows,
                    len: 0,
                    base_pos: 0,
                    bias: DeviceBuffer::zeroed(m.heads * rows * 4)?,
                    fold: None,
                    state: f32s(m.markov_rank)?,
                    latent: f32s(m.markov_rank)?,
                })
            }
            None => None,
        };
        Ok(Self {
            max: max_ctx,
            len: 0,
            slots: out,
            // The relative-bias tile is [heads, extent]: it does not grow with the context (issue #158).
            bias: f32s(i.heads * i.rel_extent)?,
            tau: f32s(1)?,
            feats: f32s(4 * if draft.is_some() { FEAT_ROWS } else { DRAFT_ROWS } * c.hidden)?,
            feat_rows: 0,
            draft,
            snap: None,
        })
    }

    /// Back to an empty context: no position, no key, no history (zeros are the reference's left padding).
    pub fn reset(&mut self) -> Result<()> {
        let st = Stream::null();
        for s in &self.slots {
            for b in [&s.past_k, &s.past_v, &s.past_y, &s.past_moe] {
                b.memset(0, &st)?;
            }
        }
        st.synchronize()?;
        self.len = 0;
        if let Some(d) = self.draft.as_mut() {
            d.len = 0;
            d.base_pos = 0;
        }
        self.snap = None;
        Ok(())
    }

    /// Keeps the three-row history of every row of a `t`-row forward, so that a verify can be rolled back to any of
    /// its rows (`rollback`). Allocated once per session and reused while `t` does not grow.
    fn snapshot(&mut self, t: usize) -> Result<()> {
        if self.snap.as_ref().is_some_and(|s| s.rows >= t) {
            return Ok(());
        }
        let mut hist = Vec::with_capacity(self.slots.len());
        for s in &self.slots {
            // `[3, C]` f32: C is the width of the stream the history belongs to.
            let kvd = s.past_k.len() / 12;
            let hd = s.past_y.len() / 12;
            hist.push([
                DeviceBuffer::zeroed(t * 3 * kvd * 4)?,
                DeviceBuffer::zeroed(t * 3 * kvd * 4)?,
                DeviceBuffer::zeroed(t * 3 * hd * 4)?,
                DeviceBuffer::zeroed(t * 3 * hd * 4)?,
            ]);
        }
        self.snap = Some(Snap { rows: t, hist });
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Inkling-2's session: the cache of its own graph and **none** of the generic per-layer state.
///
/// `Model::new_session` allocates `kv`, `conv`, `rec`, `attn_part` and the MTP buffers for every stack it carries,
/// and nothing in the looped graph reads them: the session keeps its keys in `Cache`. That was 20.5 GiB at 256K of
/// state the model never touched (20 sliding layers x 2 x 128 MiB at 65536 = 80 KiB a position on top of the 34.5 the
/// full slots account for), the state the pre-flight could not see (issue #110). The 4-byte `attn_part` placeholder
/// is what `crate::hy4::new_session` leaves for the same reason: the field is not optional and the batching paths that
/// read it are the ones this family does not take (`Model::can_batch` is false for a looped stack).
pub(crate) fn new_session(model: &crate::engine::Model, max_ctx: usize) -> Result<crate::session::Session> {
    let asked = malleus_hip::alloc_ledger_bytes;
    let before = asked();
    let cache = Cache::new(model, max_ctx)?;
    if before > 0 {
        // `MALLEUS_ALLOC_LEDGER=1`: the pre-flight sizes this session with `session_bytes`, and this is the number to
        // check it against. The `attn_part` placeholder below comes after it.
        let got = asked() - before;
        let planned = session_bytes(&model.cfg, max_ctx);
        if got != planned {
            eprintln!("inkling2: the session of {max_ctx} tokens asked {got} bytes, the plan says {planned}");
        }
    }
    Ok(crate::session::Session {
        max_ctx,
        rope: None,
        rope_factor: 1.0,
        mscale: 1.0,
        len: 0,
        next_pos: 0,
        kv: Vec::new(),
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
        idx_sel_part: None,
        idx_sel_cache: Vec::new(),
        idx_cache_len: 0,
        mtp_idx: None,
        idx_list: None,
        ple_snap: None,
        ngram_prev: Vec::new(),
        inkling2: Some(Box::new(cache)),
        hy4_spec: None,
        hy4_draft: Default::default(),
        ix_tk: Vec::new(),
        ix_gk: Vec::new(),
        ix_pooled: Vec::new(),
        ix_pos: None,
    })
}

/// The slot a layer's attention writes to: the physical layer in the first pass, then one of the extra passes'
/// slots after the 24 physical ones.
fn slot_of(i: &Inkling2Config, n_layers: usize, pass: usize, l: usize) -> usize {
    if pass == 0 || l < i.loop_start || l >= i.loop_end {
        return l;
    }
    n_layers + (pass - 1) * (i.loop_end - i.loop_start) + (l - i.loop_start)
}

/// Where a layer's keys, values and history go, and (on the one-token path) the query row of the relative bias and
/// the position factor. Taken per layer, so the borrow of the session's cache lasts one layer.
struct Place<'a> {
    slot: &'a mut Slot,
    /// The four per-row history snapshots of this slot, or `None` outside a multi-row forward: a verify keeps the
    /// history of every row so that a rollback of its tail can put one back.
    snap: Option<&'a [DeviceBuffer; 4]>,
    bias: &'a DeviceBuffer,
    tau: &'a DeviceBuffer,
    /// Position of the first row of this call.
    pos: usize,
    /// One token: the decode kernels (sconv with the history shift, attention over the cache) run instead of the
    /// whole-sequence ones.
    one: bool,
}

/// Keeps the last three rows of a convolution's input in `past` ([3, C], oldest first), so the next call convolves
/// across the boundary. `pos` is the position of the first row of `x`; rows before the sequence stay zero.
fn keep_last3(
    model: &Model,
    st: &Stream,
    past: &DeviceBuffer,
    x: &DeviceBuffer,
    pos: usize,
    t: usize,
    c: usize,
    snap: Option<&DeviceBuffer>,
) -> Result<()> {
    // A multi-row step also keeps the history of *every* row it saw: a verify that only commits a prefix of its rows
    // has to put the history of the last committed row back, which no end-of-step buffer can answer.
    if let Some(sn) = snap.filter(|_| t > 1) {
        model.k.get("inkling2_hist_snap").launch(
            ((t * c).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[sn.into(), past.into(), x.into(), pos.into(), t.into(), c.into()],
        )?;
    }
    for i in 0..3 {
        // The row of the new history is global position pos + t - 3 + i: from `x` when the call has it, from the old
        // history otherwise. Increasing i keeps every read ahead of the write that would clobber it.
        let g = pos as isize + t as isize - 3 + i as isize;
        if g < 0 {
            continue;
        }
        let (src, off) = if g >= pos as isize {
            (x, (g as usize - pos) * c * 4)
        } else {
            (past, (g as usize + 3 - pos) * c * 4)
        };
        past.copy_from_shared(i * c * 4, src, off, c * 4, st)?;
    }
    Ok(())
}

/// The buffers the grouped MoE of a whole chunk needs: the pairs sorted by expert, the gathered int8 activations,
/// the expert outputs per sorted row. Allocated only when the chunk is long enough to take the grouped path.
struct MoeGrouped {
    /// [T, hidden] and [T, hidden / 128]: the rotated int8 activations of the tokens.
    xq: DeviceBuffer,
    xsc: DeviceBuffer,
    /// [P, hidden] and [P, hidden / 128]: the same rows gathered once per (token, slot) pair, in sorted order.
    xg: DeviceBuffer,
    xgs: DeviceBuffer,
    /// [E + 1] row offsets per expert, [P] sorted row → pair, [P] pair → sorted row, and the tiles of both sizes.
    offs: DeviceBuffer,
    perm: DeviceBuffer,
    pos: DeviceBuffer,
    tiles: DeviceBuffer,
    ntiles: DeviceBuffer,
    /// [P, 2 * inter] gate|up of every pair (the checkpoint's interleaved order), [P, inter] the activated rows,
    /// [P, inter] + [P, inter / 128] the same quantized for the down GEMM, [P, hidden] the expert outputs.
    gu: DeviceBuffer,
    act: DeviceBuffer,
    aq: DeviceBuffer,
    aqs: DeviceBuffer,
    y: DeviceBuffer,
}

impl MoeGrouped {
    fn new(t: usize, h: usize, inter: usize, experts: usize, topk: usize) -> Result<Self> {
        let p = t * topk;
        let e1 = experts + 1;
        Ok(Self {
            xq: DeviceBuffer::alloc(t * h)?,
            xsc: f32s(t * h / 128)?,
            xg: DeviceBuffer::alloc(p * h)?,
            xgs: f32s(p * h / 128)?,
            offs: DeviceBuffer::alloc((e1 + 1) * 4)?,
            perm: DeviceBuffer::alloc(p * 4)?,
            pos: DeviceBuffer::alloc(p * 4)?,
            // 64-row tiles, then the 32-row remainder of each expert after the `off2` slot of the first list.
            tiles: DeviceBuffer::alloc((p.div_ceil(64) + 3 * e1 + 2) * 8)?,
            ntiles: DeviceBuffer::alloc(8)?,
            gu: f32s(p * 2 * inter)?,
            act: f32s(p * inter)?,
            aq: DeviceBuffer::alloc(p * inter)?,
            aqs: f32s(p * inter / 128)?,
            y: f32s(p * h)?,
        })
    }
}

/// The buffers a layer reads and writes, allocated once per `step` and reused by every layer and every pass. The
/// graph runs on the whole sequence, so there is no per-layer state: the loop's extra passes replay the same
/// buffers from the prelude's output.
struct Graph {
    t: usize,
    h: usize,
    eps: f32,
    x: DeviceBuffer,
    blocks: Vec<DeviceBuffer>,
    pre_loop_hidden: DeviceBuffer,
    block_residual: DeviceBuffer,
    xr: DeviceBuffer,
    xh: DeviceBuffer,
    xsc: DeviceBuffer,
    normed: DeviceBuffer,
    states: DeviceBuffer,
    mix_out: DeviceBuffer,
    msg_out: DeviceBuffer,
    msg: [DeviceBuffer; 4],
    msg_queries: DeviceBuffer,
    msg_q4: DeviceBuffer,
    attn_in: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    r: DeviceBuffer,
    gate: DeviceBuffer,
    sink: DeviceBuffer,
    attn_out: DeviceBuffer,
    y: DeviceBuffer,
    moe_out: DeviceBuffer,
    logits: DeviceBuffer,
    route_ids: DeviceBuffer,
    route_w: DeviceBuffer,
    gu: DeviceBuffer,
    act: DeviceBuffer,
    ones_k: DeviceBuffer,
    past: DeviceBuffer,
    /// The grouped MoE of the prefill: None on the one-token path, where the per-pair kernels read each expert once.
    moe: Option<MoeGrouped>,
    qin: DeviceBuffer,
    kin: DeviceBuffer,
    vin: DeviceBuffer,
    qn: DeviceBuffer,
    kn: DeviceBuffer,
    vn: DeviceBuffer,
    kt: DeviceBuffer,
    vt: DeviceBuffer,
    yt: DeviceBuffer,
    taud: Option<DeviceBuffer>,
}

impl Graph {
    fn new(model: &Model, t: usize, pos: usize) -> Result<Self> {
        let c = &model.cfg;
        let i = c.inkling2.as_ref().context("no inkling2 config")?;
        let h = c.hidden;
        let heads = i.heads;
        let kv = i.kv_heads;
        let hd = i.head_dim;
        let qd = heads * hd;
        let kvd = kv * hd;
        let kmax = qd.max(h);
        // The log-scaled positions, for the global positions this chunk covers: 1 below the floor, so 1 for any
        // sequence shorter than it.
        let mut tau: Vec<f32> = (0..t)
            .map(|p| 1.0 + i.log_alpha * (((pos + p + 1) as f32) / i.log_floor as f32).max(1.0).ln())
            .collect();
        if tau.iter().all(|x| *x == 1.0) {
            tau.clear();
        }
        let taud = if tau.is_empty() {
            None
        } else {
            Some(DeviceBuffer::from_slice(&tau)?)
        };
        Ok(Self {
            t,
            h,
            eps: c.eps,
            x: f32s(t * h)?,
            blocks: Vec::with_capacity(8),
            pre_loop_hidden: f32s(t * h)?,
            block_residual: f32s(t * h)?,
            xr: f32s(t * kmax)?,
            xh: DeviceBuffer::alloc(t * kmax * 2)?,
            xsc: f32s(t * kmax.div_ceil(128))?,
            normed: f32s(t * h)?,
            states: f32s(8 * t * h)?,
            mix_out: f32s(t * 3 * h)?,
            msg_out: f32s(t * 4 * h)?,
            msg: [f32s(t * h)?, f32s(t * h)?, f32s(t * h)?, f32s(t * h)?],
            msg_queries: f32s((i.loop_end - i.loop_start) * 4 * h)?,
            msg_q4: f32s(4 * h)?,
            attn_in: f32s(t * h)?,
            q: f32s(t * qd)?,
            k: f32s(t * kvd)?,
            v: f32s(t * kvd)?,
            r: f32s(t * heads * i.d_rel)?,
            gate: f32s(t * qd)?,
            sink: f32s(t * heads)?,
            attn_out: f32s(t * qd)?,
            y: f32s(t * h)?,
            moe_out: f32s(t * h)?,
            logits: f32s(t * i.experts)?,
            route_ids: DeviceBuffer::zeroed(t * i.top_k * 4)?,
            route_w: f32s(t * i.top_k)?,
            gu: f32s(t * i.top_k * 2 * i.moe_inter)?,
            act: f32s(t * i.top_k * i.moe_inter)?,
            ones_k: DeviceBuffer::from_slice(&vec![1.0f32; kvd])?,
            past: DeviceBuffer::zeroed(3 * h * 4)?,
            moe: Some(MoeGrouped::new(t, h, i.moe_inter, i.experts, i.top_k)?),
            qin: f32s(t * h)?,
            kin: f32s(t * h)?,
            vin: f32s(t * h)?,
            qn: f32s(t * h)?,
            kn: f32s(t * h)?,
            vn: f32s(t * h)?,
            kt: f32s(t * kvd)?,
            vt: f32s(t * kvd)?,
            yt: f32s(t * h)?,
            taud,
        })
    }

    /// Runs one decoder layer. `messages` is `Some((previous core blocks, previous pass output))` on the loop's
    /// extra passes: the four cross-pass queries then mix a message bank into the attention and MLP inputs.
    /// `place` is the session's slot for this layer's attention when the caller has a cache: the keys and values
    /// are written there and, with `place.one`, the convolution and the attention run on the one-token path.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &mut self,
        model: &Model,
        st: &Stream,
        i: &Inkling2Config,
        w: &Weights,
        l: usize,
        messages: Option<(&[DeviceBuffer], &DeviceBuffer)>,
        mut place: Option<Place<'_>>,
    ) -> Result<()> {
        let one = place.as_ref().is_some_and(|p| p.one);
        let layer = &w.layers[l];
        let full = layer.full;
        let t = self.t;
        let h = self.h;
        let eps = self.eps;
        let heads = i.heads;
        let kv = i.kv_heads;
        let hd = i.head_dim;
        let d_rel = i.d_rel;
        let kvd = kv * hd;
        let experts = i.experts;
        let topk = i.top_k;

        self.block_residual.copy_from_shared(0, &self.x, 0, t * h * 4, st)?;

        // 1. The attention input: the raw state for layer 0, the attention residual above it.
        let nq = if full { 3 } else { 1 };
        if l == 0 {
            self.qin.copy_from_shared(0, &self.x, 0, t * h * 4, st)?;
            self.kin.copy_from_shared(0, &self.x, 0, t * h * 4, st)?;
            self.vin.copy_from_shared(0, &self.x, 0, t * h * 4, st)?;
        } else {
            mix(
                model,
                st,
                &self.mix_out,
                &self.states,
                &self.blocks,
                &self.x,
                &w.attn_queries[l - 1],
                nq,
                t,
                h,
                eps,
                0,
            )?;
            copy_nq(model, &self.mix_out, &self.qin, 0, t, h, nq, st)?;
            if full {
                copy_nq(model, &self.mix_out, &self.kin, 1, t, h, 3, st)?;
                copy_nq(model, &self.mix_out, &self.vin, 2, t, h, 3, st)?;
            } else {
                self.kin.copy_from_shared(0, &self.qin, 0, t * h * 4, st)?;
                self.vin.copy_from_shared(0, &self.qin, 0, t * h * 4, st)?;
            }
        }

        // 1b. The cross-pass message gates, on the extra passes only. The four queries of a core layer attend over
        // the previous pass's core blocks and its output; the controller then mixes the four messages into the
        // query, key, value and MLP inputs.
        if let Some((prev_blocks, prev_hidden)) = messages {
            let ci = l - i.loop_start;
            self.msg_q4
                .copy_from_shared(0, &self.msg_queries, ci * 4 * h * 4, 4 * h * 4, st)?;
            mix(
                model,
                st,
                &self.msg_out,
                &self.states,
                prev_blocks,
                prev_hidden,
                &self.msg_q4,
                4,
                t,
                h,
                eps,
                0,
            )?;
            for qi in 0..4 {
                copy_nq(model, &self.msg_out, &self.msg[qi], qi, t, h, 4, st)?;
            }
            let ctrl = layer.controller.as_ref().context("loop layer without controller")?;
            // A sliding layer's attention input is one query for all three streams, so each gate reads the same state
            // and writes its own copy; a full layer has three.
            let (cq, ck, cv) = if full {
                (&self.qin, &self.kin, &self.vin)
            } else {
                self.attn_in.copy_from_shared(0, &self.qin, 0, t * h * 4, st)?;
                (&self.qin, &self.attn_in, &self.attn_in)
            };
            // One launch for the three: each was a single block on a single compute unit, and three of those in a row
            // are three latencies. Per gate the arithmetic is unchanged.
            control3(
                model,
                st,
                [&self.qin, &self.kin, &self.vin],
                [cq, ck, cv],
                [&self.msg[0], &self.msg[1], &self.msg[2]],
                [&ctrl.attn, &ctrl.k, &ctrl.v],
                t,
                h,
                i.loop_rank,
                eps,
            )?;
        }

        // 2. The input norms and the projections.
        let (dyn_k, dyn_v) = if full {
            let (q_norm, k_norm, v_norm) = match &layer.sconv {
                Sconv::Dynamic {
                    q_norm, k_norm, v_norm, ..
                } => (q_norm, k_norm, v_norm),
                _ => unreachable!(),
            };
            rms(model, st, &self.qn, &self.qin, q_norm, t, h, eps)?;
            rms(model, st, &self.kn, &self.kin, k_norm, t, h, eps)?;
            rms(model, st, &self.vn, &self.vin, v_norm, t, h, eps)?;
            let (dk, dv) = (f32s(t * kv * i.sconv_k)?, f32s(t * kv * i.sconv_k)?);
            let (k_proj, v_proj) = match &layer.sconv {
                Sconv::Dynamic { k_proj, v_proj, .. } => (k_proj, v_proj),
                _ => unreachable!(),
            };
            raw_gemm(model, st, &dk, &self.kn, k_proj, t, kv * i.sconv_k, h)?;
            raw_gemm(model, st, &dv, &self.vn, v_proj, t, kv * i.sconv_k, h)?;
            (Some(dk), Some(dv))
        } else {
            let an = layer.attn_norm.as_ref().context("sliding layer without attn_norm")?;
            rms(model, st, &self.qn, &self.qin, an, t, h, eps)?;
            if messages.is_some() {
                rms(model, st, &self.kn, &self.kin, an, t, h, eps)?;
                rms(model, st, &self.vn, &self.vin, an, t, h, eps)?;
            } else {
                self.kn.copy_from_shared(0, &self.qn, 0, t * h * 4, st)?;
                self.vn.copy_from_shared(0, &self.qn, 0, t * h * 4, st)?;
            }
            (None, None)
        };
        lin(
            model,
            st,
            &mut self.xr,
            &mut self.xh,
            &mut self.xsc,
            &self.q,
            &self.qn,
            &layer.wq,
            t,
        )?;
        lin(
            model,
            st,
            &mut self.xr,
            &mut self.xh,
            &mut self.xsc,
            &self.k,
            &self.kn,
            &layer.wk,
            t,
        )?;
        lin(
            model,
            st,
            &mut self.xr,
            &mut self.xh,
            &mut self.xsc,
            &self.v,
            &self.vn,
            &layer.wv,
            t,
        )?;
        lin(
            model,
            st,
            &mut self.xr,
            &mut self.xh,
            &mut self.xsc,
            &self.r,
            &self.qn,
            &layer.wr,
            t,
        )?;
        lin(
            model,
            st,
            &mut self.xr,
            &mut self.xh,
            &mut self.xsc,
            &self.gate,
            &self.qn,
            &layer.gate,
            t,
        )?;
        raw_gemm(model, st, &self.sink, &self.qn, &layer.sink, t, heads, h)?;
        // 3. The short convolutions of the keys and the values, then the clips.
        let (k_w, v_w) = match &layer.sconv {
            Sconv::Local { k, v } => (k, v),
            Sconv::Dynamic { k_static, v_static, .. } => (k_static, v_static),
        };
        let div = if full { 3.0 } else { (i.sconv_k as f32 + 1.0).sqrt() };
        {
            // The history of a chunk is the last three rows of its input; the one-token kernel shifts it itself.
            let (past_k, past_v) = match place.as_ref() {
                Some(p) => (&p.slot.past_k, &p.slot.past_v),
                None => (&self.past, &self.past),
            };
            sconv2(
                model,
                st,
                &self.kt,
                &self.k,
                &self.vt,
                &self.v,
                k_w,
                v_w,
                dyn_k.as_ref(),
                dyn_v.as_ref(),
                past_k,
                past_v,
                t,
                kvd,
                kv,
                div,
                one,
            )?;
        }
        if let Some(p) = place.as_mut()
            && !one
        {
            keep_last3(model, st, &p.slot.past_k, &self.k, p.pos, t, kvd, p.snap.map(|s| &s[0]))?;
            keep_last3(model, st, &p.slot.past_v, &self.v, p.pos, t, kvd, p.snap.map(|s| &s[1]))?;
        }
        self.k.copy_from_shared(0, &self.kt, 0, t * kvd * 4, st)?;
        self.v.copy_from_shared(0, &self.vt, 0, t * kvd * 4, st)?;
        let tau = match &place {
            Some(p) if full && p.one => Some(p.tau),
            _ => {
                if full {
                    self.taud.as_ref()
                } else {
                    None
                }
            }
        };
        clip(model, st, &self.q, &layer.q_clip, tau, t, heads, hd, i.qk_clip)?;
        clip(model, st, &self.k, &self.ones_k, None, t, kv, hd, i.qk_clip)?;
        rms_unnorm(model, st, &self.v, t, kv, hd, eps)?;
        clip(model, st, &self.r, &layer.rel_clip, tau, t, heads, d_rel, i.rel_clip)?;
        // The keys and values of this layer, as attention reads them, across every position the session holds.
        if let Some(p) = place.as_mut() {
            for (dst, dsc, src) in [(&p.slot.k, &p.slot.ks, &self.k), (&p.slot.v, &p.slot.vs, &self.v)] {
                model.k.get("inkling2_kv_store").launch(
                    (((t * kv).div_ceil(8)) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        dst.into(),
                        dsc.into(),
                        src.into(),
                        p.pos.into(),
                        t.into(),
                        p.slot.rows.into(),
                        p.slot.ring.into(),
                        kvd.into(),
                        hd.into(),
                    ],
                )?;
            }
        }
        let window = if full { 0 } else { i.window };
        let scale = 1.0 / (hd as f32).sqrt();
        // Rows whose keys are in the session's cache: one for a decode step, the whole chunk when the cache already
        // holds a prefix (a verify, or a chunk of a prompt). A chunk that starts at 0 is the fresh prefill and keeps
        // the tiled kernel, which is four times faster than the per-row one.
        let cached = match &place {
            Some(p) if p.one => 1,
            Some(p) if p.pos > 0 => t,
            _ => 0,
        };
        if cached >= 16 && wmma_attention(hd) {
            // A prompt chunk that continues a session: the tile reads the cache instead of the chunk's own keys, so a
            // long prompt is processed in pieces at the tiled kernel's speed. The rows of a *verify* stay on the
            // per-row path below (it is at most six rows): bit-identity with the serial step is what `spec-exact`
            // checks, and the serial step is a one-token call.
            let p = place.as_ref().context("cache chunk without a slot")?;
            if std::env::var_os("MALLEUS_INKLING2_CHUNK_DEBUG").is_some() {
                eprintln!(
                    "chunk tile: layer {l} pos {} t {t} rows {} ring {} window {window} qlast {}",
                    p.pos,
                    p.slot.rows,
                    p.slot.ring,
                    p.pos + t - 1
                );
            }
            attn_wmma(
                model,
                st,
                &self.attn_out,
                &self.q,
                &self.k,
                &self.v,
                Some((
                    &p.slot.k,
                    &p.slot.ks,
                    &p.slot.v,
                    &p.slot.vs,
                    p.slot.rows,
                    p.slot.ring,
                    p.pos,
                )),
                &self.r,
                &layer.rel_logits,
                &self.sink,
                &self.gate,
                t,
                heads,
                kv,
                hd,
                d_rel,
                i.rel_extent,
                window,
                scale,
            )?;
        } else if cached > 0 {
            // Row by row through the one-token kernels: the row of a verify is then computed by the same code that
            // computes it one step at a time, which is what makes the verify bit-identical.
            let p = place.as_ref().context("cache row without a slot")?;
            for j in 0..cached {
                let pos_j = p.pos + j;
                let k0 = if window > 0 {
                    (pos_j + 1).saturating_sub(window)
                } else {
                    0
                };
                let nk = pos_j + 1 - k0;
                let qd = (j * heads * hd * 4) as u64;
                model.k.get("inkling2_rel_bias_dec").launch(
                    ((heads * i.rel_extent).div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        p.bias.into(),
                        Arg::U64(self.r.addr() + (j * heads * d_rel * 4) as u64),
                        (&layer.rel_logits).into(),
                        heads.into(),
                        d_rel.into(),
                        i.rel_extent.into(),
                    ],
                )?;
                // A sliding layer's cache is a ring: the first key of the window sits at `k0 % ring` and wraps
                // around once, so the kernel is handed the wrapped start and the ring length.
                let ring = p.slot.ring;
                let k0r = if ring > 0 { k0 % ring } else { k0 };
                model.k.get("inkling2_attn_dec").launch(
                    (heads as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        Arg::U64(self.attn_out.addr() + qd),
                        Arg::U64(self.q.addr() + qd),
                        (&p.slot.k).into(),
                        (&p.slot.ks).into(),
                        (&p.slot.v).into(),
                        (&p.slot.vs).into(),
                        p.bias.into(),
                        Arg::U64(self.sink.addr() + (j * heads * 4) as u64),
                        Arg::U64(self.gate.addr() + qd),
                        k0r.into(),
                        nk.into(),
                        heads.into(),
                        kv.into(),
                        hd.into(),
                        ring.into(),
                        pos_j.into(),
                        i.rel_extent.into(),
                        scale.into(),
                    ],
                )?;
            }
        } else {
            attn(
                model,
                st,
                &self.attn_out,
                &self.q,
                &self.k,
                &self.v,
                &self.r,
                &layer.rel_logits,
                &self.sink,
                &self.gate,
                t,
                heads,
                kv,
                hd,
                d_rel,
                i.rel_extent,
                window,
                scale,
            )?;
        }
        lin(
            model,
            st,
            &mut self.xr,
            &mut self.xh,
            &mut self.xsc,
            &self.y,
            &self.attn_out,
            &layer.wo,
            t,
        )?;
        {
            let past = match place.as_ref() {
                Some(p) => &p.slot.past_y,
                None => &self.past,
            };
            sconv_one(
                model,
                st,
                &self.yt,
                &self.y,
                &layer.attn_sconv,
                past,
                t,
                h,
                (i.sconv_k as f32 + 1.0).sqrt(),
                one,
            )?;
        }
        if let Some(p) = place.as_mut()
            && !one
        {
            keep_last3(model, st, &p.slot.past_y, &self.y, p.pos, t, h, p.snap.map(|s| &s[2]))?;
        }
        self.y.copy_from_shared(0, &self.yt, 0, t * h * 4, st)?;

        // 4. The block bookkeeping.
        let boundary = l.is_multiple_of(i.attn_res_block);
        if boundary {
            let pushed = f32s(t * h)?;
            pushed.copy_from_shared(0, &self.block_residual, 0, t * h * 4, st)?;
            self.blocks.push(pushed);
            self.x.copy_from_shared(0, &self.y, 0, t * h * 4, st)?;
        } else {
            self.x.copy_from_shared(0, &self.block_residual, 0, t * h * 4, st)?;
            add(model, st, &self.x, &self.y, t * h)?;
        }
        // 5. The MLP input: the attention residual over the block stack.
        mix(
            model,
            st,
            &self.mix_out,
            &self.states,
            &self.blocks,
            &self.x,
            &w.mlp_queries[l],
            1,
            t,
            h,
            eps,
            0,
        )?;
        if messages.is_some() {
            let ctrl = layer.controller.as_ref().context("loop layer without controller")?;
            control(
                model,
                st,
                &self.mix_out,
                &self.mix_out,
                &self.msg[3],
                &ctrl.mlp,
                t,
                h,
                i.loop_rank,
                eps,
            )?;
        }
        // 6. The MoE.
        rms(model, st, &self.normed, &self.mix_out, &layer.mlp_norm, t, h, eps)?;
        raw_gemm(model, st, &self.logits, &self.normed, &layer.moe.gate, t, experts, h)?;
        model.k.get("inkling2_moe_route").launch(
            (t as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&self.route_ids).into(),
                (&self.route_w).into(),
                (&self.logits).into(),
                (&layer.moe.bias).into(),
                experts.into(),
                topk.into(),
                i.route_scale.into(),
            ],
        )?;
        if t >= crate::weights::WMMA_MIN_TOKENS && std::env::var_os("MALLEUS_INKLING2_MOE_PAIRS").is_none() {
            // A chunk: every expert's weights cross the bus once for all the tokens that picked it, not once per
            // (token, slot) pair, and the activations are int8 (the W4A8 GEMMs). The rows of a *verify* cannot take
            // this path: the int8 activations do not round like the fp32 of a one-token step, so the committed token
            // would depend on how many rows were checked.
            self.moe_grouped(model, st, i, w, l)?;
        } else if std::env::var_os("MALLEUS_INKLING2_MOE_PAIRS").is_some() {
            // The old per-pair path, kept for an A/B: `moe_gate_up` reads an expert once per (token, slot) pair.
            self.moe_pairs(model, st, i, w, l)?;
        } else {
            // A decode step or a verify: the pairs sorted by expert, every expert's rows in one GEMV. The per-row sum
            // does not depend on how many rows the launch has, so a verify row is the serial step bit for bit, and an
            // expert's weights cross the bus once however many tokens picked it.
            self.moe_grp(model, st, i, w, l)?;
        }
        {
            let past = match place.as_ref() {
                Some(p) => &p.slot.past_moe,
                None => &self.past,
            };
            sconv_one(
                model,
                st,
                &self.yt,
                &self.moe_out,
                &layer.mlp_sconv,
                past,
                t,
                h,
                (i.sconv_k as f32 + 1.0).sqrt(),
                one,
            )?;
        }
        if let Some(p) = place.as_mut()
            && !one
        {
            keep_last3(
                model,
                st,
                &p.slot.past_moe,
                &self.moe_out,
                p.pos,
                t,
                h,
                p.snap.map(|s| &s[3]),
            )?;
        }
        self.moe_out.copy_from_shared(0, &self.yt, 0, t * h * 4, st)?;
        add(model, st, &self.x, &self.moe_out, t * h)?;
        Ok(())
    }

    /// The routed experts, one launch per (token, slot) pair: `moe_gate_up` reads the expert's weights once for every
    /// token that picked it, so the traffic of a verify grows with its width. Kept for the A/B
    /// (`MALLEUS_INKLING2_MOE_PAIRS=1`); `moe_grp` is the same arithmetic with the pairs sorted by expert.
    fn moe_pairs(&mut self, model: &Model, st: &Stream, i: &Inkling2Config, w: &Weights, l: usize) -> Result<()> {
        let (t, h) = (self.t, self.h);
        let (inter, topk) = (i.moe_inter, i.top_k);
        let p = t * topk;
        let layer = &w.layers[l];
        model.k.get("moe_gate_up").launch(
            ((2 * inter).div_ceil(32) as u32, p as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&self.gu).into(),
                (&self.normed).into(),
                (&layer.moe.gate_up.0).into(),
                (&layer.moe.gate_up.1).into(),
                (&self.route_ids).into(),
                topk.into(),
                (2 * inter).into(),
                h.into(),
                (&model.signs).into(),
            ],
        )?;
        model.k.get("inkling2_moe_act").launch(
            ((p * inter / 128) as u32, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[
                (&self.act).into(),
                (&self.gu).into(),
                (&model.signs).into(),
                inter.into(),
                i.gate_softcap.into(),
                i.up_softcap.into(),
            ],
        )?;
        model.k.get("moe_down").launch(
            (h.div_ceil(32) as u32, t as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&self.moe_out).into(),
                (&self.act).into(),
                (&layer.moe.down.0).into(),
                (&layer.moe.down.1).into(),
                (&self.route_ids).into(),
                (&self.route_w).into(),
                topk.into(),
                h.into(),
                inter.into(),
                0i32.into(),
            ],
        )?;

        Ok(())
    }

    /// The routed experts of a decode step or a verify: `moe_sort` orders the (token, slot) pairs by expert, so each
    /// used expert's rows go through one GEMV and its weights cross the bus once however many tokens picked it. The
    /// per-row sum does not depend on the row count, which is what lets a verify row be the serial step bit for bit.
    fn moe_grp(&mut self, model: &Model, st: &Stream, i: &Inkling2Config, w: &Weights, l: usize) -> Result<()> {
        let (t, h) = (self.t, self.h);
        let (inter, experts, topk) = (i.moe_inter, i.experts, i.top_k);
        let p = t * topk;
        let layer = &w.layers[l];
        let g = self.moe.as_ref().context("no grouped MoE scratch")?;
        model.k.get("moe_sort").launch(
            (1, 1, 1),
            (1024, 1, 1),
            0,
            st,
            &[
                (&g.offs).into(),
                (&g.perm).into(),
                (&g.pos).into(),
                (&g.tiles).into(),
                (&g.ntiles).into(),
                (&self.route_ids).into(),
                p.into(),
                (experts + 1).into(),
                // A tile holds at most the rows the kernel is instantiated for (`GEMV_MAX_T`): a wider tile would
                // leave the rows past that count uncomputed.
                (crate::weights::GEMV_MAX_T as i32).into(),
                0i32.into(),
                0i32.into(),
            ],
        )?;
        // The two kernels of a decode step have no tile loop to run (one row per expert), so they are their own
        // instantiation; a verify takes the one that fits up to six rows.
        let one = if t == 1 { "1" } else { "" };
        model.k.get(&format!("moe_gate_up_grp{one}")).launch(
            ((2 * inter).div_ceil(32) as u32, p as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&self.gu).into(),
                (&self.normed).into(),
                (&layer.moe.gate_up.0).into(),
                (&layer.moe.gate_up.1).into(),
                (&g.offs).into(),
                (&g.perm).into(),
                (&g.tiles).into(),
                (&g.ntiles).into(),
                topk.into(),
                (2 * inter).into(),
                h.into(),
                (&model.signs).into(),
            ],
        )?;
        model.k.get("inkling2_moe_act").launch(
            ((p * inter / 128) as u32, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[
                (&self.act).into(),
                (&self.gu).into(),
                (&model.signs).into(),
                inter.into(),
                i.gate_softcap.into(),
                i.up_softcap.into(),
            ],
        )?;
        // The down projection writes one row per pair (expert-sorted); `moe_combine` then sums the slots of a token in
        // the order the one-pair kernel weighted them, so the two agree bit for bit.
        model.k.get(&format!("moe_down_grp{one}")).launch(
            (h.div_ceil(32) as u32, p as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&g.y).into(),
                (&self.act).into(),
                (&layer.moe.down.0).into(),
                (&layer.moe.down.1).into(),
                (&g.offs).into(),
                (&g.perm).into(),
                (&g.tiles).into(),
                (&g.ntiles).into(),
                h.into(),
                inter.into(),
            ],
        )?;
        model.k.get("moe_combine").launch(
            (t as u32, h.div_ceil(1024) as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&self.moe_out).into(),
                (&g.y).into(),
                (&self.route_w).into(),
                (&g.pos).into(),
                topk.into(),
                h.into(),
                0i32.into(),
            ],
        )?;
        Ok(())
    }

    /// The routed experts of a whole chunk, grouped by expert: `moe_sort` orders the (token, slot) pairs so each
    /// expert's weights cross the bus once for every token that picked it. `moe_gate_up`/`moe_down` read them once
    /// per pair, which is the same traffic as a decode step per token and is what used to cap the prefill.
    fn moe_grouped(&mut self, model: &Model, st: &Stream, i: &Inkling2Config, w: &Weights, l: usize) -> Result<()> {
        let g = self.moe.as_ref().context("no grouped MoE scratch")?;
        let layer = &w.layers[l];
        let (t, h) = (self.t, self.h);
        let (inter, experts, topk) = (i.moe_inter, i.experts, i.top_k);
        let p = t * topk;
        let e1 = experts + 1;
        // Tile rows: 128 once experts average more than ~48 rows (the weights are read once per tile), 64 below,
        // and the remainder of an expert on the next smaller kernel.
        let rows = p / experts;
        let bm = if rows >= *crate::engine::MOE_BM128_ROWS_ENV {
            128
        } else {
            crate::engine::MOE_BM
        };
        let split = (bm == 64 || rows < 200) && std::env::var("MALLEUS_MOE_SPLIT").map_or(true, |v| v != "0");
        let off2 = p.div_ceil(bm) + 1;
        let max_tiles = if split { p / bm } else { p.div_ceil(bm) + e1 };
        let (k1, k2) = match bm {
            128 => ("moe_w4a8_m128", "moe_w4a8"),
            _ => ("moe_w4a8", "moe_w4a8_m32"),
        };
        // 1. What the grouped GEMMs read: the activations rotated (the MLS4 weights are stored in that basis) and
        //    quantized to int8, one scale per 128-wide group.
        model.k.get("rot_quant_i8_w").launch(
            ((t * h / 128) as u32, 1, 1),
            (32, 1, 1),
            0,
            st,
            &[
                (&g.xq).into(),
                (&g.xsc).into(),
                (&self.normed).into(),
                (&model.signs).into(),
                h.into(),
            ],
        )?;
        // 2. The pairs, grouped by expert: offsets, the two permutations, and the tiles of both sizes.
        model.k.get("moe_sort").launch(
            (1, 1, 1),
            (1024, 1, 1),
            0,
            st,
            &[
                (&g.offs).into(),
                (&g.perm).into(),
                (&g.pos).into(),
                (&g.tiles).into(),
                (&g.ntiles).into(),
                (&self.route_ids).into(),
                p.into(),
                e1.into(),
                bm.into(),
                (if split { bm / 2 } else { 0 }).into(),
                off2.into(),
            ],
        )?;
        // 3. One copy of the token's activations per pair, in sorted order, with its 128-wide scales.
        model.k.get("moe_gather_g").launch(
            (p as u32, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[
                (&g.xg).into(),
                (&g.xgs).into(),
                (&g.xq).into(),
                (&g.xsc).into(),
                (&g.perm).into(),
                topk.into(),
                h.into(),
            ],
        )?;
        // 4. gate|up: `w13` is interleaved (gate at 2j, up at 2j+1), so N = 2 * inter keeps the checkpoint's order
        //    and the activation reads it as it is.
        self.moe_gemm(
            model,
            st,
            &g.gu,
            &g.xg,
            &g.xgs,
            &layer.moe.gate_up,
            2 * inter,
            h,
            off2,
            max_tiles,
            e1,
            k1,
            k2,
        )?;
        model.k.get("inkling2_moe_act").launch(
            ((p * inter / 128) as u32, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[
                (&g.act).into(),
                (&g.gu).into(),
                (&model.signs).into(),
                inter.into(),
                i.gate_softcap.into(),
                i.up_softcap.into(),
            ],
        )?;
        // 5. The activation is already in the rotated basis: quantize its rows, do not rotate them again.
        model.k.get("inkling2_quant_i8").launch(
            ((p * inter / 128) as u32, 1, 1),
            (32, 1, 1),
            0,
            st,
            &[(&g.aq).into(), (&g.aqs).into(), (&g.act).into()],
        )?;
        // 6. down, then the pair weights fold the sorted rows back into their tokens.
        self.moe_gemm(
            model,
            st,
            &g.y,
            &g.aq,
            &g.aqs,
            &layer.moe.down,
            h,
            inter,
            off2,
            max_tiles,
            e1,
            k1,
            k2,
        )?;
        model.k.get("moe_combine").launch(
            (t as u32, h.div_ceil(1024) as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&self.moe_out).into(),
                (&g.y).into(),
                (&self.route_w).into(),
                (&g.pos).into(),
                topk.into(),
                h.into(),
                0i32.into(),
            ],
        )?;
        Ok(())
    }

    /// One grouped W4A8 GEMM: the 64-row tiles, then the 32-row remainders.
    #[allow(clippy::too_many_arguments)]
    fn moe_gemm(
        &self,
        model: &Model,
        st: &Stream,
        out: &DeviceBuffer,
        x: &DeviceBuffer,
        xs: &DeviceBuffer,
        w: &(DeviceBuffer, DeviceBuffer),
        n: usize,
        k: usize,
        off2: usize,
        max_tiles: usize,
        e1: usize,
        k1: &str,
        k2: &str,
    ) -> Result<()> {
        let g = self.moe.as_ref().context("no grouped MoE scratch")?;
        let nb = n.div_ceil(128) as u32;
        let launch = |name: &str, tiles: u64, cnt: u64, nt: usize| -> Result<()> {
            if nt == 0 {
                return Ok(());
            }
            model.k.get(name).launch(
                (nt as u32 * nb, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    out.into(),
                    x.into(),
                    xs.into(),
                    (&w.0).into(),
                    (&w.1).into(),
                    (&g.offs).into(),
                    Arg::U64(tiles),
                    Arg::U64(cnt),
                    n.into(),
                    k.into(),
                ],
            )?;
            Ok(())
        };
        launch(k1, g.tiles.addr(), g.ntiles.addr(), max_tiles)?;
        launch(k2, g.tiles.addr() + 8 * off2 as u64, g.ntiles.addr() + 4, 2 * e1)?;
        Ok(())
    }
}

/// Runs `tokens` through the whole graph and returns the logits of every position, `[t, vocab]` row-major.
///
/// With a `cache` the keys, the values and the convolution histories land in the session, and a single token runs
/// the one-token kernels against them instead of the whole prompt again.
fn run(model: &Model, tokens: &[u32], cache: Option<&mut Cache>, n_logits: usize) -> Result<Vec<f32>> {
    run_mm(model, tokens, cache, n_logits, &[])
}

/// [`run`], with rows `[start, start + n)` of the input embeddings replaced by `buf` ([n, hidden], f32): the image
/// rows an Inkling-2 prompt carries in place of its `<|unused_…|>` placeholders.
fn run_mm(
    model: &Model,
    tokens: &[u32],
    cache: Option<&mut Cache>,
    n_logits: usize,
    inserts: &[(usize, &DeviceBuffer)],
) -> Result<Vec<f32>> {
    let c = &model.cfg;
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let w = model.inkling2.as_ref().context("no inkling2 weights")?;
    let st = &model.stream;
    let t = tokens.len();
    ensure!(t > 0, "inkling2: empty input");
    let h = c.hidden;
    let mut cache = cache;
    let (pos, one) = match cache.as_deref() {
        Some(k) => {
            ensure!(
                k.len() + t <= k.max,
                "inkling2: {} tokens over the {} a session holds",
                k.len() + t,
                k.max
            );
            // A chunk that continues a cache (a speculative verify, or the pieces of a prompt) reads the keys before
            // it from the cache; only a chunk that *starts* a prompt takes the tiled attention.
            (k.len(), t == 1)
        }
        None => (0, false),
    };
    // A chunk that continues a cache reads the keys of its own earlier rows back from the session's ring, which is
    // `window + PREFILL_CHUNK + DRAFT_ROWS + 1` rows deep: a longer piece would overwrite keys its own first rows
    // still read, silently. The scheduler feeds pieces of `PREFILL_CHUNK`; this stops any other caller before the
    // cache is corrupted instead of after.
    let ring_depth = PREFILL_CHUNK + DRAFT_ROWS + 1;
    ensure!(
        pos == 0 || t <= ring_depth,
        "inkling2: a cached chunk of {t} rows at position {pos} overruns the sliding ring ({ring_depth} rows): a \
         chunk that continues a prefix goes in pieces of at most {PREFILL_CHUNK}"
    );
    if let Some(k) = cache.as_deref_mut()
        && one
    {
        // The log-scaled position factor of this token, 1 below the floor.
        let tau = 1.0 + i.log_alpha * (((pos + 1) as f32) / i.log_floor as f32).max(1.0).ln();
        k.tau.upload(&[tau])?;
    }
    // The per-row history snapshot exists for one thing: rolling back the tail of a verify that the draft proposed.
    // Without the head there is no rollback, and a prefill chunk of 512 rows would pay for a snapshot nobody reads
    // (`rocprofv3` at a 16K fill: 1.5 % of the prefill and a 27 MB allocation, 6 335 launches).
    if let Some(k) = cache.as_deref_mut()
        && t > 1
        && k.draft.is_some()
    {
        k.snapshot(t)?;
    }
    let mut g = Graph::new(model, t, pos)?;

    let ids32: Vec<i32> = tokens.iter().map(|&v| v as i32).collect();
    let idbuf = DeviceBuffer::from_slice(&ids32)?;
    model.k.get("embed_bf16").launch(
        (t as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&g.x).into(), (&w.embed).into(), (&idbuf).into(), h.into()],
    )?;
    rms(model, st, &g.x, &g.x, &w.embed_norm, t, h, c.eps)?;
    // The image features replace the placeholder's embedding rows after the embedding's own norm, which is where the
    // reference splices them (`masked_scatter` on `embed_tokens(input_ids)`).
    for (start, buf) in inserts {
        let rows = buf.len() / (h * 4);
        ensure!(*start + rows <= t, "inkling2: image rows out of range");
        g.x.copy_from(start * h * 4, buf, 0, rows * h * 4, st)?;
    }

    // The cross-pass queries, [core, 4, hidden]: the checkpoint stores two rows per core layer in each table, and
    // the message attention reads [query, key, value, mlp] as [q[2i], kv[2i], kv[2i+1], q[2i+1]].
    let core = i.loop_end - i.loop_start;
    for ci in 0..core {
        g.msg_queries
            .copy_from_shared(ci * 4 * h * 4, &w.cross_pass_q, 2 * ci * h * 4, h * 4, st)?;
        g.msg_queries
            .copy_from_shared((ci * 4 + 1) * h * 4, &w.cross_pass_kv, 2 * ci * h * 4, h * 4, st)?;
        g.msg_queries
            .copy_from_shared((ci * 4 + 2) * h * 4, &w.cross_pass_kv, (2 * ci + 1) * h * 4, h * 4, st)?;
        g.msg_queries
            .copy_from_shared((ci * 4 + 3) * h * 4, &w.cross_pass_q, (2 * ci + 1) * h * 4, h * 4, st)?;
    }

    // The target states the MTP head reads: the state entering three of the visits the loop makes, and the final
    // normalized one. `target_layers` holds the visit indices (`5, 23, 41, 48` for this checkpoint) and a visit
    // index counts every application of a layer, the loop's repeats included.
    let core_len = i.loop_end - i.loop_start;
    let visit_of = |l: usize, pass: usize| -> usize {
        if l < i.loop_start {
            l
        } else if l < i.loop_end {
            i.loop_start + pass * core_len + (l - i.loop_start)
        } else {
            i.loop_start + i.loop_passes * core_len + (l - i.loop_end)
        }
    };
    let draft_visit = |l: usize, pass: usize| -> Option<usize> {
        let v = visit_of(l, pass);
        i.mtp.as_ref()?.target_layers.iter().position(|&x| x == v)
    };
    // The rows whose four target states the step keeps: the whole step up to one prefill chunk, because the draft's
    // context wants a K/V row per committed token and not only per row it drafts from (`FEAT_ROWS`).
    let nr = t.min(FEAT_ROWS);
    let feat = |cache: &mut Option<&mut Cache>, gx: &DeviceBuffer, j: usize| -> Result<()> {
        let Some(k) = cache.as_deref_mut() else { return Ok(()) };
        if k.draft.is_none() {
            return Ok(());
        }
        k.feats
            .copy_from_shared(j * FEAT_ROWS * h * 4, gx, (t - nr) * h * 4, nr * h * 4, st)?;
        k.feat_rows = nr;
        Ok(())
    };

    // The prelude, then the first pass over the core layers.
    for l in 0..i.loop_start {
        if let Some(j) = draft_visit(l, 0) {
            feat(&mut cache, &g.x, j)?;
        }
        let place = place_of(cache.as_deref_mut(), slot_of(i, c.n_layers, 0, l), pos, one);
        g.layer(model, st, i, w, l, None, place)?;
    }
    g.pre_loop_hidden.copy_from_shared(0, &g.x, 0, t * h * 4, st)?;
    let num_pre_blocks = (0..i.loop_start).filter(|l| l.is_multiple_of(i.attn_res_block)).count();
    for l in i.loop_start..i.loop_end {
        if let Some(j) = draft_visit(l, 0) {
            feat(&mut cache, &g.x, j)?;
        }
        let place = place_of(cache.as_deref_mut(), slot_of(i, c.n_layers, 0, l), pos, one);
        g.layer(model, st, i, w, l, None, place)?;
    }

    // The extra passes restart from the prelude's output and its block stack, and read the previous pass's core
    // blocks (everything after the first one) and its output as the message bank.
    for pass in 1..i.loop_passes {
        let prev_blocks: Vec<DeviceBuffer> = g.blocks.split_off(num_pre_blocks + 1);
        let prev_hidden = f32s(t * h)?;
        prev_hidden.copy_from_shared(0, &g.x, 0, t * h * 4, st)?;
        g.x.copy_from_shared(0, &g.pre_loop_hidden, 0, t * h * 4, st)?;
        g.blocks.truncate(num_pre_blocks);
        for l in i.loop_start..i.loop_end {
            if let Some(j) = draft_visit(l, pass) {
                feat(&mut cache, &g.x, j)?;
            }
            let place = place_of(cache.as_deref_mut(), slot_of(i, c.n_layers, pass, l), pos, one);
            g.layer(model, st, i, w, l, Some((&prev_blocks, &prev_hidden)), place)?;
        }
    }

    // The tail.
    for l in i.loop_end..c.n_layers {
        if let Some(j) = draft_visit(l, 0) {
            feat(&mut cache, &g.x, j)?;
        }
        let place = place_of(cache.as_deref_mut(), slot_of(i, c.n_layers, 0, l), pos, one);
        g.layer(model, st, i, w, l, None, place)?;
    }
    if let Some(k) = cache.as_mut() {
        k.len += t;
    }

    // The final attention residual, the norm and the head, on the last `out_rows` rows only: a generation step
    // wants one row, and the head is 201 024 x 2304 — running it on a whole prompt is a 1.65 GB download at 8K.
    // With the draft head on, the norm runs on every row the step captured instead (the fourth target state is the
    // normalized state of each committed token); the head itself still runs on the rows asked for.
    let out_rows = n_logits.clamp(1, t);
    let draft_rows = match cache.as_ref().and_then(|k| k.draft.as_ref()) {
        Some(_) => t.min(FEAT_ROWS),
        None => out_rows,
    };
    let norm_rows = draft_rows.max(out_rows);
    let row0 = t - norm_rows;
    mix(
        model,
        st,
        &g.mix_out,
        &g.states,
        &g.blocks,
        &g.x,
        &w.output_query,
        1,
        out_rows,
        h,
        c.eps,
        row0,
    )?;
    rms(model, st, &g.normed, &g.mix_out, &w.norm, norm_rows, h, c.eps)?;
    if let Some(j) = i.mtp.as_ref().and_then(|m| m.target_layers.len().checked_sub(1)) {
        // The last feature is the final normalized state: the rows the step kept, aligned with the other three.
        let nr = nr.min(norm_rows);
        // The planner only gives the features a prefill chunk's worth of rows when the head is on; with it off the
        // buffer holds `DRAFT_ROWS` and nothing reads them.
        if let Some(k) = cache.as_deref_mut()
            && k.draft.is_some()
        {
            k.feats.copy_from_shared(
                j * FEAT_ROWS * h * 4,
                &g.normed,
                (norm_rows - nr) * h * 4,
                nr * h * 4,
                st,
            )?;
        }
    }
    let logits_out = f32s(out_rows * c.vocab)?;
    // The head reads the last `out_rows` rows of the normalized state, which sit past the first of `norm_rows` once
    // the draft head has asked for them all: with the draft off they are the same rows and nothing is copied.
    let head_copy = if norm_rows != out_rows {
        Some(f32s(out_rows * h)?)
    } else {
        None
    };
    if let Some(b) = &head_copy {
        b.copy_from_shared(0, &g.normed, (norm_rows - out_rows) * h * 4, out_rows * h * 4, st)?;
    }
    let head_src = head_copy.as_ref().unwrap_or(&g.normed);
    lin(
        model,
        st,
        &mut g.xr,
        &mut g.xh,
        &mut g.xsc,
        &logits_out,
        head_src,
        &w.unembed,
        out_rows,
    )?;
    softcap(model, st, &logits_out, out_rows * c.vocab, i.logit_softcap)?;
    // Debug: the target states this step kept, one file per step. The invariant they have to satisfy is that a verify
    // row and a serial step at the same position carry the same bytes (`MALLEUS_INKLING2_MTP_DUMP`).
    if let (Ok(dir), Some(k)) = (std::env::var("MALLEUS_INKLING2_MTP_DUMP"), cache.as_ref()) {
        st.synchronize()?;
        let v = k.feats.download::<f32>(k.feats.len() / 4)?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/mtp_feats_{pos}_{t}.f32"), bytes)?;
        eprintln!(
            "mtp features: pos {pos}, {t} rows, {} of {DRAFT_ROWS} kept",
            k.feat_rows
        );
    }
    // The committed rows' draft context, from the features the step kept: one K/V row per token and draft layer.
    if let Some(k) = cache {
        let rows = k.feat_rows;
        if rows > 0 {
            let start = k.len - rows;
            draft_commit(model, k, start, rows)?;
        }
    }
    Ok(logits_out.download(out_rows * c.vocab)?)
}

/// The logits of the last `n_logits` positions of `tokens`, appending to the session's cache when there is one.
pub fn logits(model: &Model, cache: Option<&mut Cache>, tokens: &[u32], n_logits: usize) -> Result<Vec<f32>> {
    run(model, tokens, cache, n_logits)
}

/// [`logits`], with image rows spliced into the input embeddings (see [`run_mm`]).
pub fn logits_mm(
    model: &Model,
    cache: Option<&mut Cache>,
    tokens: &[u32],
    n_logits: usize,
    inserts: &[(usize, &DeviceBuffer)],
) -> Result<Vec<f32>> {
    run_mm(model, tokens, cache, n_logits, inserts)
}

/// `Place` for one layer of the session's cache, or `None` without a cache.
fn place_of(cache: Option<&mut Cache>, slot: usize, pos: usize, one: bool) -> Option<Place<'_>> {
    cache.map(|c| {
        let Cache {
            slots, bias, tau, snap, ..
        } = c;
        let snap = snap.as_ref().filter(|s| s.rows > 1).map(|s| &s.hist[slot]);
        Place {
            slot: &mut slots[slot],
            snap,
            bias: &*bias,
            tau: &*tau,
            pos,
            one,
        }
    })
}

fn rms(
    model: &Model,
    st: &Stream,
    out: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    rows: usize,
    n: usize,
    eps: f32,
) -> Result<()> {
    model.k.get("rmsnorm").launch(
        (rows as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[out.into(), x.into(), w.into(), n.into(), eps.into(), 0i32.into()],
    )?;
    Ok(())
}

fn add(model: &Model, st: &Stream, y: &DeviceBuffer, x: &DeviceBuffer, n: usize) -> Result<()> {
    model.ew(st, "add_inplace", n, &[y.into(), x.into(), n.into()])?;
    Ok(())
}

fn lin(
    model: &Model,
    st: &Stream,
    xr: &mut DeviceBuffer,
    xh: &mut DeviceBuffer,
    xsc: &mut DeviceBuffer,
    out: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &Linear,
    t: usize,
) -> Result<()> {
    model.gemm(st, xr, xh, xsc, out, crate::engine::Act::X(x), w, t, false)?;
    Ok(())
}

fn raw_gemm(
    model: &Model,
    st: &Stream,
    out: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    t: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    model.k.get("gemm_bf16").launch(
        (n.div_ceil(8) as u32, t.div_ceil(8) as u32, 1),
        (256, 1, 1),
        0,
        st,
        &[
            out.into(),
            x.into(),
            w.into(),
            t.into(),
            n.into(),
            k.into(),
            0i32.into(),
        ],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// `row0` selects the first row of the block stack this call reads: the head of a prompt only needs its last rows.
fn mix(
    model: &Model,
    st: &Stream,
    out: &DeviceBuffer,
    states: &DeviceBuffer,
    blocks: &[DeviceBuffer],
    partial: &DeviceBuffer,
    query: &DeviceBuffer,
    nq: usize,
    t: usize,
    h: usize,
    eps: f32,
    row0: usize,
) -> Result<()> {
    let n = blocks.len() + 1;
    for (j, b) in blocks.iter().enumerate() {
        states.copy_from_shared(j * t * h * 4, b, row0 * h * 4, t * h * 4, st)?;
    }
    states.copy_from_shared((n - 1) * t * h * 4, partial, row0 * h * 4, t * h * 4, st)?;
    model.k.get("inkling2_attn_res").launch(
        ((t * nq) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            out.into(),
            query.into(),
            states.into(),
            t.into(),
            n.into(),
            h.into(),
            nq.into(),
            eps.into(),
        ],
    )?;
    Ok(())
}

/// Copies the `qi`-th of `nq` outputs of a mix (`[t, nq, h]`) into a `[t, h]` buffer.
fn copy_nq(
    model: &Model,
    src: &DeviceBuffer,
    dst: &DeviceBuffer,
    qi: usize,
    t: usize,
    h: usize,
    nq: usize,
    st: &Stream,
) -> Result<()> {
    model.k.get("inkling2_copy_nq").launch(
        ((t * h).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[dst.into(), src.into(), nq.into(), qi.into(), h.into(), t.into()],
    )?;
    Ok(())
}

/// The cross-pass controller: `out = (1 - tanh(forget)) * cur + tanh(gain) * message`, with the gate read from
/// the RMS-normed concatenation of the state and the message.
#[allow(clippy::too_many_arguments)]
fn control(
    model: &Model,
    st: &Stream,
    out: &DeviceBuffer,
    cur: &DeviceBuffer,
    msg: &DeviceBuffer,
    gate: &Gate,
    t: usize,
    h: usize,
    rank: usize,
    eps: f32,
) -> Result<()> {
    model.k.get("inkling2_control").launch(
        (t as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            out.into(),
            cur.into(),
            msg.into(),
            (&gate.down).into(),
            (&gate.gain).into(),
            (&gate.forget).into(),
            h.into(),
            rank.into(),
            eps.into(),
        ],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sconv2(
    model: &Model,
    st: &Stream,
    yk: &DeviceBuffer,
    xk: &DeviceBuffer,
    yv: &DeviceBuffer,
    xv: &DeviceBuffer,
    k_w: &DeviceBuffer,
    v_w: &DeviceBuffer,
    dyn_k: Option<&DeviceBuffer>,
    dyn_v: Option<&DeviceBuffer>,
    past_k: &DeviceBuffer,
    past_v: &DeviceBuffer,
    t: usize,
    c: usize,
    heads: usize,
    div: f32,
    one: bool,
) -> Result<()> {
    for (y, x, w, dyn_w, past) in [(yk, xk, k_w, dyn_k, past_k), (yv, xv, v_w, dyn_v, past_v)] {
        sconv_launch(model, st, y, x, w, dyn_w, past, t, c, heads, div, one)?;
    }
    Ok(())
}

/// One short convolution: the whole-sequence kernel, or the one-token one that shifts the history itself.
#[allow(clippy::too_many_arguments)]
fn sconv_launch(
    model: &Model,
    st: &Stream,
    y: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    dyn_w: Option<&DeviceBuffer>,
    past: &DeviceBuffer,
    t: usize,
    c: usize,
    heads: usize,
    div: f32,
    one: bool,
) -> Result<()> {
    let d: Arg = match dyn_w {
        Some(x) => x.into(),
        None => Arg::Ptr(0),
    };
    if one {
        // The one-token kernel has no `T`: one row, and the history shifts in place.
        model.k.get("inkling2_sconv_dec").launch(
            (c.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                y.into(),
                x.into(),
                w.into(),
                d,
                past.into(),
                c.into(),
                heads.into(),
                div.into(),
            ],
        )?;
        return Ok(());
    }
    model.k.get("inkling2_sconv").launch(
        ((t * c).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            y.into(),
            x.into(),
            w.into(),
            d,
            past.into(),
            t.into(),
            c.into(),
            heads.into(),
            div.into(),
        ],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sconv_one(
    model: &Model,
    st: &Stream,
    y: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    past: &DeviceBuffer,
    t: usize,
    c: usize,
    div: f32,
    one: bool,
) -> Result<()> {
    sconv_launch(model, st, y, x, w, None, past, t, c, 1, div, one)
}

fn clip(
    model: &Model,
    st: &Stream,
    y: &DeviceBuffer,
    scale: &DeviceBuffer,
    tau: Option<&DeviceBuffer>,
    t: usize,
    heads: usize,
    dim: usize,
    thr: f32,
) -> Result<()> {
    let ta: Arg = match tau {
        Some(x) => x.into(),
        None => Arg::Ptr(0),
    };
    model.k.get("inkling2_l2_clip").launch(
        ((t * heads) as u32, 1, 1),
        (128, 1, 1),
        0,
        st,
        &[
            y.into(),
            y.into(),
            scale.into(),
            ta,
            t.into(),
            heads.into(),
            dim.into(),
            thr.into(),
        ],
    )?;
    Ok(())
}

fn rms_unnorm(
    model: &Model,
    st: &Stream,
    y: &DeviceBuffer,
    t: usize,
    heads: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    model.k.get("inkling2_rms_unnorm").launch(
        ((t * heads) as u32, 1, 1),
        (128, 1, 1),
        0,
        st,
        &[y.into(), y.into(), (t * heads).into(), dim.into(), eps.into()],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attn(
    model: &Model,
    st: &Stream,
    out: &DeviceBuffer,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    rel: &DeviceBuffer,
    proj: &DeviceBuffer,
    sink: &DeviceBuffer,
    gate: &DeviceBuffer,
    t: usize,
    heads: usize,
    nkv: usize,
    dim: usize,
    d_rel: usize,
    extent: usize,
    window: usize,
    scale: f32,
) -> Result<()> {
    // The tensor-core tile is the same block shape (16 query rows of one head) with the dots on WMMA: Q, K and V go
    // through fp16, the bias is still added in fp32. `MALLEUS_INKLING2_ATTN_F32=1` keeps the fp32 kernel for an A/B.
    if wmma_attention(dim) {
        attn_wmma(
            model, st, out, q, k, v, None, rel, proj, sink, gate, t, heads, nkv, dim, d_rel, extent, window, scale,
        )
    } else {
        model.k.get("inkling2_attn").launch(
            ((t.div_ceil(16) * heads) as u32, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[
                out.into(),
                q.into(),
                k.into(),
                v.into(),
                rel.into(),
                proj.into(),
                sink.into(),
                gate.into(),
                t.into(),
                heads.into(),
                nkv.into(),
                dim.into(),
                d_rel.into(),
                extent.into(),
                window.into(),
                scale.into(),
            ],
        )?;
        Ok(())
    }
}

/// Whether the tiled attention runs on the tensor cores: the tile holds a head's 128 dims, and the fp32 kernel is
/// kept for an A/B.
fn wmma_attention(dim: usize) -> bool {
    dim == 128 && std::env::var_os("MALLEUS_INKLING2_ATTN_F32").is_none()
}

/// A tile of the attention: the chunk's own keys and values when `cache` is `None`, the session's fp16 cache
/// ([rows, nkv, dim], position `p` at row `p % ring` when the layer slides) otherwise.
#[allow(clippy::too_many_arguments)]
fn attn_wmma(
    model: &Model,
    st: &Stream,
    out: &DeviceBuffer,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    cache: Option<(
        &DeviceBuffer,
        &DeviceBuffer,
        &DeviceBuffer,
        &DeviceBuffer,
        usize,
        usize,
        usize,
    )>,
    rel: &DeviceBuffer,
    proj: &DeviceBuffer,
    sink: &DeviceBuffer,
    gate: &DeviceBuffer,
    t: usize,
    heads: usize,
    nkv: usize,
    dim: usize,
    d_rel: usize,
    extent: usize,
    window: usize,
    scale: f32,
) -> Result<()> {
    let name = if cache.is_some() {
        "inkling2_attn_wmma_cached"
    } else {
        "inkling2_attn_wmma"
    };
    // Without a cache the tile reads this call's own keys and values (`CACHED == false`) and never touches the four
    // cache pointers, so they can point at anything valid.
    let (ck, cks, cv, cvs, rows, ring, pos) = cache.unwrap_or((k, sink, v, sink, 0, 0, 0));
    model.k.get(name).launch(
        ((t.div_ceil(16) * heads) as u32, 1, 1),
        (128, 1, 1),
        0,
        st,
        &[
            out.into(),
            q.into(),
            // Without a cache the kernels read this call's own keys and values; with one they are not read at all.
            k.into(),
            v.into(),
            ck.into(),
            cv.into(),
            cks.into(),
            cvs.into(),
            rows.into(),
            ring.into(),
            pos.into(),
            rel.into(),
            proj.into(),
            sink.into(),
            gate.into(),
            t.into(),
            heads.into(),
            nkv.into(),
            dim.into(),
            d_rel.into(),
            extent.into(),
            window.into(),
            scale.into(),
        ],
    )?;
    Ok(())
}

/// The three attention gates of an extra pass in one launch (see `inkling2_control3`).
#[allow(clippy::too_many_arguments)]
fn control3(
    model: &Model,
    st: &Stream,
    out: [&DeviceBuffer; 3],
    cur: [&DeviceBuffer; 3],
    msg: [&DeviceBuffer; 3],
    c: [&Gate; 3],
    t: usize,
    hidden: usize,
    rank: usize,
    eps: f32,
) -> Result<()> {
    model.k.get("inkling2_control3").launch(
        (t as u32, 3, 1),
        (256, 1, 1),
        0,
        st,
        &[
            out[0].into(),
            cur[0].into(),
            msg[0].into(),
            (&c[0].down).into(),
            (&c[0].gain).into(),
            (&c[0].forget).into(),
            out[1].into(),
            cur[1].into(),
            msg[1].into(),
            (&c[1].down).into(),
            (&c[1].gain).into(),
            (&c[1].forget).into(),
            out[2].into(),
            cur[2].into(),
            msg[2].into(),
            (&c[2].down).into(),
            (&c[2].gain).into(),
            (&c[2].forget).into(),
            hidden.into(),
            rank.into(),
            eps.into(),
        ],
    )?;
    Ok(())
}

fn softcap(model: &Model, st: &Stream, y: &DeviceBuffer, n: usize, cap: f32) -> Result<()> {
    model.k.get("inkling2_softcap").launch(
        (n.div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[y.into(), y.into(), n.into(), cap.into()],
    )?;
    Ok(())
}

/// `n` greedy tokens after `tokens`, on a cache of its own: the prompt in one forward, then a token at a time.
pub fn generate(model: &Model, tokens: &[u32], n: usize) -> Result<Vec<u32>> {
    let c = &model.cfg;
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let mut cache = Cache::new(model, tokens.len() + n + 1)?;
    let mut out = Vec::with_capacity(n);
    let all = run(model, tokens, Some(&mut cache), 1)?;
    let mut next = argmax(&all[all.len() - c.vocab..], i.unpadded_vocab);
    for _ in 0..n {
        out.push(next);
        if c.eos.contains(&next) {
            break;
        }
        next = argmax(&run(model, &[next], Some(&mut cache), 1)?, i.unpadded_vocab);
    }
    Ok(out)
}

// ---- the MTP head (issue #110, item 6): a block of drafts the verify then checks ----

/// Whether the draft head drafts. Opt-in (`MALLEUS_MTP=1`) until its acceptance is measured in service: a head that
/// proposes badly pays for the verify without earning it back.
pub fn draft_enabled() -> bool {
    std::env::var_os("MALLEUS_MTP").is_some_and(|v| v != "0")
}

/// The draft head's open questions, as a number each: the block's positions, which rows the head reads, whether the
/// anchor row is normalized, and the rope. They exist because the head has no reference implementation to compare
/// against, only its acceptance; `MALLEUS_MTP_*` picks a variant and the default is the one the numbers chose.
fn knob(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The draft head's context: for every row of the step just run, the mix of its four target states through the
/// layer's `k_proj_ctx`/`v_proj_ctx`, the per-head `k_norm_ctx` and the rope. One K/V row per committed token and
/// draft layer, which is what the block attends to. Called from `run_mm` with the rows the step kept.
fn draft_commit(model: &Model, k: &mut Cache, pos: usize, rows: usize) -> Result<()> {
    let c = &model.cfg;
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let w = model.inkling2.as_ref().context("no inkling2 weights")?;
    let (Some(m), Some(g)) = (w.mtp.as_ref(), i.mtp.as_ref()) else {
        return Ok(());
    };
    let Some(d) = k.draft.as_mut() else { return Ok(()) };
    let h = c.hidden;
    let (nkv, hd) = (g.kv_heads, g.head_dim);
    let kvd = nkv * hd;
    let rope_mode = knob("MALLEUS_MTP_ROPE", 0) as i32;
    if pos != d.base_pos + d.len || d.len + rows > d.rows {
        // A step that carried more rows than the features hold (a whole prompt in one forward, as `eval` runs it):
        // the context restarts at the rows it has instead of pretending the missing ones are there.
        d.len = 0;
        d.base_pos = pos;
    }
    let rows = rows.min(d.rows - d.len);
    if rows == 0 {
        return Ok(());
    }
    let st = &model.stream;
    let pv: Vec<i32> = (0..rows).map(|r| (d.base_pos + d.len + r) as i32).collect();
    let pbuf = DeviceBuffer::from_slice(&pv)?;
    let mix = f32s(rows * h)?;
    let kb = f32s(rows * kvd)?;
    let vb = f32s(rows * kvd)?;
    for s in 0..g.layers {
        let l = &m.layers[s];
        let alpha = DeviceBuffer::from_slice(&m.alpha[s * 4..s * 4 + 4])?;
        model.k.get("inkling2_draft_mix").launch(
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&mix).into(),
                (&k.feats).into(),
                (&alpha).into(),
                (&l.dflare_norm).into(),
                rows.into(),
                0i32.into(),
                FEAT_ROWS.into(),
                h.into(),
                g.eps.into(),
            ],
        )?;
        if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
            draft_dump(&dir, &format!("commit_mix{s}"), st, &mix, rows, h)?;
            draft_dump(&dir, &format!("alpha{s}"), st, &alpha, 4, 1)?;
        }
        raw_gemm(model, st, &kb, &mix, &l.k_ctx, rows, kvd, h)?;
        if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
            draft_dump(&dir, &format!("commit_kraw{s}"), st, &kb, rows, kvd)?;
        }
        rms(model, st, &kb, &kb, &l.k_norm_ctx, rows * nkv, hd, g.eps)?;
        model.k.get("inkling2_rope_neox").launch(
            ((rows * nkv * hd / 2).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&kb).into(),
                (&pbuf).into(),
                rows.into(),
                nkv.into(),
                hd.into(),
                g.rope_theta.into(),
                rope_mode.into(),
            ],
        )?;
        raw_gemm(model, st, &vb, &mix, &l.v_ctx, rows, kvd, h)?;
        for (dst, dsc, src) in [(&d.k[s], &d.ks[s], &kb), (&d.v[s], &d.vs[s], &vb)] {
            model.k.get("inkling2_kv_store").launch(
                ((rows * nkv).div_ceil(8) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    dst.into(),
                    dsc.into(),
                    src.into(),
                    (d.base_pos + d.len).into(),
                    rows.into(),
                    d.rows.into(),
                    0i32.into(),
                    kvd.into(),
                    hd.into(),
                ],
            )?;
        }
    }
    d.len += rows;
    Ok(())
}

/// The folded Markov bias: `unembed @ hidden_out` as `[rank, vocab]` bf16 and `unembed @ hidden_out.bias` as
/// `[vocab]` f32. `lin()` computes `X @ W^T`, so the GEMM wants `hidden_out` transposed into the rows of its input;
/// 118 GFLOP on the quantized unembed, once per session.
fn markov_fold(model: &Model, d: &mut Draft) -> Result<()> {
    if d.fold.is_some() {
        return Ok(());
    }
    let c = &model.cfg;
    let w = model.inkling2.as_ref().context("no inkling2 weights")?;
    let m = w.mtp.as_ref().context("no MTP head")?;
    let g = c
        .inkling2
        .as_ref()
        .and_then(|i| i.mtp.as_ref())
        .context("no MTP config")?;
    let (h, rank, vocab) = (c.hidden, g.markov_rank, c.vocab);
    let st = &model.stream;
    let xt = f32s(rank * h)?;
    model.k.get("inkling2_bf16_transpose").launch(
        ((h * rank).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&xt).into(), (&m.markov.hidden_out.0).into(), h.into(), rank.into()],
    )?;
    let mut xr = f32s(rank * h)?;
    let mut xh = f32s(rank * h / 2)?;
    let mut xsc = f32s(rank * h.div_ceil(128))?;
    // [rank, vocab] f32, then the bf16 copy the walk reads.
    let fold32 = f32s(rank * vocab)?;
    lin(model, st, &mut xr, &mut xh, &mut xsc, &fold32, &xt, &w.unembed, rank)?;
    let fold = DeviceBuffer::alloc(rank * vocab * 2)?;
    model.k.get("inkling2_to_bf16").launch(
        ((rank * vocab).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&fold).into(), (&fold32).into(), (rank * vocab).into()],
    )?;
    // The same projection of `hidden_out.bias`: one row through the unembed.
    let hb = f32s(h)?;
    model.k.get("inkling2_bf16_rows").launch(
        (h.div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            (&hb).into(),
            (&m.markov.hidden_out.1).into(),
            0i32.into(),
            0i32.into(),
            h.into(),
        ],
    )?;
    let mut yr = f32s(h)?;
    let mut yh = f32s(h / 2)?;
    let mut ysc = f32s(h.div_ceil(128))?;
    let fbias = f32s(vocab)?;
    lin(model, st, &mut yr, &mut yh, &mut ysc, &fbias, &hb, &w.unembed, 1)?;
    st.synchronize()?;
    d.fold = Some((fold, fbias));
    Ok(())
}

/// Debug: the draft head's stages as files, so a reference implementation can be diffed against the engine stage by
/// stage instead of only through the acceptance (`MALLEUS_MTP_DUMP=<dir>`, see `scripts/inkling2_mtp_ref.py`): the
/// anchor's four target states, the draft cache, each layer's input norm, keys, values and attention output, the
/// block's hidden after every layer, its normalized form, the Markov walk's inputs and latent.
///
/// Every file is rewritten on every call, and a draft call is the only place they are all written: read them after a
/// run that ends with a draft, and never mix them with the commit-time dumps (`commit_*`), which come from the
/// forward *before* the draft and therefore carry different features.
fn draft_dump_half(dir: &str, name: &str, st: &Stream, buf: &DeviceBuffer, n: usize) -> Result<()> {
    st.synchronize()?;
    let v: Vec<u16> = buf.download::<u16>(n)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{dir}/{name}.f16"), bytes)?;
    Ok(())
}

fn draft_dump(dir: &str, name: &str, st: &Stream, buf: &DeviceBuffer, n: usize, h: usize) -> Result<()> {
    st.synchronize()?;
    let v = buf.download::<f32>(n * h)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{dir}/{name}.f32"), bytes)?;
    Ok(())
}

/// The drafts of the block that follows `anchor`: the block's rows are the anchor token's embedding and `mask_embed`
/// for the proposals, the context is the draft cache, and the head's base logits are the target's `unembed` on the
/// proposal rows. `n` of them at most (`DRAFT_ROWS - 1`).
pub fn draft(model: &Model, cache: &mut Cache, anchor: u32, n: usize) -> Result<Vec<u32>> {
    let c = &model.cfg;
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let w = model.inkling2.as_ref().context("no inkling2 weights")?;
    let (Some(m), Some(g)) = (w.mtp.as_ref(), i.mtp.as_ref()) else {
        return Ok(Vec::new());
    };
    let feat_rows = cache.feat_rows;
    let Cache { feats, draft: slot, .. } = cache;
    let Some(d) = slot.as_mut() else {
        return Ok(Vec::new());
    };
    let feats = &*feats;
    let (heads, nkv, hd, inter) = (g.heads, g.kv_heads, g.head_dim, g.inter);
    let (kvd, qd) = (nkv * hd, heads * hd);
    let h = c.hidden;
    let n = n.min(DRAFT_ROWS - 1).min(d.rows.saturating_sub(d.len + 1));
    if n == 0 || d.len == 0 {
        return Ok(Vec::new());
    }
    let rows = n + 1;
    let st = &model.stream;
    let base = d.len;
    let rope_mode = knob("MALLEUS_MTP_ROPE", 0) as i32;
    let pos_mode = knob("MALLEUS_MTP_POS", 0);
    let logit_rows = knob("MALLEUS_MTP_LOGITS", 0);
    let emb_norm = knob("MALLEUS_MTP_EMB", 0);
    let anchor_pos = (d.base_pos + base - 1) as i32 + pos_mode as i32;
    let scale = 1.0 / (hd as f32).sqrt();
    // Row 0 is the anchor token, at the anchor's own position; the proposals are the mask row, one position each.
    let x = f32s(rows * h)?;
    let idb = DeviceBuffer::from_slice(&[anchor as i32])?;
    model.k.get("embed_bf16").launch(
        (1, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&x).into(), (&w.embed).into(), (&idb).into(), h.into()],
    )?;
    let anchor_row = knob("MALLEUS_MTP_ANCHOR", 0) == 0;
    for r in if anchor_row { 1 } else { 0 }..rows {
        model.k.get("inkling2_bf16_rows").launch(
            (h.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&x).into(),
                (&m.mask_embed).into(),
                0i32.into(),
                (r * h).into(),
                h.into(),
            ],
        )?;
    }
    let pv: Vec<i32> = (0..rows).map(|r| anchor_pos + r as i32).collect();
    let pbuf = DeviceBuffer::from_slice(&pv)?;
    let cur = f32s(rows * h)?;
    cur.copy_from_shared(0, &x, 0, rows * h * 4, st)?;
    if emb_norm == 1 {
        // The anchor row normalized with the text stack's embedding norm (the other candidate for "the anchor
        // token's embedding").
        let n0 = f32s(h)?;
        rms(model, st, &n0, &cur, &w.embed_norm, 1, h, c.eps)?;
        cur.copy_from_shared(0, &n0, 0, h * 4, st)?;
    }
    let normed = f32s(rows * h)?;
    let proj = f32s(rows * h)?;
    let q = f32s(rows * qd)?;
    let kb = f32s(rows * kvd)?;
    let vb = f32s(rows * kvd)?;
    let attn = f32s(rows * qd)?;
    let act = f32s(rows * inter)?;
    let up = f32s(rows * inter)?;
    // The GEMM input's three forms (`gemm` checks them): the wide one is the down projection's 9216-value input.
    let kmax = inter.max(h).max(qd);
    let mut xr = f32s(rows * kmax)?;
    let mut xh = f32s(rows * kmax / 2)?;
    let mut xsc = f32s(rows * kmax.div_ceil(128))?;
    for s in 0..g.layers {
        let l = &m.layers[s];
        if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
            draft_dump(&dir, &format!("layer_in{s}"), st, &cur, rows, h)?;
        }
        rms(model, st, &normed, &cur, &l.input_norm, rows, h, g.eps)?;
        if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
            draft_dump(&dir, &format!("layer_xn{s}"), st, &normed, rows, h)?;
        }
        lin(model, st, &mut xr, &mut xh, &mut xsc, &q, &normed, &l.q, rows)?;
        lin(model, st, &mut xr, &mut xh, &mut xsc, &kb, &normed, &l.k, rows)?;
        lin(model, st, &mut xr, &mut xh, &mut xsc, &vb, &normed, &l.v, rows)?;
        rms(model, st, &q, &q, &l.q_norm, rows * heads, hd, g.eps)?;
        rms(model, st, &kb, &kb, &l.k_norm, rows * nkv, hd, g.eps)?;
        for (dst, src, nh) in [(&q, &q, heads), (&kb, &kb, nkv)] {
            let _ = dst;
            model.k.get("inkling2_rope_neox").launch(
                ((rows * nh * hd / 2).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    src.into(),
                    (&pbuf).into(),
                    rows.into(),
                    nh.into(),
                    hd.into(),
                    g.rope_theta.into(),
                    rope_mode.into(),
                ],
            )?;
        }
        // The block's own keys go into the same cache the context lives in, after it: row `r` sees every committed
        // token and the rows of the block before it.
        for (dst, dsc, src) in [(&d.k[s], &d.ks[s], &kb), (&d.v[s], &d.vs[s], &vb)] {
            model.k.get("inkling2_kv_store").launch(
                ((rows * nkv).div_ceil(8) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    dst.into(),
                    dsc.into(),
                    src.into(),
                    base.into(),
                    rows.into(),
                    d.rows.into(),
                    0i32.into(),
                    kvd.into(),
                    hd.into(),
                ],
            )?;
        }
        let local = g.local.contains(&s);
        for r in 0..rows {
            let all = base + r + 1;
            let (k0, nk) = if local && all > g.window {
                (all - g.window, g.window)
            } else {
                (0, all)
            };
            model.k.get("inkling2_attn_dec").launch(
                (heads as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    Arg::U64(attn.addr() + (r * qd * 4) as u64),
                    Arg::U64(q.addr() + (r * qd * 4) as u64),
                    (&d.k[s]).into(),
                    (&d.ks[s]).into(),
                    (&d.v[s]).into(),
                    (&d.vs[s]).into(),
                    Arg::Ptr(0),
                    Arg::Ptr(0),
                    Arg::Ptr(0),
                    k0.into(),
                    nk.into(),
                    heads.into(),
                    nkv.into(),
                    hd.into(),
                    0i32.into(),
                    0i32.into(),
                    scale.into(),
                ],
            )?;
        }
        if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
            draft_dump(&dir, &format!("blk_attn{s}"), st, &attn, rows, qd)?;
            draft_dump(&dir, &format!("blk_q{s}"), st, &q, rows, qd)?;
            draft_dump(&dir, &format!("blk_k{s}"), st, &kb, rows, kvd)?;
            draft_dump(&dir, &format!("blk_v{s}"), st, &vb, rows, kvd)?;
        }
        lin(model, st, &mut xr, &mut xh, &mut xsc, &proj, &attn, &l.o, rows)?;
        if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
            draft_dump(&dir, &format!("blk_o{s}"), st, &proj, rows, h)?;
        }
        add(model, st, &cur, &proj, rows * h)?;
        rms(model, st, &normed, &cur, &l.post_attention_norm, rows, h, g.eps)?;
        lin(model, st, &mut xr, &mut xh, &mut xsc, &act, &normed, &l.gate, rows)?;
        lin(model, st, &mut xr, &mut xh, &mut xsc, &up, &normed, &l.up, rows)?;
        model.ew(
            st,
            "silu_mul",
            rows * inter,
            &[(&act).into(), (&act).into(), (&up).into(), (rows * inter).into()],
        )?;
        lin(model, st, &mut xr, &mut xh, &mut xsc, &proj, &act, &l.down, rows)?;
        add(model, st, &cur, &proj, rows * h)?;
        if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
            draft_dump(&dir, &format!("block_h{s}"), st, &cur, rows, h)?;
            draft_dump(&dir, &format!("ctx_k{s}"), st, &kb, rows, kvd)?;
            draft_dump(&dir, &format!("ctx_v{s}"), st, &vb, rows, kvd)?;
        }
    }
    rms(model, st, &normed, &cur, &m.norm, rows, h, g.eps)?;
    if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
        draft_dump(&dir, "normed", st, &normed, rows, h)?;
    }
    let r0 = if logit_rows == 0 && anchor_row { 1 } else { 0 };
    let head_in = f32s(n * h)?;
    head_in.copy_from_shared(0, &normed, r0 * h * 4, n * h * 4, st)?;
    let logits = f32s(n * c.vocab)?;
    lin(model, st, &mut xr, &mut xh, &mut xsc, &logits, &head_in, &w.unembed, n)?;

    // The Markov half: the anchor's own mixture (`alpha[-1]` with `markov_head.dflare_norm`) is the walk's context,
    // and the walk runs over the rows in order because a row's bias conditions on the token before it — the anchor
    // for the first, the proposal before it after that.
    if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
        let anchor_feats = f32s(4 * h)?;
        for j in 0..4 {
            anchor_feats.copy_from_shared(
                (j * h) * 4,
                feats,
                (j * FEAT_ROWS * h + (feat_rows.max(1) - 1) * h) * 4,
                h * 4,
                st,
            )?;
        }
        draft_dump(&dir, "anchor_feats", st, &anchor_feats, 4, h)?;
        draft_dump(&dir, "feats_planes", st, feats, 4 * FEAT_ROWS, h)?;
        for s in 0..g.layers {
            draft_dump_half(&dir, &format!("ctx_cache_k{s}"), st, &d.k[s], base * kvd)?;
            draft_dump_half(&dir, &format!("ctx_cache_v{s}"), st, &d.v[s], base * kvd)?;
        }
        let meta = format!(
            "{{\"anchor\": {anchor}, \"pos\": {}, \"rows\": {rows}, \"feat_rows\": {feat_rows}, \"h\": {h}, \
             \"heads\": {heads}, \"kv_heads\": {nkv}, \"head_dim\": {hd}, \"inter\": {inter}, \"vocab\": {},\
             \"rank\": {}, \"theta\": {}, \"eps\": {}, \"window\": {}}}",
            d.base_pos + base - 1,
            c.vocab,
            g.markov_rank,
            g.rope_theta,
            g.eps,
            g.window
        );
        std::fs::write(format!("{dir}/mtp.json"), meta)?;
    }
    let rank = g.markov_rank;
    let hid_mode = knob("MALLEUS_MTP_HID", 0);
    let prev_mode = knob("MALLEUS_MTP_PREV", 0);
    let frnn = f32s(h)?;
    let alpha_m = DeviceBuffer::from_slice(&m.alpha[12..16])?;
    let arow = (feat_rows.max(1) - 1) as i32;
    model.k.get("inkling2_draft_mix").launch(
        (1, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            (&frnn).into(),
            feats.into(),
            (&alpha_m).into(),
            (&m.markov.dflare_norm).into(),
            1i32.into(),
            arow.into(),
            FEAT_ROWS.into(),
            h.into(),
            g.eps.into(),
        ],
    )?;
    // The walk starts from a zero state for every block (`markov_head.init_state`).
    d.state.memset(0, st)?;
    markov_fold(model, d)?;
    let (fold, fbias) = d.fold.as_ref().context("markov fold")?;
    let walk = f32s(rank + rank + h + h)?;
    if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
        draft_dump(&dir, "frnn", st, &frnn, 1, h)?;
    }
    // The walk's context: the anchor's Markov mixture, or its normalized hidden state.
    if knob("MALLEUS_MTP_FRNN", 0) == 1 {
        walk.copy_from_shared((2 * rank + h) * 4, &normed, (rows - 1) * h * 4, h * 4, st)?;
    } else {
        walk.copy_from_shared((2 * rank + h) * 4, &frnn, 0, h * 4, st)?;
    }
    let markov = knob("MALLEUS_MTP_MARKOV", 1) == 1;
    let mut drafts: Vec<u32> = Vec::with_capacity(n);
    let mut prev = anchor;
    for r in 0..n {
        if !markov {
            let out = DeviceBuffer::alloc(4)?;
            let row = Arg::U64(logits.addr() + (r * c.vocab * 4) as u64);
            model.k.get("inkling2_softcap").launch(
                (c.vocab.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[row, row, c.vocab.into(), i.logit_softcap.into()],
            )?;
            model.k.get("logits_argmax").launch(
                (1, 1, 1),
                (1024, 1, 1),
                0,
                st,
                &[(&out).into(), row, c.vocab.into()],
            )?;
            st.synchronize()?;
            let got: Vec<i32> = out.download(1)?;
            prev = got[0] as u32;
            drafts.push(prev);
            continue;
        }
        // [state | markov_w1[prev] | hidden | f_rnn]
        walk.copy_from_shared(0, &d.state, 0, rank * 4, st)?;
        model.k.get("inkling2_bf16_rows").launch(
            (rank.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&walk).into(),
                (&m.markov.w1).into(),
                ((prev as usize * rank) as i32).into(),
                (rank as i32).into(),
                rank.into(),
            ],
        )?;
        let hid = if hid_mode == 1 { &cur } else { &normed };
        walk.copy_from_shared(2 * rank * 4, hid, (r0 + r) * h * 4, h * 4, st)?;
        if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
            draft_dump(&dir, &format!("walk{r}"), st, &walk, rank + rank + h + h, 1)?;
        }
        model.k.get("inkling2_markov_step").launch(
            (1, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&d.state).into(),
                (&d.latent).into(),
                (&walk).into(),
                (&m.markov.rnn_proj.0).into(),
                (&m.markov.rnn_proj.1).into(),
                rank.into(),
                (rank + rank + h + h).into(),
            ],
        )?;
        if let Ok(dir) = std::env::var("MALLEUS_MTP_DUMP") {
            draft_dump(&dir, &format!("latent{r}"), st, &d.latent, rank, 1)?;
            draft_dump(&dir, &format!("state{r}"), st, &d.state, rank, 1)?;
        }
        let row = Arg::U64(logits.addr() + (r * c.vocab * 4) as u64);
        model.k.get("inkling2_markov_bias").launch(
            (c.vocab.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                row,
                fold.into(),
                (&d.latent).into(),
                fbias.into(),
                c.vocab.into(),
                rank.into(),
            ],
        )?;
        model.k.get("inkling2_softcap").launch(
            (c.vocab.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[row, row, c.vocab.into(), i.logit_softcap.into()],
        )?;
        let out = DeviceBuffer::alloc(4)?;
        model
            .k
            .get("logits_argmax")
            .launch((1, 1, 1), (1024, 1, 1), 0, st, &[(&out).into(), row, c.vocab.into()])?;
        st.synchronize()?;
        let got: Vec<i32> = out.download(1)?;
        prev = if prev_mode == 1 { anchor } else { got[0] as u32 };
        drafts.push(prev);
    }
    Ok(drafts)
}

/// Drops the last `n` tokens of the preceding multi-row forward: the session's keys and history, and the draft head's
/// context with them.
pub fn rollback(model: &Model, cache: &mut Cache, n: usize) -> Result<()> {
    let _ = model;
    let Some(snap) = cache.snap.take() else {
        return Ok(());
    };
    let t = snap.rows;
    ensure!(n < t, "rollback of {n} tokens after a {t}-row verify");
    let keep = t - n;
    let st = Stream::null();
    for (s, hist) in cache.slots.iter().zip(&snap.hist) {
        for (i, past) in [&s.past_k, &s.past_v, &s.past_y, &s.past_moe].into_iter().enumerate() {
            if keep == 0 {
                past.memset(0, &st)?;
            } else {
                let w = past.len() / 12;
                past.copy_from_shared(0, &hist[i], (keep - 1) * 3 * w * 4, 3 * w * 4, &st)?;
            }
        }
    }
    st.synchronize()?;
    cache.len -= n;
    // The rows of the last step that were *not* committed stop being features: the draft head reads the anchor as
    // `feats[feat_rows - 1]`, and after a partly accepted verify that is the last committed row, not the last row of
    // the verify. This is the bug the stage-by-stage diff found.
    cache.feat_rows = cache.feat_rows.saturating_sub(n);
    if let Some(d) = cache.draft.as_mut() {
        d.len = d.len.saturating_sub(n);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// The map against the real file: `MALLEUS_INKLING2_FILE=/path/to/Inkling-2-Mini-mls4.malleus
    /// cargo test -p malleus-model inkling2 -- --ignored`. The file is 15 GB, so it is not part of the suite.
    #[test]
    #[ignore = "needs MALLEUS_INKLING2_FILE"]
    fn the_map_covers_the_file() {
        let path = std::env::var("MALLEUS_INKLING2_FILE").expect("MALLEUS_INKLING2_FILE");
        let g = malleus_gguf::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
        let cfg_json = g.meta_str("openmalleus.hf_config").unwrap();
        let c = Config::from_json(cfg_json, None).unwrap();
        let lines = check(&g, &c).unwrap();
        for l in &lines {
            println!("{l}");
        }
        assert!(lines[0].starts_with("1384 of 1384"), "{}", lines[0]);
        assert_eq!(lines.len(), 4, "{lines:?}");
    }

    /// A three-layer stack whose loop repeats two of them: the slot plan is what `session_bytes` prices and what
    /// `Cache::new` allocates (issue #110).
    const TINY: &str = r#"{
        "model_type": "inkling2",
        "text_config": {
            "model_type": "inkling2", "hidden_size": 128, "num_hidden_layers": 3,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "d_rel": 2, "rel_extent": 4, "log_scaling_n_floor": 8, "log_scaling_alpha": 0.1,
            "qk_l2_clip_threshold": 1.0, "rel_l2_clip_threshold": 1.0,
            "n_layers_per_attn_res_block": 1, "sconv_kernel_size": 4, "sliding_window_size": 4,
            "n_routed_experts": 4, "num_experts_per_tok": 1, "n_shared_experts": 0,
            "intermediate_size": 128, "route_scale": 1.0, "mlp_gate_softcap": 1.0,
            "mlp_up_softcap": 1.0, "final_logit_softcapping": 1.0,
            "unpadded_vocab_size": 64, "vocab_size": 64, "rms_norm_eps": 1e-6,
            "local_layer_ids": [0, 1], "scoring_func": "sigmoid", "use_gate_bias": false,
            "loop_config": {"core_start": 1, "core_end": 3, "num_passes": 2, "gate_rank": 8}
        }
    }"#;

    #[test]
    fn the_slot_plan_is_what_a_session_is_priced_with() {
        let c = Config::from_json(TINY, None).unwrap();
        let i = c.inkling2.as_ref().unwrap();
        // Two physical layers (1 and 2) in the loop, run twice: 3 + 2 slots, the extra ones being the loop's second
        // pass over layer 1 (sliding) and layer 2 (full).
        assert_eq!(n_slots(i, c.n_layers), 5);
        assert_eq!(
            (0..5).map(|s| slot_layer(i, c.n_layers, s)).collect::<Vec<_>>(),
            [0, 1, 2, 1, 2]
        );
        // A sliding slot is a ring of window + PREFILL_CHUNK + DRAFT_ROWS + 1 rows, a full one the whole context; a
        // context shorter than the ring makes the ring the whole context, and then it is not a ring at all.
        let ring = 4 + PREFILL_CHUNK + DRAFT_ROWS + 1;
        assert_eq!(slot_shape(i, false, 1024), (ring, ring));
        assert_eq!(slot_shape(i, true, 1024), (1024, 0));
        assert_eq!(slot_shape(i, false, 128), (128, 0));
        // Priced by hand: 3 rings and 2 whole contexts of keys and values (kvd = 4, fp16, K and V), the four
        // three-row histories of each slot, the bias row, tau and the MTP rows.
        let kvd = 4u64;
        let hist = 2 * (3 * kvd * 4) + 2 * (3 * 128 * 4);
        let want = 3 * ring as u64 * kvd * 2 * 2
            + 2 * 1024 * kvd * 2 * 2
            + 5 * hist
            + 2 * 1024 * 4
            + 4
            + 4 * DRAFT_ROWS as u64 * 128 * 4;
        assert_eq!(session_bytes(&c, 1024), want);
    }
}
