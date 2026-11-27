//! DFlash speculative draft (issue #12): a small transformer that proposes a **block** of tokens for a target model
//! to verify in one pass, as an alternative to the MTP head that some checkpoints carry inside themselves.
//!
//! What it is, from the checkpoint (`Ornith-1.5-35B-A3B-DFlash`) and the only reference implementation that exists
//! (SGLang ships it, the checkpoint does not: `auto_map` names a module that is not in the repository):
//!
//! - `fc` projects the **target's hidden states at `dflash_config.target_layer_ids`**, concatenated, into the draft's
//!   hidden size (8 layers x 2048 -> 2048 for Ornith), then `hidden_norm`. That projection is the whole context the
//!   draft ever sees: it never runs on the prompt, its context KV is built from those features.
//! - 6 transformer layers (GQA, qk-norm, sliding window on most of them) run over the **block** of positions, where
//!   position 0 is the last known token and positions 1.. are `mask_token_id`: the block's hidden states become the
//!   proposals, scored with the **target's** `lm_head`.
//! - No embedding and no `lm_head` of its own: it borrows the target's, so it is not a standalone model.
//!
//! This module loads the checkpoint. The forward, the capture of the target's layer features and the drafting loop
//! are the next increments: the plan, and the reason the block has to stay within the target's verify width
//! (`GEMV_MAX_T = 6` today, so a block of 7), are in the issue.

use crate::kernels::Kernels;
use crate::weights::{Linear, Weights};
use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::DeviceBuffer;
use serde_json::Value;

/// `dflash_config` plus the transformer geometry, from the draft's `config.json`.
#[derive(Clone, Debug)]
pub struct DflashConfig {
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
    /// Positions per block: 0 is the anchor, 1.. are the proposals.
    pub block_size: usize,
    /// Token the unknown positions of the block are filled with.
    pub mask_token_id: u32,
    /// Target layers whose hidden states `fc` reads, in order.
    pub target_layer_ids: Vec<usize>,
    /// Sliding window of the layers that use it, and how many of them do.
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
}

impl DflashConfig {
    pub fn from_json(config: &str) -> Result<Self> {
        let c: Value = serde_json::from_str(config)?;
        let u = |k: &str| -> Result<usize> {
            c[k].as_u64().with_context(|| format!("draft config.json has no {k}"))?;
            Ok(c[k].as_u64().unwrap_or(0) as usize)
        };
        let d = &c["dflash_config"];
        let du = |k: &str| -> Result<usize> {
            d[k].as_u64()
                .with_context(|| format!("draft config.json has no dflash_config.{k}"))?;
            Ok(d[k].as_u64().unwrap_or(0) as usize)
        };
        let rope_theta = c["rope_parameters"]["rope_theta"]
            .as_f64()
            .or_else(|| c["rope_theta"].as_f64())
            .unwrap_or(10_000.0) as f32;
        let layer_types: Vec<String> = c["layer_types"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let hidden = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        let head_dim = c["head_dim"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(hidden / heads.max(1));
        let cfg = DflashConfig {
            hidden,
            inter: u("intermediate_size")?,
            layers: u("num_hidden_layers")?,
            heads,
            kv_heads: u("num_key_value_heads")?,
            head_dim,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta,
            block_size: du("block_size")?,
            mask_token_id: du("mask_token_id")? as u32,
            target_layer_ids: d["target_layer_ids"]
                .as_array()
                .with_context(|| "draft config.json has no dflash_config.target_layer_ids")?
                .iter()
                .filter_map(|x| x.as_u64().map(|v| v as usize))
                .collect(),
            sliding_window: c["sliding_window"].as_u64().unwrap_or(0) as usize,
            layer_types,
        };
        ensure!(cfg.layers > 0 && cfg.hidden > 0, "draft config has no layers");
        ensure!(
            !cfg.target_layer_ids.is_empty(),
            "draft config has no dflash_config.target_layer_ids: the draft would have no context to read"
        );
        ensure!(
            cfg.heads.is_multiple_of(cfg.kv_heads.max(1)),
            "draft config: {} heads is not a multiple of {} kv heads",
            cfg.heads,
            cfg.kv_heads
        );
        Ok(cfg)
    }

    /// Features `fc` reads: one hidden vector per target layer.
    pub fn num_context_features(&self) -> usize {
        self.target_layer_ids.len()
    }

    /// Tokens a block proposes, i.e. the width the target has to verify.
    pub fn proposals(&self) -> usize {
        self.block_size.saturating_sub(1)
    }

    pub fn layer_is_sliding(&self, layer: usize) -> bool {
        self.layer_types.get(layer).is_some_and(|t| t.starts_with("sliding"))
    }
}

pub struct DflashLayer {
    pub input_norm: DeviceBuffer,
    pub post_norm: DeviceBuffer,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub q_norm: DeviceBuffer,
    pub k_norm: DeviceBuffer,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

pub struct DflashDraft {
    pub cfg: DflashConfig,
    /// Target features -> draft hidden.
    pub fc: Linear,
    pub hidden_norm: DeviceBuffer,
    pub layers: Vec<DflashLayer>,
    pub norm: DeviceBuffer,
    pub weight_bytes: usize,
}

/// One MLS4 tensor read back as a `Linear` with no prefill copy: a draft never prefills more than a block.
fn linear(src: &dyn TensorSource, name: &str, n: usize, k: usize) -> Result<Linear> {
    let wn = format!("{name}.weight");
    let info = src.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
    ensure!(
        info.dtype == DType::Mls4,
        "{wn}: expected MLS4 (convert the draft without --prefill-int8 and with the mls4 family), got {:?}",
        info.dtype
    );
    let shape: usize = info.shape.iter().product();
    ensure!(
        info.shape.len() == 2 && info.shape[0] == n && info.shape[1] == k,
        "{wn}: expected [{n}, {k}], got {:?}",
        info.shape
    );
    // MLS4 stores an fp16 scale per 128-group row (the MLS8 prefill copy is the one with f32 per-row scales).
    let sinfo = src
        .info(&format!("{wn}.scales"))
        .with_context(|| format!("missing tensor {wn}.scales"))?;
    let groups = k / malleus_gguf::quant::BLOCK;
    ensure!(
        sinfo.dtype == DType::F16 && sinfo.shape == vec![n, groups],
        "{wn}.scales: expected f16 [{n}, {groups}], got {:?} {:?}",
        sinfo.dtype,
        sinfo.shape
    );
    let sc = src.read_all(&format!("{wn}.scales"))?;
    ensure!(shape > 0, "{wn}: empty");
    Ok(Linear {
        w: Weights::Mls4 {
            q: DeviceBuffer::from_slice(&src.read_all(&wn)?)?,
            sc: DeviceBuffer::from_slice(&sc)?,
        },
        prefill: None,
        n,
        k,
    })
}

fn norm(src: &dyn TensorSource, name: &str, n: usize) -> Result<DeviceBuffer> {
    let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
    ensure!(
        info.dtype == DType::BF16 && info.shape == vec![n],
        "{name}: expected bf16 [{n}], got {:?} {:?}",
        info.dtype,
        info.shape
    );
    Ok(DeviceBuffer::from_slice(&src.read_all(name)?)?)
}

impl DflashDraft {
    /// Loads the draft from a converted `.malleus` file. `kernels` is not used yet: the forward is the next
    /// increment, and keeping it in the signature means the caller does not change when it arrives.
    pub fn load(config: &str, src: &dyn TensorSource, _kernels: &Kernels) -> Result<Self> {
        let cfg = DflashConfig::from_json(config)?;
        let h = cfg.hidden;
        let fc = linear(src, "fc", h, cfg.num_context_features() * h)?;
        let hidden_norm = norm(src, "hidden_norm.weight", h)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("layers.{i}");
            layers.push(DflashLayer {
                input_norm: norm(src, &format!("{p}.input_layernorm.weight"), h)?,
                post_norm: norm(src, &format!("{p}.post_attention_layernorm.weight"), h)?,
                q: linear(src, &format!("{p}.self_attn.q_proj"), cfg.heads * cfg.head_dim, h)?,
                k: linear(src, &format!("{p}.self_attn.k_proj"), cfg.kv_heads * cfg.head_dim, h)?,
                v: linear(src, &format!("{p}.self_attn.v_proj"), cfg.kv_heads * cfg.head_dim, h)?,
                o: linear(src, &format!("{p}.self_attn.o_proj"), h, cfg.heads * cfg.head_dim)?,
                q_norm: norm(src, &format!("{p}.self_attn.q_norm.weight"), cfg.head_dim)?,
                k_norm: norm(src, &format!("{p}.self_attn.k_norm.weight"), cfg.head_dim)?,
                gate: linear(src, &format!("{p}.mlp.gate_proj"), cfg.inter, h)?,
                up: linear(src, &format!("{p}.mlp.up_proj"), cfg.inter, h)?,
                down: linear(src, &format!("{p}.mlp.down_proj"), h, cfg.inter)?,
            });
        }
        let norm_final = norm(src, "norm.weight", h)?;
        let mut weight_bytes = 0usize;
        let mut count = |l: &Linear| weight_bytes += weights_bytes(&l.w);
        count(&fc);
        for l in &layers {
            for lin in [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down] {
                count(lin);
            }
        }
        weight_bytes += (2 + cfg.layers * 4) * h * 2 + cfg.layers * 2 * cfg.head_dim * 2;
        if cfg.proposals() > verify_width() {
            eprintln!(
                "note: the block proposes {} tokens, the target verifies {} drafts per pass today: a usable block is \
                 {} positions (1 anchor + {})",
                cfg.proposals(),
                verify_width(),
                verify_width() + 1,
                verify_width()
            );
        }
        Ok(DflashDraft {
            cfg,
            fc,
            hidden_norm,
            layers,
            norm: norm_final,
            weight_bytes,
        })
    }

    /// One line per field, for `malleus draft-info`: the numbers a port is checked against.
    pub fn describe(&self) -> Vec<String> {
        let c = &self.cfg;
        let mut out = vec![
            format!(
                "dflash draft: {} layers, hidden {}, {} q / {} kv heads of {}, inter {}, {:.1} MB of weights",
                c.layers,
                c.hidden,
                c.heads,
                c.kv_heads,
                c.head_dim,
                c.inter,
                self.weight_bytes as f64 / 1e6
            ),
            format!(
                "  context: fc reads {} target layers {:?} -> {} features",
                c.num_context_features(),
                c.target_layer_ids,
                c.num_context_features() * c.hidden
            ),
            format!(
                "  block: {} positions (1 anchor + {} proposals), mask token {}, sliding window {} on {} of {} layers",
                c.block_size,
                c.proposals(),
                c.mask_token_id,
                c.sliding_window,
                (0..c.layers).filter(|&i| c.layer_is_sliding(i)).count(),
                c.layers
            ),
            format!(
                "  rope theta {}, eps {:e}; embedding and lm_head are the target's",
                c.rope_theta, c.eps
            ),
        ];
        if c.proposals() > verify_width() {
            out.push(format!(
                "  warning: {} proposals against a verify width of {} drafts: the block has to be {} positions (see \
                 issue #12)",
                c.proposals(),
                verify_width(),
                verify_width() + 1
            ));
        }
        out
    }
}

/// Bytes of one weight tensor, for the size report.
fn weights_bytes(w: &Weights) -> usize {
    match w {
        // MLS1 is a DeepSeek format: a draft head does not carry one.
        Weights::Mls1 { .. } => 0,
        Weights::Bf16(b) => b.len(),
        Weights::Mls2 { q, sc } | Weights::Mls4 { q, sc } | Weights::Mls3 { q, sc } | Weights::Mls8 { q, sc } => {
            q.len() + sc.len()
        }
    }
}

/// Drafts the target's verify path accepts in one pass: `GEMV_MAX_T - 1` (the kernel's register budget, and the
/// measured that widening it is a loss). A block of `verify_width() + 1` positions fits it.
fn verify_width() -> usize {
    crate::weights::GEMV_MAX_T - 1
}

/// Fails loudly on a checkpoint that is not a DFlash draft, instead of half-loading it.
pub fn ensure_is_draft(config: &str) -> Result<()> {
    let c: Value = serde_json::from_str(config)?;
    if c["dflash_config"].is_null() {
        bail!("not a DFlash draft: config.json has no dflash_config");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORNITH: &str = r#"{
        "architectures": ["DFlashDraftModel"],
        "attention_bias": false,
        "dflash_config": {"block_size": 16, "mask_token_id": 248077,
                          "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37]},
        "head_dim": 128,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention"],
        "max_window_layers": 6,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 6,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
        "sliding_window": 4096,
        "vocab_size": 248320
    }"#;

    #[test]
    fn ornith_draft_geometry() {
        let c = DflashConfig::from_json(ORNITH).unwrap();
        assert_eq!(
            (c.layers, c.hidden, c.heads, c.kv_heads, c.head_dim),
            (6, 2048, 32, 8, 128)
        );
        assert_eq!(c.num_context_features(), 8);
        assert_eq!(c.num_context_features() * c.hidden, 16384, "fc input");
        assert_eq!(c.block_size, 16);
        assert_eq!(c.proposals(), 15);
        assert_eq!(c.mask_token_id, 248077);
        assert_eq!(c.sliding_window, 4096);
        assert!(c.layer_is_sliding(0) && !c.layer_is_sliding(5));
        assert_eq!(c.rope_theta, 10_000_000.0);
    }

    #[test]
    fn a_checkpoint_without_dflash_config_is_rejected() {
        assert!(ensure_is_draft(r#"{"dflash_config": {"block_size": 8}}"#).is_ok());
        assert!(ensure_is_draft(r#"{"model_type": "qwen3_5_moe"}"#).is_err());
        assert!(DflashConfig::from_json(r#"{"hidden_size": 2048}"#).is_err());
    }
}
