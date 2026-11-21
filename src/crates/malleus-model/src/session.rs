//! Per-sequence state (KV, conv and recurrent state, indexer pools, MTP head), the scratch of one
//! forward, and the snapshots a scheduler pools.

use anyhow::{Result, ensure};
use malleus_hip::{Arg, DeviceBuffer, Stream};

/// Per-sequence state: KV cache for attention layers, conv + recurrent state for DeltaNet layers.
pub struct Session {
    pub max_ctx: usize,
    pub len: usize,
    pub next_pos: i64,
    pub(crate) kv: Vec<(DeviceBuffer, DeviceBuffer)>,
    pub(crate) conv: Vec<DeviceBuffer>,
    pub(crate) rec: Vec<DeviceBuffer>,
    /// Per-split partial results of the split-key attention (inputs shorter than ATTN_WMMA_MIN tokens).
    pub(crate) attn_part: DeviceBuffer,
    /// MLA split partials for Ling, sized for GEMV_MAX_T tokens; None for models without MLA.
    pub(crate) mla_part: Option<DeviceBuffer>,
    /// DeltaNet states after each token but the last of a verify step, per layer (rollback).
    pub(crate) conv_snap: Vec<DeviceBuffer>,
    /// Per DeltaNet layer, two buffers of rank-1 update records (verify writes one while reading the other).
    pub(crate) dn_rec: Vec<[DeviceBuffer; 2]>,
    /// Accepted records not yet applied to the recurrent state, and the buffer holding them.
    pub(crate) dn_np: usize,
    pub(crate) dn_par: usize,
    /// Tokens of the last verify step (0: no rollback possible).
    pub(crate) snap_t: usize,
    /// MTP head: its KV cache, rows written, and hidden states (with positions) waiting for their next token.
    pub(crate) mtp_kv: Vec<(DeviceBuffer, DeviceBuffer)>,
    pub(crate) mtp_len: usize,
    pub(crate) mtp_h: Option<DeviceBuffer>,
    pub(crate) mtp_pos: Vec<[i32; 3]>,
    /// Output hidden state and position of the last MTP row (input of a chained draft).
    pub(crate) mtp_out: Option<DeviceBuffer>,
    pub(crate) mtp_out_pos: [i32; 3],
    /// Rows of chained drafts in the MTP KV cache, dropped before the next real rows.
    pub(crate) mtp_chain: usize,
    /// PLE: normalized conv inputs of the last rows and the tokens before the next input.
    pub(crate) ple_state: Option<DeviceBuffer>,
    pub(crate) ngram_ctx: Vec<u32>,
    /// QSA indexer of each attention layer (pooled block keys fp16, raw keys and positions of the last incomplete
    /// block), then the scores of a group of queries and the selection bitsets of the last input (with the unions of
    /// 64 queries), grown on demand.
    pub(crate) idx_kb: Vec<DeviceBuffer>,
    pub(crate) idx_tk: Vec<DeviceBuffer>,
    pub(crate) idx_tp: Vec<DeviceBuffer>,
    pub(crate) idx_scores: Option<DeviceBuffer>,
    pub(crate) idx_sel: Option<(DeviceBuffer, DeviceBuffer, usize)>,
    /// Last selection bitset of each indexer layer, kept for the decode reuse (one row per layer).
    pub(crate) idx_sel_cache: Vec<DeviceBuffer>,
    /// Host-side validity: the context length at which `idx_sel_cache` was written (0 = invalid).
    pub(crate) idx_cache_len: usize,
    /// Indexer of the MTP attention layer (pooled keys, tail keys, tail positions).
    pub(crate) mtp_idx: Option<[DeviceBuffer; 3]>,
    /// Short inputs with sparse attention: the selected blocks of each query as a list (and their counts).
    pub(crate) idx_list: Option<(DeviceBuffer, usize)>,
    /// Speculative verify with PLE: the conv state before the step and the normalized rows of its tokens (rollback),
    /// and the n-gram context before the step with the step's tokens.
    pub(crate) ple_snap: Option<DeviceBuffer>,
    pub(crate) ngram_prev: Vec<u32>,
    /// Inkling-2's decode state: one KV slot per layer and pass, the histories of its short convolutions, the bias
    /// row of the current query. Its own graph, not the `kv`/`conv` of the carried families.
    pub(crate) inkling2: Option<Box<crate::inkling2::Cache>>,
    /// Hy4's DFly2: what the next draft block reads from the target's last call.
    pub(crate) hy4_spec: Option<Box<crate::hy4::SpecState>>,
    pub(crate) hy4_draft: crate::hy4::DraftStats,
    /// Hy4 compressed indexer: per layer (None for the sliding layers) the token keys [`max_ctx`, `index_dim`] fp16
    /// and the pooled group keys [`max_ctx / pool`, `index_dim`] fp16, the groups already pooled (host side: a
    /// closed group never changes again), and the 3-D position of every token slot, shared by the layers.
    pub(crate) ix_tk: Vec<Option<DeviceBuffer>>,
    pub(crate) ix_gk: Vec<Option<DeviceBuffer>>,
    pub(crate) ix_pooled: Vec<usize>,
    pub(crate) ix_pos: Option<DeviceBuffer>,
}

/// Session state at a token boundary besides the KV caches (`Model::snapshot`, `Model::restore`).
pub struct Snapshot {
    pub len: usize,
    pub(crate) next_pos: i64,
    pub(crate) conv: Vec<DeviceBuffer>,
    pub(crate) rec: Vec<DeviceBuffer>,
    pub(crate) ple_state: Option<DeviceBuffer>,
    pub(crate) ngram_ctx: Vec<u32>,
    pub(crate) idx_tk: Vec<DeviceBuffer>,
    pub(crate) idx_tp: Vec<DeviceBuffer>,
    pub(crate) mtp_len: usize,
    pub(crate) mtp_h: Option<DeviceBuffer>,
    pub(crate) mtp_pos: Vec<[i32; 3]>,
    pub(crate) mtp_idx: Option<[DeviceBuffer; 2]>,
}

impl Snapshot {
    /// GPU bytes held.
    pub fn bytes(&self) -> usize {
        let v = |b: &[DeviceBuffer]| b.iter().map(|x| x.len()).sum::<usize>();
        v(&self.conv) + v(&self.rec) + self.ple_state.as_ref().map_or(0, |b| b.len()) + v(&self.idx_tk)
    }
}

/// Copies one state buffer, replacing the destination allocation when the size changed. A pooled snapshot keeps
/// its buffers from one request to the next: on these APUs the GPU memory *is* system RAM, and `hipMalloc` of a
/// buffer the size of one layer's state has to find pages while the weights hold every other page in sight.
pub(crate) fn copy_state(dst: &mut DeviceBuffer, src: &DeviceBuffer, st: &Stream) -> Result<()> {
    if dst.len() != src.len() {
        *dst = DeviceBuffer::alloc(src.len())?;
    }
    dst.copy_from(0, src, 0, src.len(), st)?;
    Ok(())
}

pub(crate) fn copy_states(dst: &mut Vec<DeviceBuffer>, src: &[DeviceBuffer], st: &Stream) -> Result<()> {
    if dst.len() != src.len() {
        *dst = src
            .iter()
            .map(|b| -> Result<DeviceBuffer> { Ok(DeviceBuffer::alloc(b.len())?) })
            .collect::<Result<_>>()?;
    }
    dst.iter_mut().zip(src).try_for_each(|(d, b)| copy_state(d, b, st))
}

pub(crate) fn copy_state_opt(dst: &mut Option<DeviceBuffer>, src: &Option<DeviceBuffer>, st: &Stream) -> Result<()> {
    match src {
        None => {
            *dst = None;
        }
        Some(s) => {
            if dst.as_ref().is_none_or(|d| d.len() != s.len()) {
                *dst = Some(DeviceBuffer::alloc(s.len())?);
            }
            if let Some(d) = dst.as_mut() {
                d.copy_from(0, s, 0, s.len(), st)?;
            }
        }
    }
    Ok(())
}

/// Recurrent and attention state a stack of layers reads and updates.
/// One row-run of a batched forward: consecutive scratch rows that belong to one session, with that session's
/// state.
pub(crate) struct Run<'a> {
    pub(crate) ls: &'a LayerStates<'a>,
    /// Row of the batch the run starts at: the stateful kernels are launched with this offset, because every
    /// intermediate buffer packs its rows at the tensor's own width (which changes with the mixer).
    pub(crate) off: usize,
    pub(crate) rows: usize,
    /// Address of this run's KV position (i32 on the device): the attention kernels read it from memory instead of a
    /// launch immediate, so a batch cannot share the single one `upload_step` writes.
    pub(crate) len_dev: u64,
}

/// The runs of one batched forward. One run per session for a decode step; a run may be longer (a prefill piece, a
/// verify chain) once the callers use it.
pub(crate) struct Batch<'a> {
    pub(crate) runs: Vec<Run<'a>>,
}

pub(crate) struct LayerStates<'a> {
    pub(crate) kv: &'a [(DeviceBuffer, DeviceBuffer)],
    pub(crate) conv: &'a [DeviceBuffer],
    pub(crate) rec: &'a [DeviceBuffer],
    pub(crate) len: usize,
    pub(crate) max_ctx: usize,
    pub(crate) attn_part: &'a DeviceBuffer,
    pub(crate) mla_part: Option<&'a DeviceBuffer>,
    pub(crate) conv_snaps: Option<&'a [DeviceBuffer]>,
    /// DeltaNet update records: (buffers, pending count, buffer with the pending ones, record this call).
    pub(crate) dn: Option<(&'a [[DeviceBuffer; 2]], usize, usize, bool)>,
    pub(crate) ple_state: Option<&'a DeviceBuffer>,
    /// Verify: where `ple` saves the conv state and the step's normalized rows.
    pub(crate) ple_snap: Option<&'a DeviceBuffer>,
    pub(crate) idx: Option<IdxStates<'a>>,
}

/// QSA indexer state for a stack of layers; `sel` is set when some query of the input has more complete blocks than
/// the budget: (scores, selections, 64-query unions, words per selection).
pub(crate) struct IdxStates<'a> {
    pub(crate) kb: &'a [DeviceBuffer],
    pub(crate) tk: &'a [DeviceBuffer],
    pub(crate) tp: &'a [DeviceBuffer],
    pub(crate) sel: Option<(&'a DeviceBuffer, &'a DeviceBuffer, &'a DeviceBuffer, usize)>,
    /// Block lists of the selections (rows, then the counts after `rows` rows).
    pub(crate) list: Option<(&'a DeviceBuffer, usize)>,
    /// Decode reuse: last selection bitset per indexer layer, and whether this forward may use it.
    pub(crate) cache: Option<&'a [DeviceBuffer]>,
    pub(crate) reuse: bool,
}

/// Queries scored per pass of the indexer (bounds the score buffer).
impl IdxStates<'_> {
    pub(crate) fn sel_args(&self) -> [Arg; 3] {
        match self.sel {
            Some((_, sel, uni, _)) => [sel.into(), uni.into(), Arg::U64(0)],
            None => [Arg::U64(0), Arg::U64(0), Arg::U64(0)],
        }
    }
}

/// Inputs from this many tokens use the WMMA prefill attention; shorter ones the split-key kernel.
pub(crate) const ATTN_WMMA_MIN: usize = 16;
pub(crate) const ATTN_SPLIT: usize = 256;
/// i32 slots before the token ids in `Scratch::hdr`: the KV position, padded to 16 bytes.
pub(crate) const STEP_HDR: usize = 4;
/// Largest host copy (in i32) `Scratch::upload_step` makes in one piece; a longer step uploads its three parts.
pub(crate) const STEP_UPLOAD_MAX: usize = 1 << 16;

pub(crate) struct Scratch {
    pub(crate) cap: usize,
    pub(crate) x: DeviceBuffer,
    pub(crate) hn: DeviceBuffer,
    pub(crate) big0: DeviceBuffer,
    pub(crate) big1: DeviceBuffer,
    pub(crate) big2: DeviceBuffer,
    pub(crate) small0: DeviceBuffer,
    pub(crate) small1: DeviceBuffer,
    pub(crate) small2: DeviceBuffer,
    pub(crate) small3: DeviceBuffer,
    pub(crate) xr: DeviceBuffer,
    pub(crate) xh: DeviceBuffer,
    pub(crate) xsc: DeviceBuffer,
    /// The inputs the host writes at every step, in one allocation so that a decode step uploads them with one
    /// host copy instead of three: [len, pad x 3, ids x cap, pos x 3 cap] as i32. `len_dev`, `ids` and `pos` are
    /// windows into it.
    pub(crate) hdr: DeviceBuffer,
    pub(crate) ids: DeviceBuffer,
    pub(crate) pos: DeviceBuffer,
    /// The session's KV position (sess.len) as a single i32 on the device, uploaded once per forward.
    /// Attention kernels read the write/read offset from here instead of a launch immediate, so the
    /// kernel node parameters stay constant across decode steps and the forward can be graph-captured.
    pub(crate) len_dev: DeviceBuffer,
    /// MTP inputs: hidden rows and the [embedding; hidden] concatenation.
    pub(crate) mh: DeviceBuffer,
    pub(crate) cat: DeviceBuffer,
    /// Chunked DeltaNet terms (deltanet_chunk_prep): Wn, Ut0, Qg, P, KdT, gamma_C.
    pub(crate) dn: [DeviceBuffer; 6],
    pub(crate) moe: Option<MoeScratch>,
    pub(crate) hc: Option<HcScratch>,
    /// Indexer projection [T, (heads + 1) * dim], normalized and roped queries, raw keys.
    pub(crate) idx: Option<[DeviceBuffer; 3]>,
    /// KDA (Ling): the conv output of the k and v streams. q's goes to `big1`, and the raw projections are free
    /// again once `conv_state_save` has read them, so two extra rows are enough for the whole mixer.
    pub(crate) kda: Option<[DeviceBuffer; 2]>,
    /// MLA (Ling): absorbed query per (token, head) for one prefill tile, and the weighted latent.
    pub(crate) mla_qt: Option<DeviceBuffer>,
    pub(crate) mla_lat: Option<DeviceBuffer>,
}

/// Hyper-connection buffers: stream state and its norm, the down/inject/up products of the mixer, the output of a
/// mixer or FFN, and the PLE n-gram embedding.
pub(crate) struct HcScratch {
    pub(crate) hc: DeviceBuffer,
    pub(crate) hcn: DeviceBuffer,
    pub(crate) hd: DeviceBuffer,
    pub(crate) ha: DeviceBuffer,
    pub(crate) hu: DeviceBuffer,
    /// Injection weights, alternating between consecutive mixes (a fused mix reads one and writes the other).
    pub(crate) inj: [DeviceBuffer; 2],
    pub(crate) mo: DeviceBuffer,
    pub(crate) ple_e: DeviceBuffer,
    /// 1 / rms of every stream (hc_quant_row).
    pub(crate) inv: DeviceBuffer,
}

pub(crate) struct MoeScratch {
    /// Router logits [T, E + 1], expert id and weight of each (token, slot) pair.
    pub(crate) logits: DeviceBuffer,
    pub(crate) ids: DeviceBuffer,
    /// `MALLEUS_EXPERT_CACHE`: the same pairs with the expert id replaced by its slot in the layer's pool.
    pub(crate) slot_ids: std::sync::Mutex<DeviceBuffer>,
    pub(crate) w: DeviceBuffer,
    /// Decode: gate|up outputs and rotated silu(gate) * up per pair; prefill: silu(gate) * up per sorted row.
    pub(crate) gu: DeviceBuffer,
    pub(crate) act: DeviceBuffer,
    /// Prefill (moe_sort): row offsets per expert, sorted row → pair, pair → sorted row, tiles and their count.
    pub(crate) offs: DeviceBuffer,
    pub(crate) perm: DeviceBuffer,
    pub(crate) pos: DeviceBuffer,
    pub(crate) tiles: DeviceBuffer,
    pub(crate) ntiles: DeviceBuffer,
    /// Prefill: gathered int8 inputs, quantized expert activations (with row scales), expert outputs per sorted row.
    pub(crate) xg: DeviceBuffer,
    pub(crate) xgs: DeviceBuffer,
    pub(crate) aq: DeviceBuffer,
    pub(crate) aqs: DeviceBuffer,
    pub(crate) y: DeviceBuffer,
}

/// Tokens per chunk of the chunked DeltaNet; shorter inputs use the token-by-token scan.
pub(crate) const DN_CHUNK: usize = 64;

impl Scratch {
    /// Uploads the KV position, the token ids and their positions of a step.
    ///
    /// One host copy for a decode step: each synchronous copy costs tens of microseconds with the GPU idle, and a
    /// speculative round makes this call half a dozen times.
    pub(crate) fn upload_step(&mut self, len: i32, ids: &[i32], pos: &[i32]) -> Result<()> {
        ensure!(
            ids.len() <= self.cap && pos.len() == 3 * ids.len(),
            "step larger than the scratch"
        );
        let n = STEP_HDR + self.cap + pos.len();
        if n > STEP_UPLOAD_MAX {
            self.len_dev.upload(&[len])?;
            self.ids.upload(ids)?;
            self.pos.upload(pos)?;
            return Ok(());
        }
        let mut h = vec![0i32; n];
        h[0] = len;
        h[STEP_HDR..STEP_HDR + ids.len()].copy_from_slice(ids);
        h[STEP_HDR + self.cap..].copy_from_slice(pos);
        Ok(self.hdr.upload(&h)?)
    }
}

pub(crate) fn f32s(n: usize) -> Result<DeviceBuffer> {
    Ok(DeviceBuffer::alloc(n * 4)?)
}

/// Average rows per expert from which the W4A8 grouped GEMMs use 128-row tiles (MALLEUS_MOE_BM128 overrides).
/// `MALLEUS_TRACE_ROUTE=<path>`: append the experts the router picked, one line per MoE call and token. It is the
/// input of the expert-cache measurements (scripts/expert_cache.py): how skewed the routing is, how much of the
/// working set a cache of N GiB would hit, and how much a token would ask the disk. Debug: it synchronizes the
/// stream once per layer, so a trace run is slower than a normal one.
pub(crate) static ENV_TRACE_ROUTE: std::sync::LazyLock<Option<String>> =
    std::sync::LazyLock::new(|| std::env::var("MALLEUS_TRACE_ROUTE").ok());

/// Appends `layer call row id:w ...` for every (token, slot) of one MoE routing decision. The shared expert comes
/// out with id `experts` (Ling): the cache measures the routed ones, so the reader drops ids past the last expert.
pub(crate) fn trace_routes(st: &Stream, ms: &MoeScratch, layer: usize, t: usize, slots: usize) -> Result<()> {
    let Some(path) = ENV_TRACE_ROUTE.as_deref() else {
        return Ok(());
    };
    st.synchronize()?;
    let ids: Vec<i32> = ms.ids.download(t * slots)?;
    let w: Vec<f32> = ms.w.download(t * slots)?;
    let mut text = String::new();
    for row in 0..t {
        text.push_str(&format!("{layer} {t} {row}"));
        for slot in 0..slots {
            let j = row * slots + slot;
            text.push_str(&format!(" {}:{:.6}", ids[j], w[j]));
        }
        text.push('\n');
    }
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(text.as_bytes())?;
    Ok(())
}
