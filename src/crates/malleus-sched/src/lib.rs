#![forbid(unsafe_code)]

//! Scheduler: one GPU thread owns the loaded model and up to `MALLEUS_SLOTS` sessions.
//!
//! Requests are admitted first come first served, one per free slot, and every round gives each active session one
//! step: a prefill piece, or a token of decode. A long prompt therefore no longer freezes the conversations already
//! running next to it — one piece is the longest a decoding session waits. One slot is what this module always did:
//! one session, one request at a time, the whole context, byte for byte the same work order.
//!
//! This first version splits the *waiting*, not the weight read: the sessions take turns, each step still loads the
//! weights for one token (issue #33, v1). Sharing one forward between the sessions is the next step and needs the
//! attention kernels to carry a per-row KV base.
//!
//! The server may know several models (see [`catalog`]); this module keeps one of them resident. A request for
//! another model drops the current `Worker` and loads the requested one on the same GPU thread, because the HIP
//! context is per thread and the weights share the device memory with the host.

pub mod catalog;

pub use catalog::{Catalog, ModelSpec};

use anyhow::{Context, Result};
use malleus_hip::DeviceBuffer;
use malleus_model::image_pre::{PreConfig, decode, preprocess_inkling2, preprocess_rgb};
use malleus_model::mrope;
use malleus_model::{GEMV_MAX_T, Model, Session, Snapshot};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio::sync::mpsc as tmpsc;

#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub seed: u64,
    /// Repetition control, llama.cpp semantics: with `repeat_penalty` different from 1.0 a token that already appears
    /// in the last `repeat_last_n` of the sequence has its logit divided by the penalty (or multiplied, when the logit
    /// is negative). `frequency_penalty`/`presence_penalty` are the OpenAI ones, applied in logit space before the
    /// temperature. All off by default: the sampling stays bit-identical to what it was before they existed.
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: 0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

impl Sampling {
    fn penalizes(&self) -> bool {
        self.repeat_last_n > 0
            && (self.repeat_penalty != 1.0 || self.frequency_penalty != 0.0 || self.presence_penalty != 0.0)
    }
}

/// The logits row to sample from: `None` when no penalty is active, so the caller keeps using the row it has.
///
/// `history` is the sequence so far and `tail` the tokens accepted in a speculative round after it; the window is the
/// last `repeat_last_n` of `history ++ tail`, which is what makes the serial step and the speculative loop see the
/// same history (a draft that gets committed is a token like any other).
fn penalized(logits: &[f32], s: &Sampling, history: &[u32], tail: &[u32]) -> Option<Vec<f32>> {
    if !s.penalizes() {
        return None;
    }
    let mut counts: Vec<(u32, u32)> = Vec::with_capacity(s.repeat_last_n.min(64));
    for &id in history.iter().rev().chain(tail.iter().rev()).take(s.repeat_last_n) {
        match counts.iter_mut().find(|(t, _)| *t == id) {
            Some((_, c)) => *c += 1,
            None => counts.push((id, 1)),
        }
    }
    if counts.is_empty() {
        return None;
    }
    let mut row = logits.to_vec();
    for (id, count) in counts {
        let Some(v) = row.get_mut(id as usize) else {
            continue;
        };
        let mut l = *v;
        if s.repeat_penalty != 1.0 {
            l = if l > 0.0 {
                l / s.repeat_penalty
            } else {
                l * s.repeat_penalty
            };
        }
        l -= s.presence_penalty;
        l -= s.frequency_penalty * count as f32;
        *v = l;
    }
    Some(row)
}

pub struct Request {
    /// The catalog entry the API resolved for this request. The worker switches to it when it is not the loaded one.
    pub model: Arc<ModelSpec>,
    pub prompt: Vec<u32>,
    /// Token sequences a prompt-lookup draft may come from besides the prompt itself: the original text of the recent
    /// messages (line numbers and JSON escapes removed by the API layer). A tool result reaches the model as a
    /// line-numbered rendering and is written back as plain text, so without these the tokens the model is about to
    /// copy are never in the prompt's own token stream.
    pub drafts: Vec<Vec<u32>>,
    /// Raw encoded images (PNG/JPEG/WebP/GIF), in the order their `<|image_pad|>` placeholders appear.
    pub images: Vec<Vec<u8>>,
    /// Tokenizer id of `<|image_pad|>`; required when `images` is not empty.
    pub image_token: Option<u32>,
    pub max_tokens: usize,
    pub sampling: Sampling,
    pub stop_tokens: Vec<u32>,
    /// Per-emitted-token temperature-1 log-probabilities in the `Token` events (OpenAI `logprobs`). Zero-cost when
    /// false: the sampling path is untouched, only the events grow a field.
    pub logprobs: bool,
    /// How many top alternatives each `Token` event also carries; 0 = none (capped by the API layer, 20 like OpenAI).
    pub top_logprobs: usize,
    /// Also keep the states after this many prompt tokens (the start of the last chat turn: the next request of
    /// the conversation shares it even when the previous answer comes back rewritten).
    pub snapshot_at: Option<usize>,
    pub cancel: Arc<AtomicBool>,
    pub events: tmpsc::UnboundedSender<Event>,
}

#[derive(Debug)]
pub enum Event {
    Started {
        prompt_tokens: usize,
        cached_tokens: usize,
    },
    /// A generated token. `top` carries its log-probability data only when the request asked for logprobs;
    /// otherwise it is None and emitting is exactly as cheap as a bare integer event.
    Token {
        id: u32,
        top: Option<TokenProbs>,
    },
    /// The worker started loading this model. Streaming clients can keep their connection open while it loads.
    Loading {
        model: String,
    },
    Done(Finish),
    Error(String),
}

/// Temperature-1 log-probabilities of one emitted token, from the logits row that produced it.
#[derive(Debug, Clone)]
pub struct TokenProbs {
    pub id: u32,
    /// The token's own log-probability (log-softmax over the whole vocabulary).
    pub logprob: f32,
    /// `(id, logprob)` of the alternatives, most likely first; up to the request's `top_logprobs`.
    pub top: Vec<(u32, f32)>,
}

/// The top-N of a logits row plus the token's own probability, as temperature-1 log-probabilities (log-softmax over
/// the full vocabulary). The top-N is kept as a partial queue the size of the request (an insert into a Vec of at
/// most `top` entries, never a vocabulary sort); when `top` is 0 only the token's own log-probability is returned,
/// for `logprobs: true` without `top_logprobs`.
pub fn token_probs(logits: &[f32], id: u32, top: usize) -> TokenProbs {
    let mut max = f32::NEG_INFINITY;
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(top.min(64));
    for (i, &l) in logits.iter().enumerate() {
        max = max.max(l);
        if top == 0 {
            continue;
        }
        // Position of the new entry in the descending-by-logit queue; equal logits order by id so the result is
        // deterministic.
        let pos = best.iter().position(|&(b, j)| l > b || (l == b && i < j as usize));
        match pos {
            Some(p) => {
                best.insert(p, (l, i as u32));
                if best.len() > top {
                    best.pop();
                }
            }
            None if best.len() < top => best.push((l, i as u32)),
            None => {}
        }
    }
    let mut norm = 0.0f64;
    for &l in logits {
        norm += ((l - max) as f64).exp();
    }
    let lse = max as f64 + norm.ln();
    let top = best.into_iter().map(|(l, i)| (i, (l as f64 - lse) as f32)).collect();
    TokenProbs {
        id,
        logprob: logits
            .get(id as usize)
            .map_or(f32::NEG_INFINITY, |&l| (l as f64 - lse) as f32),
        top,
    }
}

#[derive(Debug, Clone)]
pub struct Finish {
    pub reason: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

/// What the GPU thread takes from its channel. One queue for everything, so FIFO holds: a load or an unload never
/// jumps ahead of a request already in flight.
pub enum Command {
    /// Generate for one request, switching models first when it names another one.
    Request(Box<Request>),
    /// Load a model without generating, so the load is finished before the request that needs it arrives.
    Load(Arc<ModelSpec>),
    /// Drop the model with this id when it is the resident one, freeing the device memory.
    Unload(Arc<str>),
}

/// Where one catalog entry stands. `Failed` carries why the last load failed, so a client polling the catalog stops
/// waiting instead of waiting for a load that will never come.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelState {
    Loading,
    Loaded,
    Unloaded,
    Failed(String),
}

/// One transition, as served on the status stream and as read back by the catalog.
#[derive(Clone, Debug)]
pub struct ModelEvent {
    pub id: Arc<str>,
    pub state: ModelState,
}

/// What the GPU thread has resident, what it is loading and the last load that failed. Cloned into the API, which
/// reads it on every catalog request; every setter also broadcasts, so a watching client needs no polling and a
/// polling client sees exactly the same thing.
#[derive(Clone)]
pub struct Status {
    loaded: Arc<std::sync::Mutex<Option<Arc<ModelSpec>>>>,
    loading: Arc<std::sync::Mutex<Option<Arc<ModelSpec>>>>,
    failed: Arc<std::sync::Mutex<Option<ModelEvent>>>,
    events: broadcast::Sender<ModelEvent>,
}

impl Default for Status {
    fn default() -> Self {
        Status {
            loaded: Arc::default(),
            loading: Arc::default(),
            failed: Arc::default(),
            events: broadcast::channel(STATUS_CHANNEL).0,
        }
    }
}

fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Status {
    /// A load started: the previous model is gone by then (its memory has to be free for the new one) and a failure
    /// of this same id no longer describes it.
    fn set_loading(&self, spec: &Arc<ModelSpec>) {
        *lock(&self.loaded) = None;
        *lock(&self.loading) = Some(spec.clone());
        self.clear_failure(&spec.id);
        self.emit(&spec.id, ModelState::Loading);
    }

    fn set_loaded(&self, spec: &Arc<ModelSpec>) {
        *lock(&self.loading) = None;
        *lock(&self.loaded) = Some(spec.clone());
        self.clear_failure(&spec.id);
        self.emit(&spec.id, ModelState::Loaded);
    }

    fn set_unloaded(&self, id: &Arc<str>) {
        *lock(&self.loaded) = None;
        self.emit(id, ModelState::Unloaded);
    }

    fn set_failed(&self, id: &Arc<str>, message: String) {
        *lock(&self.loading) = None;
        *lock(&self.failed) = Some(ModelEvent {
            id: id.clone(),
            state: ModelState::Failed(message.clone()),
        });
        self.emit(id, ModelState::Failed(message));
    }

    fn clear_failure(&self, id: &Arc<str>) {
        let mut failed = lock(&self.failed);
        if failed.as_ref().is_some_and(|f| f.id == *id) {
            *failed = None;
        }
    }

    fn emit(&self, id: &Arc<str>, state: ModelState) {
        // No receiver is not an error: the catalog and the stream are optional views of the same state.
        let _ = self.events.send(ModelEvent { id: id.clone(), state });
    }

    pub fn loaded(&self) -> Option<Arc<ModelSpec>> {
        lock(&self.loaded).clone()
    }

    pub fn loading(&self) -> Option<Arc<ModelSpec>> {
        lock(&self.loading).clone()
    }

    /// The last load that failed, whoever it was for: the caller compares ids.
    pub fn failure(&self) -> Option<ModelEvent> {
        lock(&self.failed).clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ModelEvent> {
        self.events.subscribe()
    }
}

/// How many transitions a slow `/models/sse` reader may fall behind before it is told it lagged. A load is two
/// transitions, so this covers far more switching than a client can use.
const STATUS_CHANNEL: usize = 64;

pub struct Engine {
    tx: mpsc::Sender<Command>,
    /// Resident, loading and last failure, shared with the API.
    status: Status,
    pub queued: Arc<AtomicUsize>,
    /// Wall clock (ms since the epoch) at which the request in flight started, 0 when the worker is idle. Read by
    /// the watchdog: a GPU kernel cannot be cancelled from the host, so an overrun has to end the process.
    busy_since: Arc<AtomicU64>,
}

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
        .0 as u32
}

/// Prompt lookup: the tokens that followed the latest earlier occurrence of the last `n` tokens of `hist` (up to
/// `max`), or none.
pub fn lookup_drafts(hist: &[u32], sources: &[Vec<u32>], n: usize, max: usize) -> Vec<u32> {
    if max == 0 || hist.len() < n || n == 0 {
        return vec![];
    }
    let key = &hist[hist.len() - n..];
    // The longest continuation wins, not the most recent match. A tool result reaches the prompt with its line
    // numbers, so a match inside the prompt's own tokens stops at the next line break (the numbered prefix is not what
    // the model writes) while the same text in `sources` continues across lines. Taking the first match instead would
    // mean the short one always shadows the long one, which is what the sources are here for.
    let mut best: Vec<u32> = vec![];
    // The match may not overlap the key itself.
    for i in (0..hist.len() - n).rev() {
        if &hist[i..i + n] == key {
            let end = (i + n + max).min(hist.len());
            if i + n < end && end - (i + n) > best.len() {
                best = hist[i + n..end].to_vec();
            }
            break;
        }
    }
    // Then the original text of the recent messages, newest first.
    for s in sources.iter().rev() {
        if s.len() <= n || best.len() >= max {
            break;
        }
        for i in (0..=s.len() - n).rev() {
            if &s[i..i + n] == key {
                let end = (i + n + max).min(s.len());
                if i + n < end && end - (i + n) > best.len() {
                    best = s[i + n..end].to_vec();
                }
                break;
            }
        }
    }
    best
}

/// Greedy MTP drafts: up to `max` tokens, chaining while the product of the draft probabilities stays >= `pmin`.
/// With `hist` (the tokens so far, ending with the pending ones), a prompt-lookup continuation that starts with the
/// MTP draft and is longer than one token replaces the chained drafts.
pub fn draft_tokens(
    model: &Model,
    sess: &mut Session,
    pend: &[u32],
    max: usize,
    pmin: f32,
    hist: Option<(&[u32], &[Vec<u32>])>,
) -> Result<Vec<u32>> {
    if model.has_block_draft() {
        return model.block_draft(sess, *pend.last().context("no pending token")?, max);
    }
    let (d, mut p) = model.mtp_draft(sess, pend)?;
    if let Some((h, sources)) = hist {
        let lk = lookup_drafts(h, sources, env_or("MALLEUS_LOOKUP_NGRAM", LOOKUP_NGRAM), max);
        if lk.len() >= 2 && lk[0] == d {
            return Ok(lk);
        }
    }
    let mut ds = vec![d];
    while ds.len() < max && p >= pmin {
        let (d, q) = model.mtp_chain(sess, *ds.last().unwrap())?;
        ds.push(d);
        p *= q;
    }
    Ok(ds)
}

/// Candidate tokens and their probabilities after temperature, top-k, min-p and top-p (normalized).
fn distribution(logits: &[f32], s: &Sampling) -> (Vec<u32>, Vec<f32>) {
    let k = if s.top_k == 0 { 256 } else { s.top_k.min(1024) };
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    idx.truncate(k);
    idx.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    let m = logits[idx[0] as usize];
    let mut p: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i as usize] - m) / s.temperature).exp())
        .collect();
    let sum: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= sum);
    let mut keep = p.len();
    if s.min_p > 0.0 {
        keep = keep.min(p.iter().take_while(|&&v| v >= s.min_p * p[0]).count().max(1));
    }
    if s.top_p < 1.0 {
        let mut c = 0.0;
        for (i, v) in p.iter().enumerate() {
            c += v;
            if c >= s.top_p {
                keep = keep.min(i + 1);
                break;
            }
        }
    }
    idx.truncate(keep);
    p.truncate(keep);
    let total: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= total);
    (idx, p)
}

/// Rows past the real vocabulary are not tokens (the unembedding can be padded, see `Config::real_vocab`): they are
/// set to -inf before anything samples or normalizes a row, so a padded id can never be generated. A no-op for the
/// families whose head has no padding.
fn mask_pad(row: &mut [f32], real: usize) {
    if real < row.len() {
        row[real..].fill(f32::NEG_INFINITY);
    }
}

fn pick(idx: &[u32], p: &[f32], mut r: f32) -> u32 {
    for (i, &v) in p.iter().enumerate() {
        r -= v;
        if r <= 0.0 {
            return idx[i];
        }
    }
    *idx.iter()
        .zip(p)
        .rev()
        .find(|(_, v)| **v > 0.0)
        .map_or(&idx[idx.len() - 1], |(i, _)| i)
}

pub fn sample(logits: &[f32], s: &Sampling, rng: &mut impl FnMut() -> f32) -> u32 {
    if s.temperature <= 0.0 {
        return argmax(logits);
    }
    let (idx, p) = distribution(logits, s);
    pick(&idx, &p, rng())
}

/// Speculative sampling with a deterministic draft: keeps `draft` with probability p(draft), otherwise samples
/// from p without it, so the output follows p exactly. Returns (token, draft accepted).
pub fn sample_spec(logits: &[f32], s: &Sampling, draft: u32, rng: &mut impl FnMut() -> f32) -> (u32, bool) {
    if s.temperature <= 0.0 {
        let t = argmax(logits);
        return (t, t == draft);
    }
    let (idx, mut p) = distribution(logits, s);
    let pd = idx.iter().position(|&t| t == draft).map_or(0.0, |i| p[i]);
    if rng() < pd {
        return (draft, true);
    }
    if let Some(i) = idx.iter().position(|&t| t == draft) {
        p[i] = 0.0;
    }
    let rest = 1.0 - pd;
    if rest <= 0.0 {
        return (draft, true);
    }
    (pick(&idx, &p, rng() * rest), false)
}

/// Prompt tokens per forward (MALLEUS_PREFILL_CHUNK overrides): larger chunks read the MoE expert weights fewer
/// times per token, at the cost of scratch memory.
fn prefill_chunk(model: &Model) -> usize {
    // Inkling-2's attention reads the session's cache for the prefix now (`inkling2_attn_wmma_cached`), so a long
    // prompt can go in pieces: the pieces are what keeps the per-forward scratch bounded, and the sliding cache's
    // ring is sized for exactly this chunk.
    if model.cfg.family == malleus_model::Family::Inkling2 {
        return malleus_model::inkling2::PREFILL_CHUNK;
    }
    malleus_model::memplan::prefill_chunk(&model.cfg)
}

/// The `(start, end)` blocks an image occupies in the expanded prompt.
fn image_blocks(inserts: &[(usize, DeviceBuffer)], hidden: usize) -> Vec<(usize, usize)> {
    inserts.iter().map(|(s, b)| (*s, *s + b.len() / (hidden * 4))).collect()
}

/// Ranges of one image prompt's prefill: at most `chunk` tokens each, and never cut across an image block.
///
/// An image prompt used to be a single forward, with `positions` (T,3) and the embedding inserts indexed against
/// the whole prompt. Both are per token, so they can be sliced per piece; what cannot be sliced is an insert,
/// because `forward_mm` copies it whole (`copy_from(start, buf, 0, rows)`, source offset 0). Hence the cuts.
///
/// The cost of not chunking is the reason this exists: `Model::forward` sizes its scratch on the token count, so
/// one forward over a 30k-token agent prompt with a screenshot asks ~13 GiB of GTT (= system RAM on these APUs)
/// in one go, and leaves it cached afterwards (the vision prefill fix).
fn image_pieces(len: usize, blocks: &[(usize, usize)], chunk: usize) -> Vec<(usize, usize)> {
    let chunk = chunk.max(1);
    let mut pieces = Vec::new();
    let mut start = 0;
    while start < len {
        let mut end = (start + chunk).min(len);
        for &(bs, be) in blocks {
            if bs < end && be > end {
                // This block would be cut by `end`: stop before it, or take it whole when the piece starts there
                // (an image block wider than the chunk gets a piece of its own).
                end = if bs > start { bs } else { be.min(len) };
                break;
            }
        }
        pieces.push((start, end));
        start = end;
    }
    pieces
}

const DEFAULT_DRAFTS: usize = 5;
/// A chain of drafts goes on while the product of their probabilities stays above this. At 0.5 the tail of the
/// chain was mostly rejected: each such draft costs a pass of the draft head and a row of the verify. Measured on
/// Flash-Next, eight prompts: 56.5 tok/s at 0.5, 58.3 at 0.65, 59.5 at 0.8, and a plateau up to 0.95.
pub const DEFAULT_DRAFT_PMIN: f32 = 0.8;

/// Patches (16 px each) an image may be reduced to. The vision tower's attention is **quadratic** in the patches:
/// measured on gfx1150 with this model, 512x512 (1024 patches) encodes in ~3.8 s and a 12 Mpx phone photo
/// (47 000 patches) needs hours, which blocked the only GPU worker with the service still answering `/v1/models`.
/// 2048 patches (~15 s by that scaling) is the default; `MALLEUS_VISION_MAX_PATCHES` raises it for models whose
/// tower can afford more.
const DEFAULT_VISION_MAX_PATCHES: usize = 2048;

/// Seconds a single request may run before the watchdog ends the process (the unit restarts it). 0 disables it.
/// Without this, any long or pathological request hangs the service until someone restarts it by hand.
const DEFAULT_REQUEST_TIMEOUT: u64 = 600;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
/// Tokens matched by prompt lookup: `MALLEUS_LOOKUP_NGRAM`, default 3. A longer context matches less often but its
/// continuation is more likely to be right, which is what pays on code and structured text.
const LOOKUP_NGRAM: usize = 3;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

struct Worker {
    spec: Arc<ModelSpec>,
    model: Model,
    /// Sessions resident, one per slot: each holds its own KV, recurrent state, prompt cache and snapshots, so
    /// several requests take turns on the GPU instead of queueing behind each other.
    slots: Vec<Slot>,
    /// MTP drafts per decode step (0: plain decode).
    drafts: usize,
    draft_pmin: f32,
    /// Greedy steps read back the argmax instead of the logits row (`MALLEUS_STEP_ARGMAX=0` keeps the host fold,
    /// for the A/B).
    step_argmax: bool,
    /// Prompt-lookup drafts from the context when they agree with the MTP head.
    lookup: bool,
    /// Prompt snapshots kept per slot.
    max_snaps: usize,
    /// Prompt tokens processed per step: the whole prefill chunk with one slot, a slice of it when several sessions
    /// take turns (`MALLEUS_PREFILL_QUANTUM`), because one piece is what a decoding session waits for.
    quantum: usize,
}

/// One session and the prompt state that belongs to it. This split is what makes two requests independent on one
/// GPU: the weights are shared, the state is not.
struct Slot {
    sess: Session,
    /// Tokens whose KV rows this session holds, from the start of its context.
    cached: Vec<u32>,
    /// Prompt snapshots (tokens, states), least recently used first; at most `max_snaps`.
    snaps: Vec<(Vec<u32>, Snapshot)>,
    /// Snapshots whose GPU buffers are already allocated, waiting to be refilled: a prompt is snapshotted on every
    /// request, and allocating the ~100 state buffers costs far more than copying them (`Model::new_snapshot`).
    snap_pool: Vec<Snapshot>,
}

impl Slot {
    /// Hands a retired snapshot's buffers back to the pool; freeing them would give the pages right back to the
    /// allocator we are trying not to call on the request path.
    fn recycle(&mut self, snap: Snapshot) {
        if self.snap_pool.len() < 2 {
            self.snap_pool.push(snap);
        }
    }

    /// A snapshot to fill: one from the pool, else the buffers of the least recently used snapshot when all
    /// `max_snaps` are full (it is the one about to be dropped anyway), else a fresh allocation.
    fn take_snap(&mut self, model: &Model, max_snaps: usize) -> Option<Snapshot> {
        self.snap_pool.pop().or_else(|| {
            if !self.snaps.is_empty() && self.snaps.len() >= max_snaps {
                return Some(self.snaps.remove(0).1);
            }
            model.new_snapshot(&self.sess).ok()
        })
    }

    /// Keeps the session states for the tokens in `cached` (least recently used snapshots dropped first).
    fn push_snapshot(&mut self, model: &Model, max_snaps: usize) {
        if max_snaps == 0 {
            return;
        }
        let mut snap = match self.take_snap(model, max_snaps) {
            Some(s) => s,
            None => return,
        };
        if model.snapshot_into(&self.sess, &mut snap).is_err() {
            self.recycle(snap);
            return;
        }
        if let Some(i) = self.snaps.iter().position(|(t, _)| *t == self.cached) {
            let old = self.snaps.remove(i).1;
            self.recycle(old);
        }
        self.snaps.push((self.cached.clone(), snap));
        while self.snaps.len() > max_snaps {
            let old = self.snaps.remove(0).1;
            self.recycle(old);
        }
    }

    /// Back to an empty context, keeping the snapshots (they are keyed by their own tokens).
    fn reset(&mut self, model: &Model) -> Result<()> {
        model.reset_session(&mut self.sess)?;
        self.cached.clear();
        Ok(())
    }
}

/// Where a step of a request stands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Prompt tokens still to process; nothing has been emitted yet.
    Prefill,
    /// One token (or one verify round) per step.
    Decode,
    /// Finished during a batched step: the scheduler reports it and frees the slot.
    Finished,
}

/// The outcome of one step: the request runs on, or it is over (finished, cancelled, or out of context).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Progress {
    More,
    Done,
}

/// One request being served. Everything that has to survive from one step to the next and used to live in the body
/// of the old `serve()`: the pieces still to prefill, the sampler, the emitted count and the pending drafts.
struct Active {
    req: Request,
    slot: usize,
    phase: Phase,
    /// Ranges `(start, len)` of `req.prompt` still to prefill, in order.
    pieces: Vec<(usize, usize)>,
    piece: usize,
    /// Index in `pieces` after which a snapshot is kept (the start of the last chat turn).
    mid: Option<usize>,
    image_req: bool,
    image_pos: Option<Vec<[i32; 3]>>,
    image_inserts: Vec<(usize, DeviceBuffer)>,
    t0: Instant,
    t1: Instant,
    /// Prefill time, fixed when the first token is sampled.
    prefill_s: f64,
    /// Logits of the last prefill piece, waiting for the first token to be sampled off them.
    logits: Vec<f32>,
    rng: Rng,
    n: usize,
    tok: u32,
    pend: Vec<u32>,
    top: Option<TokenProbs>,
    reason: FinishReason,
    max_tokens: usize,
    ctx: usize,
}

impl Active {
    /// Emits one token with its log-probability data (None when logprobs were not asked); false when generation
    /// must stop.
    fn emit(&mut self, t: u32, top: Option<TokenProbs>) -> bool {
        self.n += 1;
        if self.req.stop_tokens.contains(&t) {
            self.reason = FinishReason::Stop;
            return false;
        }
        if self.req.events.send(Event::Token { id: t, top }).is_err() || self.req.cancel.load(Ordering::Relaxed) {
            self.reason = FinishReason::Cancelled;
            return false;
        }
        self.n < self.max_tokens
    }

    fn finish(&self) -> Finish {
        Finish {
            reason: self.reason,
            prompt_tokens: self.req.prompt.len(),
            completion_tokens: self.n,
            prefill_s: self.prefill_s,
            decode_s: self.t1.elapsed().as_secs_f64(),
        }
    }
}

/// Splits `[start, end)` into pieces of at most `quantum` tokens.
fn prompt_pieces(start: usize, end: usize, quantum: usize) -> Vec<(usize, usize)> {
    let q = quantum.max(1);
    (start..end).step_by(q).map(|s| (s, (s + q).min(end) - s)).collect()
}

/// The free slot a request starts on: the one whose session already holds the longest prefix of `prompt`, so a
/// conversation keeps hitting its own prompt cache whatever the interleaving, else the first free one.
fn pick_slot(free: &[usize], cached: &[&[u32]], prompt: &[u32]) -> Option<usize> {
    free.iter().copied().max_by_key(|&i| {
        let held = cached.get(i).copied().unwrap_or(&[]);
        let lcp = held.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        (lcp, std::cmp::Reverse(i))
    })
}

impl Worker {
    /// Serves one request on the DeepSeek-V4.1 host graph: the prompt in one prefill, then a position per token,
    /// with the events the device path emits so the API needs no branch of its own. Greedy, like `deepseek-run`:
    /// at 12.6 s a token the sampler is not what this path is measuring.
    fn serve_deepseek(&mut self, req: Request) -> Result<()> {
        let t0 = Instant::now();
        let _ = req.events.send(Event::Started {
            prompt_tokens: req.prompt.len(),
            cached_tokens: 0,
        });
        let mut prefill_s = 0.0;
        let mut emitted = 0usize;
        let mut reason = FinishReason::Length;
        let out = self
            .model
            .deepseek_generate(&req.prompt, req.max_tokens, |step, out, row| {
                if step == 0 {
                    prefill_s = t0.elapsed().as_secs_f64();
                }
                let id = *out.last().expect("a generated token");
                let top = req.logprobs.then(|| token_probs(row, id, req.top_logprobs));
                emitted = out.len();
                if req.stop_tokens.contains(&id) {
                    reason = FinishReason::Stop;
                    return false;
                }
                if req.events.send(Event::Token { id, top }).is_err() || req.cancel.load(Ordering::Relaxed) {
                    reason = FinishReason::Cancelled;
                    return false;
                }
                true
            })?;
        let decode_s = t0.elapsed().as_secs_f64() - prefill_s;
        eprintln!(
            "deepseek_v41: {} prompt + {} tokens, prefill {prefill_s:.0}s, decode {:.1}s a token, {reason:?}",
            req.prompt.len(),
            out.len(),
            if out.is_empty() {
                0.0
            } else {
                decode_s / out.len() as f64
            },
        );
        let _ = req.events.send(Event::Done(Finish {
            reason,
            prompt_tokens: req.prompt.len(),
            completion_tokens: emitted,
            prefill_s,
            decode_s,
        }));
        Ok(())
    }

    /// Prepares one request on `slot`: expands images, finds the prompt cache and emits `Started`. Nothing runs on
    /// the GPU here; the prefill is `prefill_step`.
    fn begin(&mut self, mut req: Request, slot: usize) -> Result<Active> {
        // Images are expanded and encoded before anything else: the prompt the rest of this function sees is the
        // one with the `<|image_pad|>` placeholder replaced by the grid's tokens, and its mrope positions are the
        // ones the vision tower's merge size implies.
        let image_req = !req.images.is_empty();
        let mut image_pos: Option<Vec<[i32; 3]>> = None;
        let mut image_inserts: Vec<(usize, DeviceBuffer)> = Vec::new();
        if image_req {
            let token = req.image_token.context("images without an <|image_pad|> token id")?;
            // Hy4 owns its tower in `hy4_vision` and its own preprocessing/positions; the carried Qwen models use
            // the generic `vision`; Inkling-2 has its own graph again (`inkling2_vision`), with no resize at all and
            // one image token per 40x40 tile. The three differ in the merge grid and in the token count, so the
            // expansion below is family-specific.
            let hy4_merge = self.model.hy4_vision_merge();
            let inkling2 = self.model.inkling2_vision().is_some();
            let (merge, patch) = match hy4_merge {
                Some(m) => (m, 16),
                // Inside the tower: the placeholder expands one token per tile, and the 2x2 merge is patches.
                None if inkling2 => (1, 40),
                None => {
                    let v = self.model.vision.as_ref().context("model has no vision tower")?;
                    (v.cfg.merge, v.cfg.patch)
                }
            };
            let mut expanded = Vec::with_capacity(req.prompt.len() + 256);
            let mut grids = Vec::new();
            let mut hgrids: Vec<(usize, usize, usize)> = Vec::new();
            let mut starts = Vec::new();
            let mut merged = Vec::new();
            let mut images = req.images.iter();
            for &tok in &req.prompt {
                if tok != token {
                    expanded.push(tok);
                    continue;
                }
                let bytes = images.next().context("more image placeholders than images")?;
                let (rgb, h, w) = decode(bytes, &PreConfig::default())?;
                // The tower's attention is quadratic in the patches, so the image is reduced to a patch budget
                // instead of the 16.7 Mpx the processor would accept (~65 000 patches, hours of GPU).
                let max_patches = env_or("MALLEUS_VISION_MAX_PATCHES", DEFAULT_VISION_MAX_PATCHES).max(1);
                let start = expanded.len();
                if let Some(vis) = self.model.inkling2_vision() {
                    // The processor pads to a multiple of 40 rather than resizing, so the budget is expressed in
                    // 20x20 patches — what the tower's quadratic attention actually costs — and bounds the padded
                    // image here.
                    let pre = preprocess_inkling2(
                        &rgb,
                        h,
                        w,
                        &PreConfig {
                            patch,
                            merge,
                            temporal: 1,
                            max_pixels: max_patches * 20 * 20,
                            ..Default::default()
                        },
                    )?;
                    let rows = pre.grid_h * pre.grid_w;
                    expanded.extend(std::iter::repeat_n(token, rows));
                    starts.push(start);
                    merged.push(vis.encode(&self.model, &pre)?.1);
                    continue;
                }
                if hy4_merge.is_some() {
                    let (feat, gh, gw) = self.model.hy4_image_features(&rgb, h, w, max_patches)?;
                    anyhow::ensure!(
                        gh % merge == 0 && gw % merge == 0,
                        "image grid {gh}x{gw} is not a multiple of merge {merge}"
                    );
                    let (hm, wm) = (gh / merge, gw / merge);
                    let rows = hm * (wm + 1);
                    anyhow::ensure!(
                        feat.len() == rows * self.model.cfg.hidden,
                        "hy4 image features: {} values, {rows} rows of {}",
                        feat.len(),
                        self.model.cfg.hidden
                    );
                    expanded.extend(std::iter::repeat_n(token, rows));
                    hgrids.push((1, gh, gw));
                    starts.push(start);
                    merged.push(DeviceBuffer::from_slice(&feat)?);
                    continue;
                }
                let vision = self.model.vision.as_ref().context("model has no vision tower")?;
                let pre = preprocess_rgb(
                    &rgb,
                    h,
                    w,
                    &PreConfig {
                        max_pixels: max_patches * patch * patch,
                        ..Default::default()
                    },
                )?;
                anyhow::ensure!(
                    pre.n_patches() <= max_patches,
                    "image {w}x{h} needs {} vision patches, over MALLEUS_VISION_MAX_PATCHES={max_patches}",
                    pre.n_patches()
                );
                if pre.grid_h * patch < h || pre.grid_w * patch < w {
                    eprintln!(
                        "image {w}x{h} -> {}x{} ({} patches, cap {max_patches}): the vision tower is quadratic in \
                         the patches, MALLEUS_VISION_MAX_PATCHES raises the cap",
                        pre.grid_w * patch,
                        pre.grid_h * patch,
                        pre.n_patches()
                    );
                }
                anyhow::ensure!(pre.grid_t == 1, "video input not supported");
                anyhow::ensure!(
                    pre.grid_h % merge == 0 && pre.grid_w % merge == 0,
                    "image grid {}x{} is not a multiple of merge {merge}",
                    pre.grid_h,
                    pre.grid_w
                );
                let (gh, gw) = (pre.grid_h / merge, pre.grid_w / merge);
                expanded.extend(std::iter::repeat_n(token, gh * gw));
                grids.push((gh, gw));
                starts.push(start);
                let (_, m) = vision.encode(&self.model.k, &pre)?;
                merged.push(m);
            }
            anyhow::ensure!(images.next().is_none(), "more images than image placeholders");
            req.prompt = expanded;
            // Inkling-2 has no RoPE on the text (its positions are relative), so there is nothing to derive from the
            // grid: the graph reads the rows it is given.
            let pos = if inkling2 {
                None
            } else {
                Some(match hy4_merge {
                    Some(_) => self.model.hy4_image_positions(&req.prompt, token, &hgrids)?,
                    None => mrope::positions(&req.prompt, token, &grids, 0)?.0,
                })
            };
            image_pos = pos;
            image_inserts = starts.into_iter().zip(merged).collect();
        }
        let max_snaps = self.max_snaps;
        let model = &self.model;
        let Slot {
            sess,
            cached,
            snaps,
            snap_pool,
        } = &mut self.slots[slot];
        let ctx = sess.max_ctx;
        anyhow::ensure!(
            req.prompt.len() < ctx,
            "prompt of {} tokens exceeds the context of {ctx}",
            req.prompt.len()
        );
        let max_tokens = req.max_tokens.min(ctx - req.prompt.len());
        // The session's rope extension follows the request: the factor that covers the prompt plus the tokens asked
        // for, from the ladder 1 (the checkpoint's own window) / 2 / 4. A different factor is a cache miss — the rows
        // already in the session were rotated with the old table — so the session restarts and its snapshots go.
        let factor = model.rope_factor_for(req.prompt.len() + max_tokens)?;
        if model.set_session_rope(sess, factor)? {
            model.reset_session(sess)?;
            cached.clear();
            snaps.clear();
        }
        // A prompt with images cannot reuse a token-prefix snapshot: two different images of the same grid have
        // the same tokens but different embeddings. Image requests restart from an empty session and drop the
        // snapshots, so a later text request cannot restore a state that contains an image either.
        let reuse =
            !image_req && cached.len() < req.prompt.len() && req.prompt.starts_with(&cached[..]) && !cached.is_empty();
        if !reuse {
            // The longest snapshot that is a prefix of the prompt and whose tokens the session's KV rows still hold
            // (they were not overwritten since): restart there instead of from the beginning.
            let lcp = cached.iter().zip(&req.prompt).take_while(|(a, b)| a == b).count();
            let best = if image_req {
                None
            } else {
                snaps
                    .iter()
                    .enumerate()
                    .filter(|(_, (toks, snap))| {
                        snap.len == toks.len()
                            && model.can_restore(sess, snap)
                            && toks.len() <= lcp
                            && toks.len() < req.prompt.len()
                            && req.prompt.starts_with(toks)
                    })
                    .max_by_key(|(_, (toks, _))| toks.len())
                    .map(|(i, _)| i)
            };
            match best {
                Some(i) => {
                    // Most recently used last; the snapshot is moved out because restoring borrows the session.
                    let e = snaps.remove(i);
                    model.restore(sess, &e.1)?;
                    cached.truncate(e.0.len());
                    snaps.push(e);
                }
                None => {
                    model.reset_session(sess)?;
                    cached.clear();
                }
            }
        }
        if image_req {
            while let Some((_, snap)) = snaps.pop() {
                if snap_pool.len() < 2 {
                    snap_pool.push(snap);
                }
            }
        }
        let cached_tokens = cached.len();
        let _ = req.events.send(Event::Started {
            prompt_tokens: req.prompt.len(),
            cached_tokens,
        });
        let t0 = Instant::now();
        // Prefill pieces: at most `quantum` tokens each, and never cut across an image block. An image prompt is
        // chunked like a text one: its `positions` and its embedding inserts are per token, so they are sliced per
        // piece; what cannot be sliced is an insert, because `forward_mm` copies it whole. Inkling-2's looped graph
        // reads the prompt in one forward (chunked prefill is issue #74's third item), so it gets one piece.
        let total = req.prompt.len();
        let quantum = if self.model.is_inkling2() {
            total.max(1)
        } else {
            self.quantum
        };
        let pieces: Vec<(usize, usize)> = if image_req {
            let blocks = image_blocks(&image_inserts, self.model.cfg.hidden);
            image_pieces(total - cached_tokens, &blocks, quantum)
                .into_iter()
                .map(|(s, e)| (cached_tokens + s, e - s))
                .collect()
        } else {
            prompt_pieces(cached_tokens, total, quantum)
        };
        let snap_at = req
            .snapshot_at
            .filter(|&c| max_snaps > 0 && c > cached_tokens && c < total);
        let mid = if image_req {
            None
        } else {
            snap_at.and_then(|at| {
                let mut best = None;
                for (i, (s, len)) in pieces.iter().enumerate() {
                    let end = s + len;
                    if end <= at && end < total {
                        best = Some(i);
                    }
                }
                best
            })
        };
        ok_or_empty(pieces.len())?;
        Ok(Active {
            req,
            slot,
            phase: Phase::Prefill,
            pieces,
            piece: 0,
            mid,
            image_req,
            image_pos,
            image_inserts,
            t0,
            t1: Instant::now(),
            prefill_s: 0.0,
            logits: Vec::new(),
            rng: Rng(0),
            n: 0,
            tok: 0,
            pend: Vec::new(),
            top: None,
            reason: FinishReason::Length,
            max_tokens,
            ctx,
        })
    }

    /// Processes one prefill piece; on the last one it samples the first token and hands the request to the decode
    /// steps.
    fn prefill_step(&mut self, act: &mut Active) -> Result<Progress> {
        let max_snaps = self.max_snaps;
        let model = &self.model;
        if act.req.cancel.load(Ordering::Relaxed) {
            self.slots[act.slot].reset(model)?;
            act.reason = FinishReason::Cancelled;
            act.prefill_s = act.t0.elapsed().as_secs_f64();
            act.t1 = Instant::now();
            return Ok(Progress::Done);
        }
        let (start, len) = act.pieces[act.piece];
        let real = model.cfg.real_vocab();
        let tp = Instant::now();
        let mut logits = {
            let Slot { sess, cached, .. } = &mut self.slots[act.slot];
            let logits = if act.image_req {
                let pos = act.image_pos.as_ref().map(|p| &p[start..start + len]);
                let ins: Vec<(usize, &DeviceBuffer)> = act
                    .image_inserts
                    .iter()
                    .filter_map(|(s, b)| {
                        let rows = b.len() / (model.cfg.hidden * 4);
                        (*s >= start && *s + rows <= start + len).then(|| (*s - start, b))
                    })
                    .collect();
                model.forward_mm(sess, &act.req.prompt[start..start + len], pos, 1, &ins)?
            } else {
                model.forward(sess, &act.req.prompt[start..start + len], None, 1)?
            };
            cached.extend_from_slice(&act.req.prompt[start..start + len]);
            logits
        };
        mask_pad(&mut logits, real);
        if std::env::var_os("MALLEUS_DEBUG_SNAPS").is_some() {
            eprintln!("piece {} tok: {:.1} ms", len, tp.elapsed().as_secs_f64() * 1e3);
        }
        act.piece += 1;
        if act.mid == Some(act.piece - 1) {
            let ts = Instant::now();
            self.slots[act.slot].push_snapshot(model, max_snaps);
            if std::env::var_os("MALLEUS_DEBUG_SNAPS").is_some() {
                eprintln!("snap mid: {:.1} ms", ts.elapsed().as_secs_f64() * 1e3);
            }
        }
        if act.piece < act.pieces.len() {
            return Ok(Progress::More);
        }
        // Snapshot at the end of the prompt: the next turn of a conversation usually extends it (the previous
        // answer may come back rewritten, e.g. without its reasoning, so the session itself would not match).
        let tsn = Instant::now();
        self.slots[act.slot].push_snapshot(model, max_snaps);
        if std::env::var_os("MALLEUS_DEBUG_SNAPS").is_some() {
            eprintln!("snap end: {:.1} ms", tsn.elapsed().as_secs_f64() * 1e3);
        }
        // First token: sampled from the last prefill piece's logits.
        act.prefill_s = act.t0.elapsed().as_secs_f64();
        act.t1 = Instant::now();
        act.phase = Phase::Decode;
        act.rng = Rng(act.req.sampling.seed ^ 0x5EED);
        act.logits = logits;
        let Slot { cached, .. } = &mut self.slots[act.slot];
        let pen = penalized(&act.logits, &act.req.sampling, cached, &[]);
        let row = pen.as_deref().unwrap_or(&act.logits);
        let tok = sample(row, &act.req.sampling, &mut || act.rng.next_f32());
        act.top = if act.req.logprobs {
            Some(token_probs(row, tok, act.req.top_logprobs))
        } else {
            None
        };
        act.tok = tok;
        act.pend = vec![tok];
        act.logits = Vec::new();
        Ok(Progress::More)
    }

    /// One decode step for every session that is decoding, in one pass over the weights (issue #35): the model runs
    /// `Model::decode_batch`, then each session advances its own sampler, cache and events. A finished request is
    /// marked `Phase::Finished`, so the caller drains the marks and never has to keep indices across a removal.
    fn decode_batch_step(&mut self, active: &mut [Active]) -> Result<()> {
        let idxs: Vec<usize> = (0..active.len())
            .filter(|&i| active[i].phase == Phase::Decode)
            .collect();
        if idxs.is_empty() {
            return Ok(());
        }
        let mut order: Vec<usize> = Vec::with_capacity(idxs.len());
        let real = self.model.cfg.real_vocab();
        let mut logits = {
            // One run per session, in slot order: the projections are row independent, so the run order only has to
            // be the same as `order`.
            let wanted: Vec<(usize, usize, u32)> = idxs.iter().map(|&i| (i, active[i].slot, active[i].tok)).collect();
            let mut runs: Vec<(&mut Session, u32)> = Vec::with_capacity(wanted.len());
            for (si, slot) in self.slots.iter_mut().enumerate() {
                if let Some(&(ai, _, tok)) = wanted.iter().find(|(_, s, _)| *s == si) {
                    order.push(ai);
                    runs.push((&mut slot.sess, tok));
                }
            }
            self.model.decode_batch(&mut runs)?
        };
        for (pos, &ai) in order.iter().enumerate() {
            let slot = active[ai].slot;
            let act = &mut active[ai];
            if act.n >= act.max_tokens || self.slots[slot].sess.len + 1 > act.ctx {
                act.phase = Phase::Finished;
                continue;
            }
            let (tok, top) = (act.tok, act.top.take());
            if !act.emit(tok, top) {
                act.phase = Phase::Finished;
                continue;
            }
            mask_pad(&mut logits[pos], real);
            let Slot { cached, .. } = &mut self.slots[slot];
            let pen = penalized(&logits[pos], &act.req.sampling, cached, &[]);
            cached.push(tok);
            let row = pen.as_deref().unwrap_or(&logits[pos]);
            let next = sample(row, &act.req.sampling, &mut || act.rng.next_f32());
            act.top = if act.req.logprobs {
                Some(token_probs(row, next, act.req.top_logprobs))
            } else {
                None
            };
            act.pend = vec![next];
            act.tok = next;
        }
        Ok(())
    }

    /// Runs one decode step: one token, or one verify round when the MTP head is drafting.
    fn decode_step(&mut self, act: &mut Active) -> Result<Progress> {
        let (drafts, draft_pmin, lookup) = (self.drafts, self.draft_pmin, self.lookup);
        let model = &self.model;
        let Slot { sess, cached, .. } = &mut self.slots[act.slot];
        let vocab = model.cfg.vocab;
        let real = model.cfg.real_vocab();
        if act.n >= act.max_tokens || sess.len + 1 > act.ctx {
            return Ok(Progress::Done);
        }
        let (tok, top) = (act.tok, act.top.take());
        if !act.emit(tok, top) {
            return Ok(Progress::Done);
        }
        let spec = drafts > 0
            && sess.len + 2 <= act.ctx
            && (model.has_block_draft() || model.mtp_pending(sess) == act.pend.len());
        // Greedy without penalties and without `logprobs`: the token is the argmax, computed on the device, and
        // the row never crosses the bus. Sampled or penalized requests keep the host fold.
        let greedy = self.step_argmax
            && model.supports_step_argmax()
            && !act.req.logprobs
            && act.req.sampling.temperature <= 0.0
            && !act.req.sampling.penalizes();
        if !spec {
            if greedy {
                let next = model.forward_argmax(sess, &[tok])?;
                cached.push(tok);
                act.pend = vec![next];
                act.tok = next;
                return Ok(Progress::More);
            }
            let mut logits = model.forward(sess, &[tok], None, 1)?;
            mask_pad(&mut logits, real);
            cached.push(tok);
            let pen = penalized(&logits, &act.req.sampling, cached, &[]);
            let row = pen.as_deref().unwrap_or(&logits);
            let next = sample(row, &act.req.sampling, &mut || act.rng.next_f32());
            act.top = if act.req.logprobs {
                Some(token_probs(row, next, act.req.top_logprobs))
            } else {
                None
            };
            act.pend = vec![next];
            act.tok = next;
            return Ok(Progress::More);
        }
        let max = drafts.min(act.ctx - sess.len - 1);
        cached.push(tok);
        let ds = draft_tokens(
            model,
            sess,
            &act.pend,
            max,
            draft_pmin,
            lookup.then_some((&cached[..], &act.req.drafts[..])),
        )?;
        cached.pop();
        let nd = ds.len();
        let toks: Vec<u32> = std::iter::once(tok).chain(ds.iter().copied()).collect();
        if greedy {
            // The verify's rows stay on the device: row `i` predicts `toks[i + 1]`, so draft `i` is accepted when
            // the argmax of row `i` is that draft, and the first row that disagrees gives the fallback token.
            let a = model.forward_verify_argmax(sess, &toks)?;
            let mut acc = 0;
            while acc < nd && a[acc] == ds[acc] {
                acc += 1;
            }
            let y = a[acc];
            model.rollback(sess, nd - acc)?;
            cached.push(tok);
            cached.extend_from_slice(&ds[..acc]);
            cached.push(y);
            for &d in &ds[..acc] {
                if !act.emit(d, None) {
                    return Ok(Progress::Done);
                }
            }
            act.pend = ds[..acc].to_vec();
            act.pend.push(y);
            act.tok = y;
            act.top = None;
            return Ok(Progress::More);
        }
        let mut l = model.forward_verify(sess, &toks)?;
        for row in l.chunks_mut(vocab) {
            mask_pad(row, real);
        }
        cached.push(tok);
        let mut acc = 0;
        let mut y = 0;
        while acc <= nd {
            let row = &l[acc * vocab..(acc + 1) * vocab];
            // A draft accepted in this round is a token of the sequence like any other: the row that samples
            // after it must see it, or the penalty would differ from the serial step.
            let pen = penalized(row, &act.req.sampling, cached, &ds[..acc]);
            let row = pen.as_deref().unwrap_or(row);
            if acc == nd {
                y = sample(row, &act.req.sampling, &mut || act.rng.next_f32());
                break;
            }
            let (t, ok) = sample_spec(row, &act.req.sampling, ds[acc], &mut || act.rng.next_f32());
            if !ok {
                y = t;
                break;
            }
            acc += 1;
        }
        model.rollback(sess, nd - acc)?;
        cached.extend_from_slice(&ds[..acc]);
        // One event per accepted draft, from the verify row that produced each of them: row `i` of the verify
        // predicts `toks[i + 1]`, so draft `i` carries the log-probabilities of slice `i`.
        for i in 0..acc {
            let d = ds[i];
            let t = if act.req.logprobs {
                Some(token_probs(&l[i * vocab..(i + 1) * vocab], d, act.req.top_logprobs))
            } else {
                None
            };
            if !act.emit(d, t) {
                return Ok(Progress::Done);
            }
        }
        // The next token's data, from the row it was sampled off: slice `acc` is the fallback row when draft
        // `acc` was rejected, the last verify row when all `n_drafts` were accepted.
        act.top = if act.req.logprobs {
            Some(token_probs(&l[acc * vocab..(acc + 1) * vocab], y, act.req.top_logprobs))
        } else {
            None
        };
        act.pend = ds[..acc].to_vec();
        act.pend.push(y);
        act.tok = y;
        Ok(Progress::More)
    }
}

/// `begin` refuses a prompt that leaves nothing to prefill: it would sample the first token from an empty logits
/// row. (A prompt is at least one token and `cached` is always a proper prefix of it, so this is a guard, not a
/// branch.)
fn ok_or_empty(pieces: usize) -> Result<()> {
    anyhow::ensure!(pieces > 0, "nothing to prefill");
    Ok(())
}

/// Builds the GPU state of one model. It runs on the GPU thread: `load` must too, because the HIP context is per
/// thread. The warm-up allocates the scratch of one prefill chunk before the first request.
/// Prompt tokens per prefill step when several sessions take turns (`MALLEUS_PREFILL_QUANTUM`).
///
/// Measured on Qwen3.8-27B (gfx1150, 2026-11-18): a neighbour decodes one token per `quantum / prefill_rate` seconds
/// while a long prefill runs, and the long prompt itself pays the smaller piece only when it gets short. A 20 043
/// token prompt alone: 178 tok/s at 2048 and at 256, 155 at 128. The same prompt with a second client generating 48
/// tokens: 114 s / 119 s (2048), 121 s / 76 s (256), 139 s / 46 s (128), against 113 s / 120 s with one session
/// (no interleaving at all). 256 is the point where the neighbour's stream keeps moving (a token every ~1.5 s) and
/// the prefill pays nothing on its own; merging the decode rows into the prefill forward removes the tradeoff
/// entirely (issue #33, v2).
const DEFAULT_PREFILL_QUANTUM: usize = 256;

fn build_worker(spec: &Arc<ModelSpec>, load: &mut impl FnMut(&ModelSpec) -> Result<Model>) -> Result<Worker> {
    let ctx = spec.ctx;
    let model = load(spec)?;
    // A DeepSeek-V4.1 graph (host or device) runs its own state, one request at a time: no session, no warm-up
    // and no device memory to plan, so every session below is skipped.
    if model.is_deepseek() {
        return Ok(Worker {
            spec: spec.clone(),
            drafts: 0,
            draft_pmin: DEFAULT_DRAFT_PMIN,
            step_argmax: false,
            lookup: false,
            max_snaps: 0,
            quantum: 1,
            model,
            slots: Vec::new(),
        });
    }
    // Sessions resident: `MALLEUS_SLOTS`, one session each, with the context split between them as a batched server
    // does. One slot is what this server always did: one session holding the whole context.
    let slots = malleus_model::memplan::sessions().max(1);
    let per_slot = (ctx / slots).max(2);
    let mut sessions = Vec::with_capacity(slots);
    for _ in 0..slots {
        sessions.push(Slot {
            sess: model.new_session(per_slot)?,
            cached: Vec::new(),
            snaps: Vec::new(),
            snap_pool: Vec::new(),
        });
    }
    if slots > 1 {
        eprintln!("slots: {slots} sessions of {per_slot} tokens ({ctx} in all)");
    }
    malleus_hip::print_alloc_ledger("sessions built, nothing warmed up");
    let chunk = prefill_chunk(&model).min(per_slot.saturating_sub(1)).max(1);
    if std::env::var("MALLEUS_WARMUP").map_or(true, |v| v != "0") {
        let n = chunk;
        let toks: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 50000).collect();
        let t0 = Instant::now();
        model.forward(&mut sessions[0].sess, &toks, None, 1)?;
        model.reset_session(&mut sessions[0].sess)?;
        eprintln!("warm-up: {n} tokens in {:.1}s", t0.elapsed().as_secs_f64());
        // The number the pre-flight's estimate is checked against: the weights, one session and the working set of a
        // full chunk are all allocated now.
        eprintln!(
            "memory: {:.1} GiB live on the device after the warm-up",
            malleus_hip::device_bytes_live() as f64 / (1u64 << 30) as f64
        );
        malleus_hip::print_alloc_ledger("after the warm-up");
    }
    // Prompt tokens per step: with one slot the whole chunk, exactly as before, because nobody waits for it; with
    // several, a slice of it, because one piece is what the sessions decoding next to it wait for.
    let quantum = env_or(
        "MALLEUS_PREFILL_QUANTUM",
        if slots == 1 {
            chunk
        } else {
            chunk.min(DEFAULT_PREFILL_QUANTUM)
        },
    )
    .clamp(1, chunk);
    // Prompt snapshots are per session, so the total is split: N sessions with 8 snapshots each would otherwise
    // keep N times the state copies the memory plan was written for.
    let snaps_total: usize = env_or("MALLEUS_PROMPT_SNAPSHOTS", 8);
    let max_snaps = if snaps_total == 0 {
        0
    } else {
        (snaps_total / slots).max(1)
    };
    Ok(Worker {
        spec: spec.clone(),
        drafts: if model.has_mtp() {
            env_or("MALLEUS_SPEC_DRAFTS", DEFAULT_DRAFTS).min(GEMV_MAX_T - 1)
        } else {
            0
        },
        draft_pmin: env_or("MALLEUS_DRAFT_PMIN", DEFAULT_DRAFT_PMIN),
        step_argmax: std::env::var("MALLEUS_STEP_ARGMAX").map_or(true, |v| v != "0"),
        lookup: std::env::var("MALLEUS_LOOKUP").map_or(true, |v| v != "0"),
        max_snaps,
        quantum,
        model,
        slots: sessions,
    })
}

/// Queues what a command asks for: a request goes to the FIFO the scheduler admits from, a load or an unload waits
/// in `deferred` until the requests in flight are done, so it never preempts them.
fn route(
    cmd: Command,
    pending: &mut std::collections::VecDeque<Box<Request>>,
    deferred: &mut std::collections::VecDeque<Command>,
    queued: &AtomicUsize,
) {
    match cmd {
        Command::Request(req) => {
            queued.fetch_sub(1, Ordering::Relaxed);
            pending.push_back(req);
        }
        other => deferred.push_back(other),
    }
}

/// Makes `model` the resident one, loading it when it is not already. The old model has to go first: it is dropped,
/// the device memory comes back, and only then is the new one measured against what is free, because two of these
/// files do not fit together on the machines this engine targets. `notify` is the events channel of the request that
/// triggered the switch, which gets `Event::Loading` before the thread blocks; an explicit load has no such channel.
fn ensure_loaded(
    worker: &mut Option<Worker>,
    model: &Arc<ModelSpec>,
    load: &mut impl FnMut(&ModelSpec) -> Result<Model>,
    status: &Status,
    notify: Option<&tmpsc::UnboundedSender<Event>>,
) -> Result<()> {
    if worker.as_ref().is_some_and(|w| w.spec.id == model.id) {
        return Ok(());
    }
    *worker = None;
    status.set_loading(model);
    if let Some(events) = notify {
        let _ = events.send(Event::Loading {
            model: model.id.to_string(),
        });
    }
    let _ = malleus_hip::synchronize();
    eprintln!("loading model {} from {}", model.id, model.path.display());
    let t0 = Instant::now();
    match build_worker(model, load) {
        Ok(w) => {
            eprintln!("model {} ready in {:.1}s", model.id, t0.elapsed().as_secs_f64());
            status.set_loaded(model);
            *worker = Some(w);
            Ok(())
        }
        Err(e) => {
            status.set_failed(&model.id, format!("{e:#}"));
            Err(e)
        }
    }
}

impl Engine {
    pub fn start(
        initial: Option<Arc<ModelSpec>>,
        mut load: impl FnMut(&ModelSpec) -> Result<Model> + Send + 'static,
    ) -> Result<Engine> {
        let (tx, rx) = mpsc::channel::<Command>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        let queued = Arc::new(AtomicUsize::new(0));
        let q2 = queued.clone();
        let busy_since = Arc::new(AtomicU64::new(0));
        let b2 = busy_since.clone();
        let status = Status::default();
        let s2 = status.clone();
        let slots = malleus_model::memplan::sessions().max(1);
        std::thread::Builder::new().name("malleus-gpu".into()).spawn(move || {
            let status = s2;
            let mut worker: Option<Worker> = None;
            let mut pending: std::collections::VecDeque<Box<Request>> = std::collections::VecDeque::new();
            let mut deferred: std::collections::VecDeque<Command> = std::collections::VecDeque::new();
            let mut active: Vec<Active> = Vec::new();
            if let Some(spec) = initial {
                match ensure_loaded(&mut worker, &spec, &mut load, &status, None) {
                    Ok(()) => {}
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                }
            }
            let _ = ready_tx.send(Ok(()));
            loop {
                if active.is_empty() && pending.is_empty() && deferred.is_empty() {
                    // Nothing to do: sleep on the channel, as an idle service must.
                    match rx.recv() {
                        Ok(cmd) => route(cmd, &mut pending, &mut deferred, &q2),
                        Err(_) => break,
                    }
                }
                while let Ok(cmd) = rx.try_recv() {
                    route(cmd, &mut pending, &mut deferred, &q2);
                }
                // A load or an unload sits behind the queue and the requests in flight, as it always has.
                if active.is_empty() && pending.is_empty() {
                    while let Some(cmd) = deferred.pop_front() {
                        match cmd {
                            Command::Load(model) => {
                                if let Err(e) = ensure_loaded(&mut worker, &model, &mut load, &status, None) {
                                    eprintln!("loading model {} failed: {e:#}", model.id);
                                }
                            }
                            Command::Unload(id) => {
                                if worker.as_ref().is_some_and(|w| w.spec.id == id) {
                                    worker = None;
                                    let _ = malleus_hip::synchronize();
                                    status.set_unloaded(&id);
                                    eprintln!("unloaded model {id}");
                                }
                            }
                            Command::Request(_) => unreachable!("requests are routed to `pending`"),
                        }
                    }
                }
                // Admit from the FIFO: one model resident, one request per free slot.
                while active.len() < slots {
                    let Some(head) = pending.front() else { break };
                    let want = head.model.clone();
                    if !worker.as_ref().is_some_and(|w| w.spec.id == want.id) {
                        if !active.is_empty() {
                            // The resident sessions come first; the switch waits for them.
                            break;
                        }
                        let req = pending.pop_front().expect("a queued request");
                        let events = req.events.clone();
                        if let Err(e) = ensure_loaded(&mut worker, &want, &mut load, &status, Some(&events)) {
                            let _ = events.send(Event::Error(format!("loading model `{}`: {e:#}", want.id)));
                        } else {
                            pending.push_front(req);
                        }
                        continue;
                    }
                    let req = pending.pop_front().expect("a queued request");
                    let events = req.events.clone();
                    // A DeepSeek-V4.1 graph has no session and no interleaving: the one request runs to completion
                    // on this thread and the queue waits behind it. No watchdog either: a token is the graph's
                    // speed (12.6 s on the host, ~0.1 s on the device), not a hang.
                    if worker
                        .as_ref()
                        .expect("a worker is loaded before serving")
                        .model
                        .is_deepseek()
                    {
                        let w = worker.as_mut().expect("a worker is loaded before serving");
                        if let Err(e) = w.serve_deepseek(*req) {
                            let _ = events.send(Event::Error(format!("{e:#}")));
                        }
                        continue;
                    }
                    // The slot whose session already holds the longest prefix of this prompt, else any free one.
                    let busy: Vec<usize> = active.iter().map(|a| a.slot).collect();
                    let free: Vec<usize> = (0..slots).filter(|s| !busy.contains(s)).collect();
                    let slot = {
                        let w = worker.as_ref().expect("a worker is loaded before serving");
                        let cached: Vec<&[u32]> = w.slots.iter().map(|s| &s.cached[..]).collect();
                        pick_slot(&free, &cached, &req.prompt).unwrap_or_else(|| free.first().copied().unwrap_or(0))
                    };
                    let w = worker.as_mut().expect("a worker is loaded before serving");
                    match w.begin(*req, slot) {
                        Ok(act) => {
                            // The watchdog counts the request, not the model load: a slow disk must not look like a
                            // hung kernel.
                            if active.is_empty() {
                                b2.store(now_ms(), Ordering::Relaxed);
                            }
                            active.push(act);
                        }
                        Err(e) => {
                            let _ = events.send(Event::Error(format!("{e:#}")));
                        }
                    }
                }
                if active.is_empty() {
                    b2.store(0, Ordering::Relaxed);
                    continue;
                }
                // One round: every active session gets one step, in arrival order. A session that finishes frees its
                // slot for the next round, so a queued request joins without waiting for the others to drain.
                let mut i = 0;
                while i < active.len() {
                    if active[i].phase != Phase::Prefill {
                        i += 1;
                        continue;
                    }
                    let w = worker.as_mut().expect("a worker is loaded before serving");
                    match w.prefill_step(&mut active[i]) {
                        Ok(Progress::More) => i += 1,
                        Ok(Progress::Done) => {
                            let act = active.remove(i);
                            let _ = act.req.events.send(Event::Done(act.finish()));
                        }
                        Err(e) => {
                            let act = active.remove(i);
                            if let Some(w) = worker.as_mut() {
                                let _ = w.slots[act.slot].reset(&w.model);
                            }
                            let _ = act.req.events.send(Event::Error(format!("{e:#}")));
                        }
                    }
                }
                // The decoding sessions share one pass over the weights when the model supports it; otherwise they
                // take turns, one step each per round (v1 of #33).
                let batchable = active.iter().any(|a| a.phase == Phase::Decode)
                    && worker.as_ref().is_some_and(|w| w.model.can_batch());
                if batchable {
                    let w = worker.as_mut().expect("a worker is loaded before serving");
                    match w.decode_batch_step(&mut active) {
                        Ok(()) => {
                            let mut i = 0;
                            while i < active.len() {
                                if active[i].phase == Phase::Finished {
                                    let act = active.remove(i);
                                    let _ = act.req.events.send(Event::Done(act.finish()));
                                } else {
                                    i += 1;
                                }
                            }
                        }
                        Err(e) => {
                            let message = format!("{e:#}");
                            let mut i = 0;
                            while i < active.len() {
                                if active[i].phase != Phase::Decode {
                                    i += 1;
                                    continue;
                                }
                                let act = active.remove(i);
                                if let Some(w) = worker.as_mut() {
                                    let _ = w.slots[act.slot].reset(&w.model);
                                }
                                let _ = act.req.events.send(Event::Error(message.clone()));
                            }
                        }
                    }
                } else {
                    let mut i = 0;
                    while i < active.len() {
                        if active[i].phase != Phase::Decode {
                            i += 1;
                            continue;
                        }
                        let w = worker.as_mut().expect("a worker is loaded before serving");
                        match w.decode_step(&mut active[i]) {
                            Ok(Progress::More) => i += 1,
                            Ok(Progress::Done) => {
                                let act = active.remove(i);
                                let _ = act.req.events.send(Event::Done(act.finish()));
                            }
                            Err(e) => {
                                let act = active.remove(i);
                                if let Some(w) = worker.as_mut() {
                                    let _ = w.slots[act.slot].reset(&w.model);
                                }
                                let _ = act.req.events.send(Event::Error(format!("{e:#}")));
                            }
                        }
                    }
                }
                if active.is_empty() {
                    b2.store(0, Ordering::Relaxed);
                }
            }
        })?;
        ready_rx.recv()??;
        start_watchdog(busy_since.clone())?;
        Ok(Engine {
            tx,
            status,
            queued,
            busy_since,
        })
    }

    pub fn submit(&self, req: Request) -> Result<()> {
        self.queued.fetch_add(1, Ordering::Relaxed);
        self.tx
            .send(Command::Request(Box::new(req)))
            .map_err(|_| anyhow::anyhow!("engine thread stopped"))
    }

    /// Loads `model` on the GPU thread without waiting for a request. Returns as soon as the command is queued: a load
    /// of a file this size takes a minute, far longer than a client keeps a request open. Follow [`Status`] (or
    /// [`Engine::loaded`]) for the outcome.
    pub fn load(&self, model: Arc<ModelSpec>) -> Result<()> {
        self.tx
            .send(Command::Load(model))
            .map_err(|_| anyhow::anyhow!("engine thread stopped"))
    }

    /// Drops the model with this id when it is the resident one. Queued behind the request in flight, if any.
    pub fn unload(&self, id: Arc<str>) -> Result<()> {
        self.tx
            .send(Command::Unload(id))
            .map_err(|_| anyhow::anyhow!("engine thread stopped"))
    }

    /// The status the API reads: what is resident, what is loading, the last failure, and the transition stream.
    pub fn status(&self) -> &Status {
        &self.status
    }

    /// The model resident now, if any.
    pub fn loaded(&self) -> Option<Arc<ModelSpec>> {
        self.status.loaded()
    }

    /// The model being loaded now, if any.
    pub fn loading(&self) -> Option<Arc<ModelSpec>> {
        self.status.loading()
    }

    /// How long the request in flight has been running, or `None` when the worker is idle.
    pub fn busy_for(&self) -> Option<Duration> {
        match self.busy_since.load(Ordering::Relaxed) {
            0 => None,
            t => Some(Duration::from_millis(now_ms().saturating_sub(t))),
        }
    }
}

/// Ends the process when a request overruns `MALLEUS_REQUEST_TIMEOUT` seconds. HIP has no way to cancel a kernel
/// already running, so a request that overruns cannot be abandoned in place: the alternative to restarting is
/// leaving the only GPU worker busy forever, with the service still answering `/v1/models` and every later request
/// queued behind it. Exiting is what makes the unit's `Restart=on-failure` recover the service on its own.
fn start_watchdog(busy_since: Arc<AtomicU64>) -> Result<()> {
    let timeout = env_or("MALLEUS_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT);
    if timeout == 0 {
        return Ok(());
    }
    std::thread::Builder::new()
        .name("malleus-watchdog".into())
        .spawn(move || {
            let limit = Duration::from_secs(timeout);
            loop {
                std::thread::sleep(Duration::from_millis(500));
                let started = busy_since.load(Ordering::Relaxed);
                if started == 0 {
                    continue;
                }
                let running = Duration::from_millis(now_ms().saturating_sub(started));
                if running > limit {
                    eprintln!(
                        "watchdog: a request has been running for {:.0}s, over MALLEUS_REQUEST_TIMEOUT={timeout}s; \
                         a GPU kernel cannot be cancelled, so the process exits and the service restarts",
                        running.as_secs_f64()
                    );
                    std::process::exit(70);
                }
            }
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pieces of a prompt tile it exactly: no gap, no overlap, never more than the quantum, one token at least.
    #[test]
    fn prompt_pieces_tile_the_prompt() {
        for (start, end, q) in [(0usize, 5000usize, 2048usize), (0, 1, 8), (17, 18, 4), (0, 4096, 4096)] {
            let p = prompt_pieces(start, end, q);
            assert_eq!(p.first().unwrap().0, start, "the pieces start at the first token");
            let last = p.last().unwrap();
            assert_eq!(last.0 + last.1, end, "the pieces reach the last token");
            for w in p.windows(2) {
                assert_eq!(w[0].0 + w[0].1, w[1].0, "contiguous: no gaps and no overlaps");
            }
            for (s, len) in &p {
                assert!(*len > 0 && *len <= q, "piece {s}+{len} over the quantum {q}");
            }
        }
        assert!(prompt_pieces(5, 5, 16).is_empty(), "nothing left to prefill");
    }

    /// A conversation keeps hitting its own prompt cache: the free slot that already holds a prefix of the prompt
    /// wins, and with nothing in common the lowest index does (a fresh request does not walk from slot to slot).
    #[test]
    fn the_slot_with_the_longest_prefix_wins() {
        let a = [1u32, 2, 3, 4, 5, 6];
        let b = [1u32, 2, 3, 9];
        let cached: [&[u32]; 2] = [&a, &b];
        assert_eq!(pick_slot(&[0, 1], &cached, &[1, 2, 3, 4, 5, 7]), Some(0));
        assert_eq!(pick_slot(&[0, 1], &cached, &[1, 2, 3, 9, 9]), Some(1));
        assert_eq!(pick_slot(&[0, 1], &cached, &[7, 7]), Some(0));
        assert_eq!(pick_slot(&[1], &cached, &[1, 2, 3, 4]), Some(1));
        assert_eq!(pick_slot(&[], &cached, &[1]), None, "no free slot, no admission");
    }

    #[test]
    fn image_pieces_cover_the_prompt_and_never_cut_a_block() {
        // Two image blocks (10..18 and 33..45) inside a 50-token prompt, chunk 16.
        let blocks = [(10, 18), (33, 45)];
        let p = image_pieces(50, &blocks, 16);
        assert_eq!(p.first().unwrap().0, 0, "the pieces start at the first token");
        assert_eq!(p.last().unwrap().1, 50, "the pieces reach the last token");
        for w in p.windows(2) {
            assert_eq!(w[0].1, w[1].0, "contiguous: no gaps and no overlaps");
        }
        for (s, e) in &p {
            assert!(e > s && e - s <= 16, "piece {s}..{e} over the chunk");
            for (bs, be) in blocks {
                // A piece may contain a whole block, and may begin or end exactly on an edge; what it must never do
                // is put a boundary strictly inside one, because `forward_mm` copies the insert whole.
                assert!(
                    !(*s > bs && *s < be) && !(*e > bs && *e < be),
                    "piece {s}..{e} cuts the image block {bs}..{be}"
                );
            }
        }
        // Greedy: 0-10 stops before the first block, 10-26 takes it whole, 26-33 stops before the second, 33-49
        // takes it whole, 49-50 is the tail.
        assert_eq!(p, vec![(0, 10), (10, 26), (26, 33), (33, 49), (49, 50)]);
    }

    #[test]
    fn image_pieces_without_images_are_the_plain_chunks() {
        let p = image_pieces(70, &[], 16);
        assert_eq!(p, vec![(0, 16), (16, 32), (32, 48), (48, 64), (64, 70)]);
        assert!(image_pieces(0, &[], 16).is_empty(), "an empty prompt has no piece");
        // A block ending on the chunk boundary adds no cut; a block wider than the chunk still fits a piece of its
        // own, and the tokens after it go back to the chunk size.
        assert_eq!(image_pieces(32, &[(16, 32)], 16), vec![(0, 16), (16, 32)]);
        assert_eq!(image_pieces(40, &[(4, 34)], 16), vec![(0, 4), (4, 34), (34, 40)]);
        // A block that already fits inside a piece does not cut it.
        assert_eq!(image_pieces(10, &[(2, 4)], 16), vec![(0, 10)]);
    }

    #[test]
    fn a_padded_row_is_never_sampled() {
        // The head of this checkpoint has 201 024 rows and 200 058 tokens: the padded tail can win an argmax and a
        // softmax alike, and the id it returns has no tokenizer entry.
        let (real, vocab) = (6usize, 8usize);
        let mut row = vec![0.0f32; vocab];
        row[real + 1] = 9.0;
        mask_pad(&mut row, real);
        assert!(row[real..].iter().all(|v| *v == f32::NEG_INFINITY));
        let s = Sampling {
            temperature: 1.0,
            ..Default::default()
        };
        for r in [0.0, 0.5, 0.999] {
            assert!(sample(&row, &s, &mut || r) < real as u32, "sampled a padded id");
        }
        assert!(sample(&row, &Sampling::default(), &mut || 0.5) < real as u32);
        // And a mask wider than the row (a family whose head is not padded) is a no-op.
        let mut wide = vec![1.0f32; 4];
        mask_pad(&mut wide, 4);
        assert_eq!(wide, vec![1.0f32; 4]);
    }

    #[test]
    fn repeat_penalty_pushes_a_repeated_token_down() {
        // Token 3 keeps winning; after it has been generated it must lose to token 5.
        let mut logits = vec![0.0f32; 8];
        logits[3] = 4.0;
        logits[5] = 3.0;
        let greedy = Sampling::default();
        assert_eq!(
            sample(&logits, &greedy, &mut || 0.5),
            3,
            "without a penalty the argmax wins"
        );
        let s = Sampling {
            repeat_penalty: 1.5,
            repeat_last_n: 64,
            ..Default::default()
        };
        let row = penalized(&logits, &s, &[3, 7, 3], &[]).expect("penalty active");
        assert_eq!(sample(&row, &s, &mut || 0.5), 5, "4.0 / 1.5 = 2.67 < 3.0");
        // Only the last `repeat_last_n` tokens count: with a window of 1 the window is the 7 only, the 3 is out and
        // the argmax stays where it was.
        let narrow = Sampling {
            repeat_penalty: 1.5,
            repeat_last_n: 1,
            ..Default::default()
        };
        let row = penalized(&logits, &narrow, &[3, 7], &[]).expect("the window still holds the 7");
        assert_eq!(sample(&row, &narrow, &mut || 0.5), 3, "the 3 is outside the window");
        // Off by default and with the penalties off the row is not even copied.
        assert!(penalized(&logits, &Sampling::default(), &[3], &[]).is_none());
        // Frequency and presence are additive in logit space: three occurrences of 3 cost 3 * 0.5 and 0.2.
        let s = Sampling {
            frequency_penalty: 0.5,
            presence_penalty: 0.2,
            ..Default::default()
        };
        let row = penalized(&logits, &s, &[3, 3, 3], &[]).expect("penalty active");
        assert!((row[3] - (4.0 - 1.5 - 0.2)).abs() < 1e-6, "got {}", row[3]);
        // A draft accepted in a speculative round is part of the window like any other token.
        let s = Sampling {
            repeat_penalty: 1.5,
            ..Default::default()
        };
        assert!(penalized(&logits, &s, &[7], &[3]).is_some());
    }

    #[test]
    fn greedy_and_top_k() {
        let l = vec![0.0, 5.0, 1.0, 4.9];
        assert_eq!(sample(&l, &Sampling::default(), &mut || 0.5), 1);
        let s = Sampling {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        };
        assert_eq!(sample(&l, &s, &mut || 0.99), 1);
        let s = Sampling {
            temperature: 1.0,
            top_k: 2,
            ..Default::default()
        };
        assert_eq!(sample(&l, &s, &mut || 0.99), 3);
    }

    #[test]
    fn token_probs_are_the_full_softmax_in_log_space() {
        // lse = ln(&Sigma; exp), and every reported probability is exp(logit - lse); the alternatives' probabilities
        // (and the token's own, which may be outside the top-N) are all proper probabilities.
        let l = vec![0.0, 1.0, 2.0, 3.0, -0.5];
        let p = token_probs(&l, 3, 3);
        assert_eq!(p.id, 3);
        let lse = (l.iter().map(|&v| (v as f64).exp()).sum::<f64>()).ln();
        let mut expect: Vec<(u32, f32)> = (0..l.len() as u32)
            .map(|i| (i, (l[i as usize] as f64 - lse) as f32))
            .collect();
        expect.sort_by(|&(ia, la), &(ib, lb)| (lb as f64 - la as f64).total_cmp(&0.0).then(ia.cmp(&ib)));
        assert_eq!(p.top, expect[0..3].to_vec(), "most likely first, top-N only");
        let own = expect.iter().find(|(i, _)| *i == p.id).unwrap();
        assert!((p.logprob - own.1).abs() < 1e-6, "own logprob follows the full softmax");
        let sum: f64 = p.top.iter().map(|&(_, lp)| lp.exp() as f64).sum();
        assert!(
            sum <= 1.0 + 1e-6,
            "the alternatives' probabilities sum, exp'd, stay below 1"
        );
        for (_, lp) in &p.top {
            assert!(*lp <= 1e-6, "no probability above 1");
        }
        // The sampled token's own probability is reported even when it is not in the top-N.
        let own4 = expect.iter().find(|(i, _)| *i == 4).unwrap();
        let p4 = token_probs(&l, 4, 1);
        assert!(
            (p4.logprob - own4.1).abs() < 1e-6,
            "id outside the top-N still gets its logprob"
        );
        assert_eq!(p4.top.len(), 1);
        // No alternatives asked: only the token's own log-probability.
        let p0 = token_probs(&l, 3, 0);
        assert!(p0.top.is_empty());
        assert!((p0.logprob - own.1).abs() < 1e-6);
    }

    #[test]
    fn speculative_sampling_keeps_the_distribution() {
        let l = vec![0.0, 1.0, 2.0, 0.5];
        let s = Sampling {
            temperature: 1.0,
            ..Default::default()
        };
        let mut rng = Rng(7);
        let n = 200_000;
        let mut counts = [0usize; 4];
        for i in 0..n {
            let (t, _) = sample_spec(&l, &s, (i % 4) as u32, &mut || rng.next_f32());
            counts[t as usize] += 1;
        }
        let z: f32 = l.iter().map(|v| v.exp()).sum();
        for (c, v) in counts.iter().zip(&l) {
            assert!((*c as f32 / n as f32 - v.exp() / z).abs() < 0.005, "{counts:?}");
        }
        assert_eq!(sample_spec(&l, &Sampling::default(), 2, &mut || 0.5), (2, true));
        assert_eq!(sample_spec(&l, &Sampling::default(), 1, &mut || 0.5), (2, false));
    }
}

#[cfg(test)]
mod lookup_tests {
    use super::lookup_drafts;

    /// The lookup takes the longest continuation it can find, not the most recent: a tool result sits in the prompt
    /// with its line numbers, so the match there stops at the next line break while the original text keeps going.
    #[test]
    fn longest_continuation_wins() {
        let hist = [10u32, 11, 12, 20, 10, 11, 12]; // in-prompt match continues with one token
        let original = [10u32, 11, 12, 30, 31, 32, 33, 34];
        let got = lookup_drafts(&hist, &[original.to_vec()], 3, 5);
        assert_eq!(
            got,
            vec![30, 31, 32, 33, 34],
            "the short in-prompt match must not shadow the long one"
        );
    }

    #[test]
    fn falls_back_to_the_prompt() {
        let hist = [4u32, 5, 6, 100, 200, 300, 4, 5, 6];
        let got = lookup_drafts(&hist, &[vec![42, 43, 44]], 3, 3);
        assert_eq!(got, vec![100, 200, 300]);
    }

    #[test]
    fn no_match_and_short_history() {
        let hist = [1u32, 2, 3, 4];
        assert!(lookup_drafts(&hist, &[vec![9, 9, 9]], 3, 5).is_empty());
        assert!(lookup_drafts(&[1u32, 2], &[], 3, 5).is_empty());
        assert!(lookup_drafts(&[1u32, 2, 3, 1, 2, 3], &[], 3, 0).is_empty());
    }
}

#[cfg(test)]
mod status_tests {
    use super::*;

    fn spec(id: &str) -> Arc<ModelSpec> {
        Arc::new(ModelSpec {
            id: Arc::from(id),
            path: std::path::PathBuf::from("/dev/null"),
            ctx: 8,
            eos: Arc::from(Vec::<u32>::new()),
        })
    }

    /// The state the API reads: one resident model at a time, and a failure that stays with the id it happened on
    /// until that id is loaded again. A failure of one model must not leak onto another.
    #[test]
    fn a_failure_belongs_to_the_model_that_failed() {
        let st = Status::default();
        st.set_loading(&spec("a"));
        assert_eq!(st.loading().map(|s| s.id.to_string()), Some("a".to_string()));
        assert!(st.loaded().is_none());
        st.set_loaded(&spec("a"));
        assert_eq!(st.loaded().map(|s| s.id.to_string()), Some("a".to_string()));
        assert!(st.loading().is_none());

        st.set_failed(&spec("b").id, "no memory".to_string());
        assert_eq!(st.failure().map(|f| f.id.to_string()), Some("b".to_string()));

        // Dropping `a` leaves `b`'s failure in place.
        st.set_unloaded(&spec("a").id);
        assert!(st.loaded().is_none());
        assert_eq!(st.failure().map(|f| f.id.to_string()), Some("b".to_string()));

        // A new attempt at `b` clears the failure it belongs to, and it is cleared while the load runs, not after.
        st.set_loading(&spec("b"));
        assert!(st.failure().is_none());
        assert!(st.loaded().is_none());
        assert_eq!(st.loading().map(|s| s.id.to_string()), Some("b".to_string()));
    }

    /// Every setter broadcasts, in order, so a watching client sees the same transitions a polling one does.
    #[test]
    fn transitions_are_broadcast_in_order() {
        let st = Status::default();
        let mut rx = st.subscribe();
        st.set_loading(&spec("a"));
        st.set_loaded(&spec("a"));
        st.set_loaded(&spec("b"));
        st.set_unloaded(&spec("b").id);
        let seen: Vec<(String, ModelState)> = (0..4)
            .map(|_| {
                let e = rx.try_recv().expect("a broadcast transition");
                (e.id.to_string(), e.state)
            })
            .collect();
        assert_eq!(
            seen,
            [
                ("a".to_string(), ModelState::Loading),
                ("a".to_string(), ModelState::Loaded),
                ("b".to_string(), ModelState::Loaded),
                ("b".to_string(), ModelState::Unloaded),
            ]
        );
        assert!(rx.try_recv().is_err());
    }
}
