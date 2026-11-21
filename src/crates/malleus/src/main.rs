#![forbid(unsafe_code)]

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use malleus_gguf::gguf::Gguf;
use malleus_gguf::safetensors::SafeTensors;
use malleus_model::kernels::{Kernels, select_device};
use malleus_model::vision::{VisionConfig, VisionModel};
use malleus_model::{Config, Family, GEMV_MAX_T, LoadOptions, Model, Session, dense_w4a4};
use malleus_sched::{Catalog, ModelSpec};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// `malleus --version`: the version the build was made from (the git tag, set by `build.rs`) and the license with
/// where the source is. Apache-2.0 asks a redistribution to carry the attribution notices of `NOTICE`, so pointing at
/// them is the part of the notice that travels with the binary.
const LONG_VERSION: &str = concat!(
    env!("MALLEUS_BUILD_VERSION"),
    "\nCopyright 2026 DuckPage di Olimpi e Rusu S.n.c. and the Malleus contributors.\n",
    "License Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>.\n",
    "Attribution notices: NOTICE. Third-party components: THIRD-PARTY-NOTICES.md.\n",
    "There is NO WARRANTY, to the extent permitted by law.\n",
    "Source: https://github.com/duckpage/malleus"
);

#[derive(Parser)]
#[command(name = "malleus", version, long_version = LONG_VERSION, about = "Malleus inference engine")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List HIP devices
    Devices,
    /// Per-position perplexity, top-1 agreement and KL against scripts/eval_ref.py
    Eval {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        set: PathBuf,
        #[arg(long)]
        blobs: PathBuf,
        /// Tokens per forward call (prefill chunk)
        #[arg(long, default_value_t = 256)]
        chunk: usize,
    },
    /// Compare the engine against golden references produced by scripts/golden.py
    Golden {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        golden: PathBuf,
        #[arg(long)]
        blobs: PathBuf,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        #[arg(long)]
        greedy: bool,
        #[arg(long)]
        only: Option<String>,
        /// Feed the prompts one token at a time (the decode weights and kernels) instead of one prefill
        #[arg(long)]
        decode: bool,
    },
    /// Convert a safetensors bf16 model directory to an Malleus model file (.malleus)
    Convert {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Also store an MLS8 copy of the weights for the prefill (`<name>.prefill`): twice the file, ~35 GB more
        /// on the device for a 35B MoE, in exchange for reading the prompt at 8 bits instead of 4 (measured: eval
        /// top-1 93.87 -> 97.02 %). Off by default: a single MLS4 copy is the faster file.
        #[arg(long)]
        prefill_int8: bool,
        /// Deprecated: the single copy is the default now, this flag does nothing.
        #[arg(long, hide = true)]
        no_prefill_int8: bool,
        /// With `--prefill-int8`, leave the routed experts without their MLS8 copy: their prefill reads MLS4
        /// nibbles with int8 activations (W4A8), halving the bytes of the widest prefill block while the dense
        /// linears keep their W8A8 path.
        #[arg(long)]
        no_prefill_int8_experts: bool,
        /// Comma-separated layer indices whose FFN stays MLS8 in decode
        #[arg(long, value_delimiter = ',')]
        keep_int8_layers: Vec<usize>,
        /// Directory with GPTQ-calibrated MLS4 tensors written by scripts/calibrate.py
        #[arg(long)]
        mls4_from: Option<PathBuf>,
        /// MoE layers whose gate/up are stored at 3 bit: `family` (the range its model document chose), `off`,
        /// or an inclusive `A-B`
        #[arg(long, default_value = "family")]
        expert_mls3: String,
        /// MoE layers whose routed gate/up are stored at 2 bit: `family` (Qwen3.8-Flash-Next: every routed expert
        /// of the main stack), `off`, or an inclusive `A-B`
        #[arg(long, default_value = "off")]
        expert_mls2: String,
        /// Also store the routed down projection at 2 bit instead of 4
        #[arg(long)]
        mls2_down: bool,
        /// Also store the hyper-connection mixing as MLS4 for decode, keeping the MLS8 copy for prefill
        #[arg(long)]
        hc_mls4: bool,
    },
    /// Dequantise a checkpoint whose routed experts are packed (NVFP4 / MXFP8) into a bf16 safetensors
    /// directory: the file `convert` and the layer-by-layer reference read
    Dequant {
        /// The safetensors directory to read
        #[arg(long)]
        model: PathBuf,
        /// The safetensors directory to write
        #[arg(long)]
        out: PathBuf,
    },
    /// Rewrite a model file with a single weight copy: drop the MLS8 prefill copies, keep every other tensor
    /// byte-for-byte. For a file converted before `--prefill-int8` became opt-in (no bf16 source needed).
    Slim {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Keep the prefill copy of the tensors whose name contains one of these (comma-separated), e.g.
        /// `hyper_connection`: the prefill of that family has no 4-bit form yet and would lose quality
        #[arg(long, value_delimiter = ',')]
        keep: Vec<String>,
    },
    /// Start the OpenAI / Anthropic compatible API server
    Serve {
        /// Directory whose `*.malleus` files the server exposes by file stem
        #[arg(long, env = "MALLEUS_MODELS_DIR")]
        models_dir: PathBuf,
        /// Catalog id to preload and use when a request omits `model` (default: the only model, if there is one)
        #[arg(long, env = "MALLEUS_DEFAULT_MODEL")]
        default_model: Option<String>,
        #[arg(long, env = "MALLEUS_HOST", default_value = "0.0.0.0")]
        host: String,
        #[arg(long, env = "MALLEUS_PORT", default_value_t = 8090)]
        port: u16,
        #[arg(long, env = "MALLEUS_API_KEY")]
        api_key: Option<String>,
        #[arg(long, env = "MALLEUS_CTX", default_value_t = 32768)]
        ctx: usize,
        /// Reasoning default when the request does not say (Qwen default: on)
        #[arg(long, env = "MALLEUS_DEFAULT_THINKING", default_value_t = true, action = clap::ArgAction::Set)]
        default_thinking: bool,
        #[arg(long, env = "MALLEUS_MAX_TOKENS", default_value_t = 8192)]
        max_tokens: usize,
    },
    /// Inspect a converted DFlash speculative draft: geometry, context layers, block and verify width
    DraftInfo {
        #[arg(long)]
        model: PathBuf,
    },
    /// Write the Latin-script token ids (one per line) for MALLEUS_DRAFT_VOCAB_FILE
    DraftVocab {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Exit 0 if the local server answers /health (for container health checks)
    Health {
        #[arg(long, env = "MALLEUS_PORT", default_value_t = 8090)]
        port: u16,
    },
    /// Measure prefill and decode speed on synthetic tokens
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "512")]
        prompt: Vec<usize>,
        #[arg(long, default_value_t = 32)]
        decode: usize,
        #[arg(long, default_value_t = 512)]
        chunk: usize,
        /// Also time steps of this many tokens (speculative verify cost)
        #[arg(long)]
        verify: Option<usize>,
        /// Grow one session to these context depths and time `chunk` prefill tokens and `decode` tokens at each
        #[arg(long, value_delimiter = ',')]
        depth: Vec<usize>,
    },
    /// Compare the WMMA prefill GEMM against the reference MLS4 GEMM on random data
    KernelTest,
    /// Check that a speculative verify is exact: the logits of 2..=6 tokens verified in one step against the same
    /// tokens fed one at a time, bit for bit. Exits non-zero when a row differs.
    SpecExact {
        #[arg(long)]
        model: PathBuf,
        /// A golden file (golden/<model>/golden.json): its token ids are the text
        #[arg(long)]
        golden: PathBuf,
        /// Which prompt of the golden file
        #[arg(long, default_value = "long-4500")]
        only: String,
        /// Tokens fed one at a time before the verify step (comma list: several depths)
        #[arg(long, value_delimiter = ',', default_value = "64")]
        prefix: Vec<usize>,
        /// Only this verify width (default: every width from 2)
        #[arg(long)]
        width: Option<usize>,
        /// Tokens generated twice after the first prefix, serially and through the speculative loop (drafts,
        /// verify, rollback), comparing every committed row (0: skip)
        #[arg(long, default_value_t = 96)]
        replay: usize,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
    },
    /// One pass over the weights for several sessions must give every row what the serial step gives it: the same
    /// prompt is advanced serially in one session and batched in `sessions - 1` others, comparing the argmax of every
    /// row at every step (issue #35).
    BatchExact {
        #[arg(long)]
        model: PathBuf,
        /// A golden file (golden/<model>/golden.json): its token ids are the prompt
        #[arg(long)]
        golden: PathBuf,
        /// Which prompt of the golden file
        #[arg(long, default_value = "long-context")]
        only: String,
        /// Sessions in the batch (one of them is the serial reference is not counted: 2 means one + one)
        #[arg(long, default_value_t = 2)]
        sessions: usize,
        /// Decode steps to compare
        #[arg(long, default_value_t = 32)]
        steps: usize,
        /// Prompt tokens to keep (the golden prompts are short; this only truncates)
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
    },
    /// Measure GPU memory bandwidth (read and copy)
    BenchBw {
        #[arg(long, default_value_t = 2.0)]
        gb: f64,
    },
    /// Generate a reply to a single user prompt (greedy)
    Run {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, short)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long)]
        think: bool,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// Speculative decoding with this many MTP drafts per step (0: off)
        #[arg(long, default_value_t = 0)]
        spec: usize,
        /// Debug: read whitespace-separated token ids from a file instead of encoding the prompt text
        #[arg(long)]
        prompt_ids: Option<PathBuf>,
        /// Diagnostics (#10): write the logits of the last prefill position as raw f32, to measure how sensitive the
        /// model is to removing a block of context.
        #[arg(long)]
        dump_logits: Option<PathBuf>,
    },
}

/// A file is a converted Malleus model; a directory is an upstream safetensors model.
fn is_converted(p: &Path) -> bool {
    p.is_file()
}

/// Opens a converted model and checks that it is an Malleus file of a supported family.
/// Prints what the draft declares and proves the weights load: the checkpoint is 738 MB, so this is the cheap way to
/// check a conversion before the forward exists (issue #12).
fn draft_info(path: &Path) -> Result<()> {
    let g = Gguf::open(path)?;
    let arch = g.malleus_arch()?;
    anyhow::ensure!(arch == "dflash", "`{arch}` is not a DFlash draft");
    let cfg = g.meta_str("openmalleus.hf_config")?;
    malleus_model::dflash::ensure_is_draft(cfg)?;
    let k = malleus_model::kernels::Kernels::load(&malleus_model::kernels::select_device()?)?;
    let draft = malleus_model::dflash::DflashDraft::load(cfg, &g, &k)?;
    for line in draft.describe() {
        println!("{line}");
    }
    Ok(())
}

fn open_converted(path: &Path) -> Result<Gguf> {
    let g = Gguf::open(path)?;
    let arch = g.malleus_arch()?;
    anyhow::ensure!(
        matches!(arch, "qwen35" | "qwen4exp" | "ling3" | "hy4" | "inkling2"),
        "unsupported model family `{arch}`"
    );
    Ok(g)
}

/// What the run is about to ask the machine for, before it asks (see `malleus_model::memplan`).
///
/// The weights are exact (the file size, an upper bound when only the decode copies are loaded) and the
/// per-session state is exact (it mirrors `Model::new_session`); the working set of a chunk is a reserve
/// the caller sizes with `MALLEUS_MEM_RESERVE_GIB`. `load_model` prints what the run then measured.
///
/// `chunk` is the largest forward call the run makes (`None`: the scheduler's prefill chunk).
fn preflight(path: &Path, ctx: usize, chunk: Option<usize>) -> Result<()> {
    let prefill_int8 = std::env::var("MALLEUS_PREFILL_WEIGHTS").map_or(true, |v| v != "none");
    let mut tables = 0;
    let (cfg, weights) = if is_converted(path) {
        let g = open_converted(path)?;
        let cfg = Config::from_json(
            g.meta_str("openmalleus.hf_config")?,
            g.meta_str("openmalleus.generation_config").ok(),
        )?;
        let weights = malleus_model::memplan::weights_on_device(&g, prefill_int8);
        tables = malleus_model::memplan::tables_on_disk(&g);
        eprintln!(
            "memory: {:.1} GiB of the {:.1} GiB file go on the device (the hashed tables stay on disk)",
            weights as f64 / (1u64 << 30) as f64,
            std::fs::metadata(path)?.len() as f64 / (1u64 << 30) as f64
        );
        (cfg, weights)
    } else {
        let mut bytes = 0u64;
        for e in std::fs::read_dir(path)? {
            let e = e?;
            if e.file_type()?.is_file() {
                bytes += e.metadata()?.len();
            }
        }
        (Config::from_hf(path)?, bytes)
    };
    let mtp = std::env::var("MALLEUS_MTP").map_or(true, |v| v != "0");
    // The context is the budget of all the sessions together, as it is for a batched server: every session holds
    // `ctx / slots` of it.
    let sessions = malleus_model::memplan::sessions();
    let per_slot = (ctx / sessions).max(1);
    let usage = malleus_model::memplan::Usage {
        weights,
        session: malleus_model::memplan::session_bytes(&cfg, per_slot, mtp),
        sessions,
        reserve: malleus_model::memplan::reserve_bytes(
            &cfg,
            chunk
                .unwrap_or_else(|| malleus_model::memplan::prefill_chunk(&cfg))
                .min(ctx),
            std::env::var("MALLEUS_MOE_Y_HALF").is_ok_and(|v| v != "0"),
        ),
        host: malleus_model::memplan::host_reserve_bytes(tables),
    };
    malleus_model::memplan::check(
        &path.display().to_string(),
        per_slot,
        &usage,
        malleus_model::memplan::MemInfo::read(),
    )
}

fn load_model(path: &Path) -> Result<Model> {
    let arch = select_device()?;
    let k = Kernels::load(&arch)?;
    let t0 = Instant::now();
    let progress = |i: usize, n: usize| {
        if i.is_multiple_of(8) || i == n {
            eprint!("\rloading layers {i}/{n}");
        }
    };
    let prefill_int8 = std::env::var("MALLEUS_PREFILL_WEIGHTS").map_or(true, |v| v != "none");
    let with_vision = std::env::var("MALLEUS_VISION").map(|v| v != "0").unwrap_or(true);
    let load_vision = |m: &mut Model, src: &dyn malleus_gguf::TensorSource, cfg_json: &str| -> Result<()> {
        // Hy4-flash and Inkling-2 have towers of their own, loaded with their own weights.
        if matches!(m.cfg.family, Family::Hy4 | Family::Inkling2) {
            return Ok(());
        }
        if let Some(vc) = VisionConfig::from_json(cfg_json)?
            && with_vision
            && src.info("model.visual.patch_embed.proj.weight").is_some()
        {
            let v = VisionModel::load(vc, src, "model.visual.")?;
            m.weight_bytes += v.weight_bytes;
            m.vision = Some(v);
        }
        Ok(())
    };
    let m = if is_converted(path) {
        let g = open_converted(path)?;
        let cfg_json = g.meta_str("openmalleus.hf_config")?;
        let cfg = Config::from_json(cfg_json, g.meta_str("openmalleus.generation_config").ok())?;
        let opts = LoadOptions {
            hadamard_seed: g.meta_u64("openmalleus.hadamard_seed")?,
            prefill_int8,
            prefill_families: std::env::var("MALLEUS_PREFILL_INT8_FAMILIES")
                .map(|v| v.split(',').filter(|x| !x.is_empty()).map(String::from).collect())
                .unwrap_or_default(),
            mtp: std::env::var("MALLEUS_MTP").map_or(true, |v| v != "0"),
            ..Default::default()
        };
        let mut m = Model::load(cfg, &g, &opts, k, &progress)?;
        load_vision(&mut m, &g, cfg_json)?;
        m
    } else {
        let st = SafeTensors::open_dir(path)?;
        let mut m = Model::load(Config::from_hf(path)?, &st, &LoadOptions::default(), k, &progress)?;
        load_vision(&mut m, &st, &std::fs::read_to_string(path.join("config.json"))?)?;
        m
    };
    eprintln!(
        "\nloaded {:.1} GB on {arch} in {:.0}s",
        m.weight_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    let (free, total) = malleus_hip::mem_get_info().unwrap_or((0, 0));
    eprintln!(
        "memory: {:.1} GB live on the device (peak {:.1}), GTT {:.1} GB free of {:.1}",
        malleus_hip::device_bytes_live() as f64 / 1e9,
        malleus_hip::device_bytes_peak() as f64 / 1e9,
        free as f64 / 1e9,
        total as f64 / 1e9
    );
    // What the file actually allows, not what the flags would prefer: a single-copy file has no MLS8 weights to
    // read, so the prefill runs on the MLS4 nibbles with int8 activations (W4A8).
    eprintln!(
        "dense prefill GEMMs: {}",
        if dense_w4a4() {
            "W4A4 (int4 WMMA on the MLS4 weights)"
        } else if m.prefill_copies {
            "W8A8 (fused MLS8 copies)"
        } else {
            "W4A8 (MLS4 weights, no MLS8 copies)"
        }
    );
    Ok(m)
}

fn draft_vocab(model: &Path, out: &Path) -> Result<()> {
    let chat = load_chat(model)?;
    let ids = chat.latin_vocab_ids();
    let mut text = String::with_capacity(ids.len() * 8);
    for id in &ids {
        text.push_str(&id.to_string());
        text.push('\n');
    }
    std::fs::write(out, text)?;
    eprintln!("{} Latin-script tokens -> {}", ids.len(), out.display());
    Ok(())
}

fn load_chat(path: &Path) -> Result<malleus_chat::Chat> {
    if is_converted(path) {
        let g = open_converted(path)?;
        malleus_chat::Chat::from_parts(
            g.meta_str("tokenizer.huggingface.json")?.as_bytes(),
            g.meta_str("tokenizer.chat_template")?.to_string(),
        )
    } else {
        malleus_chat::Chat::load(path)
    }
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
    x.iter().map(|&v| v as f64 - lse).collect()
}

/// The speculative verify against the serial step, bit for bit (`malleus spec-exact`).
///
/// A verify of `w` tokens has to give every one of them the logits it gets when the tokens are fed one at a
/// time: the drafts only propose, so the committed text must not depend on how many rows a step checked. The
/// verifies run first, so that with `MALLEUS_DUMP_DIR` set the last one-token step dumped is the reference of
/// row 0 (the reference then stops there, and only row 0 is compared).
/// The batched decode of several sessions must equal the serial decode of the same state, row by row.
///
/// One session is advanced serially, `sessions` others in one batched step of the same token; every row's argmax is
/// compared at every step, and the token the serial path chose is what the batch is fed next. A difference means the
/// row offsets or the shared buffers of `Model::decode_batch` are wrong, which is exactly the failure this engine
/// had while `decode_batch` was being written.
fn batch_exact(
    model: &Path,
    golden: &Path,
    only: &str,
    sessions: usize,
    steps: usize,
    limit: Option<usize>,
    ctx: usize,
) -> Result<()> {
    let g: serde_json::Value = serde_json::from_reader(std::fs::File::open(golden)?)?;
    let entry = g["text"]
        .as_array()
        .context("text")?
        .iter()
        .find(|e| e["id"].as_str() == Some(only))
        .with_context(|| format!("no prompt `{only}` in the golden file"))?;
    let mut ids: Vec<u32> = entry["input_ids"]
        .as_array()
        .context("ids")?
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    if let Some(n) = limit {
        ids.truncate(n);
    }
    ensure!(!ids.is_empty(), "empty prompt");
    ensure!(sessions >= 1, "--sessions must be at least 1");
    let m = load_model(model)?;
    ensure!(m.can_batch(), "this model has no batched decode");
    let argmax = |row: &[f32]| -> u32 {
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        best as u32
    };
    let mut serial = m.new_session(ctx)?;
    let mut batched: Vec<Session> = (0..sessions).map(|_| m.new_session(ctx)).collect::<Result<_>>()?;
    // The prompt is the same for every session, and so is its first token.
    let first = m.forward_argmax(&mut serial, &ids)?;
    let mut tok = first;
    for s in batched.iter_mut() {
        let t = m.forward_argmax(s, &ids)?;
        ensure!(t == first, "the prefills disagree ({t} against {first})");
    }
    println!(
        "prompt {} tokens, first token {first}, {sessions} batched session(s)",
        ids.len()
    );
    for step in 0..steps {
        let row = m.forward(&mut serial, &[tok], None, 1)?;
        let want = argmax(&row);
        let mut runs: Vec<(&mut Session, u32)> = batched.iter_mut().map(|s| (s, tok)).collect();
        let rows = m.decode_batch(&mut runs)?;
        for (i, r) in rows.iter().enumerate() {
            ensure!(r.len() == row.len(), "row {i} is not a logits row");
            let got = argmax(r);
            ensure!(
                got == want,
                "step {step}: row {i} chose {got}, the serial step chose {want}"
            );
            let same = r.iter().zip(&row).all(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "step {step:3} row {i}: argmax {got} {}",
                if same {
                    "identical"
                } else {
                    "same argmax, logits differ"
                }
            );
        }
        tok = want;
    }
    println!("every step agrees with the serial decode");
    Ok(())
}

fn spec_exact(
    model: &Path,
    golden: &Path,
    only: &str,
    prefixes: &[usize],
    width: Option<usize>,
    replay: usize,
    ctx: usize,
) -> Result<()> {
    let g: serde_json::Value = serde_json::from_reader(std::fs::File::open(golden)?)?;
    let entry = g["text"]
        .as_array()
        .context("text")?
        .iter()
        .find(|e| e["id"].as_str() == Some(only))
        .with_context(|| format!("no prompt `{only}` in the golden file"))?;
    let ids: Vec<u32> = entry["input_ids"]
        .as_array()
        .context("ids")?
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let widths: Vec<usize> = match width {
        Some(w) => vec![w],
        None => (2..=GEMV_MAX_T).collect(),
    };
    ensure!(
        widths.iter().all(|w| (2..=GEMV_MAX_T).contains(w)),
        "verify width out of 2..={GEMV_MAX_T}"
    );
    let maxw = widths.iter().copied().max().unwrap_or(2);
    let m = load_model(model)?;
    ensure!(m.has_mtp(), "the model has no MTP head: it never verifies");
    let vocab = m.cfg.vocab;
    // Debug, for the state a verify leaves behind: MALLEUS_SPEC_STATE=serial|verify with MALLEUS_DUMP_DIR feeds
    // the prefix, then `width` tokens one at a time or in one verify, then one more token alone, whose per-layer
    // dumps (`X<layer>_*_t1`) are what two runs with the two values are compared on.
    if let Ok(mode) = std::env::var("MALLEUS_SPEC_STATE") {
        let (p, w) = (prefixes[0], maxw);
        ensure!(p + w < ids.len(), "prefix {p} out of the prompt");
        let mut sess = m.new_session(ctx)?;
        for &id in &ids[..p] {
            m.forward(&mut sess, &[id], None, 1)?;
        }
        if mode == "verify" {
            m.forward_verify(&mut sess, &ids[p..p + w])?;
        } else {
            for &id in &ids[p..p + w] {
                m.forward(&mut sess, &[id], None, 1)?;
            }
        }
        m.forward(&mut sess, &[ids[p + w]], None, 1)?;
        return Ok(());
    }
    let ref_rows = if std::env::var_os("MALLEUS_DUMP_DIR").is_some() {
        1
    } else {
        maxw
    };
    let mut bad = 0usize;
    for &prefix in prefixes {
        ensure!(
            prefix >= 1 && prefix + maxw <= ids.len(),
            "prefix {prefix} out of the prompt"
        );
        let mut ver = Vec::new();
        for &w in &widths {
            let mut sess = m.new_session(ctx)?;
            for &id in &ids[..prefix] {
                m.forward(&mut sess, &[id], None, 1)?;
            }
            ver.push(m.forward_verify(&mut sess, &ids[prefix..prefix + w])?);
        }
        let mut sess = m.new_session(ctx)?;
        let mut reference = Vec::with_capacity(ref_rows * vocab);
        for (i, &id) in ids[..prefix + ref_rows].iter().enumerate() {
            let l = m.forward(&mut sess, &[id], None, 1)?;
            if i >= prefix {
                reference.extend(l);
            }
        }
        for (&w, v) in widths.iter().zip(&ver) {
            for r in 0..w.min(ref_rows) {
                let (a, b) = (&reference[r * vocab..(r + 1) * vocab], &v[r * vocab..(r + 1) * vocab]);
                let diff = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                let same_top = argmax(a) == argmax(b);
                bad += usize::from(diff > 0);
                println!(
                    "prefix {prefix:5} width {w} row {r}: {}{}",
                    if diff == 0 {
                        "identical".to_string()
                    } else {
                        format!("{diff} of {vocab} logits differ, max |d| {max:.3e}")
                    },
                    if same_top { "" } else { "  ARGMAX DIFFERS" }
                );
            }
        }
    }
    ensure!(bad == 0, "{bad} verify rows differ from the serial step");
    println!("every verify row is identical to the serial step");
    if replay > 0 && std::env::var_os("MALLEUS_DUMP_DIR").is_none() {
        spec_replay(&m, &ids[..prefixes[0]], replay, ctx)?;
    }
    Ok(())
}

/// Generates `n` tokens after `prompt` twice, one token at a time and through the scheduler's speculative loop
/// (MTP drafts, verify, rollback of the rejected tail), and compares every committed row bit for bit: a single
/// verify can be exact and the state it leaves behind still be wrong.
fn spec_replay(m: &Model, prompt: &[u32], n: usize, ctx: usize) -> Result<()> {
    let vocab = m.cfg.vocab;
    let mut sess = m.new_session(ctx)?;
    let mut l = Vec::new();
    for &id in prompt {
        l = m.forward(&mut sess, &[id], None, 1)?;
    }
    // Serial: `rows[i]` is what token `toks[i]` produced, and `toks[i + 1]` its argmax.
    let mut toks = vec![argmax(&l) as u32];
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let l = m.forward(&mut sess, &[toks[i]], None, 1)?;
        toks.push(argmax(&l) as u32);
        rows.push(l);
    }
    let mut sess = m.new_session(ctx)?;
    for &id in prompt {
        m.forward(&mut sess, &[id], None, 1)?;
    }
    let max = std::env::var("MALLEUS_SPEC_DRAFTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5usize)
        .min(GEMV_MAX_T - 1);
    let pmin: f32 = std::env::var("MALLEUS_DRAFT_PMIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(malleus_sched::DEFAULT_DRAFT_PMIN);
    let (mut i, mut pend, mut rounds, mut rejected) = (0usize, vec![toks[0]], 0usize, 0usize);
    while i < n {
        let ds = malleus_sched::draft_tokens(m, &mut sess, &pend, max.min(n - i), pmin, None)?;
        let step: Vec<u32> = std::iter::once(toks[i]).chain(ds.iter().copied()).collect();
        let v = if step.len() > 1 {
            m.forward_verify(&mut sess, &step)?
        } else {
            m.forward(&mut sess, &step, None, 1)?
        };
        rounds += 1;
        let mut acc = 0;
        loop {
            let (row, want) = (&v[acc * vocab..(acc + 1) * vocab], &rows[i + acc]);
            let diff = row.iter().zip(want).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            if diff > 0 {
                let max_d = row.iter().zip(want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                anyhow::bail!(
                    "replay: token {} (round {rounds}, row {acc} of a {}-row verify, {rejected} drafts rejected \
                     so far): {diff} of {vocab} logits differ from the serial step, max |d| {max_d:.3e}",
                    i + acc,
                    step.len()
                );
            }
            if acc == ds.len() || i + acc + 1 >= n || ds[acc] != toks[i + acc + 1] {
                break;
            }
            acc += 1;
        }
        rejected += ds.len() - acc;
        m.rollback(&mut sess, ds.len() - acc)?;
        pend = ds[..acc].to_vec();
        pend.push(toks[i + acc + 1]);
        i += acc + 1;
    }
    println!("replay: {n} tokens in {rounds} rounds ({rejected} drafts rejected), every committed row identical");
    Ok(())
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
        .0
}

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| p.display().to_string())?;
    Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn eval(model: &Path, set: &Path, blobs: &Path, chunk: usize) -> Result<()> {
    let e: serde_json::Value = serde_json::from_reader(std::fs::File::open(set)?)?;
    let k = e["topk"].as_u64().context("topk")? as usize;
    let topk = std::fs::read(blobs.join("eval").join("topk.bin"))?;
    let m = load_model(model)?;
    let vocab = m.cfg.vocab;
    // Inkling-2's attention runs over the whole prompt in one forward: a chunk after the first is not supported
    // yet (issue #70), so the document goes in whole.
    let chunk = match m.cfg.family {
        Family::Inkling2 => usize::MAX,
        _ => chunk,
    };
    let u = |v: &serde_json::Value| -> Vec<u64> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect() };
    let (mut pos, mut tot) = (0usize, [0f64; 5]);
    for d in e["docs"].as_array().context("docs")? {
        let ids: Vec<u32> = u(&d["input_ids"]).into_iter().map(|x| x as u32).collect();
        let ref_arg = u(&d["argmax"]);
        let ref_nll: Vec<f64> = d["nll"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let mut sess = m.new_session(ids.len() + 1)?;
        let mut acc = [0f64; 5];
        for (ci, c) in ids.chunks(chunk).enumerate() {
            let logits = m.forward(&mut sess, c, None, c.len())?;
            for (j, row) in logits.chunks_exact(vocab).enumerate() {
                let i = ci * chunk + j;
                let lp = log_softmax(row);
                let rec = &topk[(pos + i) * k * 8..(pos + i + 1) * k * 8];
                let (ri, rv) = rec.split_at(k * 4);
                let (mut kl, mut pr, mut po) = (0f64, 0f64, 0f64);
                for (a, b) in ri.as_chunks::<4>().0.iter().zip(rv.as_chunks::<4>().0) {
                    let (id, lr) = (i32::from_le_bytes(*a) as usize, f32::from_le_bytes(*b) as f64);
                    kl += lr.exp() * (lr - lp[id]);
                    pr += lr.exp();
                    po += lp[id].exp();
                }
                let (rr, ro) = ((1.0 - pr).max(1e-12), (1.0 - po).max(1e-12));
                kl += rr * (rr.ln() - ro.ln());
                acc[0] += kl;
                acc[1] += (argmax(row) == ref_arg[i] as usize) as u8 as f64;
                if i + 1 < ids.len() {
                    acc[2] += -lp[ids[i + 1] as usize];
                    acc[3] += ref_nll[i];
                    acc[4] += 1.0;
                }
            }
        }
        let n = ids.len() as f64;
        println!(
            "{:14} ppl {:8.3} (ref {:8.3}, {:+.2}%)  top1 {:6.2}%  KL {:.5}",
            d["id"].as_str().unwrap_or("?"),
            (acc[2] / acc[4]).exp(),
            (acc[3] / acc[4]).exp(),
            ((acc[2] - acc[3]) / acc[4]).exp_m1() * 100.0,
            acc[1] / n * 100.0,
            acc[0] / n
        );
        tot[0] += acc[0];
        tot[1] += acc[1];
        tot[2] += acc[2];
        tot[3] += acc[3];
        tot[4] += acc[4];
        pos += ids.len();
    }
    let n = pos as f64;
    println!(
        "TOTAL          ppl {:8.3} (ref {:8.3}, {:+.2}%)  top1 {:6.2}%  KL {:.5}",
        (tot[2] / tot[4]).exp(),
        (tot[3] / tot[4]).exp(),
        ((tot[2] - tot[3]) / tot[4]).exp_m1() * 100.0,
        tot[1] / n * 100.0,
        tot[0] / n
    );
    Ok(())
}

fn golden(
    model: &Path,
    golden: &Path,
    blobs: &Path,
    ctx: usize,
    greedy: bool,
    only: Option<&str>,
    decode: bool,
) -> Result<()> {
    let g: serde_json::Value = serde_json::from_reader(std::fs::File::open(golden)?)?;
    let m = load_model(model)?;
    let vocab = m.cfg.vocab;
    let mut worst_kl: f64 = 0.0;
    let (mut top1_ok, mut top1_n, mut pos_ok, mut pos_n) = (0, 0, 0usize, 0usize);
    for e in g["text"].as_array().context("text")? {
        let id = e["id"].as_str().unwrap_or("?");
        if only.is_some_and(|o| !o.split(',').any(|x| x == id)) {
            continue;
        }
        let ids: Vec<u32> = e["input_ids"]
            .as_array()
            .context("ids")?
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let ref_argmax: Vec<usize> = e["argmax"]
            .as_array()
            .context("argmax")?
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let mut sess = m.new_session(ctx)?;
        let t0 = Instant::now();
        let logits = if decode {
            let mut l = Vec::with_capacity(ids.len() * vocab);
            for &id in &ids {
                l.extend(m.forward(&mut sess, &[id], None, 1)?);
            }
            l
        } else {
            m.forward(&mut sess, &ids, None, ids.len())?
        };
        let dt = t0.elapsed().as_secs_f64();
        let ours: Vec<usize> = logits.chunks_exact(vocab).map(argmax).collect();
        let agree = ours.iter().zip(&ref_argmax).filter(|(a, b)| a == b).count();
        pos_ok += agree;
        pos_n += ids.len();
        let last = &logits[(ids.len() - 1) * vocab..];
        let rl = read_f32(&blobs.join("text").join(id).join("logits_last.f32"))?;
        let (lp_ref, lp_our) = (log_softmax(&rl), log_softmax(last));
        let kl: f64 = lp_ref.iter().zip(&lp_our).map(|(r, o)| r.exp() * (r - o)).sum();
        worst_kl = worst_kl.max(kl);
        let t1 = argmax(last) == argmax(&rl);
        top1_ok += t1 as usize;
        top1_n += 1;
        print!(
            "{id:14} T={:4} prefill {:6.2}s ({:5.1} tok/s)  pos-argmax {agree}/{}  last top1 {}  KL {kl:.5}",
            ids.len(),
            dt,
            ids.len() as f64 / dt,
            ids.len(),
            if t1 { "ok" } else { "MISMATCH" }
        );
        if greedy && let Some(refg) = e["greedy"].as_array() {
            let mut refg: Vec<u32> = refg.iter().map(|v| v.as_u64().unwrap() as u32).collect();
            if let Some(i) = refg.iter().position(|t| m.cfg.eos.contains(t)) {
                refg.truncate(i + 1);
            }
            let mut next = argmax(last) as u32;
            let mut out = vec![];
            let t0 = Instant::now();
            for _ in 0..refg.len() {
                out.push(next);
                if m.cfg.eos.contains(&next) {
                    break;
                }
                let l = m.forward(&mut sess, &[next], None, 1)?;
                next = argmax(&l) as u32;
            }
            let dt = t0.elapsed().as_secs_f64();
            let same = out.iter().zip(&refg).take_while(|(a, b)| a == b).count();
            print!(
                "  greedy {same}/{} identical, decode {:.2} tok/s",
                refg.len(),
                out.len() as f64 / dt
            );
        }
        println!();
    }
    if let Some(imgs) = g["images"].as_array() {
        let img_dir = golden.parent().context("golden dir")?.join("images");
        for e in imgs {
            let id = e["id"].as_str().unwrap_or("?");
            if only.is_some_and(|o| !o.split(',').any(|x| x == id)) {
                continue;
            }
            golden_image(&m, e, &img_dir, &blobs.join("images").join(id), ctx, greedy)?;
        }
    }
    println!(
        "SUMMARY: last-token top1 {top1_ok}/{top1_n}, all-position argmax {pos_ok}/{pos_n} ({:.2}%), worst KL {worst_kl:.5}",
        100.0 * pos_ok as f64 / pos_n.max(1) as f64
    );
    Ok(())
}

fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn golden_image(m: &Model, e: &serde_json::Value, img_dir: &Path, blob: &Path, ctx: usize, greedy: bool) -> Result<()> {
    use malleus_model::image_pre::{PreConfig, decode, preprocess_rgb};
    let id = e["id"].as_str().unwrap_or("?");
    let v = m.vision.as_ref().context("model has no vision tower")?;
    let bytes = std::fs::read(img_dir.join(format!("{id}.png")))?;
    let (rgb, h, w) = decode(&bytes, &PreConfig::default())?;
    let pc = PreConfig {
        max_pixels: 16777216,
        ..Default::default()
    };
    let pre = preprocess_rgb(&rgb, h, w, &pc)?;
    let grid: Vec<usize> = e["grid_thw"]
        .as_array()
        .context("grid")?
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let ref_px = read_f32(&blob.join("pixel_values.f32"))?;
    let px_ok = [pre.grid_t, pre.grid_h, pre.grid_w] == grid[..];
    let px_err = if px_ok && ref_px.len() == pre.pixels.len() {
        pre.pixels
            .iter()
            .zip(&ref_px)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    } else {
        f32::NAN
    };
    let t0 = Instant::now();
    let (vit, merged) = v.encode(&m.k, &pre)?;
    let enc_s = t0.elapsed().as_secs_f64();
    let vit = vit.download::<f32>(vit.len() / 4)?;
    let mg = merged.download::<f32>(merged.len() / 4)?;
    let vit_err = rel_err(&vit, &read_f32(&blob.join("vit_out.f32"))?);
    let mg_err = rel_err(&mg, &read_f32(&blob.join("merger_out.f32"))?);
    let ids: Vec<u32> = e["input_ids"]
        .as_array()
        .context("ids")?
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let image_token = 248056u32;
    let start = ids.iter().position(|&t| t == image_token).context("no image token")?;
    let mm = v.cfg.merge;
    let (pos, _) = malleus_model::mrope::positions(&ids, image_token, &[(pre.grid_h / mm, pre.grid_w / mm)], 0)?;
    let mut sess = m.new_session(ctx)?;
    let logits = m.forward_mm(&mut sess, &ids, Some(&pos), 1, &[(start, &merged)])?;
    let rl = read_f32(&blob.join("logits_last.f32"))?;
    let (lp_ref, lp_our) = (log_softmax(&rl), log_softmax(&logits));
    let kl: f64 = lp_ref.iter().zip(&lp_our).map(|(r, o)| r.exp() * (r - o)).sum();
    print!(
        "img {id:14} grid {:?} px max|d| {px_err:.4} vit rel {vit_err:.4} merger rel {mg_err:.4} encode {enc_s:.2}s top1 {} KL {kl:.5}",
        grid,
        if argmax(&logits) == argmax(&rl) {
            "ok"
        } else {
            "MISMATCH"
        }
    );
    if greedy && let Some(refg) = e["greedy"].as_array() {
        let mut refg: Vec<u32> = refg.iter().map(|v| v.as_u64().unwrap() as u32).collect();
        if let Some(i) = refg.iter().position(|t| m.cfg.eos.contains(t)) {
            refg.truncate(i + 1);
        }
        let mut next = argmax(&logits) as u32;
        let mut out = vec![];
        for _ in 0..refg.len() {
            out.push(next);
            if m.cfg.eos.contains(&next) {
                break;
            }
            let p = [[sess.next_pos as i32; 3]];
            let l = m.forward(&mut sess, &[next], Some(&p), 1)?;
            next = argmax(&l) as u32;
        }
        let same = out.iter().zip(&refg).take_while(|(a, b)| a == b).count();
        print!("  greedy {same}/{}", refg.len());
    }
    println!();
    Ok(())
}

fn run(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    think: bool,
    ctx: usize,
    max_drafts: usize,
    prompt_ids: Option<&Path>,
    dump_logits: Option<&Path>,
) -> Result<()> {
    preflight(model, ctx, None)?;
    let spec = max_drafts > 0;
    let pmin: f32 = std::env::var("MALLEUS_DRAFT_PMIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(malleus_sched::DEFAULT_DRAFT_PMIN);
    let chat = load_chat(model)?;
    let m = load_model(model)?;
    // Inkling-2 has no KV cache yet: the graph re-runs the whole sequence for every token, so it takes the prompt
    // here and generates on its own path.
    if m.cfg.family == Family::Inkling2 {
        let msgs = serde_json::json!([{ "role": "user", "content": prompt }]);
        let opts = malleus_chat::RenderOptions {
            enable_thinking: Some(think),
            ..Default::default()
        };
        let ids = chat.encode(&chat.render(&msgs, &opts)?)?;
        eprintln!("inkling2: prompt {} tok", ids.len());
        let t0 = Instant::now();
        let out = m.inkling2_generate(&ids, max_tokens)?;
        let dt = t0.elapsed().as_secs_f64();
        let mut dec = malleus_chat::StreamDecoder::new();
        for tok in &out {
            print!("{}", dec.push(&chat, *tok)?);
        }
        println!();
        eprintln!("{} tok in {dt:.2}s ({:.2} tok/s)", out.len(), out.len() as f64 / dt);
        return Ok(());
    }
    if let Some(path) = prompt_ids {
        let ids: Vec<u32> = std::fs::read_to_string(path)?
            .split_whitespace()
            .map(|x| x.parse::<u32>())
            .collect::<std::result::Result<_, _>>()?;
        let mut sess = m.new_session(ctx)?;
        let chunk = std::env::var("MALLEUS_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(32768)
            .max(64);
        let t0 = Instant::now();
        let mut last = Vec::new();
        for part in ids.chunks(chunk) {
            last = m.forward(&mut sess, part, None, 1)?;
        }
        let dt = t0.elapsed().as_secs_f64();
        if let Some(path) = dump_logits
            && !last.is_empty()
        {
            let mut bytes = Vec::with_capacity(last.len() * 4);
            for v in &last {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(path, &bytes)?;
            eprintln!("logit dell'ultima posizione in {}", path.display());
        }
        if let Some(path) = std::env::var_os("MALLEUS_ATTN_MASS") {
            m.dump_attn_mass(std::path::Path::new(&path))?;
            eprintln!("attention masses written to {}", std::path::Path::new(&path).display());
        }
        println!(
            "prompt {} tok, TTFT {:.2}s ({:.1} tok/s)",
            ids.len(),
            dt,
            ids.len() as f64 / dt
        );
        if max_tokens > 1 && !last.is_empty() {
            let mut next = argmax(&last) as u32;
            let gpu_argmax = std::env::var("MALLEUS_ARGMAX_GPU").map_or(true, |v| v != "0");
            let mut out = Vec::with_capacity(max_tokens);
            for _ in 0..max_tokens {
                out.push(next);
                if m.cfg.eos.contains(&next) {
                    break;
                }
                next = if gpu_argmax {
                    m.forward_argmax(&mut sess, &[next])?
                } else {
                    argmax(&m.forward(&mut sess, &[next], None, 1)?) as u32
                };
            }
            println!("ids: {out:?}");
        }
        return Ok(());
    }
    let prompts: Vec<String> = match prompt.strip_prefix('@') {
        Some(f) => std::fs::read_to_string(f)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.replace("\\n", "\n"))
            .collect(),
        None => vec![prompt.to_string()],
    };
    let mut sess = m.new_session(ctx)?;
    let (mut steps, mut accepted, mut tot_n, mut tot_s) = (0usize, 0usize, 0usize, 0f64);
    let (mut t_draft, mut t_verify, mut vsteps) = (0f64, 0f64, 0usize);
    for prompt in &prompts {
        m.reset_session(&mut sess)?;
        let msgs = serde_json::json!([{ "role": "user", "content": prompt }]);
        let opts = malleus_chat::RenderOptions {
            enable_thinking: Some(think),
            ..Default::default()
        };
        let ids = chat.encode(&chat.render(&msgs, &opts)?)?;
        let t0 = Instant::now();
        let l = m.forward(&mut sess, &ids, None, 1)?;
        let ttft = t0.elapsed().as_secs_f64();
        if let Some(path) = dump_logits {
            let mut bytes = Vec::with_capacity(l.len() * 4);
            for v in &l {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(path, &bytes)?;
        }
        let mut next = argmax(&l) as u32;
        let mut dec = malleus_chat::StreamDecoder::new();
        let t1 = Instant::now();
        let mut n = 0;
        use std::io::Write;
        let mut pend = vec![next];
        // Tokens in the session plus the pending one (prompt lookup drafts).
        let mut hist = ids.clone();
        let lookup = std::env::var("MALLEUS_LOOKUP").map_or(true, |v| v != "0");
        let v = m.cfg.vocab;
        let (s0, a0) = (steps, accepted);
        while n < max_tokens && !m.cfg.eos.contains(&next) {
            print!("{}", dec.push(&chat, next)?);
            std::io::stdout().flush()?;
            n += 1;
            if !spec {
                next = if std::env::var("MALLEUS_ARGMAX_GPU").map_or(true, |v| v != "0") {
                    m.forward_argmax(&mut sess, &[next])?
                } else {
                    argmax(&m.forward(&mut sess, &[next], None, 1)?) as u32
                };
                continue;
            }
            let ta = Instant::now();
            hist.push(next);
            // `run` has the prompt as one string, so there is no tool-result text to draft from.
            const NO_SOURCES: &[Vec<u32>] = &[];
            let ds = malleus_sched::draft_tokens(
                &m,
                &mut sess,
                &pend,
                max_drafts,
                pmin,
                lookup.then_some((&hist[..], NO_SOURCES)),
            )?;
            hist.pop();
            let nd = ds.len();
            let tb = Instant::now();
            let toks: Vec<u32> = std::iter::once(next).chain(ds.iter().copied()).collect();
            let l = m.forward_verify(&mut sess, &toks)?;
            t_draft += (tb - ta).as_secs_f64();
            t_verify += tb.elapsed().as_secs_f64();
            steps += nd;
            vsteps += 1;
            let mut acc = 0;
            while acc < nd && argmax(&l[acc * v..(acc + 1) * v]) as u32 == ds[acc] {
                acc += 1;
            }
            accepted += acc;
            m.rollback(&mut sess, nd - acc)?;
            let y = argmax(&l[acc * v..(acc + 1) * v]) as u32;
            let mut stop = false;
            for &d in &ds[..acc] {
                if n >= max_tokens || m.cfg.eos.contains(&d) {
                    stop = true;
                    break;
                }
                print!("{}", dec.push(&chat, d)?);
                n += 1;
            }
            if stop {
                break;
            }
            hist.push(next);
            hist.extend_from_slice(&ds[..acc]);
            pend = ds[..acc].to_vec();
            pend.push(y);
            next = y;
        }
        println!();
        let dt = t1.elapsed().as_secs_f64();
        tot_n += n;
        tot_s += dt;
        eprintln!(
            "prompt {} tok, TTFT {:.2}s ({:.1} tok/s), decode {n} tok at {:.2} tok/s{}",
            ids.len(),
            ttft,
            ids.len() as f64 / ttft,
            n as f64 / dt,
            if spec {
                format!(", MTP {}/{} drafts accepted", accepted - a0, steps - s0)
            } else {
                String::new()
            }
        );
    }
    if prompts.len() > 1 || spec {
        eprintln!(
            "total: {tot_n} tok at {:.2} tok/s{}",
            tot_n as f64 / tot_s,
            if spec {
                format!(
                    ", MTP {accepted}/{steps} accepted ({:.1}%)",
                    100.0 * accepted as f64 / steps.max(1) as f64
                )
            } else {
                String::new()
            }
        );
        if spec {
            let k = 1e3 / vsteps.max(1) as f64;
            eprintln!(
                "per step: draft {:.1} ms, verify {:.1} ms, {:.2} tokens",
                t_draft * k,
                t_verify * k,
                tot_n as f64 / vsteps.max(1) as f64
            );
        }
    }
    if malleus_model::expert_cache::enabled() {
        let (hits, misses, bytes) = malleus_model::expert_cache::stats();
        let total = hits + misses;
        eprintln!(
            "expert cache: {:.2}% of the pairs hit ({} / {total}), {misses} experts read, {:.2} GB from disk",
            if total > 0 {
                100.0 * hits as f64 / total as f64
            } else {
                0.0
            },
            hits,
            bytes as f64 / 1e9
        );
    }
    Ok(())
}

fn bench_bw(gb: f64) -> Result<()> {
    use malleus_hip::{Arg, DeviceBuffer, Event, Stream};
    let arch = select_device()?;
    let k = Kernels::load(&arch)?;
    let bytes = (gb * 1e9) as usize / 16 * 16;
    let n = (bytes / 16) as u64;
    let a = DeviceBuffer::zeroed(bytes)?;
    let b = DeviceBuffer::zeroed(bytes)?;
    let out = DeviceBuffer::zeroed(16)?;
    let s = Stream::null();
    for grid in [256u32, 1024, 4096] {
        let mut best = (0.0f64, 0.0f64);
        for _ in 0..5 {
            let (e0, e1) = (Event::new()?, Event::new()?);
            e0.record(&s)?;
            k.get("bw_read").launch(
                (grid, 1, 1),
                (256, 1, 1),
                0,
                &s,
                &[(&a).into(), Arg::U64(n), (&out).into()],
            )?;
            e1.record(&s)?;
            best.0 = best.0.max(bytes as f64 / (e1.elapsed_ms(&e0)? as f64 * 1e6));
            e0.record(&s)?;
            k.get("bw_copy").launch(
                (grid, 1, 1),
                (256, 1, 1),
                0,
                &s,
                &[(&b).into(), (&a).into(), Arg::U64(n)],
            )?;
            e1.record(&s)?;
            best.1 = best.1.max(2.0 * bytes as f64 / (e1.elapsed_ms(&e0)? as f64 * 1e6));
        }
        println!(
            "{arch} grid {grid:5}: read {:6.1} GB/s, copy (r+w) {:6.1} GB/s",
            best.0, best.1
        );
    }
    Ok(())
}

type ChatMap = HashMap<Arc<str>, Arc<malleus_chat::Chat>>;

/// Builds the catalog the server exposes and loads each model's tokenizer once. Every entry comes from a
/// `*.malleus` file directly under `MALLEUS_MODELS_DIR`; a broken file is skipped with a warning. The default is
/// `MALLEUS_DEFAULT_MODEL` when set, otherwise the only model when the directory holds exactly one.
fn build_catalog(models_dir: &Path, ctx: usize, default_model: Option<String>) -> Result<(Arc<Catalog>, ChatMap)> {
    let dir = models_dir
        .canonicalize()
        .with_context(|| format!("models directory {}", models_dir.display()))?;
    let paths = malleus_sched::catalog::files_in(&dir)?;
    if paths.is_empty() {
        bail!("no models found under {}", dir.display());
    }
    let mut specs: Vec<ModelSpec> = Vec::new();
    let mut chats: ChatMap = HashMap::new();
    for path in paths {
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .with_context(|| format!("{} has no UTF-8 file stem", path.display()))?
            .to_string();
        let chat = match load_chat(&path) {
            Ok(chat) => chat,
            Err(e) => {
                eprintln!("warning: skipping {}: {e:#}", path.display());
                continue;
            }
        };
        let mut eos = vec![];
        for t in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = chat.token_id(t) {
                eos.push(id);
            }
        }
        // The model's own end tokens (generation_config): Hy4 ends a turn with `<｜hy_end:opensource｜>`, which the
        // two Qwen names above do not cover, so the server never stopped on it.
        if is_converted(&path)
            && let Ok(g) = open_converted(&path)
            && let Ok(cfg) = g.meta_str("openmalleus.hf_config")
            && let Ok(c) = Config::from_json(cfg, g.meta_str("openmalleus.generation_config").ok())
        {
            for id in c.eos {
                if !eos.contains(&id) {
                    eos.push(id);
                }
            }
        }
        let spec = ModelSpec {
            id: Arc::from(id.as_str()),
            path,
            ctx,
            eos: Arc::from(eos),
        };
        chats.insert(spec.id.clone(), Arc::new(chat));
        specs.push(spec);
    }
    if specs.is_empty() {
        bail!("no loadable models found under {}", dir.display());
    }
    let default_id = match default_model {
        Some(id) => Some(id),
        None if specs.len() == 1 => Some(specs[0].id.to_string()),
        None => None,
    };
    let catalog = Arc::new(Catalog::new(specs, default_id.as_deref())?);
    Ok((catalog, chats))
}

fn serve(
    models_dir: PathBuf,
    default_model: Option<String>,
    host: String,
    port: u16,
    api_key: Option<String>,
    ctx: usize,
    default_thinking: bool,
    max_tokens: usize,
) -> Result<()> {
    let (catalog, chats) = build_catalog(&models_dir, ctx, default_model)?;
    // The catalog default is loaded before `Engine::start` returns, preserving the old startup pre-flight. Without
    // one the first request loads its model and pays that cost.
    let initial = catalog.default();
    let load = move |spec: &ModelSpec| {
        preflight(&spec.path, spec.ctx, None)?;
        load_model(&spec.path)
    };
    let engine = malleus_sched::Engine::start(initial, load)?;
    let cfg = malleus_api::Config {
        api_key,
        default_thinking,
        max_tokens_default: max_tokens,
    };
    let app = malleus_api::router(engine, catalog.clone(), chats, cfg);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()?;
    let default = catalog
        .default()
        .map(|s| format!(", default {}", s.id))
        .unwrap_or_else(|| ", no default: requests must name a model".to_string());
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
        eprintln!(
            "serving {} model(s) on {host}:{port} (ctx {ctx}{default})",
            catalog.entries().len()
        );
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}

fn health(port: u16) -> Result<()> {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port))?;
    s.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    s.write_all(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")?;
    let mut buf = String::new();
    s.read_to_string(&mut buf)?;
    anyhow::ensure!(
        buf.starts_with("HTTP/1.1 200") || buf.starts_with("HTTP/1.0 200"),
        "unhealthy: {}",
        buf.lines().next().unwrap_or("")
    );
    Ok(())
}

fn bench(model: &Path, prompts: &[usize], decode: usize, chunk: usize, verify: Option<usize>) -> Result<()> {
    let ctx = prompts.iter().copied().max().unwrap_or(512) + decode * (1 + verify.unwrap_or(1)) + 1;
    preflight(model, ctx, Some(chunk))?;
    let m = load_model(model)?;
    let max = prompts.iter().copied().max().unwrap_or(512) + decode * (1 + verify.unwrap_or(1)) + 1;
    let mut sess = m.new_session(max)?;
    let gpu_argmax = std::env::var("MALLEUS_ARGMAX_GPU").map_or(true, |v| v != "0");
    for &n in prompts {
        m.reset_session(&mut sess)?;
        let ids: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 50000).collect();
        let t0 = Instant::now();
        let mut l = vec![];
        for c in ids.chunks(chunk) {
            l = m.forward(&mut sess, c, None, 1)?;
        }
        let pp = t0.elapsed().as_secs_f64();
        let mut next = argmax(&l) as u32;
        let t1 = Instant::now();
        for _ in 0..decode {
            next = if gpu_argmax {
                m.forward_argmax(&mut sess, &[next])?
            } else {
                argmax(&m.forward(&mut sess, &[next], None, 1)?) as u32
            };
        }
        let tg = t1.elapsed().as_secs_f64();
        println!(
            "pp{n}: {:.1} tok/s ({pp:.2}s)   tg{decode}: {:.2} tok/s",
            n as f64 / pp,
            decode as f64 / tg
        );
        if let Some(v) = verify {
            let toks = vec![next; v];
            let t2 = Instant::now();
            for _ in 0..decode {
                m.forward(&mut sess, &toks, None, v)?;
            }
            let ms = t2.elapsed().as_secs_f64() * 1e3 / decode as f64;
            println!(
                "  verify {v} tokens: {ms:.1} ms/step (decode {:.1} ms/token)",
                tg * 1e3 / decode as f64
            );
        }
    }
    Ok(())
}

fn bench_depth(model: &Path, depths: &[usize], decode: usize, chunk: usize) -> Result<()> {
    let m = load_model(model)?;
    let max = depths.iter().copied().max().unwrap_or(0) + depths.len() * (chunk + decode) + 1;
    let mut sess = m.new_session(max)?;
    let gpu_argmax = std::env::var("MALLEUS_ARGMAX_GPU").map_or(true, |v| v != "0");
    let (start, mut len) = (Instant::now(), 0usize);
    let tok = |i: usize| 1000 + (i as u32 * 7919) % 50000;
    // The first depth is the reference: what is asked here is how little the two rates degrade as the context
    // grows, not what they are in absolute terms, so the ratio goes next to the number.
    let mut ref_pp = 0f64;
    let mut ref_tg = 0f64;
    println!(
        "| depth | pp{chunk} tok/s | pp vs {base} | tg{decode} tok/s | tg vs {base} | fill tok/s | elapsed s |\n|---|---|---|---|---|---|---|",
        base = depths.first().copied().unwrap_or(0)
    );
    for &d in depths {
        let (t0, from) = (Instant::now(), len);
        while len < d {
            let ids: Vec<u32> = (len..d.min(len + chunk)).map(tok).collect();
            m.forward(&mut sess, &ids, None, 1)?;
            len += ids.len();
        }
        let fill = (len - from) as f64 / t0.elapsed().as_secs_f64().max(1e-9);
        let ids: Vec<u32> = (len..len + chunk).map(tok).collect();
        let t1 = Instant::now();
        let l = m.forward(&mut sess, &ids, None, 1)?;
        let pp = chunk as f64 / t1.elapsed().as_secs_f64();
        len += chunk;
        let mut next = argmax(&l) as u32;
        let t2 = Instant::now();
        for _ in 0..decode {
            next = if gpu_argmax {
                m.forward_argmax(&mut sess, &[next])?
            } else {
                argmax(&m.forward(&mut sess, &[next], None, 1)?) as u32
            };
        }
        let tg = decode as f64 / t2.elapsed().as_secs_f64();
        len += decode;
        if ref_pp == 0.0 {
            ref_pp = pp;
            ref_tg = tg;
        }
        println!(
            "| {d} | {pp:.1} | {:.0} % | {tg:.2} | {:.0} % | {fill:.1} | {:.0} |",
            pp / ref_pp * 100.0,
            tg / ref_tg * 100.0,
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn kernel_test() -> Result<()> {
    use malleus_hip::{Arg, DeviceBuffer, Stream};
    let arch = select_device()?;
    let k = Kernels::load(&arch)?;
    let s = Stream::null();
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    let signs = DeviceBuffer::from_slice(&malleus_gguf::quant::hadamard_signs(malleus_gguf::quant::HADAMARD_SEED))?;
    let (n, kd) = (256usize, 1024usize);
    let q: Vec<u8> = (0..n * kd / 2).map(|_| ((rnd() + 1.0) * 127.9) as u8).collect();
    let sc: Vec<u16> = (0..n * kd / 128)
        .map(|_| half::f16::from_f32(0.01 + 0.01 * rnd().abs()).to_bits())
        .collect();
    let (qd, scd) = (DeviceBuffer::from_slice(&q)?, DeviceBuffer::from_slice(&sc)?);
    for t in [5usize, 24, 50, 130, 700] {
        let x: Vec<f32> = (0..t * kd).map(|_| rnd() * 3.0).collect();
        let xd = DeviceBuffer::from_slice(&x)?;
        let xr = DeviceBuffer::alloc(t * kd * 4)?;
        let xh = DeviceBuffer::alloc(t * kd * 2)?;
        let (o1, o2) = (DeviceBuffer::zeroed(t * n * 4)?, DeviceBuffer::zeroed(t * n * 4)?);
        let chunks = (t * kd / 128) as u32;
        k.get("hadamard128").launch(
            (chunks, 1, 1),
            (128, 1, 1),
            0,
            &s,
            &[(&xr).into(), (&xd).into(), (&signs).into(), kd.into()],
        )?;
        k.get("gemm_mls4").launch(
            (n.div_ceil(16) as u32, t.div_ceil(8) as u32, 1),
            (256, 1, 1),
            0,
            &s,
            &[
                (&o1).into(),
                (&xr).into(),
                (&qd).into(),
                (&scd).into(),
                t.into(),
                n.into(),
                kd.into(),
                0i32.into(),
            ],
        )?;
        k.get("hadamard128_h").launch(
            (chunks, 1, 1),
            (128, 1, 1),
            0,
            &s,
            &[(&xh).into(), (&xd).into(), (&signs).into(), kd.into()],
        )?;
        for (name, bm) in [("gemm_wmma_m32", 32usize), ("gemm_wmma_m64", 64), ("gemm_wmma", 128)] {
            o2.memset(0, &s)?;
            let args: [Arg; 9] = [
                (&o2).into(),
                (&xh).into(),
                (&qd).into(),
                (&scd).into(),
                t.into(),
                n.into(),
                kd.into(),
                0i32.into(),
                0i32.into(),
            ];
            k.get(name).launch(
                (n.div_ceil(128) as u32, t.div_ceil(bm) as u32, 1),
                (256, 1, 1),
                0,
                &s,
                &args,
            )?;
            s.synchronize()?;
            let (a, b) = (o1.download::<f32>(t * n)?, o2.download::<f32>(t * n)?);
            println!("T={t:4} {name:14} rel err {:.2e}", rel_err(&b, &a));
        }
        let xq = DeviceBuffer::alloc(t * kd)?;
        let xsc = DeviceBuffer::alloc(t * kd / 128 * 4)?;
        k.get("rot_quant_i8").launch(
            (chunks, 1, 1),
            (128, 1, 1),
            0,
            &s,
            &[(&xq).into(), (&xsc).into(), (&xd).into(), (&signs).into(), kd.into()],
        )?;
        // The same MLS4 weights with int4 activations: gemm_a8_impl shares one body with gemm_w4a8, so both
        // operand widths are checked against the same reference. W4A4 quantizes the activations to 4 bits, so its
        // error is expected to sit about an order above W4A8 rather than near it.
        let xq4 = DeviceBuffer::alloc(t * kd / 2)?;
        let xsc4 = DeviceBuffer::alloc(t * kd / 128 * 4)?;
        k.get("rot_quant_i4").launch(
            (chunks, 1, 1),
            (128, 1, 1),
            0,
            &s,
            &[(&xq4).into(), (&xsc4).into(), (&xd).into(), (&signs).into(), kd.into()],
        )?;
        for (name, x, xs, bm) in [
            ("gemm_w4a8_m32", &xq, &xsc, 32usize),
            ("gemm_w4a8_m64", &xq, &xsc, 64),
            ("gemm_w4a8", &xq, &xsc, 128),
            ("gemm_w4a4_m32", &xq4, &xsc4, 32),
            ("gemm_w4a4_m64", &xq4, &xsc4, 64),
            ("gemm_w4a4", &xq4, &xsc4, 128),
        ] {
            o2.memset(0, &s)?;
            let args: [Arg; 9] = [
                (&o2).into(),
                x.into(),
                xs.into(),
                (&qd).into(),
                (&scd).into(),
                t.into(),
                n.into(),
                kd.into(),
                0i32.into(),
            ];
            k.get(name).launch(
                ((n.div_ceil(128) * t.div_ceil(bm)) as u32, 1, 1),
                (256, 1, 1),
                0,
                &s,
                &args,
            )?;
            s.synchronize()?;
            let (a, b) = (o1.download::<f32>(t * n)?, o2.download::<f32>(t * n)?);
            println!("T={t:4} {name:16} rel err {:.2e}", rel_err(&b, &a));
        }
        // The grouped MoE GEMMs on one expert that holds every row: offs [0, t], one tile per BM rows. Both operand
        // widths share moe_a4_impl, so this covers the W4A8 path and the W4A4 one; the GLU epilogue is the same code
        // with a different tail.
        let offs = DeviceBuffer::from_slice(&[0i32, t as i32])?;
        for (name, x, xs, bm) in [
            ("moe_w4a8_m32", &xq, &xsc, 32usize),
            ("moe_w4a8", &xq, &xsc, 64),
            ("moe_w4a8_m128", &xq, &xsc, 128),
            ("moe_w4a4_m32", &xq4, &xsc4, 32),
            ("moe_w4a4", &xq4, &xsc4, 64),
            ("moe_w4a4_m128", &xq4, &xsc4, 128),
        ] {
            let nt = t.div_ceil(bm);
            let tv: Vec<i32> = (0..nt).flat_map(|i| [0i32, (i * bm) as i32]).collect();
            let (tiles, ntl) = (DeviceBuffer::from_slice(&tv)?, DeviceBuffer::from_slice(&[nt as i32])?);
            o2.memset(0, &s)?;
            let args: [Arg; 10] = [
                (&o2).into(),
                x.into(),
                xs.into(),
                (&qd).into(),
                (&scd).into(),
                (&offs).into(),
                (&tiles).into(),
                (&ntl).into(),
                n.into(),
                kd.into(),
            ];
            k.get(name)
                .launch(((nt * n.div_ceil(128)) as u32, 1, 1), (256, 1, 1), 0, &s, &args)?;
            s.synchronize()?;
            let (a, b) = (o1.download::<f32>(t * n)?, o2.download::<f32>(t * n)?);
            println!("T={t:4} {name:16} rel err {:.2e}", rel_err(&b, &a));
        }
    }
    // MLS2 (2-bit) GEMV/GEMM against a host reference built from the same quantizer: the kernel's own rounding is
    // the only thing under test, so the weights are generated, rotated and quantized here, then dequantized back.
    {
        use malleus_gguf::quant::{self, BLOCK, INT2_GROUP_BYTES, Int2Grid};
        let (n2, kd2) = (256usize, 1024usize);
        let signs2 = quant::hadamard_signs(malleus_gguf::quant::HADAMARD_SEED);
        let (mut q2, mut sc2, mut wdeq) = (
            vec![0u8; n2 * kd2 / 4],
            vec![0u16; n2 * kd2 / BLOCK],
            vec![0f32; n2 * kd2],
        );
        for r in 0..n2 {
            let mut row: Vec<f32> = (0..kd2).map(|_| rnd()).collect();
            quant::rotate_row(&mut row, &signs2);
            for (g, grp) in row.as_chunks::<BLOCK>().0.iter().enumerate() {
                let (bits, sc) = quant::quantize_group_int2(grp, Int2Grid::E2m1);
                let at = r * (kd2 / 4) + g * INT2_GROUP_BYTES;
                q2[at..at + INT2_GROUP_BYTES].copy_from_slice(&bits);
                sc2[r * (kd2 / BLOCK) + g] = half::f16::from_f32(sc).to_bits();
                let mut dq = [0f32; BLOCK];
                quant::dequant_int2(&bits, sc, Int2Grid::E2m1, &mut dq);
                wdeq[r * kd2 + g * BLOCK..r * kd2 + (g + 1) * BLOCK].copy_from_slice(&dq);
            }
        }
        let (qd2, scd2) = (DeviceBuffer::from_slice(&q2)?, DeviceBuffer::from_slice(&sc2)?);
        for (t, name) in [(1usize, "gemv_mls2"), (5, "gemm_mls2"), (24, "gemm_mls2")] {
            let x: Vec<f32> = (0..t * kd2).map(|_| rnd() * 3.0).collect();
            let mut xr = x.clone();
            for row in xr.chunks_mut(kd2) {
                quant::rotate_row(row, &signs2);
            }
            let xd = DeviceBuffer::from_slice(&xr)?;
            let out = DeviceBuffer::zeroed(t * n2 * 4)?;
            let grid = if t == 1 {
                (n2.div_ceil(32) as u32, 1, 1)
            } else {
                (n2.div_ceil(16) as u32, t.div_ceil(8) as u32, 1)
            };
            k.get(name).launch(
                grid,
                (256, 1, 1),
                0,
                &s,
                &[
                    (&out).into(),
                    (&xd).into(),
                    (&qd2).into(),
                    (&scd2).into(),
                    t.into(),
                    n2.into(),
                    kd2.into(),
                    0i32.into(),
                ],
            )?;
            let mut reference = vec![0f32; t * n2];
            for a in 0..t {
                for r in 0..n2 {
                    reference[a * n2 + r] = (0..kd2).map(|kk| xr[a * kd2 + kk] * wdeq[r * kd2 + kk]).sum();
                }
            }
            let got = out.download::<f32>(t * n2)?;
            println!("T={t:4} {name:16} rel err {:.2e}", rel_err(&got, &reference));
            assert!(
                rel_err(&got, &reference) < 1e-3,
                "{name} disagree with the host reference"
            );
        }
        // The grouped W2A8 prefill kernel on one expert that owns every row, against the same host reference with
        // the activation quantized exactly as the kernel sees it.
        for (t, bm) in [(5usize, 64usize), (24, 64), (130, 64), (130, 128)] {
            let x: Vec<f32> = (0..t * kd2).map(|_| rnd() * 3.0).collect();
            let xd = DeviceBuffer::from_slice(&x)?;
            let (xq, xsc) = (DeviceBuffer::zeroed(t * kd2)?, DeviceBuffer::zeroed(t * kd2 / 128 * 4)?);
            k.get("rot_quant_i8").launch(
                ((t * kd2 / 128) as u32, 1, 1),
                (128, 1, 1),
                0,
                &s,
                &[(&xq).into(), (&xsc).into(), (&xd).into(), (&signs).into(), kd2.into()],
            )?;
            let nt = t.div_ceil(bm);
            let tv: Vec<i32> = (0..nt).flat_map(|i| [0i32, (i * bm) as i32]).collect();
            let (tiles, ntl) = (DeviceBuffer::from_slice(&tv)?, DeviceBuffer::from_slice(&[nt as i32])?);
            let offs = DeviceBuffer::from_slice(&[0i32, t as i32])?;
            let out = DeviceBuffer::zeroed(t * n2 * 4)?;
            let args: [Arg; 10] = [
                (&out).into(),
                (&xq).into(),
                (&xsc).into(),
                (&qd2).into(),
                (&scd2).into(),
                (&offs).into(),
                (&tiles).into(),
                (&ntl).into(),
                n2.into(),
                kd2.into(),
            ];
            k.get(if bm == 128 { "moe_w2a8_m128" } else { "moe_w2a8" }).launch(
                ((nt * n2.div_ceil(128)) as u32, 1, 1),
                (256, 1, 1),
                0,
                &s,
                &args,
            )?;
            let xi: Vec<i8> = xq.download(t * kd2)?;
            let xs: Vec<f32> = xsc.download(t * kd2 / 128)?;
            let mut reference = vec![0f32; t * n2];
            for a in 0..t {
                for r in 0..n2 {
                    reference[a * n2 + r] = (0..kd2)
                        .map(|kk| xi[a * kd2 + kk] as f32 * xs[a * (kd2 / 128) + kk / 128] * wdeq[r * kd2 + kk])
                        .sum();
                }
            }
            let got = out.download::<f32>(t * n2)?;
            let err = rel_err(&got, &reference);
            println!("T={t:4} moe_w2a8(bm={bm:3}) rel err {err:.2e}");
            assert!(err < 1e-3, "moe_w2a8 disagrees with the host reference");
        }
    }
    // The one-wave rotation + quantizer must produce the same bytes as the 128-thread shared-memory one, which is
    // the reference the converter's host rotation agrees with: same arithmetic, same order, so the bytes are equal
    // rather than close.
    {
        let (rows, kw) = (8usize, 1024usize);
        let x: Vec<f32> = (0..rows * kw).map(|_| rnd() * 3.0).collect();
        let xd = DeviceBuffer::from_slice(&x)?;
        let chunks = (rows * kw / 128) as u32;
        for (old, new, i4) in [
            ("rot_quant_i8", "rot_quant_i8_w", false),
            ("rot_quant_i4", "rot_quant_i4_w", true),
        ] {
            let nb = if i4 { rows * kw / 2 } else { rows * kw };
            let (qa, qb) = (DeviceBuffer::zeroed(nb)?, DeviceBuffer::zeroed(nb)?);
            let (sa, sb) = (
                DeviceBuffer::zeroed(chunks as usize * 4)?,
                DeviceBuffer::zeroed(chunks as usize * 4)?,
            );
            for (name, q, sc, block) in [(old, &qa, &sa, 128u32), (new, &qb, &sb, 32)] {
                k.get(name).launch(
                    (chunks, 1, 1),
                    (block, 1, 1),
                    0,
                    &s,
                    &[q.into(), sc.into(), (&xd).into(), (&signs).into(), kw.into()],
                )?;
            }
            s.synchronize()?;
            let (a, b) = (qa.download::<u8>(nb)?, qb.download::<u8>(nb)?);
            let (fa, fb) = (
                sa.download::<f32>(chunks as usize)?,
                sb.download::<f32>(chunks as usize)?,
            );
            let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            let sdiff = fa.iter().zip(&fb).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            println!("{new:16} vs {old:12} bytes diff {diff} / {nb}, scale diff {sdiff} / {chunks}");
            assert_eq!((diff, sdiff), (0, 0), "{new} must match {old} byte for byte");
        }
    }
    // ---- Inkling-2 (issue #62): the short convolution, the per-head transforms, the relative bias and the
    // attention, each against a replica of the reference arithmetic on the host.
    {
        let (t, c, heads, dim) = (6usize, 256usize, 2usize, 128usize);
        let x: Vec<f32> = (0..t * c).map(|_| rnd()).collect();
        let w: Vec<u16> = (0..c * 4)
            .map(|_| half::bf16::from_f32(rnd() * 0.1).to_bits())
            .collect();
        let past: Vec<f32> = (0..3 * c).map(|_| rnd()).collect();
        let xd = DeviceBuffer::from_slice(&x)?;
        let wd = DeviceBuffer::from_slice(&w)?;
        let pd = DeviceBuffer::from_slice(&past)?;
        let yd = DeviceBuffer::zeroed(t * c * 4)?;
        k.get("inkling2_sconv").launch(
            ((t * c).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &s,
            &[
                (&yd).into(),
                (&xd).into(),
                (&wd).into(),
                Arg::U64(0),
                (&pd).into(),
                t.into(),
                c.into(),
                heads.into(),
                (5.0f32.sqrt()).into(),
            ],
        )?;
        s.synchronize()?;
        let got = yd.download::<f32>(t * c)?;
        let mut want = vec![0f32; t * c];
        for tt in 0..t {
            for cc in 0..c {
                let mut acc = 0f32;
                for lag in 0..4 {
                    let pp = tt as isize - lag as isize;
                    let vv = if pp >= 0 {
                        x[pp as usize * c + cc]
                    } else {
                        past[(pp + 3) as usize * c + cc]
                    };
                    acc += half::bf16::from_bits(w[cc * 4 + (3 - lag)]).to_f32() * vv;
                }
                want[tt * c + cc] = (x[tt * c + cc] + acc) / 5.0f32.sqrt();
            }
        }
        println!("inkling2_sconv     rel err {:.2e}", rel_err(&want, &got));
        // `dim` 16 is the relative-bias stream: narrower than the block, the case the shared-memory reduction used
        // to read past.
        for dim in [dim, 16usize] {
            for (name, scaled, tau) in [("inkling2_l2_clip", true, true), ("inkling2_l2_clip", true, false)] {
                let rows = t * heads;
                let xa: Vec<f32> = (0..rows * dim).map(|_| rnd() * 4.0).collect();
                let sc: Vec<f32> = (0..heads * dim).map(|_| 1.0 + 0.2 * rnd()).collect();
                let ta: Vec<f32> = (0..t).map(|_| 1.0 + 0.1 * rnd().abs()).collect();
                let xad = DeviceBuffer::from_slice(&xa)?;
                let scd = DeviceBuffer::from_slice(&sc)?;
                let tad = DeviceBuffer::from_slice(&ta)?;
                let yo = DeviceBuffer::zeroed(rows * dim * 4)?;
                let thr = 8.0f32;
                let tau_arg: Arg = if tau { (&tad).into() } else { Arg::U64(0) };
                k.get(name).launch(
                    (rows as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    &s,
                    &[
                        (&yo).into(),
                        (&xad).into(),
                        (&scd).into(),
                        tau_arg,
                        t.into(),
                        heads.into(),
                        dim.into(),
                        thr.into(),
                    ],
                )?;
                s.synchronize()?;
                let got = yo.download::<f32>(rows * dim)?;
                let mut want = vec![0f32; rows * dim];
                for r in 0..rows {
                    let v: Vec<f32> = (0..dim).map(|i| xa[r * dim + i] * sc[(r % heads) * dim + i]).collect();
                    let norm = v.iter().map(|q| q * q).sum::<f32>().sqrt();
                    let kk = (thr / (norm + 1e-6)).min(1.0) * if tau { ta[r / heads] } else { 1.0 };
                    for i in 0..dim {
                        want[r * dim + i] = v[i] * kk;
                    }
                }
                println!(
                    "{name:18} tau={tau:<5} dim={dim:3} rel err {:.2e} (scaled={scaled})",
                    rel_err(&want, &got)
                );
            }
        }
        let rows = t * heads;
        let xa: Vec<f32> = (0..rows * dim).map(|_| rnd() * 3.0).collect();
        let xad = DeviceBuffer::from_slice(&xa)?;
        let yo = DeviceBuffer::zeroed(rows * dim * 4)?;
        k.get("inkling2_rms_unnorm").launch(
            (rows as u32, 1, 1),
            (128, 1, 1),
            0,
            &s,
            &[(&yo).into(), (&xad).into(), rows.into(), dim.into(), 1e-6f32.into()],
        )?;
        s.synchronize()?;
        let got = yo.download::<f32>(rows * dim)?;
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            let ms = (0..dim).map(|i| xa[r * dim + i] * xa[r * dim + i]).sum::<f32>() / dim as f32;
            for i in 0..dim {
                want[r * dim + i] = xa[r * dim + i] / (ms + 1e-6).sqrt();
            }
        }
        println!("inkling2_rms_unnorm rel err {:.2e}", rel_err(&want, &got));
        // The same with a width below the block: the relative-bias-shaped case.
        let (rw, dw) = (rows, 16usize);
        let xw: Vec<f32> = (0..rw * dw).map(|_| rnd() * 3.0).collect();
        let xwd = DeviceBuffer::from_slice(&xw)?;
        let yw = DeviceBuffer::zeroed(rw * dw * 4)?;
        k.get("inkling2_rms_unnorm").launch(
            (rw as u32, 1, 1),
            (128, 1, 1),
            0,
            &s,
            &[(&yw).into(), (&xwd).into(), rw.into(), dw.into(), 1e-6f32.into()],
        )?;
        s.synchronize()?;
        let got = yw.download::<f32>(rw * dw)?;
        let mut want = vec![0f32; rw * dw];
        for r in 0..rw {
            let ms = (0..dw).map(|i| xw[r * dw + i] * xw[r * dw + i]).sum::<f32>() / dw as f32;
            for i in 0..dw {
                want[r * dw + i] = xw[r * dw + i] / (ms + 1e-6).sqrt();
            }
        }
        println!("inkling2_rms_unnorm dim=16 rel err {:.2e}", rel_err(&want, &got));
        // Relative bias and attention: one sliding-shaped case with a whole window.
        let (t2, h2, nkv2, drel, extent) = (5usize, 2usize, 1usize, 4usize, 3usize);
        let rel: Vec<f32> = (0..t2 * h2 * drel).map(|_| rnd()).collect();
        let proj: Vec<f32> = (0..drel * extent).map(|_| rnd()).collect();
        let reld = DeviceBuffer::from_slice(&rel)?;
        let projd = DeviceBuffer::from_slice(&proj)?;
        let biasd = DeviceBuffer::zeroed(h2 * t2 * t2 * 4)?;
        k.get("inkling2_rel_bias").launch(
            (((h2 * t2 * t2) as f32 / 256.0).ceil() as u32, 1, 1),
            (256, 1, 1),
            0,
            &s,
            &[
                (&biasd).into(),
                (&reld).into(),
                (&projd).into(),
                t2.into(),
                h2.into(),
                drel.into(),
                extent.into(),
            ],
        )?;
        s.synchronize()?;
        let bias = biasd.download::<f32>(h2 * t2 * t2)?;
        let mut wantb = vec![0f32; h2 * t2 * t2];
        for hh in 0..h2 {
            for qq in 0..t2 {
                for ss in 0..t2 {
                    let d = qq as isize - ss as isize;
                    if d >= 0 && (d as usize) < extent {
                        let mut acc = 0f32;
                        for i in 0..drel {
                            acc += rel[(qq * h2 + hh) * drel + i] * proj[i * extent + d as usize];
                        }
                        wantb[(hh * t2 + qq) * t2 + ss] = acc;
                    }
                }
            }
        }
        println!("inkling2_rel_bias  rel err {:.2e}", rel_err(&wantb, &bias));
        let qa: Vec<f32> = (0..t2 * h2 * dim).map(|_| rnd()).collect();
        let ka: Vec<f32> = (0..t2 * nkv2 * dim).map(|_| rnd()).collect();
        let va: Vec<f32> = (0..t2 * nkv2 * dim).map(|_| rnd()).collect();
        let ga: Vec<f32> = (0..t2 * h2 * dim).map(|_| rnd() * 3.0).collect();
        let sa: Vec<f32> = (0..t2 * h2).map(|_| rnd()).collect();
        let (qad, kad, vad, gad, sad) = (
            DeviceBuffer::from_slice(&qa)?,
            DeviceBuffer::from_slice(&ka)?,
            DeviceBuffer::from_slice(&va)?,
            DeviceBuffer::from_slice(&ga)?,
            DeviceBuffer::from_slice(&sa)?,
        );
        let oa = DeviceBuffer::zeroed(t2 * h2 * dim * 4)?;
        let scale = 1.0f32 / (dim as f32).sqrt();
        k.get("inkling2_attn").launch(
            ((t2 * h2) as u32, 1, 1),
            (128, 1, 1),
            0,
            &s,
            &[
                (&oa).into(),
                (&qad).into(),
                (&kad).into(),
                (&vad).into(),
                (&biasd).into(),
                (&sad).into(),
                (&gad).into(),
                t2.into(),
                h2.into(),
                nkv2.into(),
                dim.into(),
                0.into(),
                scale.into(),
            ],
        )?;
        s.synchronize()?;
        let got = oa.download::<f32>(t2 * h2 * dim)?;
        let mut want = vec![0f32; t2 * h2 * dim];
        for tt in 0..t2 {
            for hh in 0..h2 {
                let kh = hh / (h2 / nkv2);
                let mut sc = vec![0f32; tt + 1];
                let mut mx = sa[tt * h2 + hh];
                for ss in 0..=tt {
                    let mut acc = 0f32;
                    for i in 0..dim {
                        acc += qa[(tt * h2 + hh) * dim + i] * ka[(ss * nkv2 + kh) * dim + i];
                    }
                    sc[ss] = acc * scale + bias[(hh * t2 + tt) * t2 + ss];
                    mx = mx.max(sc[ss]);
                }
                let sum: f32 = sc.iter().map(|v| (v - mx).exp()).sum::<f32>() + (sa[tt * h2 + hh] - mx).exp();
                for i in 0..dim {
                    let mut acc = 0f32;
                    for ss in 0..=tt {
                        acc += (sc[ss] - mx).exp() / sum * va[(ss * nkv2 + kh) * dim + i];
                    }
                    let g = ga[(tt * h2 + hh) * dim + i];
                    want[(tt * h2 + hh) * dim + i] = acc / (1.0 + (-g).exp());
                }
            }
        }
        println!("inkling2_attn      rel err {:.2e}", rel_err(&want, &got));
        // Attention residual: 4 states, 3 queries, 2304 wide, at 2 positions.
        let (tr, nr, hd, nqr) = (2usize, 4usize, 2304usize, 3usize);
        let xr: Vec<f32> = (0..tr * nr * hd).map(|_| rnd()).collect();
        let qr: Vec<f32> = (0..nqr * hd).map(|_| rnd()).collect();
        let (xrd, qrd) = (DeviceBuffer::from_slice(&xr)?, DeviceBuffer::from_slice(&qr)?);
        let ord = DeviceBuffer::zeroed(tr * nqr * hd * 4)?;
        k.get("inkling2_attn_res").launch(
            ((tr * nqr) as u32, 1, 1),
            (256, 1, 1),
            0,
            &s,
            &[
                (&ord).into(),
                (&qrd).into(),
                (&xrd).into(),
                tr.into(),
                nr.into(),
                hd.into(),
                nqr.into(),
                1e-6f32.into(),
            ],
        )?;
        s.synchronize()?;
        let got = ord.download::<f32>(tr * nqr * hd)?;
        let mut want = vec![0f32; tr * nqr * hd];
        for tt in 0..tr {
            for qi in 0..nqr {
                let mut sc = vec![0f32; nr];
                let mut fac = vec![0f32; nr];
                for j in 0..nr {
                    let row = &xr[(j * tr + tt) * hd..(j * tr + tt + 1) * hd];
                    let ms = row.iter().map(|v| v * v).sum::<f32>() / hd as f32;
                    fac[j] = 1.0 / (ms + 1e-6).sqrt();
                    sc[j] = row
                        .iter()
                        .zip(&qr[qi * hd..(qi + 1) * hd])
                        .map(|(v, qv)| v * fac[j] * qv)
                        .sum::<f32>()
                        / (hd as f32).sqrt();
                }
                let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                let sum: f32 = sc.iter().map(|v| (v - mx).exp()).sum();
                for i in 0..hd {
                    let mut acc = 0f32;
                    for j in 0..nr {
                        acc += (sc[j] - mx).exp() / sum * xr[(j * tr + tt) * hd + i] * fac[j];
                    }
                    want[(tt * nqr + qi) * hd + i] = acc;
                }
            }
        }
        println!("inkling2_attn_res  rel err {:.2e}", rel_err(&want, &got));
        // The decode path's three kernels against the same arithmetic at one token.
        let pdc = DeviceBuffer::from_slice(&past)?;
        let ydc = DeviceBuffer::zeroed(c * 4)?;
        let x1 = DeviceBuffer::from_slice(&x[..c])?;
        k.get("inkling2_sconv_dec").launch(
            (c.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &s,
            &[
                (&ydc).into(),
                (&x1).into(),
                (&wd).into(),
                Arg::U64(0),
                (&pdc).into(),
                c.into(),
                heads.into(),
                (5.0f32.sqrt()).into(),
            ],
        )?;
        s.synchronize()?;
        let got = ydc.download::<f32>(c)?;
        let gotp = pdc.download::<f32>(3 * c)?;
        let mut want = vec![0f32; c];
        for cc in 0..c {
            let mut acc = 0f32;
            for lag in 0..4 {
                let vv = if lag == 0 { x[cc] } else { past[(3 - lag) * c + cc] };
                acc += half::bf16::from_bits(w[cc * 4 + (3 - lag)]).to_f32() * vv;
            }
            want[cc] = (x[cc] + acc) / 5.0f32.sqrt();
        }
        let mut wantp = past.clone();
        for cc in 0..c {
            wantp[cc] = past[c + cc];
            wantp[c + cc] = past[2 * c + cc];
            wantp[2 * c + cc] = x[cc];
        }
        println!(
            "inkling2_sconv_dec rel err {:.2e}, history rel err {:.2e}",
            rel_err(&want, &got),
            rel_err(&wantp, &gotp)
        );
        // The relative bias of one query at `pos` over the keys [k0, k0 + nk).
        let (posb, k0b, nkb) = (4usize, 1usize, 4usize);
        let relb: Vec<f32> = (0..h2 * drel).map(|_| rnd()).collect();
        let relbd = DeviceBuffer::from_slice(&relb)?;
        let biasdb = DeviceBuffer::zeroed(h2 * nkb * 4)?;
        k.get("inkling2_rel_bias_dec").launch(
            ((h2 * nkb).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &s,
            &[
                (&biasdb).into(),
                (&relbd).into(),
                (&projd).into(),
                posb.into(),
                k0b.into(),
                nkb.into(),
                h2.into(),
                drel.into(),
                extent.into(),
            ],
        )?;
        s.synchronize()?;
        let biasb = biasdb.download::<f32>(h2 * nkb)?;
        let mut wantb = vec![0f32; h2 * nkb];
        for hh in 0..h2 {
            for ss in 0..nkb {
                let d = posb as isize - (k0b + ss) as isize;
                if d >= 0 && (d as usize) < extent {
                    let mut acc = 0f32;
                    for i in 0..drel {
                        acc += relb[hh * drel + i] * proj[i * extent + d as usize];
                    }
                    wantb[hh * nkb + ss] = acc;
                }
            }
        }
        println!("inkling2_rel_bias_dec rel err {:.2e}", rel_err(&wantb, &biasb));
        // One query over a cache of `nkb` keys from `k0b`, `posb` being its last one.
        let biasd = DeviceBuffer::from_slice(&wantb)?;
        let od = DeviceBuffer::zeroed(h2 * dim * 4)?;
        k.get("inkling2_attn_dec").launch(
            (h2 as u32, 1, 1),
            (128, 1, 1),
            0,
            &s,
            &[
                (&od).into(),
                (&qad).into(),
                (&kad).into(),
                (&vad).into(),
                (&biasd).into(),
                (&sad).into(),
                (&gad).into(),
                k0b.into(),
                nkb.into(),
                h2.into(),
                nkv2.into(),
                dim.into(),
                scale.into(),
            ],
        )?;
        s.synchronize()?;
        let got = od.download::<f32>(h2 * dim)?;
        let mut want = vec![0f32; h2 * dim];
        for hh in 0..h2 {
            let kh = hh / (h2 / nkv2);
            let sc: Vec<f32> = (0..nkb)
                .map(|ss| {
                    let mut acc = 0f32;
                    for i in 0..dim {
                        acc += qa[hh * dim + i] * ka[((k0b + ss) * nkv2 + kh) * dim + i];
                    }
                    acc * scale + wantb[hh * nkb + ss]
                })
                .collect();
            let mx = sc.iter().copied().fold(sa[hh], f32::max);
            let sum: f32 = sc.iter().map(|v| (v - mx).exp()).sum::<f32>() + (sa[hh] - mx).exp();
            for i in 0..dim {
                let mut acc = 0f32;
                for ss in 0..nkb {
                    acc += (sc[ss] - mx).exp() / sum * va[((k0b + ss) * nkv2 + kh) * dim + i];
                }
                let g = ga[hh * dim + i];
                want[hh * dim + i] = acc / (1.0 + (-g).exp());
            }
        }
        println!("inkling2_attn_dec  rel err {:.2e}", rel_err(&want, &got));
        // Router and expert activation.
        let (er, topk, rs) = (128usize, 8usize, 8.0f32);
        let lg: Vec<f32> = (0..t * er).map(|_| rnd() * 3.0).collect();
        let bs: Vec<f32> = (0..er).map(|_| rnd() * 0.1).collect();
        let (lgd, bsd) = (DeviceBuffer::from_slice(&lg)?, DeviceBuffer::from_slice(&bs)?);
        let ids = DeviceBuffer::zeroed(t * topk * 4)?;
        let wts = DeviceBuffer::zeroed(t * topk * 4)?;
        k.get("inkling2_moe_route").launch(
            (t as u32, 1, 1),
            (256, 1, 1),
            0,
            &s,
            &[
                (&ids).into(),
                (&wts).into(),
                (&lgd).into(),
                (&bsd).into(),
                er.into(),
                topk.into(),
                rs.into(),
            ],
        )?;
        s.synchronize()?;
        let gid = ids.download::<i32>(t * topk)?;
        let gw = wts.download::<f32>(t * topk)?;
        let mut wid = vec![0i32; t * topk];
        let mut ww = vec![0f32; t * topk];
        for tt in 0..t {
            let mut order: Vec<usize> = (0..er).collect();
            order.sort_by(|&a, &b| {
                let sa = 1.0f32 / (1.0 + (-lg[tt * er + a]).exp()) + bs[a];
                let sb = 1.0f32 / (1.0 + (-lg[tt * er + b]).exp()) + bs[b];
                sb.partial_cmp(&sa).unwrap().then(a.cmp(&b))
            });
            let sel: Vec<usize> = order[..topk].to_vec();
            let mut sum = 0f32;
            for &e in &sel {
                sum += 1.0 / (1.0 + (-lg[tt * er + e]).exp());
            }
            for (j, &e) in sel.iter().enumerate() {
                wid[tt * topk + j] = e as i32;
                ww[tt * topk + j] = (1.0 / (1.0 + (-lg[tt * er + e]).exp())) / sum * rs;
            }
        }
        let ids_ok = gid == wid;
        println!(
            "inkling2_moe_route ids equal={ids_ok} weights rel err {:.2e}",
            rel_err(&ww, &gw)
        );
        // Activation: I = 512, 3 pairs, against the host formula.
        let (ii, np) = (512usize, 3usize);
        let gu: Vec<f32> = (0..np * 2 * ii).map(|_| rnd() * 8.0).collect();
        let gud = DeviceBuffer::from_slice(&gu)?;
        let act = DeviceBuffer::zeroed(np * ii * 4)?;
        k.get("inkling2_moe_act").launch(
            ((np * ii / 128) as u32, 1, 1),
            (128, 1, 1),
            0,
            &s,
            &[
                (&act).into(),
                (&gud).into(),
                (&signs).into(),
                ii.into(),
                4.0f32.into(),
                25.0f32.into(),
            ],
        )?;
        s.synchronize()?;
        let ga = act.download::<f32>(np * ii)?;
        let mut wa = vec![0f32; np * ii];
        let sgn = malleus_gguf::quant::hadamard_signs(malleus_gguf::quant::HADAMARD_SEED);
        for p in 0..np {
            let mut row = vec![0f32; ii];
            for b in 0..ii / 128 {
                let mut blk = [0f32; 128];
                for i in 0..128 {
                    let pos = b * 128 + i;
                    let g = gu[p * 2 * ii + 2 * pos];
                    let u = gu[p * 2 * ii + 2 * pos + 1];
                    blk[i] = (g / 4.0).tanh() * 4.0 / (1.0 + (-g).exp()) * (u / 25.0).tanh() * 25.0 * sgn[i] as f32;
                }
                for h in [1usize, 2, 4, 8, 16, 32, 64] {
                    let mut next = blk;
                    for i in 0..128 {
                        next[i] = if i & h == 0 {
                            blk[i] + blk[i ^ h]
                        } else {
                            blk[i ^ h] - blk[i]
                        };
                    }
                    blk = next;
                }
                for i in 0..128 {
                    row[b * 128 + i] = blk[i] * 0.088_388_35;
                }
            }
            wa[p * ii..(p + 1) * ii].copy_from_slice(&row);
        }
        println!("inkling2_moe_act   rel err {:.2e}", rel_err(&wa, &ga));
        // Loop controller: hidden 512, rank 8, two tokens.
        let (hc, rk, tc) = (512usize, 8usize, 2usize);
        let cur: Vec<f32> = (0..tc * hc).map(|_| rnd()).collect();
        let msg: Vec<f32> = (0..tc * hc).map(|_| rnd()).collect();
        let dn: Vec<u16> = (0..rk * 2 * hc)
            .map(|_| half::bf16::from_f32(rnd() * 0.05).to_bits())
            .collect();
        let gw: Vec<u16> = (0..hc * rk)
            .map(|_| half::bf16::from_f32(rnd() * 0.05).to_bits())
            .collect();
        let fw: Vec<u16> = (0..hc * rk)
            .map(|_| half::bf16::from_f32(rnd() * 0.05).to_bits())
            .collect();
        let (curd, msgd) = (DeviceBuffer::from_slice(&cur)?, DeviceBuffer::from_slice(&msg)?);
        let (dnd, gwd, fwd) = (
            DeviceBuffer::from_slice(&dn)?,
            DeviceBuffer::from_slice(&gw)?,
            DeviceBuffer::from_slice(&fw)?,
        );
        let outc = DeviceBuffer::zeroed(tc * hc * 4)?;
        k.get("inkling2_control").launch(
            (tc as u32, 1, 1),
            (256, 1, 1),
            0,
            &s,
            &[
                (&outc).into(),
                (&curd).into(),
                (&msgd).into(),
                (&dnd).into(),
                (&gwd).into(),
                (&fwd).into(),
                hc.into(),
                rk.into(),
                1e-6f32.into(),
            ],
        )?;
        s.synchronize()?;
        let got = outc.download::<f32>(tc * hc)?;
        let mut want = vec![0f32; tc * hc];
        for tt in 0..tc {
            let c = &cur[tt * hc..(tt + 1) * hc];
            let m = &msg[tt * hc..(tt + 1) * hc];
            let fc = 1.0 / ((c.iter().map(|v| v * v).sum::<f32>() / hc as f32) + 1e-6).sqrt();
            let fm = 1.0 / ((m.iter().map(|v| v * v).sum::<f32>() / hc as f32) + 1e-6).sqrt();
            let mut gate = vec![0f32; rk];
            for j in 0..rk {
                let mut acc = 0f32;
                for i in 0..hc {
                    acc += half::bf16::from_bits(dn[j * 2 * hc + i]).to_f32() * c[i] * fc;
                    acc += half::bf16::from_bits(dn[j * 2 * hc + hc + i]).to_f32() * m[i] * fm;
                }
                gate[j] = acc;
            }
            for i in 0..hc {
                let mut g = 0f32;
                let mut f = 0f32;
                for j in 0..rk {
                    g += half::bf16::from_bits(gw[i * rk + j]).to_f32() * gate[j];
                    f += half::bf16::from_bits(fw[i * rk + j]).to_f32() * gate[j];
                }
                want[tt * hc + i] = c[i] * (1.0 - f.tanh()) + g.tanh() * m[i];
            }
        }
        println!("inkling2_control   rel err {:.2e}", rel_err(&want, &got));
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Devices => {
            for d in malleus_hip::devices()? {
                println!("{} {} {}", d.index, d.arch, d.name);
            }
            let (f, t) = malleus_hip::mem_info()?;
            println!("memory: {:.1} GB free / {:.1} GB", f as f64 / 1e9, t as f64 / 1e9);
        }
        Cmd::BenchBw { gb } => bench_bw(gb)?,
        Cmd::KernelTest => kernel_test()?,
        Cmd::Bench {
            model,
            decode,
            chunk,
            depth,
            ..
        } if !depth.is_empty() => bench_depth(&model, &depth, decode, chunk)?,
        Cmd::Bench {
            model,
            prompt,
            decode,
            chunk,
            verify,
            ..
        } => bench(&model, &prompt, decode, chunk, verify)?,
        Cmd::Serve {
            models_dir,
            default_model,
            host,
            port,
            api_key,
            ctx,
            default_thinking,
            max_tokens,
        } => serve(
            models_dir,
            default_model,
            host,
            port,
            api_key,
            ctx,
            default_thinking,
            max_tokens,
        )?,
        Cmd::DraftInfo { model } => draft_info(&model)?,
        Cmd::DraftVocab { model, out } => draft_vocab(&model, &out)?,
        Cmd::Health { port } => health(port)?,
        Cmd::Convert {
            model,
            out,
            prefill_int8,
            no_prefill_int8,
            no_prefill_int8_experts,
            keep_int8_layers,
            mls4_from,
            expert_mls3,
            expert_mls2,
            mls2_down,
            hc_mls4,
        } => {
            let mls3_layers = match expert_mls3.as_str() {
                "family" => malleus_convert::Mls3Layers::Family,
                "off" => malleus_convert::Mls3Layers::Off,
                s => {
                    let (a, b) = s.split_once('-').ok_or_else(|| {
                        anyhow::anyhow!("--expert-mls3 `{s}`: expected `family`, `off` or an inclusive `A-B`")
                    })?;
                    malleus_convert::Mls3Layers::Range(a.trim().parse()?, b.trim().parse()?)
                }
            };
            let mls2_layers = match expert_mls2.as_str() {
                "family" => malleus_convert::Mls2Layers::Family,
                "off" => malleus_convert::Mls2Layers::Off,
                s => {
                    let (a, b) = s.split_once('-').ok_or_else(|| {
                        anyhow::anyhow!("--expert-mls2 `{s}`: expected `family`, `off` or an inclusive `A-B`")
                    })?;
                    malleus_convert::Mls2Layers::Range(a.trim().parse()?, b.trim().parse()?)
                }
            };
            if no_prefill_int8 {
                eprintln!("note: --no-prefill-int8 is the default now, the flag does nothing");
            }
            let opts = malleus_convert::Options {
                prefill_int8,
                no_prefill_int8_experts,
                keep_int8_layers,
                mls4_from,
                mls3_layers,
                mls2_layers,
                mls2_down,
                hc_mls4,
            };
            if out.extension().and_then(|e| e.to_str()) != Some(malleus_gguf::gguf::EXTENSION) {
                eprintln!(
                    "note: Malleus model files use the .{} extension",
                    malleus_gguf::gguf::EXTENSION
                );
            }
            let t0 = Instant::now();
            malleus_convert::convert(&model, &out, &opts, &|m| eprintln!("{m}"))?;
            eprintln!("converted in {:.0}s", t0.elapsed().as_secs_f64());
        }
        Cmd::Dequant { model, out } => {
            let t0 = Instant::now();
            let s = malleus_convert::dequant::dequantize(&model, &out)?;
            eprintln!(
                "{} tensors, {:.1} GB, {} experts dequantised, {} companions folded, in {} shard(s)",
                s.tensors,
                s.bytes as f64 / 1e9,
                s.experts,
                s.dropped,
                s.shards.len()
            );
            if s.amax_ratio.1 > 0.0 {
                eprintln!(
                    "amax / max|w| over the NVFP4 experts: {:.3} .. {:.3}",
                    s.amax_ratio.0, s.amax_ratio.1
                );
            }
            eprintln!("dequantised in {:.0}s", t0.elapsed().as_secs_f64());
        }
        Cmd::Slim {
            model,
            out,
            keep: keep_families,
        } => {
            if out.extension().and_then(|e| e.to_str()) != Some(malleus_gguf::gguf::EXTENSION) {
                eprintln!(
                    "note: Malleus model files use the .{} extension",
                    malleus_gguf::gguf::EXTENSION
                );
            }
            let g = open_converted(&model)?;
            // The prefill copy is a pair: the int8 weights and their row scales. Both go, or the loader would see
            // half a copy and ignore it.
            let keep = |n: &str| {
                let prefill = n.ends_with(".prefill") || n.ends_with(".prefill.scales");
                !prefill || keep_families.iter().any(|f| n.contains(f.as_str()))
            };
            let dropped: u64 = g
                .tensors()
                .iter()
                .filter(|t| !keep(&t.name))
                .map(|t| t.size as u64)
                .sum();
            anyhow::ensure!(
                dropped > 0,
                "nothing to drop from {}: already a single-copy file, or --keep covers every copy",
                model.display()
            );
            let t0 = Instant::now();
            let (kept, removed) = malleus_gguf::gguf::rewrite_keeping(&g, &g.metadata, &out, &keep)?;
            let size = std::fs::metadata(&out)?.len();
            eprintln!(
                "single copy: {kept} tensors kept, {removed} dropped ({:.2} GB), {:.2} GB written in {:.0}s",
                dropped as f64 / 1e9,
                size as f64 / 1e9,
                t0.elapsed().as_secs_f64()
            );
            if out
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.contains("mls8"))
            {
                eprintln!(
                    "warning: {} says mls8 but carries a single MLS4 copy: name it `<model>-mls4.malleus`",
                    out.display()
                );
            }
        }
        Cmd::Run {
            model,
            prompt,
            max_tokens,
            think,
            ctx,
            spec,
            prompt_ids,
            dump_logits,
        } => run(
            &model,
            &prompt,
            max_tokens,
            think,
            ctx,
            spec,
            prompt_ids.as_deref(),
            dump_logits.as_deref(),
        )?,
        Cmd::Eval {
            model,
            set,
            blobs,
            chunk,
        } => eval(&model, &set, &blobs, chunk)?,
        Cmd::Golden {
            model,
            golden: g,
            blobs,
            ctx,
            greedy,
            only,
            decode,
        } => golden(&model, &g, &blobs, ctx, greedy, only.as_deref(), decode)?,
        Cmd::SpecExact {
            model,
            golden: g,
            only,
            prefix,
            width,
            replay,
            ctx,
        } => spec_exact(&model, &g, &only, &prefix, width, replay, ctx)?,
        Cmd::BatchExact {
            model,
            golden: g,
            only,
            sessions,
            steps,
            limit,
            ctx,
        } => batch_exact(&model, &g, &only, sessions, steps, limit, ctx)?,
    }
    Ok(())
}
