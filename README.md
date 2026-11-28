# Malleus

![Malleus, forged for local AI](https://raw.githubusercontent.com/duckpage/duckpage-brand/refs/heads/main/brands/malleus/logo/svg/cover/cover-dark.svg)

<div align="center">

[🤗 Hugging Face](https://huggingface.co/duckpage) / [🌐 duckpage.dev](https://www.duckpage.dev)

</div>

**Malleus** is DuckPage's open-source LLM inference engine. Today it targets AMD RDNA 3.5 APUs — **gfx1150**
(Strix Point) and **gfx1151** (Strix Halo) — with a Rust host and hand-written HIP kernels, built for one thing: the
long prompt. Other GPUs are in progress.

We put AI on the machines people already own — your computer, your server, your data, nothing sent outside: no
third-party cloud, no sharing, the control stays with you. Malleus is how we push that to the limit on consumer
hardware. It is the engine we run in production, and it is public because it is the concrete proof of the work.

**The advantage today is prompt processing.** Strix APUs share 96–128 GB of memory with the GPU — cheap capacity
for large models at the bandwidth of a laptop — so a long prompt is the case general engines serve worst. On
Ornith-1.5-35B-A3B, Malleus prefills **2.1x faster than llama.cpp in a fresh session and 2.8x at 131 072 tokens**
(196 against 69 tok/s), so a 110 000-token prompt costs 4.5 minutes instead of 12.7. Decode is at parity — at 4
bits a step reads every weight it touches — and quality sits where llama.cpp's `Q4_K_M` sits; the MTP head is how
we go past parity. If the prompts are short, any good engine will do; this one is for the long ones.
[MODELS.md](MODELS.md) has the curves, the method and the numbers.

Nothing here is generic: every kernel targets those two chips, the model files are quantized for their memory
hierarchy, and prefill chunking, the KV and indexer budgets, the MTP head and the expert cache keep a long context
alive on one APU. The engine is readable and auditable: Rust host, HIP kernels, written only from public sources,
with no code, kernels or weight formats from any closed engine.

## Status

| | |
|---|---|
| Models | Five families carried, with the file sizes on the device and the tokens/s measured: [MODELS.md](MODELS.md) |
| Hardware | gfx1150 (development and production), gfx1151 (supported); other GPUs in progress |
| API | OpenAI Chat Completions + Anthropic Messages, streaming, tool calls, `logprobs`, image input, one port |
| Speculative decoding | MTP head + chained drafts, verified in one pass |

### Where it stands

On the model we run in production, Qwen3.8-Flash-Next at **512K of context** ([MODELS.md](MODELS.md) for the full
sweep and the method):

- **prefill**: 1455 tok/s at 1.5K tokens, 1390 at 262K and **1258 at 512K** on one prompt;
- **decode**: 62 tok/s at a fresh context, 58.7 at 32K, 46-63 on 512K requests, with the MTP head and the prompt
  lookup;
- **the window costs nothing to short prompts**: the session picks its own rotation (#164), so a request inside the
  checkpoint's 262 144 positions runs on native frequencies, bit-identical to the 262K configuration;
- **quality**: `golden` 20/21 and 93.41 % of all-position argmax against the bf16 checkpoint, worst KL 0.30965, and
  retrieval verified at both windows (10/10 needle at 262K and at 512K, `idx-recall` 12/12).

A dense model (Qwen3.8-27B) is bandwidth limited on both engines: Malleus measures ~5 % ahead (pp512 209.3
against 191.7 tok/s, tg128 5.02 against 4.74), and the MTP head breaks the ~5.8 tok/s that ~80 GB/s allows a 4-bit
27B.

## How it works

- **Weights**: `.malleus` files (a GGUF v3 container, architecture `malleus.<family>`) with **MLS4** (4-bit,
  randomized Hadamard rotation, group size 128, GPTQ-calibrated) for decode *and* prompt processing; `convert
  --prefill-int8` adds an **MLS8** (int8 per channel) copy so the prompt is read at 8 bits instead — twice the file,
  better prompts, decode unchanged. The name says which one it is: `-mls4` is one copy, `-mls4-mls8` is both.
  `--expert-mls2` stores the routed expert gate/up at **2 bits** (with `--mls2-down` the down projection too); it is
  opt-in and off by default, and [MODELS.md](MODELS.md) has the measurement that keeps it there.
- **Kernels**: hand-written HIP for gfx1150 and gfx1151 — GEMV for decode, fused W8A8 WMMA GEMMs for prefill, chunked
  Gated DeltaNet, flash-decoding attention, vision tower.
- **Host**: a single Rust binary (`malleus`); `unsafe` confined to the HIP bindings; no Python at runtime.
- **Security**: no outbound network connections; rootless, read-only container.

## Quick start

Requirements: a Linux host with the `amdgpu` driver, `/dev/kfd` and the GPU render node, and a GTT limit large
enough for the model.

```sh
# build the dev container (Podman or Docker)
podman build -f deploy/Containerfile --target runtime \
  --build-arg MALLEUS_GPU_TARGETS="gfx1150;gfx1151" -t malleus:dev .

# convert an upstream bf16 model to a .malleus file: one 4-bit copy, the faster file
malleus convert --model /models/Qwen3.8-27B --out /models/qwen38-27b-mls4.malleus

# the same with an 8-bit copy of the weights for prompt processing: twice the file, better prompts
malleus convert --model /models/Qwen3.8-27B --out /models/qwen38-27b-mls4-mls8.malleus --prefill-int8

# run a prompt
malleus run --model qwen38-27b-mls4.malleus --prompt "Hello!"

# serve every .malleus file in a directory: /v1/models lists them and each request chooses with "model"
malleus serve --models-dir /models
```

With `--models-dir`, **one model stays resident at a time**: a request for another model unloads the current one
and loads the new one on the GPU thread, so its first request pays the load and starts with an empty prompt cache.
The tokenizer and EOS ids follow the selected model; context length and tuning stay server-wide. Without
`MALLEUS_DEFAULT_MODEL`, a request that omits `model` uses the only model, and is refused when there are several.

The same port serves two API surfaces. Under `/v1` it is OpenAI- and Anthropic-compatible (`/v1/models`,
`/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/messages/count_tokens`). At the root it serves the
router shape that model-management clients speak:

| Endpoint | What it answers |
|---|---|
| `GET /models` | The same catalog with `status.value` (`loaded` / `loading` / `unloaded` / `failed`), `source`, `meta.n_ctx` and the input modalities, so a client knows what it can use and what it may load. |
| `GET /props?model=<id>` | The model's chat template (a template with `enable_thinking` is how a client learns the model takes a reasoning flag) and `models_autoload: false`: an unloaded model is never routed to implicitly. |
| `POST /models/load` | `{"model": id}`. Answers at once and the entry goes `loading` → `loaded`: a model this size takes ~30 s to a minute, far longer than a client keeps a request open. |
| `POST /models/unload` | `{"model": id}`. Queued behind the request in flight, so a generation is never cut; the entry then reads `unloaded` and the device memory is back. |
| `GET /models/sse` | The load and unload transitions as server-sent `model_status` frames, from the subscription on (the catalog is the snapshot). |
| `POST /models` | A download request: `501`, because this engine serves files that were converted on disk and makes no outbound connections. |

Use the render node of your AMD GPU (`ls -l /dev/dri/by-path/`). A Podman quadlet and a Compose file are in
[deploy/](deploy/). Other commands (`malleus --help`): `bench` (`--depth 0,8192,…,131072` grows one session and
times prefill and decode at each depth: the measurement behind [MODELS.md](MODELS.md)), `bench-bw`, `spec-exact`
(a speculative verify must be bit-identical to the serial step, or it exits non-zero), `kernel-test`, `eval`,
`golden`, `devices`, `draft-info`, `draft-vocab`, `health`, `slim` (rewrite a `.malleus` without its MLS8 prefill
copies, without the bf16 source), `inspect` (check a safetensors checkpoint against its family's tensor map, name by
name; with `--image` it also runs that family's vision tower on a picture). `bench/needle.py` checks over HTTP — so on any engine — that a unique fact planted
in the middle of a 110 000-token text is still recalled, and reports prefill and decode at that depth;
`bench/api_bench.py` does prefill and decode over HTTP from the server's own timings. The standard `malleus golden`
stops below the sparse-attention threshold; `bench/golden-long.sh` runs the 11 000-token QSA entry against its bf16
reference (the blob is generated once with `scripts/ref_layerwise.py --long 11000 --attn eager`).
`bench/idx-recall.sh` plants a unique fact in a 16K/32K token stream and checks that its block survives the
indexer selection in the question rows; `MALLEUS_RECALL_FIXTURES` selects other fixture globs (the 256K fixtures
are generated into the golden volume, not committed).

## Environment variables

Every setting is also a flag (`malleus serve --help`); the environment is what a container uses. "Off" means the
engine behaves as if the variable were unset. Defaults that depend on the model or the GPU say so.

### Server

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_MODELS_DIR` | directory (required) | Every `*.malleus` file directly under it is exposed by `/v1/models`; the id is the file stem. |
| `MALLEUS_DEFAULT_MODEL` | id (off; the only model when there is one) | Catalog id to preload and use when a request omits `model`. |
| `MALLEUS_HOST` | `0.0.0.0` | Address the API listens on. |
| `MALLEUS_PORT` | `8090` | Port for the API (Chat Completions, Messages, `/health`). |
| `MALLEUS_API_KEY` | string (off) | When set, every request needs `Authorization: Bearer <key>`. |
| `MALLEUS_CTX` | tokens, default `32768` | Maximum context. Sizes the KV state, so it is the main memory dial. |
| `MALLEUS_MAX_TOKENS` | tokens, default `8192` | Default reply budget when the request does not set `max_tokens`. |
| `MALLEUS_DEFAULT_THINKING` | `true` / `false` (default `true`) | Reasoning on/off when the request does not say. |
| `MALLEUS_VISION` | `0` turns it off | Load the vision tower when the checkpoint has one. |
| `MALLEUS_REQUEST_TIMEOUT` | seconds, default `600`, `0` disables | A request running longer than this ends the process and the unit restarts it: HIP cannot cancel a kernel already running. Give a slow model a budget larger than its worst legitimate request. |

### Loading and memory

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_PREFILL_WEIGHTS` | `none` (default: load them) | `none` skips the MLS8 prefill copies: prompt processing runs on MLS4, less memory, lower quality. |
| `MALLEUS_PREFILL_INT8_FAMILIES` | comma list of name substrings, or `none` (default: all) | Which tensors keep their MLS8 prefill copy. `none` drops all of them. |
| `MALLEUS_MEM_RESERVE_GIB` | GiB, default: estimated from the prefill chunk | Working set the memory plan counts for one forward call; set it to override the estimate. |
| `MALLEUS_HOST_RESERVE_GIB` | GiB, default `8` | RAM the memory plan leaves to the host for the page cache of a table read from disk (the n-gram table); `0` for a model without one. |
| `MALLEUS_BUF_POOL` | `0` to disable, default on | Keeps freed device buffers of up to 16 MiB (512 MiB in all) for the next allocation of the same size, so a decode step makes no `hipMalloc`/`hipFree`. |
| `MALLEUS_MEM_IGNORE` | any value except `0`/`false` | Skip the memory pre-flight and let the allocator fail instead. |
| `MALLEUS_SLOTS` | sessions, default `1` | Sessions kept resident: each in-flight request holds one and the context is split between them (`MALLEUS_CTX / slots` each). With more than one slot the decoding sessions **share one pass over the weights** one token each, which is where the aggregate rate multiplies (Qwen3.8-27B on gfx1150: 5.0 tok/s with one client, 12.8 with four), and a long prefill no longer freezes the conversations already running. Batched decode needs a flat model (no MoE, hyper-connections, PLE, indexer, KDA or MLA) and **no MTP head** — the speculative chain is per session; on any other model, or with `MALLEUS_MTP` on, the sessions take turns again and the aggregate stays at the single-request rate. `1` is the old behaviour, unchanged. The prompt snapshots (`MALLEUS_PROMPT_SNAPSHOTS`) are per session and therefore split too. |
| `MALLEUS_DECODE_INT8` | `1` (default off) | Decode on the MLS8 prefill copies of the non-expert linears: quality over speed. |
| `MALLEUS_ARGMAX_GPU` | `1` (default on) | Greedy decode computes the argmax on the device and reads back 4 bytes instead of the 1 MB logits row. Output identical; the host copy is gone, speed within noise (34.8-35.0 tok/s at 256K). `0` keeps the host argmax. |
| `MALLEUS_LMHEAD_W4` | `1` (default off) | Convert `lm_head` to MLS4 at load: half its bytes per token, a few argmax points. |
| `MALLEUS_HC_W4` | `0` (default on) | Drop the hyper-connection MLS4 weights and decode them from the MLS8 copy. |
| `MALLEUS_MTP` | `0` turns it off | Load and use the MTP head (speculative decoding). |
| `MALLEUS_MTP_INT8` | `0` (default on) | The MTP head decodes on the int8 copy: more accepted drafts, more bytes. `0` keeps it at 4 bits. |
| `MALLEUS_MTP_HIDDEN` | `post` (default: last layer output) | Feed the MTP head the post-norm hidden state. |
| `MALLEUS_MTP_CHAIN` | `pre` (default `post`) | Chained drafts start from the MTP output before its final norm. |
| `MALLEUS_DRAFT_HEAD` | `int8` (default: build it) | `int8` does not build the reduced int8 draft head from the `lm_head` rows. |
| `MALLEUS_DRAFT_VOCAB` | N (default: full vocabulary, min 1024) | Score drafts on the first N token ids only. |
| `MALLEUS_DRAFT_VOCAB_FILE` | path (off) | Token ids, one per line, that drafts are restricted to (`malleus draft-vocab` writes one). |

### Prefill and kernels

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_PREFILL_CHUNK` | tokens, default `16384` with hyper-connections else `2048` | Tokens per prompt-processing chunk; the expert weights are streamed once per chunk. The working set grows with it (about 1 MiB per token on Flash-Next), and the memory plan counts it. |
| `MALLEUS_PREFILL_QUANTUM` | tokens, default: the chunk with one slot, `min(chunk, 256)` with more | Prompt tokens processed per step when several sessions take turns: it is what a decoding session waits for, so a neighbour gets one token every `quantum / prefill_rate` seconds (measured on Qwen3.8-27B: 11 s at 2048, 1.5 s at 256, 0.9 s at 128, and 256 loses nothing on the prefill alone while 128 loses 13 %). |
| `MALLEUS_PREFILL_GEMM` | `f16` (default int8/W4A8) | Prompt GEMMs on fp16 activations instead of int8. |
| `MALLEUS_DENSE_W4A4` | `1`, `w4a8`, `acts`, `2`, `3` (default off) | `1`: int4 activations on the plain dense GEMMs. `w4a8`: also 4-bit weights for the projections whose input needs a transform (`o_proj`, `out_proj`). `acts`: int4 activations there too — faster, about −1.6 points of argmax. |
| `MALLEUS_DENSE_W4A8_ACTS` | `0` (default on) | `0` keeps those projections on int8, for an A/B of the line above. |
| `MALLEUS_MOE_W4A4` | `1` on gfx1151, `0` on gfx1150; also `glu` | Activation format of the routed experts in prefill: `1` int4, `glu` int4 on gate/up only, `0` int8. |
| `MALLEUS_EXPERT_CACHE` | GiB (default off) | Keep the routed experts on disk and cache the hot ones in per-layer slot pools; decode reads slot indices, prefill streams the whole layer (bit-identical, much slower prefill). Issue #14. |
| `MALLEUS_MOE_Y_HALF` | `1` (default off, f32) | Write the MoE intermediate `y` in fp16: half its bytes, measured +3,5 % of prefill. Ignored for checkpoints with int8 expert copies. |
| `MALLEUS_MOE_BM128` | rows per expert, default `48` | From this average rows-per-expert the grouped MoE switches to 128-row tiles. |
| `MALLEUS_MOE_SPLIT` | `0` (default on) | `0` drops the second, half-height tile list the MoE uses for the remainders of each expert. |
| `MALLEUS_W4_MAX_T` | tokens, default `0` | Inputs up to this many tokens use the 4-bit weights (W4A8). `0` never. |
| `MALLEUS_HC_FUSED` | `0` (default on) | Disable the fused hyper-connection kernels used by single-token decode. |
| `MALLEUS_MLA_PREFILL` | `1` (default off) | Opt-in tiled MLA prefill. |
| `MALLEUS_DN_SCAN` | any value (default off) | Debug: DeltaNet falls back to the token-by-token scan instead of the chunked one. |

### Attention and indexer

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_ATTN_SEL` | `0` (default on) | Disable the sparse block selection in decode and verify. |
| `MALLEUS_ATTN_GATHER` | `0` (default on) | Disable the sparse gather in prefill: attention runs dense. `1` forces the per-query gather instead of the architecture default. |
| `MALLEUS_GATHER_Q1` | `1` on gfx1151, `0` on gfx1150 | Sparse gather shape: `1` is one query per block (128 threads, four waves), `0` the four-query union. On gfx1151 `1` measured +8…+11 % of prefill at 64–98K tokens; gfx1150 keeps the union, which measured 6–9 % faster there at 12/24K. |
| `MALLEUS_GATHER_MIN` | multiple of the attention budget, default `5` | Context length from which prefill switches to the gather. Lower is faster and less exact; `3` measured +2.2 % of prefill at 24k tokens. |
| `MALLEUS_IDX_SCORE_TILE` | `1` on gfx1151 with a 4-head indexer, `0` otherwise | Score a 16-query tile while walking 64-key tiles; `0` uses the 4-query kernel. Scores are bit-identical; measured 7.95 → 5.27 s at 98K tokens. |
| `MALLEUS_IDX_SELECT` | `12` on gfx1151 from 12 288 blocks, `0` otherwise | `12` = three radix passes of 12/12/8 bits (4096 bins), `0` = four passes of 8 bits. Same bitset; the 4096-bin histogram wins only on long rows. |
| `MALLEUS_IDX_KT` | key tiles per score block, default `8` in prefill and `1` in decode | Tuning of the tiled score kernel; 8 reuses the q tile, 1 keeps the key parallelism of decode. |
| `MALLEUS_IDX_GROUP` | `4` on gfx1151 when the 12/12/8 select is active, `1` otherwise | Score and select one representative query per group of N, copy its selection to the group and mask the few future keys in the gather. 4 makes the 32K-256K curve flat (`1 453` to `1 422` tok/s at 256K); `1` restores per-query selection. Prefill only: decode/verify (`nq = 1..6`) always fall back to `1`. |
| `MALLEUS_IDX_DECODE_REUSE` | `1` (default on) | Decode: reuse the previous token's block selection until a new 4-token block completes, then recompute. At 256K serial decode 31,0 → 34,2 tok/s; at 11K teacher-forced the argmax/KL are unchanged. `0` recomputes every token. |
| `MALLEUS_IDX_STATS` | any value (default off) | Debug: print the density of the selections and of their unions; `all` prints every layer. |
| `MALLEUS_IDX_DUMP_SEL` | path (default off) | Debug: write the selection bitsets of the last sub-batch, one file per indexer layer, for `bench/idx-recall.sh`. |

### Speculative decoding and the prompt cache

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_SPEC_DRAFTS` | drafts per round, default `5` (capped at `GEMV_MAX_T - 1`) | Drafts verified per round; `0` turns speculation off. |
| `MALLEUS_DRAFT_PMIN` | f32, default `0.8` | A chain of drafts goes on while the product of their probabilities stays above this. |
| `MALLEUS_LOOKUP` | `0` (default on) | Disable the prompt-lookup drafter (copies from the request's own text). |
| `MALLEUS_LOOKUP_NGRAM` | tokens, default `3` | Length of the n-gram the prompt lookup matches. |
| `MALLEUS_LOOKUP_NORM` | `1` (default off) | Also draft from the original text of the recent messages, with line numbers and JSON escapes removed. Measured no gain at 5 drafts; it is the substrate for deeper rounds. |
| `MALLEUS_PROMPT_SNAPSHOTS` | N, default `8` | Prompt states kept for prefix reuse; `0` disables the snapshot cache. |
| `MALLEUS_WARMUP` | `0` (default on) | Skip the warm-up request at startup (it costs a chunk of prefill, it saves the first request a lot). |

### N-gram table, rope, build, debug

| Variable | Values (default) | What it does |
|---|---|---|
| `MALLEUS_NGRAM_THREADS` | threads, default `256` | Threads that read the n-gram table for a prompt; a decode step (up to 16 tokens) reads its rows on the calling thread. |
| `MALLEUS_NGRAM_WARM` | `1` (default off) | Read the whole n-gram table at startup to warm the page cache. |
| `MALLEUS_ROPE_YARN` | factor > 1, default `1` (off) | YaRN scaling to go past the native 262144 positions (the container uses `2` at 512k). |
| `MALLEUS_ROPE_ORIG` | positions, default `262144` | Native rope length the YaRN ramp is built on. |
| `MALLEUS_GPU_TARGETS` | build time, default `gfx1150;gfx1151` | Architectures the kernels are compiled for. |
| `MALLEUS_RPATH` | build time, any value | Link the binary with `rpath=$ORIGIN/../lib`, which the container image needs. |
| `MALLEUS_DUMP_DIR` | directory (off) | Debug: write layer buffers as `{tag}.f32` for a layer-by-layer comparison. |
| `MALLEUS_EXPERT_CACHE_VERIFY` | any value (default off) | Debug: after every expert fetch, read the slot back and compare it with the file. |
| `MALLEUS_TRACE_ROUTE` | file path (off) | Debug: append the experts the router picked (`layer t row id:weight ...`), the input of `scripts/expert_cache.py`. Synchronizes once per layer, so a trace run is slower. |
| `MALLEUS_DEBUG_SNAPS` | any value (default off) | Debug: log the prompt-snapshot bookkeeping. |
| `MALLEUS_TIMING` | any value (default off) | Debug: print timings of the n-gram embedding step. |

## Documentation and contributing

- **Models and their measured performance** — the files we carry, their size on the device and the tokens/s per
  prompt length, engine against engine — are in [MODELS.md](MODELS.md).
- **Everything in the repository is in English** — code, comments, commit messages, issues, these docs. Engineering
  notes and plans are not kept here: what must not be lost goes into the issue that motivated it.
- **How to contribute** — build, workflow and the rules for changes — is in [AGENTS.md](AGENTS.md) and
  [CONTRIBUTING.md](CONTRIBUTING.md): every change goes through an issue, a dedicated branch and a pull request.

## License

**Apache License 2.0** ([LICENSE](LICENSE)): use it, modify it, ship it in your product, closed or open, as long as
you keep the [NOTICE](NOTICE) and the attribution notices — no CLA, no copyleft, and a patent grant on top. The code
is not the moat: the tuning and the operations are.

| | |
|---|---|
| Code | Apache License 2.0 — [LICENSE](LICENSE) |
| Attribution and third parties | [NOTICE](NOTICE), [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) |
| Names and logo | not licensed with the code — [TRADEMARKS.md](TRADEMARKS.md) |
| Contributions | Apache-2.0, signed off (`git commit -s`, [DCO](DCO)) — [CONTRIBUTING.md](CONTRIBUTING.md) |
| Model weights | keep the license of their original model |
| Redistributed in the image | ROCm runtime libraries, Debian base — licenses in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) |

The software is provided **"as is"**, without warranty of any kind, express or implied, and without liability for any
damages arising from its use — see the [Apache-2.0](LICENSE) *Disclaimer of Warranty* and *Limitation of Liability*
(sections 7 and 8). The converted model weights keep the license of their original models and are not ours to warrant.
