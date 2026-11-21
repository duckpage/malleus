#!/usr/bin/env python3
"""Generate golden references with transformers (offline tool, never shipped).

Writes small files (prompts, token ids, greedy outputs, summaries) to golden/<name>/ and
large tensors (logits, vision intermediates) to --blob-dir.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

TEXT_PROMPTS = [
    ("en-fact", "What is the capital of France? Answer in one word."),
    ("en-explain", "Explain in two sentences why the sky is blue."),
    ("en-list", "List three prime numbers greater than 100."),
    ("en-story", "Write the opening sentence of a mystery novel set in a lighthouse."),
    ("en-math", "If a train travels 120 km in 1.5 hours, what is its average speed in km/h?"),
    ("en-reason", "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?"),
    ("it-fact", "Qual è il fiume più lungo d'Italia?"),
    ("it-explain", "Spiega in due frasi cos'è la fotosintesi."),
    ("it-translate", "Traduci in inglese: 'Domani andiamo al mare se non piove.'"),
    ("it-email", "Scrivi una breve email per spostare una riunione da lunedì a martedì."),
    ("it-recipe", "Quali sono gli ingredienti della carbonara tradizionale?"),
    ("code-py", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("code-rs", "Write a Rust function that reverses a string slice and returns a String."),
    ("code-sql", "Write a SQL query that returns the 5 most recent orders from table orders(id, created_at)."),
    ("code-bug", "Find the bug:\n\ndef mean(xs):\n    return sum(xs) / len(xs) + 1\n"),
    ("code-explain", "What does `git rebase -i HEAD~3` do?"),
    ("code-regex", "Write a regex that matches an Italian fiscal code (codice fiscale)."),
    ("mixed-json", "Return a JSON object with keys name and age for a person named Marco aged 42."),
    ("long-context", "Summarize: " + " ".join(
        ["The committee met on Monday to discuss the budget for the new library."] * 12)),
    ("raw-continue", None),
]
RAW_TEXT = "The quick brown fox jumps over the lazy dog. In 1969, humans first"

GREEDY_IDS = ["en-explain", "it-explain", "code-py", "raw-continue"]


def make_images(out_dir: Path):
    """Synthetic images (no downloads): shapes, text, diagrams, gradients, odd sizes and aspect ratios."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()
    imgs = {}

    im = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 260, 260), fill="red")
    d.rectangle((280, 280, 460, 460), fill="blue")
    d.polygon([(380, 40), (480, 220), (280, 220)], fill="green")
    imgs["shapes"] = im

    im = Image.new("RGB", (800, 240), (250, 250, 240))
    d = ImageDraw.Draw(im)
    d.text((20, 40), "Malleus 2026", fill="black", font=font)
    d.text((20, 120), "Totale: 42,50 EUR", fill="darkred", font=font)
    imgs["text"] = im

    im = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(im)
    for i, (x, lbl) in enumerate([(40, "A"), (260, "B"), (480, "C")]):
        d.rectangle((x, 200, x + 120, 280), outline="black", width=3)
        d.text((x + 50, 225), lbl, fill="black", font=font)
        if i < 2:
            d.line((x + 120, 240, x + 220, 240), fill="black", width=3)
            d.polygon([(x + 220, 240), (x + 205, 230), (x + 205, 250)], fill="black")
    imgs["diagram"] = im

    im = Image.new("RGB", (1024, 256))
    px = im.load()
    for y in range(256):
        for x in range(1024):
            px[x, y] = (x * 255 // 1023, y, 128)
    imgs["gradient-wide"] = im

    im = Image.new("RGB", (200, 900), "white")
    d = ImageDraw.Draw(im)
    for i in range(9):
        d.rectangle((20, 20 + i * 98, 180, 100 + i * 98), fill=(i * 28, 100, 255 - i * 28))
    imgs["tall-bars"] = im

    im = Image.new("RGB", (96, 96), "yellow")
    ImageDraw.Draw(im).ellipse((20, 20, 76, 76), fill="black")
    imgs["tiny"] = im

    rng = np.random.default_rng(1234)
    arr = (rng.random((600, 900, 3)) * 60).astype(np.uint8)
    arr[:300] += np.array([90, 140, 200], dtype=np.uint8)
    arr[300:] += np.array([60, 120, 40], dtype=np.uint8)
    imgs["landscape"] = Image.fromarray(arr)

    im = Image.new("RGB", (1280, 720), (30, 30, 30))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1280, 40), fill=(60, 60, 60))
    d.text((10, 6), "terminal - bash", fill="white", font=font)
    for i, line in enumerate(["$ cargo build --release", "   Compiling malleus v0.1.0", "error[E0308]: mismatched types"]):
        d.text((20, 70 + i * 40), line, fill="lightgreen" if i < 2 else "red", font=font)
    imgs["screenshot"] = im

    im = Image.new("RGB", (480, 480), "white")
    d = ImageDraw.Draw(im)
    vals = [3, 7, 2, 9, 5]
    for i, v in enumerate(vals):
        d.rectangle((40 + i * 85, 440 - v * 40, 100 + i * 85, 440), fill="steelblue")
    d.line((30, 440, 460, 440), fill="black", width=2)
    imgs["bar-chart"] = im

    im = Image.new("RGB", (333, 517), "white")
    d = ImageDraw.Draw(im)
    for i in range(0, 333, 37):
        d.line((i, 0, i, 517), fill="gray")
    for j in range(0, 517, 37):
        d.line((0, j, 333, j), fill="gray")
    d.text((100, 240), "7 x 6 = ?", fill="black", font=font)
    imgs["odd-size-grid"] = im

    for k, v in imgs.items():
        v.save(out_dir / f"{k}.png")
    return imgs


IMAGE_QUESTIONS = {
    "shapes": "Describe the shapes and their colors.",
    "text": "What text is written in this image?",
    "diagram": "Describe this diagram.",
    "gradient-wide": "What do you see?",
    "tall-bars": "How many colored bars are there?",
    "tiny": "What is in this image?",
    "landscape": "Describe this image briefly.",
    "screenshot": "What error is shown in this terminal?",
    "bar-chart": "Which bar is the tallest?",
    "odd-size-grid": "Solve the expression in the image.",
}


def load_on_gpu(model_dir: str):
    """Builds the model directly on the GPU and streams shards there (the LXC host has only 32 GB of RAM)."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(model_dir)
    if cfg.model_type.startswith("inkling2"):
        return load_inkling2_on_gpu(model_dir)
    with torch.device("cuda"):
        model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16)
    params = dict(model.named_parameters())
    loaded = set()
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cuda:0") as st:
            for k in st.keys():
                if k.startswith("mtp."):
                    continue
                if k not in params:
                    raise KeyError(f"unexpected checkpoint tensor {k}")
                with torch.no_grad():
                    params[k].copy_(st.get_tensor(k))
                loaded.add(k)
    missing = set(params) - loaded
    if missing:
        raise KeyError(f"parameters not in checkpoint: {sorted(missing)[:5]}")
    model.eval()
    return model


def load_inkling2_on_gpu(model_dir: str):
    """Inkling-2 ships quantized expert weights (NVFP4/MXFP8) that `from_pretrained` dequantizes on the host.

    Reading the shards on the GPU keeps the dequantization on the device: the host stays at ~32 GB while the
    expanded bf16 weights (~40 GiB) land in GTT. This is the memory-safe path for the LXC host.
    """
    import transformers.modeling_utils as mu
    from transformers import AutoModelForImageTextToText

    orig = mu.safe_open

    def safe_open_gpu(*a, **kw):
        kw["device"] = "cuda:0"
        return orig(*a, **kw)

    mu.safe_open = safe_open_gpu
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="cuda"
        )
    finally:
        mu.safe_open = orig
    model.eval()
    return model


def save_f32(path: Path, t: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    t.detach().float().cpu().contiguous().numpy().tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3.8-27B")
    ap.add_argument("--out", default="golden/qwen38-27b")
    ap.add_argument("--blob-dir", default="/golden/qwen38-27b")
    ap.add_argument("--greedy-tokens", type=int, default=256)
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--reasoning-effort", default=None,
                    help="template reasoning effort for models that use it (inkling2: none|low|...)")
    ap.add_argument("--loop-passes", type=int, default=None,
                    help="inkling2: override loop_num_passes (1 = the base graph, for debugging)")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    import transformers

    out, blob = Path(args.out), Path(args.blob_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(args.model)
    t0 = time.time()
    model = load_on_gpu(args.model)
    if args.loop_passes is not None and hasattr(model.config, "text_config"):
        model.config.text_config.loop_num_passes = args.loop_passes
        model.model.language_model.config.loop_num_passes = args.loop_passes
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    meta = {"transformers": transformers.__version__, "torch": torch.__version__, "dtype": "bfloat16", "text": [], "images": []}
    with torch.no_grad():
        for pid, text in TEXT_PROMPTS:
            if text is None:
                ids = proc.tokenizer(RAW_TEXT, return_tensors="pt").input_ids
            else:
                msgs = [{"role": "user", "content": text}]
                ids = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                               reasoning_effort=args.reasoning_effort,
                                               tokenize=True, return_tensors="pt", return_dict=True)["input_ids"]
            ids = ids.to("cuda")
            logits = model(input_ids=ids).logits[0].float()
            save_f32(blob / "text" / pid / "logits_last.f32", logits[-1])
            lp = torch.log_softmax(logits, -1)
            top = torch.topk(lp[-1], 10)
            entry = {"id": pid, "input_ids": ids[0].tolist(), "argmax": logits.argmax(-1).tolist(),
                     "top10_last": {"ids": top.indices.tolist(), "logprobs": top.values.tolist()}}
            if pid in GREEDY_IDS and not args.skip_greedy:
                t1 = time.time()
                g = model.generate(input_ids=ids, max_new_tokens=args.greedy_tokens, do_sample=False,
                                   temperature=None, top_p=None, top_k=None)
                entry["greedy"] = g[0, ids.shape[1]:].tolist()
                entry["greedy_text"] = proc.tokenizer.decode(entry["greedy"])
                print(f"  greedy {pid}: {time.time() - t1:.0f}s", flush=True)
            meta["text"].append(entry)
            print(pid, ids.shape[1], proc.tokenizer.decode(entry["argmax"][-1:]), flush=True)
            json.dump(meta, open(out / "golden.json", "w"), indent=1)

        if not args.skip_images:
            imgs = make_images(out / "images")
            for name, im in imgs.items():
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": IMAGE_QUESTIONS[name]}]}]
                prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                                  reasoning_effort=args.reasoning_effort, tokenize=False)
                inp = proc(text=[prompt], images=[im], return_tensors="pt").to("cuda")
                pv = inp["pixel_values"]
                grid = inp["image_grid_thw"]
                vis = model.model.visual(pv.to(torch.bfloat16), grid_thw=grid)
                d = blob / "images" / name
                save_f32(d / "pixel_values.f32", pv)
                save_f32(d / "vit_out.f32", vis.last_hidden_state)
                save_f32(d / "merger_out.f32", vis.pooler_output)
                logits = model(**inp).logits[0].float()
                save_f32(d / "logits_last.f32", logits[-1])
                g = model.generate(**inp, max_new_tokens=48, do_sample=False, temperature=None, top_p=None, top_k=None)
                n_in = inp["input_ids"].shape[1]
                entry = {"id": name, "size": list(im.size), "input_ids": inp["input_ids"][0].tolist(),
                         "grid_thw": grid[0].tolist(), "pixel_shape": list(pv.shape),
                         "argmax_last": int(logits[-1].argmax()),
                         "greedy": g[0, n_in:].tolist(), "greedy_text": proc.tokenizer.decode(g[0, n_in:])}
                meta["images"].append(entry)
                print(name, grid[0].tolist(), entry["greedy_text"][:80].replace("\n", " "), flush=True)
                json.dump(meta, open(out / "golden.json", "w"), indent=1)

    json.dump(meta, open(out / "golden.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
