#!/usr/bin/env python3
"""Probe Gemma 4 E2B residual-stream magnitudes per layer to determine whether
ANE runs need anemll-style FP16 scaling.

We load the HF text decoder, run a small set of representative prompts in two
dtypes:

* bf16  — training dtype, no overflow → ground-truth residual magnitudes
* fp16  — what ANE actually sees in production → check for overflow / NaN

For each layer we record max-abs, mean-abs, and NaN flag. Decision rule:

* bf16 max-abs < 30_000               → no scaling needed
* 30_000 ≤ bf16 max-abs < 60_000     → α ≈ 0.7 (margin for fp16)
* bf16 max-abs ≥ 60_000 or fp16 NaN  → α ≈ 0.5 (anemll's E2B hypothesis)

Output: markdown summary printed to stdout.

Usage:
    python conversion/probe_residual_overflow.py \\
        --hf-dir output/gemma4-e2b/hf_model \\
        --num-prompts 8 --max-tokens 256
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A small set of representative prompts: short factual, code, multilingual,
# long-context-style. We are NOT measuring task accuracy — only residual
# magnitudes — so 8-16 prompts is plenty.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "日本で一番高い山は",
    "Quantum mechanics describes the behavior of particles at",
    "1, 1, 2, 3, 5, 8, 13, 21, 34,",
    "The mitochondria is the powerhouse of the cell. This is because",
    # Long context to stress later layers / global attention more:
    ("In a small village nestled between two mountains, there lived a baker "
     "who was known throughout the region for his exceptional sourdough. "
     "Every morning before dawn, he would wake up to tend his starter, "
     "feeding it carefully with rye flour and filtered water. The villagers "
     "would line up before sunrise to receive their daily loaves. One day,"),
    ("System: You are a helpful assistant.\nUser: Explain how a transistor "
     "works in simple terms.\nAssistant:"),
]


def _stats(t: torch.Tensor) -> Dict[str, float]:
    finite = torch.isfinite(t)
    if not finite.any():
        return {"max_abs": math.inf, "mean_abs": math.inf, "nan_frac": 1.0}
    finite_t = t[finite]
    return {
        "max_abs": float(finite_t.abs().max()),
        "mean_abs": float(finite_t.abs().mean()),
        "nan_frac": float((~finite).float().mean()),
    }


def run_probe(hf_dir: str, dtype: torch.dtype, prompts: List[str],
              max_tokens: int, device: str) -> List[Dict]:
    print(f"\n[probe] dtype={dtype}  device={device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir, dtype=dtype, low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    print(f"[probe] loaded in {time.time()-t0:.1f}s", flush=True)

    n_layers = model.config.text_config.num_hidden_layers
    # hidden_states tuple is len = n_layers + 1 (includes embedding output at idx 0)
    per_layer: List[List[Dict[str, float]]] = [[] for _ in range(n_layers + 1)]

    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=max_tokens).input_ids.to(device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True,
                        return_dict=True, use_cache=False)
        hs = out.hidden_states  # tuple of (B, T, H)
        for li, h in enumerate(hs):
            per_layer[li].append(_stats(h.detach().cpu().float()))
        del out, hs
        print(f"  prompt {i+1}/{len(prompts)}  T={ids.shape[1]}  "
              f"final_max_abs={per_layer[-1][-1]['max_abs']:.1f}", flush=True)

    rows = []
    for li, samples in enumerate(per_layer):
        max_abs = max(s["max_abs"] for s in samples)
        mean_abs = sum(s["mean_abs"] for s in samples) / len(samples)
        nan = any(s["nan_frac"] > 0 for s in samples)
        rows.append({
            "layer": li,
            "name": "embed_out" if li == 0 else f"after_L{li-1}",
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "nan": nan,
        })
    return rows


def decide_alpha(bf16_rows: List[Dict], fp16_rows: List[Dict]) -> str:
    bf16_max = max(r["max_abs"] for r in bf16_rows)
    fp16_nan = any(r["nan"] for r in fp16_rows)
    fp16_max = max(r["max_abs"] for r in fp16_rows if not math.isinf(r["max_abs"]))

    if fp16_nan or bf16_max >= 60_000:
        return f"α ≈ 0.5  (bf16_max={bf16_max:.0f}, fp16_nan={fp16_nan})"
    if bf16_max >= 30_000:
        return f"α ≈ 0.7  (bf16_max={bf16_max:.0f}, fp16_max={fp16_max:.0f})"
    return f"NO SCALING NEEDED  (bf16_max={bf16_max:.0f}, fp16_max={fp16_max:.0f})"


def print_table(label: str, rows: List[Dict]) -> None:
    print(f"\n## {label}")
    print(f"{'layer':>6}  {'name':<14} {'max_abs':>10}  {'mean_abs':>10}  {'NaN'}")
    for r in rows:
        nan_mark = "‼️" if r["nan"] else ""
        ma = "inf" if math.isinf(r["max_abs"]) else f"{r['max_abs']:.1f}"
        ave = "inf" if math.isinf(r["mean_abs"]) else f"{r['mean_abs']:.2f}"
        print(f"{r['layer']:>6}  {r['name']:<14} {ma:>10}  {ave:>10}  {nan_mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="output/gemma4-e2b/hf_model")
    ap.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--device", default="cpu",
                    help="cpu or mps. (CPU is safer for fp16 overflow detection.)")
    ap.add_argument("--skip-fp16", action="store_true",
                    help="Skip fp16 pass (bf16 ground truth only)")
    args = ap.parse_args()

    if not os.path.isdir(args.hf_dir):
        sys.exit(f"hf_dir not found: {args.hf_dir}")

    prompts = PROMPTS[:args.num_prompts]

    bf16_rows = run_probe(args.hf_dir, torch.bfloat16, prompts,
                          args.max_tokens, args.device)
    print_table("bf16 (ground truth)", bf16_rows)

    if args.skip_fp16:
        return

    fp16_rows = run_probe(args.hf_dir, torch.float16, prompts,
                          args.max_tokens, args.device)
    print_table("fp16 (production proxy)", fp16_rows)

    print("\n## Decision\n")
    print(decide_alpha(bf16_rows, fp16_rows))


if __name__ == "__main__":
    main()
