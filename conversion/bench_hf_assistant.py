#!/usr/bin/env python3
"""HF runtime baseline + assistant bench for Gemma 4 MTP.

Runs `target.generate()` with and without `assistant_model=drafter` on a
fixed set of prompts, captures per-step accept counts, and reports
tok/s + accept distribution. Establishes a ceiling for what the
official drafter can deliver on this machine, independent of our
CoreML port.

Usage:
    .mtp_venv/bin/python conversion/bench_hf_assistant.py [--device mps|cpu] [--max-tokens 96]
"""
from __future__ import annotations
import argparse
import sys
import time
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Patch SinglePositionMultiTokenCandidateGenerator to record per-call accept counts.
# We intercept `update_candidate_strategy` since it receives `num_matches`.
from transformers.generation import candidate_generator as _cg

_per_call_log: list[tuple[int, int]] = []  # (num_matches, num_proposed)
_orig_update = _cg.SinglePositionMultiTokenCandidateGenerator.update_candidate_strategy


def _patched_update(self, input_ids, scores, num_matches):
    # Record this round's matches before delegating to the base impl.
    n_proposed = int(self.num_assistant_tokens)
    _per_call_log.append((int(num_matches), n_proposed))
    return _orig_update(self, input_ids, scores, num_matches)


_cg.SinglePositionMultiTokenCandidateGenerator.update_candidate_strategy = _patched_update


PROMPTS: List[tuple[str, str]] = [
    ("capitals",
     "The capital of France is Paris. The capital of Germany is Berlin. "
     "The capital of Italy is"),
    ("essay",
     "Write a 250-word essay about the history of speculative decoding "
     "for language models. Cover EAGLE, Medusa, and MTP."),
    ("code",
     "Complete this Python function:\n```python\ndef fibonacci(n):\n    "
     "if n < 2:\n        return n\n    a, b = 0, 1\n    for i in range"),
    ("qa",
     "Q: What is the boiling point of water at sea level?\nA:"),
]


def _run_target_only(target, tok, prompt: str, max_new_tokens: int, device: str) -> tuple[float, int]:
    inputs = tok(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    n = out.shape[1] - inputs.input_ids.shape[1]
    return n / dt, n


def _run_assisted(target, drafter, tok, prompt: str, max_new_tokens: int,
                  device: str) -> tuple[float, int, list[tuple[int, int]]]:
    _per_call_log.clear()
    inputs = tok(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            assistant_model=drafter, pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    n = out.shape[1] - inputs.input_ids.shape[1]
    return n / dt, n, list(_per_call_log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--target", default="google/gemma-4-E2B-it")
    ap.add_argument("--assistant", default="google/gemma-4-E2B-it-assistant")
    args = ap.parse_args()

    device = args.device
    print(f"[bench] device={device}  max_tokens={args.max_tokens}")
    print(f"[bench] target={args.target}")
    print(f"[bench] assistant={args.assistant}")

    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"[bench] dtype={dtype}")
    print("[bench] loading target ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    print("[bench] loading assistant ...", flush=True)
    drafter = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.target)
    print("[bench] models loaded", flush=True)

    print(f"\n{'name':<10} {'baseline tok/s':>16} {'assisted tok/s':>16} "
          f"{'speedup':>8} {'avg accept':>11} {'rounds':>7}")
    print("-" * 80)

    for name, prompt in PROMPTS:
        try:
            base_tps, base_n = _run_target_only(
                target, tok, prompt, args.max_tokens, device)
        except Exception as e:
            print(f"{name:<10} baseline FAILED: {e}")
            continue
        try:
            asst_tps, asst_n, log = _run_assisted(
                target, drafter, tok, prompt, args.max_tokens, device)
        except Exception as e:
            print(f"{name:<10} assisted FAILED: {e}")
            continue

        if log:
            total_match = sum(m for m, _ in log)
            total_proposed = sum(p for _, p in log)
            avg_accept = total_match / total_proposed if total_proposed else 0.0
        else:
            avg_accept = float("nan")
        print(f"{name:<10} {base_tps:>16.2f} {asst_tps:>16.2f} "
              f"{asst_tps/base_tps:>7.2f}x {avg_accept:>10.2%} {len(log):>7d}")

    print("\nNotes:")
    print("- accept = sum(num_matches) / sum(num_proposed) across all drafter calls.")
    print("- baseline is plain target.generate; assisted is target.generate + drafter.")


if __name__ == "__main__":
    main()
