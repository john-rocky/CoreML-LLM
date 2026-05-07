#!/usr/bin/env python3
"""HF transformers MTP bench — gold-standard reference for what vendor's
universal drafter actually delivers on a non-CoreML stack.

Runs Gemma 4 E2B (bf16) + assistant (bf16) on MPS via transformers's
SinglePositionMultiTokenCandidateGenerator (auto-selected when
`assistant_model` is the assistant repo). Measures tok/s with and
without MTP on the same prompts our CoreML bench uses, so we can
compare apples-to-apples and decide whether our gap is INT4-specific
or implementation-specific.
"""
from __future__ import annotations
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "fib_recursion":
        "find the nth Fibonacci number using recursion.",
    "repeat":
        'Say "yes" 30 times.',
    "stack":
        "Explain the difference between a stack and a queue in 4 bullet "
        "points, with a Python code example showing each.",
    "qsort":
        "def quicksort(arr):",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="google/gemma-4-E2B-it")
    ap.add_argument("--assistant", default="google/gemma-4-E2B-it-assistant")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompts", nargs="*", default=list(PROMPTS.keys()))
    args = ap.parse_args()

    print(f"Loading target {args.target} (bf16) on {args.device} ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16).eval().to(args.device)
    print(f"Loading assistant {args.assistant} (bf16) on {args.device} ...")
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=torch.bfloat16).eval().to(args.device)
    tok = AutoTokenizer.from_pretrained(args.target)

    print(f"\nMax new tokens: {args.max_new_tokens}")
    print(f"\n{'Prompt':<14} {'mode':<10} {'tok/s':>8} {'gen_tokens':>11}")
    print("-" * 50)

    for pname in args.prompts:
        prompt = PROMPTS[pname]
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(args.device)

        for mode, kwargs in [
            ("no_MTP", {}),
            ("MTP",    {"assistant_model": assistant}),
        ]:
            # warm-up (cuts compile/cache cost)
            _ = target.generate(**ids, max_new_tokens=8, do_sample=False, **kwargs)
            torch.mps.synchronize() if args.device == "mps" else None

            t0 = time.perf_counter()
            out = target.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, **kwargs)
            torch.mps.synchronize() if args.device == "mps" else None
            dt = time.perf_counter() - t0
            n_new = out.shape[1] - ids["input_ids"].shape[1]
            print(f"{pname:<14} {mode:<10} {n_new/dt:>8.2f} {n_new:>11}")
        print()


if __name__ == "__main__":
    main()
