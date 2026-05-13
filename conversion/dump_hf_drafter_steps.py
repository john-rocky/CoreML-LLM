#!/usr/bin/env python3
"""Dump HF drafter proposals + target argmax per step on the capitals prompt.

Goal: ground-truth comparison against our Swift `[MtpDbg]` log so we can
see EXACTLY where the Mac/CoreML port diverges.

For each drafter call, prints:
  - prefill position (= input_ids.shape[1] - 1)
  - constant position_ids HF passes to drafter
  - drafter proposal token IDs (K of them)
  - target's verify argmax per slot
  - matchCount

Usage:
    .mtp_venv/bin/python conversion/dump_hf_drafter_steps.py [--prompt-id capitals]
"""
from __future__ import annotations
import argparse
import sys
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import candidate_generator as _cg

# Captured per-step logs.
_steps: list[dict] = []
_orig_get = _cg.SinglePositionMultiTokenCandidateGenerator.get_candidates


def _patched_get(self, input_ids, model_kwargs, model_outputs,
                 is_first_iteration, n_last_matches, **kwargs):
    """Wrap get_candidates to capture the proposed tokens + position."""
    candidate_input_ids, candidate_logits = _orig_get(
        self, input_ids, model_kwargs, model_outputs, is_first_iteration,
        n_last_matches, **kwargs)
    if not is_first_iteration:
        proposed = candidate_input_ids[0, input_ids.shape[1]:].tolist()
        _steps.append({
            "round": len(_steps) + 1,
            "input_len": int(input_ids.shape[1]),
            "position_ids": int(input_ids.shape[1] - 1),  # HF constant pos
            "proposed": proposed,
            "n_last_matches": int(n_last_matches),
        })
    return candidate_input_ids, candidate_logits


_cg.SinglePositionMultiTokenCandidateGenerator.get_candidates = _patched_get


PROMPTS = {
    "capitals": "The capital of France is Paris. The capital of Germany is Berlin. "
                "The capital of Italy is",
    "essay": "Write a 250-word essay about the history of speculative decoding "
             "for language models. Cover EAGLE, Medusa, and MTP.",
    "translate": (
        "Translate the following 10 sentences from English to French.\n"
        "1. Hello, how are you today?\n2. The cat is on the table.\n"
        "3. I would like a cup of coffee.\n4. Where is the train station?\n"
        "5. We are going to the beach tomorrow.\n6. She reads books every "
        "evening.\n7. They live in a small village.\n8. The weather is "
        "very nice.\n9. Could you help me, please?\n10. I love this song."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-id", choices=list(PROMPTS), default="capitals")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = ap.parse_args()

    print(f"[dump] prompt={args.prompt_id} max_tokens={args.max_tokens} device={args.device}")
    dtype = torch.float16 if args.device == "mps" else torch.float32
    target = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=dtype,
        low_cpu_mem_usage=True).to(args.device).eval()
    drafter = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it-assistant", dtype=dtype,
        low_cpu_mem_usage=True).to(args.device).eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

    prompt = PROMPTS[args.prompt_id]
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(args.device)
    print(f"[dump] prompt='{prompt}'")
    print(f"[dump] prompt_tokens={inputs.input_ids.shape[1]}")
    print(f"[dump] prompt token_ids[-10:]={inputs.input_ids[0, -10:].tolist()}")

    _steps.clear()
    with torch.no_grad():
        out = target.generate(
            **inputs, max_new_tokens=args.max_tokens, do_sample=False,
            assistant_model=drafter, pad_token_id=tok.eos_token_id)

    n_new = out.shape[1] - inputs.input_ids.shape[1]
    new_ids = out[0, inputs.input_ids.shape[1]:].tolist()
    print(f"\n[dump] generated {n_new} tokens: {new_ids}")
    print(f"[dump] decoded: '{tok.decode(new_ids)}'")
    print(f"[dump] {len(_steps)} drafter rounds:")
    for s in _steps:
        print(f"  round={s['round']:>2}  input_len={s['input_len']:>3}  "
              f"position_ids={s['position_ids']:>3}  "
              f"n_last_matches={s['n_last_matches']}  "
              f"proposed={s['proposed']}")


if __name__ == "__main__":
    main()
