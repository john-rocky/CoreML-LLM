#!/usr/bin/env python3
"""HF per-round detail: log every (num_matches, num_proposed) tuple.

This tells us how HF's adaptive `num_assistant_tokens` walks during a
chat-templated long-output run, and whether our static K_USE=2 is
fighting against HF's adaptive optimum.
"""
from __future__ import annotations
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import candidate_generator as _cg

_log: list[tuple[int, int]] = []
_orig = _cg.SinglePositionMultiTokenCandidateGenerator.update_candidate_strategy


def _patched(self, input_ids, scores, num_matches):
    _log.append((int(num_matches), int(self.num_assistant_tokens)))
    return _orig(self, input_ids, scores, num_matches)


_cg.SinglePositionMultiTokenCandidateGenerator.update_candidate_strategy = _patched


def main():
    target = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=torch.float32,
        low_cpu_mem_usage=True).eval()
    drafter = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it-assistant", dtype=torch.float32,
        low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    prompt = ("Write a complete Python module that implements a doubly-linked list with "
              "insert, delete, search, reverse, and to_list methods. Include type hints, "
              "docstrings on every method, and a test suite using unittest with at least "
              "10 test cases.")
    msgs = [{"role": "user", "content": prompt}]
    chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    chat_ids = tok(chat_text, return_tensors="pt", add_special_tokens=False).input_ids
    print(f"chat_prompt_len = {chat_ids.shape[1]}")

    with torch.no_grad():
        out = target.generate(input_ids=chat_ids, max_new_tokens=384,
                               do_sample=False, pad_token_id=tok.eos_token_id,
                               assistant_model=drafter)
    n = out.shape[1] - chat_ids.shape[1]
    print(f"generated {n} tokens; rounds={len(_log)}")

    # Distribution of (matches, proposed)
    print("\nFirst 30 rounds:")
    for i, (m, p) in enumerate(_log[:30], 1):
        print(f"  round {i:>3}: matched {m} of {p}")

    print("\nDistribution of num_proposed per round:")
    cnt = Counter(p for _, p in _log)
    for k, v in sorted(cnt.items()):
        print(f"  proposed={k:>2}: {v:>4} rounds")

    print("\nMatch-rate breakdown by num_proposed:")
    for p in sorted(cnt.keys()):
        sub = [m for m, pp in _log if pp == p]
        if not sub: continue
        rate = sum(sub) / (len(sub) * p) if p else 0
        print(f"  proposed={p:>2}: rounds={len(sub):>4}  total_matched={sum(sub):>4}  rate={rate:.1%}")


if __name__ == "__main__":
    main()
