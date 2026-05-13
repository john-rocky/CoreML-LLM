#!/usr/bin/env python3
"""HF runtime in fp16 to check whether precision is a major drag.

Mac: torch.float16 on CPU works (slow); MPS gives speed but might
have its own numerics. Compare fp32 vs fp16 to see how much accept
the Swift fp16/INT4 ANE setup is "supposed" to lose just from precision.
"""
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import candidate_generator as _cg

_log: list[tuple[int, int]] = []
_orig = _cg.SinglePositionMultiTokenCandidateGenerator.update_candidate_strategy


def _patched(self, input_ids, scores, num_matches):
    _log.append((int(num_matches), int(self.num_assistant_tokens)))
    return _orig(self, input_ids, scores, num_matches)


_cg.SinglePositionMultiTokenCandidateGenerator.update_candidate_strategy = _patched

PROMPT = ("Write a complete Python module that implements a doubly-linked list with "
          "insert, delete, search, reverse, and to_list methods. Include type hints, "
          "docstrings on every method, and a test suite using unittest with at least "
          "10 test cases.")


def _bench(label, target, drafter, tok, input_ids):
    _log.clear()
    with torch.no_grad():
        out = target.generate(input_ids=input_ids, max_new_tokens=384,
                              do_sample=False, pad_token_id=tok.eos_token_id,
                              assistant_model=drafter)
    n = out.shape[1] - input_ids.shape[1]
    accept = (sum(m for m, _ in _log) / sum(p for _, p in _log)) if _log else 0
    full = sum(1 for m, p in _log if m == p)
    zero = sum(1 for m, _ in _log if m == 0)
    print(f"{label:<25} gen={n:>4}t  rounds={len(_log):>4}  "
          f"accept={accept:.1%}  full={full}  zero={zero}")


def main():
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    msgs = [{"role": "user", "content": PROMPT}]
    chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    chat_ids = tok(chat_text, return_tensors="pt", add_special_tokens=False).input_ids

    for label, dtype in [("fp32", torch.float32), ("fp16", torch.float16),
                         ("bf16", torch.bfloat16)]:
        try:
            target = AutoModelForCausalLM.from_pretrained(
                "google/gemma-4-E2B-it", dtype=dtype,
                low_cpu_mem_usage=True).eval()
            drafter = AutoModelForCausalLM.from_pretrained(
                "google/gemma-4-E2B-it-assistant", dtype=dtype,
                low_cpu_mem_usage=True).eval()
            _bench(label, target, drafter, tok, chat_ids)
        except Exception as e:
            print(f"{label} FAILED: {e}")


if __name__ == "__main__":
    main()
