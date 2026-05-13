#!/usr/bin/env python3
"""HF bench WITH chat template applied — match Swift's actual prompt format.

If Swift Mac drops to ~29 % vs HF (raw) 90 %, the lurking variable is
the chat template tokens. This bench reruns HF on the chat-templated
prompt to see whether HF accept also drops, isolating the effect.
"""
from __future__ import annotations
import time
import torch
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
    prompt = ("The capital of France is Paris. The capital of Germany is Berlin. "
              "The capital of Italy is")

    print(f"\n{'mode':<25} {'baseline tps':>12} {'asst tps':>10} {'speedup':>8} {'accept':>8} {'rounds':>7}")
    print("-" * 80)

    # Run 1: raw prompt, no template.
    raw_ids = tok(prompt, return_tensors="pt")
    print(f"raw_prompt_len = {raw_ids.input_ids.shape[1]} tokens")

    def _bench(label: str, input_ids):
        _log.clear()
        # baseline
        t0 = time.perf_counter()
        with torch.no_grad():
            out = target.generate(input_ids=input_ids, max_new_tokens=64,
                                   do_sample=False, pad_token_id=tok.eos_token_id)
        base_tps = (out.shape[1] - input_ids.shape[1]) / (time.perf_counter() - t0)
        # assisted
        _log.clear()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = target.generate(input_ids=input_ids, max_new_tokens=64,
                                   do_sample=False, pad_token_id=tok.eos_token_id,
                                   assistant_model=drafter)
        asst_tps = (out.shape[1] - input_ids.shape[1]) / (time.perf_counter() - t0)
        rounds = len(_log)
        accept = sum(m for m, _ in _log) / sum(p for _, p in _log) if _log else 0
        print(f"{label:<25} {base_tps:>12.2f} {asst_tps:>10.2f} {asst_tps/base_tps:>7.2f}x "
              f"{accept:>7.2%} {rounds:>7d}")

    _bench("raw_prompt", raw_ids.input_ids)

    # Run 2: with chat template (user-only message).
    msgs = [{"role": "user", "content": prompt}]
    chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    chat_ids = tok(chat_text, return_tensors="pt", add_special_tokens=False).input_ids
    print(f"chat_prompt_len = {chat_ids.shape[1]} tokens")
    _bench("chat_template_applied", chat_ids)


if __name__ == "__main__":
    main()
