#!/usr/bin/env python3
"""HF bench v2 — verify chat-template drag claim with proper sample sizes.

Earlier (`bench_hf_chat_template.py`) I concluded chat-template halves
accept (90 % → 33 %) but the chat-template run only had 3 drafter
rounds before the model hit EOS. That's not a reliable accept measure.

This bench:
  - Uses prompts that force LONG outputs (50+ rounds of drafting).
  - Tries multiple chat templates / formats.
  - Disables early stop where possible.
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


PROMPTS = [
    ("capitals_list",
     "List the capitals of these 30 countries: France, Germany, Italy, Spain, "
     "Japan, China, Korea, India, Brazil, Argentina, Mexico, Canada, Australia, "
     "Russia, Egypt, Kenya, Nigeria, South Africa, Turkey, Greece, Portugal, "
     "Ireland, Netherlands, Belgium, Sweden, Norway, Finland, Denmark, "
     "Poland, Austria. Respond with one country and capital per line, no extra text."),
    ("code_long",
     "Write a complete Python module that implements a doubly-linked list with "
     "insert, delete, search, reverse, and to_list methods. Include type hints, "
     "docstrings on every method, and a test suite using unittest with at least "
     "10 test cases."),
    ("essay_long",
     "Write a 500-word essay on the impact of generative AI on software "
     "engineering productivity, covering code generation, testing, debugging, "
     "and documentation."),
    ("qa_long",
     "Explain in detail how a CPU executes an instruction, covering fetch, "
     "decode, execute, memory access, and writeback stages. Discuss pipelining "
     "and out-of-order execution. Aim for 400+ words."),
]


def _bench(label, target, drafter, tok, input_ids, max_new_tokens=384):
    _log.clear()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target.generate(input_ids=input_ids, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    base_dt = time.perf_counter() - t0
    base_n = out.shape[1] - input_ids.shape[1]
    base_tps = base_n / base_dt

    _log.clear()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target.generate(input_ids=input_ids, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id,
                              assistant_model=drafter)
    asst_dt = time.perf_counter() - t0
    asst_n = out.shape[1] - input_ids.shape[1]
    asst_tps = asst_n / asst_dt
    rounds = len(_log)
    if _log:
        accept = sum(m for m, _ in _log) / sum(p for _, p in _log)
    else:
        accept = float("nan")
    print(f"{label:<35} {base_n:>4}t/{base_tps:>5.1f}tps  "
          f"{asst_n:>4}t/{asst_tps:>5.1f}tps  "
          f"{asst_tps/base_tps:>5.2f}x  acc={accept:>5.1%}  rounds={rounds}")


def main():
    target = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=torch.float32,
        low_cpu_mem_usage=True).eval()
    drafter = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it-assistant", dtype=torch.float32,
        low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

    print(f"\n{'prompt':<35} {'baseline':>14}  {'assisted':>14}  "
          f"speedup  {'accept':>7}  rounds")
    print("-" * 100)

    for name, prompt in PROMPTS:
        # raw form
        raw = tok(prompt, return_tensors="pt").input_ids
        _bench(f"raw_{name} ({raw.shape[1]}tok)",
               target, drafter, tok, raw)

        # chat-template form
        msgs = [{"role": "user", "content": prompt}]
        chat_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        chat = tok(chat_text, return_tensors="pt",
                   add_special_tokens=False).input_ids
        _bench(f"chat_{name} ({chat.shape[1]}tok)",
               target, drafter, tok, chat)
        print()


if __name__ == "__main__":
    main()
