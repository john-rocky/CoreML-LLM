"""Phase 1: generate the PyTorch parity oracle for Qwen3.5-0.8B text-only.

Loads Qwen/Qwen3.5-0.8B via Qwen3_5ForCausalLM + Qwen3_5TextConfig (vision + MTP
excluded without patching), runs fixed prompts through both the chunked (prefill)
and recurrent (decode) Gated-DeltaNet paths, and saves logits + top-10 tokens as
the reference for later CoreML parity checks.

Output: conversion/qwen3_5_reference_logits.pt
"""

import time
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-0.8B"
OUTPUT = Path(__file__).parent / "qwen3_5_reference_logits.pt"

# Spans 8 -> ~80 tokens to exercise both prefill-chunk boundaries and short seqs.
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Write a haiku about the ocean:",
    "Q: What is 17 * 23? A:",
    "Once upon a time in a small village nestled between tall mountains, there lived",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The key difference between a linked list and an array is that",
    "In the year 2050, artificial intelligence had advanced to the point where",
    "The following Python function computes the factorial of a non-negative integer:\n\ndef factorial(n):\n    ",
    "Translate to Japanese: The quick brown fox jumps over the lazy dog.",
]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device):
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    dtype = torch.float32  # oracle precision; SSM inside already fp32 per config
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return model, cfg


@torch.no_grad()
def run_prefill(model, input_ids):
    out = model(input_ids=input_ids, use_cache=False)
    return out.logits  # (1, S, V)


@torch.no_grad()
def run_recurrent_decode(model, input_ids):
    """Feed tokens one at a time with use_cache=True. Returns last-token logits
    after each step (1, V) stacked into (S, V)."""
    past = None
    step_logits = []
    for i in range(input_ids.shape[1]):
        tok = input_ids[:, i:i+1]
        out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        step_logits.append(out.logits[:, -1, :].squeeze(0).cpu())
    return torch.stack(step_logits, dim=0)  # (S, V)


def cos_sim(a, b, eps=1e-8):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return (a @ b / (a.norm() * b.norm() + eps)).item()


def main():
    device = pick_device()
    print(f"device: {device}")
    t0 = time.time()
    model, cfg = load_model(device)
    print(f"loaded in {time.time()-t0:.1f}s, "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    records = []
    for prompt in PROMPTS:
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        S = input_ids.shape[1]
        print(f"\nprompt (S={S}): {prompt!r}")

        t1 = time.time()
        logits_prefill = run_prefill(model, input_ids).squeeze(0).cpu()  # (S, V)
        t_prefill = time.time() - t1

        t2 = time.time()
        logits_recurrent = run_recurrent_decode(model, input_ids)         # (S, V)
        t_recurrent = time.time() - t2

        cos_per_pos = torch.tensor([
            cos_sim(logits_prefill[i], logits_recurrent[i]) for i in range(S)
        ])
        cos_mean = cos_per_pos.mean().item()
        cos_min = cos_per_pos.min().item()

        # Top-k at the LAST position
        topk_prefill = torch.topk(logits_prefill[-1], k=10)
        next_tok_pref = topk_prefill.indices[0].item()
        next_tok_text = tok.decode([next_tok_pref])
        print(f"  prefill  {t_prefill:.2f}s  recurrent {t_recurrent:.2f}s")
        print(f"  prefill vs recurrent cos mean={cos_mean:.6f} min={cos_min:.6f}")
        print(f"  next token: {next_tok_pref} ({next_tok_text!r})")

        records.append({
            "prompt": prompt,
            "input_ids": input_ids.cpu(),
            "logits_prefill": logits_prefill.half(),
            "logits_recurrent": logits_recurrent.half(),
            "cos_per_pos": cos_per_pos,
            "top10_last_ids": topk_prefill.indices,
            "top10_last_vals": topk_prefill.values,
            "next_token_text": next_tok_text,
            "timings": {"prefill_s": t_prefill, "recurrent_s": t_recurrent},
        })

    torch.save({
        "model_id": MODEL_ID,
        "config": cfg.to_dict(),
        "records": records,
    }, OUTPUT)
    print(f"\nsaved {OUTPUT} ({OUTPUT.stat().st_size/1e6:.1f} MB)")

    worst = min(r["cos_per_pos"].min().item() for r in records)
    mean = sum(r["cos_per_pos"].mean().item() for r in records) / len(records)
    print(f"\noverall parity: mean cos={mean:.6f}, worst single-position cos={worst:.6f}")
    if worst < 0.999:
        print("WARNING: prefill vs recurrent divergence > 0.001 — investigate")


if __name__ == "__main__":
    main()
