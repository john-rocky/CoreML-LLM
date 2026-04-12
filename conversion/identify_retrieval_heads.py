#!/usr/bin/env python3
"""Classify Gemma 4 attention heads as retrieval vs streaming.

This is a training-free proxy for DuoAttention (Xiao et al., ICLR 2025). The
full DuoAttention method uses synthetic passkey data + learnable per-head gates
with sparsity regularization. As a first-pass proxy we use **retrieval-head
attention analysis** (Wu et al., 2024 — "Retrieval Head Mechanistically
Explains Long-Context Factuality"): we plant a needle in a haystack, run the
target with output_attentions, and count for each (layer, head) how often it
attends from the query position to the needle position.

Heads that consistently copy from far back = retrieval heads (need full KV).
Heads that don't = streaming heads (safe to cap to sink+window on ANE).

Output: eagle3 style heads.json with per-layer per-head mask.

Usage (Colab, A100 preferred):
    pip install -q -U transformers datasets
    python conversion/identify_retrieval_heads.py \\
        --model-id google/gemma-4-E2B-it \\
        --output /content/drive/MyDrive/heads_gemma4_e2b.json \\
        --num-prompts 64 \\
        --ctx 8192

Runtime: ~30-60 min on A100 for 64 × 8K prompts.

Follow-up: once heads.json is produced, `gemma4_swa_chunks.py` must be
modified to use a short sink+window KV buffer for streaming heads while
keeping full KV for retrieval heads. That is a Tier-2 conversion task, not
in this script.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm


# ── Needle-in-a-haystack prompt generation ──────────────────────────────────

FILLER_TOPICS = [
    "The history of transistors began in the early 20th century.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The French Revolution reshaped European political thought.",
    "Machine learning uses statistical methods to find patterns in data.",
    "Ocean currents distribute heat across the globe, affecting climate.",
    "The invention of the printing press revolutionized knowledge diffusion.",
    "Electromagnetic induction is fundamental to modern power generation.",
    "Ancient Egyptian civilization developed along the Nile for millennia.",
    "Tectonic plates slowly reshape the earth's surface over geological time.",
    "The Renaissance brought renewed interest in classical art and science.",
]


def make_needle_prompt(ctx_tokens: int, tokenizer, rng: random.Random) -> tuple[str, int, int, str]:
    """Build a needle-in-haystack prompt.

    Returns (prompt_text, needle_start_char, needle_length_tokens_hint, needle_answer).
    The hint is used later to locate the needle position in token space.
    """
    # Random 5-digit passkey
    key = "".join(rng.choice(string.digits) for _ in range(5))
    needle = f" The special passkey is {key}. Please remember it. "

    # Fill until close to ctx_tokens
    filler_parts = []
    token_budget = int(ctx_tokens * 0.95)  # leave room for question/answer
    rough = 0
    while rough < token_budget:
        s = rng.choice(FILLER_TOPICS) + " "
        filler_parts.append(s)
        rough += len(s.split())

    # Insert needle at a random position in the middle 50-80% of the haystack
    pos = rng.randint(len(filler_parts) // 2, int(len(filler_parts) * 0.8))
    filler_parts.insert(pos, needle)
    haystack = "".join(filler_parts)

    question = f"\n\nQuestion: What is the special passkey mentioned above? Answer: "
    prompt = haystack + question
    return prompt, key


# ── Head attention scoring via hooks ────────────────────────────────────────

class HeadAttentionRecorder:
    """Captures per-head attention weight at the final query position."""

    def __init__(self):
        self.records = []   # list of (layer_idx, attn_tensor[num_heads, seq_len])
        self._handles = []

    def _hook(self, layer_idx):
        def fn(module, inputs, output):
            # output may be (attn_out, attn_weights) depending on model impl.
            # For SDPA-based Gemma 4, attention weights are NOT returned by default.
            # Users need to patch the attention to save attn_weights.
            pass
        return fn

    def clear(self):
        self.records.clear()


def collect_attention_via_forward_with_weights(model, input_ids, device):
    """Try to get attention weights. Modern HF forces output_attentions=True
    to disable SDPA fast path and use eager attention (which supports weights)."""
    out = model(
        input_ids=input_ids,
        output_attentions=True,
        output_hidden_states=False,
        use_cache=False,
    )
    # out.attentions: tuple of (B, num_heads, seq, seq) per layer
    return out.attentions


# ── Scoring logic ───────────────────────────────────────────────────────────

def score_heads(attentions, needle_start: int, needle_end: int, query_pos: int):
    """For each (layer, head), compute attention mass from query_pos on
    needle_start..needle_end-1 as the "retrieval signal".

    attentions: tuple of L tensors, each (1, H, seq, seq)
    returns: dict layer_idx -> np.array of shape (H,)
    """
    import numpy as np
    scores = {}
    for l, attn in enumerate(attentions):
        # attn[:, :, query_pos, :] = attention distribution from query over keys
        row = attn[0, :, query_pos, :]                     # (H, seq)
        mass = row[:, needle_start:needle_end].sum(-1)     # (H,)
        scores[l] = mass.float().cpu().numpy()
    return scores


def locate_needle(tokenizer, prompt: str, key: str) -> tuple[int, int]:
    """Return (start_tok, end_tok_exclusive) of the substring containing the key."""
    ids_full = tokenizer.encode(prompt)
    # Use a marker substring that includes the key to locate. Fall back to key alone.
    markers = [f"passkey is {key}", key]
    for m in markers:
        if m in prompt:
            before = prompt.split(m, 1)[0]
            ids_before = tokenizer.encode(before)
            ids_marker = tokenizer.encode(m, add_special_tokens=False)
            start = len(ids_before)
            end = start + len(ids_marker)
            if 0 <= start < end <= len(ids_full):
                return start, end
    return -1, -1


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    ap.add_argument("--output", type=str, default="./heads_gemma4_e2b.json")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-frac", type=float, default=0.5,
                    help="Fraction of heads (top by retrieval score) kept as retrieval heads.")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = args.device

    print(f"Loading {args.model_id}...")
    try:
        from transformers import Gemma4ForConditionalGeneration as TCls
    except Exception:
        from transformers import AutoModelForCausalLM as TCls
    from transformers import AutoTokenizer
    target = TCls.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map=device,
        attn_implementation="eager"   # required to get attn weights
    )
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    tcfg = target.config.text_config if hasattr(target.config, "text_config") else target.config
    num_heads = tcfg.num_attention_heads
    num_layers = tcfg.num_hidden_layers
    print(f"layers={num_layers}, heads={num_heads}")

    import numpy as np
    score_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    count = 0

    t0 = time.time()
    for _ in tqdm(range(args.num_prompts), desc="needle prompts"):
        prompt, key = make_needle_prompt(args.ctx, tokenizer, rng)
        start, end = locate_needle(tokenizer, prompt, key)
        if start < 0:
            continue

        ids = tokenizer.encode(
            prompt, return_tensors="pt",
            truncation=True, max_length=args.ctx
        ).to(device)
        if ids.shape[1] <= end:
            # Needle got truncated out; skip
            continue
        query_pos = ids.shape[1] - 1

        with torch.no_grad():
            attentions = collect_attention_via_forward_with_weights(target, ids, device)
        if attentions is None:
            print("WARN: model returned no attentions. Ensure attn_implementation='eager'.")
            return 1

        scores = score_heads(attentions, start, end, query_pos)
        for l, arr in scores.items():
            score_sum[l] += arr
        count += 1

    if count == 0:
        print("ERROR: zero valid prompts; cannot compute head scores.")
        return 1

    # Average and normalize
    avg = score_sum / count                          # (L, H) retrieval mass
    flat = avg.flatten()
    threshold = np.quantile(flat, 1 - args.top_frac)
    mask = (avg >= threshold).astype(int)            # 1 = retrieval, 0 = streaming
    retrieval_count = int(mask.sum())
    total_heads = num_layers * num_heads
    print(f"retrieval heads: {retrieval_count}/{total_heads} "
          f"({100 * retrieval_count / total_heads:.1f}%, target {args.top_frac*100:.0f}%)")

    # Save
    out = {
        "model_id": args.model_id,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_prompts": count,
        "ctx": args.ctx,
        "top_frac": args.top_frac,
        "scores": avg.tolist(),                      # L × H, float
        "is_retrieval": mask.tolist(),               # L × H, 0/1
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {args.output}")

    # Report per-layer retrieval share
    per_layer = mask.sum(axis=1)
    print("\nretrieval heads per layer (max = {}):".format(num_heads))
    for l, n in enumerate(per_layer):
        bar = "█" * int(n)
        print(f"  L{l:2d}: {n:2d} {bar}")

    dt = time.time() - t0
    print(f"\ndone in {dt/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
