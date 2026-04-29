#!/usr/bin/env python3
"""Generate real-prompt calibration data for `linear_quantize_activations`.

Stage 1 retry: synthetic N(0, 0.5) calibration produced cos sim 0.108
on chunk_1 (see docs/experiments/SESSION_2026_04_26_W4A8_HOLD.md). Real Gemma 4
embeddings have channel-wise outliers / heavy-tailed distributions that
random Gaussian doesn't capture.

This script writes a `.npz` containing one calibration sample per
(prompt, token-position) pair, using the actual embed_tokens lookup +
per_layer_embed lookup + RoPE table values that the production runtime
feeds into chunk_1 at decode time.

What we DON'T capture: KV cache state. The cml9 calibrator uses fresh
`make_state()` per sample (per the converter's
`_patch_calibrator_for_stateful` patch), so attention reads zero KV
during calibration regardless. That's a known limitation; the input
distribution is what matters here.

Usage:
  python conversion/gen_calib_data_real.py \
      --hf-dir /path/to/gemma4-e2b/hf_model \
      --output conversion/calibration_data/gemma4_chunk1_real.npz \
      --prompts-per-tokens 32 \
      --tokens-per-prompt 4

Output `.npz`: zero-padded samples named "sample_NNN" each containing
all 10 chunk_1 input arrays (hidden_states, masks, per_layer_raw,
cos_s/sin_s/cos_f/sin_f, current_pos, ring_pos).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model


# Diverse prompt set — English, Japanese, code, math. Goal is to span
# the embedding distribution that production sees, not to replicate any
# specific eval set.
PROMPTS = [
    "Hello, how are you today?",
    "The capital of France is",
    "import torch\nimport numpy as np\n",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "こんにちは、お元気ですか？",
    "東京の天気を教えてください。",
    "人工知能とは",
    "The quick brown fox jumps over the lazy dog.",
    "Solve: 2 + 2 * 3 =",
    "What is the meaning of life?",
    "Write a haiku about autumn:",
    "List five prime numbers:",
    "C'est la vie, mes amis.",
    "Pourquoi le ciel est-il bleu?",
    "El gato negro duerme.",
    "Bonjour le monde",
    "for i in range(10):\n    print(i)",
    "SELECT * FROM users WHERE id = 1;",
    "<html><head><title>Hello</title></head>",
    "git commit -m \"fix bug\"",
    "Translate to English: Guten Tag",
    "What year did WW2 end?",
    "The mitochondria is the",
    "0.1 + 0.2 = 0.30000000000000004",
    "function add(a, b) { return a + b; }",
    "李白《静夜思》：床前明月光",
    "안녕하세요 반갑습니다",
    "1, 1, 2, 3, 5, 8, 13, 21,",
    "The largest ocean on Earth is the",
    "Photosynthesis converts",
    "println(\"Hello, World!\")",
    "Once upon a time in a far-away",
]


def load_tokenizer(hf_dir: str):
    """Use the raw `tokenizers` library — transformers' GemmaTokenizerFast
    auto-load trips on transformers 4.x extra_special_tokens schema for
    Gemma 4. The tokenizer.json contents are identical either way."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(hf_dir, "tokenizer.json"))


def build_inputs_for_token(model: Gemma4Model, token_id: int, pos: int,
                            ctx: int, W: int) -> dict:
    """Construct the full chunk_1 input dict for a given token at a given
    position. Matches what Swift's Gemma4StatefulEngine builds at decode."""
    cfg = model.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim

    with torch.no_grad():
        # Token embedding × sqrt(hidden_size) (Gemma post-embed scale).
        ids = torch.tensor([[token_id]], dtype=torch.long)
        h = model.embed_tokens(ids).to(torch.float16)
        h = h * (hidden ** 0.5)
        hidden_states = h.view(1, 1, hidden).numpy().astype(np.float16)

        # Per-layer raw lookup × per_layer_embed_scale (sqrt(per_layer_dim)).
        plr = model.embed_tokens_per_layer(ids).to(torch.float16)
        plr = plr * model.per_layer_embed_scale
        per_layer_raw = plr.view(1, 1, nlayers * pld).numpy().astype(np.float16)

        # RoPE tables at this position.
        cos_s = model.cos_sliding[pos:pos + 1].view(1, 1, 1, hd_s).to(
            torch.float16).numpy().astype(np.float16)
        sin_s = model.sin_sliding[pos:pos + 1].view(1, 1, 1, hd_s).to(
            torch.float16).numpy().astype(np.float16)
        cos_f = model.cos_full[pos:pos + 1].view(1, 1, 1, hd_f).to(
            torch.float16).numpy().astype(np.float16)
        sin_f = model.sin_full[pos:pos + 1].view(1, 1, 1, hd_f).to(
            torch.float16).numpy().astype(np.float16)

    # Causal masks: zero for valid positions, large negative for future.
    mask_full = np.zeros((1, 1, 1, ctx), dtype=np.float16)
    if pos + 1 < ctx:
        mask_full[0, 0, 0, pos + 1:] = -1e4
    mask_sliding = np.zeros((1, 1, 1, W), dtype=np.float16)
    # Sliding-window with valid range [max(0, pos-W+1), pos]:
    valid_start = max(0, pos - W + 1)
    if valid_start > 0:
        mask_sliding[0, 0, 0, :valid_start] = -1e4
    if pos + 1 < W:
        mask_sliding[0, 0, 0, pos + 1:] = -1e4

    return {
        "hidden_states":         hidden_states,
        "causal_mask_full":      mask_full,
        "causal_mask_sliding":   mask_sliding,
        "per_layer_raw":         per_layer_raw,
        "cos_s":                 cos_s,
        "sin_s":                 sin_s,
        "cos_f":                 cos_f,
        "sin_f":                 sin_f,
        "current_pos":           np.array([pos], dtype=np.int32),
        "ring_pos":              np.array([pos % W], dtype=np.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True,
                    help="HuggingFace Gemma 4 E2B model directory.")
    ap.add_argument("--output", required=True,
                    help="Destination .npz path.")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--max-prompts", type=int, default=32,
                    help="Number of prompts (cycles PROMPTS list).")
    ap.add_argument("--tokens-per-prompt", type=int, default=4,
                    help="How many leading tokens of each prompt to "
                         "capture as separate calibration samples.")
    ap.add_argument("--max-samples", type=int, default=64,
                    help="Hard cap on total samples (cml9 calibration "
                         "is ~6 s/sample, so 64 ≈ 6-7 min).")
    args = ap.parse_args()

    print(f"Loading Gemma 4 E2B from {args.hf_dir}...")
    model = Gemma4Model.from_pretrained(args.hf_dir, context_length=args.ctx)
    model.eval()
    cfg = model.config
    W = cfg.sliding_window
    print(f"  hidden={cfg.hidden_size}  pld={cfg.hidden_size_per_layer_input}  "
          f"layers={cfg.num_hidden_layers}  head_dim={cfg.head_dim}  "
          f"global_head_dim={cfg.global_head_dim}  ctx={args.ctx}  W={W}")

    print(f"Loading tokenizer...")
    tok = load_tokenizer(args.hf_dir)

    prompts = PROMPTS[:args.max_prompts]
    samples: list[dict] = []
    for p_idx, prompt in enumerate(prompts):
        ids = tok.encode(prompt).ids  # raw `tokenizers.Encoding`
        # First N tokens of each prompt; pos increments per token.
        for t_idx, tok_id in enumerate(ids[:args.tokens_per_prompt]):
            sample = build_inputs_for_token(
                model, int(tok_id), pos=t_idx, ctx=args.ctx, W=W)
            samples.append(sample)
            if len(samples) >= args.max_samples:
                break
        if len(samples) >= args.max_samples:
            break

    print(f"Generated {len(samples)} samples from {len(prompts)} prompts.")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as .npz with one entry per sample, prefixed name__field.
    flat = {}
    for i, s in enumerate(samples):
        for k, v in s.items():
            flat[f"sample_{i:03d}__{k}"] = v
    flat["_meta_num_samples"] = np.array([len(samples)], dtype=np.int32)
    flat["_meta_input_names"] = np.array(list(samples[0].keys()))
    np.savez_compressed(out_path, **flat)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved {len(samples)} samples to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
