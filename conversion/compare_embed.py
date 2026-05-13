#!/usr/bin/env python3
"""Compare HF's embed_tokens(token_id) * sqrt(hidden) vs Swift's
embed_tokens (INT8 dequantized + scaled). Localizes whether the chunk1
input is correct.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def _swift_embed(token_id: int, embed_tokens_path: str, scales_path: str,
                 vocab_size: int, hidden_size: int, embed_scale: float) -> np.ndarray:
    """Mirror EmbeddingLookup.lookup() in Swift."""
    raw = np.fromfile(embed_tokens_path, dtype=np.int8)
    assert raw.size == vocab_size * hidden_size
    raw = raw.reshape(vocab_size, hidden_size)
    scales_raw = np.fromfile(scales_path, dtype=np.uint16)
    assert scales_raw.size == vocab_size
    scales_fp16 = scales_raw.view(np.float16).astype(np.float32)
    row = raw[token_id].astype(np.float32)
    row_scale = scales_fp16[token_id] / 127.0 * embed_scale
    return (row * row_scale).astype(np.float16).astype(np.float32)


def main():
    target = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=torch.float16, low_cpu_mem_usage=True).eval()
    embed_tokens = target.model.language_model.embed_tokens
    print(f"HF embed_tokens type: {type(embed_tokens).__name__}")
    print(f"HF embed_scale = {float(embed_tokens.embed_scale.float()):.6f}")

    for token_id in [2, 105, 818, 5279, 11702]:
        hf_emb = embed_tokens(torch.tensor([[token_id]])).detach().float().numpy()[0, 0]
        sw_emb = _swift_embed(
            token_id,
            "output/gemma4-e2b/bundle/embed_tokens_q8.bin",
            "output/gemma4-e2b/bundle/embed_tokens_scales.bin",
            vocab_size=262144, hidden_size=1536,
            embed_scale=math.sqrt(1536))
        cos = float(np.dot(hf_emb, sw_emb) / (np.linalg.norm(hf_emb) * np.linalg.norm(sw_emb) + 1e-8))
        diff = np.abs(hf_emb - sw_emb)
        print(f"  token {token_id:>6}: cos={cos:.6f}  |hf|={np.linalg.norm(hf_emb):.2f}  "
              f"|sw|={np.linalg.norm(sw_emb):.2f}  max_diff={diff.max():.4f}  mean_diff={diff.mean():.5f}")


if __name__ == "__main__":
    main()
