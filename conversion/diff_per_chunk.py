#!/usr/bin/env python3
"""Per-chunk parity audit: Swift chunk1/2/3/4 outputs vs HF per-layer
hidden states at the same position.

Localizes which chunk first diverges (cosine drops below 0.999), so we
can drill into that chunk's MIL ops to find the bug.
"""
from __future__ import annotations
import numpy as np
import torch


def _load(name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(f"/tmp/mtp_chunks_swift/{name}.fp16", dtype=np.uint16)
    return raw.view(np.float16).reshape(shape).astype(np.float32)


def _diff(label: str, hf: np.ndarray, sw: np.ndarray):
    a = hf.flatten()
    b = sw.flatten()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    abs_diff = np.abs(a - b)
    print(f"{label:<35} cos={cos:.6f}  "
          f"|a|={np.linalg.norm(a):.2f}  |b|={np.linalg.norm(b):.2f}  "
          f"max_diff={abs_diff.max():.4f}  mean_diff={abs_diff.mean():.5f}")


def main():
    cap = torch.load("output/mtp_probe/hf_per_layer_capitals.pt", map_location="cpu",
                     weights_only=False)
    N_full = int(cap["prompt_token_ids"].shape[0])  # = 29 (28 prompt + 1 bootstrap)
    print(f"HF prompt+bootstrap len = {N_full}")
    bootstrap_pos = N_full - 1  # = 28

    # Swift dumps are at position 28 (= bootstrap_pos) after running predictStep.
    sw_h1 = _load("h1", (1, 1, 1536))[0, 0]
    sw_h2 = _load("h2", (1, 1, 1536))[0, 0]
    sw_h3 = _load("h3", (1, 1, 1536))[0, 0]
    sw_h4_postnorm = _load("h4_postnorm", (1, 1, 1536))[0, 0]

    # Chunk boundaries in our 4-chunk legacy bundle (verified earlier):
    # chunk1 = L0-7  → output is post-L7
    # chunk2 = L8-14 → output is post-L14
    # chunk3 = L15-24 → output is post-L24
    # chunk4 = L25-34 + final_norm → output is post-final-norm
    # HF hidden_states[0] = embed (post-scale), [1..N] = post-layer 0..N-1.
    # So post-L7 = HF[8], post-L14 = HF[15], post-L24 = HF[25],
    # post-final-norm = HF[-1] (last entry).
    hs = cap["hidden_states"]
    print(f"HF hidden_states len = {len(hs)}")

    hf_post_l7 = hs[8][0, bootstrap_pos].float().numpy()
    hf_post_l14 = hs[15][0, bootstrap_pos].float().numpy()
    hf_post_l24 = hs[25][0, bootstrap_pos].float().numpy()
    hf_postnorm = hs[-1][0, bootstrap_pos].float().numpy()

    print(f"\nPosition compared = {bootstrap_pos}")
    print(f"HF[0]  = embed scale check: |x|={np.linalg.norm(hs[0][0, bootstrap_pos]):.2f}")
    print(f"HF[8]  = post-L7 (chunk1 boundary)")
    print(f"HF[15] = post-L14 (chunk2 boundary)")
    print(f"HF[25] = post-L24 (chunk3 boundary)")
    print(f"HF[{len(hs)-1}] = post-final-norm (chunk4 output)")

    _diff("chunk1 (post-L7)", hf_post_l7, sw_h1)
    _diff("chunk2 (post-L14)", hf_post_l14, sw_h2)
    _diff("chunk3 (post-L24)", hf_post_l24, sw_h3)
    _diff("chunk4 (post-final-norm)", hf_postnorm, sw_h4_postnorm)


if __name__ == "__main__":
    main()
