#!/usr/bin/env python3
"""Numerically diff HF target's L13/L14 K/V vs Swift's kv13/kv14.

HF: tensor shape (1, 1, N, head_dim), positions 0..N-1.
Swift kv13: (1, 1, 512, 256) right-aligned. With N=28 prompt + bootstrap K[N=28],
            29 slots filled at indices 483..511 (oldest 0 at 483, newest 28 at 511).
Swift kv14: (1, 1, 2048, 512) left-aligned. 29 slots at indices 0..28.

Comparison:
- For positions 0..N-1 (where HF has data), compute element-wise diff.
- Slot 511 of Swift kv13 (= position 28 = bootstrap K) is "extra" and not in HF capture.
"""
from __future__ import annotations
import numpy as np
import torch


def _load_swift(name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(f"/tmp/mtp_swift_kv/{name}.fp16", dtype=np.uint16)
    return raw.view(np.float16).reshape(shape).astype(np.float32)


def _diff(label: str, hf: np.ndarray, swift: np.ndarray) -> None:
    """Both must be same shape."""
    assert hf.shape == swift.shape, f"{label}: shape {hf.shape} vs {swift.shape}"
    abs_diff = np.abs(hf - swift)
    cosines = []
    for i in range(hf.shape[2]):
        a = hf[0, 0, i].flatten()
        b = swift[0, 0, i].flatten()
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
        cos = float(np.dot(a, b) / denom)
        cosines.append(cos)
    print(f"\n{label}: shape={hf.shape}")
    print(f"  per-position cosine: min={min(cosines):.6f} max={max(cosines):.6f} "
          f"mean={sum(cosines)/len(cosines):.6f}")
    print(f"  per-position cosine all positions:")
    for i, c in enumerate(cosines):
        flag = " <-- LOW" if c < 0.99 else ""
        print(f"    pos {i:>3}: {c:.6f}{flag}")
    print(f"  abs_diff stats: max={abs_diff.max():.4f} mean={abs_diff.mean():.5f} "
          f"max_pos={int(np.argmax(abs_diff.reshape(-1)) // np.prod(abs_diff.shape[3:]))}")


def main():
    cap = torch.load("output/mtp_probe/hf_kv_chat_capitals.pt", map_location="cpu",
                     weights_only=False)
    N = int(cap["prompt_token_ids"].shape[0])
    print(f"HF prompt N={N}, next argmax = {cap['next_argmax']}")

    hf_swa_k = cap["swa_k"].float().numpy()        # (1, 1, N, 256)
    hf_swa_v = cap["swa_v"].float().numpy()        # (1, 1, N, 256)
    hf_full_k = cap["full_k"].float().numpy()      # (1, 1, N, 512)
    hf_full_v = cap["full_v"].float().numpy()      # (1, 1, N, 512)

    swift_kv13_k = _load_swift("kv13_k", (1, 1, 512, 256))
    swift_kv13_v = _load_swift("kv13_v_raw", (1, 1, 512, 256))
    swift_kv14_k = _load_swift("kv14_k", (1, 1, 2048, 512))
    swift_kv14_v = _load_swift("kv14_v_raw", (1, 1, 2048, 512))

    # Swift sliding cache: right-aligned, M positions in slots [W-M..W-1].
    # With Swift's prompt being chat-templated: prefill writes 0..N-1 (28 slots),
    # bootstrap writes N (=28). So 29 positions in slots [W-29..W-1] = [483..511].
    # We want positions 0..N-1 = slots [483..510] (slot 511 is bootstrap K).
    W = 512
    M_swift = 29   # number of positions Swift has written so far
    bootstrap_pos = N  # = 28
    # Swift slots [W-M..W-1] = positions [0..M-1]
    swift_swa_k_aligned = swift_kv13_k[:, :, W - M_swift:W - 1, :]  # positions 0..N-1 (28 slots)
    swift_swa_v_aligned = swift_kv13_v[:, :, W - M_swift:W - 1, :]
    print(f"Swift sliding K aligned to HF positions: shape={swift_swa_k_aligned.shape}")

    # Swift full cache: left-aligned. positions 0..M-1 in slots 0..M-1.
    swift_full_k_aligned = swift_kv14_k[:, :, :N, :]  # positions 0..N-1
    swift_full_v_aligned = swift_kv14_v[:, :, :N, :]
    print(f"Swift full K aligned to HF positions: shape={swift_full_k_aligned.shape}")

    _diff("L13 (sliding) K", hf_swa_k, swift_swa_k_aligned)
    _diff("L13 (sliding) V", hf_swa_v, swift_swa_v_aligned)
    _diff("L14 (full) K", hf_full_k, swift_full_k_aligned)
    _diff("L14 (full) V", hf_full_v, swift_full_v_aligned)

    # Last hidden state: HF (1, N, 1536) — pick last position N-1.
    hf_last_hidden = cap["last_hidden"].float().numpy()  # (1, N, 1536)
    swift_last_hidden = _load_swift("last_decode_hidden", (1, 1, 1536))
    # Compare HF[N-1] vs Swift[0, 0]
    a = hf_last_hidden[0, N - 1].flatten()
    b = swift_last_hidden[0, 0].flatten()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    print(f"\nlast_hidden (post-final-norm) at last prompt position:")
    print(f"  HF[N-1={N-1}] vs Swift[0,0]  cosine={cos:.6f}")
    print(f"  HF magnitude: {np.linalg.norm(a):.2f}  Swift magnitude: {np.linalg.norm(b):.2f}")


if __name__ == "__main__":
    main()
