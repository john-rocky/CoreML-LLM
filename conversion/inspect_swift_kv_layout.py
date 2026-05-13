#!/usr/bin/env python3
"""Inspect Swift K cache layout: per-slot magnitude reveals where positions sit.

If positions are right-aligned with newest at slot W-1, we expect:
  - magnitudes near 0 in slots 0..W-M-1
  - magnitudes nonzero in slots W-M..W-1
And the "newest" position should have a similar magnitude pattern to the
"oldest" — both are real K. So magnitude alone doesn't tell us order.

But: HF's last position (N-1) was input prompt[27] (token id from chat template).
Swift's bootstrap input was target's argmax = 818 ("The"). Different tokens
at the same position in different runs.

What we CAN check: do the FIRST few positions (where token ids match: BOS=2,
105, 2364, 107) produce identical K? If yes, it tells us where in the cache
position 0 lives.
"""
from __future__ import annotations
import numpy as np
import torch


def _load_swift(name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(f"/tmp/mtp_swift_kv/{name}.fp16", dtype=np.uint16)
    return raw.view(np.float16).reshape(shape).astype(np.float32)


def main():
    cap = torch.load("output/mtp_probe/hf_kv_chat_capitals.pt", map_location="cpu",
                     weights_only=False)
    hf_swa_k = cap["swa_k"].float().numpy()        # (1, 1, N=28, 256)
    hf_full_k = cap["full_k"].float().numpy()      # (1, 1, N=28, 512)
    print(f"HF prompt N={hf_swa_k.shape[2]}")

    swift_kv13_k = _load_swift("kv13_k", (1, 1, 512, 256))
    swift_kv14_k = _load_swift("kv14_k", (1, 1, 2048, 512))

    # Per-slot magnitude in Swift's kv13_k.
    mag13 = np.linalg.norm(swift_kv13_k[0, 0], axis=-1)  # (512,)
    nz13 = np.where(mag13 > 1e-3)[0]
    print(f"\nSwift kv13_k non-zero slots: {len(nz13)}  range [{nz13.min()}..{nz13.max()}]")
    print(f"  first 5 nz slots: {nz13[:5].tolist()}  mags: {mag13[nz13[:5]].round(3).tolist()}")
    print(f"  last 5  nz slots: {nz13[-5:].tolist()}  mags: {mag13[nz13[-5:]].round(3).tolist()}")

    # Per-slot magnitude in Swift's kv14_k.
    mag14 = np.linalg.norm(swift_kv14_k[0, 0], axis=-1)  # (2048,)
    nz14 = np.where(mag14 > 1e-3)[0]
    print(f"\nSwift kv14_k non-zero slots: {len(nz14)}  range [{nz14.min()}..{nz14.max()}]")

    # HF magnitudes for reference.
    hf_mag13 = np.linalg.norm(hf_swa_k[0, 0], axis=-1)
    hf_mag14 = np.linalg.norm(hf_full_k[0, 0], axis=-1)
    print(f"\nHF L13 K per-position magnitudes: {hf_mag13.round(3).tolist()}")
    print(f"HF L14 K per-position magnitudes: {hf_mag14.round(3).tolist()}")
    print(f"\nSwift kv13_k slot magnitudes (last 30): {mag13[-30:].round(3).tolist()}")
    print(f"Swift kv14_k slot magnitudes (first 30): {mag14[:30].round(3).tolist()}")

    # Also try alignment hypotheses.
    N = hf_swa_k.shape[2]
    W = 512
    M_swift = 29  # 28 prompt + 1 bootstrap

    print("\n=== Alignment test: try several mappings ===")

    def _try(label, slice_indices: list[int]):
        """slice_indices[i] = which Swift slot maps to HF position i (i in 0..N-1)."""
        sw = swift_kv13_k[0, 0, slice_indices, :]     # (N, 256)
        hf = hf_swa_k[0, 0]                          # (N, 256)
        # per-position cosine
        cos = np.einsum('ij,ij->i', hf, sw) / (np.linalg.norm(hf, axis=-1) * np.linalg.norm(sw, axis=-1) + 1e-8)
        print(f"{label:<60} mean cos = {cos.mean():.4f}  (first 5: {cos[:5].round(4).tolist()})")

    # Hypothesis 1: right-aligned, position i at slot W-M+i (so newest at W-1).
    _try("RA newest at slot 511, position 0 at slot W-M (=483)",
         [W - M_swift + i for i in range(N)])
    # Hypothesis 2: right-aligned reversed (newest at slot W-M).
    _try("RA-rev newest at slot W-M=483, position 0 at slot W-1=511",
         [W - 1 - i for i in range(N)])
    # Hypothesis 3: position i at slot W-M+i+1 (skipping slot W-M for bootstrap)
    _try("RA shifted by 1 (bootstrap at oldest, prompt at newer slots)",
         [W - M_swift + 1 + i for i in range(N)])
    # Hypothesis 4: left-aligned (newest at slot N-1)
    _try("LA position 0 at slot 0, newest at slot N-1",
         list(range(N)))


if __name__ == "__main__":
    main()
