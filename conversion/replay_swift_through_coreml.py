#!/usr/bin/env python3
"""Replay Swift's actual round-1 drafter inputs through the CoreML model.

If CoreML produces the right answer with HF capture but the wrong answer
with Swift's dump, we know exactly which inputs Swift mangled.
"""
from __future__ import annotations
import os
import numpy as np
import coremltools as ct


SHAPES = {
    "embed_token": (1, 1, 1536),
    "proj_act": (1, 1, 1536),
    "kv13_k": (1, 1, 512, 256),
    "kv13_v": (1, 1, 256, 512),
    "kv14_k": (1, 1, 2048, 512),
    "kv14_v": (1, 1, 512, 2048),
    "mask_full": (1, 1, 1, 2048),
    "mask_swa": (1, 1, 1, 512),
    "cos_full": (1, 256),
    "cos_swa": (1, 128),
    "sin_full": (1, 256),
    "sin_swa": (1, 128),
}


def _load(name: str) -> np.ndarray:
    path = f"/tmp/mtp_swift_r1/{name}.fp16"
    raw = np.fromfile(path, dtype=np.uint16).view(np.float16)
    return raw.reshape(SHAPES[name])


def main():
    model = ct.models.MLModel("mtp_drafter_ctx2k.mlpackage",
                              compute_units=ct.ComputeUnit.CPU_AND_NE)
    feed = {name: _load(name) for name in SHAPES}
    print("[swift-replay] feeding Swift round-1 inputs to CoreML drafter")
    out = model.predict(feed)
    top_k = out["top_k_indices"].flatten().tolist()
    top_v = out["top_k_values"].flatten().tolist()
    print(f"[swift-replay] CoreML top-k = {top_k}")
    print(f"[swift-replay] CoreML top-v = {[f'{v:.3f}' for v in top_v]}")

    # Also dump some sanity stats on Swift inputs.
    et = _load("embed_token").astype(np.float32)
    pa = _load("proj_act").astype(np.float32)
    kv13k = _load("kv13_k").astype(np.float32)
    kv14k = _load("kv14_k").astype(np.float32)
    print(f"\n[stats] embed_token   min={et.min():.3f} max={et.max():.3f} mean={et.mean():.4f} std={et.std():.4f}")
    print(f"[stats] proj_act      min={pa.min():.3f} max={pa.max():.3f} mean={pa.mean():.4f} std={pa.std():.4f}")
    # Per-slot non-zero detection on K caches.
    kv13_nz = (np.abs(kv13k).sum(axis=-1) > 1e-6)[0, 0]   # (512,)
    kv14_nz = (np.abs(kv14k).sum(axis=-1) > 1e-6)[0, 0]   # (2048,)
    print(f"[stats] kv13_k non-zero slots: {int(kv13_nz.sum())} of 512  "
          f"first={int(np.argmax(kv13_nz))}  last={int(512 - np.argmax(kv13_nz[::-1]) - 1)}")
    print(f"[stats] kv14_k non-zero slots: {int(kv14_nz.sum())} of 2048  "
          f"first={int(np.argmax(kv14_nz))}  last={int(2048 - np.argmax(kv14_nz[::-1]) - 1)}")
    # Mask check.
    msk_swa = _load("mask_swa").astype(np.float32)[0, 0, 0]   # (512,)
    msk_full = _load("mask_full").astype(np.float32)[0, 0, 0] # (2048,)
    msk_swa_allowed = (msk_swa > -1e3)
    msk_full_allowed = (msk_full > -1e3)
    print(f"[stats] mask_swa  allowed slots: {int(msk_swa_allowed.sum())} of 512  "
          f"first={int(np.argmax(msk_swa_allowed))}  last={int(512 - np.argmax(msk_swa_allowed[::-1]) - 1)}")
    print(f"[stats] mask_full allowed slots: {int(msk_full_allowed.sum())} of 2048  "
          f"first={int(np.argmax(msk_full_allowed))}  last={int(2048 - np.argmax(msk_full_allowed[::-1]) - 1)}")

    # Decode the top-1 token if a tokenizer is locally available.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
        print(f"\n[decode] Swift's nextID=5279 → '{tok.decode([5279])}'")
        print(f"[decode] Swift's CoreML top-1={top_k[0]} → '{tok.decode([top_k[0]])}'")
        print(f"[decode] HF expected '236761' → '{tok.decode([236761])}'")
    except Exception as e:
        print(f"[decode] tokenizer unavailable: {e}")


if __name__ == "__main__":
    main()
