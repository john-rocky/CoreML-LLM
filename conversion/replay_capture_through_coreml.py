#!/usr/bin/env python3
"""Replay HF capture through the compiled CoreML drafter (.mlmodelc).

If our PyTorch port matches HF on real K/V (proven by
`replay_capture_through_port.py`), but the compiled CoreML mlpackage
differs, we know the bug is in build_mtp_drafter.py / coremltools
conversion / weights load.

If both port and CoreML match HF, the bug is in Swift wiring (mask
layout, KV cache positioning, RoPE table content, etc.).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import coremltools as ct


def _load_partial_rope(theta: float, head_dim: int, partial: float, max_pos: int) -> tuple[np.ndarray, np.ndarray]:
    rope_angles = int(partial * head_dim // 2)
    inv_rot = 1.0 / (theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float32) / head_dim))
    nope = head_dim // 2 - rope_angles
    inv_freq = np.concatenate([inv_rot, np.zeros(nope, dtype=np.float32)]) if nope > 0 else inv_rot
    t = np.arange(max_pos, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float16), np.sin(emb).astype(np.float16)


def _make_sliding_mask(W: int, valid: int) -> np.ndarray:
    """Right-aligned: last `valid` positions (slots W-valid..W-1) allowed."""
    mask = np.full((1, 1, 1, W), -65504.0, dtype=np.float16)
    start = W - valid
    mask[..., start:] = 0
    return mask


def _make_full_mask(C: int, valid: int) -> np.ndarray:
    """Left-aligned: first `valid` positions (slots 0..valid-1) allowed."""
    mask = np.full((1, 1, 1, C), -65504.0, dtype=np.float16)
    mask[..., :valid] = 0
    return mask


def main():
    cap = torch.load("output/mtp_probe/hf_capture.pt", map_location="cpu", weights_only=False)
    print(f"[coreml-replay] cap: input_len={cap['input_ids'].shape[1]} pos={int(cap['position_ids'][0,0])}")

    # Load CoreML model.
    pkg = "mtp_drafter_ctx2k.mlpackage"
    print(f"[coreml-replay] loading {pkg} ...", flush=True)
    model = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # Build inputs.
    pos = int(cap["position_ids"][0, 0].item())
    valid = pos + 1  # K cache contains positions 0..pos
    valid_seen = cap["sliding_k"].shape[2]  # actual ctx in capture
    print(f"[coreml-replay] valid (mask) = {valid}  valid_seen (cache) = {valid_seen}")

    # Sliding cache: drafter expects (1, 1, W=512, head_dim=256). Position
    # captured K from HF (length valid_seen) into the right-aligned tail.
    W = 512
    swa_hd = 256
    sk = np.zeros((1, 1, W, swa_hd), dtype=np.float16)
    sv = np.zeros((1, 1, swa_hd, W), dtype=np.float16)
    cap_sk = cap["sliding_k"].numpy().astype(np.float16)        # (1, 1, valid_seen, 256)
    cap_sv = cap["sliding_v"].numpy().astype(np.float16)        # (1, 1, valid_seen, 256)
    sk[:, :, W - valid_seen:, :] = cap_sk
    sv[:, :, :, W - valid_seen:] = cap_sv.transpose(0, 1, 3, 2)  # (1,1,256,valid_seen)
    print(f"[coreml-replay] sliding K right-aligned, valid_seen slots written")

    # Full cache: drafter expects (1, 1, C=2048, head_dim=512). Place positions 0..valid_seen-1 left-aligned.
    C = 2048
    full_hd = 512
    fk = np.zeros((1, 1, C, full_hd), dtype=np.float16)
    fv = np.zeros((1, 1, full_hd, C), dtype=np.float16)
    cap_fk = cap["full_k"].numpy().astype(np.float16)            # (1, 1, valid_seen, 512)
    cap_fv = cap["full_v"].numpy().astype(np.float16)            # (1, 1, valid_seen, 512)
    fk[:, :, :valid_seen, :] = cap_fk
    fv[:, :, :, :valid_seen] = cap_fv.transpose(0, 1, 3, 2)
    print(f"[coreml-replay] full K left-aligned, valid_seen slots written")

    # RoPE tables: partial_rotary 0.25 on full, full rotary on swa.
    cos_swa_tbl, sin_swa_tbl = _load_partial_rope(10000.0, 256, 1.0, max_pos=valid + 1)
    cos_full_tbl, sin_full_tbl = _load_partial_rope(1_000_000.0, 512, 0.25, max_pos=valid + 1)
    cos_swa = cos_swa_tbl[pos:pos + 1, :128]   # first half
    sin_swa = sin_swa_tbl[pos:pos + 1, :128]
    cos_full = cos_full_tbl[pos:pos + 1, :256]
    sin_full = sin_full_tbl[pos:pos + 1, :256]

    # Mask: drafter swa mask right-aligned, full mask left-aligned.
    mask_swa = _make_sliding_mask(W, valid_seen)
    mask_full = _make_full_mask(C, valid_seen)

    # Embed inputs.
    embed_token = cap["last_token_embedding"].numpy().astype(np.float16)   # (1, 1, 1536)
    proj_act = cap["last_hidden_state"].numpy().astype(np.float16)         # (1, 1, 1536)

    feed = {
        "embed_token": embed_token,
        "proj_act": proj_act,
        "kv13_k": sk,
        "kv13_v": sv,
        "kv14_k": fk,
        "kv14_v": fv,
        "cos_swa": cos_swa,
        "sin_swa": sin_swa,
        "cos_full": cos_full,
        "sin_full": sin_full,
        "mask_swa": mask_swa,
        "mask_full": mask_full,
    }
    print("[coreml-replay] running prediction ...", flush=True)
    out = model.predict(feed)
    print(f"[coreml-replay] outputs keys = {list(out.keys())}")
    top_k_indices = out["top_k_indices"] if "top_k_indices" in out else out.get("top_k_ids")
    top_k_values = out.get("top_k_values")
    if top_k_indices is not None:
        print(f"[coreml-replay] CoreML top-k indices = {top_k_indices.flatten().tolist()}")
        print(f"[coreml-replay] CoreML top-k values  = {top_k_values.flatten().tolist() if top_k_values is not None else 'n/a'}")
    hf_top1 = int(cap["drafter_token"][0, 0].item())
    print(f"[coreml-replay] HF top-1 = {hf_top1}")
    if top_k_indices is not None:
        match = int(top_k_indices.flatten()[0]) == hf_top1
        print(f"[coreml-replay] CoreML top-1 == HF top-1: {match}")


if __name__ == "__main__":
    main()
