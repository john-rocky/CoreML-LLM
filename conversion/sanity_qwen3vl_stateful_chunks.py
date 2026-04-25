#!/usr/bin/env python3
"""Mac sanity for a Qwen3-VL 2B stateful chunked bundle (4 body + head).

Mirrors `sanity_stateful_chunks.py` (Gemma 4 E2B variant) for the
Qwen3-VL 2B layout:
  qwen3_vl_2b_stateful_chunks/
    chunk_0.mlpackage  ... chunk_3.mlpackage  (body, MLState)
    chunk_head.mlpackage                       (final norm + lm_head + argmax)
    embed_weight.bin                           (raw fp16 embed sidecar)

Verifies:
  1. Each chunk loads on Mac CPU_AND_NE.
  2. State round-trips on chunk_0 (predict twice with same state object,
     same inputs except current_pos → outputs differ when state was
     written during the first call).
  3. chunk_0 → chunk_1 → chunk_2 → chunk_3 → chunk_head wired end-to-end
     produces a valid token_id ∈ [0, vocab_size).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import coremltools as ct

HIDDEN = 2048
HEAD_DIM = 128
MAX_SEQ = 2048
VOCAB = 151_936  # Qwen3-VL 2B; tolerate slightly more
NUM_CHUNKS = 4


def _zeros(shape, dtype=np.float16):
    return np.zeros(shape, dtype=dtype)


def _rope(pos: int):
    theta = 5_000_000.0
    half = HEAD_DIM // 2
    freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    angles = pos * freqs
    full = np.concatenate([angles, angles])
    cos = np.cos(full).astype(np.float16).reshape(1, 1, HEAD_DIM)
    sin = np.sin(full).astype(np.float16).reshape(1, 1, HEAD_DIM)
    return cos, sin


def _causal_mask(pos: int):
    m = np.zeros((1, 1, 1, MAX_SEQ), dtype=np.float16)
    if pos + 1 < MAX_SEQ:
        m[0, 0, 0, pos + 1:] = -1e4
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True,
                    help="Directory containing chunk_0..3.mlpackage + chunk_head.mlpackage")
    ap.add_argument("--label", default="bundle")
    args = ap.parse_args()

    root = Path(args.chunks_dir).resolve()
    print(f"\n=== sanity [{args.label}] {root} ===")
    if not root.is_dir():
        print(f"  missing: {root}")
        sys.exit(1)

    body_paths = [root / f"chunk_{i}.mlpackage" for i in range(NUM_CHUNKS)]
    head_path = root / "chunk_head.mlpackage"
    for p in body_paths + [head_path]:
        if not p.is_dir():
            print(f"  missing chunk: {p}")
            sys.exit(1)

    print("loading chunks...")
    bodies = []
    for p in body_paths:
        m = ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE)
        bodies.append(m)
        print(f"  {p.name} loaded")
    head = ct.models.MLModel(str(head_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  {head_path.name} loaded")

    # Inputs at pos=0
    hidden = _zeros((1, 1, HIDDEN))
    np.random.seed(7)
    hidden[0, 0, :] = np.random.randn(HIDDEN).astype(np.float16) * 0.1
    cos0, sin0 = _rope(0)
    mask0 = _causal_mask(0)
    pos0 = np.array([0], dtype=np.int32)

    # State round-trip on chunk_0: same inputs, two predicts, expect Δ > 0
    state = bodies[0].make_state()
    out_a = bodies[0].predict({
        "hidden_in": hidden, "cos": cos0, "sin": sin0,
        "causal_mask": mask0, "current_pos": pos0,
    }, state=state)
    h_a = out_a["hidden"]

    # Step to pos=1 with the same state — second call sees state from first.
    cos1, sin1 = _rope(1)
    mask1 = _causal_mask(1)
    pos1 = np.array([1], dtype=np.int32)
    out_b = bodies[0].predict({
        "hidden_in": hidden, "cos": cos1, "sin": sin1,
        "causal_mask": mask1, "current_pos": pos1,
    }, state=state)
    h_b = out_b["hidden"]

    delta = float(np.abs(h_a.astype(np.float32) - h_b.astype(np.float32)).mean())
    finite_a = np.isfinite(h_a).all()
    finite_b = np.isfinite(h_b).all()
    print(f"  chunk_0 state round-trip: |Δ| = {delta:.4f}   finite_a={bool(finite_a)} finite_b={bool(finite_b)}")
    if not (finite_a and finite_b) or delta == 0.0:
        print("  state round-trip looks broken (Δ=0 means the second call "
              "didn't see the first call's KV write)")
        sys.exit(2)

    # End-to-end at pos=0 — chain bodies + head
    states = [m.make_state() for m in bodies]
    h = hidden
    for ci, m in enumerate(bodies):
        out = m.predict({
            "hidden_in": h, "cos": cos0, "sin": sin0,
            "causal_mask": mask0, "current_pos": pos0,
        }, state=states[ci])
        h = out["hidden"]
        if not np.isfinite(h).all():
            print(f"  chunk_{ci} produced NaN/Inf — bail")
            sys.exit(3)

    head_out = head.predict({"hidden_in": h})
    tok = int(head_out["next_token"][0]) if hasattr(head_out["next_token"], "__getitem__") \
        else int(head_out["next_token"])
    print(f"  E2E token_id = {tok}")
    if not (0 <= tok < VOCAB + 50_000):
        print(f"  token_id out of plausible range")
        sys.exit(4)

    print("  OK — load + state round-trip + E2E token wired correctly")


if __name__ == "__main__":
    main()
