#!/usr/bin/env python3
"""EAGLE-3 e_next embed parity diagnosis (Track 7).

Compares the fp16 token embedding table the TRAINER uses (tied to `lm_head`,
see `train_eagle3_ttt.py:282`) against the int8-dequantized table the
DEPLOYED draft sees on-device (via `EmbeddingLookup.swift:35`).

The draft's `e_next` input at inference is produced by `ChunkedEngine.
embedToken()` (ChunkedEngine.swift:1411), which calls `embedTokens.lookup`,
which applies:

    fp16_device[i] = int8[tok, i] * (fp16_scale[tok] / 127.0) * embedScale

At training time the same tensor is produced by direct indexing of the
fp16 lm_head buffer multiplied by `embed_scale` (train_eagle3_ttt.py:374).
If the two are not numerically equivalent the draft's attention runs on
OOD inputs at inference, degrading accept rate.

Usage:
    python conversion/diagnose_embed_parity.py \
        --int8      ~/Downloads/gemma4-e2b-eagle3-sideload/embed_tokens_q8.bin \
        --scales    ~/Downloads/gemma4-e2b-eagle3-sideload/embed_tokens_scales.bin \
        --fp16      ~/Downloads/lm_head_weight.bin \
        --vocab 262144 --hidden 1536 --embed-scale 39.191835884530846 \
        --n-sample 10000 --seed 0
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np


def dequantize_rows(int8_path: Path, scales_path: Path, rows: np.ndarray,
                   hidden: int, embed_scale: float) -> np.ndarray:
    """Reproduce `EmbeddingLookup.lookup` in NumPy for the given token rows.

    Returns (len(rows), hidden) float32 tensor that the on-device draft
    would see as its `e_next` input.
    """
    # int8 table shape: (vocab, hidden); memmap to avoid loading 402 MB.
    q8 = np.memmap(int8_path, dtype=np.int8, mode="r").reshape(-1, hidden)
    # fp16 scales, one per row. Size must match int8 rows.
    scales = np.fromfile(scales_path, dtype=np.float16)
    if q8.shape[0] != scales.shape[0]:
        raise ValueError(
            f"row count mismatch: int8={q8.shape[0]} scales={scales.shape[0]}")
    rowq = np.asarray(q8[rows], dtype=np.float32)           # (N, hidden)
    rows_scale = scales[rows].astype(np.float32) / 127.0    # (N,)
    out = rowq * rows_scale[:, None] * float(embed_scale)   # (N, hidden)
    return out


def lm_head_rows(fp16_path: Path, rows: np.ndarray, vocab: int, hidden: int,
                 embed_scale: float) -> np.ndarray:
    """Reproduce trainer's e_next: `embed_table[tok] * embed_scale`.

    The trainer uses `embed_table = lm_head_t` (train_eagle3_ttt.py:282)
    and multiplies by `embed_scale` (train_eagle3_ttt.py:374-375).
    """
    size = os.path.getsize(fp16_path)
    expected = vocab * hidden * 2  # fp16
    if size != expected:
        raise ValueError(
            f"lm_head size mismatch: got {size}, expected {expected} "
            f"(vocab={vocab} hidden={hidden} fp16)")
    tbl = np.memmap(fp16_path, dtype=np.float16, mode="r").reshape(vocab, hidden)
    out = np.asarray(tbl[rows], dtype=np.float32) * float(embed_scale)
    return out


def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity, fp32."""
    an = np.linalg.norm(a, axis=-1) + 1e-30
    bn = np.linalg.norm(b, axis=-1) + 1e-30
    return (a * b).sum(axis=-1) / (an * bn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", type=Path, required=True)
    ap.add_argument("--scales", type=Path, required=True)
    ap.add_argument("--fp16", type=Path, required=True,
                    help="fp16 lm_head weight dump (vocab, hidden). Trainer "
                         "uses this as embed_table via tied embeddings.")
    ap.add_argument("--vocab", type=int, default=262144)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--embed-scale", type=float, default=39.191835884530846)
    ap.add_argument("--n-sample", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = rng.choice(args.vocab, size=args.n_sample, replace=False)
    rows.sort()

    print(f"[parity] sampling {len(rows)} tokens from vocab={args.vocab}")
    print(f"[parity] int8 table : {args.int8}")
    print(f"[parity] scales     : {args.scales}")
    print(f"[parity] fp16 table : {args.fp16}")
    print(f"[parity] embed_scale: {args.embed_scale:.6f}")

    e_device = dequantize_rows(args.int8, args.scales, rows,
                               args.hidden, args.embed_scale)
    e_train = lm_head_rows(args.fp16, rows, args.vocab, args.hidden,
                           args.embed_scale)

    cs = cos_sim(e_train, e_device)
    abs_err = np.abs(e_train - e_device)
    per_row_max = abs_err.max(axis=-1)
    per_row_mean = abs_err.mean(axis=-1)

    n_train = np.linalg.norm(e_train, axis=-1)
    n_device = np.linalg.norm(e_device, axis=-1)
    rel_err = (per_row_max / (n_train + 1e-30))

    print()
    print("=" * 60)
    print(f"{'metric':<28} {'mean':>12} {'min':>12} {'max':>12}")
    print("-" * 60)
    print(f"{'cos_sim(train,device)':<28} {cs.mean():>12.6f} "
          f"{cs.min():>12.6f} {cs.max():>12.6f}")
    print(f"{'abs_err per element':<28} {per_row_mean.mean():>12.4e} "
          f"{per_row_mean.min():>12.4e} {per_row_mean.max():>12.4e}")
    print(f"{'abs_err per row (max)':<28} {per_row_max.mean():>12.4e} "
          f"{per_row_max.min():>12.4e} {per_row_max.max():>12.4e}")
    print(f"{'rel_err (row_max/|train|)':<28} {rel_err.mean():>12.4e} "
          f"{rel_err.min():>12.4e} {rel_err.max():>12.4e}")
    print(f"{'|e_train|':<28} {n_train.mean():>12.3f} "
          f"{n_train.min():>12.3f} {n_train.max():>12.3f}")
    print(f"{'|e_device|':<28} {n_device.mean():>12.3f} "
          f"{n_device.min():>12.3f} {n_device.max():>12.3f}")
    print("=" * 60)

    # Flag tokens where cos_sim drops. Int8 rounding typically gives
    # cos ≥ 0.9999; anything below 0.998 points at a pathological row.
    low = cs < 0.998
    print(f"\n[parity] rows with cos_sim < 0.998: {int(low.sum())} / {len(rows)}")
    if low.any():
        idx = np.argsort(cs)[:10]
        print("[parity] worst 10 rows (token_id, cos_sim, row_max_abs):")
        for i in idx:
            print(f"  tok={int(rows[i]):>6d}  cos={cs[i]:.6f}  "
                  f"max_abs={per_row_max[i]:.4e}  |train|={n_train[i]:.2f}")

    # Verdict heuristic: mean cos ≥ 0.9995 and max rel err ≤ 1% → no drift.
    if cs.mean() >= 0.9995 and rel_err.max() <= 0.01:
        verdict = "NO DRIFT (int8 quantization is within fp16 noise)"
    elif cs.mean() >= 0.999:
        verdict = "BORDERLINE (quantization noise measurable; unlikely to drop accept)"
    else:
        verdict = "DRIFT (train↔inference e_next mismatch — fix trainer)"
    print(f"\n[parity] verdict: {verdict}")


if __name__ == "__main__":
    main()
