#!/usr/bin/env python3.12
"""Probe INT4 quantization quality on the Gemma 4 E2B per-layer embedding (PLE).

Compares the original BF16 PLE matrix against:
  1. Production INT8 row-wise (the on-disk format we currently ship).
  2. Synthetic INT8 row-wise (sanity check the methodology vs production).
  3. INT4 row-wise symmetric.
  4. INT4 grouped symmetric (g=32, g=64, g=128).

Reports per-row cosine similarity stats vs both BF16 (ground truth) and the
production INT8 (the baseline we'd be replacing).

Usage:
    python3.12 scripts/probe_ple_int4.py

No deps beyond torch / safetensors / numpy. Mac-side only; ~5 min on
Mac Studio (M2 Ultra). Peak RAM ~6 GB.
"""
from __future__ import annotations

import os
import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


HF_MODEL = "/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/output/gemma4-e2b/hf_model/model.safetensors"
PROD_INT8 = "/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/build/gemma4_stateful_ab/linear/gemma4_e2b_stateful_chunks/embed_tokens_per_layer_q8.bin"
PROD_SCALES = "/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/build/gemma4_stateful_ab/linear/gemma4_e2b_stateful_chunks/embed_tokens_per_layer_scales.bin"
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"


def load_bf16_ple() -> np.ndarray:
    """Load the BF16 PLE table from safetensors → fp32 numpy (rows × cols)."""
    print(f"loading BF16 PLE from safetensors ...")
    t0 = time.time()
    with safe_open(HF_MODEL, framework="pt") as f:
        t = f.get_tensor(PLE_KEY)  # bf16 [262144, 8960]
    arr = t.to(torch.float32).numpy()
    print(f"  shape={arr.shape}, dtype={arr.dtype}, took {time.time()-t0:.2f}s")
    return arr


def load_prod_int8() -> tuple[np.ndarray, np.ndarray]:
    """Load production INT8 + per-row FP16 scales."""
    print(f"loading production INT8 PLE from disk ...")
    raw = np.fromfile(PROD_INT8, dtype=np.int8)
    scales = np.fromfile(PROD_SCALES, dtype=np.float16)
    n_rows = scales.shape[0]
    n_cols = raw.shape[0] // n_rows
    int8 = raw.reshape(n_rows, n_cols)
    print(f"  int8 shape={int8.shape}, scales shape={scales.shape}")
    return int8, scales


def dequant_prod(int8: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Match EmbeddingLookup.swift: out = int8 × (scale_fp16 / 127.0) × 1.0."""
    s = (scales.astype(np.float32) / 127.0)[:, None]
    return int8.astype(np.float32) * s


def quant_row_int(x: np.ndarray, n_bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row symmetric quantization to n_bits.

    Returns (q, scale_per_row, dequantized).
    """
    qmax = (1 << (n_bits - 1)) - 1  # 127 for INT8, 7 for INT4
    absmax = np.max(np.abs(x), axis=1)  # per-row
    # Avoid div-by-zero
    absmax_safe = np.where(absmax > 0, absmax, 1.0)
    scale = absmax_safe / qmax  # per-row scale
    q = np.round(x / scale[:, None]).clip(-qmax, qmax).astype(np.int32)
    deq = q.astype(np.float32) * scale[:, None]
    return q, scale, deq


def quant_grouped_int(x: np.ndarray, n_bits: int, group_size: int) -> np.ndarray:
    """Per-group symmetric quantization → dequantized fp32.

    Quantizes each row in groups of `group_size` columns, with a per-group
    scale. Returns the dequantized matrix.
    """
    n_rows, n_cols = x.shape
    assert n_cols % group_size == 0, f"cols {n_cols} not divisible by group {group_size}"
    qmax = (1 << (n_bits - 1)) - 1
    n_groups = n_cols // group_size
    x_g = x.reshape(n_rows, n_groups, group_size)
    absmax = np.max(np.abs(x_g), axis=2, keepdims=True)
    absmax_safe = np.where(absmax > 0, absmax, 1.0)
    scale = absmax_safe / qmax
    q = np.round(x_g / scale).clip(-qmax, qmax).astype(np.int32)
    deq = (q.astype(np.float32) * scale).reshape(n_rows, n_cols)
    return deq


def cos_sim_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity per row, shape (n_rows,)."""
    # batched dot for memory efficiency
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    chunk = 16384
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        ai = a[i:j].astype(np.float64)
        bi = b[i:j].astype(np.float64)
        num = np.einsum("ij,ij->i", ai, bi)
        den = np.linalg.norm(ai, axis=1) * np.linalg.norm(bi, axis=1)
        # Both could be zero for unused vocab rows → cos = 1.0 (consistent zero)
        cs = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
        out[i:j] = cs
    return out


def stats(name: str, cs: np.ndarray):
    pcts = [50, 90, 99, 99.9]
    pct_vals = np.percentile(cs, pcts)
    print(f"  {name:35s}  "
          f"mean={cs.mean():.6f}  min={cs.min():.6f}  "
          f"p50={pct_vals[0]:.6f}  p99={pct_vals[2]:.6f}  p99.9={pct_vals[3]:.6f}")


def main():
    fp32 = load_bf16_ple()  # [262144, 8960]
    prod_int8, prod_scales = load_prod_int8()
    prod_deq = dequant_prod(prod_int8, prod_scales)
    print(f"  prod_deq shape={prod_deq.shape}, dtype={prod_deq.dtype}")
    print()

    # 1) Production INT8 vs BF16 (ground truth) — the *baseline* we'd be replacing
    print("=== BASELINE (production INT8 on disk) ===")
    cs_prod_vs_bf16 = cos_sim_per_row(prod_deq, fp32)
    stats("prod INT8 vs BF16", cs_prod_vs_bf16)

    # 2) Synthetic INT8 row-wise (methodology sanity check)
    print()
    print("=== SYNTHETIC RECIPES vs BF16 (the ground truth) ===")
    _, _, deq_int8_row = quant_row_int(fp32, 8)
    cs_int8_row = cos_sim_per_row(deq_int8_row, fp32)
    stats("synth INT8 row-wise vs BF16", cs_int8_row)

    # 3) INT4 row-wise (the simplest replacement)
    _, _, deq_int4_row = quant_row_int(fp32, 4)
    cs_int4_row = cos_sim_per_row(deq_int4_row, fp32)
    stats("INT4 row-wise vs BF16", cs_int4_row)

    # 4) INT4 grouped (g=128, g=64, g=32)
    for g in (128, 64, 32):
        deq_int4_g = quant_grouped_int(fp32, 4, g)
        cs_int4_g = cos_sim_per_row(deq_int4_g, fp32)
        stats(f"INT4 group={g} vs BF16", cs_int4_g)
        del deq_int4_g

    # 5) Compare INT4 variants against production INT8 (the deployable baseline)
    print()
    print("=== INT4 RECIPES vs PROD INT8 (deployable baseline) ===")
    cs_vs_prod = cos_sim_per_row(deq_int4_row, prod_deq)
    stats("INT4 row-wise vs prod INT8", cs_vs_prod)
    for g in (128, 64, 32):
        deq_int4_g = quant_grouped_int(fp32, 4, g)
        cs_int4_g_vs_prod = cos_sim_per_row(deq_int4_g, prod_deq)
        stats(f"INT4 group={g} vs prod INT8", cs_int4_g_vs_prod)
        del deq_int4_g

    # 6) Storage size comparison (excluding scales)
    print()
    print("=== ON-DISK STORAGE (rows × cols × bytes_per_value) ===")
    n_rows, n_cols = fp32.shape
    size_mb = lambda b: b / (1 << 20)
    bf16_bytes = n_rows * n_cols * 2
    int8_data_bytes = n_rows * n_cols * 1
    int8_scales_bytes = n_rows * 2  # fp16 row scale
    int4_data_bytes = n_rows * n_cols // 2  # 4 bits per value
    int4_row_scales_bytes = n_rows * 2  # fp16 row scale
    int4_g128_scales = n_rows * (n_cols // 128) * 2
    int4_g64_scales = n_rows * (n_cols // 64) * 2
    int4_g32_scales = n_rows * (n_cols // 32) * 2

    print(f"  BF16 reference         : {size_mb(bf16_bytes):>7.1f} MB")
    print(f"  prod INT8 + row scales : {size_mb(int8_data_bytes + int8_scales_bytes):>7.1f} MB  "
          f"(saves {size_mb(bf16_bytes - int8_data_bytes - int8_scales_bytes):.1f} MB vs BF16)")
    print(f"  INT4 row + row scales  : {size_mb(int4_data_bytes + int4_row_scales_bytes):>7.1f} MB  "
          f"(saves {size_mb(int8_data_bytes + int8_scales_bytes - int4_data_bytes - int4_row_scales_bytes):.1f} MB vs INT8)")
    for g, sc in [(128, int4_g128_scales), (64, int4_g64_scales), (32, int4_g32_scales)]:
        total = int4_data_bytes + sc
        save = (int8_data_bytes + int8_scales_bytes) - total
        print(f"  INT4 group={g:3d} + scales : {size_mb(total):>7.1f} MB  "
              f"(saves {size_mb(save):.1f} MB vs INT8)")


if __name__ == "__main__":
    main()
