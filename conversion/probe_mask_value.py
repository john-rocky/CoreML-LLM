#!/usr/bin/env python3
"""Probe whether ANE softmax handles mask=-Inf (0xFC00) differently from
mask=-1e4 (0xF0E2). Apple's ml-ane-transformers explicitly recommends -1e4
("recommended float value for preventing attention is -1e4… composition of
multiple masks while staying in the float16-friendly range") but our
production runtime fills with 0xFC00 = -Inf.

We don't compose masks in production (each layer takes one mask), so the
overflow argument doesn't directly apply — but ANE's softmax kernel may still
treat the values differently. This probe converts a tiny softmax-over-masked-
scores model and compares ANE outputs for the two mask fill values.

Mathematically identical results expected. Any measurable divergence means
ANE silicon distinguishes the two; that would justify swapping the runtime
constant to -1e4 to be safe.

Usage:
    python conversion/probe_mask_value.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct

ROOT = os.path.dirname(os.path.abspath(__file__))


class MaskedSoftmax(nn.Module):
    """softmax(scores + mask) — minimal model exposing the kernel under test.

    Shapes mirror our production single-token decode attention:
      scores: (1, num_heads=8, q=1, ctx=512)
      mask:   (1, 1,         q=1, ctx=512)
    """

    def forward(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.softmax(scores + mask, dim=-1)


def _build_coreml(num_heads: int, ctx: int, out_path: str) -> None:
    model = MaskedSoftmax().eval()
    sample = (
        torch.zeros(1, num_heads, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
    )
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="scores", shape=sample[0].shape, dtype=np.float16),
            ct.TensorType(name="mask",   shape=sample[1].shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="probs")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    if os.path.exists(out_path):
        import shutil
        shutil.rmtree(out_path)
    mlmodel.save(out_path)


def _make_mask(ctx: int, attended_len: int, fill_bits: int) -> np.ndarray:
    """Build (1,1,1,ctx) mask: 0 for first `attended_len`, fill_bits otherwise.

    fill_bits is a fp16 bit pattern (uint16). 0xFC00 = -Inf, 0xF0E2 = -1e4.
    """
    arr = np.zeros((1, 1, 1, ctx), dtype=np.uint16)
    arr[..., attended_len:] = fill_bits
    return arr.view(np.float16)


def _make_scores(num_heads: int, ctx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Magnitudes in [-10, 10] approximate real post-scale Q@K^T values.
    return rng.uniform(-10.0, 10.0, size=(1, num_heads, 1, ctx)).astype(np.float16)


def _run(model_path: str, scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import coremltools as ct  # local re-import for type narrowing
    m = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    out = m.predict({"scores": scores, "mask": mask})
    return out["probs"]


def _compare(label: str, a: np.ndarray, b: np.ndarray) -> None:
    diff = (a.astype(np.float32) - b.astype(np.float32))
    mask_zero = (np.abs(a) < 1e-30) & (np.abs(b) < 1e-30)
    nonzero_diff = np.abs(diff[~mask_zero])
    print(f"\n[{label}]  shape={a.shape}")
    print(f"  max|Δ|        : {np.abs(diff).max():.3e}")
    print(f"  mean|Δ|       : {np.abs(diff).mean():.3e}")
    print(f"  L∞ over nonzero positions: "
          f"{nonzero_diff.max() if nonzero_diff.size else 0:.3e}")
    print(f"  any NaN in A  : {np.isnan(a).any()}   any NaN in B: {np.isnan(b).any()}")
    # Top-1 probability mass agreement (we mostly care if the most-likely
    # position changes between mask values).
    a1 = a.argmax(axis=-1)
    b1 = b.argmax(axis=-1)
    print(f"  argmax agree  : {(a1 == b1).mean()*100:.1f}%   ({a1.size} positions)")


def main():
    NUM_HEADS = 8
    CTX = 512
    ATTENDED = 200  # first 200 positions unmasked, last 312 masked
    NEG_INF = 0xFC00
    NEG_1E4 = 0xF0E2

    tmp = tempfile.mkdtemp(prefix="mask_probe_")
    pkg = os.path.join(tmp, "masked_softmax.mlpackage")
    print(f"[build] converting MaskedSoftmax → {pkg}")
    t = time.time()
    _build_coreml(NUM_HEADS, CTX, pkg)
    print(f"[build] done in {time.time()-t:.1f}s")

    # Run several seeds so we average over score patterns.
    diffs = []
    for seed in range(5):
        scores = _make_scores(NUM_HEADS, CTX, seed)
        mask_inf = _make_mask(CTX, ATTENDED, NEG_INF)
        mask_1e4 = _make_mask(CTX, ATTENDED, NEG_1E4)

        out_inf = _run(pkg, scores, mask_inf)
        out_1e4 = _run(pkg, scores, mask_1e4)

        _compare(f"seed={seed}", out_inf, out_1e4)
        diffs.append(np.abs(out_inf.astype(np.float32) - out_1e4.astype(np.float32)).max())

    print(f"\n=== Summary across 5 seeds ===")
    print(f"  worst max|Δ|: {max(diffs):.3e}")
    print(f"  mean max|Δ|: {np.mean(diffs):.3e}")
    if max(diffs) < 1e-3:
        print(f"  → CoreML/ANE handles -Inf and -1e4 IDENTICALLY. Keep -Inf.")
    else:
        print(f"  → divergence detected. Apple's -1e4 recommendation is load-bearing.")


if __name__ == "__main__":
    main()
