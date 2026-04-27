#!/usr/bin/env python3
"""Mac probe: chunk_1 sparse-vs-W4 baseline latency + cos sim.

Loads two chunk_1.mlpackage variants (e.g. W4 LUT only and N:M sparse +
W4 LUT joint), runs identical T=1 decode steps on both, and reports:
  - mean / p50 / p95 latency per step on CPU_AND_NE
  - cos sim and max-abs diff on hidden_states_out + per_layer_combined_out
  - bundle size on disk

Usage:
    python conversion/probe_sparse_chunk1.py \
        --baseline /tmp/g4_3chunk_sparse_probe/baseline_w4/chunk_1.mlpackage \
        --variant  /tmp/g4_3chunk_sparse_probe/sparse_w4/chunk_1.mlpackage \
        --steps 30 --warmup 3
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import numpy as np
import coremltools as ct


CTX = 2048
W = 512
HIDDEN = 1536
NLAYERS = 35
PLD = 256
HD_S = 256
HD_F = 512


def _bundle_size_mb(p: str) -> float:
    p = Path(p)
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def _make_inputs(pos: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((1, 1, HIDDEN), dtype=np.float32).astype(np.float16) * 0.1
    plr = rng.standard_normal((1, 1, NLAYERS * PLD), dtype=np.float32).astype(np.float16) * 0.1
    cos_s = rng.uniform(-1, 1, (1, 1, 1, HD_S)).astype(np.float16)
    sin_s = rng.uniform(-1, 1, (1, 1, 1, HD_S)).astype(np.float16)
    cos_f = rng.uniform(-1, 1, (1, 1, 1, HD_F)).astype(np.float16)
    sin_f = rng.uniform(-1, 1, (1, 1, 1, HD_F)).astype(np.float16)

    mask_full = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask_full[0, 0, 0, :pos + 1] = 0
    mask_sliding = np.full((1, 1, 1, W), -65504.0, dtype=np.float16)
    mask_sliding[0, 0, 0, :min(pos + 1, W)] = 0

    return {
        "hidden_states": h,
        "causal_mask_full": mask_full,
        "causal_mask_sliding": mask_sliding,
        "per_layer_raw": plr,
        "cos_s": cos_s, "sin_s": sin_s,
        "cos_f": cos_f, "sin_f": sin_f,
        "current_pos": np.array([pos], dtype=np.int32),
        "ring_pos":    np.array([pos % W], dtype=np.int32),
    }


def _bench(pkg_path: str, steps: int, warmup: int):
    print(f"\n=== {os.path.basename(os.path.dirname(pkg_path))} ===")
    print(f"  size: {_bundle_size_mb(pkg_path):.1f} MB")
    t0 = time.time()
    m = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  loaded in {time.time()-t0:.1f}s")
    state = m.make_state()

    times = []
    last_out = None
    for step in range(warmup + steps):
        feats = _make_inputs(step)
        t0 = time.time()
        out = m.predict(feats, state=state)
        dt = (time.time() - t0) * 1000
        if step >= warmup:
            times.append(dt)
        last_out = out

    times = np.asarray(times)
    print(f"  steps={steps}  mean={times.mean():.2f} ms  "
          f"p50={np.percentile(times, 50):.2f}  "
          f"p95={np.percentile(times, 95):.2f}")
    return last_out, times


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="W4 LUT only chunk_1.mlpackage")
    ap.add_argument("--variant",  required=True,
                    help="Joint sparse + palettized chunk_1.mlpackage")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    out_b, t_b = _bench(args.baseline, args.steps, args.warmup)
    out_v, t_v = _bench(args.variant,  args.steps, args.warmup)

    print("\n=== latency delta ===")
    print(f"  baseline mean: {t_b.mean():.2f} ms")
    print(f"  variant  mean: {t_v.mean():.2f} ms")
    delta = (t_v.mean() / t_b.mean() - 1.0) * 100
    print(f"  variant / baseline: {t_v.mean() / t_b.mean():.3f}x  ({delta:+.1f}%)")

    print("\n=== output parity ===")
    for name in ["hidden_states_out", "per_layer_combined_out"]:
        if name in out_b and name in out_v:
            a = out_b[name]; b = out_v[name]
            cs = _cos_sim(a, b)
            md = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
            print(f"  {name:30s} cos={cs:.6f}  max_abs_diff={md:.4e}")


if __name__ == "__main__":
    main()
