#!/usr/bin/env python3
"""Benchmark T=1 decode through the 1-chunk all-in-one stateful model.

Single mlpackage: L0-34 + lm_head + argmax in one forward.

Usage:
    python conversion/benchmark_stateful_1chunk.py \
        --pkg /tmp/g4_1chunk/model.mlpackage --ctx 2048 --steps 30
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import coremltools as ct


def _z(shape):
    return np.zeros(shape, dtype=np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--function", default="infer")
    args = ap.parse_args()

    cu = ct.ComputeUnit.CPU_ONLY if args.cpu_only else ct.ComputeUnit.CPU_AND_NE
    ctx = args.ctx
    W = 512
    hidden = 1536
    nlayers = 35
    pld = 256
    total_pld = nlayers * pld
    hd_s = 256
    hd_f = 512

    print(f"\n{'='*60}")
    print(f"1-chunk stateful benchmark: {os.path.basename(args.pkg)}, "
          f"ctx={ctx}, fn={args.function}")
    print(f"Compute units: {cu}")
    print(f"{'='*60}")

    print("Loading model...")
    t0 = time.time()
    m = ct.models.MLModel(args.pkg, compute_units=cu, function_name=args.function)
    print(f"  loaded in {time.time()-t0:.1f}s")

    state = m.make_state()

    h_in = _z((1, 1, hidden))
    plr = _z((1, 1, total_pld))
    cos_s = _z((1, 1, 1, hd_s))
    sin_s = _z((1, 1, 1, hd_s))
    cos_f = _z((1, 1, 1, hd_f))
    sin_f = _z((1, 1, 1, hd_f))

    times = []
    print(f"\nRunning {args.steps} decode steps...")
    for step in range(args.steps):
        pos = step
        ring = pos % W
        mask_full = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float16)
        mask_full[0, 0, 0, :pos+1] = 0
        mask_sliding = np.full((1, 1, 1, W), -65504.0, dtype=np.float16)
        valid = min(pos + 1, W)
        mask_sliding[0, 0, 0, :valid] = 0
        cur_pos = np.array([pos], dtype=np.int32)
        ring_pos = np.array([ring], dtype=np.int32)

        t0 = time.time()
        _ = m.predict({
            "hidden_states": h_in,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "per_layer_raw": plr,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "current_pos": cur_pos, "ring_pos": ring_pos,
        }, state=state)
        dt = time.time() - t0
        times.append(dt)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"  Step {step+1}: {dt*1000:.1f}ms")

    skip = 5
    arr = times[skip:]
    mean, std, mn, mx = np.mean(arr)*1000, np.std(arr)*1000, np.min(arr)*1000, np.max(arr)*1000
    print(f"\n  TOTAL: {mean:.1f}ms ±{std:.1f} (min={mn:.1f}, max={mx:.1f})")
    print(f"  Throughput: {1000/mean:.1f} tok/s")


if __name__ == "__main__":
    main()
