"""Mac bench — Gemma 4 vision encoder: new ANE-targeted build vs legacy GPU build.

Runs `N` predictions on each model and reports median / p90 latencies.
Input data is zeros — enough to pull weights, kick off kernels, and
measure steady-state per-call cost. Parity vs HF is verified at
convert time (`cosine ≈ 1.0` for the ANE build); this bench is
strictly about latency.
"""
from __future__ import annotations

import argparse
import os
import statistics as stats
import time
from pathlib import Path

import coremltools as ct
import numpy as np


# Legacy vision.mlmodelc input shape: (1, 2520, 768) fp32 + (1, 2520, 2) int32
# New vision.ane.mlpackage input shape: (1, 2304, 768) fp16 + (1, 2304, 2) int32
LEGACY_PATCHES = 2520
ANE_PATCHES = 2304
PATCH_DIM = 768
GRID_SIDE = 48


def _pos_ids_square(num_patches: int, pad: int = 0) -> np.ndarray:
    arr = np.zeros((1, num_patches, 2), dtype=np.int32)
    k = 0
    for py in range(GRID_SIDE):
        for px in range(GRID_SIDE):
            arr[0, k, 0] = px
            arr[0, k, 1] = py
            k += 1
    if pad and k < num_patches:
        arr[0, k:] = -1
    return arr


def _bench(model: ct.models.MLModel, feed: dict, n: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        model.predict(feed)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(feed)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "n": n,
        "median_ms": stats.median(samples),
        "p90_ms": samples[int(0.9 * n)],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ane", required=True,
                    help="Path to vision.ane.mlpackage (the new ANE-targeted build)")
    ap.add_argument("--legacy",
                    help="Path to legacy vision.mlmodelc (GPU baseline). Optional.")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    # ---- ANE build ---------------------------------------------------------
    print("=" * 72)
    print(f"NEW  : {args.ane}")
    print("=" * 72)
    pv_ane = np.zeros((1, ANE_PATCHES, PATCH_DIM), dtype=np.float16)
    pid_ane = _pos_ids_square(ANE_PATCHES, pad=0)
    ane_feed = {"pixel_values": pv_ane, "pixel_position_ids": pid_ane}

    for units_name, units in [
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
    ]:
        print(f"\n-- compute_units={units_name} --")
        t0 = time.perf_counter()
        m = ct.models.MLModel(args.ane, compute_units=units)
        print(f"  load: {(time.perf_counter() - t0) * 1000:.1f} ms")
        r = _bench(m, ane_feed, n=args.iters)
        print(f"  predict x{r['n']}: median={r['median_ms']:.1f} ms  "
              f"p90={r['p90_ms']:.1f}  min={r['min_ms']:.1f}  max={r['max_ms']:.1f}")
        del m

    # ---- Legacy build (optional) ------------------------------------------
    if args.legacy and os.path.exists(args.legacy):
        print("\n" + "=" * 72)
        print(f"LEGACY: {args.legacy}")
        print("=" * 72)
        # Legacy model expects fp32 pixel values and (1, 2520, 768) shape.
        pv_leg = np.zeros((1, LEGACY_PATCHES, PATCH_DIM), dtype=np.float32)
        pid_leg = _pos_ids_square(LEGACY_PATCHES, pad=1)
        leg_feed = {"pixel_values": pv_leg, "pixel_position_ids": pid_leg}

        for units_name, units in [
            ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
            ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ]:
            print(f"\n-- compute_units={units_name} --")
            t0 = time.perf_counter()
            m = ct.models.MLModel(args.legacy, compute_units=units)
            print(f"  load: {(time.perf_counter() - t0) * 1000:.1f} ms")
            r = _bench(m, leg_feed, n=args.iters)
            print(f"  predict x{r['n']}: median={r['median_ms']:.1f} ms  "
                  f"p90={r['p90_ms']:.1f}  min={r['min_ms']:.1f}  max={r['max_ms']:.1f}")
            del m


if __name__ == "__main__":
    main()
