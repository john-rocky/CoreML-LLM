#!/usr/bin/env python3
"""Mac CPU_AND_NE latency: chunk_1 W4 baseline vs W4A8 (Stage 1).

Tests whether `linear_quantize_activations` on top of W4 LUT yields
the expected memory-bandwidth halving on Apple ANE for the 8-layer
Gemma 4 stateful chunk_1 graph.

Both variants must already be built at:
  /tmp/g4_w4a8/w4_linear/chunk_1.mlpackage     (--linear-projections)
  /tmp/g4_w4a8/w4a8_linear/chunk_1.mlpackage   (--linear-projections --activation-quant)

Inputs are zeros — we measure dispatch latency, not output correctness.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct

ROOT = Path("/tmp/g4_w4a8")

HIDDEN = 1536
PLD = 256
NLAYERS = 35
HKV = 1
HD_S = 256
HD_F = 512
CTX = 512
W = 512


def _zeros(shape, dtype=np.float16):
    return np.zeros(shape, dtype=dtype)


def _make_inputs():
    return {
        "hidden_states":         _zeros((1, 1, HIDDEN)),
        "causal_mask_full":      _zeros((1, 1, 1, CTX)),
        "causal_mask_sliding":   _zeros((1, 1, 1, W)),
        "per_layer_raw":         _zeros((1, 1, NLAYERS * PLD)),
        "cos_s":                 _zeros((1, 1, 1, HD_S)),
        "sin_s":                 _zeros((1, 1, 1, HD_S)),
        "cos_f":                 _zeros((1, 1, 1, HD_F)),
        "sin_f":                 _zeros((1, 1, 1, HD_F)),
        "current_pos":           np.zeros((1,), dtype=np.int32),
        "ring_pos":              np.zeros((1,), dtype=np.int32),
    }


def measure(label: str, pkg_path: Path, iters: int = 20, warmup: int = 3):
    print(f"\n[{label}] {pkg_path}")
    if not pkg_path.is_dir():
        print(f"  missing: {pkg_path}")
        return None
    size_mb = sum(f.stat().st_size for f in pkg_path.rglob('*')
                  if f.is_file()) / 1024 / 1024
    print(f"  size: {size_mb:.1f} MB")
    t0 = time.time()
    m = ct.models.MLModel(str(pkg_path),
                          compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  load: {time.time()-t0:.1f}s")
    state = m.make_state()
    inputs = _make_inputs()
    for _ in range(warmup):
        m.predict(inputs, state=state)
    times = []
    for _ in range(iters):
        t = time.time()
        m.predict(inputs, state=state)
        times.append((time.time() - t) * 1000)
    arr = np.array(times)
    print(f"  iters={iters}  median={np.median(arr):.2f} ms  "
          f"mean={arr.mean():.2f}  min={arr.min():.2f}  max={arr.max():.2f}  "
          f"std={arr.std():.2f}")
    return arr, size_mb


def main():
    a = measure("A (W4 + Linear)",
                ROOT / "w4_linear" / "chunk_1.mlpackage")
    b = measure("B (W4A8 + Linear)",
                ROOT / "w4a8_linear" / "chunk_1.mlpackage")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if a is None or b is None:
        print("  one or both variants missing")
        return 1
    a_arr, a_size = a
    b_arr, b_size = b
    am, bm = float(np.median(a_arr)), float(np.median(b_arr))
    delta = (bm / am - 1.0) * 100.0
    print(f"  A median: {am:.2f} ms ({a_size:.1f} MB)")
    print(f"  B median: {bm:.2f} ms ({b_size:.1f} MB)")
    print(f"  delta:    {bm-am:+.2f} ms ({delta:+.1f}%)")
    print(f"  size delta: {b_size-a_size:+.1f} MB")
    print()
    if delta < -10:
        print("  W4A8 GO signal: > -10% Mac latency, "
              "memory-bandwidth halving consistent with theory.")
    elif delta < -3:
        print("  W4A8 weak signal: -3..-10% Mac latency. iPhone test required.")
    elif delta < 5:
        print("  parity. ANE shader chose same path; iPhone may differ.")
    else:
        print(f"  W4A8 slower by {delta:.1f}% on Mac. "
              "ANE may decompose to non-fused ops or calibration was off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
