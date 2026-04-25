#!/usr/bin/env python3
"""Mac CPU_AND_NE latency comparison: chunk_1 Conv2d vs nn.Linear, W4 LUT.

Tests whether the +21% Mac W4 latency gap MBA observed at 5-layer scale
(commit 72d30b3) reproduces at 8-layer chunk_1 scale on the real Gemma 4
graph (with attention, RoPE, per-layer embed, KV state writes).

Inputs are zeros (correctness doesn't matter — we just measure dispatch
latency through the full chunk graph). State is created fresh each variant.

Both variants must already be built at:
  /tmp/g4_chunk1_ab/conv/chunk_1.mlpackage      (default Conv2d)
  /tmp/g4_chunk1_ab/linear/chunk_1.mlpackage    (--linear-projections)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct

ROOT = Path("/tmp/g4_chunk1_ab")

# Match the build_gemma4_e2b_stateful_chunks.py chunk_1 input spec.
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
        print(f"  ❌ missing: {pkg_path}")
        return None
    t0 = time.time()
    m = ct.models.MLModel(str(pkg_path),
                          compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  load: {time.time()-t0:.1f}s")
    state = m.make_state()
    inputs = _make_inputs()
    # Warmup
    for _ in range(warmup):
        m.predict(inputs, state=state)
    # Time
    times = []
    for _ in range(iters):
        t = time.time()
        m.predict(inputs, state=state)
        times.append((time.time() - t) * 1000)
    arr = np.array(times)
    print(f"  iters={iters}  median={np.median(arr):.2f} ms  "
          f"mean={arr.mean():.2f}  min={arr.min():.2f}  max={arr.max():.2f}  "
          f"std={arr.std():.2f}")
    return arr


def main():
    a = measure("A (Conv2d-1×1 wrapper)",
                ROOT / "conv" / "chunk_1.mlpackage")
    b = measure("B (nn.Linear native)",
                ROOT / "linear" / "chunk_1.mlpackage")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if a is None or b is None:
        print("  ❌ one or both variants missing")
        return 1
    am, bm = float(np.median(a)), float(np.median(b))
    delta = (bm / am - 1.0) * 100.0
    print(f"  A median: {am:.2f} ms")
    print(f"  B median: {bm:.2f} ms")
    print(f"  delta:    {bm-am:+.2f} ms ({delta:+.1f}%)")
    print()
    if abs(delta) < 5:
        print("  → parity. MBA 5-layer +21% does NOT scale to chunk_1.")
        print("    Migration likely safe; iPhone validation still confirms.")
    elif delta < -5:
        print("  → B faster. Strong migration GO signal.")
    elif delta < 15:
        print("  → B slower by 5-15%. Mac quirk persists at scale but smaller.")
        print("    iPhone test gates production migration.")
    else:
        print("  → B slower by 15%+. Mac quirk is real and scales.")
        print("    iPhone may show same; migration HOLD probable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
