#!/usr/bin/env python3
"""Mac CPU_AND_NE latency comparison for Qwen3-VL 2B chunk_0:
Conv2dLinear (default) vs nn.Linear (--linear-projections), INT8.

Mirrors `probe_chunk1_linear_w4_latency.py` for the Qwen3-VL 2B
stateful chunk shape. Both bundles must already be built at:

  /tmp/q3vl_linear_test/conv/qwen3_vl_2b_stateful_chunks/chunk_0.mlpackage
  /tmp/q3vl_linear_test/linear/qwen3_vl_2b_stateful_chunks/chunk_0.mlpackage

Inputs are zeros — we only measure dispatch latency through the chunk
graph. State is created fresh per variant.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct

ROOT = Path("/tmp/q3vl_linear_test")

# Match Qwen3-VL 2B stateful chunk_0 input spec.
HIDDEN = 2048
HEAD_DIM = 128
MAX_SEQ = 2048


def _zeros(shape, dtype=np.float16):
    return np.zeros(shape, dtype=dtype)


def _make_inputs():
    return {
        "hidden_in":   _zeros((1, 1, HIDDEN)),
        "cos":         _zeros((1, 1, HEAD_DIM)),
        "sin":         _zeros((1, 1, HEAD_DIM)),
        "causal_mask": _zeros((1, 1, 1, MAX_SEQ)),
        "current_pos": np.zeros((1,), dtype=np.int32),
    }


def measure(label: str, pkg_path: Path, iters: int = 20, warmup: int = 3):
    print(f"\n[{label}] {pkg_path}")
    if not pkg_path.is_dir():
        print(f"  missing: {pkg_path}")
        return None
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
    return float(np.median(arr))


def main():
    conv_path = ROOT / "conv" / "qwen3_vl_2b_stateful_chunks" / "chunk_0.mlpackage"
    linear_path = ROOT / "linear" / "qwen3_vl_2b_stateful_chunks" / "chunk_0.mlpackage"

    conv_med = measure("Conv2dLinear", conv_path)
    linear_med = measure("nn.Linear", linear_path)

    if conv_med is not None and linear_med is not None:
        delta = (linear_med - conv_med) / conv_med * 100
        print(f"\n  Conv2d:  {conv_med:.2f} ms")
        print(f"  Linear:  {linear_med:.2f} ms")
        print(f"  Δ:       {delta:+.1f} %  ({'Linear faster' if delta < 0 else 'Linear slower'})")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
