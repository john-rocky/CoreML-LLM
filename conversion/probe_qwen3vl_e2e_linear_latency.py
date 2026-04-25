#!/usr/bin/env python3
"""Mac CPU_AND_NE latency comparison: full Qwen3-VL 2B 4-chunk + head
Conv2dLinear vs nn.Linear bundles, INT8.

Times one decode step end-to-end (chunk_0..3 + head) so we capture the
real per-token cost. Also reports per-component breakdown so we can
attribute any Linear-vs-Conv delta to a specific chunk.

Bundles must already be built at:
  /tmp/q3vl_conv_full/qwen3_vl_2b_stateful_chunks/
  /tmp/q3vl_linear_full/qwen3_vl_2b_stateful_chunks/
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct

HIDDEN = 2048
HEAD_DIM = 128
MAX_SEQ = 2048


def _zeros(shape, dtype=np.float16):
    return np.zeros(shape, dtype=dtype)


def measure_bundle(label: str, root: Path, iters: int = 30, warmup: int = 5):
    print(f"\n[{label}] {root}")
    body_paths = [root / f"chunk_{i}.mlpackage" for i in range(4)]
    head_path = root / "chunk_head.mlpackage"
    for p in body_paths + [head_path]:
        if not p.is_dir():
            print(f"  missing: {p}")
            return None

    bodies = []
    for p in body_paths:
        m = ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE)
        bodies.append(m)
    head = ct.models.MLModel(str(head_path), compute_units=ct.ComputeUnit.CPU_AND_NE)

    states = [m.make_state() for m in bodies]

    hidden = _zeros((1, 1, HIDDEN))
    cos = _zeros((1, 1, HEAD_DIM))
    sin = _zeros((1, 1, HEAD_DIM))
    mask = _zeros((1, 1, 1, MAX_SEQ))
    pos = np.zeros((1,), dtype=np.int32)

    def step():
        h = hidden
        for ci, m in enumerate(bodies):
            out = m.predict({
                "hidden_in": h, "cos": cos, "sin": sin,
                "causal_mask": mask, "current_pos": pos,
            }, state=states[ci])
            h = out["hidden"]
        head_out = head.predict({"hidden_in": h})
        return int(head_out["next_token"][0])

    # Warmup
    for _ in range(warmup):
        step()
    # Time E2E
    times_total = []
    for _ in range(iters):
        t = time.time()
        step()
        times_total.append((time.time() - t) * 1000)
    arr = np.array(times_total)
    print(f"  E2E iters={iters}  median={np.median(arr):.2f} ms  "
          f"mean={arr.mean():.2f}  min={arr.min():.2f}  max={arr.max():.2f}  "
          f"std={arr.std():.2f}")

    # Per-component breakdown
    per_chunk = [[] for _ in range(4)]
    per_head = []
    for _ in range(iters):
        h = hidden
        for ci, m in enumerate(bodies):
            t = time.time()
            out = m.predict({
                "hidden_in": h, "cos": cos, "sin": sin,
                "causal_mask": mask, "current_pos": pos,
            }, state=states[ci])
            per_chunk[ci].append((time.time() - t) * 1000)
            h = out["hidden"]
        t = time.time()
        head.predict({"hidden_in": h})
        per_head.append((time.time() - t) * 1000)

    print(f"  per-component (median ms):")
    for ci in range(4):
        a = np.array(per_chunk[ci])
        print(f"    chunk_{ci}:  median={np.median(a):.2f}  mean={a.mean():.2f}")
    a = np.array(per_head)
    print(f"    chunk_head: median={np.median(a):.2f}  mean={a.mean():.2f}")

    return float(np.median(arr)), [float(np.median(np.array(per_chunk[ci]))) for ci in range(4)] + [float(np.median(np.array(per_head)))]


def main():
    a = measure_bundle("Conv2dLinear", Path("/tmp/q3vl_conv_full/qwen3_vl_2b_stateful_chunks"))
    b = measure_bundle("nn.Linear",    Path("/tmp/q3vl_linear_full/qwen3_vl_2b_stateful_chunks"))

    if a and b:
        a_e2e, a_per = a
        b_e2e, b_per = b
        delta = (b_e2e - a_e2e) / a_e2e * 100
        print("\n=== summary ===")
        print(f"  E2E      Conv={a_e2e:.2f} ms  Linear={b_e2e:.2f} ms  Δ={delta:+.1f} %")
        names = ["chunk_0", "chunk_1", "chunk_2", "chunk_3", "chunk_head"]
        for i, n in enumerate(names):
            d = (b_per[i] - a_per[i]) / a_per[i] * 100 if a_per[i] > 0 else 0
            print(f"  {n:11s} Conv={a_per[i]:.2f}  Linear={b_per[i]:.2f}  Δ={d:+.1f} %")
        toks_conv = 1000.0 / a_e2e
        toks_lin = 1000.0 / b_e2e
        print(f"\n  decode tok/s (Mac):  Conv={toks_conv:.1f}   Linear={toks_lin:.1f}")


if __name__ == "__main__":
    main()
