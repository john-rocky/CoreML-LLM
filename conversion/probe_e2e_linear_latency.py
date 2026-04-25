#!/usr/bin/env python3
"""End-to-end Mac CPU_AND_NE latency comparison: full 4-chunk Gemma 4 E2B
forward (chunk_1 → 2 → 3 → 4) for the Conv2d-1×1 wrapper variant vs the
nn.Linear native variant. W4 LUT palettize.

The chunk_1 single-chunk probe (probe_chunk1_linear_w4_latency.py) showed
parity at chunk_1 scale (+0.9%) — but Plan 3 migration decision needs
the E2E sum (especially chunk_4 lm_head Conv2d→Linear, where the projection
goes 2304→262144).

Both bundles must already be built at:
  /tmp/g4_chunk1_ab/conv/{chunk_1,chunk_2,chunk_3,chunk_4}.mlpackage
  /tmp/g4_chunk1_ab/linear/{chunk_1,chunk_2,chunk_3,chunk_4}.mlpackage
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct

ROOT = Path("/tmp/g4_chunk1_ab")

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


def _load(label: str, bundle_dir: Path):
    ts = []
    chunks = []
    for i in range(1, 5):
        p = bundle_dir / f"chunk_{i}.mlpackage"
        if not p.is_dir():
            print(f"  ❌ missing: {p}")
            return None
        t = time.time()
        m = ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE)
        ts.append(time.time() - t)
        chunks.append(m)
    print(f"[{label}] load times: {[f'{t:.1f}s' for t in ts]}, total {sum(ts):.1f}s")
    return chunks


def _e2e_step(chunks, state1, state2, position: int, prev_token_id: int = 0):
    """One full forward through chunks 1→2→3→4. Returns per-chunk ms +
    total ms. State is mutated in place."""
    # Common inputs
    cos_s = _zeros((1, 1, 1, HD_S))
    sin_s = _zeros((1, 1, 1, HD_S))
    cos_f = _zeros((1, 1, 1, HD_F))
    sin_f = _zeros((1, 1, 1, HD_F))
    causal_mask_full = _zeros((1, 1, 1, CTX))
    causal_mask_sliding = _zeros((1, 1, 1, W))
    current_pos = np.array([position], dtype=np.int32)
    ring_pos = np.array([position % W], dtype=np.int32)

    timings = []
    # chunk_1: takes per_layer_raw, has state
    inputs = {
        "hidden_states": _zeros((1, 1, HIDDEN)),
        "causal_mask_full": causal_mask_full,
        "causal_mask_sliding": causal_mask_sliding,
        "per_layer_raw": _zeros((1, 1, NLAYERS * PLD)),
        "cos_s": cos_s, "sin_s": sin_s,
        "cos_f": cos_f, "sin_f": sin_f,
        "current_pos": current_pos, "ring_pos": ring_pos,
    }
    t = time.time()
    out1 = chunks[0].predict(inputs, state=state1)
    timings.append((time.time() - t) * 1000)

    # chunk_2: takes per_layer_combined, has state
    inputs = {
        "hidden_states": np.asarray(out1["hidden_states_out"]).astype(np.float16),
        "causal_mask_full": causal_mask_full,
        "causal_mask_sliding": causal_mask_sliding,
        "per_layer_combined": np.asarray(out1["per_layer_combined_out"]).astype(np.float16),
        "cos_s": cos_s, "sin_s": sin_s,
        "cos_f": cos_f, "sin_f": sin_f,
        "current_pos": current_pos, "ring_pos": ring_pos,
    }
    t = time.time()
    out2 = chunks[1].predict(inputs, state=state2)
    timings.append((time.time() - t) * 1000)

    # chunk_3: stateless, reads kv13/14
    inputs = {
        "hidden_states": np.asarray(out2["hidden_states_out"]).astype(np.float16),
        "causal_mask_full": causal_mask_full,
        "causal_mask_sliding": causal_mask_sliding,
        "per_layer_combined": np.asarray(out1["per_layer_combined_out"]).astype(np.float16),
        "cos_s": cos_s, "sin_s": sin_s,
        "cos_f": cos_f, "sin_f": sin_f,
        "kv13_k": np.asarray(out2["kv13_k"]).astype(np.float16),
        "kv13_v": np.asarray(out2["kv13_v"]).astype(np.float16),
        "kv14_k": np.asarray(out2["kv14_k"]).astype(np.float16),
        "kv14_v": np.asarray(out2["kv14_v"]).astype(np.float16),
    }
    t = time.time()
    out3 = chunks[2].predict(inputs)
    timings.append((time.time() - t) * 1000)

    # chunk_4: stateless, lm_head + argmax
    inputs = {
        "hidden_states": np.asarray(out3["hidden_states_out"]).astype(np.float16),
        "causal_mask_full": causal_mask_full,
        "causal_mask_sliding": causal_mask_sliding,
        "per_layer_combined": np.asarray(out1["per_layer_combined_out"]).astype(np.float16),
        "cos_s": cos_s, "sin_s": sin_s,
        "cos_f": cos_f, "sin_f": sin_f,
        "kv13_k": np.asarray(out2["kv13_k"]).astype(np.float16),
        "kv13_v": np.asarray(out2["kv13_v"]).astype(np.float16),
        "kv14_k": np.asarray(out2["kv14_k"]).astype(np.float16),
        "kv14_v": np.asarray(out2["kv14_v"]).astype(np.float16),
    }
    t = time.time()
    out4 = chunks[3].predict(inputs)
    timings.append((time.time() - t) * 1000)

    return timings, int(np.asarray(out4["token_id"]).flat[0])


def measure(label: str, bundle_dir: Path, iters: int = 20, warmup: int = 3):
    print(f"\n=== {label} ===")
    chunks = _load(label, bundle_dir)
    if chunks is None:
        return None
    state1 = chunks[0].make_state()
    state2 = chunks[1].make_state()

    # Warmup + advance position so state has actual writes (closer to real decode)
    for i in range(warmup):
        _e2e_step(chunks, state1, state2, position=i)

    per_chunk = [[], [], [], []]
    totals = []
    last_token = -1
    for i in range(iters):
        timings, tok = _e2e_step(chunks, state1, state2, position=warmup + i)
        for k, t in enumerate(timings):
            per_chunk[k].append(t)
        totals.append(sum(timings))
        last_token = tok
    arr = np.array(totals)
    pc = [np.array(c) for c in per_chunk]
    print(f"  iters={iters}  total median={np.median(arr):.2f} ms  "
          f"mean={arr.mean():.2f}  std={arr.std():.2f}")
    print(f"  per-chunk medians: "
          f"c1={np.median(pc[0]):.2f}  c2={np.median(pc[1]):.2f}  "
          f"c3={np.median(pc[2]):.2f}  c4={np.median(pc[3]):.2f}")
    print(f"  last token_id: {last_token}")
    return arr, pc


def main():
    a = measure("A (Conv2d-1×1 wrapper, W4)", ROOT / "conv")
    b = measure("B (nn.Linear native, W4)", ROOT / "linear")

    print("\n" + "=" * 60)
    print("E2E VERDICT")
    print("=" * 60)
    if a is None or b is None:
        print("  ❌ one or both bundles incomplete")
        return 1
    a_total, a_pc = a
    b_total, b_pc = b
    am, bm = float(np.median(a_total)), float(np.median(b_total))
    delta = (bm / am - 1.0) * 100.0
    print(f"  E2E A median: {am:.2f} ms")
    print(f"  E2E B median: {bm:.2f} ms")
    print(f"  delta:        {bm-am:+.2f} ms ({delta:+.1f}%)")
    print()
    print("  Per-chunk delta (B-A) ms:")
    names = ["chunk_1", "chunk_2", "chunk_3", "chunk_4 (lm_head)"]
    for i in range(4):
        am_i, bm_i = float(np.median(a_pc[i])), float(np.median(b_pc[i]))
        d_i = bm_i - am_i
        d_pct = (bm_i / am_i - 1.0) * 100.0 if am_i > 0 else 0.0
        print(f"    {names[i]:<22} {bm_i-am_i:+.2f} ms ({d_pct:+.1f}%) "
              f"[A {am_i:.2f}, B {bm_i:.2f}]")

    print()
    if abs(delta) < 5:
        print("  → E2E parity. Migration GO; iPhone validation confirms.")
    elif delta < -5:
        print("  → B faster E2E. Strong migration GO.")
    elif delta < 15:
        print("  → B slower 5-15% E2E. iPhone test gates.")
    else:
        print("  → B slower 15%+ E2E. Migration HOLD.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
