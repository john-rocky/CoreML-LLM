#!/usr/bin/env python3
"""Benchmark T=1 decode through the 3-chunk merged stateful bundle.

Layout:
  chunk_1: L0-7,  own KV, computes PLE
  chunk_2: L8-24, merged (own KV L8-14 + KV-shared L15-24)
  chunk_3: L25-34 + lm_head + argmax (= old chunk_4)

Mirrors `benchmark_stateful_chunks.py` but skips the chunk_3 input pass
because the merged middle now drives directly into chunk_3 (the final
lm_head chunk). One fewer Mac↔ANE roundtrip per token.

Usage:
    python conversion/benchmark_stateful_3chunks.py \
        --bundle /tmp/g4_3chunk/multi --ctx 2048 --steps 30
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
    ap.add_argument("--bundle", required=True)
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
    print(f"3-chunk stateful benchmark: {os.path.basename(args.bundle)}, "
          f"ctx={ctx}, fn={args.function}")
    print(f"Compute units: {cu}")
    print(f"{'='*60}")

    print("Loading models...")
    def _load(name):
        path = os.path.join(args.bundle, f"{name}.mlpackage")
        t0 = time.time()
        m = ct.models.MLModel(path, compute_units=cu, function_name=args.function)
        print(f"  {name} loaded in {time.time()-t0:.1f}s")
        return m

    c1 = _load("chunk_1")
    c2 = _load("chunk_2")    # merged
    c3 = _load("chunk_3")    # final + lm_head

    state1 = c1.make_state()
    state2 = c2.make_state()

    h_in = _z((1, 1, hidden))
    plr = _z((1, 1, total_pld))
    cos_s = _z((1, 1, 1, hd_s))
    sin_s = _z((1, 1, 1, hd_s))
    cos_f = _z((1, 1, 1, hd_f))
    sin_f = _z((1, 1, 1, hd_f))

    times_total, times_c1, times_c2, times_c3 = [], [], [], []

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
        t1 = time.time()
        out1 = c1.predict({
            "hidden_states": h_in,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "per_layer_raw": plr,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "current_pos": cur_pos, "ring_pos": ring_pos,
        }, state=state1)
        dt_c1 = time.time() - t1
        h1 = out1["hidden_states_out"]
        plc = out1["per_layer_combined_out"]

        t2 = time.time()
        out2 = c2.predict({
            "hidden_states": h1,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "per_layer_combined": plc,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "current_pos": cur_pos, "ring_pos": ring_pos,
        }, state=state2)
        dt_c2 = time.time() - t2
        h2 = out2["hidden_states_out"]
        kv13_k = out2["kv13_k"]; kv13_v = out2["kv13_v"]
        kv14_k = out2["kv14_k"]; kv14_v = out2["kv14_v"]

        t3 = time.time()
        _ = c3.predict({
            "hidden_states": h2,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "per_layer_combined": plc,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "kv13_k": kv13_k, "kv13_v": kv13_v,
            "kv14_k": kv14_k, "kv14_v": kv14_v,
        })
        dt_c3 = time.time() - t3
        dt_total = time.time() - t0

        times_total.append(dt_total)
        times_c1.append(dt_c1); times_c2.append(dt_c2); times_c3.append(dt_c3)

        if step == 0 or (step + 1) % 10 == 0:
            print(f"  Step {step+1}: total={dt_total*1000:.1f}ms "
                  f"[c1={dt_c1*1000:.1f} c2={dt_c2*1000:.1f} c3={dt_c3*1000:.1f}]")

    skip = 5
    def stats(a):
        a = a[skip:]
        return np.mean(a)*1000, np.std(a)*1000, np.min(a)*1000, np.max(a)*1000

    print(f"\n{'='*60}")
    print(f"Results (ctx={ctx}, fn={args.function}, "
          f"steps {skip+1}-{args.steps}):")
    print(f"{'='*60}")
    for name, arr in [("chunk1", times_c1), ("chunk2(merged)", times_c2),
                      ("chunk3(final)", times_c3), ("TOTAL", times_total)]:
        mean, std, mn, mx = stats(arr)
        print(f"  {name:14s}: {mean:6.1f}ms ±{std:4.1f} (min={mn:.1f}, max={mx:.1f})")
    mean_total = np.mean(times_total[skip:]) * 1000
    print(f"\n  Throughput: {1000/mean_total:.1f} tok/s")


if __name__ == "__main__":
    main()
