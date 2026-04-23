#!/usr/bin/env python3
"""Mac-side benchmark: compare prefill variant dispatch cost across
the 4 `prefill_b<N>` functions in a multifunction mlpackage.

Isolates the "does smaller N actually dispatch faster" question from
end-to-end TTFT noise. Loads prefill_chunk1.mlpackage with each
function_name in turn, runs `--iterations` predicts on dummy inputs,
reports mean / min / p99 latency.

Run on Mac Studio (has ANE). compute_units=cpuAndNeuralEngine mirrors
the device config as closely as possible — Apple's ANE on M-series
is a different revision but the relative cost across variant sizes
should be indicative.

Usage:
    python conversion/benchmark_prefill_multifunction.py \\
        --mlpackage ./output/gemma4-e2b/prefill_multifunction/prefill_chunk1.mlpackage \\
        --sizes 64 128 256 512 \\
        --iterations 20 --warmup 3
"""
from __future__ import annotations
import argparse
import time
import statistics
from pathlib import Path
import numpy as np
import coremltools as ct

# Shapes match PrefillChunk1 inputs — see conversion/build_prefill_multifunction.py::_chunk1_specs
HIDDEN_SIZE = 1536
NUM_LAYERS = 35
PER_LAYER_DIM = 256
TOTAL_PLD = NUM_LAYERS * PER_LAYER_DIM


def make_inputs(N: int, real_len: int | None = None) -> dict:
    # Build causal mask: position i attends to positions 0..i within real_len,
    # everything beyond real_len is masked to -inf. Mirrors what Swift's
    # makePrefillCausalMask produces at runtime. real_len=None → full N
    # causal (upper triangle masked), no padding mask.
    mask = np.zeros((1, 1, N, N), dtype=np.float16)
    neg_inf = np.float16(-65504.0)  # fp16 min
    if real_len is None:
        real_len = N
    for i in range(N):
        # Self-attention row i: pad positions >real_len-1 are masked.
        # Also mask future positions i+1..N.
        for j in range(N):
            if j > i or j >= real_len or i >= real_len:
                mask[0, 0, i, j] = neg_inf
    return {
        "hidden_states":   np.zeros((1, N, HIDDEN_SIZE), dtype=np.float16),
        "causal_mask":     mask,
        "per_layer_raw":   np.zeros((1, N, TOTAL_PLD),   dtype=np.float16),
        "cos_s":           np.zeros((1, 1, N, 256),      dtype=np.float16),
        "sin_s":           np.zeros((1, 1, N, 256),      dtype=np.float16),
        "cos_f":           np.zeros((1, 1, N, 512),      dtype=np.float16),
        "sin_f":           np.zeros((1, 1, N, 512),      dtype=np.float16),
    }


def time_variant(mlpackage: str, function_name: str, N: int,
                 iterations: int, warmup: int, compute_units,
                 real_len: int | None = None) -> dict:
    load_t0 = time.perf_counter()
    m = ct.models.MLModel(mlpackage, function_name=function_name,
                          compute_units=compute_units)
    load_dt = time.perf_counter() - load_t0

    inputs = make_inputs(N, real_len=real_len)
    for _ in range(warmup):
        _ = m.predict(inputs)

    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = m.predict(inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "function": function_name,
        "N": N,
        "load_ms": load_dt * 1000,
        "mean_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p99_ms": latencies[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", type=str, required=True,
                    help="Path to a multifunction prefill_chunk*.mlpackage")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[64, 128, 256, 512])
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--compute-units", type=str, default="cpuAndNeuralEngine",
                    choices=["cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly", "all"])
    ap.add_argument("--real-len", type=int, default=None,
                    help="Set causal mask to simulate real_len tokens valid; "
                         "positions >= real_len masked to -inf. Default: real_len = N (full).")
    ap.add_argument("--function-name", type=str, default=None,
                    help="Override function_name when loading (e.g. pass empty/None for "
                         "single-variant mlpackages where the default `main` is used). "
                         "When set, the same name is reused for every --sizes entry.")
    args = ap.parse_args()

    cu = {
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
        "cpuAndGPU":          ct.ComputeUnit.CPU_AND_GPU,
        "cpuOnly":            ct.ComputeUnit.CPU_ONLY,
        "all":                ct.ComputeUnit.ALL,
    }[args.compute_units]

    print(f"\nBenchmark: {args.mlpackage}")
    print(f"  compute_units = {args.compute_units}, "
          f"iterations = {args.iterations}, warmup = {args.warmup}")
    print(f"\n{'function':<16} {'N':>4} {'load':>9} {'mean':>9} {'min':>9} {'p50':>9} {'p99':>9}")
    print("-" * 68)

    results: list[dict] = []
    for N in args.sizes:
        rl = args.real_len if args.real_len is not None else N
        rl = min(rl, N)  # clip to N
        fn = args.function_name if args.function_name is not None else f"prefill_b{N}"
        r = time_variant(args.mlpackage, fn, N,
                          args.iterations, args.warmup, cu,
                          real_len=rl)
        results.append(r)
        print(f"{r['function']:<16} {r['N']:>4} "
              f"{r['load_ms']:>7.0f}ms {r['mean_ms']:>7.1f}ms "
              f"{r['min_ms']:>7.1f}ms {r['p50_ms']:>7.1f}ms {r['p99_ms']:>7.1f}ms")

    if len(results) >= 2:
        largest = max(results, key=lambda r: r["N"])
        smallest = min(results, key=lambda r: r["N"])
        print(f"\nRatio smallest / largest (mean): "
              f"{smallest['mean_ms']/largest['mean_ms']:.2f}x")
        print(f"(compute work ratio = {smallest['N']/largest['N']:.3f}x — "
              f"if dispatch-dominated, mean ratio ≈ 1)")


if __name__ == "__main__":
    main()
