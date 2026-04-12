#!/usr/bin/env python3
"""Benchmark prefill latency: ANE vs GPU (A19 Pro tensor cores).

Measures the TTFT component of inference by running the prefill path with
two different compute units on the same mlpackage set, reporting delta.

The GPU variant is produced by `conversion/build_prefill_gpu.py`; this
script loads both variants and times N prefill iterations each.

Usage (Mac, real device required for MLModel prediction; simulator lacks ANE):
    python conversion/benchmark_prefill.py \\
        --ane-dir ./output/gemma4-e2b/ane/prefill \\
        --gpu-dir ./output/gemma4-e2b/gpu/prefill \\
        --prompt-len 2048 \\
        --iterations 10

Reports:
  - per-chunk latency (ms) for each compute unit
  - aggregate (sum of 4 chunks) TTFT
  - speedup ratio
  - any chunks that failed to load on the target compute unit (compile error)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def benchmark_chunk(mlpackage_path: Path, compute_units_str: str, dummy_inputs: dict,
                    iterations: int, warmup: int = 2) -> dict:
    """Load + run a single mlpackage N times, return timing stats."""
    import coremltools as ct
    try:
        cu_enum = {
            "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
            "cpuAndGPU":          ct.ComputeUnit.CPU_AND_GPU,
            "cpuOnly":            ct.ComputeUnit.CPU_ONLY,
            "all":                ct.ComputeUnit.ALL,
        }[compute_units_str]
    except KeyError:
        raise ValueError(f"unknown compute units: {compute_units_str}")

    try:
        m = ct.models.MLModel(str(mlpackage_path), compute_units=cu_enum)
    except Exception as e:
        return {"error": f"load failed: {e}"}

    # Warmup
    for _ in range(warmup):
        _ = m.predict(dummy_inputs)

    # Time
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = m.predict(dummy_inputs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    return {
        "iterations": iterations,
        "min_ms":    latencies[0],
        "p50_ms":    latencies[len(latencies) // 2],
        "max_ms":    latencies[-1],
        "mean_ms":   sum(latencies) / len(latencies),
    }


def make_dummy_inputs(mlpackage_path: Path) -> dict:
    """Inspect the model and build zero-filled inputs matching its shape spec."""
    import coremltools as ct
    import numpy as np
    m = ct.models.MLModel(str(mlpackage_path))
    spec = m.get_spec()
    inputs: dict = {}
    for inp in spec.description.input:
        name = inp.name
        t = inp.type
        if t.WhichOneof("Type") == "multiArrayType":
            shape = tuple(t.multiArrayType.shape)
            arr = np.zeros(shape, dtype=np.float16)
            inputs[name] = arr
        else:
            raise RuntimeError(f"unsupported input type for {name}")
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ane-dir", type=str, required=True,
                    help="Directory with prefill_chunk{1..4}.mlpackage (ANE variant)")
    ap.add_argument("--gpu-dir", type=str, required=True,
                    help="Directory with prefill_chunk{1..4}_gpu.mlpackage (GPU variant)")
    ap.add_argument("--prompt-len", type=int, default=2048,
                    help="Target prompt length (informational; mlpackage shapes govern actual)")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    if platform.system() != "Darwin":
        print("WARN: CoreML prediction requires macOS. This run will likely error.")

    ane_dir = Path(args.ane_dir)
    gpu_dir = Path(args.gpu_dir)

    results = {"iterations": args.iterations, "warmup": args.warmup, "chunks": []}
    agg_ane_mean = 0.0
    agg_gpu_mean = 0.0

    for i in range(1, 5):
        ane_path = ane_dir / f"prefill_chunk{i}.mlpackage"
        gpu_path = gpu_dir / f"prefill_chunk{i}_gpu.mlpackage"
        chunk_res = {"chunk": i}

        if not ane_path.exists():
            print(f"chunk{i}: ANE package missing at {ane_path}")
            chunk_res["ane_error"] = "missing file"
        else:
            print(f"\nchunk{i} — benchmarking ANE path...")
            dummy = make_dummy_inputs(ane_path)
            ane_stats = benchmark_chunk(ane_path, "cpuAndNeuralEngine", dummy,
                                         args.iterations, args.warmup)
            chunk_res["ane"] = ane_stats
            if "mean_ms" in ane_stats:
                agg_ane_mean += ane_stats["mean_ms"]
                print(f"  ANE p50={ane_stats['p50_ms']:.2f}ms mean={ane_stats['mean_ms']:.2f}ms")

        if not gpu_path.exists():
            print(f"chunk{i}: GPU package missing at {gpu_path}")
            chunk_res["gpu_error"] = "missing file"
        else:
            print(f"chunk{i} — benchmarking GPU path...")
            dummy = make_dummy_inputs(gpu_path)
            gpu_stats = benchmark_chunk(gpu_path, "cpuAndGPU", dummy,
                                         args.iterations, args.warmup)
            chunk_res["gpu"] = gpu_stats
            if "mean_ms" in gpu_stats:
                agg_gpu_mean += gpu_stats["mean_ms"]
                print(f"  GPU p50={gpu_stats['p50_ms']:.2f}ms mean={gpu_stats['mean_ms']:.2f}ms")

        if "mean_ms" in chunk_res.get("ane", {}) and "mean_ms" in chunk_res.get("gpu", {}):
            speedup = chunk_res["ane"]["mean_ms"] / chunk_res["gpu"]["mean_ms"]
            chunk_res["gpu_speedup_vs_ane"] = speedup
            print(f"  speedup (ANE/GPU): {speedup:.2f}x")

        results["chunks"].append(chunk_res)

    # Aggregate
    print("\n── Aggregate (sum of 4 chunks) ──")
    print(f"  ANE total: {agg_ane_mean:.2f} ms")
    print(f"  GPU total: {agg_gpu_mean:.2f} ms")
    if agg_ane_mean > 0 and agg_gpu_mean > 0:
        print(f"  end-to-end speedup: {agg_ane_mean / agg_gpu_mean:.2f}x")
        results["aggregate"] = {
            "ane_total_ms": agg_ane_mean,
            "gpu_total_ms": agg_gpu_mean,
            "end_to_end_speedup": agg_ane_mean / agg_gpu_mean,
            "ttft_seconds_ane": agg_ane_mean / 1000.0,
            "ttft_seconds_gpu": agg_gpu_mean / 1000.0,
        }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f: json.dump(results, f, indent=2)
        print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
