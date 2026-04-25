"""
Probe: coremltools 9.0 `allowLowPrecisionAccumulationOnGPU` runtime hint
on Gemma 4 still-image vision encoder (cpuAndGPU).

A vs B comparison:
  A: default config (allowLowPrecisionAccumulationOnGPU = false)
  B: hint enabled  (allowLowPrecisionAccumulationOnGPU = true)

Reports:
  - median / p99 latency for both
  - max abs diff between A and B output (numerical impact)
  - verdict: GO / NEUTRAL / HOLD

Requires coremltools >= 9.0, macOS 15+.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.models import CompiledMLModel


DEFAULT_MODEL = (
    "/Users/majimadaisuke/Downloads/coreml-llm-artifacts/"
    "staging-2k-fast-prefill/gemma4-e2b/vision.mlmodelc"
)


def make_inputs(seed: int = 0) -> dict[str, np.ndarray]:
    """Match Gemma 4 vision encoder schema:
        pixel_values        Float32 (1, 2520, 768)  pre-patchified
        pixel_position_ids  Int32   (1, 2520, 2)
    """
    rng = np.random.default_rng(seed)
    pixel_values = rng.standard_normal((1, 2520, 768), dtype=np.float32) * 0.5
    # Position ids: row, col grid (typical Gemma 4 vision layout). Values do not
    # affect latency materially; choose a deterministic sequence.
    rows = np.arange(2520, dtype=np.int32) // 56
    cols = np.arange(2520, dtype=np.int32) % 56
    pixel_position_ids = np.stack([rows, cols], axis=-1).reshape(1, 2520, 2).astype(np.int32)
    return {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids}


def time_predict(model: CompiledMLModel, inputs: dict, *, warmup: int, iters: int):
    for _ in range(warmup):
        out = model.predict(inputs)
    samples_ms = []
    last_out = None
    for _ in range(iters):
        t0 = time.perf_counter()
        last_out = model.predict(inputs)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()
    return samples_ms, last_out


def percentile(sorted_samples: list[float], p: float) -> float:
    if not sorted_samples:
        return float("nan")
    k = (len(sorted_samples) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_samples) - 1)
    return sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f)


def summarize(label: str, samples_ms: list[float]) -> dict:
    med = statistics.median(samples_ms)
    p99 = percentile(samples_ms, 0.99)
    mean = statistics.mean(samples_ms)
    p10 = percentile(samples_ms, 0.10)
    print(f"  [{label}] median={med:7.2f} ms  mean={mean:7.2f} ms  "
          f"p10={p10:7.2f} ms  p99={p99:7.2f} ms  n={len(samples_ms)}")
    return {"median": med, "p99": p99, "mean": mean, "p10": p10}


def diff_outputs(a: dict, b: dict) -> tuple[float, float]:
    keys = sorted(set(a) & set(b))
    max_abs = 0.0
    max_rel = 0.0
    for k in keys:
        av = np.asarray(a[k], dtype=np.float32)
        bv = np.asarray(b[k], dtype=np.float32)
        d = np.abs(av - bv)
        max_abs = max(max_abs, float(d.max()))
        denom = np.maximum(np.abs(av), 1e-6)
        max_rel = max(max_rel, float((d / denom).max()))
    return max_abs, max_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    print(f"coremltools = {ct.__version__}")
    print(f"model       = {model_path}")
    print(f"compute     = cpuAndGPU")
    print(f"warmup={args.warmup}  iters={args.iters}  seed={args.seed}")

    inputs = make_inputs(seed=args.seed)
    for k, v in inputs.items():
        print(f"  input  {k}: {v.dtype} {tuple(v.shape)}")

    print("\n[load A] default (no optimization_hints)")
    model_a = CompiledMLModel(
        str(model_path),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    print("[load B] allowLowPrecisionAccumulationOnGPU = True")
    model_b = CompiledMLModel(
        str(model_path),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        optimization_hints={"allowLowPrecisionAccumulationOnGPU": True},
    )

    print("\n[bench A] default")
    samples_a, last_a = time_predict(model_a, inputs, warmup=args.warmup, iters=args.iters)
    stats_a = summarize("A default", samples_a)

    print("\n[bench B] hint=True")
    samples_b, last_b = time_predict(model_b, inputs, warmup=args.warmup, iters=args.iters)
    stats_b = summarize("B hint=T ", samples_b)

    # Re-run A once more after B to detect first-load cache effects skewing A.
    print("\n[bench A2] default (re-run after B, sanity)")
    samples_a2, _ = time_predict(model_a, inputs, warmup=2, iters=args.iters)
    stats_a2 = summarize("A2 def  ", samples_a2)

    # Use min(A, A2) median as the fair baseline for A.
    a_median_fair = min(stats_a["median"], stats_a2["median"])

    print("\n[diff] output A vs B (single-shot, last sample)")
    max_abs, max_rel = diff_outputs(last_a, last_b)
    print(f"  max_abs_diff = {max_abs:.6f}")
    print(f"  max_rel_diff = {max_rel:.6f}")

    delta_pct = (stats_b["median"] - a_median_fair) / a_median_fair * 100.0
    print("\n=== SUMMARY ===")
    print(f"  A median (fair = min of two runs) : {a_median_fair:7.2f} ms")
    print(f"  B median                          : {stats_b['median']:7.2f} ms")
    print(f"  delta                             : {delta_pct:+6.2f} %  "
          f"({'B faster' if delta_pct < 0 else 'B slower'})")
    print(f"  max abs diff                      : {max_abs:.4e}")

    # Verdict per spec:
    #   delta <= -10% AND max_abs < 1e-2   -> GO
    #   |delta| <= 2%                      -> NEUTRAL (HOLD)
    #   delta > 0                          -> regression -> ABORT
    #   else                               -> NEUTRAL (HOLD)
    if not np.isfinite(max_abs):
        verdict = "ABORT (NaN output)"
    elif delta_pct > 2.0:
        verdict = "ABORT (regression)"
    elif delta_pct <= -10.0 and max_abs < 1e-2:
        verdict = "GO (commit hint)"
    elif delta_pct <= -10.0 and max_abs >= 1e-2:
        verdict = "HOLD (latency win but accuracy loss too large)"
    elif abs(delta_pct) <= 2.0:
        verdict = "NEUTRAL / HOLD (no measurable benefit)"
    else:
        verdict = f"INCONCLUSIVE (delta={delta_pct:+.2f}%)"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
