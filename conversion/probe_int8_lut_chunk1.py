#!/usr/bin/env python3
"""Round 8: chunk_1 baseline (W4 LUT FP16 entries) vs INT8 LUT entries
(joint_compression=True). Latency + ANE placement audit.

Mac CPU_AND_NE only. Per ROUND8_FINDINGS.md candidate #1.

Usage from repo root:
  python conversion/probe_int8_lut_chunk1.py \
      --baseline /tmp/r8_int8lut/baseline/chunk_1.mlpackage \
      --int8lut  /tmp/r8_int8lut/int8lut/chunk_1.mlpackage
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import coremltools as ct


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


def measure(label, pkg_path, iters=20, warmup=3):
    print(f"\n[{label}] {pkg_path}")
    if not Path(pkg_path).is_dir():
        print(f"  missing: {pkg_path}")
        return None
    size_mb = sum(f.stat().st_size for f in Path(pkg_path).rglob('*')
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
    return {"times": arr, "size_mb": size_mb, "model": m}


def audit_ane(label, pkg_path):
    print(f"\n[ANE audit {label}]")
    try:
        m = ct.models.MLModel(str(pkg_path),
                              compute_units=ct.ComputeUnit.CPU_AND_NE)
        compiled = m.get_compiled_model_path()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        dev = Counter()
        for fn in plan.model_structure.program.functions.values():
            for op in fn.block.operations:
                a = plan.get_compute_device_usage_for_mlprogram_operation(op)
                d = ("const" if (a is None and op.operator_name == "const")
                     else (a.preferred_compute_device.__class__.__name__
                           if a else "unknown"))
                dev[d] += 1
        total = sum(v for k, v in dev.items() if k != "const")
        ane = dev.get("MLNeuralEngineComputeDevice", 0)
        ane_pct = (ane / total * 100) if total else 0.0
        print(f"  ANE: {ane}/{total} ({ane_pct:.1f}%)")
        for k, v in dev.most_common():
            print(f"    {k}: {v}")
        return ane_pct
    except Exception as e:
        print(f"  audit failed: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--int8lut", required=True)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print("=" * 70)
    print("ROUND 8 — INT8 LUT entries probe (chunk_1, Mac CPU+NE)")
    print("=" * 70)

    base = measure("baseline (FP16 LUT)", args.baseline, iters=args.iters)
    int8 = measure("int8 LUT entries", args.int8lut, iters=args.iters)

    base_ane = audit_ane("baseline", args.baseline)
    int8_ane = audit_ane("int8lut", args.int8lut)

    if base and int8:
        b_med = float(np.median(base["times"]))
        i_med = float(np.median(int8["times"]))
        delta_ms = i_med - b_med
        delta_pct = (delta_ms / b_med * 100) if b_med else 0.0
        size_delta = int8["size_mb"] - base["size_mb"]
        size_pct = (size_delta / base["size_mb"] * 100) if base["size_mb"] else 0.0
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"size:    baseline={base['size_mb']:.1f} MB  "
              f"int8lut={int8['size_mb']:.1f} MB  "
              f"Δ={size_delta:+.1f} ({size_pct:+.1f}%)")
        print(f"latency: baseline={b_med:.2f} ms  "
              f"int8lut={i_med:.2f} ms  "
              f"Δ={delta_ms:+.2f} ms ({delta_pct:+.1f}%)")
        print(f"ANE %:   baseline={base_ane:.1f}%  int8lut={int8_ane:.1f}%")
        if delta_pct < -2:
            verdict = f"PROMISING ({-delta_pct:.1f}% faster)"
        elif delta_pct < 2:
            verdict = "NEUTRAL (within noise)"
        else:
            verdict = f"REGRESSION ({delta_pct:.1f}% slower)"
        print(f"\nverdict (Mac latency only): {verdict}")
        print("Note: Mac latency may not predict iPhone — true test is iPhone bench.")


if __name__ == "__main__":
    main()
