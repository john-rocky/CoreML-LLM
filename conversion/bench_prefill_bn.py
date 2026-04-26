#!/usr/bin/env python3
"""Latency comparison: T=N prefill_bN vs T=1 infer ×N (single chunk).

Measures Mac-side per-call latency for the multifunction package's
prefill_b<T> and infer functions, run on the same chunk so the only
variable is the batch size. T=N speedup vs T=1×N is the headline
prefill TTFT win for Stage 3.

Usage:
    python conversion/bench_prefill_bn.py \
        --pkg /tmp/g4_prefill/multi/chunk_1.mlpackage --T 8 --reps 50
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import coremltools as ct


def _function_inputs(model, function_name):
    spec = model.get_spec()
    desc = spec.description
    if desc.input:
        return list(desc.input)
    for fdesc in desc.functions:
        if fdesc.name == function_name:
            return list(fdesc.input)
    return []


def _build_inputs(model, function_name):
    feats = {}
    for inp in _function_inputs(model, function_name):
        if inp.type.WhichOneof("Type") != "multiArrayType":
            continue
        shape = tuple(int(d) for d in inp.type.multiArrayType.shape)
        dt = inp.type.multiArrayType.dataType
        if dt == 65568:
            feats[inp.name] = np.zeros(shape, dtype=np.float16)
        elif dt in (131104, 131088):
            feats[inp.name] = np.zeros(shape, dtype=np.int32)
        else:
            feats[inp.name] = np.zeros(shape, dtype=np.float32)
    return feats


def _bench(model, fn, state_factory, reps, warmup=5):
    feats = _build_inputs(model, fn)
    state = state_factory()
    for _ in range(warmup):
        model.predict(feats, state=state)
    state = state_factory()
    t0 = time.perf_counter()
    for _ in range(reps):
        model.predict(feats, state=state)
    dt = (time.perf_counter() - t0) / reps
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    pkg = Path(args.pkg)
    units = ct.ComputeUnit.CPU_AND_NE

    print(f"package: {pkg}")
    print(f"T={args.T}, reps={args.reps}")

    m_pref = ct.models.MLModel(str(pkg), function_name=f"prefill_b{args.T}",
                                compute_units=units)
    m_inf = ct.models.MLModel(str(pkg), function_name="infer",
                                compute_units=units)
    pref_dt = _bench(m_pref, f"prefill_b{args.T}", m_pref.make_state, args.reps)
    inf_dt = _bench(m_inf, "infer", m_inf.make_state, args.reps)

    print(f"\n  T={args.T} prefill_b{args.T}:  "
          f"{pref_dt*1000:.2f} ms/call (= {pref_dt/args.T*1000:.2f} ms/tok)")
    print(f"  T=1 infer ×{args.T}:          "
          f"{inf_dt*1000:.2f} ms/call ({inf_dt*args.T*1000:.2f} ms total)")
    speedup = (inf_dt * args.T) / pref_dt
    print(f"  speedup vs T=1×{args.T}: {speedup:.2f}×")


if __name__ == "__main__":
    main()
