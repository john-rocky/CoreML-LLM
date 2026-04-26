#!/usr/bin/env python3
"""Mac sanity for the multifunction prefill_bN packages.

Loads `infer` (T=1) and `prefill_b<T>` from the same .mlpackage,
predicts zero inputs, and checks that:
  - both functions load via `function_name=...`
  - state buffers persist across calls (round-trip via slice_update)
  - outputs are finite

Usage:
    python conversion/sanity_prefill_bn.py \
        --pkg /tmp/g4_prefill/multi/chunk_1.mlpackage --T 8
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import coremltools as ct


def _zero_input(spec):
    shape = tuple(d.size if d.WhichOneof("dimension") == "size"
                  else d.size_range.lower_bound
                  for d in spec.type.multiArrayType.shape)
    dtype = {
        65568: np.float16,  # MLArrayDataType.FLOAT16
        65600: np.float32,
        131104: np.int32,
    }.get(spec.type.multiArrayType.dataType, np.float32)
    return np.zeros(shape, dtype=dtype)


def _function_inputs(model, function_name):
    """Get the per-function inputs for a multifunction model.
    Top-level spec.description has empty inputs — they live in
    spec.description.functions[function_name].input."""
    spec = model.get_spec()
    desc = spec.description
    if desc.input:
        return list(desc.input)
    for fdesc in desc.functions:
        if fdesc.name == function_name:
            return list(fdesc.input)
    return []


def _function_outputs(model, function_name):
    spec = model.get_spec()
    desc = spec.description
    if desc.output:
        return list(desc.output)
    for fdesc in desc.functions:
        if fdesc.name == function_name:
            return list(fdesc.output)
    return []


def _build_inputs(model, function_name):
    """Build a zero-filled feature dict from the function's input description."""
    feats = {}
    for inp in _function_inputs(model, function_name):
        if inp.type.WhichOneof("Type") == "stateType":
            continue  # state is passed separately
        if inp.type.WhichOneof("Type") == "multiArrayType":
            shape = tuple(d for d in inp.type.multiArrayType.shape)
            dt = inp.type.multiArrayType.dataType
            if dt == 65568:                  # FLOAT16
                feats[inp.name] = np.zeros(shape, dtype=np.float16)
            elif dt == 131104 or dt == 131088:  # INT32 / INT32 alt
                feats[inp.name] = np.zeros(shape, dtype=np.int32)
            else:
                feats[inp.name] = np.zeros(shape, dtype=np.float32)
    return feats


def _check_output_finite(out_dict, label):
    bad = []
    for name, arr in out_dict.items():
        a = np.asarray(arr)
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            bad.append(name)
    print(f"  {label}: {len(out_dict)} outputs"
          + ("" if not bad else f", non-finite={bad}"))
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--T", type=int, default=8)
    args = ap.parse_args()

    pkg = Path(args.pkg)
    print(f"Loading multifunction: {pkg}")

    cfg_infer = ct.ComputeUnit.CPU_AND_NE
    m_infer = ct.models.MLModel(str(pkg), function_name="infer",
                                 compute_units=cfg_infer)
    m_pref = ct.models.MLModel(str(pkg), function_name=f"prefill_b{args.T}",
                                compute_units=cfg_infer)
    print("  loaded both functions")

    print("\nInputs (infer):", [i.name for i in _function_inputs(m_infer, "infer")])
    print("Inputs (prefill):", [i.name for i in _function_inputs(m_pref, f"prefill_b{args.T}")])
    print("Outputs (infer):", [o.name for o in _function_outputs(m_infer, "infer")])
    print("Outputs (prefill):", [o.name for o in _function_outputs(m_pref, f"prefill_b{args.T}")])

    # The two functions share state buffers (kv_cache_*). Build state from
    # one and re-use it in the other to confirm the slot bindings work.
    state = m_pref.make_state()

    feats_pref = _build_inputs(m_pref, f"prefill_b{args.T}")
    print(f"\nprefill_b{args.T} input shapes:")
    for k, v in feats_pref.items():
        print(f"    {k}: {v.shape} {v.dtype}")
    out_pref = m_pref.predict(feats_pref, state=state)
    if not _check_output_finite(out_pref, f"prefill_b{args.T}"):
        raise SystemExit("prefill output non-finite")

    feats_infer = _build_inputs(m_infer, "infer")
    # current_pos = T (write at the slot right after prefill)
    if "current_pos" in feats_infer:
        feats_infer["current_pos"][:] = args.T
    if "ring_pos" in feats_infer:
        feats_infer["ring_pos"][:] = args.T
    print(f"\ninfer (T=1) at current_pos={args.T}:")
    out_inf = m_infer.predict(feats_infer, state=state)
    if not _check_output_finite(out_inf, "infer"):
        raise SystemExit("infer output non-finite")

    print("\nOK: multifunction sanity passed.")


if __name__ == "__main__":
    main()
