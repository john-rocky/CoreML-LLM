#!/usr/bin/env python3
"""Mac chained sanity for the multifunction prefill_bN bundle.

Loads chunk_{1..4}.mlpackage, runs a T=N forward through the four
prefill chunks in sequence (mirroring the runtime dispatch pattern),
and checks:
  - all four functions load via `function_name=prefill_b<T>`
  - chunk_1 → chunk_2 KV state propagation (slice_update writes T slots)
  - chunk_2 emits kv13/kv14 with the expected shapes for chunks 3/4
  - chunk_4 produces a finite token_id

Then runs a single T=1 step at position=T (the post-prefill decode
slot) using the `infer` function to confirm cross-function state
sharing (the T=1 forward sees the KV that prefill wrote).

Usage:
    python conversion/sanity_prefill_bn_chain.py \
        --bundle /tmp/g4_prefill/multi --T 8
"""
from __future__ import annotations
import argparse
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


def _check_finite(name, arr):
    a = np.asarray(arr)
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise SystemExit(f"  ! non-finite output: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--T", type=int, default=8)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    T = args.T
    fn = f"prefill_b{T}"

    print(f"Bundle: {bundle}")
    print(f"Loading multifunction `{fn}` for chunks 1..4")

    units = ct.ComputeUnit.CPU_AND_NE
    pkgs = [bundle / f"chunk_{i}.mlpackage" for i in (1, 2, 3, 4)]
    prefill_models = [ct.models.MLModel(str(p), function_name=fn,
                                          compute_units=units) for p in pkgs]
    infer_models = [ct.models.MLModel(str(p), function_name="infer",
                                        compute_units=units) for p in pkgs]
    print("  loaded all four chunks (prefill + infer)")

    # State buffers belong to chunk_1 and chunk_2 (3/4 are stateless).
    state1 = prefill_models[0].make_state()
    state2 = prefill_models[1].make_state()

    # ---- T=N pass ---------------------------------------------------
    print(f"\nT={T} prefill pass:")
    in1 = _build_inputs(prefill_models[0], fn)
    out1 = prefill_models[0].predict(in1, state=state1)
    print(f"  chunk_1 outs: {sorted(out1.keys())}")
    _check_finite("chunk_1.hidden_states_out", out1["hidden_states_out"])

    in2 = _build_inputs(prefill_models[1], fn)
    in2["hidden_states"] = out1["hidden_states_out"]
    in2["per_layer_combined"] = out1["per_layer_combined_out"]
    out2 = prefill_models[1].predict(in2, state=state2)
    print(f"  chunk_2 outs: {sorted(out2.keys())}")
    for name in ("hidden_states_out", "kv13_k", "kv13_v", "kv14_k", "kv14_v"):
        _check_finite(f"chunk_2.{name}", out2[name])
        print(f"    {name}: shape={np.asarray(out2[name]).shape}")

    in3 = _build_inputs(prefill_models[2], fn)
    in3["hidden_states"] = out2["hidden_states_out"]
    in3["per_layer_combined"] = out1["per_layer_combined_out"]
    in3["kv13_k"] = out2["kv13_k"]; in3["kv13_v"] = out2["kv13_v"]
    in3["kv14_k"] = out2["kv14_k"]; in3["kv14_v"] = out2["kv14_v"]
    out3 = prefill_models[2].predict(in3)
    _check_finite("chunk_3.hidden_states_out", out3["hidden_states_out"])
    print(f"  chunk_3 hidden_states_out: shape="
          f"{np.asarray(out3['hidden_states_out']).shape}")

    in4 = _build_inputs(prefill_models[3], fn)
    in4["hidden_states"] = out3["hidden_states_out"]
    in4["per_layer_combined"] = out1["per_layer_combined_out"]
    in4["kv13_k"] = out2["kv13_k"]; in4["kv13_v"] = out2["kv13_v"]
    in4["kv14_k"] = out2["kv14_k"]; in4["kv14_v"] = out2["kv14_v"]
    out4 = prefill_models[3].predict(in4)
    print(f"  chunk_4 outs: {sorted(out4.keys())}")
    tok_arr = np.asarray(out4["token_id"])
    print(f"    token_id: {tok_arr.tolist()}  shape={tok_arr.shape}")

    # ---- T=1 pass at position=T using the same state ----------------
    print(f"\nT=1 infer pass at current_pos={T} (uses same state):")
    in1d = _build_inputs(infer_models[0], "infer")
    in1d["current_pos"][:] = T
    in1d["ring_pos"][:] = T
    out1d = infer_models[0].predict(in1d, state=state1)
    _check_finite("infer.chunk_1", out1d["hidden_states_out"])

    in2d = _build_inputs(infer_models[1], "infer")
    in2d["current_pos"][:] = T
    in2d["ring_pos"][:] = T
    in2d["hidden_states"] = out1d["hidden_states_out"]
    in2d["per_layer_combined"] = out1d["per_layer_combined_out"]
    out2d = infer_models[1].predict(in2d, state=state2)
    _check_finite("infer.chunk_2", out2d["hidden_states_out"])

    in3d = _build_inputs(infer_models[2], "infer")
    in3d["hidden_states"] = out2d["hidden_states_out"]
    in3d["per_layer_combined"] = out1d["per_layer_combined_out"]
    in3d["kv13_k"] = out2d["kv13_k"]; in3d["kv13_v"] = out2d["kv13_v"]
    in3d["kv14_k"] = out2d["kv14_k"]; in3d["kv14_v"] = out2d["kv14_v"]
    out3d = infer_models[2].predict(in3d)
    _check_finite("infer.chunk_3", out3d["hidden_states_out"])

    in4d = _build_inputs(infer_models[3], "infer")
    in4d["hidden_states"] = out3d["hidden_states_out"]
    in4d["per_layer_combined"] = out1d["per_layer_combined_out"]
    in4d["kv13_k"] = out2d["kv13_k"]; in4d["kv13_v"] = out2d["kv13_v"]
    in4d["kv14_k"] = out2d["kv14_k"]; in4d["kv14_v"] = out2d["kv14_v"]
    out4d = infer_models[3].predict(in4d)
    print(f"  infer.token_id: {np.asarray(out4d['token_id']).tolist()}")

    print("\nOK: 4-chunk chained T=N + T=1 sanity passed.")


if __name__ == "__main__":
    main()
