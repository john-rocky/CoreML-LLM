#!/usr/bin/env python3
"""Mac sanity-check: load compiled chunk3 .mlmodelc and run a single forward
pass with zero inputs. Verify it loads, runs, and produces a sane token_id.
This is NOT a perf test — actual A/B happens on iPhone.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import coremltools as ct


def run_chunk3(mlmodelc_path: str, mlpackage_path: str) -> None:
    print(f"\n[sanity] {mlmodelc_path}")
    if not os.path.isdir(mlmodelc_path):
        sys.exit(f"  not found: {mlmodelc_path}")

    # Read input/output spec from the source .mlpackage; CompiledMLModel doesn't
    # expose get_spec().
    pkg_model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = pkg_model.get_spec()

    t = time.time()
    model = ct.models.CompiledMLModel(
        mlmodelc_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  loaded mlmodelc in {time.time()-t:.1f}s")

    input_specs = {i.name: tuple(int(d) for d in i.type.multiArrayType.shape)
                   for i in spec.description.input}
    print(f"  inputs ({len(input_specs)}):")
    for name, shp in input_specs.items():
        print(f"    {name:<26} {shp}")

    inputs = {name: np.zeros(shp, dtype=np.float16) for name, shp in input_specs.items()}

    t = time.time()
    out = model.predict(inputs)
    print(f"\n  predict in {(time.time()-t)*1000:.1f}ms")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"    {k:<26} shape={tuple(v.shape)}  dtype={v.dtype}  "
                  f"sample={v.flat[0]}")
        else:
            print(f"    {k:<26} {v}")

    tok = out.get("token_id")
    if tok is None:
        sys.exit("  ❌ no token_id output")
    tok_v = int(np.asarray(tok).flat[0])
    if tok_v < 0 or tok_v >= 262144:
        sys.exit(f"  ❌ token_id={tok_v} out of vocab range")
    print(f"  ✅ token_id={tok_v} (valid)")


def main():
    pairs = [
        ("output/gemma4-e2b/chunks_3way_lmsplit8/chunk3_3way.mlmodelc",
         "output/gemma4-e2b/chunks_3way_lmsplit8/chunk3_3way.mlpackage"),
        ("output/gemma4-e2b/chunks_3way_lmsplit16/chunk3_3way.mlmodelc",
         "output/gemma4-e2b/chunks_3way_lmsplit16/chunk3_3way.mlpackage"),
    ]
    for mc, mp in pairs:
        run_chunk3(mc, mp)
    print("\n[sanity] all variants loaded and produced valid output")


if __name__ == "__main__":
    main()
