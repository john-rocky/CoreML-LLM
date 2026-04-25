#!/usr/bin/env python3
"""Compare MIL op mix + ANE placement for Qwen3-VL 2B chunk_0
Conv2dLinear vs nn.Linear bundles."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import coremltools as ct

ROOT = Path("/tmp/q3vl_linear_test")


def audit(label: str, pkg_path: Path):
    print(f"\n[{label}] {pkg_path}")
    if not pkg_path.is_dir():
        print(f"  missing: {pkg_path}")
        return
    m = ct.models.MLModel(str(pkg_path),
                          compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = m.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    op_counter: Counter[str] = Counter()
    dev_counter: Counter[str] = Counter()
    for fn in plan.model_structure.program.functions.values():
        for op in fn.block.operations:
            op_counter[op.operator_name] += 1
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            d = ("const" if (a is None and op.operator_name == "const")
                 else (a.preferred_compute_device.__class__.__name__ if a else "unknown"))
            dev_counter[d] += 1

    total = sum(dev_counter.values())
    compute = total - dev_counter.get("const", 0)
    ane = dev_counter.get("MLNeuralEngineComputeDevice", 0)
    pct = 100 * ane / compute if compute else 0.0
    print(f"  total ops:    {total}")
    print(f"  compute ops:  {compute} (excl const)")
    print(f"  ANE %:        {pct:.1f}  ({ane}/{compute})")
    print(f"  top device split:")
    for d, n in dev_counter.most_common():
        print(f"    {d:42s} {n}")
    print(f"  top 12 op kinds:")
    for op_name, n in op_counter.most_common(12):
        print(f"    {op_name:32s} {n}")
    return op_counter, total, compute, ane


def main():
    a = audit("Conv2dLinear", ROOT / "conv" / "qwen3_vl_2b_stateful_chunks" / "chunk_0.mlpackage")
    b = audit("nn.Linear",    ROOT / "linear" / "qwen3_vl_2b_stateful_chunks" / "chunk_0.mlpackage")
    if a and b:
        oa, ta, ca, na = a
        ob, tb, cb, nb = b
        print("\n=== summary ===")
        print(f"  total ops    Conv2d={ta}  Linear={tb}  Δ={tb-ta:+d}  ({(tb-ta)/ta*100:+.1f} %)")
        print(f"  compute ops  Conv2d={ca}  Linear={cb}  Δ={cb-ca:+d}  ({(cb-ca)/ca*100:+.1f} %)")
        print(f"  ANE %        Conv2d={100*na/ca:.1f}  Linear={100*nb/cb:.1f}")


if __name__ == "__main__":
    main()
