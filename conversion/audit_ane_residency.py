#!/usr/bin/env python3
"""Audit ANE residency of a compiled .mlmodelc / .mlpackage via MLComputePlan.

Mirrors the Swift ComputePlanAudit logic but runs on macOS during conversion
so the 3-chunk consolidation gate (does the 17-layer merged chunk stay on
ANE?) can be answered without the iPhone in hand.

Usage:
    python conversion/audit_ane_residency.py <model.mlpackage> [<model2> ...]
"""
from __future__ import annotations

import sys
from collections import Counter

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def _iter_mlprogram_ops(model_structure):
    prog = getattr(model_structure, "program", None)
    if prog is None:
        return
    for func_name, func in prog.functions.items():
        for block in (func.block,):
            yield from _iter_block(block, func_name)


def _iter_block(block, func_name):
    for op in block.operations:
        yield func_name, op
        for nested in getattr(op, "blocks", ()) or ():
            yield from _iter_block(nested, func_name)


def _device_label(usage) -> str:
    if usage is None:
        return "unknown"
    # coremltools 9.0 renamed the attribute to `preferred_compute_device`;
    # earlier versions exposed it as `preferred`.  Try both so this works
    # against either coremltools.
    preferred = (
        getattr(usage, "preferred_compute_device", None)
        or getattr(usage, "preferred", None)
    )
    if preferred is None:
        return "unknown"
    name = type(preferred).__name__
    if "Neural" in name or "ANE" in name:
        return "ANE"
    if "GPU" in name:
        return "GPU"
    if "CPU" in name:
        return "CPU"
    return name


def audit(path: str) -> None:
    print(f"\n=== {path} ===")
    try:
        plan = MLComputePlan.load_from_path(path=path,
                                            compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as e:
        print(f"  load_from_path failed: {e}")
        return

    ms = plan.model_structure
    by_device = Counter()
    by_device_by_op = Counter()
    total = 0

    for func_name, op in _iter_mlprogram_ops(ms):
        # `const` ops aren't dispatched to a compute device — skip so the
        # ANE/CPU/GPU percentages reflect actual compute work, not weights.
        if op.operator_name == "const":
            continue
        try:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        except Exception:
            usage = None
        dev = _device_label(usage)
        by_device[dev] += 1
        by_device_by_op[(op.operator_name, dev)] += 1
        total += 1

    if total == 0:
        print("  (no MLProgram operations found — not an MLProgram?)")
        return

    print(f"  total ops: {total}")
    for dev, n in sorted(by_device.items(), key=lambda kv: -kv[1]):
        print(f"    {dev}: {n}  ({100.0 * n / total:.2f}%)")

    non_ane = [(op, dev, n) for (op, dev), n in by_device_by_op.items() if dev != "ANE"]
    if non_ane:
        print("  non-ANE ops:")
        for op, dev, n in sorted(non_ane, key=lambda t: -t[2]):
            print(f"    [{dev:3s}] {op:<24s} {n}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        audit(p)


if __name__ == "__main__":
    main()
