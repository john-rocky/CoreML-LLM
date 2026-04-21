"""Validate probe methodology: does Gemma 4 E2B (known 99.78% ANE in prod) also
show high ANE placement via MLComputePlan on this Mac? If yes, our single-op
probes are giving real signal. If no, single-op probes are misleading on Mac.
"""
import sys
from pathlib import Path
from collections import Counter

import coremltools as ct


def main(mlpackage_path: str):
    path = Path(mlpackage_path)
    print(f"loading: {path}")
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = model.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    program = plan.model_structure.program
    if program is None:
        print("no program")
        return

    total_counts: Counter = Counter()
    op_type_by_dev: dict[str, Counter] = {}
    for func_name, func in program.functions.items():
        for op in func.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = a.preferred_compute_device.__class__.__name__ if a else "unknown"
            total_counts[dev] += 1
            op_type_by_dev.setdefault(dev, Counter())[op.operator_name] += 1

    total = sum(total_counts.values())
    print(f"\ntotal ops: {total}")
    for dev, n in total_counts.most_common():
        print(f"  {dev}: {n} ({100*n/total:.1f}%)")
    for dev, counter in op_type_by_dev.items():
        print(f"\n=== {dev} op types (top 10) ===")
        for op_type, n in counter.most_common(10):
            print(f"  {op_type}: {n}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/output/gemma4-e2b-final/model.mlpackage")
