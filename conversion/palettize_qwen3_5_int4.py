"""Phase 4b: palettize the fp16 Qwen3.5-0.8B mlpackage to INT4.

Loads an existing fp16 mlpackage (produced by Phase 4a), applies INT4
palettization using the Gemma 4 / MTP recipe (kmeans, per_grouped_channel,
group_size=32), saves the INT4 mlpackage, then runs `.predict()` against the
Phase 1 oracle prompts on-device and verifies parity cos >= 0.995 and top-1
next-token match.

Usage:
    python palettize_qwen3_5_int4.py \\
        --fp16-path /tmp/qwen3_5_0_8b_fp16_seq64.mlpackage \\
        --out-path /tmp/qwen3_5_0_8b_int4_seq64.mlpackage
"""
from collections import Counter
from pathlib import Path
import argparse
import time

import numpy as np
import torch

import coremltools as ct
import coremltools.optimize.coreml as cto


ORACLE_PATH = Path(__file__).parent / "qwen3_5_reference_logits.pt"
DEFAULT_SEQ_LEN = 64


def palettize(fp16_path: Path, out_path: Path):
    print(f"loading fp16 mlpackage: {fp16_path}")
    t0 = time.time()
    mlm = ct.models.MLModel(str(fp16_path))
    print(f"  loaded in {time.time()-t0:.1f}s")

    cfg = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(
            mode="kmeans", nbits=4,
            granularity="per_grouped_channel", group_size=32,
        )
    )
    print("palettizing (kmeans, nbits=4, group_size=32, per_grouped_channel)...")
    t1 = time.time()
    mlm_q = cto.palettize_weights(mlm, cfg)
    print(f"  palettized in {time.time()-t1:.1f}s")

    mlm_q.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved {out_path} ({size_mb:.1f} MB)")
    return out_path, size_mb


def cos_sim(a, b):
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-12))


def parity_predict(mlpackage_path: Path, oracle: dict, seq_len: int, pad_id: int = 0):
    """Load the mlpackage on-device and run `predict()` for each oracle prompt.
    Returns (worst_pos_cos, mean_cos, top1_rate)."""
    print(f"\n=== parity via MLModel.predict ({mlpackage_path.name}) ===")
    mlm = ct.models.MLModel(str(mlpackage_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    per_prompt = []
    for rec in oracle["records"]:
        ids = rec["input_ids"].numpy()  # (1, S_real)
        S_real = ids.shape[1]
        if S_real > seq_len:
            continue
        padded = np.full((1, seq_len), pad_id, dtype=np.int32)
        padded[:, :S_real] = ids.astype(np.int32)

        out = mlm.predict({"input_ids": padded})
        logits = out["logits"]  # (1, seq_len, V)
        logits_real = logits[0, :S_real, :]  # (S_real, V)
        ref = rec["logits_prefill"].float().numpy()  # (S_real, V)

        per_pos_cos = np.array([cos_sim(logits_real[i], ref[i]) for i in range(S_real)])
        mean_c = float(per_pos_cos.mean())
        min_c = float(per_pos_cos.min())
        last_c = float(per_pos_cos[-1])

        pred_top1 = int(np.argmax(logits_real[-1]))
        ref_top1 = int(rec["top10_last_ids"][0].item())
        top1 = pred_top1 == ref_top1

        print(f"  S={S_real:3d}  mean={mean_c:.6f}  min={min_c:.6f}  last={last_c:.6f}  "
              f"top1_match={top1}  prompt={rec['prompt'][:35]!r}")
        per_prompt.append({"mean": mean_c, "min": min_c, "top1": top1})

    worst = min(p["min"] for p in per_prompt)
    mean = sum(p["mean"] for p in per_prompt) / len(per_prompt)
    top1_rate = sum(p["top1"] for p in per_prompt) / len(per_prompt)
    print(f"\n  overall: mean={mean:.6f}  worst_pos={worst:.6f}  "
          f"top1_match={top1_rate*100:.0f}% "
          f"({sum(p['top1'] for p in per_prompt)}/{len(per_prompt)})")
    return worst, mean, top1_rate


def audit_placement(path: Path):
    """Same audit as Phase 4a — excludes static const from the ANE% denominator."""
    print(f"\n=== placement audit ({path.name}) ===")
    reloaded = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    dev_counts: Counter = Counter()
    op_type_by_dev: dict[str, Counter] = {}
    for func_name, func in program.functions.items():
        for op in func.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if a is None:
                dev = "const" if op.operator_name == "const" else "unknown"
            else:
                dev = a.preferred_compute_device.__class__.__name__
            dev_counts[dev] += 1
            op_type_by_dev.setdefault(dev, Counter())[op.operator_name] += 1

    total = sum(dev_counts.values())
    ane = dev_counts.get("MLNeuralEngineComputeDevice", 0)
    cpu = dev_counts.get("MLCPUComputeDevice", 0)
    gpu = dev_counts.get("MLGPUComputeDevice", 0)
    const = dev_counts.get("const", 0)
    unknown = dev_counts.get("unknown", 0)
    compute_total = ane + cpu + gpu + unknown
    ane_pct = 100 * ane / compute_total if compute_total else 0.0

    print(f"  total ops: {total}  (compute={compute_total}, const={const})")
    print(f"    ANE:     {ane} ({ane_pct:.2f}% of compute)")
    print(f"    CPU:     {cpu}")
    if gpu: print(f"    GPU:     {gpu}")
    if unknown: print(f"    unknown: {unknown}")
    for dev in ("MLCPUComputeDevice", "MLNeuralEngineComputeDevice"):
        c = op_type_by_dev.get(dev, Counter())
        if not c:
            continue
        print(f"  === {dev} top ops ===")
        for op_type, n in c.most_common(15):
            print(f"    {op_type}: {n}")
    return ane_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16-path", required=True, type=str)
    ap.add_argument("--out-path", required=True, type=str)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--skip-palettize", action="store_true",
                    help="skip palettization; only re-run parity/audit against --out-path")
    args = ap.parse_args()

    fp16_path = Path(args.fp16_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_palettize:
        palettize(fp16_path, out_path)

    oracle = torch.load(ORACLE_PATH, map_location="cpu", weights_only=False)

    print("\n== fp16 baseline ==")
    worst_fp16, mean_fp16, top1_fp16 = parity_predict(fp16_path, oracle, args.seq_len)

    print("\n== INT4 palettized ==")
    worst_q, mean_q, top1_q = parity_predict(out_path, oracle, args.seq_len)

    ane_pct = audit_placement(out_path)
    int4_size = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    fp16_size = sum(f.stat().st_size for f in fp16_path.rglob('*') if f.is_file()) / 1e6

    print(f"\n=== Phase 4b summary ===")
    print(f"  fp16: {fp16_size:.1f} MB  parity worst_pos={worst_fp16:.6f}  top1={top1_fp16*100:.0f}%")
    print(f"  int4: {int4_size:.1f} MB  parity worst_pos={worst_q:.6f}  top1={top1_q*100:.0f}%  "
          f"ANE={ane_pct:.2f}%")
    print(f"  compression: {fp16_size/int4_size:.2f}x")
    if worst_q >= 0.995 and ane_pct >= 95.0 and top1_q >= 0.9:
        print("  GATE PASSED")
    else:
        print(f"  GATE FAILED (need worst_pos>=0.995, ANE>=95%, top1>=90%)")


if __name__ == "__main__":
    main()
