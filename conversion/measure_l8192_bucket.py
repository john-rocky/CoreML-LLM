#!/usr/bin/env python3
"""Workstream B gate: is a fixed L=8192 ANE bucket worth shipping?

The >max-bucket catch-all today is a flexible RangeDim **GPU** model (~10× slower than a
fixed ANE bucket, because flexible shapes force ANE fallback). Inputs of 4097–8192 tokens
take that slow path. This measures whether a **fixed L=8192 ANE bucket** instead:

  1. **stays on the ANE** — compile the int8 bucket, tally op→device via MLComputePlan
     (reuses audit_ane_residency.py). Gate: ~99% ANE (like the smaller buckets), not a
     fall-off to CPU/GPU.
  2. **is faster** than the dynamic GPU catch-all at a long (~8000-token) input — warm
     median of MLModel.predict: ANE bucket (pooled_fp16, padded to 8192, CPU_AND_NE) vs
     the dyn8192 GPU model (pooled_fp16, non-padded actual length, CPU_AND_GPU).
  3. **holds fidelity** — cosine of both vs the fp32 `Reference` oracle (gate ≥ 0.99).

Ship decision (printed): ship iff ANE-resident AND faster than the GPU catch-all.

Prereqs (build first; each ~1.1 GB):
    python conversion/build_pplx_embed_bundle.py --model pplx-embed --max-seq-len 8192
    python conversion/build_pplx_embed_bundle.py --model pplx-embed --max-seq-len 8192 \
        --output-mode pooled_fp16
    python conversion/build_pplx_embed_bundle.py --model pplx-embed --dynamic-upper 8192 \
        --output-mode pooled_fp16    # the GPU catch-all to compare against

Usage:
    uv run python conversion/measure_l8192_bucket.py --n-tokens 8000 --iters 5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "..", "output", "pplx-embed")
ANE_FP16 = os.path.join(OUT, "L8192-pooled_fp16", "encoder.mlpackage")
ANE_INT8 = os.path.join(OUT, "L8192-int8", "encoder.mlpackage")
DYN_FP16 = os.path.join(OUT, "dyn8192-pooled_fp16", "encoder.mlpackage")

SAMPLE = (
    "Embeddings map text into dense vectors so semantically similar passages land near "
    "each other. 東京は日本の首都であり、世界有数の大都市圏を形成しています。 "
    "La inteligencia artificial avanza rápido. Машинное обучение меняет обработку языка. "
    "Retrieval-augmented generation grounds large language models in external knowledge. "
)


def _compile(pkg: str) -> str:
    out_dir = os.path.dirname(pkg)
    mlmodelc = os.path.join(out_dir, "encoder.mlmodelc")
    if not os.path.isdir(mlmodelc):
        subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out_dir],
                       check=True, capture_output=True)
    return mlmodelc


def _residency(mlmodelc: str) -> tuple[float, int, Counter]:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    from audit_ane_residency import _iter_mlprogram_ops, _device_label

    plan = MLComputePlan.load_from_path(path=mlmodelc, compute_units=ct.ComputeUnit.CPU_AND_NE)
    by_dev: Counter = Counter()
    total = 0
    for _f, op in _iter_mlprogram_ops(plan.model_structure):
        if op.operator_name == "const":
            continue
        try:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        except Exception:
            usage = None
        by_dev[_device_label(usage)] += 1
        total += 1
    return (100.0 * by_dev.get("ANE", 0) / total if total else 0.0), total, by_dev


def _time(pkg: str, inputs: dict, units, iters: int, warmup: int):
    import coremltools as ct
    m = ct.models.MLModel(pkg, compute_units=units)
    out = None
    for _ in range(warmup):
        out = m.predict(inputs)
    ts = []
    for _ in range(iters):
        t = time.time()
        out = m.predict(inputs)
        ts.append((time.time() - t) * 1000.0)
    emb = np.asarray(out["embedding"]).astype(np.float32).reshape(1, -1)
    return float(np.median(ts)), emb


def main() -> int:
    ap = argparse.ArgumentParser(description="L=8192 ANE bucket ship gate")
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--n-tokens", type=int, default=8000, help="valid tokens in the long input")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--fidelity-gate", type=float, default=0.99)
    args = ap.parse_args()

    for p, what in ((ANE_FP16, "L8192 pooled_fp16"), (DYN_FP16, "dyn8192 pooled_fp16")):
        if not os.path.isdir(p):
            print(f"MISSING {what}: {p}\n(build it first — see this script's header.)")
            return 1

    from config import MODEL_REGISTRY
    hf_repo = args.hf_repo or MODEL_REGISTRY["pplx-embed"].hf_repo
    import pplx_embed_reference as R
    import torch
    print(f"[ref] loading fp32 oracle {hf_repo} …")
    ref = R.Reference(hf_repo)
    tok = ref.tokenizer

    # Long multilingual input → n_tokens valid tokens.
    text = SAMPLE
    while len(tok.encode(text)) < args.n_tokens:
        text = text + " " + SAMPLE
    ids = tok([text], return_tensors="np", truncation=True, max_length=args.n_tokens)["input_ids"][0]
    n = int(ids.shape[0])
    print(f"[input] {n} valid tokens")

    # fp32 reference pooled (this is the heavy step at long L).
    print("[ref] fp32 forward (slow at long L) …")
    with torch.inference_mode():
        ids_t = torch.from_numpy(ids.astype(np.int64)).view(1, -1)
        mask_t = torch.ones((1, n), dtype=torch.float32)
        hidden = ref.model(input_ids=ids_t, attention_mask=mask_t).last_hidden_state.float()
        ref_pooled = R.masked_mean(hidden, mask_t).numpy().astype(np.float32)

    # --- 1. residency (int8 bucket if built, else the pooled_fp16 encoder body) -------
    res_pkg = ANE_INT8 if os.path.isdir(ANE_INT8) else ANE_FP16
    print(f"\n[1] ANE residency of {os.path.relpath(res_pkg, OUT)} …")
    ane_pct, total, by_dev = _residency(_compile(res_pkg))
    print(f"    ANE {ane_pct:.2f}%  ({total} ops; {dict(by_dev)})")

    import coremltools as ct
    L = 8192
    # --- 2a. ANE bucket: pad to 8192, CPU_AND_NE ------------------------------------
    pid = np.zeros((1, L), dtype=np.int32)
    pid[0, :n] = ids
    pam = np.zeros((1, L), dtype=np.float16)
    pam[0, :n] = 1.0
    print(f"\n[2a] ANE L8192 bucket latency (CPU_AND_NE, padded to {L}) …")
    ane_ms, ane_emb = _time(ANE_FP16, {"input_ids": pid, "attention_mask": pam},
                            ct.ComputeUnit.CPU_AND_NE, args.iters, args.warmup)
    ane_cos = float(R.cosine_similarity(ane_emb, ref_pooled)[0])
    print(f"     median {ane_ms:.1f} ms   cosine {ane_cos:.5f}")

    # --- 2b. dynamic GPU model: actual length, CPU_AND_GPU --------------------------
    did = ids.astype(np.int32).reshape(1, n)
    dam = np.ones((1, n), dtype=np.float16)
    print(f"\n[2b] dynamic GPU model latency (CPU_AND_GPU, actual {n}) …")
    gpu_ms, gpu_emb = _time(DYN_FP16, {"input_ids": did, "attention_mask": dam},
                            ct.ComputeUnit.CPU_AND_GPU, args.iters, args.warmup)
    gpu_cos = float(R.cosine_similarity(gpu_emb, ref_pooled)[0])
    print(f"     median {gpu_ms:.1f} ms   cosine {gpu_cos:.5f}")

    # --- decision -------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("L=8192 SHIP GATE")
    print("=" * 64)
    resident = ane_pct >= 99.0
    faster = ane_ms < gpu_ms
    fid_ok = ane_cos >= args.fidelity_gate
    speedup = gpu_ms / ane_ms if ane_ms else 0.0
    print(f"  ANE residency : {ane_pct:.2f}%   (resident ≥99% = {resident})")
    print(f"  latency       : ANE {ane_ms:.1f} ms  vs  GPU {gpu_ms:.1f} ms  "
          f"({speedup:.1f}× {'faster' if faster else 'SLOWER'})")
    print(f"  fidelity      : ANE cosine {ane_cos:.5f}  (≥{args.fidelity_gate} = {fid_ok})")
    ship = resident and faster and fid_ok
    print()
    if ship:
        print("  ✅ SHIP: L=8192 stays on the ANE and beats the GPU catch-all. Place "
              "L8192-int8/ in the bundle dir; Swift auto-routes (no Swift change).")
    else:
        print("  ❌ DO NOT SHIP: gate not met (see above). The >4096 path stays the GPU "
              "model; record the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
