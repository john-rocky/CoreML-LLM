#!/usr/bin/env python3
"""A/B: native `Qwen3RMSNorm` vs the shared `ANERMSNorm` cat/chunk trick on the ANE.

Isolates the RMSNorm half of the incidental finding in docs/PPLX_EMBED_GPU_RESIDENCY.md:
a from-scratch "GPU-native" encoder rebuild ran the *ANE* path ~21% faster (28.6 vs
36.1 ms at L=256), but it confounded three changes (Conv2d-1×1→Linear, cat/chunk→native
RMSNorm, layout). This script changes ONLY the RMSNorm (`norm_impl` ∈ {ane_cat, native},
see models/qwen3_encoder.py) and measures, for each L ∈ {256, 512}:

  * ANE residency  — % of non-const MLProgram ops the static planner puts on the ANE
                     (reuses audit_ane_residency.py's MLComputePlan pattern). The gate:
                     native must STAY on the ANE (≈ the ane_cat residency, ~99%).
  * latency        — CPU_AND_NE warm median of MLModel.predict (the metric that matters;
                     the ANE path is the one pplx-embed ships).
  * fidelity       — cosine of the pooled_fp16 output vs the fp32 `Reference` oracle
                     (gate ≥ 0.99). Built with `--output-mode pooled_fp16` so the Python
                     CoreML bridge can read the output.

Decision rule (printed at the end): native WINS if it is faster on CPU_AND_NE, keeps
residency ≥ ~99%, and holds cosine ≥ 0.99 at both L — then make it the pplx-embed
default. Otherwise keep ane_cat and record the negative result.

Usage:
    uv run python conversion/experiment_ane_rmsnorm.py
    uv run python conversion/experiment_ane_rmsnorm.py --lengths 256 512 --iters 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

NORM_IMPLS = ("ane_cat", "native")

# A representative multilingual paragraph; repeated/truncated to fill each bucket so the
# fidelity check exercises a realistic (long, non-trivial) input at every L.
SAMPLE_TEXT = (
    "Embeddings map text into dense vectors so that semantically similar passages land "
    "near each other. 東京は日本の首都であり、世界有数の大都市圏を形成しています。 "
    "La inteligencia artificial avanza rápido y transforma la búsqueda de información. "
    "Машинное обучение изменяет способы обработки естественного языка. "
    "Retrieval-augmented generation grounds large language models in external knowledge."
)


def _build(hf_repo: str, norm_impl: str, L: int, out_root: str) -> str:
    """Build a pooled_fp16 bucket for (norm_impl, L); returns the .mlpackage path."""
    from build_pplx_embed_bundle import build_bundle

    out_dir = os.path.join(out_root, f"{norm_impl}-L{L}")
    return build_bundle(
        hf_repo, "pplx-embed", out_dir, max_seq_len=L, output_mode="pooled_fp16",
        quantize=None, variant="plain", dynamic_upper=0, skip_if_exists=True,
        norm_impl=norm_impl,
    )


def _compile(pkg: str) -> str:
    """Compile an .mlpackage → .mlmodelc via `xcrun coremlcompiler` (skip if present).

    MLComputePlan.load_from_path wants a *compiled* model; pointed at a raw .mlpackage
    it hard-aborts (uncatchable C++ exception). Compile once, reuse.
    """
    import subprocess

    out_dir = os.path.dirname(pkg)
    mlmodelc = os.path.join(out_dir, "encoder.mlmodelc")
    if not os.path.isdir(mlmodelc):
        subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out_dir],
                       check=True, capture_output=True)
    return mlmodelc


def _ane_residency(mlmodelc: str) -> tuple[float, int, Counter]:
    """Static op→device tally via MLComputePlan. Returns (ANE %, total ops, by-device)."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    from audit_ane_residency import _iter_mlprogram_ops, _device_label

    plan = MLComputePlan.load_from_path(path=mlmodelc, compute_units=ct.ComputeUnit.CPU_AND_NE)
    by_device: Counter = Counter()
    total = 0
    for _func, op in _iter_mlprogram_ops(plan.model_structure):
        if op.operator_name == "const":
            continue
        try:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        except Exception:
            usage = None
        by_device[_device_label(usage)] += 1
        total += 1
    ane_pct = 100.0 * by_device.get("ANE", 0) / total if total else 0.0
    return ane_pct, total, by_device


def _make_inputs(tokenizer, L: int):
    """Tokenize SAMPLE_TEXT (repeated to ~fill L), pad to L. Returns (inputs, n_valid)."""
    text = SAMPLE_TEXT
    # Repeat until the tokenized length comfortably exceeds L, then truncate to L.
    while len(tokenizer.encode(text)) < L:
        text = text + " " + SAMPLE_TEXT
    enc = tokenizer([text], return_tensors="np", truncation=True, max_length=L)
    ids = enc["input_ids"][0].astype(np.int32)
    n = int(ids.shape[0])
    pid = np.zeros((1, L), dtype=np.int32)
    pid[0, :n] = ids
    pam = np.zeros((1, L), dtype=np.float16)
    pam[0, :n] = 1.0
    return {"input_ids": pid, "attention_mask": pam}, n


def _latency_and_output(pkg: str, inputs: dict, iters: int, warmup: int):
    """CPU_AND_NE warm median latency (ms) + the pooled fp16 output of the last run."""
    import coremltools as ct

    m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    out = None
    for _ in range(warmup):
        out = m.predict(inputs)
    times = []
    for _ in range(iters):
        t = time.time()
        out = m.predict(inputs)
        times.append((time.time() - t) * 1000.0)
    emb = np.asarray(out["embedding"]).astype(np.float32).reshape(1, -1)
    return float(np.median(times)), float(np.mean(times)), emb


def main() -> int:
    ap = argparse.ArgumentParser(description="ANE RMSNorm A/B (native vs cat/chunk)")
    ap.add_argument("--hf-repo", default=None, help="Override HF repo / local dir")
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "..", "output",
                                                       "pplx-embed-rmsnorm-ab"))
    ap.add_argument("--fidelity-gate", type=float, default=0.99)
    args = ap.parse_args()

    from config import MODEL_REGISTRY
    hf_repo = args.hf_repo or MODEL_REGISTRY["pplx-embed"].hf_repo

    import pplx_embed_reference as R
    print(f"[ref] loading fp32 oracle {hf_repo} …")
    ref = R.Reference(hf_repo)
    tok = ref.tokenizer

    # results[(impl, L)] = dict(ane_pct, total, latency_med, latency_mean, cosine, n)
    results: dict[tuple[str, int], dict] = {}
    for L in args.lengths:
        inputs, n = _make_inputs(tok, L)
        # fp32 reference pooled (pre-tanh) for this exact (truncated) input.
        import torch
        ids_t = torch.from_numpy(inputs["input_ids"]).to(torch.long)[:, :n]
        mask_t = torch.ones((1, n), dtype=torch.float32)
        with torch.inference_mode():
            hidden = ref.model(input_ids=ids_t, attention_mask=mask_t).last_hidden_state.float()
            ref_pooled = R.masked_mean(hidden, mask_t).numpy().astype(np.float32)

        for impl in NORM_IMPLS:
            print(f"\n=== norm_impl={impl}  L={L}  (n_valid={n}) ===")
            pkg = _build(hf_repo, impl, L, args.out_root)
            mlmodelc = _compile(pkg)
            ane_pct, total, by_dev = _ane_residency(mlmodelc)
            print(f"  residency: ANE {ane_pct:.2f}%  ({total} ops; "
                  f"{dict(by_dev)})")
            med, mean, emb = _latency_and_output(pkg, inputs, args.iters, args.warmup)
            cos = float(R.cosine_similarity(emb, ref_pooled)[0])
            print(f"  CPU_AND_NE latency: median {med:.2f} ms  mean {mean:.2f} ms")
            print(f"  fidelity cosine vs fp32: {cos:.6f}  (gate ≥ {args.fidelity_gate})")
            results[(impl, L)] = dict(ane_pct=ane_pct, total=total, latency_med=med,
                                      latency_mean=mean, cosine=cos, n=n)

    # ---- summary + decision -------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY  (norm_impl × L)")
    print("=" * 72)
    print(f"  {'impl':<8} {'L':>5} {'ANE%':>7} {'lat_med(ms)':>12} {'cosine':>9}")
    for L in args.lengths:
        for impl in NORM_IMPLS:
            r = results[(impl, L)]
            print(f"  {impl:<8} {L:>5} {r['ane_pct']:>7.2f} {r['latency_med']:>12.2f} "
                  f"{r['cosine']:>9.5f}")

    print("\nDECISION")
    native_wins_all = True
    for L in args.lengths:
        a = results[("ane_cat", L)]
        nv = results[("native", L)]
        speedup = (a["latency_med"] / nv["latency_med"] - 1.0) * 100.0
        faster = nv["latency_med"] < a["latency_med"]
        resident = nv["ane_pct"] >= 0.99 * a["ane_pct"] and nv["ane_pct"] >= 99.0
        fid_ok = nv["cosine"] >= args.fidelity_gate
        verdict = "WIN" if (faster and resident and fid_ok) else "no"
        if verdict != "WIN":
            native_wins_all = False
        print(f"  L={L}: native {nv['latency_med']:.2f} vs ane_cat {a['latency_med']:.2f} ms "
              f"({speedup:+.1f}% {'faster' if faster else 'slower'}); "
              f"ANE {nv['ane_pct']:.2f}% (resident={resident}); "
              f"cosine {nv['cosine']:.5f} (ok={fid_ok}) → {verdict}")

    print()
    if native_wins_all:
        print("  ✅ native RMSNorm WINS at all L → make norm_impl='native' the pplx-embed "
              "default (build_pplx_embed_bundle.py / encoder).")
    else:
        print("  ❌ native does not clear the gate at every L → keep ane_cat; record the "
              "negative result. (See per-L lines above.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
