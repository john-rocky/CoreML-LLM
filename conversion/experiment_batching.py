#!/usr/bin/env python3
"""Rigorous CoreML batching throughput experiment for the pplx-embed encoder.

Question: does batching (B>1) give throughput gains on Apple Silicon, and if the
earlier quick test showed FLAT docs/sec, is that real (and why) or a measurement
blind spot?

For each (L, B) we build a pooled_fp16 batched encoder, convert with coremltools
(shape (B, L)), then load+time it under three compute-unit settings and compute
per-doc latency = batch_latency / B and docs/sec = B / batch_latency.

We also:
  - audit the actual compute-device placement (MLComputePlan) of a batched model,
  - run a control (1 batch-N predict vs N sequential B=1 predicts),
  - sanity-check that batching is real (distinct input rows -> distinct outputs).

Run:
    uv run python conversion/experiment_batching.py            # full sweep
    uv run python conversion/experiment_batching.py --quick    # smaller sweep
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct  # noqa: E402
from coremltools.models.compute_plan import MLComputePlan  # noqa: E402

from models.qwen3_encoder import (  # noqa: E402
    Qwen3EncoderConfig,
    PplxEmbedModel,
    load_encoder_weights,
    apply_fp16_residual_rescale,
)

HF_REPO = "perplexity-ai/pplx-embed-v1-0.6b"
RESCALE_K = 8.0

CU = {
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
}


def snapshot_dir() -> str:
    if os.path.isdir(HF_REPO):
        return HF_REPO
    from huggingface_hub import snapshot_download
    return snapshot_download(
        HF_REPO,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.py", "1_Pooling/*"],
    )


_TORCH_CACHE: dict[int, PplxEmbedModel] = {}


def torch_model(snap: str, rope_len: int) -> PplxEmbedModel:
    """Build (once per rope_len) the weight-loaded, rescaled torch encoder."""
    if rope_len in _TORCH_CACHE:
        return _TORCH_CACHE[rope_len]
    cfg = Qwen3EncoderConfig.from_json(os.path.join(snap, "config.json"), max_seq_len=rope_len)
    model = PplxEmbedModel(cfg, output_mode="pooled_fp16").eval()
    load_encoder_weights(model.encoder, snap)
    apply_fp16_residual_rescale(model.encoder, RESCALE_K)
    _TORCH_CACHE[rope_len] = model
    return model


def build_mlpackage(snap: str, L: int, B: int, out_dir: str) -> str:
    """Convert a (B, L) pooled_fp16 encoder to a .mlpackage. Cached on disk."""
    pkg = os.path.join(out_dir, f"enc_L{L}_B{B}.mlpackage")
    if os.path.exists(pkg):
        return pkg
    os.makedirs(out_dir, exist_ok=True)
    model = torch_model(snap, rope_len=L)
    sample_ids = torch.zeros((B, L), dtype=torch.int32)
    sample_mask = torch.ones((B, L), dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(model, (sample_ids, sample_mask))
    inputs = [
        ct.TensorType(name="input_ids", shape=(B, L), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(B, L), dtype=np.float16),
    ]
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="embedding", dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS26,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(pkg)
    del traced, mlmodel
    gc.collect()
    return pkg


def make_inputs(B: int, L: int, distinct: bool = False) -> dict:
    rng = np.random.default_rng(0)
    if distinct:
        ids = rng.integers(1, 5000, size=(B, L)).astype(np.int32)
    else:
        ids = rng.integers(1, 5000, size=(1, L)).astype(np.int32)
        ids = np.repeat(ids, B, axis=0).astype(np.int32)
    mask = np.ones((B, L), dtype=np.float16)
    return {"input_ids": ids, "attention_mask": mask}


def time_model(mlmodel, feeds: dict, n_warm: int = 3, n_runs: int = 8) -> dict:
    for _ in range(n_warm):
        mlmodel.predict(feeds)
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        mlmodel.predict(feeds)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return {
        "median_s": samples[len(samples) // 2],
        "min_s": samples[0],
        "max_s": samples[-1],
        "runs": samples,
    }


def device_label(usage) -> str:
    if usage is None:
        return "unknown"
    pref = getattr(usage, "preferred_compute_device", None) or getattr(usage, "preferred", None)
    if pref is None:
        return "unknown"
    name = type(pref).__name__
    if "Neural" in name or "ANE" in name:
        return "ANE"
    if "GPU" in name:
        return "GPU"
    if "CPU" in name:
        return "CPU"
    return name


def _iter_ops(ms):
    prog = getattr(ms, "program", None)
    if prog is None:
        return
    for fn, func in prog.functions.items():
        stack = [func.block]
        while stack:
            blk = stack.pop()
            for op in blk.operations:
                yield op
                for nb in getattr(op, "blocks", ()) or ():
                    stack.append(nb)


_COMPILE_CACHE: dict[str, str] = {}


def _compiled_path(pkg: str) -> str:
    """Compile an .mlpackage to a persistent .mlmodelc once (MLComputePlan needs
    a compiled model, and the temp one from get_compiled_model_path() is deleted
    when its MLModel is GC'd — so copy it to a stable location)."""
    if pkg in _COMPILE_CACHE:
        return _COMPILE_CACHE[pkg]
    import shutil
    m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_ONLY)
    tmp = m.get_compiled_model_path()
    dst = pkg.replace(".mlpackage", ".mlmodelc")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(tmp, dst)   # copy before `m` is GC'd / tmp is cleaned
    del m
    _COMPILE_CACHE[pkg] = dst
    return dst


def audit_devices(pkg: str, compute_unit: ct.ComputeUnit) -> Counter:
    path = _compiled_path(pkg) if pkg.endswith(".mlpackage") else pkg
    plan = MLComputePlan.load_from_path(path=path, compute_units=compute_unit)
    ms = plan.model_structure
    by_dev = Counter()
    for op in _iter_ops(ms):
        if op.operator_name == "const":
            continue
        try:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        except Exception:
            usage = None
        by_dev[device_label(usage)] += 1
    return by_dev


def fmt_devs(c: Counter) -> str:
    tot = sum(c.values()) or 1
    return ", ".join(f"{d}:{n}({100*n/tot:.0f}%)" for d, n in c.most_common())


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sweep (B up to 16)")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "experiments", "batching_models"))
    ap.add_argument("--log", default=os.path.join(ROOT, "experiments", "batching_out.log"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    _logf = open(args.log, "w")
    sys.stdout = _Tee(sys.__stdout__, _logf)

    Ls = [128, 512]
    Bs = [1, 4, 16] if args.quick else [1, 4, 16, 64]
    units = ["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"]

    snap = snapshot_dir()
    print(f"snapshot: {snap}")
    print(f"Ls={Ls} Bs={Bs} units={units} runs={args.runs}\n")

    # ---- Build all needed mlpackages first (one torch model per L) ----
    pkgs: dict[tuple[int, int], str] = {}
    for L in Ls:
        for B in Bs:
            print(f"[build] L={L} B={B} ...", flush=True)
            pkgs[(L, B)] = build_mlpackage(snap, L, B, args.out_dir)
    # free torch
    _TORCH_CACHE.clear()
    gc.collect()

    # ---- Timing sweep ----
    # rows: (L, B, unit) -> result
    results = {}
    for unit in units:
        for L in Ls:
            for B in Bs:
                pkg = pkgs[(L, B)]
                try:
                    m = ct.models.MLModel(pkg, compute_units=CU[unit])
                except Exception as e:
                    print(f"  load FAIL L={L} B={B} {unit}: {e}")
                    continue
                feeds = make_inputs(B, L, distinct=False)
                try:
                    r = time_model(m, feeds, n_runs=args.runs)
                except Exception as e:
                    print(f"  predict FAIL L={L} B={B} {unit}: {e}")
                    del m
                    gc.collect()
                    continue
                bl = r["median_s"]
                results[(L, B, unit)] = {
                    "batch_lat_ms": bl * 1e3,
                    "per_doc_ms": bl / B * 1e3,
                    "docs_per_s": B / bl,
                }
                print(f"  L={L:4d} B={B:3d} {unit:12s} "
                      f"batch={bl*1e3:8.2f}ms  per-doc={bl/B*1e3:7.3f}ms  "
                      f"docs/s={B/bl:8.2f}")
                del m
                gc.collect()

    # ---- Print tables ----
    print("\n\n========== docs/sec  (rows=B, cols=unit) ==========")
    for L in Ls:
        print(f"\n--- L={L} ---")
        header = "  B   " + "".join(f"{u:>14s}" for u in units)
        print(header)
        for B in Bs:
            row = f"{B:4d}  "
            for u in units:
                r = results.get((L, B, u))
                row += f"{r['docs_per_s']:14.2f}" if r else f"{'-':>14s}"
            print(row)

    print("\n\n========== per-doc latency ms  (rows=B, cols=unit) ==========")
    for L in Ls:
        print(f"\n--- L={L} ---")
        header = "  B   " + "".join(f"{u:>14s}" for u in units)
        print(header)
        for B in Bs:
            row = f"{B:4d}  "
            for u in units:
                r = results.get((L, B, u))
                row += f"{r['per_doc_ms']:14.3f}" if r else f"{'-':>14s}"
            print(row)

    # ---- Speedup vs B=1 (docs/sec ratio) ----
    print("\n\n========== batch speedup = docs/s(B) / docs/s(B=1) ==========")
    for L in Ls:
        print(f"\n--- L={L} ---")
        header = "  B   " + "".join(f"{u:>14s}" for u in units)
        print(header)
        for B in Bs:
            row = f"{B:4d}  "
            for u in units:
                r = results.get((L, B, u))
                r1 = results.get((L, 1, u))
                if r and r1:
                    row += f"{r['docs_per_s']/r1['docs_per_s']:13.2f}x"
                else:
                    row += f"{'-':>14s}"
            print(row)

    # ---- Device audit for B=64 (or max B), L=128 ----
    maxB = Bs[-1]
    print(f"\n\n========== DEVICE PLACEMENT (L=128, B={maxB}) ==========")
    pkg = pkgs[(128, maxB)]
    for unit in units:
        try:
            c = audit_devices(pkg, CU[unit])
            print(f"  requested {unit:12s} -> {fmt_devs(c)}")
        except Exception as e:
            print(f"  audit {unit} failed: {e}")
    # also B=1 L=128 for contrast
    print(f"\n  (contrast) L=128 B=1:")
    for unit in units:
        try:
            c = audit_devices(pkgs[(128, 1)], CU[unit])
            print(f"  requested {unit:12s} -> {fmt_devs(c)}")
        except Exception as e:
            print(f"  audit {unit} failed: {e}")

    # ---- Control: 1x batch-N vs N x batch-1 (best compute unit per case) ----
    print(f"\n\n========== CONTROL: batch-N predict vs N sequential B=1 ==========")
    for unit in ["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"]:
        for L in Ls:
            B = maxB
            mN = ct.models.MLModel(pkgs[(L, B)], compute_units=CU[unit])
            m1 = ct.models.MLModel(pkgs[(L, 1)], compute_units=CU[unit])
            feedsN = make_inputs(B, L, distinct=True)
            feeds1_list = [
                {"input_ids": feedsN["input_ids"][i:i+1],
                 "attention_mask": feedsN["attention_mask"][i:i+1]}
                for i in range(B)
            ]
            # warm
            for _ in range(2):
                mN.predict(feedsN)
                m1.predict(feeds1_list[0])
            # batch-N
            tb = []
            for _ in range(5):
                t0 = time.perf_counter()
                mN.predict(feedsN)
                tb.append(time.perf_counter() - t0)
            tb.sort(); batchN = tb[len(tb)//2]
            # N sequential
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                for f in feeds1_list:
                    m1.predict(f)
                ts.append(time.perf_counter() - t0)
            ts.sort(); seqN = ts[len(ts)//2]
            print(f"  {unit:12s} L={L:4d} B={B}:  batch-N={batchN*1e3:8.1f}ms   "
                  f"{B}x(B=1)={seqN*1e3:8.1f}ms   speedup={seqN/batchN:5.2f}x")
            del mN, m1
            gc.collect()

    # ---- Sanity: distinct rows -> distinct outputs ----
    print(f"\n\n========== SANITY: batching is real (distinct rows) ==========")
    L, B = 128, min(4, maxB)
    m = ct.models.MLModel(pkgs[(L, B)], compute_units=ct.ComputeUnit.CPU_AND_NE)
    feeds = make_inputs(B, L, distinct=True)
    out = m.predict(feeds)["embedding"]
    out = np.asarray(out)
    print(f"  output shape: {out.shape}")
    # pairwise check rows differ
    allsame = True
    for i in range(B):
        for j in range(i+1, B):
            d = float(np.abs(out[i] - out[j]).max())
            if d > 1e-4:
                allsame = False
            print(f"  row {i} vs {j}: max|diff|={d:.4f}")
    print(f"  -> distinct inputs give {'DUPLICATE (BROADCAST BUG!)' if allsame else 'DISTINCT'} outputs")
    # also confirm a single row matches the B=1 model on same input
    m1 = ct.models.MLModel(pkgs[(L, 1)], compute_units=ct.ComputeUnit.CPU_AND_NE)
    o1 = np.asarray(m1.predict({"input_ids": feeds["input_ids"][0:1],
                                "attention_mask": feeds["attention_mask"][0:1]})["embedding"])
    d01 = float(np.abs(out[0] - o1[0]).max())
    print(f"  batch row0 vs B=1 model on same input: max|diff|={d01:.4f} "
          f"({'MATCH' if d01 < 0.05 else 'MISMATCH'})")

    print("\nDONE.")


if __name__ == "__main__":
    main()
