#!/usr/bin/env python3
"""Probe: does the 17-layer MergedChunk23 (L8-24) compile to CoreML for iOS 18 ANE?

Gate for the 3-chunk consolidation plan (chunk1 / merged-17 / chunk4).
Runs trace + ct.convert + optional compile. On success reports mlpackage size
and (if --compile) the .mlmodelc size; on failure prints the converter error.

Deliberately skips int4 palettization and the verify_qK path so the compile
signal is isolated from quantization and multi-function packaging.

Usage:
    python conversion/probe_3way_compile.py \
        --model gemma4-e2b --ctx 2048 \
        --output /tmp/gemma4_3way_probe --compile
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct

from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model
from models.gemma4_swa_merged import MergedChunk23
from models.gemma4_swa_chunks import SWAChunk1, SWAChunk4, compute_chunk_boundaries


fp16 = np.float16


def _resolve_hf_dir(model_name: str, override: str | None) -> str:
    if override:
        return override
    if model_name in MODEL_REGISTRY:
        from huggingface_hub import snapshot_download
        repo = MODEL_REGISTRY[model_name].hf_repo
        local = os.path.join(ROOT, "..", "output", model_name, "hf_model")
        if not os.path.isdir(local) or not any(
            fn.endswith(".safetensors") for fn in os.listdir(local)
        ):
            print(f"Downloading {repo} to {local}...")
            snapshot_download(
                repo, local_dir=local,
                allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model"],
            )
        return local
    raise SystemExit(f"unknown model {model_name}")


def _du_mb(path: str) -> float:
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024 / 1024
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
    return total / 1024 / 1024


def _convert(model, sample_inputs, input_specs, output_names, *, label: str):
    print(f"\n[{label}] tracing...")
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_inputs, check_trace=False)
    print(f"  traced in {time.time()-t:.1f}s")

    print(f"[{label}] ct.convert iOS18 / fp16 / CPU_AND_NE...")
    t = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"  converted in {time.time()-t:.1f}s")
    return mlmodel


def _save(mlmodel, path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    mlmodel.save(path)
    print(f"  saved {path} ({_du_mb(path):.1f} MB)")


def _compile(pkg: str) -> str:
    print(f"\n  compiling {pkg} to .mlmodelc ...")
    t = time.time()
    model = ct.models.MLModel(pkg)
    compiled = model.get_compiled_model_path()
    dst = pkg.replace(".mlpackage", ".mlmodelc")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(compiled, dst)
    del model
    print(f"  compiled {dst} ({_du_mb(dst):.1f} MB, {time.time()-t:.1f}s)")
    return dst


def _probe_merged(base, *, ctx: int, out_dir: str, compile_it: bool) -> None:
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    max_hd = hd_f
    nkv = cfg.num_key_value_heads

    mc = MergedChunk23(base).eval()
    ns, nf = mc.num_sliding, mc.num_full
    print(f"\nMergedChunk23: L{mc.START_C2}-{mc.END_C3 - 1} "
          f"({mc.END_C3 - mc.START_C2} layers, own-KV: {ns} sliding + {nf} full)")

    sample = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(ns, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(nf, nkv, ctx, max_hd, dtype=torch.float16),
        torch.zeros(nf, nkv, ctx, max_hd, dtype=torch.float16),
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=sample[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=sample[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=sample[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=sample[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=sample[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=sample[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=sample[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=sample[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=sample[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=sample[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=sample[12].shape, dtype=fp16),
    ]
    outputs = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
               "K_full_out", "V_full_out",
               "kv13_k", "kv13_v", "kv14_k", "kv14_v"]

    mlmodel = _convert(mc, sample, inputs, outputs, label="merged-17")
    pkg = os.path.join(out_dir, "chunk2_merged17.mlpackage")
    _save(mlmodel, pkg)
    del mlmodel

    if compile_it:
        _compile(pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--compile", action="store_true",
                    help="Also compile .mlpackage to .mlmodelc (macOS only)")
    args = ap.parse_args()

    if args.ctx is None and args.model in MODEL_REGISTRY:
        args.ctx = MODEL_REGISTRY[args.model].default_context_length
    args.output = args.output or os.path.join(ROOT, "..", "output",
                                              f"{args.model}_3way_probe")
    os.makedirs(args.output, exist_ok=True)

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir}...")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()

    print(f"N={base.config.num_hidden_layers}  producers=L{base.config.kv_sliding_producer}/"
          f"L{base.config.kv_full_producer}  W={base.config.sliding_window}  ctx={args.ctx}")
    print(f"4-chunk boundaries: {compute_chunk_boundaries(base.config)}")

    _probe_merged(base, ctx=args.ctx, out_dir=args.output, compile_it=args.compile)

    print(f"\nOK. Probe artifacts in {args.output}")


if __name__ == "__main__":
    main()
