#!/usr/bin/env python3
"""Compile probe: does the 15-layer BigChunk1 (L0-14) pass ct.convert and
coremltools' Mac compile step at iOS 18 target, fp16, CPU_AND_NE?

Smaller than the shipped 17-layer merged chunk2, so if 17 compiled, 15
should.  The probe exists so we fail fast if the PLE-inside-chunk1 path
confuses the compiler at 15 layers (which doesn't exist in the shipped
tree — chunk1 is 8 layers, merged chunk2 is 17 but has NO PLE).
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

from models.gemma4 import Gemma4Model
from models.gemma4_swa_3chunk_search import BigChunk1

fp16 = np.float16


def _du_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024 / 1024
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
    return total / 1024 / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--output", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    args.output = args.output or os.path.join(
        ROOT, "..", "output", f"{args.model}_topology_I_probe")
    os.makedirs(args.output, exist_ok=True)

    hf_dir = os.path.join(ROOT, "..", "output", args.model, "hf_model")
    print(f"Loading {args.model} from {hf_dir}  ctx={args.ctx}")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()
    cfg = base.config
    own_end = cfg.kv_full_producer + 1  # 15 for E2B

    big = BigChunk1(base, 0, own_end).eval()
    ns, nf = big.num_sliding, big.num_full
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    max_hd = hd_f
    nkv = cfg.num_key_value_heads

    print(f"\nBigChunk1 L0-{own_end-1}  {own_end} layers  own-KV: {ns} sliding + {nf} full")

    sample = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, args.ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, args.ctx, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(ns, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(nf, nkv, args.ctx, max_hd, dtype=torch.float16),
        torch.zeros(nf, nkv, args.ctx, max_hd, dtype=torch.float16),
    )
    input_specs = [
        ct.TensorType(name="hidden_states",       shape=sample[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=sample[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=sample[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",          shape=sample[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",        shape=sample[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=sample[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=sample[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                shape=sample[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                shape=sample[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",         shape=sample[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",         shape=sample[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",            shape=sample[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",            shape=sample[12].shape, dtype=fp16),
    ]
    out_names = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
                 "K_full_out", "V_full_out", "per_layer_combined_out",
                 "kv13_k", "kv13_v", "kv14_k", "kv14_v"]

    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(big, sample, check_trace=False)
    print(f"traced in {time.time()-t:.1f}s")

    t = time.time()
    mlm = ct.convert(
        traced,
        inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in out_names],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"converted in {time.time()-t:.1f}s")

    pkg = os.path.join(args.output, "big_chunk1.mlpackage")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    mlm.save(pkg)
    print(f"saved  {pkg}  ({_du_mb(pkg):.1f} MB)")

    if args.compile:
        t = time.time()
        m = ct.models.MLModel(pkg)
        compiled = m.get_compiled_model_path()
        dst = pkg.replace(".mlpackage", ".mlmodelc")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(compiled, dst)
        del m
        print(f"compiled {dst}  ({_du_mb(dst):.1f} MB, {time.time()-t:.1f}s)")


if __name__ == "__main__":
    main()
