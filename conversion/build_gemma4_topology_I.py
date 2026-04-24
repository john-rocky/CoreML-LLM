#!/usr/bin/env python3
"""Topology-I 3-chunk decode bundle for Gemma 4 E2B.

Produces:
    chunk1_topoI.mlpackage   (BigChunk1 — L0-14, 15 layers own-KV + PLE + emits kv13/kv14)
    chunk2_topoI.mlpackage   (SWAChunk3 — L15-24, 10 layers KV-shared)
    chunk3_topoI.mlpackage   (SWAChunk4 — L25-34, 10 layers KV-shared + norm + lm_head + argmax)

chunk3_topoI is architecturally identical to chunk3_3way (Topology II's
LM-head chunk) and to chunk4 (4-chunk bundle).  It is rebuilt here so
the Topology I bundle is self-contained, but on the device the three
files can be dropped alongside the 4-chunk + Topology II files for
side-by-side A/B.

Compared to Topology II (shipped):
  - chunk1 absorbs chunk2 (8 → 15 own-KV layers)
  - chunk2 covers only the first half of shared-KV layers (17 → 10)
  - chunk3 unchanged (10 layers + head)
Dispatch count unchanged (3), but the per-chunk ms distribution shifts.
ANE per-layer efficiency may differ between 15- and 17-layer chunks;
iPhone A/B is the only way to find out.
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
from models.gemma4_swa_chunks import SWAChunk3, SWAChunk4
from models.gemma4_swa_3chunk_search import BigChunk1

fp16 = np.float16


def _resolve_hf_dir(model_name: str) -> str:
    return os.path.join(ROOT, "..", "output", model_name, "hf_model")


def _du_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024 / 1024
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
    return total / 1024 / 1024


def _convert_and_palettize(model, sample, inputs, outputs, *, label, quantize):
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, check_trace=False)
    print(f"    traced in {time.time()-t:.1f}s")

    t = time.time()
    mlm = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=n) for n in outputs],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"    converted in {time.time()-t:.1f}s")

    if quantize:
        t = time.time()
        cfg = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                nbits=4, granularity="per_grouped_channel", group_size=32))
        mlm = ct.optimize.coreml.palettize_weights(mlm, cfg)
        print(f"    palettized INT4/g32 in {time.time()-t:.1f}s")
    return mlm


def _save(mlm, path):
    if os.path.exists(path):
        shutil.rmtree(path)
    mlm.save(path)
    print(f"    saved {path}  ({_du_mb(path):.1f} MB)")


def build_chunk1(base, ctx, out_pkg, *, quantize):
    cfg = base.config
    own_end = cfg.kv_full_producer + 1
    big = BigChunk1(base, 0, own_end).eval()
    ns, nf = big.num_sliding, big.num_full
    print(f"\n=== chunk1_topoI (L0-{own_end-1}, {own_end} layers, PLE+kv13/14 emit) ===")
    print(f"    own-KV: {ns} sliding + {nf} full")

    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    max_hd = hd_f
    nkv = cfg.num_key_value_heads

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
    in_specs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=sample[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=sample[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",       shape=sample[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=sample[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=sample[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=sample[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=sample[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=sample[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=sample[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=sample[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=sample[12].shape, dtype=fp16),
    ]
    out_names = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
                 "K_full_out", "V_full_out", "per_layer_combined_out",
                 "kv13_k", "kv13_v", "kv14_k", "kv14_v"]
    m = _convert_and_palettize(big, sample, in_specs, out_names,
                               label="chunk1_topoI", quantize=quantize)
    _save(m, out_pkg)


def build_chunk2(base, ctx, out_pkg, *, quantize):
    cfg = base.config
    c3_start = cfg.kv_full_producer + 1
    # Mid-point split of the shared region: L15-24 for E2B.
    n = cfg.num_hidden_layers
    c3_end = c3_start + (n - c3_start) // 2
    swa3 = SWAChunk3(base, c3_start, c3_end).eval()
    print(f"\n=== chunk2_topoI (L{c3_start}-{c3_end-1}, {c3_end - c3_start} layers, KV-shared) ===")

    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    nkv = cfg.num_key_value_heads

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
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, ctx, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, ctx, hd_f, dtype=torch.float16),
    )
    in_specs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=sample[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=sample[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=sample[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=sample[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=sample[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=sample[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=sample[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=sample[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=sample[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=sample[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=sample[12].shape, dtype=fp16),
    ]
    m = _convert_and_palettize(swa3, sample, in_specs, ["hidden_states_out"],
                               label="chunk2_topoI", quantize=quantize)
    _save(m, out_pkg)


def build_chunk3(base, ctx, out_pkg, *, quantize):
    cfg = base.config
    n = cfg.num_hidden_layers
    c3_start = cfg.kv_full_producer + 1
    c3_end = c3_start + (n - c3_start) // 2
    c4_start = c3_end
    c4_end = n
    swa4 = SWAChunk4(base, c4_start, c4_end).eval()
    print(f"\n=== chunk3_topoI (L{c4_start}-{c4_end-1} + LM head, {c4_end - c4_start} layers) ===")

    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    nkv = cfg.num_key_value_heads

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
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, ctx, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, ctx, hd_f, dtype=torch.float16),
    )
    in_specs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=sample[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=sample[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=sample[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=sample[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=sample[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=sample[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=sample[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=sample[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=sample[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=sample[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=sample[12].shape, dtype=fp16),
    ]
    m = _convert_and_palettize(swa4, sample, in_specs,
                               ["token_id", "token_logit", "hidden_states_out"],
                               label="chunk3_topoI", quantize=quantize)
    _save(m, out_pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--only", choices=("chunk1", "chunk2", "chunk3"), default=None)
    args = ap.parse_args()

    if args.ctx is None:
        args.ctx = (MODEL_REGISTRY[args.model].default_context_length
                    if args.model in MODEL_REGISTRY else 2048)
    args.output = args.output or os.path.join(ROOT, "..", "output",
                                              args.model, "chunks_topoI")
    os.makedirs(args.output, exist_ok=True)

    hf_dir = _resolve_hf_dir(args.model)
    print(f"Loading {args.model} from {hf_dir}  ctx={args.ctx}")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()

    quantize = not args.no_quantize
    c1 = os.path.join(args.output, "chunk1_topoI.mlpackage")
    c2 = os.path.join(args.output, "chunk2_topoI.mlpackage")
    c3 = os.path.join(args.output, "chunk3_topoI.mlpackage")

    if args.only in (None, "chunk1"):
        build_chunk1(base, args.ctx, c1, quantize=quantize)
    if args.only in (None, "chunk2"):
        build_chunk2(base, args.ctx, c2, quantize=quantize)
    if args.only in (None, "chunk3"):
        build_chunk3(base, args.ctx, c3, quantize=quantize)

    print("\n" + "=" * 60)
    print(f"Topology-I bundle in {args.output}/")
    for p in (c1, c2, c3):
        if os.path.exists(p):
            print(f"  {os.path.basename(p):<32s} {_du_mb(p):7.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
