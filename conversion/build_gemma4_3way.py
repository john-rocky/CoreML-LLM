#!/usr/bin/env python3
"""Build the decode-only 3-chunk variant of Gemma 4 (chunk1 / merged17 / head).

Mirrors build_verify_chunks.py but emits only three .mlpackage files and only
the decode_q1 entry point (verify_qK is skipped until 11c lands and the
speculative path is unblocked).

Layer split (E2B):
    chunk1_3way:  L0-7    (SWAChunk1, own KV)                     — 8 layers
    chunk2_3way:  L8-24   (MergedChunk23, own KV + KV-shared)      — 17 layers
    chunk3_3way:  L25-34  (SWAChunk4, KV-shared + norm + lm_head)  — 10 layers + head

Output directory (default): output/<model>/chunks_3way/
    chunk1_3way.mlpackage
    chunk2_3way.mlpackage
    chunk3_3way.mlpackage

Swift loader detects 3-chunk mode by the presence of chunk2_3way.mlpackage
(or .mlmodelc after compile). See follow-up PR for ChunkedEngine wiring.
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
from models.gemma4_swa_chunks import (
    SWAChunk1, SWAChunk4, SWAChunk4_LMSplit, compute_chunk_boundaries,
)
from models.gemma4_swa_merged import MergedChunk23

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


def _convert_and_palettize(model, sample, inputs, outputs, *, label: str, quantize: bool):
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, check_trace=False)
    print(f"    traced in {time.time()-t:.1f}s")

    t = time.time()
    mlmodel = ct.convert(
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
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg)
        print(f"    palettized INT4/g32 in {time.time()-t:.1f}s")

    return mlmodel


def _save(mlmodel, path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    mlmodel.save(path)
    print(f"    saved {path}  ({_du_mb(path):.1f} MB)")


# ----------------------------------------------------------------------
# Chunk 1: L0-7, own KV. Identical to build_verify_chunks.py's decode_q1.
# ----------------------------------------------------------------------

def build_chunk1(base, ctx: int, out_pkg: str, *, quantize: bool) -> None:
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    max_hd = hd_f
    nkv = cfg.num_key_value_heads

    boundaries = compute_chunk_boundaries(cfg)
    c1_start, c1_end = boundaries[0]
    print(f"\n=== chunk1_3way (L{c1_start}-{c1_end-1}) ===")

    swa1 = SWAChunk1(base, c1_start, c1_end).eval()
    ns1, nf1 = swa1.num_sliding, swa1.num_full
    s1 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(max(nf1, 1), nkv, ctx, max_hd, dtype=torch.float16),
        torch.zeros(max(nf1, 1), nkv, ctx, max_hd, dtype=torch.float16),
    )
    in1 = [
        ct.TensorType(name="hidden_states",      shape=s1[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s1[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s1[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s1[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",       shape=s1[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s1[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s1[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s1[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s1[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=s1[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=s1[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=s1[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=s1[12].shape, dtype=fp16),
    ]
    out1 = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
            "K_full_out", "V_full_out", "per_layer_combined_out"]
    m = _convert_and_palettize(swa1, s1, in1, out1,
                               label="chunk1_3way", quantize=quantize)
    _save(m, out_pkg)


# ----------------------------------------------------------------------
# Chunk 2 (merged): L8-24, own-KV + KV-shared. MergedChunk23 keeps
# kv13/kv14 internal but also re-outputs them (chunk3 still needs them).
# ----------------------------------------------------------------------

def build_chunk2_merged(base, ctx: int, out_pkg: str, *, quantize: bool) -> None:
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    max_hd = hd_f
    nkv = cfg.num_key_value_heads

    boundaries = compute_chunk_boundaries(cfg)
    own_range = boundaries[1]
    shared_range = boundaries[2]
    mc = MergedChunk23(base, own_range=own_range, shared_range=shared_range).eval()
    ns, nf = mc.num_sliding, mc.num_full
    n_layers = (mc.END_C2 - mc.START_C2) + (mc.END_C3 - mc.START_C3)
    print(f"\n=== chunk2_3way (L{mc.START_C2}-{mc.END_C3-1}, {n_layers} layers) ===")
    print(f"    own-KV: {ns} sliding + {nf} full")

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
    m = _convert_and_palettize(mc, sample, inputs, outputs,
                               label="chunk2_3way", quantize=quantize)
    _save(m, out_pkg)


# ----------------------------------------------------------------------
# Chunk 3 (head): L25-34 + norm + lm_head + argmax.  Mirrors the current
# 4-chunk chunk4 byte-for-byte; renamed so the bundle layout stays
# self-consistent (chunk1/chunk2/chunk3).
# ----------------------------------------------------------------------

def build_chunk3_head(base, ctx: int, out_pkg: str, *, quantize: bool,
                      lm_splits: int = 1) -> None:
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    nkv = cfg.num_key_value_heads

    boundaries = compute_chunk_boundaries(cfg)
    _, c4_end = boundaries[3]
    c4_start = boundaries[3][0]
    suffix = "" if lm_splits <= 1 else f" (lm_split={lm_splits})"
    print(f"\n=== chunk3_3way (L{c4_start}-{c4_end-1} + LM head){suffix} ===")

    if lm_splits > 1:
        swa = SWAChunk4_LMSplit(base, c4_start, c4_end, n_splits=lm_splits).eval()
    else:
        swa = SWAChunk4(base, c4_start, c4_end).eval()
    s = (
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
    inputs = [
        ct.TensorType(name="hidden_states",      shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=s[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=s[12].shape, dtype=fp16),
    ]
    outputs = ["token_id", "token_logit", "hidden_states_out"]
    m = _convert_and_palettize(swa, s, inputs, outputs,
                               label="chunk3_3way", quantize=quantize)
    _save(m, out_pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--output", default=None,
                    help="Output directory (default: output/<model>/chunks_3way)")
    ap.add_argument("--no-quantize", action="store_true",
                    help="Skip INT4 palettization (fp16 chunks — ~4× larger)")
    ap.add_argument("--only", choices=("chunk1", "chunk2", "chunk3"), default=None,
                    help="Build only one chunk (debug)")
    ap.add_argument("--lm-splits", type=int, default=1, choices=(1, 2, 4, 8, 16),
                    help="Split chunk3 lm_head into N parallel Conv2d heads. "
                         "1=baseline (current production); 16=anemll-style ANE-SRAM-friendly.")
    args = ap.parse_args()

    if args.ctx is None and args.model in MODEL_REGISTRY:
        args.ctx = MODEL_REGISTRY[args.model].default_context_length
    elif args.ctx is None:
        args.ctx = 2048
    default_subdir = "chunks_3way" if args.lm_splits == 1 else f"chunks_3way_lmsplit{args.lm_splits}"
    args.output = args.output or os.path.join(ROOT, "..", "output", args.model, default_subdir)
    os.makedirs(args.output, exist_ok=True)

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir}  ctx={args.ctx}")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()
    print(f"N={base.config.num_hidden_layers}  "
          f"producers=L{base.config.kv_sliding_producer}/L{base.config.kv_full_producer}  "
          f"W={base.config.sliding_window}")
    print(f"Quantize: {'int4 per_grouped_channel g=32' if not args.no_quantize else 'fp16'}")

    quantize = not args.no_quantize
    c1 = os.path.join(args.output, "chunk1_3way.mlpackage")
    c2 = os.path.join(args.output, "chunk2_3way.mlpackage")
    c3 = os.path.join(args.output, "chunk3_3way.mlpackage")

    if args.only in (None, "chunk1"):
        build_chunk1(base, args.ctx, c1, quantize=quantize)
    if args.only in (None, "chunk2"):
        build_chunk2_merged(base, args.ctx, c2, quantize=quantize)
    if args.only in (None, "chunk3"):
        build_chunk3_head(base, args.ctx, c3, quantize=quantize,
                          lm_splits=args.lm_splits)

    print("\n" + "=" * 60)
    print(f"3-chunk bundle written to {args.output}/")
    for p in (c1, c2, c3):
        if os.path.exists(p):
            print(f"  {os.path.basename(p):<32s} {_du_mb(p):7.1f} MB")
    print("=" * 60)
    print("Swift-side wiring of ChunkedEngine to this layout is the next PR.")


if __name__ == "__main__":
    main()
