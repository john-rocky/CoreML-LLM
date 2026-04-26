#!/usr/bin/env python3
"""Build Gemma 4 E2B stateful 1-chunk all-in-one mlpackage.

Single mlpackage containing the entire 35-layer model (PLE compute +
own-KV L0-14 + KV-shared L15-34 + final norm + lm_head + argmax).
Mirrors the Qwen3-VL single-state pattern; tests whether iPhone ANE
accepts T=8 multifunction when there is only one chunk and one
unified MLState (no chunk-boundary KV alias plumbing).

Output: <out>/model.mlpackage  (the only file — no chunk_*)

Usage:
    python conversion/build_gemma4_e2b_stateful_1chunk.py \
        --output /tmp/g4_1chunk \
        --hf-dir /path/to/gemma4-e2b/hf_model \
        --ctx 2048 --linear-projections --prefill-batches "8"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

from ane_ops import MODEL_DTYPE
from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model
from models.gemma4_swa_stateful_chunks import (
    SWAStatefulModel1Chunk, SWAStatefulModel1ChunkPrefill,
)
from build_gemma4_e2b_stateful_chunks import (
    _resolve_hf_dir, _audit_ane, _trace_and_convert_stateful,
    merge_multifunction,
)

fp16 = np.float16


def convert_model_1chunk(base, ctx, out_path, nbits, *, use_linear=False):
    print("\n" + "=" * 60)
    print("MODEL 1-CHUNK (L0-34 + lm_head + argmax) — own-KV L0-14")
    print("=" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulModel1Chunk(base, ctx,
                                     use_linear=use_linear).eval().to(MODEL_DTYPE)
    no = max(chunk.num_own, 1)

    sample = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",   shape=sample[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_raw",      shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="current_pos",        shape=(1,),            dtype=np.int32),
        ct.TensorType(name="ring_pos",           shape=(1,),            dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="token_id",     dtype=np.int32),
        ct.TensorType(name="token_logit",  dtype=fp16),
        ct.TensorType(name="hidden_normed", dtype=fp16),
    ]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * no, HKV, ctx, max_hd), dtype=fp16),
            name="kv_cache_unified",
        ),
    ]
    _trace_and_convert_stateful(
        chunk, sample, inputs, outputs, states, out_path, nbits)


def convert_model_1chunk_prefill(base, ctx, T, out_path, nbits, *,
                                   use_linear=False):
    print("\n" + "-" * 60)
    print(f"MODEL 1-CHUNK PREFILL T={T} (L0-34 + lm_head)")
    print("-" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulModel1ChunkPrefill(
        base, ctx, use_linear=use_linear, T=T).eval().to(MODEL_DTYPE)
    no = max(chunk.num_own, 1)

    sample = (
        torch.zeros(1, T, hidden, dtype=torch.float16),
        torch.zeros(1, 1, T, ctx, dtype=torch.float16),
        torch.zeros(1, 1, T, W, dtype=torch.float16),
        torch.zeros(1, T, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, T, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, T, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, T, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, T, hd_f, dtype=torch.float16),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",   shape=sample[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_raw",      shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="current_pos",        shape=(1,),            dtype=np.int32),
        ct.TensorType(name="ring_pos",           shape=(1,),            dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="token_id",     dtype=np.int32),
        ct.TensorType(name="token_logit",  dtype=fp16),
        ct.TensorType(name="hidden_normed", dtype=fp16),
    ]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * no, HKV, ctx, max_hd), dtype=fp16),
            name="kv_cache_unified",
        ),
    ]
    _trace_and_convert_stateful(
        chunk, sample, inputs, outputs, states, out_path, nbits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--output", required=True)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--nbits", type=int, default=4, choices=[0, 4, 8])
    ap.add_argument("--prefill-batches", default="",
                    help="Comma-separated batch sizes (e.g. '8').")
    ap.add_argument("--linear-projections", action="store_true")
    args = ap.parse_args()

    if args.ctx is None:
        if args.model in MODEL_REGISTRY:
            args.ctx = MODEL_REGISTRY[args.model].default_context_length
        else:
            args.ctx = 2048

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir}...")
    t0 = time.time()
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"\nctx={args.ctx}  W={base.config.sliding_window}  "
          f"hidden={base.config.hidden_size}  layers={base.config.num_hidden_layers}")
    print(f"Quantize: int{args.nbits}" if args.nbits else "Quantize: fp16")
    if args.linear_projections:
        print("Projections: nn.Linear")

    prefill_Ts = [int(x) for x in args.prefill_batches.split(",") if x.strip()]
    if prefill_Ts:
        print(f"Multifunction prefill batches: {prefill_Ts}")
        intermediate = out / "_mf_intermediate"
        intermediate.mkdir(parents=True, exist_ok=True)
    else:
        intermediate = None

    final_pkg = out / "model.mlpackage"
    if not prefill_Ts:
        convert_model_1chunk(base, args.ctx, str(final_pkg), args.nbits,
                              use_linear=args.linear_projections)
    else:
        decode_pkg = intermediate / "model_infer.mlpackage"
        convert_model_1chunk(base, args.ctx, str(decode_pkg), args.nbits,
                              use_linear=args.linear_projections)
        prefill_pkgs = []
        for T in prefill_Ts:
            ppkg = intermediate / f"model_prefill_b{T}.mlpackage"
            convert_model_1chunk_prefill(
                base, args.ctx, T, str(ppkg), args.nbits,
                use_linear=args.linear_projections)
            prefill_pkgs.append((T, ppkg))
        merge_multifunction(decode_pkg, prefill_pkgs, str(final_pkg))

    print(f"\nartifact: {final_pkg}")
    size = sum(f.stat().st_size for f in final_pkg.rglob('*')
                if f.is_file()) / 1e6
    print(f"  size: {size:.1f} MB")


if __name__ == "__main__":
    main()
