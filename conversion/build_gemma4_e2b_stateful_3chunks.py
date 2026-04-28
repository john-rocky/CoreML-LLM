#!/usr/bin/env python3
"""Build Gemma 4 stateful 3-chunk variant (merged middle).

Same as `build_gemma4_e2b_stateful_chunks.py` but emits 3 mlpackages
instead of 4 — the middle chunk merges the 4-chunk's chunk_2 (own KV)
and chunk_3 (KV-shared), keeping kv13/kv14 producer aliases internal.
Final chunk_3 = old chunk_4 (KV-shared tail + lm_head + argmax).

Layout (E2B / E4B, derived from `compute_chunk_boundaries(config)`):
  E2B (35 layers):
    chunk_1  L0-7   own KV, computes PLE        — same as 4-chunk
    chunk_2  L8-24  merged: own L8-14 + shared L15-24
    chunk_3  L25-34 + lm_head + argmax          — = old chunk_4
  E4B (42 layers):
    chunk_1  L0-11  own KV, computes PLE
    chunk_2  L12-32 merged: own L12-23 + shared L24-32
    chunk_3  L33-41 + lm_head + argmax

Multifunction `--prefill-batches "8"` adds a `prefill_b8` function to
each chunk (sharing weights via coremltools save_multifunction).

Usage:
    python conversion/build_gemma4_e2b_stateful_3chunks.py \
        --model gemma4-e2b \
        --output /tmp/g4_3chunk/multi \
        --hf-dir /path/to/gemma4-e2b/hf_model \
        --ctx 2048 --linear-projections --prefill-batches "8"

    python conversion/build_gemma4_e2b_stateful_3chunks.py \
        --model gemma4-e4b \
        --output /tmp/g4_3chunk_e4b \
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
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)

from ane_ops import MODEL_DTYPE
from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model
from models.gemma4_swa_stateful_chunks import (
    SWAStatefulChunk1, SWAStatefulChunk1Prefill,
    SWAStatefulChunk4, SWAStatefulChunk4Prefill,
    SWAStatefulMergedChunk23, SWAStatefulMergedChunk23Prefill,
    SWAStatefulChunk1Single, SWAStatefulChunk1PrefillSingle,
    SWAStatefulMergedChunk23Single, SWAStatefulMergedChunk23PrefillSingle,
)
from build_gemma4_e2b_stateful_chunks import (
    _resolve_hf_dir, _audit_ane, _trace_and_convert_stateful,
    convert_chunk1, convert_chunk1_prefill,
    convert_chunk_shared, convert_chunk_shared_prefill,
    merge_multifunction,
)
from models.gemma4_swa_chunks import compute_chunk_boundaries

fp16 = np.float16


def convert_chunk2_merged(base, ctx, out_path, nbits, *, use_linear=False,
                          own_range=None, shared_range=None):
    own = own_range or (8, 15)
    shared = shared_range or (15, 25)
    print("\n" + "=" * 60)
    print(f"CHUNK 2 MERGED (L{own[0]}-{shared[1]-1}) — "
          f"own KV L{own[0]}-{own[1]-1} + KV-shared L{shared[0]}-{shared[1]-1}")
    print("=" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulMergedChunk23(base, ctx,
                                       use_linear=use_linear,
                                       own_range=own_range,
                                       shared_range=shared_range
                                       ).eval().to(MODEL_DTYPE)
    ns, nf = max(chunk.num_sliding, 1), max(chunk.num_full, 1)

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
        ct.TensorType(name="per_layer_combined", shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="current_pos",        shape=(1,),            dtype=np.int32),
        ct.TensorType(name="ring_pos",           shape=(1,),            dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="hidden_states_out", dtype=fp16),
        ct.TensorType(name="kv13_k",            dtype=fp16),
        ct.TensorType(name="kv13_v",            dtype=fp16),
        ct.TensorType(name="kv14_k",            dtype=fp16),
        ct.TensorType(name="kv14_v",            dtype=fp16),
    ]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * ns, HKV, W, max_hd), dtype=fp16),
            name="kv_cache_sliding",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * nf, HKV, ctx, max_hd), dtype=fp16),
            name="kv_cache_full",
        ),
    ]
    _trace_and_convert_stateful(
        chunk, sample, inputs, outputs, states, out_path, nbits)


def convert_chunk2_merged_prefill(base, ctx, T, out_path, nbits, *,
                                    use_linear=False,
                                    own_range=None, shared_range=None):
    own = own_range or (8, 15)
    shared = shared_range or (15, 25)
    print("\n" + "-" * 60)
    print(f"CHUNK 2 MERGED PREFILL T={T} (L{own[0]}-{shared[1]-1})")
    print("-" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulMergedChunk23Prefill(
        base, ctx, use_linear=use_linear, T=T,
        own_range=own_range, shared_range=shared_range).eval().to(MODEL_DTYPE)
    ns, nf = max(chunk.num_sliding, 1), max(chunk.num_full, 1)

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
        ct.TensorType(name="per_layer_combined", shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="current_pos",        shape=(1,),            dtype=np.int32),
        ct.TensorType(name="ring_pos",           shape=(1,),            dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="hidden_states_out", dtype=fp16),
        ct.TensorType(name="kv13_k",            dtype=fp16),
        ct.TensorType(name="kv13_v",            dtype=fp16),
        ct.TensorType(name="kv14_k",            dtype=fp16),
        ct.TensorType(name="kv14_v",            dtype=fp16),
    ]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * ns, HKV, W, max_hd), dtype=fp16),
            name="kv_cache_sliding",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * nf, HKV, ctx, max_hd), dtype=fp16),
            name="kv_cache_full",
        ),
    ]
    _trace_and_convert_stateful(
        chunk, sample, inputs, outputs, states, out_path, nbits)


def convert_chunk1_single(base, c_start, c_end, ctx, out_path, nbits, *,
                            use_linear=False):
    print("\n" + "=" * 60)
    print(f"CHUNK 1 SINGLE-BUFFER (L{c_start}-{c_end-1})")
    print("=" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulChunk1Single(base, c_start, c_end, ctx,
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
        ct.TensorType(name="hidden_states_out",       dtype=fp16),
        ct.TensorType(name="per_layer_combined_out",  dtype=fp16),
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


def convert_chunk1_prefill_single(base, c_start, c_end, ctx, T, out_path, nbits, *,
                                    use_linear=False):
    print("\n" + "-" * 60)
    print(f"CHUNK 1 SINGLE-BUFFER PREFILL T={T} (L{c_start}-{c_end-1})")
    print("-" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulChunk1PrefillSingle(base, c_start, c_end, ctx,
                                             use_linear=use_linear, T=T
                                             ).eval().to(MODEL_DTYPE)
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
        ct.TensorType(name="hidden_states_out",       dtype=fp16),
        ct.TensorType(name="per_layer_combined_out",  dtype=fp16),
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


def convert_chunk2_merged_single(base, ctx, out_path, nbits, *, use_linear=False,
                                 own_range=None, shared_range=None):
    print("\n" + "=" * 60)
    print(f"CHUNK 2 MERGED SINGLE-BUFFER (L8-24)")
    print("=" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulMergedChunk23Single(base, ctx,
                                             use_linear=use_linear,
                                             own_range=own_range,
                                             shared_range=shared_range
                                             ).eval().to(MODEL_DTYPE)
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
        ct.TensorType(name="per_layer_combined", shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="current_pos",        shape=(1,),            dtype=np.int32),
        ct.TensorType(name="ring_pos",           shape=(1,),            dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="hidden_states_out", dtype=fp16),
        ct.TensorType(name="kv13_k",            dtype=fp16),
        ct.TensorType(name="kv13_v",            dtype=fp16),
        ct.TensorType(name="kv14_k",            dtype=fp16),
        ct.TensorType(name="kv14_v",            dtype=fp16),
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


def convert_chunk2_merged_prefill_single(base, ctx, T, out_path, nbits, *,
                                            use_linear=False,
                                            own_range=None, shared_range=None):
    print("\n" + "-" * 60)
    print(f"CHUNK 2 MERGED SINGLE-BUFFER PREFILL T={T} (L8-24)")
    print("-" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulMergedChunk23PrefillSingle(
        base, ctx, use_linear=use_linear, T=T,
        own_range=own_range, shared_range=shared_range).eval().to(MODEL_DTYPE)
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
        ct.TensorType(name="per_layer_combined", shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="current_pos",        shape=(1,),            dtype=np.int32),
        ct.TensorType(name="ring_pos",           shape=(1,),            dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="hidden_states_out", dtype=fp16),
        ct.TensorType(name="kv13_k",            dtype=fp16),
        ct.TensorType(name="kv13_v",            dtype=fp16),
        ct.TensorType(name="kv14_k",            dtype=fp16),
        ct.TensorType(name="kv14_v",            dtype=fp16),
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
    ap.add_argument("--model", default="gemma4-e2b",
                    help="Model name (gemma4-e2b or gemma4-e4b). Chunk "
                         "boundaries are derived from the HF config via "
                         "compute_chunk_boundaries(config); kv13/kv14 names "
                         "are legacy aliases (sliding/full producer slots) "
                         "shared across both models.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--nbits", type=int, default=4, choices=[0, 4, 8])
    ap.add_argument("--only-chunk", type=int, default=None, choices=[1, 2, 3])
    ap.add_argument("--prefill-batches", default="",
                    help="Comma-separated batch sizes (e.g. '8' or '8,16').")
    ap.add_argument("--linear-projections", action="store_true")
    ap.add_argument("--prune-n-m", default=None,
                    help="N:M structured sparsity, e.g. '2:4'. Applied to "
                         "weight tensors before palettize. Joint sparse + "
                         "palettized produces "
                         "constexpr_lut_to_sparse + constexpr_sparse_to_dense.")
    ap.add_argument("--single-buffer", action="store_true",
                    help="Probe variant: collapse kv_cache_sliding + "
                         "kv_cache_full into a single MLState (one buffer "
                         "per chunk, layer-index in axis 0). Tests whether "
                         "iPhone ANE multifunction T>1 accepts a unified "
                         "state where it rejected the dual-state form.")
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

    cfg = base.config
    boundaries = compute_chunk_boundaries(cfg)
    # 3-chunk: re-use boundaries[0] (chunk_1) and boundaries[3] (= old
    # chunk_4, which becomes new chunk_3). The merged middle spans
    # boundaries[1] (own KV) → boundaries[2] (KV-shared), passed into
    # SWAStatefulMergedChunk23 via own_range/shared_range so the same
    # builder works for E2B (own=L8-14, shared=L15-24) and E4B
    # (own=L12-23, shared=L24-32).
    chunk1_range = boundaries[0]
    own_range = boundaries[1]
    shared_range = boundaries[2]
    chunk3_range = boundaries[3]   # final chunk = old chunk_4
    print(f"\nctx={args.ctx}  W={cfg.sliding_window}  hidden={cfg.hidden_size}")
    print(f"3-chunk layout: c1=L{chunk1_range[0]}-{chunk1_range[1]-1}, "
          f"c2_merged=L{own_range[0]}-{shared_range[1]-1} "
          f"(own L{own_range[0]}-{own_range[1]-1} + "
          f"shared L{shared_range[0]}-{shared_range[1]-1}), "
          f"c3=L{chunk3_range[0]}-{chunk3_range[1]-1}")
    print(f"Quantize: int{args.nbits}" if args.nbits else "Quantize: fp16")
    if args.linear_projections:
        print(f"Projections: nn.Linear")

    do = (lambda n: args.only_chunk is None or args.only_chunk == n)
    use_linear = args.linear_projections

    prune_n_m = None
    if args.prune_n_m:
        try:
            n_str, m_str = args.prune_n_m.split(":")
            prune_n_m = (int(n_str), int(m_str))
            print(f"Prune: N:M sparsity = {prune_n_m[0]}:{prune_n_m[1]}")
        except ValueError:
            raise SystemExit(
                f"--prune-n-m must be 'N:M' (got '{args.prune_n_m}')")

    prefill_Ts = [int(x) for x in args.prefill_batches.split(",") if x.strip()]
    if prefill_Ts:
        print(f"Multifunction prefill batches: {prefill_Ts}")
        intermediate = out / "_mf_intermediate"
        intermediate.mkdir(parents=True, exist_ok=True)
    else:
        intermediate = None

    def _build_one(decode_fn, prefill_fn, final_name):
        final_pkg = out / f"{final_name}.mlpackage"
        if not prefill_Ts:
            decode_fn(str(final_pkg))
            return
        decode_pkg = intermediate / f"{final_name}_infer.mlpackage"
        decode_fn(str(decode_pkg))
        prefill_pkgs = []
        for T in prefill_Ts:
            ppkg = intermediate / f"{final_name}_prefill_b{T}.mlpackage"
            prefill_fn(T, str(ppkg))
            prefill_pkgs.append((T, ppkg))
        merge_multifunction(decode_pkg, prefill_pkgs, str(final_pkg))

    use_single = args.single_buffer
    if use_single:
        print("Single-buffer probe: kv_cache_unified (no sliding/full split)")

    if do(1):
        if use_single:
            _build_one(
                lambda p: convert_chunk1_single(base, *chunk1_range, args.ctx, p,
                                                  args.nbits, use_linear=use_linear),
                lambda T, p: convert_chunk1_prefill_single(
                    base, *chunk1_range, args.ctx, T, p, args.nbits,
                    use_linear=use_linear),
                "chunk_1",
            )
        else:
            _build_one(
                lambda p: convert_chunk1(base, *chunk1_range, args.ctx, p,
                                           args.nbits, use_linear=use_linear,
                                           prune_n_m=prune_n_m),
                lambda T, p: convert_chunk1_prefill(
                    base, *chunk1_range, args.ctx, T, p, args.nbits,
                    use_linear=use_linear),
                "chunk_1",
            )
    if do(2):
        if use_single:
            _build_one(
                lambda p: convert_chunk2_merged_single(base, args.ctx, p,
                                                         args.nbits,
                                                         use_linear=use_linear,
                                                         own_range=own_range,
                                                         shared_range=shared_range),
                lambda T, p: convert_chunk2_merged_prefill_single(
                    base, args.ctx, T, p, args.nbits,
                    use_linear=use_linear,
                    own_range=own_range, shared_range=shared_range),
                "chunk_2",
            )
        else:
            _build_one(
                lambda p: convert_chunk2_merged(base, args.ctx, p,
                                                  args.nbits,
                                                  use_linear=use_linear,
                                                  own_range=own_range,
                                                  shared_range=shared_range),
                lambda T, p: convert_chunk2_merged_prefill(
                    base, args.ctx, T, p, args.nbits,
                    use_linear=use_linear,
                    own_range=own_range, shared_range=shared_range),
                "chunk_2",
            )
    if do(3):
        # Final chunk = old chunk_4 (KV-shared L25-34 + lm_head + argmax)
        _build_one(
            lambda p: convert_chunk_shared(
                SWAStatefulChunk4, base, *chunk3_range, args.ctx, p,
                args.nbits, name="CHUNK 3 (final)", with_lm_head=True,
                use_linear=use_linear),
            lambda T, p: convert_chunk_shared_prefill(
                SWAStatefulChunk4Prefill, base, *chunk3_range, args.ctx, T,
                p, args.nbits, name="CHUNK 3 (final)", with_lm_head=True,
                use_linear=use_linear),
            "chunk_3",
        )

    print(f"\nartifacts under {out}")
    for p in sorted(out.iterdir()):
        size = (sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                if p.is_dir() else p.stat().st_size) / 1e6
        print(f"  {p.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
