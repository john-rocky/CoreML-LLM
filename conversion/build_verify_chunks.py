#!/usr/bin/env python3
"""Build multi-function CoreML chunks with decode_q1 + verify_qK entry points.

Each chunk mlpackage contains two functions sharing deduplicated weights:
  - decode_q1: standard Q=1 autoregressive decode (identical to existing chunks)
  - verify_qK: Q=K batched speculative verification (read-only KV cache)

The verify function runs K draft tokens through the target model in one ANE
dispatch, returning per-position argmax for acceptance/rejection.

Usage:
    python build_verify_chunks.py --output /tmp/gemma4-multi
    python build_verify_chunks.py --output /tmp/gemma4-multi --K 5
    python build_verify_chunks.py --output /tmp/gemma4-multi --ctx 8192

Requires GEMMA4_HF_DIR env var or default path to the HF model.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

from ane_ops import MODEL_DTYPE
from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import (
    SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4,
    SWAVerifyChunk1, SWAVerifyChunk2, SWAVerifyChunk3, SWAVerifyChunk4,
    compute_chunk_boundaries,
)

DEFAULT_HF_DIR = os.environ.get("GEMMA4_HF_DIR", f"{ROOT}/../output/gemma4-e2b/hf_model")
fp16 = np.float16


def _resolve_hf_dir(model_name: str, override: str | None) -> str:
    """Return a local HF dir: --hf-dir override wins, else auto-download by registry entry."""
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
    return DEFAULT_HF_DIR


def trace_and_convert(model, sample_inputs, input_specs, output_names, quantize=True):
    """Trace a PyTorch model and convert to CoreML."""
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_inputs, check_trace=False)
    print(f"    traced in {time.time()-t:.1f}s")

    t = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"    converted in {time.time()-t:.1f}s")

    if quantize:
        t = time.time()
        nbits = int(os.environ.get("PALETTIZE_NBITS", "4"))
        granularity = os.environ.get("PALETTIZE_GRANULARITY", "per_grouped_channel")
        group_size = int(os.environ.get("PALETTIZE_GROUP_SIZE", "32"))
        per_channel_scale = os.environ.get("PALETTIZE_PCS", "0") == "1"
        # PALETTIZE_KEEP_FP_KV=1 → keep `*_self_attn_(k|v)_proj_weight` as fp16
        # (skip palettize for those ops). The K/V cache is computed from these
        # weights, and INT4 noise on the K cache (cosine 0.978 vs fp16 0.993)
        # cuts MTP per-slot accept on chat code from ~0.55 (vendor) to ~0.20
        # (ours). Keeping ~5 % of weights fp16 raises chunk size ~3 %; the
        # rest of the model stays INT4 for ANE bandwidth.
        keep_fp_kv = os.environ.get("PALETTIZE_KEEP_FP_KV", "0") == "1"
        # Optionally also keep q_proj + o_proj fp16. Higher size hit but
        # cleaner attention output → cleaner K writes for next layer.
        keep_fp_attn = os.environ.get("PALETTIZE_KEEP_FP_ATTN", "0") == "1"
        cfg_kw = dict(nbits=nbits, granularity=granularity,
                      enable_per_channel_scale=per_channel_scale)
        if granularity == "per_grouped_channel":
            cfg_kw["group_size"] = group_size
        global_cfg = ct.optimize.coreml.OpPalettizerConfig(**cfg_kw)
        op_name_configs = {}
        patterns = []
        if keep_fp_kv:
            patterns += ["self_attn_k_proj", "self_attn_v_proj"]
        if keep_fp_attn:
            patterns += ["self_attn_q_proj", "self_attn_o_proj"]
        if patterns:
            from coremltools.optimize.coreml import get_weights_metadata
            md = get_weights_metadata(mlmodel, weight_threshold=2048)
            for name in md.keys():
                if any(p in name for p in patterns):
                    op_name_configs[name] = None
            print(f"    keep_fp patterns={patterns} — skipping "
                  f"{len(op_name_configs)} weights (kept fp16); "
                  f"sample: {list(op_name_configs)[:2]}")
        cfg = ct.optimize.coreml.OptimizationConfig(
            global_config=global_cfg,
            op_name_configs=op_name_configs if op_name_configs else None)
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg)
        print(f"    palettized nbits={nbits} {granularity}"
              f"{' g='+str(group_size) if granularity=='per_grouped_channel' else ''} "
              f"pcs={per_channel_scale} keep_fp_kv={keep_fp_kv} in {time.time()-t:.1f}s")

    return mlmodel


def save_temp(mlmodel, path):
    """Save mlpackage to a temporary path."""
    if os.path.exists(path):
        shutil.rmtree(path)
    mlmodel.save(path)


def build_multifunction(decode_path, verify_path, output_path):
    """Bundle decode and verify mlpackages into a single multi-function mlpackage."""
    desc = MultiFunctionDescriptor()
    desc.add_function(decode_path, "main", "decode_q1")
    desc.add_function(verify_path, "main", "verify_qK")
    desc.default_function_name = "decode_q1"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    save_multifunction(desc, output_path)

    # Report size
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(output_path)
        for f in fns
    ) / 1024 / 1024
    print(f"    saved {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build multi-function verify chunks")
    parser.add_argument("--model", type=str, default="gemma4-e2b",
                        help="Model name in MODEL_REGISTRY (gemma4-e2b | gemma4-e4b)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: ../output/<model>/chunks)")
    parser.add_argument("--hf-dir", type=str, default=None,
                        help="Override HF model directory (skip auto-download)")
    parser.add_argument("--K", type=int, default=3,
                        help="Number of draft tokens for verification (default: 3)")
    parser.add_argument("--ctx", type=int, default=None,
                        help="Context length (default: registry entry's default)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip int4 palettization")
    parser.add_argument("--awq-state", type=str, default=None,
                        help="Optional AWQ-smoothed state_dict (.pt) to load "
                             "after Gemma4Model.from_pretrained. Use "
                             "conversion/awq_smooth_gemma4.py to generate.")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep the _tmp/chunkN_{decode,verify}.mlpackage "
                             "intermediates so standalone verify-only chunks "
                             "can be pushed to iPhone without the multi-function "
                             "mlpackage (some iOS CoreML builds reject multi-function).")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(ROOT, "..", "output", args.model, "chunks")
    if args.ctx is None and args.model in MODEL_REGISTRY:
        args.ctx = MODEL_REGISTRY[args.model].default_context_length
    elif args.ctx is None:
        args.ctx = 2048

    os.makedirs(args.output, exist_ok=True)
    tmp = os.path.join(args.output, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    K = args.K
    CTX = args.ctx
    quantize = not args.no_quantize

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir}...")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=CTX)
    base.eval()

    if args.awq_state:
        print(f"Loading AWQ-smoothed state from {args.awq_state}")
        sd = torch.load(args.awq_state, map_location="cpu", weights_only=True)
        missing, unexpected = base.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys (ignored)")
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (kept original)")
        print(f"  AWQ state loaded ({len(sd)} tensors)")

    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    W = base.config.sliding_window
    hd_s = base.config.head_dim
    hd_f = base.config.global_head_dim
    max_hd = hd_f
    boundaries = compute_chunk_boundaries(base.config)

    print(f"\nK={K}, CTX={CTX}, W={W}, hidden={hidden}, pld={pld}, nlayers={nlayers}")
    print(f"head_dim={hd_s}, global_head_dim={hd_f}, num_kv_heads={base.config.num_key_value_heads}")
    print(f"KV producers: sliding=L{base.config.kv_sliding_producer}, full=L{base.config.kv_full_producer}")
    print(f"Chunk boundaries: {boundaries}")
    print(f"Quantize: {'int4' if quantize else 'fp16'}\n")

    # ================================================================
    # Common input specs for chunks 3/4 (shared KV)
    # ================================================================
    nkv = base.config.num_key_value_heads

    def shared_kv_inputs(seq, prefix_name="per_layer_combined"):
        return [
            ct.TensorType(name="hidden_states",       shape=(1, seq, hidden),       dtype=fp16),
            ct.TensorType(name="causal_mask_full",     shape=(1, 1, seq, CTX),       dtype=fp16),
            ct.TensorType(name="causal_mask_sliding",  shape=(1, 1, seq, W),         dtype=fp16),
            ct.TensorType(name=prefix_name,            shape=(1, seq, nlayers * pld), dtype=fp16),
            ct.TensorType(name="cos_s",                shape=(1, 1, seq, hd_s),      dtype=fp16),
            ct.TensorType(name="sin_s",                shape=(1, 1, seq, hd_s),      dtype=fp16),
            ct.TensorType(name="cos_f",                shape=(1, 1, seq, hd_f),      dtype=fp16),
            ct.TensorType(name="sin_f",                shape=(1, 1, seq, hd_f),      dtype=fp16),
            ct.TensorType(name="kv13_k",               shape=(1, nkv, W, hd_s),      dtype=fp16),
            ct.TensorType(name="kv13_v",               shape=(1, nkv, W, hd_s),      dtype=fp16),
            ct.TensorType(name="kv14_k",               shape=(1, nkv, CTX, hd_f),    dtype=fp16),
            ct.TensorType(name="kv14_v",               shape=(1, nkv, CTX, hd_f),    dtype=fp16),
        ]

    def shared_kv_samples(seq):
        return (
            torch.zeros(1, seq, hidden, dtype=torch.float16),
            torch.zeros(1, 1, seq, CTX, dtype=torch.float16),
            torch.zeros(1, 1, seq, W, dtype=torch.float16),
            torch.zeros(1, seq, nlayers * pld, dtype=torch.float16),
            torch.zeros(1, 1, seq, hd_s, dtype=torch.float16),
            torch.zeros(1, 1, seq, hd_s, dtype=torch.float16),
            torch.zeros(1, 1, seq, hd_f, dtype=torch.float16),
            torch.zeros(1, 1, seq, hd_f, dtype=torch.float16),
            torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
            torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
            torch.zeros(1, nkv, CTX, hd_f, dtype=torch.float16),
            torch.zeros(1, nkv, CTX, hd_f, dtype=torch.float16),
        )

    # ================================================================
    # Chunk 1: own KV cache
    # ================================================================
    c1_start, c1_end = boundaries[0]
    print("=" * 60)
    print(f"CHUNK 1 (L{c1_start}-{c1_end-1})")
    print("=" * 60)

    # -- Decode Q=1 --
    print("\n  [decode_q1]")
    swa1 = SWAChunk1(base, c1_start, c1_end).eval()
    ns1, nf1 = swa1.num_sliding, swa1.num_full
    s1 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(max(nf1, 1), nkv, CTX, max_hd, dtype=torch.float16),
        torch.zeros(max(nf1, 1), nkv, CTX, max_hd, dtype=torch.float16),
    )
    in1 = [
        ct.TensorType(name="hidden_states",       shape=s1[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=s1[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=s1[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",          shape=s1[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",        shape=s1[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=s1[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=s1[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                shape=s1[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                shape=s1[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",         shape=s1[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",         shape=s1[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",            shape=s1[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",            shape=s1[12].shape, dtype=fp16),
    ]
    out1 = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
            "K_full_out", "V_full_out", "per_layer_combined_out"]
    decode1 = trace_and_convert(swa1, s1, in1, out1, quantize=quantize)
    save_temp(decode1, f"{tmp}/chunk1_decode.mlpackage")
    del decode1

    # -- Verify Q=K (write-through) --
    print(f"\n  [verify_qK] K={K} (write-through)")
    vc1 = SWAVerifyChunk1(base, seq_len=K, start=c1_start, end=c1_end).eval()
    vs1 = (
        torch.zeros(1, K, hidden, dtype=torch.float16),
        torch.zeros(1, 1, K, CTX, dtype=torch.float16),
        torch.zeros(1, 1, K, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, K, dtype=torch.float16),  # update_indicator
        torch.zeros(1, K, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_f, dtype=torch.float16),
        torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(max(nf1, 1), nkv, CTX, max_hd, dtype=torch.float16),
        torch.zeros(max(nf1, 1), nkv, CTX, max_hd, dtype=torch.float16),
    )
    vin1 = [
        ct.TensorType(name="hidden_states",       shape=vs1[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=vs1[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=vs1[2].shape,  dtype=fp16),
        ct.TensorType(name="update_indicator",     shape=vs1[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",        shape=vs1[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=vs1[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=vs1[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                shape=vs1[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                shape=vs1[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",         shape=vs1[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",         shape=vs1[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",            shape=vs1[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",            shape=vs1[12].shape, dtype=fp16),
    ]
    # 11c: per-T K/V slices replace full-cache outputs.
    # Sliding hd=256, full hd=512. Swift zero-pads sliding writes to max_hd=512.
    vout1 = ["hidden_states_out", "per_layer_combined_out",
             "new_K_sliding", "new_V_sliding", "new_K_full", "new_V_full"]
    verify1 = trace_and_convert(vc1, vs1, vin1, vout1, quantize=quantize)
    save_temp(verify1, f"{tmp}/chunk1_verify.mlpackage")
    del verify1

    # -- Bundle --
    print("\n  [multifunction] bundling...")
    build_multifunction(
        f"{tmp}/chunk1_decode.mlpackage",
        f"{tmp}/chunk1_verify.mlpackage",
        f"{args.output}/chunk1.mlpackage")

    # ================================================================
    # Chunk 2: own KV cache; emits producer KV (kv13_*/kv14_* aliases)
    # ================================================================
    c2_start, c2_end = boundaries[1]
    print("\n" + "=" * 60)
    print(f"CHUNK 2 (L{c2_start}-{c2_end-1})")
    print("=" * 60)

    # -- Decode Q=1 --
    print("\n  [decode_q1]")
    swa2 = SWAChunk2(base, c2_start, c2_end).eval()
    ns2, nf2 = swa2.num_sliding, swa2.num_full
    s2 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(ns2, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns2, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(nf2, nkv, CTX, max_hd, dtype=torch.float16),
        torch.zeros(nf2, nkv, CTX, max_hd, dtype=torch.float16),
    )
    in2 = [
        ct.TensorType(name="hidden_states",       shape=s2[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=s2[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=s2[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",          shape=s2[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",   shape=s2[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=s2[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=s2[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                shape=s2[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                shape=s2[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",         shape=s2[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",         shape=s2[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",            shape=s2[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",            shape=s2[12].shape, dtype=fp16),
    ]
    out2 = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
            "K_full_out", "V_full_out", "kv13_k", "kv13_v", "kv14_k", "kv14_v"]
    decode2 = trace_and_convert(swa2, s2, in2, out2, quantize=quantize)
    save_temp(decode2, f"{tmp}/chunk2_decode.mlpackage")
    del decode2

    # -- Verify Q=K (write-through) --
    print(f"\n  [verify_qK] K={K} (write-through)")
    vc2 = SWAVerifyChunk2(base, seq_len=K, start=c2_start, end=c2_end).eval()
    vs2 = (
        torch.zeros(1, K, hidden, dtype=torch.float16),
        torch.zeros(1, 1, K, CTX, dtype=torch.float16),
        torch.zeros(1, 1, K, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, K, dtype=torch.float16),  # update_indicator
        torch.zeros(1, K, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_f, dtype=torch.float16),
        torch.zeros(ns2, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns2, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(nf2, nkv, CTX, max_hd, dtype=torch.float16),
        torch.zeros(nf2, nkv, CTX, max_hd, dtype=torch.float16),
    )
    vin2 = [
        ct.TensorType(name="hidden_states",       shape=vs2[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=vs2[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=vs2[2].shape,  dtype=fp16),
        ct.TensorType(name="update_indicator",     shape=vs2[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",   shape=vs2[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=vs2[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=vs2[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                shape=vs2[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                shape=vs2[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",         shape=vs2[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",         shape=vs2[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",            shape=vs2[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",            shape=vs2[12].shape, dtype=fp16),
    ]
    # 11c: per-T K/V slices replace full-cache outputs.
    # kv13_k/v + kv14_k/v are the extended within-verify caches feeding chunks 3/4
    # (NOT persisted by Swift; persistent kv13/kv14 are the L13/L14 slots of
    # kSliding2/kFull2, written via new_K_sliding/new_K_full slices).
    vout2 = ["hidden_states_out",
             "new_K_sliding", "new_V_sliding", "new_K_full", "new_V_full",
             "kv13_k", "kv13_v", "kv14_k", "kv14_v"]
    verify2 = trace_and_convert(vc2, vs2, vin2, vout2, quantize=quantize)
    save_temp(verify2, f"{tmp}/chunk2_verify.mlpackage")
    del verify2

    # -- Bundle --
    print("\n  [multifunction] bundling...")
    build_multifunction(
        f"{tmp}/chunk2_decode.mlpackage",
        f"{tmp}/chunk2_verify.mlpackage",
        f"{args.output}/chunk2.mlpackage")

    # ================================================================
    # Chunk 3: all KV-shared
    # ================================================================
    c3_start, c3_end = boundaries[2]
    print("\n" + "=" * 60)
    print(f"CHUNK 3 (L{c3_start}-{c3_end-1}, KV-shared)")
    print("=" * 60)

    # -- Decode Q=1 --
    print("\n  [decode_q1]")
    swa3 = SWAChunk3(base, c3_start, c3_end).eval()
    s3_decode = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, CTX, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, CTX, hd_f, dtype=torch.float16),
    )
    in3_decode = [
        ct.TensorType(name="hidden_states",       shape=s3_decode[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=s3_decode[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=s3_decode[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",          shape=s3_decode[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",   shape=s3_decode[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=s3_decode[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=s3_decode[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                shape=s3_decode[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                shape=s3_decode[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",               shape=s3_decode[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",               shape=s3_decode[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",               shape=s3_decode[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",               shape=s3_decode[12].shape, dtype=fp16),
    ]
    decode3 = trace_and_convert(swa3, s3_decode, in3_decode, ["hidden_states_out"],
                                quantize=quantize)
    save_temp(decode3, f"{tmp}/chunk3_decode.mlpackage")
    del decode3

    # -- Verify Q=K --
    print(f"\n  [verify_qK] K={K}")
    vc3 = SWAVerifyChunk3(base, seq_len=K, start=c3_start, end=c3_end).eval()
    s3_verify = shared_kv_samples(K)
    in3_verify = shared_kv_inputs(K)
    verify3 = trace_and_convert(vc3, s3_verify, in3_verify, ["hidden_states_out"],
                                quantize=quantize)
    save_temp(verify3, f"{tmp}/chunk3_verify.mlpackage")
    del verify3

    # -- Bundle --
    print("\n  [multifunction] bundling...")
    build_multifunction(
        f"{tmp}/chunk3_decode.mlpackage",
        f"{tmp}/chunk3_verify.mlpackage",
        f"{args.output}/chunk3.mlpackage")

    # ================================================================
    # Chunk 4: all KV-shared + norm + lm_head
    # ================================================================
    c4_start, c4_end = boundaries[3]
    print("\n" + "=" * 60)
    print(f"CHUNK 4 (L{c4_start}-{c4_end-1} + LM head, KV-shared)")
    print("=" * 60)

    # -- Decode Q=1 --
    print("\n  [decode_q1]")
    swa4 = SWAChunk4(base, c4_start, c4_end).eval()
    s4_decode = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, CTX, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, CTX, hd_f, dtype=torch.float16),
    )
    in4_decode = [
        ct.TensorType(name="hidden_states",       shape=s4_decode[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=s4_decode[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=s4_decode[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",          shape=s4_decode[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",   shape=s4_decode[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=s4_decode[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=s4_decode[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                shape=s4_decode[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                shape=s4_decode[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",               shape=s4_decode[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",               shape=s4_decode[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",               shape=s4_decode[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",               shape=s4_decode[12].shape, dtype=fp16),
    ]
    out4_decode = ["token_id", "token_logit", "hidden_states_out"]
    decode4 = trace_and_convert(swa4, s4_decode, in4_decode, out4_decode,
                                quantize=quantize)
    save_temp(decode4, f"{tmp}/chunk4_decode.mlpackage")
    del decode4

    # -- Verify Q=K --
    print(f"\n  [verify_qK] K={K}")
    vc4 = SWAVerifyChunk4(base, seq_len=K, start=c4_start, end=c4_end).eval()
    s4_verify = shared_kv_samples(K)
    in4_verify = shared_kv_inputs(K)
    verify4 = trace_and_convert(vc4, s4_verify, in4_verify,
                                ["token_ids", "hidden_states_out"],
                                quantize=quantize)
    save_temp(verify4, f"{tmp}/chunk4_verify.mlpackage")
    del verify4

    # -- Bundle --
    print("\n  [multifunction] bundling...")
    build_multifunction(
        f"{tmp}/chunk4_decode.mlpackage",
        f"{tmp}/chunk4_verify.mlpackage",
        f"{args.output}/chunk4.mlpackage")

    # ================================================================
    # Cleanup temp files (unless --keep-tmp). Retained intermediates are
    # usable as standalone chunk{N}_{decode,verify}.mlpackage which match
    # Swift's "standalone verify chunks" fallback when multi-function
    # loading is unsupported on the target OS.
    # ================================================================
    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"\n  [keep-tmp] intermediates preserved under {tmp}/")

    print(f"\n{'='*60}")
    print(f"Multi-function chunks saved to {args.output}/")
    for i, (s, e) in enumerate(boundaries, start=1):
        print(f"  chunk{i}.mlpackage — L{s}-{e-1} (decode_q1 + verify_qK, K={K})")
    print(f"\nEach mlpackage has two functions:")
    print(f"  decode_q1 (default) — standard Q=1 autoregressive decode")
    print(f"  verify_qK           — Q={K} batched speculative verification")
    print(f"Weights are deduplicated across functions (single weight.bin).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
