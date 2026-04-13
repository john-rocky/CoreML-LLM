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
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import (
    SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4,
    SWAVerifyChunk1, SWAVerifyChunk2, SWAVerifyChunk3, SWAVerifyChunk4,
)

HF_DIR = os.environ.get("GEMMA4_HF_DIR", f"{ROOT}/../output/gemma4-e2b/hf_model")
fp16 = np.float16


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
        cfg = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                nbits=4, granularity="per_grouped_channel", group_size=32))
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg)
        print(f"    palettized in {time.time()-t:.1f}s")

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
    parser.add_argument("--output", type=str, default="/tmp/gemma4-multi",
                        help="Output directory")
    parser.add_argument("--hf-dir", type=str, default=HF_DIR,
                        help="HuggingFace model directory")
    parser.add_argument("--K", type=int, default=3,
                        help="Number of draft tokens for verification (default: 3)")
    parser.add_argument("--ctx", type=int, default=2048,
                        help="Context length (default: 2048)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip int4 palettization")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tmp = os.path.join(args.output, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    K = args.K
    CTX = args.ctx
    W = 512
    quantize = not args.no_quantize

    print(f"Loading Gemma 4 E2B from {args.hf_dir}...")
    base = Gemma4Model.from_pretrained(args.hf_dir, context_length=CTX)
    base.eval()

    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512

    print(f"\nK={K}, CTX={CTX}, W={W}, hidden={hidden}, pld={pld}, nlayers={nlayers}")
    print(f"Quantize: {'int4' if quantize else 'fp16'}\n")

    # ================================================================
    # Common input specs for chunks 3/4 (shared KV)
    # ================================================================
    def shared_kv_inputs(seq, prefix_name="per_layer_combined"):
        return [
            ct.TensorType(name="hidden_states",       shape=(1, seq, hidden),       dtype=fp16),
            ct.TensorType(name="causal_mask_full",     shape=(1, 1, seq, CTX),       dtype=fp16),
            ct.TensorType(name="causal_mask_sliding",  shape=(1, 1, seq, W),         dtype=fp16),
            ct.TensorType(name=prefix_name,            shape=(1, seq, nlayers * pld), dtype=fp16),
            ct.TensorType(name="cos_s",                shape=(1, 1, seq, 256),       dtype=fp16),
            ct.TensorType(name="sin_s",                shape=(1, 1, seq, 256),       dtype=fp16),
            ct.TensorType(name="cos_f",                shape=(1, 1, seq, 512),       dtype=fp16),
            ct.TensorType(name="sin_f",                shape=(1, 1, seq, 512),       dtype=fp16),
            ct.TensorType(name="kv13_k",               shape=(1, 1, W, 256),         dtype=fp16),
            ct.TensorType(name="kv13_v",               shape=(1, 1, W, 256),         dtype=fp16),
            ct.TensorType(name="kv14_k",               shape=(1, 1, CTX, 512),       dtype=fp16),
            ct.TensorType(name="kv14_v",               shape=(1, 1, CTX, 512),       dtype=fp16),
        ]

    def shared_kv_samples(seq):
        return (
            torch.zeros(1, seq, hidden, dtype=torch.float16),
            torch.zeros(1, 1, seq, CTX, dtype=torch.float16),
            torch.zeros(1, 1, seq, W, dtype=torch.float16),
            torch.zeros(1, seq, nlayers * pld, dtype=torch.float16),
            torch.zeros(1, 1, seq, 256, dtype=torch.float16),
            torch.zeros(1, 1, seq, 256, dtype=torch.float16),
            torch.zeros(1, 1, seq, 512, dtype=torch.float16),
            torch.zeros(1, 1, seq, 512, dtype=torch.float16),
            torch.zeros(1, 1, W, 256, dtype=torch.float16),
            torch.zeros(1, 1, W, 256, dtype=torch.float16),
            torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
            torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
        )

    # ================================================================
    # Chunk 1: L0-7, 7 sliding + 1 full, owns KV cache
    # ================================================================
    print("=" * 60)
    print("CHUNK 1 (L0-7)")
    print("=" * 60)

    # -- Decode Q=1 --
    print("\n  [decode_q1]")
    swa1 = SWAChunk1(base).eval()
    s1 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(1, 1, CTX, max_hd, dtype=torch.float16),
        torch.zeros(1, 1, CTX, max_hd, dtype=torch.float16),
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

    # -- Verify Q=K --
    print(f"\n  [verify_qK] K={K}")
    vc1 = SWAVerifyChunk1(base, seq_len=K).eval()
    vs1 = (
        torch.zeros(1, K, hidden, dtype=torch.float16),
        torch.zeros(1, 1, K, CTX, dtype=torch.float16),
        torch.zeros(1, 1, K, W, dtype=torch.float16),
        torch.zeros(1, K, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, K, 256, dtype=torch.float16),
        torch.zeros(1, 1, K, 256, dtype=torch.float16),
        torch.zeros(1, 1, K, 512, dtype=torch.float16),
        torch.zeros(1, 1, K, 512, dtype=torch.float16),
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(1, 1, CTX, max_hd, dtype=torch.float16),
        torch.zeros(1, 1, CTX, max_hd, dtype=torch.float16),
    )
    vin1 = [
        ct.TensorType(name="hidden_states",       shape=vs1[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=vs1[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=vs1[2].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",        shape=vs1[3].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=vs1[4].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=vs1[5].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                shape=vs1[6].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                shape=vs1[7].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",         shape=vs1[8].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",         shape=vs1[9].shape,  dtype=fp16),
        ct.TensorType(name="K_full_in",            shape=vs1[10].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",            shape=vs1[11].shape, dtype=fp16),
    ]
    vout1 = ["hidden_states_out", "per_layer_combined_out"]
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
    # Chunk 2: L8-14, 5 sliding + 2 full, owns KV cache
    # ================================================================
    print("\n" + "=" * 60)
    print("CHUNK 2 (L8-14)")
    print("=" * 60)

    # -- Decode Q=1 --
    print("\n  [decode_q1]")
    swa2 = SWAChunk2(base).eval()
    s2 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, CTX, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, CTX, max_hd, dtype=torch.float16),
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

    # -- Verify Q=K --
    print(f"\n  [verify_qK] K={K}")
    vc2 = SWAVerifyChunk2(base, seq_len=K).eval()
    vs2 = (
        torch.zeros(1, K, hidden, dtype=torch.float16),
        torch.zeros(1, 1, K, CTX, dtype=torch.float16),
        torch.zeros(1, 1, K, W, dtype=torch.float16),
        torch.zeros(1, K, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, K, 256, dtype=torch.float16),
        torch.zeros(1, 1, K, 256, dtype=torch.float16),
        torch.zeros(1, 1, K, 512, dtype=torch.float16),
        torch.zeros(1, 1, K, 512, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, CTX, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, CTX, max_hd, dtype=torch.float16),
    )
    vin2 = [
        ct.TensorType(name="hidden_states",       shape=vs2[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",     shape=vs2[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding",  shape=vs2[2].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",   shape=vs2[3].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                shape=vs2[4].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                shape=vs2[5].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                shape=vs2[6].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                shape=vs2[7].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",         shape=vs2[8].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",         shape=vs2[9].shape,  dtype=fp16),
        ct.TensorType(name="K_full_in",            shape=vs2[10].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",            shape=vs2[11].shape, dtype=fp16),
    ]
    vout2 = ["hidden_states_out", "kv13_k", "kv13_v", "kv14_k", "kv14_v"]
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
    # Chunk 3: L15-24, all KV-shared
    # ================================================================
    print("\n" + "=" * 60)
    print("CHUNK 3 (L15-24, KV-shared)")
    print("=" * 60)

    # -- Decode Q=1 --
    print("\n  [decode_q1]")
    swa3 = SWAChunk3(base).eval()
    s3_decode = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
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
    vc3 = SWAVerifyChunk3(base, seq_len=K).eval()
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
    # Chunk 4: L25-34 + norm + lm_head, all KV-shared
    # ================================================================
    print("\n" + "=" * 60)
    print("CHUNK 4 (L25-34 + LM head, KV-shared)")
    print("=" * 60)

    # -- Decode Q=1 --
    print("\n  [decode_q1]")
    swa4 = SWAChunk4(base).eval()
    s4_decode = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
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
    vc4 = SWAVerifyChunk4(base, seq_len=K).eval()
    s4_verify = shared_kv_samples(K)
    in4_verify = shared_kv_inputs(K)
    verify4 = trace_and_convert(vc4, s4_verify, in4_verify, ["token_ids"],
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
    # Cleanup temp files
    # ================================================================
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Multi-function chunks saved to {args.output}/")
    print(f"  chunk1.mlpackage — L0-7   (decode_q1 + verify_qK, K={K})")
    print(f"  chunk2.mlpackage — L8-14  (decode_q1 + verify_qK, K={K})")
    print(f"  chunk3.mlpackage — L15-24 (decode_q1 + verify_qK, K={K})")
    print(f"  chunk4.mlpackage — L25-34 (decode_q1 + verify_qK, K={K})")
    print(f"\nEach mlpackage has two functions:")
    print(f"  decode_q1 (default) — standard Q=1 autoregressive decode")
    print(f"  verify_qK           — Q={K} batched speculative verification")
    print(f"Weights are deduplicated across functions (single weight.bin).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
