#!/usr/bin/env python3
"""Rebuild ONLY chunk 4 (decode_q1 + verify_qK) of the multi-function pack.

This is a trimmed variant of `build_verify_chunks.py` that skips chunks
1/2/3 so we can iterate quickly on chunk 4 when only its verify output
spec changes. The resulting `chunk4.mlpackage` can be dropped into an
existing staging directory alongside the untouched chunks 1/2/3.

Produced specifically for Track A / Phase C tolerance-aware acceptance:
chunk 4 verify gains a `logits_fp16` output of shape (1, K, vocab) so
the Swift bench can score drafter proposals against the target's top-N
logits without rerunning the argmax.

Usage:
    python build_verify_chunk4_only.py --output /tmp/gemma4-chunk4-tol
    python build_verify_chunk4_only.py --output /tmp/gemma4-chunk4-tol --K 3 --ctx 2048
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
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import SWAChunk4, SWAVerifyChunk4

HF_DIR = os.environ.get("GEMMA4_HF_DIR", f"{ROOT}/../output/gemma4-e2b/hf_model")
fp16 = np.float16


def trace_and_convert(model, sample_inputs, input_specs, output_names, quantize=True):
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
    if os.path.exists(path):
        shutil.rmtree(path)
    mlmodel.save(path)


def build_multifunction(decode_path, verify_path, output_path):
    desc = MultiFunctionDescriptor()
    desc.add_function(decode_path, "main", "decode_q1")
    desc.add_function(verify_path, "main", "verify_qK")
    desc.default_function_name = "decode_q1"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    save_multifunction(desc, output_path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(output_path)
        for f in fns
    ) / 1024 / 1024
    print(f"    saved {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Rebuild chunk 4 only (tolerance variant)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for chunk4.mlpackage")
    parser.add_argument("--hf-dir", type=str, default=HF_DIR)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--no-quantize", action="store_true")
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

    print(f"\nK={K}, CTX={CTX}, W={W}, hidden={hidden}, pld={pld}, nlayers={nlayers}")
    print(f"Quantize: {'int4' if quantize else 'fp16'}\n")

    def shared_kv_inputs(seq):
        return [
            ct.TensorType(name="hidden_states",       shape=(1, seq, hidden),       dtype=fp16),
            ct.TensorType(name="causal_mask_full",     shape=(1, 1, seq, CTX),       dtype=fp16),
            ct.TensorType(name="causal_mask_sliding",  shape=(1, 1, seq, W),         dtype=fp16),
            ct.TensorType(name="per_layer_combined",   shape=(1, seq, nlayers * pld), dtype=fp16),
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

    print("=" * 60)
    print("CHUNK 4 (L25-34 + LM head, KV-shared) — rebuild only")
    print("=" * 60)

    # -- Decode Q=1 (unchanged — shapes must match the main builder exactly) --
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

    # -- Verify Q=K — now exposes logits_fp16 --
    print(f"\n  [verify_qK] K={K} (with logits_fp16)")
    vc4 = SWAVerifyChunk4(base, seq_len=K).eval()
    s4_verify = shared_kv_samples(K)
    in4_verify = shared_kv_inputs(K)
    verify4 = trace_and_convert(
        vc4, s4_verify, in4_verify,
        ["token_ids", "hidden_states_out", "logits_fp16"],
        quantize=quantize)
    save_temp(verify4, f"{tmp}/chunk4_verify.mlpackage")
    del verify4

    # -- Bundle --
    print("\n  [multifunction] bundling...")
    build_multifunction(
        f"{tmp}/chunk4_decode.mlpackage",
        f"{tmp}/chunk4_verify.mlpackage",
        f"{args.output}/chunk4.mlpackage")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nDone. Wrote {args.output}/chunk4.mlpackage")


if __name__ == "__main__":
    main()
