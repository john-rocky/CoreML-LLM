#!/usr/bin/env python3
"""Build EAGLE-3 verify chunks for Gemma 4 E2B (T=3 batch-verify).

Read-only KV cache. Produces per-position argmax (token_ids, token_logits)
from chunk4. Intended callsite: SpeculativeTarget.verifyCandidates in Swift.

Outputs:
  verify_chunk1.mlpackage  — L0-7
  verify_chunk2.mlpackage  — L8-14, + extended kv13 (W+T,256), kv14 (ctx+T,512)
  verify_chunk3.mlpackage  — L15-24
  verify_chunk4.mlpackage  — L25-34 + norm + lm_head, emits (token_ids, token_logits)

Usage:
    python conversion/build_eagle3_verify.py --output ./output/eagle3-chunks
    python conversion/build_eagle3_verify.py --only chunk1 -T 3
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import time

import torch
import coremltools as ct

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_verify_chunks import (
    VerifyChunk1, VerifyChunk2, VerifyChunk3, VerifyChunk4,
)
from ane_ops import MODEL_DTYPE

HF_DIR = os.environ.get(
    "GEMMA4_HF_DIR",
    f"{ROOT}/../output/gemma4-e2b/hf_model",
)
CTX = 2048
W = 512
fp16 = ct.converters.mil.mil.types.fp16
int32 = ct.converters.mil.mil.types.int32


def do_convert(model, sample_inputs, input_specs, output_specs, save_path, quantize=True):
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_inputs, check_trace=False)
    print(f"    traced in {time.time()-t:.1f}s")

    t = time.time()
    mlmodel = ct.convert(
        traced, inputs=input_specs,
        outputs=output_specs,
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

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    mlmodel.save(save_path)
    return mlmodel


def build_chunk1(base, out_dir, T):
    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512
    print(f"\n=== verify_chunk1 (L0-7, T={T}) ===")
    c1 = VerifyChunk1(base, T=T).eval()
    s = (
        torch.zeros(1, T, hidden, dtype=torch.float16),                     # hidden_states
        torch.zeros(1, 1, T, CTX + T, dtype=torch.float16),                 # causal_mask_full
        torch.zeros(1, 1, T, W + T, dtype=torch.float16),                   # causal_mask_sliding
        torch.zeros(1, T, nlayers * pld, dtype=torch.float16),              # per_layer_raw
        torch.zeros(1, 1, T, 256, dtype=torch.float16),                     # cos_s
        torch.zeros(1, 1, T, 256, dtype=torch.float16),                     # sin_s
        torch.zeros(1, 1, T, 512, dtype=torch.float16),                     # cos_f
        torch.zeros(1, 1, T, 512, dtype=torch.float16),                     # sin_f
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),                  # K_sliding_in
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),                  # V_sliding_in
        torch.zeros(1, 1, CTX, max_hd, dtype=torch.float16),                # K_full_in
        torch.zeros(1, 1, CTX, max_hd, dtype=torch.float16),                # V_full_in
    )
    ins = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",       shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=s[11].shape, dtype=fp16),
    ]
    outs = [
        ct.TensorType(name="hidden_states_out"),
        ct.TensorType(name="per_layer_combined_out"),
        ct.TensorType(name="K_sliding_new"),
        ct.TensorType(name="V_sliding_new"),
        ct.TensorType(name="K_full_new"),
        ct.TensorType(name="V_full_new"),
    ]
    do_convert(c1, s, ins, outs, f"{out_dir}/verify_chunk1.mlpackage")


def build_chunk2(base, out_dir, T):
    hidden = base.config.hidden_size
    max_hd = 512
    print(f"\n=== verify_chunk2 (L8-14, + extended kv13/kv14, T={T}) ===")
    c2 = VerifyChunk2(base, T=T).eval()
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    s = (
        torch.zeros(1, T, hidden, dtype=torch.float16),
        torch.zeros(1, 1, T, CTX + T, dtype=torch.float16),
        torch.zeros(1, 1, T, W + T, dtype=torch.float16),
        torch.zeros(1, T, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, T, 256, dtype=torch.float16),
        torch.zeros(1, 1, T, 256, dtype=torch.float16),
        torch.zeros(1, 1, T, 512, dtype=torch.float16),
        torch.zeros(1, 1, T, 512, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),       # 5 sliding layers in L8-14
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, CTX, max_hd, dtype=torch.float16),     # 2 full layers
        torch.zeros(2, 1, CTX, max_hd, dtype=torch.float16),
    )
    ins = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=s[11].shape, dtype=fp16),
    ]
    outs = [
        ct.TensorType(name="hidden_states_out"),
        ct.TensorType(name="kv13_k_out"),
        ct.TensorType(name="kv13_v_out"),
        ct.TensorType(name="kv14_k_out"),
        ct.TensorType(name="kv14_v_out"),
        ct.TensorType(name="K_sliding_new"),
        ct.TensorType(name="V_sliding_new"),
        ct.TensorType(name="K_full_new"),
        ct.TensorType(name="V_full_new"),
        ct.TensorType(name="hidden_at_L8"),
    ]
    do_convert(c2, s, ins, outs, f"{out_dir}/verify_chunk2.mlpackage")


def build_chunk3(base, out_dir, T):
    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    print(f"\n=== verify_chunk3 (L15-24, T={T}) ===")
    c3 = VerifyChunk3(base, T=T).eval()
    s = (
        torch.zeros(1, T, hidden, dtype=torch.float16),
        torch.zeros(1, 1, T, CTX + T, dtype=torch.float16),
        torch.zeros(1, 1, T, W + T, dtype=torch.float16),
        torch.zeros(1, T, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, T, 256, dtype=torch.float16),
        torch.zeros(1, 1, T, 256, dtype=torch.float16),
        torch.zeros(1, 1, T, 512, dtype=torch.float16),
        torch.zeros(1, 1, T, 512, dtype=torch.float16),
        torch.zeros(1, 1, W + T, 256, dtype=torch.float16),     # kv13_k
        torch.zeros(1, 1, W + T, 256, dtype=torch.float16),
        torch.zeros(1, 1, CTX + T, 512, dtype=torch.float16),   # kv14_k
        torch.zeros(1, 1, CTX + T, 512, dtype=torch.float16),
    )
    ins = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=s[11].shape, dtype=fp16),
    ]
    outs = [
        ct.TensorType(name="hidden_states_out"),
        ct.TensorType(name="hidden_at_L17"),
    ]
    do_convert(c3, s, ins, outs, f"{out_dir}/verify_chunk3.mlpackage")


def build_chunk4(base, out_dir, T):
    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    print(f"\n=== verify_chunk4 (L25-34 + LM, T={T}) ===")
    c4 = VerifyChunk4(base, T=T).eval()
    s = (
        torch.zeros(1, T, hidden, dtype=torch.float16),
        torch.zeros(1, 1, T, CTX + T, dtype=torch.float16),
        torch.zeros(1, 1, T, W + T, dtype=torch.float16),
        torch.zeros(1, T, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, T, 256, dtype=torch.float16),
        torch.zeros(1, 1, T, 256, dtype=torch.float16),
        torch.zeros(1, 1, T, 512, dtype=torch.float16),
        torch.zeros(1, 1, T, 512, dtype=torch.float16),
        torch.zeros(1, 1, W + T, 256, dtype=torch.float16),
        torch.zeros(1, 1, W + T, 256, dtype=torch.float16),
        torch.zeros(1, 1, CTX + T, 512, dtype=torch.float16),
        torch.zeros(1, 1, CTX + T, 512, dtype=torch.float16),
    )
    ins = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=s[11].shape, dtype=fp16),
    ]
    outs = [
        ct.TensorType(name="token_ids",    dtype=int32),
        ct.TensorType(name="token_logits"),
        ct.TensorType(name="hidden_at_L34"),
    ]
    do_convert(c4, s, ins, outs, f"{out_dir}/verify_chunk4.mlpackage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="./output/eagle3-chunks")
    ap.add_argument("-T", "--T", type=int, default=3,
                    help="Number of candidate positions per verify call (K for speculative).")
    ap.add_argument("--only", type=str, default=None,
                    choices=[None, "chunk1", "chunk2", "chunk3", "chunk4"])
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading Gemma 4 E2B from {HF_DIR}...")
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=CTX)
    base.eval()

    targets = [args.only] if args.only else ["chunk1", "chunk2", "chunk3", "chunk4"]
    dispatch = {
        "chunk1": build_chunk1,
        "chunk2": build_chunk2,
        "chunk3": build_chunk3,
        "chunk4": build_chunk4,
    }
    for name in targets:
        dispatch[name](base, args.output, args.T)

    print(f"\nDone. verify_chunk*.mlpackage files in {args.output}/")


if __name__ == "__main__":
    main()
