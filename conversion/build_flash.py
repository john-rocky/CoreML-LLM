#!/usr/bin/env python3
"""Build Flash Decoding CoreML chunks for 8K context.

Full-attention layers use K-dim chunked attention with online softmax.
Mathematically exact — no quality loss. Each attention chunk fits in ANE SRAM.

Usage:
    python build_flash.py --output /tmp/flash-8k --chunk-size 1024
"""
from __future__ import annotations
import argparse, os, sys, shutil, time
import torch
import numpy as np
import coremltools as ct

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_swa_flash import FlashChunk1, FlashChunk2, FlashChunk3, FlashChunk4, ATTN_CHUNK_SIZE
from ane_ops import MODEL_DTYPE

HF_DIR = f"{ROOT}/output/gemma4-e2b-final/hf_model"
CTX = 8192
W = 512
fp16 = ct.converters.mil.mil.types.fp16


def do_convert(model, sample_inputs, input_specs, output_names, save_path, quantize=True, nbits=4):
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_inputs, check_trace=False)
    print(f"    traced in {time.time()-t:.1f}s")
    t = time.time()
    mlmodel = ct.convert(
        traced, inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        minimum_deployment_target=ct.target.iOS26,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"    converted in {time.time()-t:.1f}s")
    if quantize:
        t = time.time()
        cfg = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                nbits=nbits, granularity="per_grouped_channel", group_size=32))
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg)
        print(f"    palettized W{nbits} in {time.time()-t:.1f}s")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    mlmodel.save(save_path)
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(save_path) for f in fns)
    print(f"    saved ({sz/1e6:.0f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/tmp/flash-8k")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--nbits", type=int, default=4, choices=[2, 3, 4, 6, 8],
                        help="palettization bit width (2=W2A16, 4=INT4 default)")
    parser.add_argument("--chunk-only", type=int, default=None, choices=[1, 2, 3, 4],
                        help="build only the specified chunk (for quick experiments)")
    args = parser.parse_args()

    import models.gemma4_swa_flash as flash_mod
    flash_mod.ATTN_CHUNK_SIZE = args.chunk_size
    flash_mod._FULL_ATTN_CHUNKS = CTX // args.chunk_size
    cs = args.chunk_size
    print(f"Attention chunk size: {cs} (8K = {CTX//cs} chunks)")

    os.makedirs(args.output, exist_ok=True)
    print("Loading Gemma 4 E2B...")
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=CTX)
    base.eval()

    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512

    # Same input shapes as standard 8K model (drop-in replacement)
    def chunk1_io():
        s = (torch.zeros(1,1,hidden,dtype=torch.float16),
             torch.zeros(1,1,1,CTX,dtype=torch.float16),
             torch.zeros(1,1,1,W,dtype=torch.float16),
             torch.zeros(1,1,CTX,1,dtype=torch.float16),
             torch.zeros(1,1,nlayers*pld,dtype=torch.float16),
             torch.zeros(1,1,1,256,dtype=torch.float16),
             torch.zeros(1,1,1,256,dtype=torch.float16),
             torch.zeros(1,1,1,512,dtype=torch.float16),
             torch.zeros(1,1,1,512,dtype=torch.float16),
             torch.zeros(7,1,W,max_hd,dtype=torch.float16),
             torch.zeros(7,1,W,max_hd,dtype=torch.float16),
             torch.zeros(1,1,CTX,max_hd,dtype=torch.float16),
             torch.zeros(1,1,CTX,max_hd,dtype=torch.float16))
        inp = [ct.TensorType(name=n,shape=s[i].shape,dtype=fp16) for i,n in enumerate([
            "hidden_states","causal_mask_full","causal_mask_sliding","update_mask",
            "per_layer_raw","cos_s","sin_s","cos_f","sin_f",
            "K_sliding_in","V_sliding_in","K_full_in","V_full_in"])]
        out = ["hidden_states_out","K_sliding_out","V_sliding_out","K_full_out","V_full_out","per_layer_combined_out"]
        return s, inp, out

    def chunk2_io():
        s = (torch.zeros(1,1,hidden,dtype=torch.float16),
             torch.zeros(1,1,1,CTX,dtype=torch.float16),
             torch.zeros(1,1,1,W,dtype=torch.float16),
             torch.zeros(1,1,CTX,1,dtype=torch.float16),
             torch.zeros(1,1,nlayers*pld,dtype=torch.float16),
             torch.zeros(1,1,1,256,dtype=torch.float16),
             torch.zeros(1,1,1,256,dtype=torch.float16),
             torch.zeros(1,1,1,512,dtype=torch.float16),
             torch.zeros(1,1,1,512,dtype=torch.float16),
             torch.zeros(5,1,W,max_hd,dtype=torch.float16),
             torch.zeros(5,1,W,max_hd,dtype=torch.float16),
             torch.zeros(2,1,CTX,max_hd,dtype=torch.float16),
             torch.zeros(2,1,CTX,max_hd,dtype=torch.float16))
        inp = [ct.TensorType(name=n,shape=s[i].shape,dtype=fp16) for i,n in enumerate([
            "hidden_states","causal_mask_full","causal_mask_sliding","update_mask",
            "per_layer_combined","cos_s","sin_s","cos_f","sin_f",
            "K_sliding_in","V_sliding_in","K_full_in","V_full_in"])]
        out = ["hidden_states_out","K_sliding_out","V_sliding_out","K_full_out","V_full_out",
               "kv13_k","kv13_v","kv14_k","kv14_v"]
        return s, inp, out

    def chunk34_io():
        s = (torch.zeros(1,1,hidden,dtype=torch.float16),
             torch.zeros(1,1,1,CTX,dtype=torch.float16),
             torch.zeros(1,1,1,W,dtype=torch.float16),
             torch.zeros(1,1,CTX,1,dtype=torch.float16),
             torch.zeros(1,1,nlayers*pld,dtype=torch.float16),
             torch.zeros(1,1,1,256,dtype=torch.float16),
             torch.zeros(1,1,1,256,dtype=torch.float16),
             torch.zeros(1,1,1,512,dtype=torch.float16),
             torch.zeros(1,1,1,512,dtype=torch.float16),
             torch.zeros(1,1,W,256,dtype=torch.float16),
             torch.zeros(1,1,W,256,dtype=torch.float16),
             torch.zeros(1,1,CTX,512,dtype=torch.float16),
             torch.zeros(1,1,CTX,512,dtype=torch.float16))
        inp = [ct.TensorType(name=n,shape=s[i].shape,dtype=fp16) for i,n in enumerate([
            "hidden_states","causal_mask_full","causal_mask_sliding","update_mask",
            "per_layer_combined","cos_s","sin_s","cos_f","sin_f",
            "kv13_k","kv13_v","kv14_k","kv14_v"])]
        return s, inp

    # Build chunks
    only = args.chunk_only
    nb = args.nbits

    if only is None or only == 1:
        print(f"\n=== FlashChunk1 (L0-7) W{nb} ===")
        c1 = FlashChunk1(base).eval()
        s1, in1, out1 = chunk1_io()
        do_convert(c1, s1, in1, out1, f"{args.output}/chunk1.mlpackage", nbits=nb)

    if only is None or only == 2:
        print(f"\n=== FlashChunk2 (L8-14) W{nb} ===")
        c2 = FlashChunk2(base).eval()
        s2, in2, out2 = chunk2_io()
        do_convert(c2, s2, in2, out2, f"{args.output}/chunk2.mlpackage", nbits=nb)

    if only is None or only == 3:
        print(f"\n=== FlashChunk3 (L15-24 shared) W{nb} ===")
        c3 = FlashChunk3(base).eval()
        s3, in3 = chunk34_io()
        do_convert(c3, s3, in3, ["hidden_states_out"], f"{args.output}/chunk3.mlpackage", nbits=nb)

    if only is None or only == 4:
        print(f"\n=== FlashChunk4 (L25-34 + LM) W{nb} ===")
        c4 = FlashChunk4(base).eval()
        s4, in4 = chunk34_io()
        do_convert(c4, s4, in4, ["token_id","token_logit","hidden_states_out"], f"{args.output}/chunk4.mlpackage", nbits=nb)

    print(f"\n{'='*60}")
    print(f"Flash Decoding models saved to {args.output}/")
    print(f"Attention chunk size: {cs} ({CTX//cs} chunks per full-attn layer)")
    print(f"Drop-in compatible with existing 8K Swift engine")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
