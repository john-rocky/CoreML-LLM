#!/usr/bin/env python3
"""Build Windowed Full Attention (WFA) CoreML chunks.

Full-attention layers use a sliding window of FW=2048 instead of full ctx=8192.
All KV updates are shift-based. Expected to match 2K model speed (~31 tok/s).

Usage:
    python build_wfa.py --output /tmp/wfa-test --fw 2048
"""
from __future__ import annotations
import argparse, os, sys, shutil, time
import torch
import numpy as np
import coremltools as ct

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_swa_wfa import WFAChunk1, WFAChunk2, WFAChunk3, WFAChunk4, FW as DEFAULT_FW
from ane_ops import MODEL_DTYPE

HF_DIR = f"{ROOT}/output/gemma4-e2b-final/hf_model"
W = 512
fp16 = ct.converters.mil.mil.types.fp16


def do_convert(model, sample_inputs, input_specs, output_names, save_path, quantize=True):
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_inputs, check_trace=False)
    print(f"    traced in {time.time()-t:.1f}s")

    t = time.time()
    mlmodel = ct.convert(
        traced, inputs=input_specs,
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

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    mlmodel.save(save_path)
    sz = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(save_path) for f in fns
    )
    print(f"    saved ({sz/1e6:.0f} MB)")
    return mlmodel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/tmp/wfa-test")
    parser.add_argument("--fw", type=int, default=DEFAULT_FW,
                        help="Full-attention window size (default: 2048)")
    args = parser.parse_args()

    fw = args.fw

    # Patch the FW constant in the module
    import models.gemma4_swa_wfa as wfa_mod
    wfa_mod.FW = fw
    print(f"Full-attention window: FW={fw}")

    os.makedirs(args.output, exist_ok=True)

    print("Loading Gemma 4 E2B...")
    # context_length doesn't matter for WFA since all KV is windowed
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=fw)
    base.eval()

    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512

    # WFA: no update_mask needed (shift-based for all layers)
    # causal_mask_full: (1, 1, 1, FW) — sliding mask for full-attention
    # causal_mask_sliding: (1, 1, 1, W) — sliding mask for SWA

    # === Chunk 1 ===
    print(f"\n=== WFA Chunk1 (L0-7, FW={fw}) ===")
    c1 = WFAChunk1(base).eval()
    s1 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, fw, dtype=torch.float16),       # causal_mask_full (FW!)
        torch.zeros(1, 1, 1, W, dtype=torch.float16),        # causal_mask_sliding
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(1, 1, fw, max_hd, dtype=torch.float16),  # K_full (FW!)
        torch.zeros(1, 1, fw, max_hd, dtype=torch.float16),  # V_full (FW!)
    )
    in1 = [
        ct.TensorType(name="hidden_states",       shape=s1[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s1[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s1[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_raw",       shape=s1[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s1[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s1[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s1[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s1[7].shape, dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=s1[8].shape, dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=s1[9].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=s1[10].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=s1[11].shape, dtype=fp16),
    ]
    out1 = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
            "K_full_out", "V_full_out", "per_layer_combined_out"]
    do_convert(c1, s1, in1, out1, f"{args.output}/chunk1.mlpackage")

    # === Chunk 2 ===
    print(f"\n=== WFA Chunk2 (L8-14, FW={fw}) ===")
    c2 = WFAChunk2(base).eval()
    s2 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, fw, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, fw, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, fw, max_hd, dtype=torch.float16),
    )
    in2 = [
        ct.TensorType(name="hidden_states",       shape=s2[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s2[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s2[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s2[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s2[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s2[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s2[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s2[7].shape, dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=s2[8].shape, dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=s2[9].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=s2[10].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=s2[11].shape, dtype=fp16),
    ]
    out2 = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
            "K_full_out", "V_full_out", "kv13_k", "kv13_v", "kv14_k", "kv14_v"]
    do_convert(c2, s2, in2, out2, f"{args.output}/chunk2.mlpackage")

    # === Chunk 3 ===
    print(f"\n=== WFA Chunk3 (L15-24 shared, FW={fw}) ===")
    c3 = WFAChunk3(base).eval()
    s3 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, fw, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, fw, 512, dtype=torch.float16),   # kv14 is FW-sized!
        torch.zeros(1, 1, fw, 512, dtype=torch.float16),
    )
    in3 = [
        ct.TensorType(name="hidden_states",       shape=s3[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s3[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s3[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s3[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s3[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s3[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s3[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s3[7].shape, dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=s3[8].shape, dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=s3[9].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=s3[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=s3[11].shape, dtype=fp16),
    ]
    out3 = ["hidden_states_out"]
    do_convert(c3, s3, in3, out3, f"{args.output}/chunk3.mlpackage")

    # === Chunk 4 ===
    print(f"\n=== WFA Chunk4 (L25-34 shared + LM, FW={fw}) ===")
    c4 = WFAChunk4(base).eval()
    s4 = s3  # same input shapes
    in4 = in3
    out4 = ["token_id", "token_logit", "hidden_states_out"]
    do_convert(c4, s4, in4, out4, f"{args.output}/chunk4.mlpackage")

    print(f"\n{'='*60}")
    print(f"WFA models saved to {args.output}/ (FW={fw})")
    print(f"Expected performance: ~{31 if fw <= 2048 else '20-25'} tok/s on iPhone ANE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
