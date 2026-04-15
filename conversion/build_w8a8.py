#!/usr/bin/env python3
"""Build W8A8 quantized CoreML chunks for 8K context.

Compares INT4 palettization (current) vs INT8 linear weight quantization
vs W8A8 (weight+activation INT8) on the bottleneck chunk2.

Usage:
    python build_w8a8.py --output /tmp/w8a8-test
"""
from __future__ import annotations
import argparse, os, sys, shutil, time
import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4
from ane_ops import MODEL_DTYPE

HF_DIR = f"{ROOT}/output/gemma4-e2b-final/hf_model"
CTX = 8192
W = 512
fp16 = ct.converters.mil.mil.types.fp16


def build_chunk(chunk_cls, base, chunk_name, ctx, output_dir, quant_mode="int4"):
    """Build and convert one chunk with specified quantization."""
    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512

    chunk = chunk_cls(base).eval()

    # Build sample inputs based on chunk type
    if chunk_name == "chunk1":
        sample = (
            torch.zeros(1, 1, hidden, dtype=torch.float16),
            torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
            torch.zeros(1, 1, 1, W, dtype=torch.float16),
            torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
            torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
            torch.zeros(1, 1, 1, 256, dtype=torch.float16),
            torch.zeros(1, 1, 1, 256, dtype=torch.float16),
            torch.zeros(1, 1, 1, 512, dtype=torch.float16),
            torch.zeros(1, 1, 1, 512, dtype=torch.float16),
            torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
            torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
            torch.zeros(1, 1, ctx, max_hd, dtype=torch.float16),
            torch.zeros(1, 1, ctx, max_hd, dtype=torch.float16),
        )
        inputs = [
            ct.TensorType(name="hidden_states",       shape=sample[0].shape, dtype=fp16),
            ct.TensorType(name="causal_mask_full",    shape=sample[1].shape, dtype=fp16),
            ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
            ct.TensorType(name="update_mask",         shape=sample[3].shape, dtype=fp16),
            ct.TensorType(name="per_layer_raw",       shape=sample[4].shape, dtype=fp16),
            ct.TensorType(name="cos_s",               shape=sample[5].shape, dtype=fp16),
            ct.TensorType(name="sin_s",               shape=sample[6].shape, dtype=fp16),
            ct.TensorType(name="cos_f",               shape=sample[7].shape, dtype=fp16),
            ct.TensorType(name="sin_f",               shape=sample[8].shape, dtype=fp16),
            ct.TensorType(name="K_sliding_in",        shape=sample[9].shape, dtype=fp16),
            ct.TensorType(name="V_sliding_in",        shape=sample[10].shape, dtype=fp16),
            ct.TensorType(name="K_full_in",           shape=sample[11].shape, dtype=fp16),
            ct.TensorType(name="V_full_in",           shape=sample[12].shape, dtype=fp16),
        ]
        outputs = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
                    "K_full_out", "V_full_out", "per_layer_combined_out"]

    elif chunk_name == "chunk2":
        sample = (
            torch.zeros(1, 1, hidden, dtype=torch.float16),
            torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
            torch.zeros(1, 1, 1, W, dtype=torch.float16),
            torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
            torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
            torch.zeros(1, 1, 1, 256, dtype=torch.float16),
            torch.zeros(1, 1, 1, 256, dtype=torch.float16),
            torch.zeros(1, 1, 1, 512, dtype=torch.float16),
            torch.zeros(1, 1, 1, 512, dtype=torch.float16),
            torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
            torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
            torch.zeros(2, 1, ctx, max_hd, dtype=torch.float16),
            torch.zeros(2, 1, ctx, max_hd, dtype=torch.float16),
        )
        inputs = [
            ct.TensorType(name="hidden_states",       shape=sample[0].shape, dtype=fp16),
            ct.TensorType(name="causal_mask_full",    shape=sample[1].shape, dtype=fp16),
            ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
            ct.TensorType(name="update_mask",         shape=sample[3].shape, dtype=fp16),
            ct.TensorType(name="per_layer_combined",  shape=sample[4].shape, dtype=fp16),
            ct.TensorType(name="cos_s",               shape=sample[5].shape, dtype=fp16),
            ct.TensorType(name="sin_s",               shape=sample[6].shape, dtype=fp16),
            ct.TensorType(name="cos_f",               shape=sample[7].shape, dtype=fp16),
            ct.TensorType(name="sin_f",               shape=sample[8].shape, dtype=fp16),
            ct.TensorType(name="K_sliding_in",        shape=sample[9].shape, dtype=fp16),
            ct.TensorType(name="V_sliding_in",        shape=sample[10].shape, dtype=fp16),
            ct.TensorType(name="K_full_in",           shape=sample[11].shape, dtype=fp16),
            ct.TensorType(name="V_full_in",           shape=sample[12].shape, dtype=fp16),
        ]
        outputs = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
                    "K_full_out", "V_full_out", "kv13_k", "kv13_v", "kv14_k", "kv14_v"]

    elif chunk_name in ("chunk3", "chunk4"):
        sample = (
            torch.zeros(1, 1, hidden, dtype=torch.float16),
            torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
            torch.zeros(1, 1, 1, W, dtype=torch.float16),
            torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
            torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
            torch.zeros(1, 1, 1, 256, dtype=torch.float16),
            torch.zeros(1, 1, 1, 256, dtype=torch.float16),
            torch.zeros(1, 1, 1, 512, dtype=torch.float16),
            torch.zeros(1, 1, 1, 512, dtype=torch.float16),
            torch.zeros(1, 1, W, 256, dtype=torch.float16),
            torch.zeros(1, 1, W, 256, dtype=torch.float16),
            torch.zeros(1, 1, ctx, 512, dtype=torch.float16),
            torch.zeros(1, 1, ctx, 512, dtype=torch.float16),
        )
        inputs = [
            ct.TensorType(name="hidden_states",       shape=sample[0].shape, dtype=fp16),
            ct.TensorType(name="causal_mask_full",    shape=sample[1].shape, dtype=fp16),
            ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
            ct.TensorType(name="update_mask",         shape=sample[3].shape, dtype=fp16),
            ct.TensorType(name="per_layer_combined",  shape=sample[4].shape, dtype=fp16),
            ct.TensorType(name="cos_s",               shape=sample[5].shape, dtype=fp16),
            ct.TensorType(name="sin_s",               shape=sample[6].shape, dtype=fp16),
            ct.TensorType(name="cos_f",               shape=sample[7].shape, dtype=fp16),
            ct.TensorType(name="sin_f",               shape=sample[8].shape, dtype=fp16),
            ct.TensorType(name="kv13_k",              shape=sample[9].shape, dtype=fp16),
            ct.TensorType(name="kv13_v",              shape=sample[10].shape, dtype=fp16),
            ct.TensorType(name="kv14_k",              shape=sample[11].shape, dtype=fp16),
            ct.TensorType(name="kv14_v",              shape=sample[12].shape, dtype=fp16),
        ]
        if chunk_name == "chunk3":
            outputs = ["hidden_states_out"]
        else:
            outputs = ["token_id", "token_logit", "hidden_states_out"]

    print(f"\n{'='*60}")
    print(f"Building {chunk_name} (ctx={ctx}, quant={quant_mode})")
    print(f"{'='*60}")

    # Trace
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(chunk, sample, check_trace=False)
    print(f"  Traced in {time.time()-t:.1f}s")

    # Convert to CoreML (FP16, no quantization yet)
    t = time.time()
    mlmodel = ct.convert(
        traced, inputs=inputs,
        outputs=[ct.TensorType(name=n) for n in outputs],
        minimum_deployment_target=ct.target.iOS26,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"  Converted in {time.time()-t:.1f}s")

    # Apply quantization
    t = time.time()
    if quant_mode == "int4":
        # Current approach: INT4 palettization
        cfg = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                nbits=4, granularity="per_grouped_channel", group_size=32))
        mlmodel = cto.palettize_weights(mlmodel, cfg)
        print(f"  INT4 palettized in {time.time()-t:.1f}s")

    elif quant_mode == "w8":
        # INT8 weight-only linear quantization
        cfg = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=np.int8,
                granularity="per_channel",
            ))
        mlmodel = cto.linear_quantize_weights(mlmodel, cfg)
        print(f"  W8 quantized in {time.time()-t:.1f}s")

    elif quant_mode == "w8a8":
        # Step 1: Activation quantization with calibration data
        # Create calibration data - use random inputs that mimic real inference
        print("  Generating calibration data...")
        cal_data = []
        for i in range(5):
            d = {}
            for inp, s in zip(inputs, sample):
                # Use random normal data scaled to fp16 range
                arr = np.random.randn(*[x for x in s.shape]).astype(np.float16) * 0.1
                d[inp.name] = arr
            cal_data.append(d)

        act_cfg = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=np.int8,
            ))
        print("  Quantizing activations...")
        mlmodel = cto.linear_quantize_activations(mlmodel, act_cfg, cal_data)
        print(f"  A8 done in {time.time()-t:.1f}s")

        # Step 2: Weight quantization
        t2 = time.time()
        w_cfg = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=np.int8,
                granularity="per_channel",
            ))
        mlmodel = cto.linear_quantize_weights(mlmodel, w_cfg)
        print(f"  W8 done in {time.time()-t2:.1f}s")

    elif quant_mode == "fp16":
        print("  No quantization (FP16)")

    # Save
    save_path = os.path.join(output_dir, f"{chunk_name}.mlpackage")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    mlmodel.save(save_path)

    # Report size
    import subprocess
    size = subprocess.check_output(["du", "-sh", save_path]).decode().split()[0]
    print(f"  Saved to {save_path} ({size})")
    return mlmodel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/tmp/w8a8-test")
    parser.add_argument("--quant", type=str, default="w8", choices=["int4", "w8", "w8a8", "fp16"],
                        help="Quantization mode")
    parser.add_argument("--chunks", type=str, default="1,2,3,4",
                        help="Comma-separated chunk numbers to build")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading Gemma 4 E2B...")
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=CTX)
    base.eval()

    chunk_map = {
        "1": (SWAChunk1, "chunk1"),
        "2": (SWAChunk2, "chunk2"),
        "3": (SWAChunk3, "chunk3"),
        "4": (SWAChunk4, "chunk4"),
    }

    for cn in args.chunks.split(","):
        cn = cn.strip()
        if cn in chunk_map:
            cls, name = chunk_map[cn]
            build_chunk(cls, base, name, CTX, args.output, args.quant)

    print(f"\nDone! Models in {args.output}/")


if __name__ == "__main__":
    main()
