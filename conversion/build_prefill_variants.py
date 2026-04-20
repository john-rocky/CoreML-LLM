#!/usr/bin/env python3
"""Build precompiled prefill chunks at multiple static batch sizes (S1).

The runtime (`Sources/CoreMLLLM/ChunkedEngine.swift`) prefers the smallest
precompiled prefill variant whose static N is >= the live prompt length,
so that a 50-token chat turn does not pay the ANE cost of padding to 512.
This script produces sibling mlpackages named `prefill_chunk{1..4}_b{N}`
for each requested N. The default-sized `prefill_chunk{1..4}` (typically
N=512) is left untouched and remains the catch-all for long prompts.

See docs/LITERT_PERF_ADOPTIONS.md §S1 for rationale.

Usage:
    python conversion/build_prefill_variants.py \\
        --hf-dir ./output/gemma4-e2b/hf_model \\
        --output ./output/gemma4-e2b \\
        --sizes 32 64 128 256

Naming output:
    output/prefill_chunk1_b32.mlpackage
    output/prefill_chunk2_b32.mlpackage
    output/prefill_chunk3_b32.mlpackage
    output/prefill_chunk4_b32.mlpackage
    output/prefill_chunk1_b64.mlpackage
    ... etc.

Quantization: matches the default prefill build (4-bit per-grouped-channel
palettization, group_size=32) so variants are byte-comparable to the main
prefill chunks. Override with --no-quantize for FP16 debugging builds.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import coremltools as ct

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_prefill_chunks import (
    PrefillChunk1, PrefillChunk2, PrefillChunk3, PrefillChunk4,
)

fp16 = ct.converters.mil.mil.types.fp16

CTX_DEFAULT = 2048


def do_convert(model, sample_inputs, input_specs, output_names,
               save_path, quantize: bool):
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


def export_variant(base: Gemma4Model, N: int, out_dir: Path, quantize: bool):
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    total_pld = nlayers * pld

    print(f"\n=== Prefill variants N={N} ===")

    # ---- Chunk 1: layers 0-7 + PLE ----
    print(f"\n  prefill_chunk1_b{N} (L0-7 + PLE)")
    pc1 = PrefillChunk1(base).eval()
    s1 = (
        torch.zeros(1, N, hidden, dtype=torch.float16),         # hidden_states
        torch.zeros(1, 1, N, N, dtype=torch.float16),            # causal_mask
        torch.zeros(1, N, total_pld, dtype=torch.float16),       # per_layer_raw
        torch.zeros(1, 1, N, 256, dtype=torch.float16),          # cos_s
        torch.zeros(1, 1, N, 256, dtype=torch.float16),          # sin_s
        torch.zeros(1, 1, N, 512, dtype=torch.float16),          # cos_f
        torch.zeros(1, 1, N, 512, dtype=torch.float16),          # sin_f
    )
    in1 = [
        ct.TensorType(name="hidden_states",   shape=s1[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask",     shape=s1[1].shape, dtype=fp16),
        ct.TensorType(name="per_layer_raw",   shape=s1[2].shape, dtype=fp16),
        ct.TensorType(name="cos_s",           shape=s1[3].shape, dtype=fp16),
        ct.TensorType(name="sin_s",           shape=s1[4].shape, dtype=fp16),
        ct.TensorType(name="cos_f",           shape=s1[5].shape, dtype=fp16),
        ct.TensorType(name="sin_f",           shape=s1[6].shape, dtype=fp16),
    ]
    out1 = [
        "hidden_states_out", "per_layer_combined_out",
        "K0", "V0", "K1", "V1", "K2", "V2", "K3", "V3",
        "K4", "V4", "K5", "V5", "K6", "V6", "K7", "V7",
    ]
    do_convert(pc1, s1, in1, out1, str(out_dir / f"prefill_chunk1_b{N}.mlpackage"), quantize)

    # ---- Chunk 2: layers 8-14 (outputs kv13/kv14 for sharing) ----
    print(f"\n  prefill_chunk2_b{N} (L8-14, outputs kv13/kv14)")
    pc2 = PrefillChunk2(base).eval()
    s2 = (
        torch.zeros(1, N, hidden, dtype=torch.float16),
        torch.zeros(1, 1, N, N, dtype=torch.float16),
        torch.zeros(1, N, total_pld, dtype=torch.float16),
        torch.zeros(1, 1, N, 256, dtype=torch.float16),
        torch.zeros(1, 1, N, 256, dtype=torch.float16),
        torch.zeros(1, 1, N, 512, dtype=torch.float16),
        torch.zeros(1, 1, N, 512, dtype=torch.float16),
    )
    in2 = [
        ct.TensorType(name="hidden_states",         shape=s2[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask",           shape=s2[1].shape, dtype=fp16),
        ct.TensorType(name="per_layer_combined",    shape=s2[2].shape, dtype=fp16),
        ct.TensorType(name="cos_s",                 shape=s2[3].shape, dtype=fp16),
        ct.TensorType(name="sin_s",                 shape=s2[4].shape, dtype=fp16),
        ct.TensorType(name="cos_f",                 shape=s2[5].shape, dtype=fp16),
        ct.TensorType(name="sin_f",                 shape=s2[6].shape, dtype=fp16),
    ]
    out2 = [
        "hidden_states_out",
        "K0", "V0", "K1", "V1", "K2", "V2", "K3", "V3", "K4", "V4",
        "kv13_k", "kv13_v", "kv14_k", "kv14_v",
    ]
    do_convert(pc2, s2, in2, out2, str(out_dir / f"prefill_chunk2_b{N}.mlpackage"), quantize)

    # ---- Chunk 3: layers 15-24, KV-shared via kv13/kv14 ----
    print(f"\n  prefill_chunk3_b{N} (L15-24, KV-shared)")
    pc3 = PrefillChunk3(base).eval()
    s3 = (
        torch.zeros(1, N, hidden, dtype=torch.float16),
        torch.zeros(1, 1, N, N, dtype=torch.float16),
        torch.zeros(1, N, total_pld, dtype=torch.float16),
        torch.zeros(1, 1, N, 256, dtype=torch.float16),
        torch.zeros(1, 1, N, 256, dtype=torch.float16),
        torch.zeros(1, 1, N, 512, dtype=torch.float16),
        torch.zeros(1, 1, N, 512, dtype=torch.float16),
        torch.zeros(1, 1, N, 256, dtype=torch.float16),  # kv13_k
        torch.zeros(1, 1, N, 256, dtype=torch.float16),  # kv13_v
        torch.zeros(1, 1, N, 512, dtype=torch.float16),  # kv14_k
        torch.zeros(1, 1, N, 512, dtype=torch.float16),  # kv14_v
    )
    in3 = [
        ct.TensorType(name="hidden_states",         shape=s3[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask",           shape=s3[1].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",    shape=s3[2].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                 shape=s3[3].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                 shape=s3[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                 shape=s3[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                 shape=s3[6].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",                shape=s3[7].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",                shape=s3[8].shape,  dtype=fp16),
        ct.TensorType(name="kv14_k",                shape=s3[9].shape,  dtype=fp16),
        ct.TensorType(name="kv14_v",                shape=s3[10].shape, dtype=fp16),
    ]
    out3 = ["hidden_states_out"]
    do_convert(pc3, s3, in3, out3, str(out_dir / f"prefill_chunk3_b{N}.mlpackage"), quantize)

    # ---- Chunk 4: layers 25-34 + LM head + last-position pick ----
    print(f"\n  prefill_chunk4_b{N} (L25-34 + LM head)")
    pc4 = PrefillChunk4(base).eval()
    s4 = (
        torch.zeros(1, N, hidden, dtype=torch.float16),
        torch.zeros(1, 1, N, N, dtype=torch.float16),
        torch.zeros(1, N, total_pld, dtype=torch.float16),
        torch.zeros(1, 1, N, 256, dtype=torch.float16),
        torch.zeros(1, 1, N, 256, dtype=torch.float16),
        torch.zeros(1, 1, N, 512, dtype=torch.float16),
        torch.zeros(1, 1, N, 512, dtype=torch.float16),
        torch.zeros(1, 1, N, 256, dtype=torch.float16),
        torch.zeros(1, 1, N, 256, dtype=torch.float16),
        torch.zeros(1, 1, N, 512, dtype=torch.float16),
        torch.zeros(1, 1, N, 512, dtype=torch.float16),
        torch.zeros(1, N, 1, dtype=torch.float16),       # last_position_mask
    )
    in4 = [
        ct.TensorType(name="hidden_states",         shape=s4[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask",           shape=s4[1].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",    shape=s4[2].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",                 shape=s4[3].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",                 shape=s4[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",                 shape=s4[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",                 shape=s4[6].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",                shape=s4[7].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",                shape=s4[8].shape,  dtype=fp16),
        ct.TensorType(name="kv14_k",                shape=s4[9].shape,  dtype=fp16),
        ct.TensorType(name="kv14_v",                shape=s4[10].shape, dtype=fp16),
        ct.TensorType(name="last_position_mask",    shape=s4[11].shape, dtype=fp16),
    ]
    out4 = ["token_id", "token_logit"]
    do_convert(pc4, s4, in4, out4, str(out_dir / f"prefill_chunk4_b{N}.mlpackage"), quantize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True,
                    help="HuggingFace Gemma 4 E2B checkpoint directory")
    ap.add_argument("--output", required=True,
                    help="Output directory (sibling to existing prefill_chunk*.mlpackage)")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[32, 64, 128, 256],
                    help="Static batch sizes to export (default: 32 64 128 256)")
    ap.add_argument("--context-length", type=int, default=CTX_DEFAULT,
                    help="Model context length (default 2048)")
    ap.add_argument("--no-quantize", action="store_true",
                    help="Skip palettization (FP16 debug build)")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Gemma 4 E2B from {args.hf_dir}...")
    base = Gemma4Model.from_pretrained(args.hf_dir, context_length=args.context_length)
    base.eval()

    quantize = not args.no_quantize
    for N in sorted(set(args.sizes)):
        export_variant(base, N=N, out_dir=out_dir, quantize=quantize)

    print("\nAll variants exported.")
    print("Move them next to your existing prefill_chunk*.mlpackage to enable")
    print("the runtime's S1 best-match prefill router (see ChunkedEngine.swift).")


if __name__ == "__main__":
    main()
