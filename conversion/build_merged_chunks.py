#!/usr/bin/env python3
"""Build merged (2-chunk or 1-chunk) decode models for Gemma 4 E2B.

Motivation: ANE dispatch overhead is the shipping bottleneck (see
`docs/BASELINE_SPEED_AUDIT.md`). Halving chunk count halves the serial
round-trip cost per decode step.

Outputs (under --output):
  --mode two:
    merged_chunk1.mlpackage  — L0-14  + PLE,   owns KV
    merged_chunk2.mlpackage  — L15-34 + norm + lm_head + argmax

  --mode one:
    merged_full.mlpackage    — L0-34 + PLE + norm + lm_head + argmax

Both variants ship alongside the existing 4-chunk chunk1..chunk4 mlpackages
without replacing them. The Swift runtime auto-detects which layout is
present and falls back to 4-chunk if the merged variants are missing.

Usage:
    python build_merged_chunks.py --mode two  --output /tmp/gemma4-2chunk
    python build_merged_chunks.py --mode one  --output /tmp/gemma4-1chunk
    python build_merged_chunks.py --mode both --output /tmp/gemma4-merged

Requires GEMMA4_HF_DIR env var or HF model at the default path.

After conversion:
  1. Load the mlpackage on a Mac with a fresh Python session.
  2. Run `python test_merged_parity.py --ref /path/to/4chunk --merged /path/to/merged`
     to confirm cosine similarity > 0.9999 against the 4-chunk reference.
  3. Copy to iPhone, run ComputePlanAudit to verify ANE placement.
  4. Bench tok/s on device (see docs/CHUNK_CONSOLIDATION_BENCH.md).
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

from models.gemma4 import Gemma4Model
from models.gemma4_swa_merged2 import MergedChunk12, MergedChunk34
from models.gemma4_swa_merged1 import MergedChunk1

HF_DIR = os.environ.get("GEMMA4_HF_DIR", f"{ROOT}/../output/gemma4-e2b/hf_model")
fp16 = np.float16


def trace_and_convert(model, sample_inputs, input_specs, output_names,
                      save_path, quantize=True):
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

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    mlmodel.save(save_path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(save_path)
        for f in fns
    ) / 1024 / 1024
    print(f"    saved {save_path} ({size_mb:.1f} MB)")
    return mlmodel


def build_two_chunk(base, out_dir, ctx, W, quantize):
    """Build MergedChunk12 + MergedChunk34."""
    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512

    # 12 sliding + 3 full across L0-14 (matches SWAChunk1+SWAChunk2 concatenated).
    num_sliding = 12
    num_full = 3

    # ================================================================
    # Merged chunk 1: L0-14, own KV, produces kv13/kv14
    # ================================================================
    print("\n" + "=" * 60)
    print("MERGED CHUNK 1 (L0-14, 15 layers)")
    print("=" * 60)
    m12 = MergedChunk12(base).eval()
    s1 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(num_sliding, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(num_sliding, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(num_full, 1, ctx, max_hd, dtype=torch.float16),
        torch.zeros(num_full, 1, ctx, max_hd, dtype=torch.float16),
    )
    in1 = [
        ct.TensorType(name="hidden_states",       shape=s1[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s1[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s1[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s1[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",       shape=s1[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s1[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s1[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s1[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s1[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=s1[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=s1[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=s1[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=s1[12].shape, dtype=fp16),
    ]
    out1 = ["hidden_states_out",
            "K_sliding_out", "V_sliding_out",
            "K_full_out", "V_full_out",
            "kv13_k", "kv13_v", "kv14_k", "kv14_v",
            "per_layer_combined_out"]
    trace_and_convert(m12, s1, in1, out1,
                      f"{out_dir}/merged_chunk1.mlpackage", quantize=quantize)
    del m12

    # ================================================================
    # Merged chunk 2: L15-34 + norm + lm_head, all KV-shared
    # ================================================================
    print("\n" + "=" * 60)
    print("MERGED CHUNK 2 (L15-34 + LM head, 20 layers)")
    print("=" * 60)
    m34 = MergedChunk34(base).eval()
    s2 = (
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
    in2 = [
        ct.TensorType(name="hidden_states",       shape=s2[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s2[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s2[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s2[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s2[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s2[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s2[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s2[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s2[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=s2[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=s2[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=s2[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=s2[12].shape, dtype=fp16),
    ]
    out2 = ["token_id", "token_logit", "hidden_states_out"]
    trace_and_convert(m34, s2, in2, out2,
                      f"{out_dir}/merged_chunk2.mlpackage", quantize=quantize)
    del m34


def build_one_chunk(base, out_dir, ctx, W, quantize):
    """Build a single-graph 35-layer variant. Experimental — may fall off ANE."""
    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512

    num_sliding = 12
    num_full = 3

    print("\n" + "=" * 60)
    print("MERGED FULL (L0-34 + LM head, 35 layers) — experimental")
    print("=" * 60)
    mfull = MergedChunk1(base).eval()
    s = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(num_sliding, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(num_sliding, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(num_full, 1, ctx, max_hd, dtype=torch.float16),
        torch.zeros(num_full, 1, ctx, max_hd, dtype=torch.float16),
    )
    in_specs = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",       shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=s[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=s[12].shape, dtype=fp16),
    ]
    out_names = ["token_id", "token_logit", "hidden_states_out",
                 "K_sliding_out", "V_sliding_out", "K_full_out", "V_full_out"]
    try:
        trace_and_convert(mfull, s, in_specs, out_names,
                          f"{out_dir}/merged_full.mlpackage", quantize=quantize)
        print("\n[build_merged_chunks] merged_full converted. ")
        print("  NEXT STEP: compile on iPhone 17 Pro and run ComputePlanAudit")
        print("  BEFORE shipping — 35 layers may exceed the ANE per-function stability")
        print("  ceiling (historical ~15 layers).")
    except Exception as e:  # conversion can OOM or raise; surface cleanly
        print(f"\n[build_merged_chunks] 1-chunk conversion failed: {e}")
        print("  This is a known risk — 35 layers is well above the per-function ceiling.")
        print("  2-chunk variant (merged_chunk1 + merged_chunk2) is the fallback.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build 2-chunk / 1-chunk merged Gemma 4 decode mlpackages")
    parser.add_argument("--mode", type=str, choices=["two", "one", "both"],
                        default="two",
                        help="Which merged variant(s) to build")
    parser.add_argument("--output", type=str, default="/tmp/gemma4-merged",
                        help="Output directory")
    parser.add_argument("--hf-dir", type=str, default=HF_DIR,
                        help="HuggingFace model directory")
    parser.add_argument("--ctx", type=int, default=2048,
                        help="Context length (default: 2048)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip int4 palettization (for parity debugging)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    W = 512
    quantize = not args.no_quantize

    print(f"Loading Gemma 4 E2B from {args.hf_dir}...")
    base = Gemma4Model.from_pretrained(args.hf_dir, context_length=args.ctx)
    base.eval()

    print(f"\nctx={args.ctx}, W={W}, quantize={'int4' if quantize else 'fp16'}")

    if args.mode in ("two", "both"):
        build_two_chunk(base, args.output, args.ctx, W, quantize)

    if args.mode in ("one", "both"):
        build_one_chunk(base, args.output, args.ctx, W, quantize)

    print(f"\n{'='*60}")
    print(f"Merged chunks saved to {args.output}/")
    print(f"Next: run `python test_merged_parity.py` to verify numerical parity")
    print(f"against the 4-chunk reference model.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
