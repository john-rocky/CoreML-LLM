#!/usr/bin/env python3
"""Build stateful (MLState) CoreML chunks — KV cache as internal state.

Prototype: chunk2 only. If this compiles and runs on ANE, expand to chunk1.
chunk3/4 have no KV state and don't need MLState.

Usage:
    python build_stateful.py --output /tmp/stateful-8k --chunk-only 2
"""
from __future__ import annotations
import argparse, os, sys, shutil, time
import torch
import numpy as np
import coremltools as ct

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_stateful_chunks import StatefulChunk2
from ane_ops import MODEL_DTYPE

HF_DIR = f"{ROOT}/output/gemma4-e2b-final/hf_model"
CTX = 8192
W = 512
fp16 = ct.converters.mil.mil.types.fp16


def build_stateful_chunk2(base, output_dir, nbits=4):
    """Convert StatefulChunk2 with KV as MLState."""
    print(f"\n=== StatefulChunk2 (L8-14, MLState KV) ===")
    hidden = base.config.hidden_size
    nlayers = base.config.num_hidden_layers
    pld = base.config.hidden_size_per_layer_input
    max_hd = base.config.global_head_dim

    chunk = StatefulChunk2(base).eval()

    # Sample inputs — NO KV inputs (KV is internal state)
    sample_inputs = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),           # hidden_states
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),           # causal_mask_full
        torch.zeros(1, 1, 1, W, dtype=torch.float16),             # causal_mask_sliding
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),           # update_mask
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),    # per_layer_combined
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),           # cos_s
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),           # sin_s
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),           # cos_f
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),           # sin_f
    )

    input_specs = [
        ct.TensorType(name="hidden_states", shape=(1, 1, hidden), dtype=fp16),
        ct.TensorType(name="causal_mask_full", shape=(1, 1, 1, CTX), dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=(1, 1, 1, W), dtype=fp16),
        ct.TensorType(name="update_mask", shape=(1, 1, CTX, 1), dtype=fp16),
        ct.TensorType(name="per_layer_combined", shape=(1, 1, nlayers * pld), dtype=fp16),
        ct.TensorType(name="cos_s", shape=(1, 1, 1, 256), dtype=fp16),
        ct.TensorType(name="sin_s", shape=(1, 1, 1, 256), dtype=fp16),
        ct.TensorType(name="cos_f", shape=(1, 1, 1, 512), dtype=fp16),
        ct.TensorType(name="sin_f", shape=(1, 1, 1, 512), dtype=fp16),
    ]

    output_specs = [
        ct.TensorType(name="hidden_states_out"),
        ct.TensorType(name="kv13_k"),
        ct.TensorType(name="kv13_v"),
        ct.TensorType(name="kv14_k"),
        ct.TensorType(name="kv14_v"),
    ]

    # State declarations — 2 buffers: kv_sliding (10, 1, W, hd) + kv_full (4, 1, CTX, hd)
    n_sliding = 5
    n_full = 2
    state_specs = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=(n_sliding * 2, 1, W, max_hd), dtype=np.float16),
            name="kv_sliding",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=(n_full * 2, 1, CTX, max_hd), dtype=np.float16),
            name="kv_full",
        ),
    ]

    print(f"  States: kv_sliding ({n_sliding*2}, 1, {W}, {max_hd}) + kv_full ({n_full*2}, 1, {CTX}, {max_hd})")

    # Trace
    t = time.time()
    with torch.no_grad():
        chunk.kv_sliding.zero_()
        chunk.kv_full.zero_()
        traced = torch.jit.trace(chunk, sample_inputs, check_trace=False)
    print(f"  traced in {time.time()-t:.1f}s")

    # Convert with state
    t = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
        outputs=output_specs,
        states=state_specs,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"  converted in {time.time()-t:.1f}s")

    # Palettize
    if nbits:
        t = time.time()
        cfg = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                nbits=nbits, granularity="per_grouped_channel", group_size=32))
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg)
        print(f"  palettized W{nbits} in {time.time()-t:.1f}s")

    save_path = os.path.join(output_dir, "chunk2.mlpackage")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    mlmodel.save(save_path)
    sz = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fns in os.walk(save_path) for f in fns)
    print(f"  saved ({sz / 1e6:.0f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/tmp/stateful-8k")
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--chunk-only", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print("Loading Gemma 4 E2B...")
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=CTX)
    base.eval()

    if args.chunk_only == 2:
        build_stateful_chunk2(base, args.output, nbits=args.nbits)


if __name__ == "__main__":
    main()
