#!/usr/bin/env python3
"""Build a standalone chunk4_subset.mlpackage for sparse-LM-head MTP verify.

Saves ~7-10 ms per cycle on iPhone vs full 262144-vocab LM head matmul, by
computing logits only over a caller-provided candidate set (typically 1024
tokens drawn from drafter top-K + PLD history + top-N frequent tokens).

Output: `chunk4_subset.mlpackage` with a single function `verify_qK_subset`
that takes the same inputs as `verify_qK` PLUS `candidate_ids: (M,) int32`,
and returns the predicted token IDs after gather-back from the subset.

Usage:
    python conversion/build_chunk4_subset.py --K 3 --num-candidates 1024
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

from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import (
    SWAVerifyChunk4Subset, compute_chunk_boundaries,
)

fp16 = np.float16


def _resolve_hf_dir(model_name: str, override: str | None) -> str:
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
                allow_patterns=["*.safetensors", "*.json", "tokenizer*",
                                "*.txt", "*.model"],
            )
        return local
    raise SystemExit(f"unknown model {model_name}")


def _du_mb(path: str) -> float:
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024 / 1024
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
    return total / 1024 / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gemma4-e2b",
                    help="Model name in MODEL_REGISTRY")
    ap.add_argument("--output", type=str, default=None,
                    help="Output directory (default: ../output/<model>/chunks_subset)")
    ap.add_argument("--hf-dir", type=str, default=None)
    ap.add_argument("--K", type=int, default=3,
                    help="Verify batch size (Q=K)")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--num-candidates", type=int, default=1024,
                    help="Subset vocab size (caller passes this many candidate IDs)")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()

    if args.ctx is None and args.model in MODEL_REGISTRY:
        args.ctx = MODEL_REGISTRY[args.model].default_context_length
    elif args.ctx is None:
        args.ctx = 2048
    args.output = args.output or os.path.join(
        ROOT, "..", "output", args.model, "chunks_subset")
    os.makedirs(args.output, exist_ok=True)
    quantize = not args.no_quantize
    K = args.K
    M = args.num_candidates

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading Gemma 4 from {hf_dir} ...")
    base = Gemma4Model.from_pretrained(hf_dir)
    base.eval()
    cfg = base.config

    boundaries = compute_chunk_boundaries(cfg)
    c4_start, c4_end = boundaries[3]
    print(f"\n  [verify_qK_subset] K={K} M={M}, layers L{c4_start}-{c4_end-1}")

    vc4 = SWAVerifyChunk4Subset(
        base, seq_len=K, start=c4_start, end=c4_end,
        num_candidates=M).eval()

    hidden = cfg.hidden_size
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    nkv = cfg.num_key_value_heads
    nlayers = cfg.num_hidden_layers
    pld = cfg.hidden_size_per_layer_input
    CTX = args.ctx

    # Sample inputs (zeros) for tracing. NO candidate_ids — Swift handles
    # sparse matmul outside the model.
    sample = (
        torch.zeros(1, K, hidden, dtype=torch.float16),
        torch.zeros(1, 1, K, CTX, dtype=torch.float16),    # causal_mask_full
        torch.zeros(1, 1, K, W, dtype=torch.float16),      # causal_mask_sliding
        torch.zeros(1, K, nlayers * pld, dtype=torch.float16),  # per_layer_combined
        torch.zeros(1, 1, K, hd_s, dtype=torch.float16),   # cos_s
        torch.zeros(1, 1, K, hd_s, dtype=torch.float16),   # sin_s
        torch.zeros(1, 1, K, hd_f, dtype=torch.float16),   # cos_f
        torch.zeros(1, 1, K, hd_f, dtype=torch.float16),   # sin_f
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),  # kv13_k
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),  # kv13_v
        torch.zeros(1, nkv, CTX, hd_f, dtype=torch.float16),  # kv14_k
        torch.zeros(1, nkv, CTX, hd_f, dtype=torch.float16),  # kv14_v
    )

    # Smoke test forward.
    print("  Smoke test (zeros) ...")
    with torch.no_grad():
        out = vc4(*sample)
        print(f"    output token_ids: {out[0].shape} {out[0].dtype}")
        print(f"    output hidden:    {out[1].shape} {out[1].dtype}")

    inputs = [
        ct.TensorType(name="hidden_states",       shape=sample[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=sample[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",               shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",               shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",               shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",               shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=sample[8].shape, dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=sample[9].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=sample[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=sample[11].shape, dtype=fp16),
    ]
    outputs = ["normed_hidden", "hidden_states_out"]

    print("\n  Tracing...")
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(vc4, sample, check_trace=False)
    print(f"    traced in {time.time()-t:.1f}s")

    print("  Converting to CoreML ...")
    t = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=n) for n in outputs],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"    converted in {time.time()-t:.1f}s")

    if quantize:
        print("  Palettizing weights INT4 (group_size=32) ...")
        cfg_q = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                nbits=4, granularity="per_grouped_channel", group_size=32))
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg_q)

    out_path = os.path.join(args.output, "chunk4_subset.mlpackage")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    mlmodel.save(out_path)
    print(f"\n  Saved {out_path} ({_du_mb(out_path):.1f} MB)")

    print(f"\n{'='*60}")
    print(f"chunk4_subset built — pair with existing chunk1/2/3 mlmodelc.")
    print(f"  Inputs include `candidate_ids` (shape (M={M},), int32).")
    print(f"  Outputs token_ids (1, K={K}) + hidden_states_out.")
    print(f"  Swift caller builds candidate set per cycle + dispatches here.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
