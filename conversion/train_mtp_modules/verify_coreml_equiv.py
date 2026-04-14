#!/usr/bin/env python3
"""Numerical equivalence test: PyTorch MtpModuleANE vs converted CoreML mlpackage.

Prevents silent conversion drift that would invalidate overnight training.
Runs the same random inputs through both, compares outputs.

Usage:
  python3 verify_coreml_equiv.py \
      --ckpt ~/Downloads/mtp_final.pt \
      --module-idx 0 \
      --hf-dir ~/.cache/huggingface/.../gemma-4-E2B-it \
      --mlpackage /tmp/mtp_pipeline_verify/mtp_module_0.mlpackage

Thresholds (fp16 math after CoreML conversion):
  hidden_out:  max_abs_diff < 0.05 is typical; >0.2 is suspicious
  top_k_indices[0]: argmax may differ when top-1 and top-2 logits are
                    within softmax noise; acceptable if top1 logit gap is small.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_mtp_modules.mtp_modules import MtpModuleConfig
from train_mtp_modules.build_mtp_coreml import (
    MtpModuleANE,
    load_trained_module_weights,
    load_lm_head_from_hf,
)
from ane_ops import MODEL_DTYPE


def build_pt_model(ckpt_path: str, module_idx: int, hf_dir: str) -> MtpModuleANE:
    cfg = MtpModuleConfig()
    model = MtpModuleANE(cfg, include_lm_head=True).to(MODEL_DTYPE).eval()
    load_trained_module_weights(model, ckpt_path, module_idx)
    load_lm_head_from_hf(model, hf_dir)
    return model


def make_random_inputs(cfg: MtpModuleConfig, seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    H, NKV, HD, W = cfg.hidden_size, cfg.num_kv_heads, cfg.head_dim, cfg.kv_window
    half = HD // 2
    inputs = {
        # magnitude matches training data (L34 pre-norm norm ~50 for seq of 3 tokens)
        "hidden_prev": torch.randn(1, 1, H, generator=rng, dtype=MODEL_DTYPE) * 0.5,
        "embed_token": torch.randn(1, 1, H, generator=rng, dtype=MODEL_DTYPE) * 1.0,
        "kv_k_in": torch.randn(1, NKV, W, HD, generator=rng, dtype=MODEL_DTYPE) * 0.3,
        "kv_v_in": torch.randn(1, NKV, W, HD, generator=rng, dtype=MODEL_DTYPE) * 0.3,
        "cos": torch.cos(torch.arange(half, dtype=torch.float32) * 0.01).to(MODEL_DTYPE).unsqueeze(0),
        "sin": torch.sin(torch.arange(half, dtype=torch.float32) * 0.01).to(MODEL_DTYPE).unsqueeze(0),
        "mask": torch.zeros(1, 1, 1, W, dtype=MODEL_DTYPE),
        # write at an arbitrary slot (say 5)
        "update_idx": torch.zeros(1, 1, W, 1, dtype=MODEL_DTYPE),
    }
    inputs["update_idx"][0, 0, 5, 0] = 1.0
    # Mask out slots not yet written to avoid reading garbage — keep first 6 valid
    for i in range(6, W):
        inputs["mask"][0, 0, 0, i] = -65500.0  # fp16 -inf-ish
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--module-idx", type=int, required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--mlpackage", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== Loading PyTorch reference (module {args.module_idx}) ===")
    pt_model = build_pt_model(args.ckpt, args.module_idx, args.hf_dir)

    cfg = MtpModuleConfig()
    inputs = make_random_inputs(cfg, seed=args.seed)

    print("\n=== PyTorch forward ===")
    with torch.no_grad():
        pt_out = pt_model(**inputs)
    pt_top_ids, pt_top_vals, pt_hidden, pt_kv_k, pt_kv_v = pt_out
    print(f"  hidden_out: shape={tuple(pt_hidden.shape)} norm={pt_hidden.float().norm():.4f}")
    print(f"  top_k_indices[:5]: {pt_top_ids[:5].tolist()}")
    print(f"  top_k_values[:5]:  {[f'{v:.3f}' for v in pt_top_vals[:5].tolist()]}")

    print(f"\n=== Loading CoreML model ({args.mlpackage}) ===")
    import coremltools as ct
    ml = ct.models.MLModel(args.mlpackage, compute_units=ct.ComputeUnit.CPU_ONLY)

    ml_inputs = {k: v.float().numpy() for k, v in inputs.items()}
    print("\n=== CoreML predict ===")
    ml_out = ml.predict(ml_inputs)
    for k, v in ml_out.items():
        arr = np.asarray(v)
        print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")

    ml_hidden = torch.from_numpy(np.asarray(ml_out["hidden_out"]))
    ml_top_ids = np.asarray(ml_out["top_k_indices"]).flatten()
    ml_top_vals = np.asarray(ml_out["top_k_values"]).flatten()

    print("\n=== Comparison ===")
    diff_hidden = (pt_hidden.float() - ml_hidden.float()).abs()
    print(f"  hidden_out max abs diff: {diff_hidden.max().item():.6f}")
    print(f"  hidden_out mean abs diff: {diff_hidden.mean().item():.6f}")
    print(f"  hidden_out rel diff (||diff||/||pt||): "
          f"{diff_hidden.norm().item() / max(pt_hidden.float().norm().item(), 1e-9):.6f}")

    # Build outputs argmax (shape (1,)) as of the topk fix; PyTorch side emits
    # the same thing (first element of what used to be topk(8)).
    pt_argmax = int(pt_top_ids.flatten()[0].item())
    ml_argmax = int(ml_top_ids.flatten()[0])
    print(f"\n  PT argmax id: {pt_argmax}")
    print(f"  ML argmax id: {ml_argmax}")
    top1_match = pt_argmax == ml_argmax

    # KV cache outputs
    diff_kv_k = (pt_kv_k.float() - torch.from_numpy(np.asarray(ml_out["kv_k_out"])).float()).abs()
    diff_kv_v = (pt_kv_v.float() - torch.from_numpy(np.asarray(ml_out["kv_v_out"])).float()).abs()
    print(f"  kv_k_out max abs diff: {diff_kv_k.max().item():.6f}")
    print(f"  kv_v_out max abs diff: {diff_kv_v.max().item():.6f}")

    # Pass/fail
    HIDDEN_THRESH = 0.2
    ok_hidden = diff_hidden.max().item() < HIDDEN_THRESH
    ok_top = top1_match  # Must match exactly (argmax is deterministic modulo fp noise)
    print()
    print(f"  hidden_out within {HIDDEN_THRESH}: {'PASS' if ok_hidden else 'FAIL'}")
    print(f"  argmax match:              {'PASS' if ok_top else 'FAIL'}")
    if ok_hidden and ok_top:
        print("\n[OK] Conversion preserves numerical semantics.")
        sys.exit(0)
    else:
        print("\n[FAIL] Conversion has unacceptable drift.")
        sys.exit(1)


if __name__ == "__main__":
    main()
