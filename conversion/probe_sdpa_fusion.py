#!/usr/bin/env python3
"""Probe: does PyTorch F.scaled_dot_product_attention fuse to a single
CoreML op on ctools 9 / iOS 26 target, with Gemma 4's scale=1.0?

Two questions answered in one script:

1. **Numerical parity**: does SDPA(q,k,v,mask,scale=1.0) match the manual
   attn = softmax(q @ k.T + mask) @ v for fp16 inputs under Gemma 4's
   q_norm/k_norm conditions? If cos < 0.9999 the fusion is lossy.

2. **Fusion survival**: does ct.convert keep SDPA as a single op in the
   lowered MIL program, or does it decompose back to matmul+softmax+matmul?
   If decomposed, there is no speed win and we fall back to manual.

A minimal SWA block (Q=1, KV=512, n_heads=8, hd=256) is the test vehicle.
The previous 2025 test used d^(1/4) pre-scaling — this one uses scale=1.0
directly (matching the Gemma 4 post-norm setup).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct
from ane_ops import MODEL_DTYPE, ane_softmax


def manual_attention(q, k, v, mask):
    """Current Gemma 4 path in gemma4_swa_chunks.py::_run_layer_swa."""
    attn_w = torch.matmul(q, k.transpose(-1, -2))
    attn_w = attn_w + mask
    attn_w = ane_softmax(attn_w, dim=-1)
    return torch.matmul(attn_w, v)


def sdpa_attention(q, k, v, mask):
    """F.scaled_dot_product_attention with Gemma 4's post-norm scale=1.0.

    The scale kwarg was added in PyTorch 2.1; we rely on torch 2.9 in
    the lama-cml env. mask is an additive bias (0 = keep, large-neg = block).
    """
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)


class ManualBlock(torch.nn.Module):
    def forward(self, q, k, v, mask):
        return manual_attention(q, k, v, mask)


class SDPABlock(torch.nn.Module):
    def forward(self, q, k, v, mask):
        return sdpa_attention(q, k, v, mask)


def _census_ops(mlpkg: str) -> tuple[int, list[str]]:
    """Compile the mlpackage, walk its MIL via MLComputePlan, count op types.

    Returns (sdpa_like_count, sorted list of "op_type:count" strings).
    """
    import shutil
    from collections import Counter
    from coremltools.models.compute_plan import MLComputePlan
    mlmodelc = mlpkg.replace(".mlpackage", ".mlmodelc")
    m = ct.models.MLModel(mlpkg)
    compiled = m.get_compiled_model_path()
    if os.path.exists(mlmodelc):
        shutil.rmtree(mlmodelc)
    shutil.copytree(compiled, mlmodelc)
    del m

    plan = MLComputePlan.load_from_path(path=mlmodelc,
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    struct = plan.model_structure
    types: Counter = Counter()

    def walk_b(block):
        for op in block.operations:
            # op.operator_name arrives as either "const" or "ios18.const".
            name = op.operator_name
            if "." in name:
                name = name.split(".", 1)[1]
            types[name] += 1
            for b in op.blocks:
                walk_b(b)

    if hasattr(struct, "program"):
        for fname, func in struct.program.functions.items():
            walk_b(func.block)

    sdpa_names = {"scaled_dot_product_attention",
                  "scaled_dot_product_attention_sliced_q", "sdpa"}
    sdpa_count = sum(types[n] for n in sdpa_names)
    top = [f"{k}:{v}" for k, v in types.most_common(20)]
    return sdpa_count, top


def _convert(model, sample, input_names, *, label: str):
    print(f"\n[{label}] tracing + converting...")
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, check_trace=False)
    inputs = [ct.TensorType(name=n, shape=s.shape, dtype=np.float16)
              for n, s in zip(input_names, sample)]
    mlm = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="out")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"  converted in {time.time()-t:.1f}s")
    return mlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--hd", type=int, default=256)
    ap.add_argument("--kvlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Gemma 4 SWA-style Q=1, KV=W=512, 8 heads, 256 head_dim after GQA-repeat.
    B, H, Q, K, D = 1, args.heads, 1, args.kvlen, args.hd
    q = torch.randn(B, H, Q, D, dtype=torch.float16) * 0.1
    k = torch.randn(B, H, K, D, dtype=torch.float16) * 0.1
    v = torch.randn(B, H, K, D, dtype=torch.float16) * 0.1
    mask = torch.zeros(1, 1, Q, K, dtype=torch.float16)

    # -------- 1. PyTorch parity --------
    with torch.no_grad():
        out_manual = manual_attention(q, k, v, mask)
        out_sdpa = sdpa_attention(q, k, v, mask)
    diff = (out_manual.to(torch.float32) - out_sdpa.to(torch.float32))
    c = float((out_manual.flatten().to(torch.float32)
               @ out_sdpa.flatten().to(torch.float32)) /
              (out_manual.norm() * out_sdpa.norm() + 1e-12))
    max_abs = float(diff.abs().max())
    print(f"\n[parity] manual vs SDPA (fp16 inputs):")
    print(f"  cos      = {c:.8f}")
    print(f"  max_abs  = {max_abs:.4e}")
    if c < 0.9999:
        print("  WARN: cos < 0.9999 — SDPA decomposition disagrees with manual")

    # -------- 2. MIL survival --------
    sample = (q, k, v, mask)
    names = ["q", "k", "v", "mask"]

    manual_ml = _convert(ManualBlock(), sample, names, label="manual")
    sdpa_ml = _convert(SDPABlock(), sample, names, label="sdpa")

    out_dir = os.path.join(ROOT, "..", "output", "sdpa_probe")
    os.makedirs(out_dir, exist_ok=True)
    manual_pkg = os.path.join(out_dir, "manual.mlpackage")
    sdpa_pkg = os.path.join(out_dir, "sdpa.mlpackage")
    import shutil
    for p, m in [(manual_pkg, manual_ml), (sdpa_pkg, sdpa_ml)]:
        if os.path.exists(p):
            shutil.rmtree(p)
        m.save(p)

    manual_count, manual_types = _census_ops(manual_pkg)
    sdpa_count, sdpa_types = _census_ops(sdpa_pkg)

    print(f"\n[graph] manual: {manual_count} SDPA-like ops")
    print(f"        types (top 15): {manual_types}")
    print(f"[graph] sdpa:   {sdpa_count} SDPA-like ops")
    print(f"        types (top 15): {sdpa_types}")

    if sdpa_count > 0:
        print("\n  ✅ SDPA op survived in the MIL — fusion is intact on ctools 9 / iOS 18")
    else:
        print("\n  ❌ SDPA op decomposed — ct.convert lowered it to matmul/softmax chain")
        print("     → no speed gain over manual attention for this scale=1.0 + ane_softmax setup")


if __name__ == "__main__":
    main()
