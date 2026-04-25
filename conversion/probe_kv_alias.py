#!/usr/bin/env python3
"""Probe: does Gemma 4 global-attn K_proj == V_proj at the weight level?

docs/ANE_ONLY_LEVERS.md §D asserts the global-attention design guarantees
K ≡ V within each full-attn layer (producers L14/L19/L24/L29/L34 in E2B).
If true at the weight level, the inference-side k_proj / v_proj can be
aliased to one Conv2d + one output tensor, halving kv14 memory and
dropping one matmul per global layer.

This probe loads the HF weights and compares k_proj.weight vs v_proj.weight
tensor-wise. If cos=1.0 and max_abs=0, the alias is safe. Any non-zero
max_abs means the weights are genuinely different and we can't alias at
the cache level (though post-norm K and V might still differ even if
weights matched — tested separately with real activations).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.flatten().to(torch.float32)
    b32 = b.flatten().to(torch.float32)
    n = float(a32.norm() * b32.norm() + 1e-12)
    return float((a32 @ b32) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--hf-dir", default=None)
    args = ap.parse_args()

    hf_dir = args.hf_dir or os.path.join(ROOT, "..", "output", args.model, "hf_model")
    print(f"Loading {args.model} from {hf_dir}...")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=2048)
    base.eval()

    cfg = base.config
    n = cfg.num_hidden_layers
    print(f"\nN={n}, producers=L{cfg.kv_sliding_producer}/L{cfg.kv_full_producer}")
    print(f"Full-attn layers: ", end="")
    full_layers = [i for i in range(n) if cfg.is_full_attention(i)]
    print(full_layers)
    print(f"Sliding layers: {[i for i in range(n) if not cfg.is_full_attention(i)]}")

    # Check both pre-projection and layer norms
    print("\n--- k_proj.weight vs v_proj.weight (full-attn layers only) ---")
    any_mismatch = False
    for li in full_layers:
        layer = base.layers[li]
        k = layer.self_attn["k_proj"]
        v = layer.self_attn["v_proj"]
        kw = k.weight.data
        vw = v.weight.data
        if kw.shape != vw.shape:
            print(f"  L{li:2d}  shape mismatch: k={tuple(kw.shape)} v={tuple(vw.shape)}")
            any_mismatch = True
            continue
        c = _cos(kw, vw)
        d = float((kw.to(torch.float32) - vw.to(torch.float32)).abs().max())
        tag = "OK " if c > 0.9999 and d < 1e-5 else ("~~ " if c > 0.99 else "XX ")
        print(f"  {tag}L{li:2d}  shape={tuple(kw.shape)}  cos={c:.6f}  max_abs={d:.4e}")
        if tag != "OK ":
            any_mismatch = True

    print("\n--- k_norm.weight vs v_proj post-norm? (check if Gemma uses q/k norm equivalently) ---")
    for li in full_layers[:3]:
        layer = base.layers[li]
        k_norm = layer.self_attn.get("k_norm")
        if k_norm is None:
            print(f"  L{li}: no k_norm")
            continue
        print(f"  L{li:2d}  k_norm weight shape={tuple(k_norm.weight.shape)}  "
              f"mean={float(k_norm.weight.mean()):.4f}  std={float(k_norm.weight.std()):.4f}")

    if any_mismatch:
        print("\n❌ K ≠ V at the weight level — within-layer alias does NOT apply")
        print("   (docs/ANE_ONLY_LEVERS.md §D claim would need re-examination)")
    else:
        print("\n✅ All full-attn layers have K ≡ V at the weight level — alias is safe")
        print("   Next step: modify converter to emit single kv14 output + consume once in chunks 3/4")


if __name__ == "__main__":
    main()
