#!/usr/bin/env python3
"""Phase B alternate design — entire MoE layer as ONE mlpackage with gather.

Per-expert dispatch (phase_b_single_expert_mlpackage.py) showed 0.35 ms
per call on Mac ANE — too slow for 144 dispatches/token. This script
tests the alternative: pack all 60 experts as constants in a single
mlpackage, use gather at runtime to select 4, do the SwiGLU on the
gathered weights, weighted-sum the outputs.

Decision: if this brings the per-token tok/s above the per-expert
result, scale up to a chunked design (6 layers per chunk) for Phase C.

Inputs to the mlpackage:
  x_bc1t       : (1, hidden, 1, 1)  fp16  — hidden state
  topk_idx     : (4,)               int32 — selected expert indices
  topk_weights : (4,)               fp16  — routing weights for those experts

Output:
  y_bc1t       : (1, hidden, 1, 1)  fp16  — weighted sum of expert outputs

Constants:
  gate_proj_all : (60, inter, hidden)  fp16  = 60 * 1408 * 2048 * 2 = 346 MB
  up_proj_all   : (60, inter, hidden)  fp16  = 346 MB
  down_proj_all : (60, hidden, inter)  fp16  = 346 MB
  Total: ~1 GB per layer's mlpackage at fp16. INT4 palettized → ~125 MB.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import MODEL_DTYPE


HIDDEN = 2048
EXPERT_INTER = 1408
NUM_EXPERTS = 60
TOP_K = 4


class QwenMoeLayerGather(nn.Module):
    """One full layer's expert dispatch via gather. Excludes attention,
    norm, shared expert — focus on the routed-expert hot path."""

    def __init__(self,
                 hidden: int = HIDDEN,
                 inter: int = EXPERT_INTER,
                 num_experts: int = NUM_EXPERTS):
        super().__init__()
        # Store as (num_experts, out, in). Initialized small to avoid blowing up.
        self.gate_proj_all = nn.Parameter(
            (torch.randn(num_experts, inter, hidden) * 0.02).to(MODEL_DTYPE))
        self.up_proj_all = nn.Parameter(
            (torch.randn(num_experts, inter, hidden) * 0.02).to(MODEL_DTYPE))
        self.down_proj_all = nn.Parameter(
            (torch.randn(num_experts, hidden, inter) * 0.02).to(MODEL_DTYPE))

    def forward(self,
                x_bc1t: torch.Tensor,
                topk_idx: torch.Tensor,
                topk_weights: torch.Tensor) -> torch.Tensor:
        """x_bc1t: (1, hidden, 1, 1)
        topk_idx: (top_k,) int
        topk_weights: (top_k,) fp16"""
        # Gather selected experts
        # (top_k, inter, hidden)
        gate_sel = self.gate_proj_all.index_select(0, topk_idx)
        up_sel = self.up_proj_all.index_select(0, topk_idx)
        # (top_k, hidden, inter)
        down_sel = self.down_proj_all.index_select(0, topk_idx)

        # Flatten x to (1, hidden), then do batched matmul per selected
        # expert.
        x_flat = x_bc1t.reshape(1, HIDDEN)  # (1, hidden)
        # Use einsum: for each expert e, compute (1, inter) = x @ gate[e].T
        # gate_sel: (top_k, inter, hidden), x_flat: (1, hidden)
        # gate_out: (top_k, 1, inter)
        gate_out = torch.einsum("ki h, b h -> k b i", gate_sel, x_flat)
        up_out = torch.einsum("ki h, b h -> k b i", up_sel, x_flat)
        intermediate = F.silu(gate_out) * up_out  # (top_k, 1, inter)
        # down: (top_k, 1, hidden) = (top_k, 1, inter) @ down_sel.T per expert
        # down_sel: (top_k, hidden, inter)
        down_out = torch.einsum("k b i, kh i -> k b h",
                                intermediate, down_sel)
        # Weighted sum: weights (top_k,) broadcast over (top_k, 1, hidden)
        weighted = (down_out * topk_weights.view(-1, 1, 1)).sum(dim=0)
        # Back to (1, hidden, 1, 1)
        return weighted.view(1, HIDDEN, 1, 1)


def build_and_convert(out_path: str) -> str:
    m = QwenMoeLayerGather().eval().to(MODEL_DTYPE)
    print(f"[build] module size: "
          f"{sum(p.numel() for p in m.parameters())/1e6:.0f}M params")

    x_ex = torch.zeros(1, HIDDEN, 1, 1, dtype=MODEL_DTYPE)
    idx_ex = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    w_ex = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=MODEL_DTYPE)

    print(f"[trace] tracing...")
    traced = torch.jit.trace(m, (x_ex, idx_ex, w_ex))

    print(f"[convert] coremltools convert -> {out_path}")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(name="x_bc1t",
                          shape=(1, HIDDEN, 1, 1), dtype=np.float16),
            ct.TensorType(name="topk_idx",
                          shape=(TOP_K,), dtype=np.int32),
            ct.TensorType(name="topk_weights",
                          shape=(TOP_K,), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="y_bc1t", dtype=np.float16)],
    )
    mlmodel.save(out_path)
    print(f"[convert] done in {time.time()-t0:.1f}s")
    return out_path


def time_dispatch(mlpkg: str, n_runs: int = 100,
                  compute_unit: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE,
                  vary_idx: bool = True) -> dict:
    print(f"[time] loading {mlpkg} on {compute_unit}")
    model = ct.models.MLModel(mlpkg, compute_units=compute_unit)
    # Warm-up
    x = np.zeros((1, HIDDEN, 1, 1), dtype=np.float16)
    idx = np.array([0, 1, 2, 3], dtype=np.int32)
    w = np.full((TOP_K,), 0.25, dtype=np.float16)
    for _ in range(5):
        _ = model.predict({"x_bc1t": x, "topk_idx": idx, "topk_weights": w})
    rng = np.random.default_rng(0)
    times = []
    for _ in range(n_runs):
        x = rng.standard_normal((1, HIDDEN, 1, 1)).astype(np.float16)
        if vary_idx:
            idx = rng.choice(NUM_EXPERTS, TOP_K, replace=False).astype(np.int32)
        t0 = time.perf_counter()
        _ = model.predict({"x_bc1t": x, "topk_idx": idx, "topk_weights": w})
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "n": len(times),
        "min_ms": times[0],
        "p10_ms": times[len(times)//10],
        "median_ms": times[len(times)//2],
        "p90_ms": times[int(len(times)*0.9)],
        "p99_ms": times[int(len(times)*0.99)],
        "max_ms": times[-1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/qwen_moe_layer_gather.mlpackage")
    p.add_argument("--n-runs", type=int, default=200)
    p.add_argument("--report", default="/tmp/phase_b_layer_gather_report.json")
    args = p.parse_args()

    print(f"=== Phase B alternate — layer-gather MoE ===")
    build_and_convert(args.out)

    print(f"\n=== Dispatch latency (n={args.n_runs}, varied indices) ===")
    results = {}
    for name, unit in [
        ("CPU_AND_NE (ANE)", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU (GPU)", ct.ComputeUnit.CPU_AND_GPU),
        ("ALL (auto)", ct.ComputeUnit.ALL),
    ]:
        try:
            stats = time_dispatch(args.out, n_runs=args.n_runs,
                                  compute_unit=unit)
            results[name] = stats
            print(f"{name:>22}: min {stats['min_ms']:.2f}ms  "
                  f"p10 {stats['p10_ms']:.2f}  "
                  f"med {stats['median_ms']:.2f}  "
                  f"p90 {stats['p90_ms']:.2f}")
        except Exception as e:
            print(f"{name:>22}: ERROR {e}")
            results[name] = {"error": str(e)}

    # Extrapolation: per-layer = 1 routed batched-gather + 1 shared + 1 attn + 1 router
    # = 4 dispatches per layer × 24 layers = 96 dispatches per token
    print(f"\n=== Full-decode extrapolation ===")
    dispatch_per_token = 24 * 4
    print(f"Total dispatches per token: {dispatch_per_token} "
          f"(this layer batched-gather = 1 of the 4 per-layer dispatches)")
    print(f"NOTE: assuming shared + attn + router cost the SAME as one "
          f"layer-gather call (optimistic). Realistic = each is its own "
          f"latency; we only measured the batched-gather call.")
    for name, stats in results.items():
        if "median_ms" in stats:
            med = stats["median_ms"]
            # All 4 dispatches at this latency (optimistic)
            total_ms = dispatch_per_token * med
            tok_per_s = 1000.0 / total_ms
            print(f"{name:>22}: {med:.2f}ms × {dispatch_per_token} = "
                  f"{total_ms:.0f}ms → {tok_per_s:.1f} tok/s (optimistic)")

    # Gate decision
    print(f"\n=== Phase B-alt gate ===")
    GATE_DISPATCH_MS = 0.40
    best = min(s.get("median_ms", 999) for s in results.values() if "median_ms" in s)
    if best <= GATE_DISPATCH_MS:
        print(f"PASS — best median {best:.2f}ms <= gate {GATE_DISPATCH_MS}ms")
    else:
        print(f"FAIL — best median {best:.2f}ms > gate {GATE_DISPATCH_MS}ms")

    with open(args.report, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
