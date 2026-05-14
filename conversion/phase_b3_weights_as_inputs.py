#!/usr/bin/env python3
"""Phase B-3 — weights-as-inputs MoE design.

Different from Phase B-2 (all 60 experts as in-graph constants + gather):
here Swift gathers the 4 selected expert weights AHEAD of the call,
and passes them as INPUTS to the mlpackage. ANE only sees the 4 it
needs to compute.

Per-call inputs:
  x_bc1t       : (1, hidden, 1, 1) fp16
  gate_w_sel   : (4, inter, hidden) fp16  — pre-gathered gate_proj weights
  up_w_sel     : (4, inter, hidden) fp16
  down_w_sel   : (4, hidden, inter) fp16
  weights_topk : (4,) fp16                — routing weights

Constants: NONE in the mlpackage (other than the empty SwiGLU graph itself).

Output:
  y_bc1t       : (1, hidden, 1, 1) fp16   — weighted sum of expert outputs

Transfer cost analysis:
  Per call: 3 weight tensors × 4 experts × 1408 × 2048 × 2 bytes = 70 MB
  Per-token total (24 layers): 70 × 24 = 1.68 GB
  iPhone 70 GB/s ceiling: 24 ms just for weight transfer
  → token ceiling ~42 tok/s before any compute

This is a HARD upper bound (the bandwidth still has to flow). For this
design to beat current Gemma 4 (35 tok/s) we'd need compute to be
hidden behind transfer (impossible on serial ANE) OR we'd need a
training-time INT4 quantization of the weights that's accepted as
input.

Test: even at fp16, can we hit the 5ms-per-layer-call upper bound that
gives 24 × 5 = 120 ms = 8 tok/s? If yes, even fp16-input design is
useful with INT4 future variant. If no (eg 50 ms/call), nothing
salvages it.
"""
from __future__ import annotations
import argparse
import json
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
TOP_K = 4


class WeightsAsInputsExpert(nn.Module):
    """Forward with weights passed as inputs, no constants."""
    def __init__(self): super().__init__()

    def forward(self,
                x_bc1t: torch.Tensor,
                gate_w_sel: torch.Tensor,
                up_w_sel: torch.Tensor,
                down_w_sel: torch.Tensor,
                weights_topk: torch.Tensor) -> torch.Tensor:
        # x_bc1t: (1, hidden, 1, 1)
        # gate_w_sel: (top_k, inter, hidden)
        # up_w_sel:   (top_k, inter, hidden)
        # down_w_sel: (top_k, hidden, inter)
        # weights_topk: (top_k,)
        x_flat = x_bc1t.reshape(1, HIDDEN)
        # gate_out: (top_k, 1, inter)
        gate_out = torch.einsum("kih,bh->kbi", gate_w_sel, x_flat)
        up_out = torch.einsum("kih,bh->kbi", up_w_sel, x_flat)
        intermediate = F.silu(gate_out) * up_out  # (top_k, 1, inter)
        # down_out: (top_k, 1, hidden)
        down_out = torch.einsum("kbi,khi->kbh", intermediate, down_w_sel)
        weighted = (down_out * weights_topk.view(-1, 1, 1)).sum(dim=0)
        return weighted.view(1, HIDDEN, 1, 1)


def build_and_convert(out_path: str) -> str:
    m = WeightsAsInputsExpert().eval().to(MODEL_DTYPE)
    x = torch.zeros(1, HIDDEN, 1, 1, dtype=MODEL_DTYPE)
    g = torch.zeros(TOP_K, EXPERT_INTER, HIDDEN, dtype=MODEL_DTYPE)
    u = torch.zeros(TOP_K, EXPERT_INTER, HIDDEN, dtype=MODEL_DTYPE)
    d = torch.zeros(TOP_K, HIDDEN, EXPERT_INTER, dtype=MODEL_DTYPE)
    w = torch.tensor([0.25] * TOP_K, dtype=MODEL_DTYPE)
    print(f"[trace] tracing...")
    traced = torch.jit.trace(m, (x, g, u, d, w))
    print(f"[convert] {out_path}")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(name="x_bc1t", shape=(1, HIDDEN, 1, 1), dtype=np.float16),
            ct.TensorType(name="gate_w_sel", shape=(TOP_K, EXPERT_INTER, HIDDEN),
                          dtype=np.float16),
            ct.TensorType(name="up_w_sel", shape=(TOP_K, EXPERT_INTER, HIDDEN),
                          dtype=np.float16),
            ct.TensorType(name="down_w_sel", shape=(TOP_K, HIDDEN, EXPERT_INTER),
                          dtype=np.float16),
            ct.TensorType(name="weights_topk", shape=(TOP_K,), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="y_bc1t", dtype=np.float16)],
    )
    mlmodel.save(out_path)
    print(f"[convert] done in {time.time()-t0:.1f}s")
    return out_path


def time_dispatch(mlpkg: str, n_runs: int = 100,
                  unit: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE) -> dict:
    model = ct.models.MLModel(mlpkg, compute_units=unit)
    rng = np.random.default_rng(0)
    # Pre-allocate input buffers
    x = np.zeros((1, HIDDEN, 1, 1), dtype=np.float16)
    g = rng.standard_normal((TOP_K, EXPERT_INTER, HIDDEN)).astype(np.float16)
    u = rng.standard_normal((TOP_K, EXPERT_INTER, HIDDEN)).astype(np.float16)
    d = rng.standard_normal((TOP_K, HIDDEN, EXPERT_INTER)).astype(np.float16)
    w = np.full((TOP_K,), 0.25, dtype=np.float16)
    feed = {"x_bc1t": x, "gate_w_sel": g, "up_w_sel": u,
            "down_w_sel": d, "weights_topk": w}
    # warm-up
    for _ in range(5):
        _ = model.predict(feed)
    # timed runs
    times = []
    for _ in range(n_runs):
        # Vary x but keep weights stable; the question is dispatch cost.
        # Re-randomising weights every call would inflate transfer time
        # measurements (but is realistic — Swift would gather fresh per
        # layer per token). Test both modes.
        feed["x_bc1t"] = rng.standard_normal((1, HIDDEN, 1, 1)).astype(np.float16)
        t0 = time.perf_counter()
        _ = model.predict(feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {"n": len(times),
            "min_ms": times[0],
            "p10_ms": times[len(times)//10],
            "median_ms": times[len(times)//2],
            "p90_ms": times[int(len(times)*0.9)],
            "p99_ms": times[int(len(times)*0.99)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/qwen_moe_weights_as_inputs.mlpackage")
    p.add_argument("--n-runs", type=int, default=100)
    p.add_argument("--report", default="/tmp/phase_b3_report.json")
    args = p.parse_args()

    print(f"=== Phase B-3 — weights-as-inputs MoE ===")
    build_and_convert(args.out)

    print(f"\n=== Dispatch latency (n={args.n_runs}) ===")
    results = {}
    for name, unit in [
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("ALL", ct.ComputeUnit.ALL),
    ]:
        try:
            stats = time_dispatch(args.out, n_runs=args.n_runs, unit=unit)
            results[name] = stats
            print(f"{name:>12}: min {stats['min_ms']:.2f}ms  "
                  f"med {stats['median_ms']:.2f}  "
                  f"p90 {stats['p90_ms']:.2f}")
        except Exception as e:
            print(f"{name:>12}: ERROR {e}")
            results[name] = {"error": str(e)}

    # Extrapolate
    print(f"\n=== Full-decode extrapolation ===")
    dispatch_per_token = 24 * 4  # routed-expert call + shared + attn + router per layer
    for name, stats in results.items():
        if "median_ms" in stats:
            ms = stats["median_ms"]
            print(f"{name:>12}: {ms:.2f}ms × 24 layers (routed only) = "
                  f"{ms*24:.0f}ms; if ANE handles others free → "
                  f"max {1000/(ms*24):.1f} tok/s (UPPER BOUND)")

    with open(args.report, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
