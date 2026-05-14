#!/usr/bin/env python3
"""Phase B — Build one Qwen MoE expert as CoreML mlpackage and measure
ANE dispatch latency.

The make-or-break question for the whole Qwen MoE pivot: can ANE
dispatch a small (~8.65M-param SwiGLU) computation in under ~0.2 ms?
If yes, 144 expert dispatches per decode step gives 50+ tok/s on iPhone.
If no, the whole port is dead.

This script:
  1. Builds a torch nn.Module for ONE Qwen MoE expert: SwiGLU 2048→1408→2048.
     Uses Conv2dLinear (kernel_size=1) per the existing ANE recipe in
     `conversion/ane_ops.py`.
  2. Optionally loads real weights from a safetensors shard (--weights-path)
     OR uses random fp16 weights (default; the architecture test is what
     matters for dispatch latency, not weight values).
  3. Converts to mlpackage with coremltools, ComputeUnit.ALL (lets ANE
     win the placement battle).
  4. Compiles to mlmodelc.
  5. Loads via coremltools and times N invocations with T=1 input.
  6. Reports min/median/p99 latency and the extrapolated full-model
     decode tok/s.

Usage:
  python conversion/phase_b_single_expert_mlpackage.py \\
    --out /tmp/qwen_moe_expert_l0e0.mlpackage \\
    --n-runs 100

Optional (when shard 1 download finished):
  python conversion/phase_b_single_expert_mlpackage.py \\
    --weights-path /tmp/qwen15-moe-chat/model-00001-of-00008.safetensors \\
    --layer 0 --expert 0 \\
    --out /tmp/qwen_moe_expert_l0e0.mlpackage
"""
from __future__ import annotations
import argparse
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
from ane_ops import Conv2dLinear, MODEL_DTYPE


# Qwen1.5-MoE-A2.7B constants (from /tmp/qwen15-moe-chat/config.json)
HIDDEN = 2048
EXPERT_INTER = 1408


class QwenMoeSingleExpert(nn.Module):
    """SwiGLU FFN: down_proj(silu(gate_proj(x)) * up_proj(x)).

    Uses Conv2dLinear for ANE-friendly conv1x1 dispatch.

    Input shape contract for ANE: (B=1, hidden, 1, T=1).
    Output shape: (B=1, hidden, 1, T=1).
    """

    def __init__(self, hidden: int = HIDDEN, inter: int = EXPERT_INTER):
        super().__init__()
        self.gate_proj = Conv2dLinear(hidden, inter, bias=False)
        self.up_proj = Conv2dLinear(hidden, inter, bias=False)
        self.down_proj = Conv2dLinear(inter, hidden, bias=False)

    def forward(self, x_bc1t: torch.Tensor) -> torch.Tensor:
        # x_bc1t: (1, hidden, 1, 1)
        gate = self.gate_proj.forward_conv(x_bc1t)  # (1, inter, 1, 1)
        up = self.up_proj.forward_conv(x_bc1t)
        intermediate = F.silu(gate) * up
        out = self.down_proj.forward_conv(intermediate)  # (1, hidden, 1, 1)
        return out


def load_real_weights(path: str, layer: int, expert: int) -> dict:
    """Load gate/up/down for a specific expert from a safetensors shard.

    Falls back to random weights if file unavailable.
    """
    if not os.path.exists(path):
        print(f"[weights] {path} not found, falling back to random")
        return None
    from safetensors.torch import load_file
    print(f"[weights] loading {path}")
    sd = load_file(path)
    pfx = f"model.layers.{layer}.mlp.experts.{expert}"
    keys = [f"{pfx}.gate_proj.weight", f"{pfx}.up_proj.weight",
            f"{pfx}.down_proj.weight"]
    missing = [k for k in keys if k not in sd]
    if missing:
        print(f"[weights] missing keys: {missing}")
        return None
    return {
        "gate": sd[keys[0]].to(MODEL_DTYPE),
        "up": sd[keys[1]].to(MODEL_DTYPE),
        "down": sd[keys[2]].to(MODEL_DTYPE),
    }


def build_module(weights: dict | None) -> QwenMoeSingleExpert:
    m = QwenMoeSingleExpert()
    if weights is not None:
        # Conv2dLinear weights are stored as conv.weight shape
        # (out, in, 1, 1). Load Linear-style (out, in) and reshape.
        m.gate_proj.conv.weight.data = weights["gate"].unsqueeze(-1).unsqueeze(-1)
        m.up_proj.conv.weight.data = weights["up"].unsqueeze(-1).unsqueeze(-1)
        m.down_proj.conv.weight.data = weights["down"].unsqueeze(-1).unsqueeze(-1)
        print(f"[build] loaded real weights into module")
    else:
        # Random fp16
        for p in m.parameters():
            p.data = (torch.randn_like(p.data) * 0.02).to(MODEL_DTYPE)
        print(f"[build] using random fp16 weights")
    return m.eval().to(MODEL_DTYPE)


def to_mlpackage(module: QwenMoeSingleExpert, out_path: str,
                  compute_unit: ct.ComputeUnit = ct.ComputeUnit.ALL) -> str:
    """Trace + convert via coremltools."""
    example = torch.zeros(1, HIDDEN, 1, 1, dtype=MODEL_DTYPE)
    print(f"[trace] tracing with example shape {tuple(example.shape)}")
    traced = torch.jit.trace(module, example)

    print(f"[convert] coremltools convert -> {out_path}")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=compute_unit,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[ct.TensorType(name="x_bc1t",
                              shape=(1, HIDDEN, 1, 1), dtype=np.float16)],
        outputs=[ct.TensorType(name="y_bc1t", dtype=np.float16)],
    )
    mlmodel.save(out_path)
    print(f"[convert] done in {time.time()-t0:.1f}s")
    return out_path


def time_dispatch(mlpkg_path: str, n_runs: int = 100,
                  compute_unit: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE) -> dict:
    """Load mlpackage and time N predictions with random input."""
    print(f"[time] loading {mlpkg_path} on {compute_unit}")
    model = ct.models.MLModel(mlpkg_path, compute_units=compute_unit)
    # warm-up
    x = np.zeros((1, HIDDEN, 1, 1), dtype=np.float16)
    for _ in range(5):
        _ = model.predict({"x_bc1t": x})
    # timed runs
    times = []
    for i in range(n_runs):
        x = np.random.randn(1, HIDDEN, 1, 1).astype(np.float16)
        t0 = time.perf_counter()
        _ = model.predict({"x_bc1t": x})
        dt = time.perf_counter() - t0
        times.append(dt * 1000.0)  # ms
    times.sort()
    return {
        "n": len(times),
        "min_ms": times[0],
        "p10_ms": times[len(times) // 10],
        "median_ms": times[len(times) // 2],
        "p90_ms": times[int(len(times) * 0.9)],
        "p99_ms": times[int(len(times) * 0.99)],
        "max_ms": times[-1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/qwen_moe_expert_l0e0.mlpackage")
    p.add_argument("--weights-path", default=None,
                   help="optional safetensors shard with real weights")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--n-runs", type=int, default=200)
    p.add_argument("--n-experts-per-token", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=24,
                   help="for full-decode extrapolation")
    p.add_argument("--report", default="/tmp/phase_b_dispatch_report.json")
    args = p.parse_args()

    print(f"=== Phase B — single Qwen MoE expert ANE feasibility ===")

    # Load (or fake) weights
    weights = None
    if args.weights_path:
        weights = load_real_weights(args.weights_path, args.layer, args.expert)
    module = build_module(weights)

    # Convert
    to_mlpackage(module, args.out, compute_unit=ct.ComputeUnit.ALL)

    # Time on three compute placements for comparison
    print(f"\n=== Dispatch latency (n={args.n_runs}) ===")
    results = {}
    for name, unit in [
        ("CPU_AND_NE (ANE)", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU (Mac)", ct.ComputeUnit.CPU_AND_GPU),
        ("ALL (auto)", ct.ComputeUnit.ALL),
    ]:
        try:
            stats = time_dispatch(args.out, n_runs=args.n_runs, compute_unit=unit)
            results[name] = stats
            print(f"{name:>22}: min {stats['min_ms']:.2f}ms  "
                  f"p10 {stats['p10_ms']:.2f}  "
                  f"med {stats['median_ms']:.2f}  "
                  f"p90 {stats['p90_ms']:.2f}  "
                  f"p99 {stats['p99_ms']:.2f}")
        except Exception as e:
            print(f"{name:>22}: ERROR {e}")
            results[name] = {"error": str(e)}

    # Extrapolation: full decode = N_layers * (K_experts + 1 shared + 1 router) calls
    print(f"\n=== Full-decode extrapolation ===")
    print(f"Assumption: per layer = {args.n_experts_per_token} experts + 1 shared + 1 attn")
    dispatch_per_token = args.n_layers * (args.n_experts_per_token + 1 + 1)
    print(f"Total dispatches per token: {dispatch_per_token}")
    for name, stats in results.items():
        if "median_ms" in stats:
            med = stats["median_ms"]
            total_ms = dispatch_per_token * med
            tok_per_s = 1000.0 / total_ms
            print(f"{name:>22}: median {med:.2f}ms -> "
                  f"{total_ms:.0f}ms/tok -> {tok_per_s:.1f} tok/s")

    # Gate decision
    print(f"\n=== Phase B gate ===")
    GATE_DISPATCH_MS = 0.25
    best = min(s.get("median_ms", 999) for s in results.values() if "median_ms" in s)
    if best <= GATE_DISPATCH_MS:
        print(f"PASS — best median dispatch {best:.2f}ms <= gate {GATE_DISPATCH_MS}ms")
        print(f"      Full decode extrapolation: "
              f"{1000.0 / (dispatch_per_token * best):.1f} tok/s")
    else:
        print(f"FAIL — best median dispatch {best:.2f}ms > gate {GATE_DISPATCH_MS}ms")
        print(f"      Full decode at best dispatch would be only "
              f"{1000.0 / (dispatch_per_token * best):.1f} tok/s.")
        print(f"      Below current Gemma 4 baseline (~35 tok/s) -> abort Phase C")

    import json
    with open(args.report, "w") as f:
        json.dump({
            "args": vars(args),
            "dispatch_results": results,
            "extrapolation": {
                "dispatch_per_token": dispatch_per_token,
                "extrapolated_tps": {
                    name: 1000.0 / (dispatch_per_token * stats["median_ms"])
                    for name, stats in results.items() if "median_ms" in stats
                },
            },
        }, f, indent=2)
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
