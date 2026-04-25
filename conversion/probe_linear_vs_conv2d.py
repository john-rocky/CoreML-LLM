#!/usr/bin/env python3
"""coremltools 9 PR #2577 micro-PoC: does dropping the Conv2d-1×1 wrapper
help on ANE for Gemma 4-style projections, given that cml 9 now natively
supports activation-quant of `linear` ops?

Two minimal models with identical math:
  A) Conv2d(1×1) wrapped — current pipeline (requires permute/squeeze)
  B) nn.Linear direct — cml 9 native path

Both convert with iOS18 target, compute_precision=FLOAT16, CPU_AND_NE.
We compare:
  - MIL op mix (conv vs linear vs reshape/transpose/expand/squeeze)
  - ANE placement % (computePlan)
  - Predict latency on Mac (proxy only, real test is iPhone)

Caveats:
  * Mac timing != iPhone timing. This is a graph-shape and ANE-placement
    test only.
  * We don't add activation-quant in this PoC — that's a separate axis.
    If the linear-op path doesn't even reach high ANE placement, the
    cml 9 activation-quant feature on it doesn't matter for us.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

OUT = Path("/tmp/probe_linear_vs_conv2d")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 1536      # match Gemma 4 E2B
INTER  = 6144      # match Gemma 4 E2B FFN intermediate
SEQ    = 1         # decode shape

torch.manual_seed(0)
W_q  = torch.randn(HIDDEN, HIDDEN, dtype=torch.float16) * 0.02
W_o  = torch.randn(HIDDEN, HIDDEN, dtype=torch.float16) * 0.02
W_g  = torch.randn(INTER,  HIDDEN, dtype=torch.float16) * 0.02
W_u  = torch.randn(INTER,  HIDDEN, dtype=torch.float16) * 0.02
W_d  = torch.randn(HIDDEN, INTER,  dtype=torch.float16) * 0.02


class ConvBlock(nn.Module):
    """Mini transformer-ish block using Conv2d-1×1 (current pipeline pattern)."""

    def __init__(self):
        super().__init__()
        self.q = nn.Conv2d(HIDDEN, HIDDEN, 1, bias=False, dtype=torch.float16)
        self.o = nn.Conv2d(HIDDEN, HIDDEN, 1, bias=False, dtype=torch.float16)
        self.g = nn.Conv2d(HIDDEN, INTER,  1, bias=False, dtype=torch.float16)
        self.u = nn.Conv2d(HIDDEN, INTER,  1, bias=False, dtype=torch.float16)
        self.d = nn.Conv2d(INTER,  HIDDEN, 1, bias=False, dtype=torch.float16)
        with torch.no_grad():
            self.q.weight.copy_(W_q.unsqueeze(-1).unsqueeze(-1))
            self.o.weight.copy_(W_o.unsqueeze(-1).unsqueeze(-1))
            self.g.weight.copy_(W_g.unsqueeze(-1).unsqueeze(-1))
            self.u.weight.copy_(W_u.unsqueeze(-1).unsqueeze(-1))
            self.d.weight.copy_(W_d.unsqueeze(-1).unsqueeze(-1))

    def forward(self, x):
        # x: (1, 1, H)
        x_c = x.permute(0, 2, 1).unsqueeze(2)            # (1, H, 1, 1)
        q = self.q(x_c)
        o = self.o(q)
        g = nn.functional.gelu(self.g(o), approximate="tanh") * self.u(o)
        d = self.d(g)
        return d.squeeze(2).permute(0, 2, 1)             # back to (1, 1, H)


class LinearBlock(nn.Module):
    """Same math via nn.Linear directly (cml 9 native path)."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
        self.o = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
        self.g = nn.Linear(HIDDEN, INTER,  bias=False, dtype=torch.float16)
        self.u = nn.Linear(HIDDEN, INTER,  bias=False, dtype=torch.float16)
        self.d = nn.Linear(INTER,  HIDDEN, bias=False, dtype=torch.float16)
        with torch.no_grad():
            self.q.weight.copy_(W_q)
            self.o.weight.copy_(W_o)
            self.g.weight.copy_(W_g)
            self.u.weight.copy_(W_u)
            self.d.weight.copy_(W_d)

    def forward(self, x):
        q = self.q(x)
        o = self.o(q)
        g = nn.functional.gelu(self.g(o), approximate="tanh") * self.u(o)
        return self.d(g)


def _convert(model: nn.Module, label: str) -> str:
    sample = torch.zeros(1, 1, HIDDEN, dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(model.eval(), sample, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=sample.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    out = OUT / f"{label}.mlpackage"
    if out.exists():
        import shutil; shutil.rmtree(out)
    mlmodel.save(str(out))
    return str(out)


def _op_mix(pkg: str) -> Counter:
    m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_ONLY)
    counts = Counter()
    for _, fn in m.get_spec().mlProgram.functions.items():
        for _, blk in fn.block_specializations.items():
            for op in blk.operations:
                counts[op.type] += 1
    return counts


def _ane_placement(pkg: str) -> tuple[int, int, float]:
    """Match build_gemma4_e2b_stateful_chunks.py:_audit_ane exactly."""
    m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = m.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    dev = Counter()
    for fn in plan.model_structure.program.functions.values():
        for op in fn.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            d = ("const" if (a is None and op.operator_name == "const")
                 else (a.preferred_compute_device.__class__.__name__
                       if a else "unknown"))
            dev[d] += 1
    total = sum(dev.values())
    compute = total - dev.get("const", 0)
    ane = dev.get("MLNeuralEngineComputeDevice", 0)
    pct = 100.0 * ane / compute if compute else 0.0
    return ane, compute, pct


def _bench(pkg: str, iters: int = 20) -> tuple[float, float]:
    m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    x = np.zeros((1, 1, HIDDEN), dtype=np.float16)
    # warmup
    for _ in range(3):
        m.predict({"x": x})
    times = []
    for _ in range(iters):
        t = time.time()
        m.predict({"x": x})
        times.append((time.time() - t) * 1000)
    return float(np.median(times)), float(np.std(times))


def main():
    print(f"\n[probe] HIDDEN={HIDDEN}  INTER={INTER}")

    print("\n[A] converting Conv2d-1×1 wrapped block")
    pa = _convert(ConvBlock(), "A_conv2d")
    print("[B] converting nn.Linear direct block")
    pb = _convert(LinearBlock(), "B_linear")

    cnt_a = _op_mix(pa)
    cnt_b = _op_mix(pb)

    print("\n--- MIL op mix comparison ---")
    keys = sorted(set(cnt_a) | set(cnt_b))
    print(f"{'op_type':<30} {'A(conv)':>10} {'B(linear)':>10}")
    for k in keys:
        a, b = cnt_a.get(k, 0), cnt_b.get(k, 0)
        if a or b:
            print(f"  {k:<28} {a:>10} {b:>10}")

    a_ane, a_total, a_pct = _ane_placement(pa)
    b_ane, b_total, b_pct = _ane_placement(pb)
    print("\n--- ANE placement (CPU_AND_NE preferred device per op) ---")
    print(f"  A (conv2d):  {a_ane}/{a_total} = {a_pct:.1f}%")
    print(f"  B (linear):  {b_ane}/{b_total} = {b_pct:.1f}%")

    a_med, a_std = _bench(pa)
    b_med, b_std = _bench(pb)
    print("\n--- Mac latency (median over 20 iters, 3 warmup) ---")
    print(f"  A (conv2d):  {a_med:.2f} ± {a_std:.2f} ms")
    print(f"  B (linear):  {b_med:.2f} ± {b_std:.2f} ms")
    print(f"  delta:       {(b_med-a_med):+.2f} ms ({(b_med/a_med-1)*100:+.1f}%)")

    print("\n--- Verdict ---")
    if b_pct < 80 and a_pct >= 90:
        print(f"  → Conv2d-1×1 wrapper IS still needed: linear path drops to {b_pct:.0f}% ANE")
    elif b_pct >= a_pct - 2 and len(cnt_b) < len(cnt_a):
        print(f"  → nn.Linear is at least as good. Reshape ops trimmed.")
    else:
        print(f"  → ambiguous; need iPhone benchmark for final decision.")


if __name__ == "__main__":
    main()
