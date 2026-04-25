"""5-layer scale probe: nn.Conv2d(1×1) wrapper vs nn.Linear (cml9 native).

Plan 3 follow-up to the micro-PoC in docs. Builds two variants of a Gemma-4-like
decoder block (projections + GELU FFN, no attention compute) at the E2B-chunk_1
parameter scale and compares ANE placement / latency / MIL op mix.

Variant A — ConvProj: nn.Conv2d(in, out, 1, bias=False) with permute/unsqueeze
                       wrapper (current ane_ops.Conv2dLinear pipeline).
Variant B — LinearProj: nn.Linear(in, out, bias=False), no wrapper.

Hidden=1536, intermediate=6144, q-heads=8, kv-heads=1, head_dim=256, 5 layers.
Sample: (1, 1, 1536) fp16. ct.convert: iOS18 + fp16 + CPU_AND_NE.

Captures: MIL op mix (per variant), ANE placement %, 20-iter median predict
latency (warmup 3). Verdict: GO / HOLD / INTERMEDIATE.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

import coremltools as ct


HIDDEN = 1536
INTERMEDIATE = 6144
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
NUM_LAYERS = 5

Q_DIM = NUM_Q_HEADS * HEAD_DIM   # 2048
O_DIM = HIDDEN                    # 1536


# ---------- projection variants -------------------------------------------

class ConvProj(nn.Module):
    """nn.Linear-by-Conv2d(1x1), with permute/unsqueeze wrapper (variant A).

    Mirrors conversion/ane_ops.py::Conv2dLinear.forward:
        (B, S, in) -> permute(0,2,1).unsqueeze(2) -> (B, in, 1, S)
                  -> conv2d -> (B, out, 1, S)
                  -> squeeze(2).permute(0,2,1) -> (B, S, out)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = self.conv(x)
        return x.squeeze(2).permute(0, 2, 1)


class LinearProj(nn.Module):
    """nn.Linear, no wrapper (variant B — cml9 native path)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------- 1-layer Gemma-4-ish block (proj + GELU FFN, no attn compute) --

def make_block(proj_cls):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            # "attention" stand-in: q_proj then o_proj
            self.q_proj = proj_cls(HIDDEN, Q_DIM)
            self.o_proj = proj_cls(Q_DIM, O_DIM)
            # FFN
            self.gate_proj = proj_cls(HIDDEN, INTERMEDIATE)
            self.up_proj = proj_cls(HIDDEN, INTERMEDIATE)
            self.down_proj = proj_cls(INTERMEDIATE, HIDDEN)
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            attn_out = self.o_proj(self.q_proj(x))
            x = x + attn_out
            ffn_out = self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
            return x + ffn_out
    return Block


def make_stack(proj_cls, n_layers: int) -> nn.Sequential:
    Block = make_block(proj_cls)
    return nn.Sequential(*[Block() for _ in range(n_layers)])


# ---------- ct.convert + audit + latency ----------------------------------

def convert_one(model: nn.Module, sample: torch.Tensor, name: str,
                out_dir: Path) -> tuple[ct.models.MLModel, Path]:
    print(f"\n=== {name}: trace + ct.convert ===")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample)
    print(f"  traced in {time.time() - t0:.1f}s")

    t0 = time.time()
    ct_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="x", shape=sample.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  converted in {time.time() - t0:.1f}s")
    out_path = out_dir / f"{name}.mlpackage"
    ct_model.save(str(out_path))
    return ct_model, out_path


def audit(out_path: Path):
    """Compile + MLComputePlan walk. Returns (reloaded_model, op_mix,
    device_counts, ane_pct, compute_count, ane_count, non_ane_ops)."""
    reloaded = ct.models.MLModel(
        str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    op_mix: Counter[str] = Counter()
    dev_counts: Counter[str] = Counter()
    non_ane_ops: list[str] = []
    for fn in plan.model_structure.program.functions.values():
        for op in fn.block.operations:
            op_mix[op.operator_name] += 1
            assignment = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if assignment is None:
                d = "const" if op.operator_name == "const" else "unknown"
            else:
                d = assignment.preferred_compute_device.__class__.__name__
            dev_counts[d] += 1
            if d not in ("MLNeuralEngineComputeDevice", "const"):
                non_ane_ops.append(f"{op.operator_name} -> {d}")
    total = sum(dev_counts.values())
    compute = total - dev_counts.get("const", 0)
    ane = dev_counts.get("MLNeuralEngineComputeDevice", 0)
    pct = 100.0 * ane / compute if compute else 0.0
    return reloaded, op_mix, dev_counts, pct, compute, ane, non_ane_ops


def measure_latency(model: ct.models.MLModel, x_np: np.ndarray,
                    iters: int = 20, warmup: int = 3):
    for _ in range(warmup):
        model.predict({"x": x_np})
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict({"x": x_np})
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), times


def run_variant(name: str, proj_cls, sample: torch.Tensor, out_dir: Path):
    print(f"\n{'=' * 72}\n  {name} — building {NUM_LAYERS}-layer stack\n{'=' * 72}")
    stack = make_stack(proj_cls, NUM_LAYERS).eval()
    n_params = sum(p.numel() for p in stack.parameters())
    print(f"  params: {n_params / 1e6:.1f} M")

    ct_model, out_path = convert_one(stack, sample, name, out_dir)

    print(f"\n[{name}] audit (compile + MLComputePlan) …")
    _, op_mix, dev_counts, pct, compute, ane, non_ane = audit(out_path)

    print(f"\n[{name}] MIL op mix ({sum(op_mix.values())} ops total):")
    for op_type, count in sorted(op_mix.items(), key=lambda kv: -kv[1]):
        print(f"    {op_type:30s} {count}")

    print(f"\n[{name}] device counts: {dict(dev_counts)}")
    print(f"[{name}] ANE placement: {ane}/{compute} ({pct:.1f}%)")
    if non_ane:
        print(f"[{name}] non-ANE ops:")
        for line in non_ane[:30]:
            print(f"    {line}")
        if len(non_ane) > 30:
            print(f"    … ({len(non_ane) - 30} more)")

    # Re-load fresh for latency to drop any lingering compute-plan state.
    runner = ct.models.MLModel(
        str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    x_np = sample.numpy().astype(np.float16)
    median_ms, all_times = measure_latency(runner, x_np)
    print(f"\n[{name}] latency: median {median_ms:.3f} ms over 20 iter "
          f"(min {min(all_times):.3f}, max {max(all_times):.3f}, "
          f"p90 {sorted(all_times)[int(0.9 * len(all_times))]:.3f})")

    return {
        "name": name,
        "op_mix": dict(op_mix),
        "device_counts": dict(dev_counts),
        "ane_pct": pct,
        "ane_count": ane,
        "compute_count": compute,
        "non_ane_ops": non_ane,
        "latency_ms": median_ms,
        "all_latencies": all_times,
    }


# ---------- main ----------------------------------------------------------

def main():
    print(f"coremltools {ct.__version__}, torch {torch.__version__}")
    print(f"config: hidden={HIDDEN}, intermediate={INTERMEDIATE}, "
          f"q_heads={NUM_Q_HEADS}, kv_heads={NUM_KV_HEADS}, "
          f"head_dim={HEAD_DIM}, layers={NUM_LAYERS}")

    out_dir = Path(tempfile.mkdtemp(prefix="probe5_"))
    print(f"out_dir: {out_dir}")

    # Float32 sample for tracing; ct.TensorType declares fp16 for the converted
    # model's input boundary.
    sample = torch.zeros(1, 1, HIDDEN, dtype=torch.float32)

    results: dict = {}
    variants = [
        ("A_conv2d_wrapper", ConvProj),
        ("B_linear_native", LinearProj),
    ]
    for name, cls in variants:
        try:
            results[name] = run_variant(name, cls, sample, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{name}] ERROR: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
            results[name] = {"name": name, "error": str(exc)}

    # ----- summary table -----
    print(f"\n{'=' * 72}\n  SUMMARY — 5-layer Gemma-4-like, hidden={HIDDEN}, "
          f"int={INTERMEDIATE}\n{'=' * 72}")
    a = results.get("A_conv2d_wrapper", {})
    b = results.get("B_linear_native", {})
    if "error" in a or "error" in b:
        print("  one or both variants errored — see traceback above")
        return

    print(f"  {'metric':<22} {'A (Conv2d 1x1)':<22} {'B (nn.Linear)':<22}  delta")
    print(f"  {'-' * 22} {'-' * 22} {'-' * 22}  -----")
    print(f"  {'ANE placement %':<22} "
          f"{a['ane_pct']:>6.1f} %{'':<14} "
          f"{b['ane_pct']:>6.1f} %{'':<14}  "
          f"{b['ane_pct'] - a['ane_pct']:+.1f} pt")
    print(f"  {'ANE / compute':<22} "
          f"{a['ane_count']:>4} / {a['compute_count']:<14} "
          f"{b['ane_count']:>4} / {b['compute_count']:<14}")
    print(f"  {'MIL ops total':<22} "
          f"{sum(a['op_mix'].values()):>22} "
          f"{sum(b['op_mix'].values()):>22}  "
          f"{sum(b['op_mix'].values()) - sum(a['op_mix'].values()):+d}")
    print(f"  {'latency median (ms)':<22} "
          f"{a['latency_ms']:>22.3f} "
          f"{b['latency_ms']:>22.3f}  "
          f"{(b['latency_ms'] - a['latency_ms']) / a['latency_ms'] * 100:+.1f} %")

    # Op-mix delta (only differing ops)
    print("\n  [op-mix delta]  (positive = B has more)")
    all_ops = set(a["op_mix"]) | set(b["op_mix"])
    for op in sorted(all_ops):
        ca, cb = a["op_mix"].get(op, 0), b["op_mix"].get(op, 0)
        if ca != cb:
            print(f"      {op:30s} A={ca:<5} B={cb:<5} delta={cb - ca:+d}")

    # ----- verdict -----
    print(f"\n{'=' * 72}\n  VERDICT\n{'=' * 72}")
    delta_pp = b["ane_pct"] - a["ane_pct"]
    lat_ratio = b["latency_ms"] / a["latency_ms"]
    if delta_pp >= -2.0 and lat_ratio <= 1.0:
        verdict = "GO — migrate Conv2d-wrapper → nn.Linear (cml9 native)"
    elif delta_pp <= -5.0:
        verdict = "HOLD — Linear path drops ANE placement >5pt"
    else:
        verdict = (f"INTERMEDIATE — ANE delta {delta_pp:+.1f}pt, "
                   f"latency ratio {lat_ratio:.2f}x; need more data")
    print(f"  ANE placement delta (B - A):  {delta_pp:+.2f} pt")
    print(f"  latency ratio       (B / A):  {lat_ratio:.3f}x  "
          f"(<1 means B faster)")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    sys.exit(main() or 0)
