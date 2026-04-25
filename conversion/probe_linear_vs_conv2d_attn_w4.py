"""Probe v2: 5-layer Gemma-4-like with **attention** and **W4 palettize**.

Plan-3 follow-up #2 to probe_linear_vs_conv2d_5layer.py. Adds:
  - Real attention math per layer (Q/K/V/softmax/matmul over 32-token window).
    GQA via repeat_interleave (per task spec). RoPE skipped (scale=1.0).
  - W4 LUT palettize (nbits=4, per_grouped_channel, group_size=32) on fp16
    mlpackage, then re-audit.

Compares 2 wrapper variants × 2 quantization levels = 4 builds:
  - A_fp16 / A_w4  : Conv2d(1×1) wrapper (current ane_ops.Conv2dLinear path)
  - B_fp16 / B_w4  : nn.Linear native (cml9 PR #2577 target)

Per build captures: MIL op mix (conv vs linear vs constexpr_lut_to_dense),
ANE placement %, 20-iter median predict latency (Mac, CPU+ANE).

Verdict:
  - W4-after, B's ANE delta vs A ≥ 0pt → migration GO confirmed
  - W4-after, B drops ANE → linear+activation-quant not on ANE, HOLD
  - middle → more data needed
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
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)


HIDDEN = 1536
INTERMEDIATE = 6144
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
NUM_LAYERS = 5

# 32-token causal window: q_len=1, kv_len=32 → past cache length = 31
KV_LEN = 32
PAST_LEN = KV_LEN - 1
N_REP = NUM_Q_HEADS // NUM_KV_HEADS  # GQA replication factor (8)
SCALE = HEAD_DIM ** -0.5


# ---------- projection variants ------------------------------------------

class ConvProj(nn.Module):
    """Variant A: nn.Linear-by-Conv2d(1×1) with permute/unsqueeze wrapper."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = self.conv(x)
        return x.squeeze(2).permute(0, 2, 1)


class LinearProj(nn.Module):
    """Variant B: nn.Linear, no wrapper (cml9 native path)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------- 1-layer block: projections + attention + GELU FFN ------------

class AttnBlock(nn.Module):
    def __init__(self, proj_cls):
        super().__init__()
        self.q_proj = proj_cls(HIDDEN, NUM_Q_HEADS * HEAD_DIM)
        self.k_proj = proj_cls(HIDDEN, NUM_KV_HEADS * HEAD_DIM)
        self.v_proj = proj_cls(HIDDEN, NUM_KV_HEADS * HEAD_DIM)
        self.o_proj = proj_cls(NUM_Q_HEADS * HEAD_DIM, HIDDEN)
        self.gate_proj = proj_cls(HIDDEN, INTERMEDIATE)
        self.up_proj = proj_cls(HIDDEN, INTERMEDIATE)
        self.down_proj = proj_cls(INTERMEDIATE, HIDDEN)
        self.act = nn.GELU()

    def forward(
        self, x: torch.Tensor, k_past: torch.Tensor, v_past: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        # (1, 1, H) -> (1, n_heads, 1, head_dim)
        q = self.q_proj(x).view(1, 1, NUM_Q_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(1, 1, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(1, 1, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        # (1, kv, 31, D) ++ (1, kv, 1, D) -> (1, kv, 32, D)
        k_full = torch.cat([k_past, k], dim=2)
        v_full = torch.cat([v_past, v], dim=2)
        # GQA via repeat_interleave (per task spec): kv 1 -> q-heads 8
        k_full = k_full.repeat_interleave(N_REP, dim=1)
        v_full = v_full.repeat_interleave(N_REP, dim=1)
        # SDPA math (no causal mask needed, q_len=1)
        attn = torch.matmul(q, k_full.transpose(-1, -2)) * SCALE  # (1, 8, 1, 32)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_full)                          # (1, 8, 1, 256)
        out = out.permute(0, 2, 1, 3).reshape(1, 1, NUM_Q_HEADS * HEAD_DIM)
        out = self.o_proj(out)
        x = residual + out
        # FFN: GELU(gate) * up -> down
        ffn = self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        return x + ffn


class Stack5(nn.Module):
    """5 AttnBlocks in series. Forward signature is hardcoded to 11 tensors
    (x + 5 K-pasts + 5 V-pasts) so each becomes a separate ct.TensorType
    rather than a rank-5 stacked input."""

    def __init__(self, proj_cls):
        super().__init__()
        self.layers = nn.ModuleList([AttnBlock(proj_cls) for _ in range(NUM_LAYERS)])

    def forward(
        self,
        x: torch.Tensor,
        k0: torch.Tensor, k1: torch.Tensor, k2: torch.Tensor,
        k3: torch.Tensor, k4: torch.Tensor,
        v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor,
        v3: torch.Tensor, v4: torch.Tensor,
    ) -> torch.Tensor:
        ks = (k0, k1, k2, k3, k4)
        vs = (v0, v1, v2, v3, v4)
        for i, layer in enumerate(self.layers):
            x = layer(x, ks[i], vs[i])
        return x


# ---------- ct.convert + audit + latency ---------------------------------

def _make_sample_inputs():
    x = torch.zeros(1, 1, HIDDEN, dtype=torch.float32)
    kvs = [torch.zeros(1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM, dtype=torch.float32)
           for _ in range(2 * NUM_LAYERS)]
    return (x, *kvs)


def _ct_input_specs():
    return [
        ct.TensorType(name="x", shape=(1, 1, HIDDEN), dtype=np.float16),
        *[ct.TensorType(name=f"k{i}",
                        shape=(1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM),
                        dtype=np.float16)
          for i in range(NUM_LAYERS)],
        *[ct.TensorType(name=f"v{i}",
                        shape=(1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM),
                        dtype=np.float16)
          for i in range(NUM_LAYERS)],
    ]


def _predict_inputs():
    return {
        "x": np.zeros((1, 1, HIDDEN), dtype=np.float16),
        **{f"k{i}": np.zeros((1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM), dtype=np.float16)
           for i in range(NUM_LAYERS)},
        **{f"v{i}": np.zeros((1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM), dtype=np.float16)
           for i in range(NUM_LAYERS)},
    }


def convert_one(model: nn.Module, sample, name: str, out_dir: Path
                ) -> tuple[ct.models.MLModel, Path]:
    print(f"\n=== {name}: trace + ct.convert ===")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, strict=False)
    print(f"  traced in {time.time() - t0:.1f}s")

    t0 = time.time()
    ct_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=_ct_input_specs(),
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
    reloaded = ct.models.MLModel(
        str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    op_mix: Counter[str] = Counter()
    dev_counts: Counter[str] = Counter()
    non_ane: list[str] = []
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
                non_ane.append(f"{op.operator_name} -> {d}")
    total = sum(dev_counts.values())
    compute = total - dev_counts.get("const", 0)
    ane = dev_counts.get("MLNeuralEngineComputeDevice", 0)
    pct = 100.0 * ane / compute if compute else 0.0
    return reloaded, op_mix, dev_counts, pct, compute, ane, non_ane


def measure_latency(out_path: Path, iters: int = 20, warmup: int = 3
                    ) -> tuple[float, list[float]]:
    runner = ct.models.MLModel(
        str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    inputs = _predict_inputs()
    for _ in range(warmup):
        runner.predict(inputs)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        runner.predict(inputs)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), times


def report(name: str, op_mix, dev_counts, pct, compute, ane, non_ane,
           median_ms, all_times):
    print(f"\n[{name}] MIL op mix ({sum(op_mix.values())} ops total):")
    for op_type, count in sorted(op_mix.items(), key=lambda kv: -kv[1]):
        print(f"    {op_type:32s} {count}")
    print(f"\n[{name}] device counts: {dict(dev_counts)}")
    print(f"[{name}] ANE placement: {ane}/{compute} ({pct:.1f}%)")
    if non_ane:
        print(f"[{name}] non-ANE ops:")
        for line in non_ane[:30]:
            print(f"    {line}")
        if len(non_ane) > 30:
            print(f"    … ({len(non_ane) - 30} more)")
    p90 = sorted(all_times)[int(0.9 * len(all_times))]
    print(f"\n[{name}] latency: median {median_ms:.3f} ms over {len(all_times)} iter "
          f"(min {min(all_times):.3f}, max {max(all_times):.3f}, p90 {p90:.3f})")


def palettize(fp16_path: Path, w4_path: Path) -> Path:
    """W4 LUT palettize: nbits=4, per_grouped_channel, group_size=32."""
    print(f"\n--- palettize {fp16_path.name} -> {w4_path.name} (W4 LUT g=32) ---")
    m = ct.models.MLModel(str(fp16_path))
    op_cfg = OpPalettizerConfig(
        mode="kmeans",
        nbits=4,
        granularity="per_grouped_channel",
        group_size=32,
    )
    opt_cfg = OptimizationConfig(global_config=op_cfg)
    t0 = time.time()
    m_w4 = palettize_weights(m, opt_cfg)
    print(f"  palettized in {time.time() - t0:.1f}s")
    m_w4.save(str(w4_path))
    src_mb = sum(f.stat().st_size for f in fp16_path.rglob('*') if f.is_file()) / 1e6
    dst_mb = sum(f.stat().st_size for f in w4_path.rglob('*') if f.is_file()) / 1e6
    print(f"  bundle: {src_mb:.0f} MB (fp16) -> {dst_mb:.0f} MB (W4) "
          f"[{100 * dst_mb / src_mb:.1f}%]")
    return w4_path


# ---------- top-level orchestration --------------------------------------

def run_build(name: str, proj_cls, out_dir: Path) -> dict:
    sample = _make_sample_inputs()
    stack = Stack5(proj_cls).eval()
    n_params = sum(p.numel() for p in stack.parameters())
    print(f"\n{'=' * 72}\n  {name} — building Stack5 with attention\n{'=' * 72}")
    print(f"  params: {n_params / 1e6:.1f} M")
    _, fp16_path = convert_one(stack, sample, name, out_dir)
    return measure_and_report(name, fp16_path)


def measure_and_report(name: str, path: Path) -> dict:
    print(f"\n[{name}] audit (compile + MLComputePlan) …")
    _, op_mix, dev_counts, pct, compute, ane, non_ane = audit(path)
    median_ms, all_times = measure_latency(path)
    report(name, op_mix, dev_counts, pct, compute, ane, non_ane,
           median_ms, all_times)
    return {
        "name": name,
        "path": str(path),
        "op_mix": dict(op_mix),
        "device_counts": dict(dev_counts),
        "ane_pct": pct,
        "ane_count": ane,
        "compute_count": compute,
        "non_ane_ops": non_ane,
        "latency_ms": median_ms,
        "all_latencies": all_times,
    }


def main():
    print(f"coremltools {ct.__version__}, torch {torch.__version__}")
    print(f"config: hidden={HIDDEN}, intermediate={INTERMEDIATE}, "
          f"q_heads={NUM_Q_HEADS}, kv_heads={NUM_KV_HEADS}, "
          f"head_dim={HEAD_DIM}, layers={NUM_LAYERS}, "
          f"kv_window={KV_LEN}")

    out_dir = Path(tempfile.mkdtemp(prefix="probe5_attn_w4_"))
    print(f"out_dir: {out_dir}")

    results: dict[str, dict] = {}

    # ---- fp16 builds ----
    for name, cls in [("A_fp16", ConvProj), ("B_fp16", LinearProj)]:
        try:
            results[name] = run_build(name, cls, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{name}] ERROR: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
            results[name] = {"name": name, "error": str(exc)}

    # ---- W4 palettize ----
    for fp16_name, w4_name in [("A_fp16", "A_w4"), ("B_fp16", "B_w4")]:
        if "error" in results.get(fp16_name, {}):
            print(f"\nskipping {w4_name} — {fp16_name} errored")
            continue
        fp16_path = Path(results[fp16_name]["path"])
        w4_path = out_dir / f"{w4_name}.mlpackage"
        try:
            palettize(fp16_path, w4_path)
            results[w4_name] = measure_and_report(w4_name, w4_path)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{w4_name}] ERROR: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
            results[w4_name] = {"name": w4_name, "error": str(exc)}

    # ---- 4-way summary table ----
    print(f"\n{'=' * 80}\n  SUMMARY — 5-layer + attention, hidden={HIDDEN}\n{'=' * 80}")
    order = ["A_fp16", "B_fp16", "A_w4", "B_w4"]
    header = f"  {'metric':<22}"
    for k in order:
        header += f" {k:>14}"
    print(header)
    print("  " + "-" * 22 + (" " + "-" * 14) * 4)
    for metric_name, getter in [
        ("ANE %",      lambda r: f"{r['ane_pct']:.1f}%" if "error" not in r else "ERR"),
        ("ANE/compute", lambda r: f"{r['ane_count']}/{r['compute_count']}"
                        if "error" not in r else "ERR"),
        ("MIL ops total", lambda r: str(sum(r['op_mix'].values()))
                          if "error" not in r else "ERR"),
        ("latency (ms)", lambda r: f"{r['latency_ms']:.3f}"
                         if "error" not in r else "ERR"),
    ]:
        row = f"  {metric_name:<22}"
        for k in order:
            r = results.get(k, {})
            row += f" {getter(r):>14}" if r else f" {'-':>14}"
        print(row)

    # Per-build op breakdown for the relevant ops
    print("\n  [op-mix per build]")
    interesting = ("ios18.linear", "ios18.conv", "ios18.matmul",
                   "ios18.softmax", "ios18.transpose", "ios18.reshape",
                   "ios18.tile", "ios18.constexpr_lut_to_dense",
                   "ios18.constexpr_blockwise_shift_scale")
    print("  " + " " * 36 + "".join(f"{k:>10}" for k in order))
    for op in interesting:
        row = f"  {op:36s}"
        any_nonzero = False
        for k in order:
            r = results.get(k, {})
            n = r.get("op_mix", {}).get(op, 0) if "error" not in r else "-"
            if isinstance(n, int) and n > 0:
                any_nonzero = True
            row += f"{n:>10}"
        if any_nonzero:
            print(row)

    # ----- verdict -----
    print(f"\n{'=' * 80}\n  VERDICT\n{'=' * 80}")
    a16 = results.get("A_fp16", {})
    b16 = results.get("B_fp16", {})
    aw4 = results.get("A_w4", {})
    bw4 = results.get("B_w4", {})
    if any("error" in r for r in (a16, b16, aw4, bw4)):
        print("  one or more variants errored — see traceback(s) above")
        return

    fp16_delta = b16["ane_pct"] - a16["ane_pct"]
    w4_delta = bw4["ane_pct"] - aw4["ane_pct"]
    fp16_lat_ratio = b16["latency_ms"] / a16["latency_ms"]
    w4_lat_ratio = bw4["latency_ms"] / aw4["latency_ms"]
    print(f"  fp16 ANE delta (B - A):  {fp16_delta:+.2f} pt   "
          f"latency B/A: {fp16_lat_ratio:.3f}x")
    print(f"  W4   ANE delta (B - A):  {w4_delta:+.2f} pt   "
          f"latency B/A: {w4_lat_ratio:.3f}x")

    if w4_delta >= 0:
        verdict = "GO — migration confirmed (B keeps/improves ANE under W4)"
    elif w4_delta <= -5.0:
        verdict = ("HOLD — W4 path drops B below A on ANE; "
                   "linear+activation-quant likely not on ANE")
    else:
        verdict = (f"INTERMEDIATE — W4 ANE delta {w4_delta:+.1f}pt; "
                   "more data needed (try other granularity / nbits)")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    sys.exit(main() or 0)
