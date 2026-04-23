#!/usr/bin/env python3
"""Mac spike: can we ship multiple prefill-length variants of one chunk as
a single multifunction mlpackage with weights shared by hash dedup?

This doesn't touch Gemma 4; it converts a small stand-in module at two
different input sequence lengths (N=64, N=512), palettizes both to int4
per_grouped_channel (same config as production), then merges via
coremltools.utils.save_multifunction. Measures:

  1. Individual variant sizes
  2. Merged size (should be ≈ max(variants) + small graph delta, NOT sum)
  3. Whether weights/ directory inside the merged .mlpackage is deduped

Run: python conversion/spikes/multifunction_prefill_spike.py
"""
from __future__ import annotations
import os
import sys
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct
MultiFunctionDescriptor = ct.utils.MultiFunctionDescriptor
save_multifunction = ct.utils.save_multifunction

OUT = Path("/tmp/mfn_spike")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 1536   # Gemma 4 E2B hidden dim
N_LAYERS = 7    # enough weights to make dedup meaningful (~50MB int4)


class PrefillStandIn(nn.Module):
    """Stack of Linear ops — mimics a decoder chunk with shape-parametric compute.
    Identical weights across variants; only the sequence dimension N changes."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(N_LAYERS)
        ])

    def forward(self, x):  # x: (1, N, HIDDEN)
        for layer in self.layers:
            x = torch.nn.functional.silu(layer(x)) + x
        return x


def build_variant(model: nn.Module, N: int, name: str) -> Path:
    """Convert the same torch weights to an mlpackage at seq length N."""
    print(f"\n[variant {name}] tracing at N={N}")
    sample = torch.randn(1, N, HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, check_trace=False)

    print(f"[variant {name}] converting")
    mlm = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, N, HIDDEN))],
        outputs=[ct.TensorType(name="y")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    print(f"[variant {name}] palletizing int4 per_grouped_channel group=32")
    cfg = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpPalettizerConfig(
            nbits=4, granularity="per_grouped_channel", group_size=32))
    mlm = ct.optimize.coreml.palettize_weights(mlm, cfg)

    out_path = OUT / f"variant_{name}.mlpackage"
    if out_path.exists():
        shutil.rmtree(out_path)
    mlm.save(str(out_path))
    print(f"[variant {name}] saved → {out_path}")
    return out_path


def dir_size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def weights_size(p: Path) -> int:
    """Size of Data/com.apple.CoreML/weights/ directory inside mlpackage."""
    wdir = p / "Data" / "com.apple.CoreML" / "weights"
    if not wdir.exists():
        return 0
    return dir_size(wdir)


def main():
    torch.manual_seed(0)
    model = PrefillStandIn().eval()

    # Count raw trainable params
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {N_LAYERS} Linear layers, hidden={HIDDEN}, params={nparams/1e6:.1f}M")
    print(f"  fp16 size estimate: {nparams*2/1024/1024:.1f}MB")
    print(f"  int4 size estimate: {nparams*0.5/1024/1024:.1f}MB (palletize per_grouped_channel group=32)")

    v64 = build_variant(model, N=64, name="n64")
    v512 = build_variant(model, N=512, name="n512")

    s64 = dir_size(v64)
    s512 = dir_size(v512)
    w64 = weights_size(v64)
    w512 = weights_size(v512)

    print("\n=== individual variants ===")
    print(f"  N=64  total={human(s64)}  weights={human(w64)}")
    print(f"  N=512 total={human(s512)} weights={human(w512)}")

    # Merge
    print("\n[merge] save_multifunction dedup by weight hash")
    desc = MultiFunctionDescriptor()
    desc.add_function(str(v64), src_function_name="main", target_function_name="prefill_n64")
    desc.add_function(str(v512), src_function_name="main", target_function_name="prefill_n512")
    desc.default_function_name = "prefill_n512"

    merged_path = OUT / "merged.mlpackage"
    if merged_path.exists():
        shutil.rmtree(merged_path)
    save_multifunction(desc, str(merged_path))

    sM = dir_size(merged_path)
    wM = weights_size(merged_path)

    print("\n=== merged multifunction ===")
    print(f"  total={human(sM)}  weights={human(wM)}")

    # Verdict
    print("\n=== VERDICT ===")
    sum_pkg = s64 + s512
    print(f"  sum of variants : {human(sum_pkg)}")
    print(f"  merged          : {human(sM)}")
    print(f"  saved vs sum    : {human(sum_pkg - sM)}  ({(1 - sM/sum_pkg)*100:.1f}% smaller)")
    print(f"  merged / larger : {sM / max(s64, s512):.2f}x  (expect ≈1.0x if dedup works)")
    print(f"  merged weights  : {human(wM)} (vs n512 alone: {human(w512)})")

    if sM < (s64 + s512) * 0.7:
        print("\n  ✅ Dedup likely working — merged is much smaller than sum.")
    else:
        print("\n  ⚠ Dedup suspect — merged size close to sum. Investigate.")

    # Load test
    print("\n[load test]")
    try:
        m64 = ct.models.MLModel(str(merged_path), function_name="prefill_n64",
                                 compute_units=ct.ComputeUnit.CPU_ONLY)
        print(f"  loaded function prefill_n64 ✓")
        m512 = ct.models.MLModel(str(merged_path), function_name="prefill_n512",
                                  compute_units=ct.ComputeUnit.CPU_ONLY)
        print(f"  loaded function prefill_n512 ✓")
    except Exception as e:
        print(f"  ⚠ load failed: {e}")


if __name__ == "__main__":
    main()
