#!/usr/bin/env python3
"""Runtime helpers for Cascading KV Cache (Approach C).

The ANE-compiled model (see conversion/models/gemma4_swa_cascading.py) takes
`gather_idx`, `cos_p`, `sin_p` as inputs instead of computing them inline from
a Python `position` int. This file is the reference implementation of the
caller-side pre-computation. The Swift runtime must port this behavior
verbatim (it is a few arithmetic operations; not model code).

Additionally provides:
  - A CLI to precompute gather indices for a set of positions (useful for
    validation / golden-data generation).
  - A `CascadingKVRuntime` class grouping position→tensors conversion that
    unit tests can exercise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Allow running as a script from repo root
sys.path.append(str(Path(__file__).resolve().parent))
from models.gemma4_swa_cascading import CascadingConfig, build_gather_indices


class CascadingKVRuntime:
    """Pre-compute `gather_idx`, `cos_p`, `sin_p` for each decode step.

    Usage:
      rt = CascadingKVRuntime(cfg=CascadingConfig(), head_dim=256, rope_theta=10000.0)
      rt.precompute_rope_table(max_seq=8192)
      for step in range(N):
          gi, cos_p, sin_p = rt.step_inputs(position=step)
          # feed gi/cos_p/sin_p into the ANE mlpackage alongside hidden/K/V
    """

    def __init__(self, cfg: CascadingConfig, head_dim: int, rope_theta: float = 10000.0,
                 dtype: torch.dtype = torch.float16):
        self.cfg = cfg
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.dtype = dtype
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def precompute_rope_table(self, max_seq: int):
        """Build a `(max_seq, head_dim//2)` cos/sin lookup table once."""
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        pos = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        self._cos = freqs.cos().to(self.dtype)
        self._sin = freqs.sin().to(self.dtype)

    def step_inputs(self, position: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(gather_idx, cos_p, sin_p)` tensors for the given decode position.

        Shapes:
          gather_idx: (total_slots,) int32
          cos_p:      (1, head_dim/2) dtype
          sin_p:      (1, head_dim/2) dtype
        """
        if self._cos is None:
            raise RuntimeError("call precompute_rope_table(max_seq) first")
        gi = build_gather_indices(position, self.cfg).to(torch.int32)
        cos_p = self._cos[position:position + 1]
        sin_p = self._sin[position:position + 1]
        return gi, cos_p, sin_p

    def batch_precompute(self, positions: list[int]):
        """Precompute inputs for a batch of positions (e.g., for a prefill sweep
        or for golden-data generation). Returns a dict of stacked tensors."""
        if self._cos is None:
            raise RuntimeError("call precompute_rope_table(max_seq) first")
        gis, coss, sins = [], [], []
        for p in positions:
            gi, cp, sp = self.step_inputs(p)
            gis.append(gi); coss.append(cp); sins.append(sp)
        return {
            "gather_idx": torch.stack(gis, dim=0),   # (B, total_slots)
            "cos_p":      torch.stack(coss, dim=0),  # (B, 1, head_dim/2)
            "sin_p":      torch.stack(sins, dim=0),  # (B, 1, head_dim/2)
        }


# ── Swift port reference (pseudocode) ──────────────────────────────────────
#
#   let rt = CascadingKVRuntime(cfg: .default, headDim: 256, ropeTheta: 10000.0)
#   rt.precomputeRopeTable(maxSeq: 8192)
#
#   while decoding {
#     let (gi, cosP, sinP) = rt.stepInputs(position: currentPos)
#     let inputs: [String: Any] = [
#       "hidden_states": h,
#       "k_cache": kCache,
#       "v_cache": vCache,
#       "gather_idx": gi,      // MLMultiArray (Int32, shape [2564])
#       "cos_p": cosP,         // MLMultiArray (Float16, shape [1, 128])
#       "sin_p": sinP,
#     ]
#     let out = try await cascadingChunk.prediction(from: inputs)
#     ...
#   }
#
# Per-step cost of runtime pre-compute: ~2564 integer adds + clamps
# (≈ a few μs). Completely dominated by mlpackage prediction latency.


# ── CLI: dump golden tensors for validation ────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=str, default="0,10,100,512,1024,2048,4096,7000,8000",
                    help="comma-separated decode positions")
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    ap.add_argument("--output", type=str, default="./cascading_golden.pt")
    ap.add_argument("--max-seq", type=int, default=8192)
    args = ap.parse_args()

    cfg = CascadingConfig()
    print(cfg.describe())
    rt = CascadingKVRuntime(cfg=cfg, head_dim=args.head_dim, rope_theta=args.rope_theta)
    rt.precompute_rope_table(max_seq=args.max_seq)

    positions = [int(p) for p in args.positions.split(",")]
    out: dict = {}
    for p in positions:
        gi, cp, sp = rt.step_inputs(p)
        print(f"  position={p:5d}: gather_idx[{gi.shape[0]}] min={gi.min().item()} "
              f"max={gi.max().item()} cos_p[{cp.shape}] sin_p[{sp.shape}]")
        out[f"pos_{p}"] = {"gather_idx": gi, "cos_p": cp, "sin_p": sp}

    torch.save(out, args.output)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
