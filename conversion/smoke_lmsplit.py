#!/usr/bin/env python3
"""Smoke test SWAChunk4_LMSplit: instantiate and trace, compare against
SWAChunk4 baseline output to confirm the split is mathematically equivalent.
Mac-only; no CoreML conversion (that step is what build_gemma4_3way.py does).
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import MODEL_REGISTRY  # noqa: E402
from models.gemma4 import Gemma4Model  # noqa: E402
from models.gemma4_swa_chunks import (  # noqa: E402
    SWAChunk4, SWAChunk4_LMSplit, compute_chunk_boundaries,
)


def main():
    hf_dir = os.path.join(ROOT, "..", "output", "gemma4-e2b", "hf_model")
    print(f"[smoke] loading {hf_dir}  (ctx=512)")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=512).eval()
    cfg = base.config
    boundaries = compute_chunk_boundaries(cfg)
    c4_start, c4_end = boundaries[3]
    print(f"[smoke] chunk4 layers L{c4_start}-{c4_end-1}  vocab={cfg.vocab_size}")

    baseline = SWAChunk4(base, c4_start, c4_end).eval()
    splits = {n: SWAChunk4_LMSplit(base, c4_start, c4_end, n_splits=n).eval()
              for n in (2, 8, 16)}

    ctx = cfg.state_length
    W = cfg.sliding_window
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    nkv = cfg.num_key_value_heads
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim

    torch.manual_seed(0)
    sample = (
        torch.randn(1, 1, hidden, dtype=torch.float16) * 0.5,
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, ctx, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, ctx, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, ctx, hd_f, dtype=torch.float16),
    )

    print("\n[smoke] running baseline SWAChunk4")
    with torch.no_grad():
        ref_id, ref_logit, ref_h = baseline(*sample)
    print(f"  baseline argmax token_id={ref_id.item()}  logit={ref_logit.item():.3f}")

    for n, mod in splits.items():
        with torch.no_grad():
            tid, tlogit, th = mod(*sample)
        same_id = bool((tid == ref_id).all())
        max_h_diff = float((th - ref_h).abs().max())
        print(f"  n_splits={n:>2}  argmax id={tid.item():<6} same_id={same_id}  "
              f"hidden_max_diff={max_h_diff:.3e}  logit={tlogit.item():.3f}")

    print("\n[smoke] tracing n_splits=16 (no CoreML convert)")
    traced = torch.jit.trace(splits[16], sample, check_trace=False)
    print(f"  traced module has {sum(p.numel() for p in traced.parameters()):,} params")

    print("\n[smoke] OK")


if __name__ == "__main__":
    main()
