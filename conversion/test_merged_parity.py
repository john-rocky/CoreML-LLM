#!/usr/bin/env python3
"""Parity test: compare 2-chunk merged pipeline against the 4-chunk reference.

Both pipelines are pure PyTorch (pre-CoreML conversion). We use the same
Gemma4Model backbone and exercise 16 decode steps on a held-out prompt set
(same seed, same positions). For each step we compare:

  1. Final hidden_states (pre-argmax) — cosine similarity target > 0.9999
  2. Argmax token ID — must match exactly across the run

Why fp32 on CPU
---------------
On iPhone/ANE the decoder runs in fp16 with saturating arithmetic, so
intermediate products above 65504 simply clamp. PyTorch fp16 on Mac/CPU
does NOT saturate — it returns ``inf`` which propagates as NaN through the
later softmax / gating ops. Because the goal here is to prove the 2-chunk
merged pipeline is mathematically identical to the shipping 4-chunk graph
(not to reproduce ANE numerics), the whole test runs in fp32: we monkey-
patch ``MODEL_DTYPE`` before the model modules are imported, load weights
in fp32, and pass fp32 inputs. Device numerics are validated separately
on-device (see docs/CHUNK_CONSOLIDATION_BENCH.md).

Ground truth
------------
The 4-chunk SWA pipeline IS the shipping production code (see
``build_verify_chunks.py``) — it is what Swift actually runs on device,
so it serves as the ground-truth reference here. We do NOT compare against
``Gemma4MonolithicWrapper`` because it is the tracer for a legacy
single-graph export that (a) has a residual layout bug in its PLE path
(Conv2d expects NCHW, wrapper feeds BSH), and (b) is not on any shipping
code path. Device numerics (fp16, int4 palettized weights) are validated
separately via on-device benchmarks documented in
``docs/CHUNK_CONSOLIDATION_BENCH.md``.

Usage:
    python test_merged_parity.py                                 # default: 2-chunk vs 4-chunk
    python test_merged_parity.py --mode one                      # 1-chunk variant too
    GEMMA4_HF_DIR=/path/to/gemma-e2b python test_merged_parity.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# IMPORTANT: monkey-patch MODEL_DTYPE BEFORE importing the model modules so
# every ``from ane_ops import MODEL_DTYPE`` picks up fp32. Any later change
# to ``ane_ops.MODEL_DTYPE`` would NOT propagate to already-imported modules.
import ane_ops as _ane_ops
_ane_ops.MODEL_DTYPE = torch.float32
MODEL_DTYPE = torch.float32

from models.gemma4 import Gemma4Model  # noqa: E402
from models.gemma4_swa_chunks import SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4  # noqa: E402
from models.gemma4_swa_merged2 import MergedChunk12, MergedChunk34  # noqa: E402
from models.gemma4_swa_merged1 import MergedChunk1  # noqa: E402

# Also propagate fp32 to each model module's local MODEL_DTYPE binding.
# ``from ane_ops import MODEL_DTYPE`` creates a per-module name that is NOT
# re-read after import; we must overwrite each one so later attribute lookups
# inside those modules' functions see fp32.
import models.gemma4 as _g4
import models.gemma4_swa_chunks as _gc
import models.gemma4_swa_merged2 as _gm2
import models.gemma4_swa_merged1 as _gm1
for _m in (_g4, _gc, _gm2, _gm1):
    _m.MODEL_DTYPE = torch.float32

HF_DIR = os.environ.get("GEMMA4_HF_DIR", f"{ROOT}/../output/gemma4-e2b/hf_model")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().to(torch.float32)
    bf = b.flatten().to(torch.float32)
    denom = torch.norm(af) * torch.norm(bf)
    if denom.item() == 0.0:
        return 1.0
    return float(torch.dot(af, bf) / denom)


def build_decode_inputs(ctx, W, hidden, nlayers, pld, position, token_id, base):
    """Build the shared inputs for one decode step at `position`.

    Matches ``Gemma4MonolithicWrapper.forward`` exactly:
    - embedding scaled by sqrt(hidden_size)
    - per_layer_raw = embed_tokens_per_layer(ids) * per_layer_embed_scale
      (originally MISSING from this harness — caused every later PLE value
      to be off by sqrt(256) = 16, which combined with fp16 MLP overflow
      produced NaN at layer 0. The missing scale is the main harness bug.)
    - causal_mask_full uses -1e9 as the "masked" sentinel (not -inf), to
      match the monolithic wrapper and avoid inf propagation.
    """
    # Embedding + scaled
    emb = base.embed_tokens(torch.tensor([token_id], dtype=torch.long))
    emb = emb.view(1, 1, hidden).to(MODEL_DTYPE)
    emb = emb * (hidden ** 0.5)

    # Raw per-layer embedding — MUST be scaled by sqrt(per_layer_dim)
    # (see Gemma4MonolithicWrapper line 103). Dropping this scale leaves
    # per_layer_raw ~16x smaller than the projection branch, skewing the
    # PLE sum and producing garbage hidden states downstream.
    pl_raw = base.embed_tokens_per_layer(torch.tensor([token_id], dtype=torch.long))
    pl_raw = pl_raw.view(1, 1, nlayers * pld).to(MODEL_DTYPE) * base.per_layer_embed_scale

    # Causal masks (0 for allowed, -1e9 for masked — mirrors the wrapper)
    NEG_BIG = -1e9
    mask_full = torch.full((1, 1, 1, ctx), NEG_BIG, dtype=MODEL_DTYPE)
    mask_full[..., :position + 1] = 0.0
    mask_sliding = torch.full((1, 1, 1, W), NEG_BIG, dtype=MODEL_DTYPE)
    valid = min(position + 1, W)
    mask_sliding[..., W - valid:] = 0.0

    # Update mask (1 at position for full-attn, else 0)
    upd = torch.zeros(1, 1, ctx, 1, dtype=MODEL_DTYPE)
    upd[:, :, position, :] = 1.0

    # RoPE: use the base model's tables
    def pick(table, pos):
        return table[pos].unsqueeze(0).unsqueeze(0).unsqueeze(0).to(MODEL_DTYPE)
    cos_s = pick(base.cos_sliding, position)
    sin_s = pick(base.sin_sliding, position)
    cos_f = pick(base.cos_full, position)
    sin_f = pick(base.sin_full, position)
    return emb, pl_raw, mask_full, mask_sliding, upd, cos_s, sin_s, cos_f, sin_f


def run_4chunk(base, c1, c2, c3, c4, emb, pl_raw, mf, ms, upd,
               cos_s, sin_s, cos_f, sin_f,
               ksl1, vsl1, kfl1, vfl1, ksl2, vsl2, kfl2, vfl2):
    """Single decode step through SWA 4-chunk pipeline. Returns (token_id, hidden, updated caches)."""
    h, ksl1_o, vsl1_o, kfl1_o, vfl1_o, plc = c1(
        emb, mf, ms, upd, pl_raw,
        cos_s, sin_s, cos_f, sin_f,
        ksl1, vsl1, kfl1, vfl1,
    )
    h, ksl2_o, vsl2_o, kfl2_o, vfl2_o, kv13k, kv13v, kv14k, kv14v = c2(
        h, mf, ms, upd, plc,
        cos_s, sin_s, cos_f, sin_f,
        ksl2, vsl2, kfl2, vfl2,
    )
    h = c3(h, mf, ms, upd, plc,
           cos_s, sin_s, cos_f, sin_f,
           kv13k, kv13v, kv14k, kv14v)
    tid, _, normed = c4(h, mf, ms, upd, plc,
                        cos_s, sin_s, cos_f, sin_f,
                        kv13k, kv13v, kv14k, kv14v)
    return tid, normed, (ksl1_o, vsl1_o, kfl1_o, vfl1_o,
                         ksl2_o, vsl2_o, kfl2_o, vfl2_o)


def run_2chunk(m12, m34, emb, pl_raw, mf, ms, upd,
               cos_s, sin_s, cos_f, sin_f,
               ks, vs, kf, vf):
    """Single decode step through the 2-chunk pipeline."""
    (h, ks_o, vs_o, kf_o, vf_o, kv13k, kv13v, kv14k, kv14v, plc) = m12(
        emb, mf, ms, upd, pl_raw,
        cos_s, sin_s, cos_f, sin_f,
        ks, vs, kf, vf,
    )
    tid, _, normed = m34(
        h, mf, ms, upd, plc,
        cos_s, sin_s, cos_f, sin_f,
        kv13k, kv13v, kv14k, kv14v,
    )
    return tid, normed, (ks_o, vs_o, kf_o, vf_o)


def run_1chunk(mfull, emb, pl_raw, mf, ms, upd,
               cos_s, sin_s, cos_f, sin_f,
               ks, vs, kf, vf):
    tid, _, normed, ks_o, vs_o, kf_o, vf_o = mfull(
        emb, mf, ms, upd, pl_raw,
        cos_s, sin_s, cos_f, sin_f,
        ks, vs, kf, vf,
    )
    return tid, normed, (ks_o, vs_o, kf_o, vf_o)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["two", "one", "both"], default="two",
                        help="Which merged variant(s) to check")
    parser.add_argument("--hf-dir", default=HF_DIR)
    parser.add_argument("--ctx", type=int, default=512,
                        help="Context length for test (keep small — CPU torch is slow)")
    parser.add_argument("--steps", type=int, default=16, help="Decode steps to simulate")
    parser.add_argument("--tol-cos", type=float, default=0.9999,
                        help="Minimum required cosine similarity on hidden states")
    args = parser.parse_args()

    print(f"Loading Gemma 4 E2B from {args.hf_dir} (fp32 for CPU stability)...")
    base = Gemma4Model.from_pretrained(args.hf_dir, context_length=args.ctx)
    base = base.to(torch.float32)
    base.eval()

    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    W = 512
    max_hd = 512
    ctx = args.ctx

    # Build chunks once (weights shared with base) ------------------------------
    c1 = SWAChunk1(base).eval()
    c2 = SWAChunk2(base).eval()
    c3 = SWAChunk3(base).eval()
    c4 = SWAChunk4(base).eval()
    m12 = MergedChunk12(base).eval()
    m34 = MergedChunk34(base).eval()
    mfull = MergedChunk1(base).eval() if args.mode in ("one", "both") else None

    # Allocate independent KV caches per pipeline (they drift from each other
    # during the run when the underlying math disagrees).
    def alloc_ref():
        return [
            torch.zeros(7, 1, W, max_hd, dtype=MODEL_DTYPE),
            torch.zeros(7, 1, W, max_hd, dtype=MODEL_DTYPE),
            torch.zeros(1, 1, ctx, max_hd, dtype=MODEL_DTYPE),
            torch.zeros(1, 1, ctx, max_hd, dtype=MODEL_DTYPE),
            torch.zeros(5, 1, W, max_hd, dtype=MODEL_DTYPE),
            torch.zeros(5, 1, W, max_hd, dtype=MODEL_DTYPE),
            torch.zeros(2, 1, ctx, max_hd, dtype=MODEL_DTYPE),
            torch.zeros(2, 1, ctx, max_hd, dtype=MODEL_DTYPE),
        ]

    def alloc_merged():
        return [
            torch.zeros(12, 1, W, max_hd, dtype=MODEL_DTYPE),
            torch.zeros(12, 1, W, max_hd, dtype=MODEL_DTYPE),
            torch.zeros(3, 1, ctx, max_hd, dtype=MODEL_DTYPE),
            torch.zeros(3, 1, ctx, max_hd, dtype=MODEL_DTYPE),
        ]

    kvs_ref = alloc_ref()
    kvs_m2 = alloc_merged()
    kvs_m1 = alloc_merged() if mfull is not None else None

    # Deterministic test prompt (ids small enough to always lie in vocab)
    prompt_ids = [2, 106, 2364, 108, 791, 603, 573, 4791, 576, 2909, 15050, 603]
    if len(prompt_ids) < args.steps:
        prompt_ids = prompt_ids + [0] * (args.steps - len(prompt_ids))
    prompt_ids = prompt_ids[: args.steps]

    any_fail = False
    for step, tid in enumerate(prompt_ids):
        pos = step
        emb, pl_raw, mf, ms, upd, cs, ss, cf, sf = build_decode_inputs(
            ctx, W, hidden, nlayers, pld, pos, tid, base)

        tid_ref, h_ref, new_ref = run_4chunk(
            base, c1, c2, c3, c4, emb, pl_raw, mf, ms, upd,
            cs, ss, cf, sf, *kvs_ref,
        )
        kvs_ref = list(new_ref)

        if args.mode in ("two", "both"):
            tid_m2, h_m2, new_m2 = run_2chunk(
                m12, m34, emb, pl_raw, mf, ms, upd, cs, ss, cf, sf, *kvs_m2,
            )
            kvs_m2 = list(new_m2)
            cos_h = cosine(h_ref, h_m2)
            match = int(tid_ref.item()) == int(tid_m2.item())
            ok = cos_h >= args.tol_cos and match
            any_fail = any_fail or not ok
            print(f"[2-chunk] step={step:2d} cos(h)={cos_h:.6f} "
                  f"tid_ref={int(tid_ref.item()):6d} tid_m2={int(tid_m2.item()):6d} "
                  f"{'OK' if ok else 'FAIL'}")

        if args.mode in ("one", "both") and mfull is not None:
            tid_m1, h_m1, new_m1 = run_1chunk(
                mfull, emb, pl_raw, mf, ms, upd, cs, ss, cf, sf, *kvs_m1,
            )
            kvs_m1 = list(new_m1)
            cos_h = cosine(h_ref, h_m1)
            match = int(tid_ref.item()) == int(tid_m1.item())
            ok = cos_h >= args.tol_cos and match
            any_fail = any_fail or not ok
            print(f"[1-chunk] step={step:2d} cos(h)={cos_h:.6f} "
                  f"tid_ref={int(tid_ref.item()):6d} tid_m1={int(tid_m1.item()):6d} "
                  f"{'OK' if ok else 'FAIL'}")

    print()
    if any_fail:
        print("PARITY FAIL — do NOT ship. Inspect first failing step above.")
        sys.exit(1)
    else:
        print(f"PARITY OK across {args.steps} steps (cos >= {args.tol_cos}).")


if __name__ == "__main__":
    main()
