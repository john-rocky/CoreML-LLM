#!/usr/bin/env python3
"""MLState multimodal feasibility probe — single-function prefill at T=N.

Stage 3 found that **multifunction** (`prefill_bN`) prefill chunks with
dual MLState (`kv_cache_sliding` + `kv_cache_full`) are rejected by the
iPhone ANE 18 compiler with `ANECCompile FAILED 11`. That blocked Stage
6 multimodal stateful — vision needs full bidirectional within-image-pad
attention which doesn't fit in T=8 batches.

This probe asks an untested question: does **single-function** (separate
mlpackage, NOT multifunction) prefill with MLState compile on iPhone
ANE 18 at large T?

If yes → we have a path to MLState multimodal:
  prefill model (T=288, MLState) processes image span in one forward
  → memcpy state buffers → decode model (T=1, MLState) continues
If no → MLState multimodal is structurally rejected; retire that path.

Usage:
    python conversion/probe_stateful_singlefunc_prefill.py \\
        --hf-dir /path/to/hf_model \\
        --output /tmp/mlstate_probe \\
        --t 64
        # then 128, 256, 288 if 64 passes

Output: a standalone `chunk_1_prefill_T{N}.mlpackage` with MLState states.
Compile + load on Mac is the easy half; the ANE-18 verdict requires
copying the mlpackage to iPhone and trying `MLModel(contentsOf:)`.
"""

from __future__ import annotations
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from build_gemma4_e2b_stateful_chunks import convert_chunk1_prefill
from models.gemma4 import Gemma4Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default=None,
                    help="HF model dir (defaults to MODEL_REGISTRY entry).")
    ap.add_argument("--output", default="/tmp/mlstate_probe",
                    help="Output directory.")
    ap.add_argument("--t", type=int, default=64,
                    help="Prefill batch size T to probe.")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--nbits", type=int, default=4,
                    help="INT4 palettization (default; 0 disables).")
    ap.add_argument("--use-linear", action="store_true",
                    help="Plan 3 Linear projections (cml9 PR #2577).")
    args = ap.parse_args()

    if args.hf_dir is None:
        from config import MODEL_REGISTRY
        cfg_entry = MODEL_REGISTRY["gemma4-e2b"]
        from huggingface_hub import snapshot_download
        args.hf_dir = snapshot_download(cfg_entry.hf_repo)

    os.makedirs(args.output, exist_ok=True)

    print(f"[probe] hf_dir = {args.hf_dir}")
    print(f"[probe] T = {args.t}, ctx = {args.ctx}, nbits = {args.nbits}, "
          f"linear = {args.use_linear}")

    base = Gemma4Model.from_pretrained(args.hf_dir, context_length=args.ctx)
    base.eval()

    out_path = os.path.join(args.output, f"chunk_1_prefill_T{args.t}.mlpackage")
    t0 = time.time()
    convert_chunk1_prefill(
        base=base,
        c_start=0, c_end=8,        # E2B L0-7
        ctx=args.ctx,
        T=args.t,
        out_path=out_path,
        nbits=args.nbits,
        use_linear=args.use_linear,
    )
    dt = time.time() - t0
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(out_path) for f in fs
    ) / 1024 / 1024
    print(f"\n[probe] DONE in {dt:.0f}s — {out_path} ({size_mb:.1f} MB)")
    print("[probe] Next: copy to iPhone and try MLModel.load. If `ANECCompile "
          "FAILED 11` shows up, retry with smaller T. If all T<=288 fail, "
          "single-function MLState T>1 is also rejected by ANE 18 → "
          "stateful multimodal is structurally retired.")


if __name__ == "__main__":
    main()
