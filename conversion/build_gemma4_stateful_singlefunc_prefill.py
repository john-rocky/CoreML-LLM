#!/usr/bin/env python3
"""Build Gemma 4 stateful prefill chunks as **single-function**
mlpackages (no multifunction merge). Supports E2B and E4B; chunk
boundaries come from `compute_chunk_boundaries(config)`.

Stage 8 / Stage 6.5 opt-in builder. Companion to
`docs/HANDOFF_STAGE8_MLSTATE_MULTIMODAL.md` and the probe results in
`docs/MLSTATE_MULTIMODAL_PROBE.md`.

Why this is separate from `build_gemma4_e2b_stateful_chunks.py` /
`build_gemma4_e2b_stateful_3chunks.py`:

  - Those scripts emit prefill mlpackages with multifunction merge
    (`prefill_bN` baked into the same mlpackage as decode T=1).
    iPhone ANE 18 rejects multifunction T>1 + dual MLState with
    `ANECCompile FAILED 11` (Stage 3 finding).
  - Probe 1 (2026-04-28, iPhone 17 Pro A19 Pro) showed that
    **single-function** T>1 stateful prefill compiles cleanly: the
    multifunction code path is the specific blocker, not stateful
    T>1 itself.

This builder skips the multifunction merge entirely and emits each
prefill chunk as a standalone mlpackage. The Swift engine (Stage 8
work, not yet implemented) will load the prefill chunk separately
from the decode chunk and bridge KV state via `state.withMultiArray(for:)`
+ memcpy (probe 2 verified).

Layout produced (3-chunk merged variant, T=288 default):

    E2B (35 layers):
      chunk_1_prefill_T288.mlpackage       (L0-7, own KV)
      chunk_2_3way_prefill_T288.mlpackage  (L8-24 merged, own + shared)
      chunk_3_prefill_T288.mlpackage       (L25-34 + lm_head + argmax)
    E4B (42 layers):
      chunk_1_prefill_T288.mlpackage       (L0-11, own KV)
      chunk_2_3way_prefill_T288.mlpackage  (L12-32 merged, own + shared)
      chunk_3_prefill_T288.mlpackage       (L33-41 + lm_head + argmax)

Usage:
    python conversion/build_gemma4_stateful_singlefunc_prefill.py \\
        --hf-dir /path/to/gemma4-e2b/hf_model \\
        --output /tmp/stateful_singlefunc_prefill \\
        --t 288 \\
        --ctx 2048 \\
        --linear-projections

Then on iPhone:
    1. Compile each .mlpackage to .mlmodelc on the build host.
    2. Push via `xcrun devicectl device copy to ...` to the app
       sandbox (e.g. Documents/mlstate_probe/).
    3. Verify ANE compile via the "MLState Probe" buttons in
       ModelPickerView (research section). Probe 1 covers chunk_1;
       Stage 8 needs the same probe for chunk_2_3way + chunk_3.

T sizing rationale:
  - T=288 covers the full 256-token image-pad span plus ~32 tokens
    of leading text (BOS / turn markers / etc) in one forward.
  - Drop to T=224 if cross-device compile (8 GB iPhone non-Pro)
    rejects T=288 — image still fits, leading text gets truncated
    or moved.
  - T below 256 means the image span itself splits across
    forwards, defeating bidirectional attention. Don't go that low.

Cross-device compile risk:
  - iPhone 17 Pro (12 GB RAM): probe 1 PASS at T=288.
  - iPhone 17 / 16 / 15 Pro (8 GB RAM): unprobed.
  - iPhone 15 (6 GB RAM): may fail compile peak.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from build_gemma4_e2b_stateful_chunks import (
    _resolve_hf_dir,
    convert_chunk1_prefill,
    convert_chunk2_prefill,
    convert_chunk_shared_prefill,
)
from build_gemma4_e2b_stateful_3chunks import convert_chunk2_merged_prefill
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import compute_chunk_boundaries
from models.gemma4_swa_stateful_chunks import (
    SWAStatefulChunk3Prefill,
    SWAStatefulChunk4Prefill,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b",
                    help="MODEL_REGISTRY entry. Default gemma4-e2b.")
    ap.add_argument("--hf-dir", default=None,
                    help="HF model dir override (else MODEL_REGISTRY).")
    ap.add_argument("--output", default=None,
                    help="Output directory (default output/<model>/"
                         "stateful_singlefunc_prefill).")
    ap.add_argument("--t", type=int, default=288,
                    help="Prefill batch size T. Default 288 (256 image-pad "
                         "+ 32 text margin).")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--nbits", type=int, default=4,
                    help="INT4 palettization (default 4; 0 disables).")
    ap.add_argument("--linear-projections", action="store_true",
                    help="Plan 3 Linear projections (cml9 PR #2577) — "
                         "default on for Stage 3 / Stage 8 ship parity.")
    ap.add_argument("--only", choices=("chunk1", "chunk2_3way", "chunk3",
                                        "chunk2_own", "chunk3_shared",
                                        "chunk4_final"),
                    default=None,
                    help="Build only one chunk (debug).")
    ap.add_argument("--four-chunk", action="store_true",
                    help="Build 4-chunk variant: chunk_1, chunk_2 (own only), "
                         "chunk_3 (KV-shared no lm_head), chunk_4 (KV-shared "
                         "+ lm_head). Use when E4B chunk_2 merged graph is "
                         "rejected by iPhone ANE 18 (std::bad_cast at "
                         "MIL→EIR translation). Default off (3-chunk merged).")
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(
            ROOT, "..", "output", args.model,
            "stateful_singlefunc_prefill")
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"[build] model={args.model}  hf_dir={hf_dir}")
    print(f"[build] T={args.t}  ctx={args.ctx}  nbits={args.nbits}  "
          f"linear={args.linear_projections}")
    print(f"[build] output={args.output}")

    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()

    # Chunk layout (config-derived via compute_chunk_boundaries):
    #   E2B: c1=L0-7,   own=L8-14,  shared=L15-24, c4=L25-34
    #   E4B: c1=L0-11,  own=L12-23, shared=L24-32, c4=L33-41
    # The merged prefill needs own_range + shared_range so it picks
    # the right layer-index window for the kv13/kv14 producer aliases.
    boundaries = compute_chunk_boundaries(base.config)
    c1_start, c1_end = boundaries[0]
    own_range = boundaries[1]
    shared_range = boundaries[2]
    c4_start, c4_end = boundaries[3]

    if args.four_chunk:
        paths = {
            "chunk1": os.path.join(args.output, f"chunk_1_prefill_T{args.t}.mlpackage"),
            "chunk2_own": os.path.join(args.output, f"chunk_2_prefill_T{args.t}.mlpackage"),
            "chunk3_shared": os.path.join(args.output, f"chunk_3_prefill_T{args.t}.mlpackage"),
            "chunk4_final": os.path.join(args.output, f"chunk_4_prefill_T{args.t}.mlpackage"),
        }
    else:
        paths = {
            "chunk1": os.path.join(args.output, f"chunk_1_prefill_T{args.t}.mlpackage"),
            "chunk2_3way": os.path.join(args.output, f"chunk_2_3way_prefill_T{args.t}.mlpackage"),
            "chunk3": os.path.join(args.output, f"chunk_3_prefill_T{args.t}.mlpackage"),
        }

    t0 = time.time()
    if args.only in (None, "chunk1"):
        convert_chunk1_prefill(
            base=base,
            c_start=c1_start, c_end=c1_end,
            ctx=args.ctx, T=args.t,
            out_path=paths["chunk1"],
            nbits=args.nbits,
            use_linear=args.linear_projections,
        )
    if args.four_chunk:
        # 4-chunk path: chunk_2 = own only, chunk_3 = KV-shared (no lm_head),
        # chunk_4 = KV-shared + lm_head. Splits the 3-chunk merged middle so
        # each subgraph stays under iPhone ANE 18 compile budget.
        own_start, own_end = own_range
        shared_start, shared_end = shared_range
        if args.only in (None, "chunk2_own"):
            convert_chunk2_prefill(
                base=base,
                c_start=own_start, c_end=own_end,
                ctx=args.ctx, T=args.t,
                out_path=paths["chunk2_own"],
                nbits=args.nbits,
                use_linear=args.linear_projections,
            )
        if args.only in (None, "chunk3_shared"):
            convert_chunk_shared_prefill(
                chunk_cls=SWAStatefulChunk3Prefill,
                base=base,
                c_start=shared_start, c_end=shared_end,
                ctx=args.ctx, T=args.t,
                out_path=paths["chunk3_shared"],
                nbits=args.nbits,
                name="CHUNK 3 (KV-shared, no lm_head)",
                with_lm_head=False,
                use_linear=args.linear_projections,
            )
        if args.only in (None, "chunk4_final"):
            convert_chunk_shared_prefill(
                chunk_cls=SWAStatefulChunk4Prefill,
                base=base,
                c_start=c4_start, c_end=c4_end,
                ctx=args.ctx, T=args.t,
                out_path=paths["chunk4_final"],
                nbits=args.nbits,
                name="CHUNK 4 (KV-shared + lm_head)",
                with_lm_head=True,
                use_linear=args.linear_projections,
            )
    else:
        if args.only in (None, "chunk2_3way"):
            convert_chunk2_merged_prefill(
                base=base, ctx=args.ctx, T=args.t,
                out_path=paths["chunk2_3way"],
                nbits=args.nbits,
                use_linear=args.linear_projections,
                own_range=own_range,
                shared_range=shared_range,
            )
        if args.only in (None, "chunk3"):
            convert_chunk_shared_prefill(
                chunk_cls=SWAStatefulChunk4Prefill,
                base=base,
                c_start=c4_start, c_end=c4_end,
                ctx=args.ctx, T=args.t,
                out_path=paths["chunk3"],
                nbits=args.nbits,
                name="CHUNK 3 (final)",
                with_lm_head=True,
                use_linear=args.linear_projections,
            )
    print(f"\n[build] DONE in {time.time()-t0:.0f}s")
    print("=" * 60)
    layout = "4-chunk" if args.four_chunk else "3-chunk merged"
    print(f"{layout} stateful single-function prefill (T={args.t}):")
    for label, path in paths.items():
        if not os.path.exists(path):
            continue
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(path) for f in fs
        ) / 1024 / 1024
        print(f"  {label:<13s} {size_mb:>7.1f} MB  {path}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. xcrun coremlcompiler compile <each>.mlpackage <out>/")
    print("  2. devicectl push the .mlmodelc files to the app sandbox.")
    print("  3. Use ModelPickerView's 'MLState Probe (research)' buttons")
    print("     to verify iPhone ANE compile + state-buffer bridging")
    print("     (probe 1 already validated chunk_1 — repeat for the")
    print("     other chunks and on non-Pro RAM devices).")
    print("  4. See docs/HANDOFF_STAGE8_MLSTATE_MULTIMODAL.md for the")
    print("     full Stage 8 implementation plan.")


if __name__ == "__main__":
    main()
