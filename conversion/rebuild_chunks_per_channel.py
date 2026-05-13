#!/usr/bin/env python3
"""Rebuild Gemma 4 E2B chunks with per-channel INT4 palettization (T7 lever).

T7 lever (MobileLLM-Pro recipe, arxiv 2511.06719): per-channel INT4
beats group-wise INT4 on accelerators because group-wise causes
LUT-decode stall on NPUs.

Existing builds use `granularity=per_grouped_channel, group_size=32`.
This script re-palettizes the same fp16 mlpackages with
`granularity=per_channel` (no group_size) for an iPhone A/B.

PPL hit reported: per-channel 1.3% vs group-wise 0.4% (negligible for
chat-bot use).

Usage:
  pyenv shell lama-cml
  python conversion/rebuild_chunks_per_channel.py \
    --src output/gemma4-e2b/chunks_3way_fp16kv \
    --dst output/gemma4-e2b/chunks_3way_fp16kv_perch

Then compile to .mlmodelc and push to iPhone via
scripts/assemble_3way_mf_bundle.sh adapted for the new dest.
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct


def _du_mb(path: str) -> float:
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024 / 1024
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
    return total / 1024 / 1024


def repalettize(src: str, dst: str, *, nbits: int = 4,
                granularity: str = "per_channel") -> None:
    """Load an fp16 (un-palettized) mlpackage, re-palettize, save."""
    if not os.path.exists(src):
        raise FileNotFoundError(f"source mlpackage not found: {src}")
    print(f"[repalettize] loading {src}")
    t = time.time()
    mlmodel = ct.models.MLModel(src)
    print(f"  loaded in {time.time()-t:.1f}s")

    t = time.time()
    cfg_kw: dict = {"nbits": nbits, "granularity": granularity}
    cfg = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpPalettizerConfig(**cfg_kw))
    mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg)
    print(f"  palettized nbits={nbits} granularity={granularity} in {time.time()-t:.1f}s")

    if os.path.exists(dst):
        shutil.rmtree(dst)
    mlmodel.save(dst)
    size_mb = _du_mb(dst)
    print(f"  saved {dst}  ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Source directory with fp16 mlpackages (e.g. chunks_3way_fp16kv)")
    ap.add_argument("--dst", required=True,
                    help="Destination directory for per-channel mlpackages")
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--granularity", default="per_channel",
                    choices=["per_channel", "per_tensor", "per_grouped_channel"])
    ap.add_argument("--group-size", type=int, default=32,
                    help="Only used when granularity=per_grouped_channel")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # Enumerate mlpackages in source. Handle both `chunk1.mlpackage`
    # naming and `chunk1_3way.mlpackage`.
    for name in sorted(os.listdir(args.src)):
        if not name.endswith(".mlpackage"):
            continue
        src_path = os.path.join(args.src, name)
        dst_path = os.path.join(args.dst, name)
        print(f"\n=== {name} ===")
        if args.granularity == "per_grouped_channel":
            print("  (group_size will be applied; only relevant when source is unpalettized)")
        repalettize(src_path, dst_path,
                    nbits=args.nbits,
                    granularity=args.granularity)

    print("\n" + "=" * 60)
    print(f"Per-channel chunks written to {args.dst}/")
    print("Next: compile to .mlmodelc and assemble bundle for iPhone push.")
    print(f"  for pkg in {args.dst}/*.mlpackage; do")
    print(f"    xcrun coremlcompiler compile $pkg <bundle-dir>/")
    print(f"  done")


if __name__ == "__main__":
    main()
