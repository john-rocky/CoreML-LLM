#!/usr/bin/env python3
"""Compile 3-chunk decode mlpackages and drop them into an existing bundle.

Takes the output of build_gemma4_3way.py (three .mlpackage files under
output/<model>/chunks_3way/) and compiles each to .mlmodelc, then copies the
two renamed files into output/<model>/bundle/ so LLM_3CHUNK=1 at runtime
can pick them up alongside the existing 4-chunk decoders.

Preserves (does not overwrite) the existing chunk1.mlmodelc — the 3-chunk
variant's chunk1 is architecturally + bit-identical. Installs:
    chunk2_3way.mlmodelc  (= merged L8-24 17-layer decoder)
    chunk3_3way.mlmodelc  (= LM-head chunk, L25-34 + norm + lm_head + argmax)

Skipped (still sourced from 4-chunk bundle):
    chunk1.mlmodelc, chunk3.mlmodelc, chunk4.mlmodelc, prefill_chunk{1..4}.mlmodelc,
    per_layer_projection.bin, embed_tokens_*.bin, RoPE tables, tokenizer files.

Usage:
    python conversion/install_3way_bundle.py --model gemma4-e2b
    # or explicit directories:
    python conversion/install_3way_bundle.py \\
        --chunks-dir output/gemma4-e2b/chunks_3way \\
        --bundle-dir output/gemma4-e2b/bundle
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time


def _du_mb(path: str) -> float:
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024 / 1024
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
    return total / 1024 / 1024


def _compile(pkg: str, out_mlmodelc: str) -> None:
    import coremltools as ct
    print(f"  compile: {os.path.basename(pkg)} → {os.path.basename(out_mlmodelc)}")
    t = time.time()
    model = ct.models.MLModel(pkg)
    compiled = model.get_compiled_model_path()
    if os.path.exists(out_mlmodelc):
        shutil.rmtree(out_mlmodelc)
    shutil.copytree(compiled, out_mlmodelc)
    del model
    print(f"    done  {_du_mb(out_mlmodelc):.1f} MB  ({time.time()-t:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--chunks-dir", default=None)
    ap.add_argument("--bundle-dir", default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    if args.chunks_dir is None:
        args.chunks_dir = os.path.join(root, "..", "output", args.model, "chunks_3way")
    if args.bundle_dir is None:
        args.bundle_dir = os.path.join(root, "..", "output", args.model, "bundle")

    chunks_dir = os.path.abspath(args.chunks_dir)
    bundle_dir = os.path.abspath(args.bundle_dir)

    if not os.path.isdir(chunks_dir):
        raise SystemExit(f"no such chunks dir: {chunks_dir}  "
                         "(run build_gemma4_3way.py first)")
    if not os.path.isdir(bundle_dir):
        raise SystemExit(f"no such bundle dir: {bundle_dir}  "
                         "(run build_gemma4_bundle.py first)")

    # The builder's chunk1_3way is byte-equivalent to chunk1 in the 4-chunk
    # bundle, so we reuse the existing compiled chunk1.mlmodelc. Only the
    # two renamed chunks get installed.
    renames = [
        ("chunk2_3way.mlpackage", "chunk2_3way.mlmodelc"),
        ("chunk3_3way.mlpackage", "chunk3_3way.mlmodelc"),
    ]

    for pkg_name, mlc_name in renames:
        pkg = os.path.join(chunks_dir, pkg_name)
        out = os.path.join(bundle_dir, mlc_name)
        if not os.path.isdir(pkg):
            raise SystemExit(f"missing {pkg} (run build_gemma4_3way.py first)")
        _compile(pkg, out)

    # Sanity check: make sure the existing 4-chunk bundle is still intact.
    required = ["chunk1.mlmodelc", "chunk2.mlmodelc", "chunk3.mlmodelc",
                "chunk4.mlmodelc", "model_config.json"]
    missing = [r for r in required if not os.path.exists(os.path.join(bundle_dir, r))]
    if missing:
        print(f"\n[warn] 4-chunk bundle files missing in {bundle_dir}: {missing}")
        print("       LLM_3CHUNK=0 (default) will error at runtime.")

    print("\n" + "=" * 60)
    print(f"Installed 3-chunk decoders into {bundle_dir}")
    print("=" * 60)
    print("On iPhone, set LLM_3CHUNK=1 to route decode through the new chunks.")
    print("Re-deploy the bundle with devicectl:")
    print(
        "  xcrun devicectl device copy to \\\n"
        '    --device "$DEVICE" \\\n'
        "    --domain-type appDataContainer \\\n"
        "    --domain-identifier com.example.CoreMLLLMChat \\\n"
        f"    --source {bundle_dir} \\\n"
        f"    --destination Documents/Models/{args.model} \\\n"
        "    --remove-existing-content true"
    )


if __name__ == "__main__":
    main()
