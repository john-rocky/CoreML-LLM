#!/usr/bin/env python3
"""Compile Topology-I mlpackages and drop them into an existing bundle.

Usage:
    python conversion/install_topoI_bundle.py --model gemma4-e2b
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time


def _du_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024 / 1024
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
    return total / 1024 / 1024


def _compile(pkg, out_mlmodelc):
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
    chunks_dir = args.chunks_dir or os.path.join(root, "..", "output", args.model, "chunks_topoI")
    bundle_dir = args.bundle_dir or os.path.join(root, "..", "output", args.model, "bundle")
    chunks_dir = os.path.abspath(chunks_dir); bundle_dir = os.path.abspath(bundle_dir)
    if not os.path.isdir(chunks_dir):
        raise SystemExit(f"no such chunks dir: {chunks_dir}")
    if not os.path.isdir(bundle_dir):
        raise SystemExit(f"no such bundle dir: {bundle_dir}")

    for pkg_name, mlc_name in [
        ("chunk1_topoI.mlpackage", "chunk1_topoI.mlmodelc"),
        ("chunk2_topoI.mlpackage", "chunk2_topoI.mlmodelc"),
        ("chunk3_topoI.mlpackage", "chunk3_topoI.mlmodelc"),
    ]:
        pkg = os.path.join(chunks_dir, pkg_name)
        out = os.path.join(bundle_dir, mlc_name)
        if not os.path.isdir(pkg):
            raise SystemExit(f"missing {pkg}")
        _compile(pkg, out)

    print("\n" + "=" * 60)
    print(f"Installed Topology-I decoders in {bundle_dir}")
    print("=" * 60)
    print("On iPhone, set LLM_3CHUNK=1 + LLM_3CHUNK_TOPO=I to route decode via them.")
    print("devicectl:")
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
