#!/usr/bin/env python3
"""Wire up a cross-vocabulary (Qwen) drafter for an existing Gemma model.

Usage
-----
    python setup_cross_vocab_drafter.py \
        --gemma-dir /path/to/gemma4-e2b \
        --qwen-dir  /path/to/qwen2.5-0.5b

After running this, the Swift runtime will pick up the drafter at load time
and enable cross-vocabulary speculative decoding (Route B / Task 3).

What it does
------------
1. Symlinks the Qwen mlmodelc into <gemma-dir>/cross_vocab/qwen_drafter.mlmodelc
2. Builds the Qwen <-> Gemma vocab map via build_qwen_gemma_vocab_map.py
3. Prints coverage stats
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma-dir", required=True, type=Path,
                    help="Gemma model directory (contains chunk1.mlmodelc / hf_model)")
    ap.add_argument("--qwen-dir", required=True, type=Path,
                    help="Qwen 2.5 0.5B model directory (contains model.mlmodelc / hf_model)")
    args = ap.parse_args()

    gdir = args.gemma_dir.resolve()
    qdir = args.qwen_dir.resolve()

    if not gdir.is_dir():
        print(f"[setup] gemma dir missing: {gdir}", file=sys.stderr)
        return 2
    if not qdir.is_dir():
        print(f"[setup] qwen dir missing: {qdir}", file=sys.stderr)
        return 2

    qwen_model = qdir / "model.mlmodelc"
    if not qwen_model.is_dir():
        qwen_model = qdir / "model.mlpackage"
    if not qwen_model.exists():
        print(f"[setup] qwen model.mlmodelc or model.mlpackage not found under {qdir}",
              file=sys.stderr)
        return 2

    g_tokenizer = gdir / "hf_model"
    q_tokenizer = qdir / "hf_model"
    for d in (g_tokenizer, q_tokenizer):
        if not d.is_dir():
            print(f"[setup] tokenizer dir missing: {d}", file=sys.stderr)
            return 2

    cv_dir = gdir / "cross_vocab"
    cv_dir.mkdir(exist_ok=True)

    # Symlink the Qwen model so we don't double-disk-space.
    link_name = "qwen_drafter" + qwen_model.suffix
    link_target = cv_dir / link_name
    if link_target.exists() or link_target.is_symlink():
        link_target.unlink()
    os.symlink(qwen_model, link_target)
    print(f"[setup] linked {link_target} -> {qwen_model}")

    # Build the vocab map.
    map_out = cv_dir / "qwen_gemma_vocab.bin"
    builder = Path(__file__).parent / "build_qwen_gemma_vocab_map.py"
    cmd = [
        sys.executable, str(builder),
        "--qwen-tokenizer", str(q_tokenizer),
        "--gemma-tokenizer", str(g_tokenizer),
        "--output", str(map_out),
    ]
    print(f"[setup] running: {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[setup] vocab map builder failed with code {rc}", file=sys.stderr)
        return rc

    print(f"[setup] done. Drafter will auto-load from {cv_dir} at next runtime start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
