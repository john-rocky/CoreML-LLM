#!/usr/bin/env python3
"""Extract Gemma 4 lm_head weight as a standalone fp16 binary file.

Output: output/gemma4-e2b/lm_head_fp16.bin
Format: row-major (V=262144 rows × H=1536 cols), little-endian fp16.
Total size: ~800 MB. Swift loads + gathers candidate rows for sparse matmul.

INT4 palettized variant: lm_head_int4.bin (~200 MB) — only if we want to
save iPhone storage; sparse matmul still needs fp32 ops though.

Usage:
    python conversion/extract_lm_head.py --model gemma4-e2b
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--hf-dir", default=None)
    args = ap.parse_args()

    if args.hf_dir is None:
        repo = MODEL_REGISTRY[args.model].hf_repo
        local = os.path.join(ROOT, "..", "output", args.model, "hf_model")
        args.hf_dir = local

    print(f"loading {args.hf_dir}...")
    base = Gemma4Model.from_pretrained(args.hf_dir)
    base.eval()

    lm_w = base.lm_head.weight.data  # (vocab, hidden, 1, 1) for Conv2d
    if lm_w.ndim == 4:
        lm_w = lm_w.squeeze(-1).squeeze(-1)
    V, H = lm_w.shape
    print(f"  lm_head: ({V}, {H}) dtype={lm_w.dtype}")

    out_dir = args.output_dir or os.path.join(
        ROOT, "..", "output", args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lm_head_fp16.bin")

    lm_fp16 = lm_w.detach().to(torch.float16).contiguous().cpu().numpy()
    print(f"  writing {out_path} ({lm_fp16.nbytes / 1024 / 1024:.1f} MB) ...")
    with open(out_path, "wb") as f:
        f.write(lm_fp16.tobytes())
    print(f"  done")

    # Sanity check: read back first row
    sanity = np.fromfile(out_path, dtype=np.float16, count=H)
    diff = np.max(np.abs(sanity - lm_fp16[0]))
    print(f"  sanity check row 0 max diff: {diff:.2e}")

    # Also write the model_config.json piece that confirms vocab + hidden
    cfg_path = os.path.join(out_dir, "lm_head_meta.json")
    import json
    with open(cfg_path, "w") as f:
        json.dump({
            "vocab_size": V,
            "hidden_size": H,
            "dtype": "fp16",
            "shape": [V, H],
            "layout": "row_major",
        }, f, indent=2)
    print(f"  meta: {cfg_path}")


if __name__ == "__main__":
    main()
