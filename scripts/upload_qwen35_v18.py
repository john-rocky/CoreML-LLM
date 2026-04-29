#!/usr/bin/env python3
"""Upload Qwen3.5 v1.8.0 MLKV bundles (full-vocab rep_penalty path)
to HuggingFace Hub.

What changed vs v1.x:
- chunk_d emits FULL fp16 logits (1, 1, 248320) instead of in-graph
  topk[k=N]. Swift handles rep_penalty + argmax in fp32 over the
  full vocab (the iPhone A18 ANE fp16 reduction bias workaround).
- Body chunks (a, b, c) unchanged but rebuilt for consistency.
- embed_weight.bin unchanged.

Usage:
  python scripts/upload_qwen35_v18.py --model 0.8b
  python scripts/upload_qwen35_v18.py --model 2b
  python scripts/upload_qwen35_v18.py --model both --dry-run
"""

from __future__ import annotations
import argparse
import os
import sys
from huggingface_hub import HfApi, create_repo


BUNDLES = {
    "0.8b": {
        "src": "/tmp/qwen35_0_8b_mlkv_unified_rerank/qwen3_5_0_8b_decode_chunks_mlkv",
        "repo": "mlboydaisuke/qwen3.5-0.8B-CoreML",
        "remote_root": "qwen3_5_0_8b_decode_chunks_mlkv",
    },
    "2b": {
        "src": "/tmp/qwen35_2b_mlkv_logits/qwen3_5_2b_decode_chunks_mlkv",
        "repo": "mlboydaisuke/qwen3.5-2B-CoreML",
        "remote_root": "qwen3_5_2b_decode_chunks_mlkv",
    },
}

# Files inside each chunk_*.mlpackage that the iOS downloader expects.
PKG_FILES = [
    "Manifest.json",
    "Data/com.apple.CoreML/model.mlmodel",
    "Data/com.apple.CoreML/weights/weight.bin",
]


def list_files(local_root: str, remote_root: str) -> list[tuple[str, str]]:
    """Return [(local_path, remote_path), ...] for all bundle files."""
    pairs: list[tuple[str, str]] = []
    for chunk in ("chunk_a", "chunk_b", "chunk_c", "chunk_d"):
        for rel in PKG_FILES:
            local = os.path.join(local_root, f"{chunk}.mlpackage", rel)
            remote = f"{remote_root}/{chunk}.mlpackage/{rel}"
            if not os.path.exists(local):
                print(f"  SKIP missing {local}")
                continue
            pairs.append((local, remote))
    embed_local = os.path.join(local_root, "embed_weight.bin")
    if os.path.exists(embed_local):
        pairs.append((embed_local, f"{remote_root}/embed_weight.bin"))
    else:
        print(f"  SKIP missing {embed_local}")
    return pairs


def upload(model: str, dry_run: bool) -> None:
    bundle = BUNDLES[model]
    pairs = list_files(bundle["src"], bundle["remote_root"])
    total_mb = sum(os.path.getsize(l) for l, _ in pairs) / 1024 / 1024
    print(f"\n=== {model} ===  {bundle['repo']}")
    print(f"  Source: {bundle['src']}")
    print(f"  Files:  {len(pairs)} ({total_mb:.0f} MB total)")
    for l, r in pairs:
        sz = os.path.getsize(l) / 1024 / 1024
        print(f"    {sz:>7.1f} MB  →  {r}")

    if dry_run:
        print("  --dry-run: not uploading.")
        return

    api = HfApi()
    create_repo(bundle["repo"], repo_type="model", exist_ok=True)
    for local, remote in pairs:
        sz_mb = os.path.getsize(local) / 1024 / 1024
        print(f"Uploading {remote} ({sz_mb:.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=bundle["repo"],
            repo_type="model",
            commit_message=f"v1.8.0: full-vocab rep_penalty path — {os.path.basename(remote)}",
        )
    print(f"\n✅ {model} done. https://huggingface.co/{bundle['repo']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["0.8b", "2b", "both"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    targets = ["0.8b", "2b"] if args.model == "both" else [args.model]
    for m in targets:
        upload(m, args.dry_run)


if __name__ == "__main__":
    main()
