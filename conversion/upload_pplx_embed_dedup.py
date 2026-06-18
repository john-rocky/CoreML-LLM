#!/usr/bin/env python3
"""Guaranteed-minimal-transfer upload of the staged pplx-embed repo tree.

The staged tree has byte-identical `weight.bin`s across buckets (single fixed RoPE
table), but `hf upload-large-folder` re-uploads identical oids across its parallel
batches, so dedup "doesn't work" and you push ~14 GB instead of ~2.3 GB.

This uploads each UNIQUE file content exactly once, then uses HF **server-side copy**
(`CommitOperationCopy`) to materialize every duplicate path from the already-uploaded
blob — no re-upload. Net transfer = the unique blobs only (the 2 weight blobs + the
unique small/tokenizer files).

Run (stop any in-flight `hf upload-large-folder` first):
    HF_HUB_DISABLE_XET=1 uv run python conversion/upload_pplx_embed_dedup.py \
        --repo dokterbob/pplx-embed-coreml \
        --stage output/pplx-embed-coreml-stage
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

# Duplicates at/above this size are materialized via server-side Copy (no re-upload).
# Only the ~1.15 GB weight.bin clears this bar — and it is always LFS, so Copy is safe.
# Smaller duplicates (tokenizer, graph files) are just re-added; that upload is tiny and
# avoids any "Copy a non-LFS file" edge case.
_LFS_MIN = 50 * 1024 * 1024  # 50 MB


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Dedup-minimal upload of the pplx-embed stage")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--stage", default="output/pplx-embed-coreml-stage")
    args = ap.parse_args()

    stage = Path(args.stage).resolve()
    if not stage.is_dir():
        print(f"stage dir not found: {stage}")
        return 1

    from huggingface_hub import (
        HfApi, create_repo, CommitOperationAdd, CommitOperationCopy,
    )

    # Walk the staged tree; group repo paths by content sha (the dedup key).
    print("Hashing staged files (local; finds the unique blobs) …")
    entries: list[tuple[str, Path, str, int]] = []  # (repo_path, local, sha, size)
    for root, dirs, fns in os.walk(stage):
        # Skip dot-dirs (e.g. `.cache/huggingface/` that `hf upload-large-folder`
        # writes into the folder for resume tracking) — HF rejects `.cache/` paths.
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in fns:
            if fn.startswith("."):
                continue
            lp = Path(root) / fn
            rp = lp.relative_to(stage).as_posix()
            entries.append((rp, lp, _sha256(lp), lp.stat().st_size))

    canonical: dict[str, str] = {}        # sha -> first repo_path (the uploaded copy)
    adds: list = []                       # unique content to upload
    copies: list = []                     # dups → server-side copy
    readds: list = []                     # tiny dups → just re-add (cheap)
    for rp, lp, sha, size in sorted(entries, key=lambda e: e[0]):
        if sha not in canonical:
            canonical[sha] = rp
            adds.append(CommitOperationAdd(path_in_repo=rp, path_or_fileobj=str(lp)))
        elif size >= _LFS_MIN:
            copies.append(CommitOperationCopy(src_path_in_repo=canonical[sha], path_in_repo=rp))
        else:
            readds.append(CommitOperationAdd(path_in_repo=rp, path_or_fileobj=str(lp)))

    uniq_gb = sum(e[3] for e in entries if canonical[e[2]] == e[0]) / 1e9
    total_gb = sum(e[3] for e in entries) / 1e9
    print(f"  {len(entries)} files → {len(adds)} unique to upload "
          f"({uniq_gb:.2f} GB transferred) + {len(copies)} server-side copies "
          f"+ {len(readds)} tiny re-adds. (apparent total {total_gb:.1f} GB)")

    token = os.environ.get("HF_TOKEN") or None
    if token is None:
        try:
            print(f"Using cached HF login: {HfApi().whoami().get('name')}")
        except Exception:
            print("ERROR: no HF_TOKEN and no cached login (`huggingface-cli login`).")
            return 1
    api = HfApi(token=token)
    create_repo(args.repo, repo_type="model", exist_ok=True, token=token)

    # Commit 1: every unique blob + the tiny duplicates (these establish the copy sources).
    print(f"\n[1/2] Uploading {len(adds) + len(readds)} unique/small files "
          f"(~{uniq_gb:.2f} GB over the wire) …")
    api.create_commit(args.repo, operations=adds + readds, repo_type="model",
                      commit_message="upload unique blobs + small files (deduped)")

    # Commit 2: server-side copy the large duplicates from their canonical path.
    if copies:
        print(f"[2/2] Server-side copying {len(copies)} duplicate weight blobs "
              f"(no re-upload) …")
        api.create_commit(args.repo, operations=copies, repo_type="model",
                          commit_message="server-side copy deduped weight.bin across buckets")

    print(f"\n✅ Done. https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
