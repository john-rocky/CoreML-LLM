#!/usr/bin/env python3
"""Build and publish the distribution artifact for a Gemma 4 E2B CoreML release.

Fixes the "manifest points to a non-existent zip" problem by producing EITHER:

  (a) A single `gemma4-e2b-coreml.zip` artifact + SHA-256, ready to upload
      to https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml/resolve/main/
  (b) A per-file `manifest.json` that lists every raw file already on HF
      along with its size + etag, so a downloader can skip the zip entirely.

Both paths produce a `manifest.json` with no `TODO` placeholders. Pick
whichever matches how the runtime downloader is wired.

The Swift runtime (ModelDownloader.swift) already supports (b) — it
downloads each mlmodelc component separately. If you prefer to also
ship (a) for a single-click manual download, upload the zip with
`--upload`.

Usage — produce artifact from a local conversion output directory:
    python conversion/build_hf_release.py \\
        --source ./output/gemma4-e2b-final \\
        --mode per-file \\
        --output ./release/manifest.json

Usage — upload as a single zip (slower download but simpler server):
    export HF_TOKEN=hf_...
    python conversion/build_hf_release.py \\
        --source ./output/gemma4-e2b-final \\
        --mode zip \\
        --output ./release/gemma4-e2b-coreml.zip \\
        --upload \\
        --hf-repo mlboydaisuke/gemma-4-E2B-coreml

`hf_hub_download` / `upload_file` from `huggingface_hub` is used. Requires
the token to have *write* access to the target repo.

The resulting manifest.json format:

  {
    "version": "v0.5.1",
    "model_id": "gemma-4-E2B-coreml",
    "mode": "per-file",     // or "zip"
    "files": [
      { "path": "sdpa/swa/chunk1.mlmodelc/weights/weight.bin",
        "url":  "https://huggingface.co/.../weights/weight.bin",
        "size_bytes": 155436864,
        "sha256": "<hex>" },
      ...
    ],
    // OR for --mode zip:
    "archive": {
      "url": "https://huggingface.co/.../gemma4-e2b-coreml.zip",
      "size_bytes": 2700000000,
      "sha256": "<hex>"
    }
  }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def walk_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def make_per_file_manifest(source: Path, repo: str, branch: str = "main",
                            version: str = "v0.5.1") -> dict:
    """Hash every file, produce a manifest pointing to HF raw URLs."""
    base_url = f"https://huggingface.co/{repo}/resolve/{branch}"
    files_meta = []
    total = 0
    for f in walk_files(source):
        rel = f.relative_to(source).as_posix()
        size = f.stat().st_size
        total += size
        files_meta.append({
            "path":        rel,
            "url":         f"{base_url}/{rel}",
            "size_bytes":  size,
            "sha256":      sha256_of(f),
        })
        print(f"  [{size:>12,}]  {rel}")
    return {
        "version":         version,
        "model_id":        Path(repo).name,
        "mode":            "per-file",
        "repo":            repo,
        "branch":          branch,
        "total_size":      total,
        "files":           files_meta,
    }


def make_zip(source: Path, output: Path):
    """Create gemma4-e2b-coreml.zip at `output`."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): output.unlink()
    # Use `zip -r --symlinks` for maximum compatibility with macOS/iOS unzip
    # and to preserve .mlmodelc directory semantics if the user mounts them.
    # Compression level 1 for speed — the binaries are mostly already compressed.
    args = ["zip", "-r", "-1", str(output)]
    for p in sorted(source.iterdir()):
        args.append(p.name)
    print(f"running: zip ... (source={source})")
    subprocess.run(args, cwd=source, check=True)


def make_zip_manifest(zip_path: Path, repo: str, branch: str = "main",
                      version: str = "v0.5.1") -> dict:
    size = zip_path.stat().st_size
    sha = sha256_of(zip_path)
    return {
        "version":     version,
        "model_id":    Path(repo).name,
        "mode":        "zip",
        "repo":        repo,
        "branch":      branch,
        "archive": {
            "url":        f"https://huggingface.co/{repo}/resolve/{branch}/{zip_path.name}",
            "size_bytes": size,
            "sha256":     sha,
        },
    }


def upload_to_hf(local_path: Path, repo: str, path_in_repo: str, token: str):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print(f"uploading {local_path} -> hf://{repo}/{path_in_repo}")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="model",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True,
                    help="Local directory containing the CoreML release files "
                         "(e.g. ./output/gemma4-e2b-final)")
    ap.add_argument("--mode", choices=["per-file", "zip"], default="per-file")
    ap.add_argument("--output", type=str, required=True,
                    help="Output path. For per-file mode, a manifest.json path. "
                         "For zip mode, the zip path.")
    ap.add_argument("--manifest-output", type=str, default=None,
                    help="Where to write manifest.json when --mode=zip "
                         "(default: alongside the zip).")
    ap.add_argument("--hf-repo", type=str, default="mlboydaisuke/gemma-4-E2B-coreml")
    ap.add_argument("--branch", type=str, default="main")
    ap.add_argument("--version", type=str, default="v0.5.1")
    ap.add_argument("--upload", action="store_true",
                    help="Also upload the artifact (and manifest.json) to HF. "
                         "Requires HF_TOKEN env var with write scope to --hf-repo.")
    args = ap.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        print(f"ERROR: --source path does not exist: {source}")
        return 1

    t0 = time.time()
    if args.mode == "per-file":
        print(f"hashing per-file manifest for {source}...")
        manifest = make_per_file_manifest(source, args.hf_repo, args.branch, args.version)
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2))
        total_gb = manifest["total_size"] / 1e9
        print(f"\n  {len(manifest['files'])} files, total {total_gb:.2f} GB")
        print(f"  manifest: {out}")
        if args.upload:
            token = os.environ.get("HF_TOKEN")
            if not token:
                print("ERROR: HF_TOKEN env var required for --upload")
                return 1
            upload_to_hf(out, args.hf_repo, "manifest.json", token)
            print("  uploaded manifest.json -> HF")

    else:   # zip
        zip_out = Path(args.output)
        make_zip(source, zip_out)
        manifest = make_zip_manifest(zip_out, args.hf_repo, args.branch, args.version)
        mf_out = Path(args.manifest_output) if args.manifest_output else zip_out.with_name("manifest.json")
        mf_out.write_text(json.dumps(manifest, indent=2))
        size_gb = manifest["archive"]["size_bytes"] / 1e9
        print(f"\n  zip: {zip_out} ({size_gb:.2f} GB)")
        print(f"  sha256: {manifest['archive']['sha256']}")
        print(f"  manifest: {mf_out}")
        if args.upload:
            token = os.environ.get("HF_TOKEN")
            if not token:
                print("ERROR: HF_TOKEN env var required for --upload")
                return 1
            upload_to_hf(zip_out, args.hf_repo, zip_out.name, token)
            upload_to_hf(mf_out, args.hf_repo, "manifest.json", token)
            print("  uploaded zip + manifest.json -> HF")

    print(f"\ndone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
