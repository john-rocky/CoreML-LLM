#!/usr/bin/env python3
"""Fix the Gemma 4 E2B entry in the coreml-zoo manifest that the Hub app reads.

Problem:
  `https://huggingface.co/mlboydaisuke/coreml-zoo/resolve/main/models.json` has
  a gemma4_e2b entry with:
      files[0].url    = ".../gemma4-e2b-coreml.zip"   ← 404, never uploaded
      files[0].sha256 = "TODO"                         ← placeholder
  so the Hub app (CoreMLModelsApp in the CoreML-Models sample_apps) always fails
  sha256 verification (the only thing it gets back is a 404 HTML body which
  can't possibly match `TODO`).

Two fix modes here:

  --mode zip
      Build `gemma4-e2b-coreml.zip` from a local source directory, upload it to
      `mlboydaisuke/gemma-4-E2B-coreml`, compute its real sha256, patch
      `models.json` so the entry points to the now-real zip, and upload the
      updated manifest to `mlboydaisuke/coreml-zoo`.

  --mode per-file
      Leave the raw files on HF as they are (no zip needed). Replace the single
      `files[0]` entry with multiple per-file entries that list every raw file
      the runtime actually needs (chunk1-4.mlmodelc / prefill / embed tables /
      vision / audio / tokenizer / ...), each with its real size + sha256
      fetched from HF. Upload the updated manifest.

per-file mode is cheaper (no 2.7 GB zip upload, no 2.7 GB zip to maintain), and
the Hub app's DownloadManager handles it natively (FileSpec with no `archive`
field just gets moved to `Paths.modelDir(id)/<name>`). Recommend this unless a
single zip is specifically needed.

Requirements:
    pip install huggingface_hub
    export HF_TOKEN=hf_...   # must have WRITE scope on mlboydaisuke/coreml-zoo
                             # AND mlboydaisuke/gemma-4-E2B-coreml for --mode zip

Usage — per-file (recommended):
    python conversion/fix_coreml_zoo_manifest.py \\
        --mode per-file \\
        --zoo-repo mlboydaisuke/coreml-zoo \\
        --model-repo mlboydaisuke/gemma-4-E2B-coreml \\
        --upload

Usage — zip (needs a local conversion output to zip up):
    python conversion/fix_coreml_zoo_manifest.py \\
        --mode zip \\
        --source ~/path/to/gemma4-e2b-coreml-files \\
        --zoo-repo mlboydaisuke/coreml-zoo \\
        --model-repo mlboydaisuke/gemma-4-E2B-coreml \\
        --upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# Per-file list that matches the Swift runtime's buildHuggingFaceFileList
# in CoreML-LLM's ModelDownloader. Updating this list should stay in lockstep
# with that Swift function to avoid the two diverging.
GEMMA4_E2B_FILE_PATHS = [
    # Decode chunks (sdpa/swa/) — 2K-context shipping model
    "sdpa/swa/chunk1.mlmodelc/weights/weight.bin",
    "sdpa/swa/chunk1.mlmodelc/coremldata.bin",
    "sdpa/swa/chunk1.mlmodelc/model.mil",
    "sdpa/swa/chunk1.mlmodelc/metadata.json",
    "sdpa/swa/chunk1.mlmodelc/analytics/coremldata.bin",
    "sdpa/swa/chunk2.mlmodelc/weights/weight.bin",
    "sdpa/swa/chunk2.mlmodelc/coremldata.bin",
    "sdpa/swa/chunk2.mlmodelc/model.mil",
    "sdpa/swa/chunk2.mlmodelc/metadata.json",
    "sdpa/swa/chunk2.mlmodelc/analytics/coremldata.bin",
    "sdpa/swa/chunk3.mlmodelc/weights/weight.bin",
    "sdpa/swa/chunk3.mlmodelc/coremldata.bin",
    "sdpa/swa/chunk3.mlmodelc/model.mil",
    "sdpa/swa/chunk3.mlmodelc/metadata.json",
    "sdpa/swa/chunk3.mlmodelc/analytics/coremldata.bin",
    "sdpa/swa/chunk4.mlmodelc/weights/weight.bin",
    "sdpa/swa/chunk4.mlmodelc/coremldata.bin",
    "sdpa/swa/chunk4.mlmodelc/model.mil",
    "sdpa/swa/chunk4.mlmodelc/metadata.json",
    "sdpa/swa/chunk4.mlmodelc/analytics/coremldata.bin",
    # Prefill chunks
    "sdpa/prefill/prefill_chunk1.mlmodelc/coremldata.bin",
    "sdpa/prefill/prefill_chunk1.mlmodelc/model.mil",
    "sdpa/prefill/prefill_chunk1.mlmodelc/metadata.json",
    "sdpa/prefill/prefill_chunk1.mlmodelc/analytics/coremldata.bin",
    "sdpa/prefill/prefill_chunk2.mlmodelc/coremldata.bin",
    "sdpa/prefill/prefill_chunk2.mlmodelc/model.mil",
    "sdpa/prefill/prefill_chunk2.mlmodelc/metadata.json",
    "sdpa/prefill/prefill_chunk2.mlmodelc/analytics/coremldata.bin",
    "sdpa/prefill/prefill_chunk3.mlmodelc/coremldata.bin",
    "sdpa/prefill/prefill_chunk3.mlmodelc/model.mil",
    "sdpa/prefill/prefill_chunk3.mlmodelc/metadata.json",
    "sdpa/prefill/prefill_chunk3.mlmodelc/analytics/coremldata.bin",
    "sdpa/prefill/prefill_chunk4.mlmodelc/coremldata.bin",
    "sdpa/prefill/prefill_chunk4.mlmodelc/model.mil",
    "sdpa/prefill/prefill_chunk4.mlmodelc/metadata.json",
    "sdpa/prefill/prefill_chunk4.mlmodelc/analytics/coremldata.bin",
    # Config + tokenizer
    "model_config.json",
    "hf_model/tokenizer.json",
    "hf_model/tokenizer_config.json",
    "hf_model/config.json",
    # Embeddings (PLE)
    "embed_tokens_q8.bin",
    "embed_tokens_scales.bin",
    "embed_tokens_per_layer_q8.bin",
    "embed_tokens_per_layer_scales.bin",
    "per_layer_projection.bin",
    "per_layer_norm_weight.bin",
    # RoPE tables
    "swa/cos_sliding.npy",
    "swa/sin_sliding.npy",
    "swa/cos_full.npy",
    "swa/sin_full.npy",
    # Vision encoder
    "vision.mlmodelc/weights/weight.bin",
    "vision.mlmodelc/coremldata.bin",
    "vision.mlmodelc/model.mil",
    "vision.mlmodelc/metadata.json",
    "vision.mlmodelc/analytics/coremldata.bin",
    # Audio encoder
    "audio.mlmodelc/weights/weight.bin",
    "audio.mlmodelc/coremldata.bin",
    "audio.mlmodelc/model.mil",
    "audio.mlmodelc/metadata.json",
    "audio.mlmodelc/analytics/coremldata.bin",
    "mel_filterbank.bin",
    "audio_config.json",
    "output_proj_weight.npy",
    "output_proj_bias.npy",
    "embed_proj_weight.npy",
]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def fetch_current_manifest(zoo_repo: str) -> dict:
    """Download the current models.json so we can patch just the gemma entry."""
    import urllib.request
    url = f"https://huggingface.co/{zoo_repo}/resolve/main/models.json"
    print(f"fetching current manifest: {url}")
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))


def hf_file_meta(model_repo: str, path: str, api) -> tuple[int, str]:
    """Return (size_bytes, sha256_hex) for a file on HF.

    Prefer the LFS pointer's sha256 when available (common for large blobs);
    fall back to downloading and hashing the file when the file isn't LFS.
    """
    from huggingface_hub import hf_hub_url
    try:
        info = api.repo_info(repo_id=model_repo, files_metadata=True)
        for sib in info.siblings:
            if sib.rfilename == path:
                if getattr(sib, "lfs", None) and sib.lfs and sib.lfs.get("sha256"):
                    return sib.size, sib.lfs["sha256"]
                if sib.size:
                    return sib.size, _hash_hf_file(model_repo, path)
    except Exception as e:
        print(f"  WARN: repo_info failed: {e}")
    # Fallback: download and hash
    return _hash_hf_file_with_size(model_repo, path)


def _hash_hf_file(model_repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download
    with tempfile.TemporaryDirectory() as td:
        local = hf_hub_download(repo_id=model_repo, filename=path, local_dir=td,
                                local_dir_use_symlinks=False)
        return sha256_of(Path(local))


def _hash_hf_file_with_size(model_repo: str, path: str) -> tuple[int, str]:
    from huggingface_hub import hf_hub_download
    with tempfile.TemporaryDirectory() as td:
        local = hf_hub_download(repo_id=model_repo, filename=path, local_dir=td,
                                local_dir_use_symlinks=False)
        return Path(local).stat().st_size, sha256_of(Path(local))


def build_per_file_entries(model_repo: str, paths: list[str], api) -> list[dict]:
    """For each file path, produce a FileSpec-compatible dict with real size+sha256."""
    base = f"https://huggingface.co/{model_repo}/resolve/main"
    entries = []
    missing = []
    for i, p in enumerate(paths):
        print(f"  [{i+1}/{len(paths)}] {p}")
        try:
            size, sha = hf_file_meta(model_repo, p, api)
        except Exception as e:
            print(f"    MISSING: {e}")
            missing.append(p); continue
        # Use only the trailing filename as `name` so DownloadManager writes to
        # `models/<id>/<name>`. Keep the directory structure by including it
        # in the name field (DownloadManager uses appendingPathComponent
        # which handles subpath strings).
        short_name = p.replace("sdpa/swa/", "").replace("sdpa/prefill/", "").replace("hf_model/", "hf_model/").replace("swa/", "")
        entries.append({
            "name":          short_name,
            "url":           f"{base}/{p}",
            "size_bytes":    int(size),
            "sha256":        sha,
            "compute_units": "cpuAndNeuralEngine",
            "kind":          "model",
        })
    if missing:
        print(f"\n  {len(missing)} files missing from HF; skipped them.")
        for p in missing: print(f"    {p}")
    return entries


def build_zip_entry(source: Path, model_repo: str, api, *, upload: bool,
                    token: str | None) -> dict:
    """Build gemma4-e2b-coreml.zip from local source dir, optionally upload,
    return the FileSpec dict."""
    print(f"building gemma4-e2b-coreml.zip from {source}")
    zip_path = source.parent / "gemma4-e2b-coreml.zip"
    if zip_path.exists(): zip_path.unlink()
    args = ["zip", "-r", "-1", str(zip_path)]
    for p in sorted(source.iterdir()): args.append(p.name)
    subprocess.run(args, cwd=source, check=True)
    size = zip_path.stat().st_size
    sha = sha256_of(zip_path)
    print(f"  size: {size:,} bytes, sha256: {sha}")
    if upload:
        if not token: raise RuntimeError("HF_TOKEN required to upload")
        print(f"uploading to hf://{model_repo}/gemma4-e2b-coreml.zip")
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo="gemma4-e2b-coreml.zip",
            repo_id=model_repo, repo_type="model",
        )
    return {
        "name":          "gemma4-e2b-coreml.zip",
        "url":           f"https://huggingface.co/{model_repo}/resolve/main/gemma4-e2b-coreml.zip",
        "archive":       "zip",
        "size_bytes":    int(size),
        "sha256":        sha,
        "compute_units": "cpuAndNeuralEngine",
        "kind":          "model",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["per-file", "zip"], default="per-file")
    ap.add_argument("--source", type=str,
                    help="Local directory with CoreML files (required for --mode zip)")
    ap.add_argument("--zoo-repo", type=str, default="mlboydaisuke/coreml-zoo")
    ap.add_argument("--model-repo", type=str, default="mlboydaisuke/gemma-4-E2B-coreml")
    ap.add_argument("--upload", action="store_true",
                    help="Upload updated models.json (and zip, if --mode=zip) to HF")
    ap.add_argument("--output", type=str, default="./models_patched.json",
                    help="Local copy of the patched manifest")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub")
        return 1

    token = os.environ.get("HF_TOKEN")
    if args.upload and not token:
        print("ERROR: HF_TOKEN env var required for --upload (with write scope)")
        return 1
    api = HfApi(token=token) if token else HfApi()

    manifest = fetch_current_manifest(args.zoo_repo)
    target = next((m for m in manifest["models"] if m["id"] == "gemma4_e2b"), None)
    if target is None:
        print(f"ERROR: no entry with id='gemma4_e2b' in {args.zoo_repo}/models.json")
        return 1
    print(f"found gemma4_e2b entry. before: files[0] = {target['files'][0]['name']} "
          f"(sha256={target['files'][0]['sha256']})")

    if args.mode == "zip":
        if not args.source:
            print("ERROR: --source required for --mode zip")
            return 1
        source = Path(args.source).resolve()
        if not source.exists():
            print(f"ERROR: source path {source} does not exist")
            return 1
        entry = build_zip_entry(source, args.model_repo, api,
                                 upload=args.upload, token=token)
        target["files"] = [entry]
    else:
        print(f"\nfetching metadata for {len(GEMMA4_E2B_FILE_PATHS)} raw files "
              f"from {args.model_repo}...")
        entries = build_per_file_entries(args.model_repo, GEMMA4_E2B_FILE_PATHS, api)
        if not entries:
            print("ERROR: no usable files found on HF; aborting")
            return 1
        target["files"] = entries
        print(f"\nrewrote gemma4_e2b entry with {len(entries)} per-file entries")

    from datetime import date
    manifest["updated_at"] = date.today().isoformat()

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote patched manifest: {out}")

    if args.upload:
        print(f"uploading models.json -> hf://{args.zoo_repo}/models.json")
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo="models.json",
            repo_id=args.zoo_repo, repo_type="model",
        )
        print("✓ uploaded")

    print("\nDone. Hub app should work on next fetch of models.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
