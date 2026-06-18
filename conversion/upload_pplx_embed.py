#!/usr/bin/env python3
"""Publish the prebuilt pplx-embed CoreML buckets to a single HuggingFace repo.

End users should **download** the finished `.mlpackage` buckets, not regenerate them
(conversion needs the toolkit + minutes per bucket). This script mirrors the local
bundle layout into one HF repo with **per-bucket subfolders**, so a consumer pulls only
the bucket(s) they need (the Swift downloader fetches an explicit file list — see
`PplxEmbed.load(repo:buckets:)` / `Gemma3BundleDownloader`).

Single-repo rationale (see docs/PPLX_EMBED.md): the repo convention is one HF repo per
model family with subfolders; each bucket `.mlpackage` embeds the same ~1.1 GB weights
but they are **byte-identical across buckets** (bucket size only changes the traced shape
+ RoPE length), so HF content-addressed LFS stores the blob once — several repos would
not save storage.

Repo layout (target `<account>/pplx-embed-coreml`):
    L512-int8/   L1024-int8/   …   L8192-int8/   dyn8192-int8/   (plain)
    context/L512-int8/  …                                        (context variant)
    manifest.json   README.md
Each bucket subfolder mirrors the local bundle: `encoder.mlpackage/` (or `.mlmodelc/`),
`model_config.json`, `hf_model/` tokenizer json. The upstream `hf_model/*.safetensors`
are excluded (ship only the tokenizer json — matches the other CoreML repos).

This script does NOT upload directly. It compiles (optionally), stages a clean repo tree
(symlinks → manifest.json + README.md), ensures the repo exists, and prints the resumable
`hf upload-large-folder` command for you to run — that uploader is parallel, xet-accelerated,
shows realtime progress, and resumes if interrupted (re-run the same command).

Usage:
    # compile + ship both formats, plain + context:
    uv run python conversion/upload_pplx_embed.py --repo <account>/pplx-embed-coreml \
        --plain-dir output/pplx-embed --context-dir output/pplx-embed-context --compile
    # then run the printed command, e.g.:
    hf upload-large-folder <account>/pplx-embed-coreml output/pplx-embed-coreml-stage --repo-type=model
    # restrict to specific buckets:
    uv run python conversion/upload_pplx_embed.py --repo <account>/pplx-embed-coreml \
        --plain-dir output/pplx-embed --buckets L512-int8 L1024-int8 L2048-int8
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Files inside a bucket dir to never upload (the re-conversion-only weights).
_EXCLUDE_SUFFIXES = (".safetensors",)


def _is_shippable_bucket(bucket_dir: Path) -> bool:
    """A bucket ships iff its model_config.json is an int8-output, fp16-weight model.

    Mirrors PplxEmbed.swift's parseBucket filter so we publish exactly the set the
    Swift runtime will load (skips pooled_fp16 fidelity bundles + weight-quant probes).
    """
    cfg = bucket_dir / "model_config.json"
    if not cfg.exists():
        return False
    try:
        j = json.loads(cfg.read_text())
    except Exception:
        return False
    return (j.get("output_mode") == "int8"
            and (j.get("quantization_weights") or "fp16") == "fp16")


def _model_dir_name(bucket_dir: Path) -> str | None:
    """Return 'encoder.mlmodelc' if compiled present, else 'encoder.mlpackage'."""
    if (bucket_dir / "encoder.mlmodelc").is_dir():
        return "encoder.mlmodelc"
    if (bucket_dir / "encoder.mlpackage").is_dir():
        return "encoder.mlpackage"
    return None


def _compile_bucket(bucket_dir: Path) -> None:
    """Compile encoder.mlpackage → encoder.mlmodelc in-place (skip if present).

    Ship-compiled path: precompiled `.mlmodelc` removes the consumer's first-load
    `MLModel.compileModel` step and matches the repo's other CoreML releases. CoreML
    `.mlmodelc` is loadable across the deployment target's devices (iPhone + Mac).
    """
    import subprocess

    mlmodelc = bucket_dir / "encoder.mlmodelc"
    pkg = bucket_dir / "encoder.mlpackage"
    if mlmodelc.is_dir() or not pkg.is_dir():
        return
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(bucket_dir)],
                   check=True)


def _present_model_dirs(bucket_dir: Path) -> list[str]:
    """Model dirs present in this bucket (both, if shipping both formats)."""
    return [d for d in ("encoder.mlmodelc", "encoder.mlpackage")
            if (bucket_dir / d).is_dir()]


def _bucket_files(bucket_dir: Path) -> list[str]:
    """Relative file paths (POSIX) to ship for one bucket, excluding safetensors.

    Includes every present model dir (so 'ship both' uploads both encoder.mlmodelc
    and encoder.mlpackage); the Swift client selectively downloads only one format.
    """
    files: list[str] = []
    model_dirs = _present_model_dirs(bucket_dir)
    if not model_dirs:
        return files
    for model_dir in model_dirs:
        for p in sorted((bucket_dir / model_dir).rglob("*")):
            if p.is_file() and not p.name.endswith(_EXCLUDE_SUFFIXES):
                files.append(p.relative_to(bucket_dir).as_posix())
    # model_config.json + tokenizer json (exclude any safetensors defensively).
    if (bucket_dir / "model_config.json").is_file():
        files.append("model_config.json")
    hf = bucket_dir / "hf_model"
    if hf.is_dir():
        for p in sorted(hf.rglob("*")):
            if p.is_file() and not p.name.endswith(_EXCLUDE_SUFFIXES):
                files.append(p.relative_to(bucket_dir).as_posix())
    return files


def _discover_buckets(plain_dir: Path | None, context_dir: Path | None,
                      only: set[str] | None) -> list[tuple[str, Path]]:
    """Return [(repo_subfolder, local_bucket_dir)] for every shippable bucket.

    Plain buckets map to their dirname (e.g. 'L512-int8'); context buckets are
    prefixed with 'context/' (e.g. 'context/L512-int8').
    """
    out: list[tuple[str, Path]] = []
    if plain_dir and plain_dir.is_dir():
        for d in sorted(p for p in plain_dir.iterdir() if p.is_dir()):
            if only and d.name not in only:
                continue
            if _is_shippable_bucket(d):
                out.append((d.name, d))
    if context_dir and context_dir.is_dir():
        for d in sorted(p for p in context_dir.iterdir() if p.is_dir()):
            if only and d.name not in only:
                continue
            if _is_shippable_bucket(d):
                out.append((f"context/{d.name}", d))
    return out


def _build_manifest(buckets: list[tuple[str, Path]], repo: str) -> dict:
    # Size-only manifest: the Swift `load(repo:)` derives download globs from each
    # bucket's subfolder + formats, and the HF Swift Hub client's content-addressed
    # cache dedups the byte-identical weight.bin by etag on download — so no per-file
    # sha is needed here (and staging stays instant, no ~14 GB hash).
    base_url = f"https://huggingface.co/{repo}/resolve/main"
    entries = []
    total = 0
    for subfolder, bucket_dir in buckets:
        cfg = json.loads((bucket_dir / "model_config.json").read_text())
        files_meta = []
        for rel in _bucket_files(bucket_dir):
            p = bucket_dir / rel
            size = p.stat().st_size
            total += size
            repo_path = f"{subfolder}/{rel}"
            files_meta.append({
                "path": repo_path,
                "url": f"{base_url}/{repo_path}",
                "size_bytes": size,
            })
        formats = [d.split(".", 1)[1] for d in _present_model_dirs(bucket_dir)]
        entries.append({
            "subfolder": subfolder,
            "variant": cfg.get("variant", "plain"),
            "bucket": cfg.get("bucket"),
            "dynamic": bool(cfg.get("dynamic", False)),
            "dynamic_upper": cfg.get("dynamic_upper", 0),
            "max_seq_len": cfg.get("max_seq_len"),
            "norm_impl": cfg.get("norm_impl", "ane_cat"),
            "formats": formats,   # e.g. ["mlmodelc", "mlpackage"]; Swift picks one
            "files": files_meta,
        })
    # Aggregate of the formats actually shipped: "both", "mlmodelc", or "mlpackage".
    all_formats = {f for _s, d in buckets for f in
                   (x.split(".", 1)[1] for x in _present_model_dirs(d))}
    fmt = "both" if all_formats == {"mlmodelc", "mlpackage"} else next(iter(all_formats), "mlpackage")
    return {
        "model_id": Path(repo).name,
        "repo": repo,
        "format": fmt,
        "buckets": entries,
        "total_size": total,
    }


def _readme(repo: str, manifest: dict) -> str:
    rows = []
    for b in manifest["buckets"]:
        n = sum(f["size_bytes"] for f in b["files"])
        kind = "dynamic GPU catch-all" if b["dynamic"] else "fixed ANE bucket"
        rows.append(f"| `{b['subfolder']}/` | {b['variant']} | {b['bucket']} | "
                    f"{kind} | {n / 1e9:.2f} GB |")
    table = "\n".join(rows)
    return f"""\
---
language: multilingual
license: apache-2.0
base_model: perplexity-ai/pplx-embed-v1-0.6b
tags:
  - coreml
  - apple-neural-engine
  - qwen3
  - sentence-embedding
  - on-device
library_name: coreml
---

# pplx-embed for Apple CoreML (ANE-optimized)

CoreML conversion of Perplexity's
[`pplx-embed-v1-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b)
(a bidirectional Qwen3-0.6B encoder → masked-mean pool → tanh-int8 head) produced with
the [CoreML-LLM](https://github.com/john-rocky/CoreML-LLM) pipeline. Targets macOS 26.

Each subfolder is a **fixed-shape sequence-length bucket** that stays resident on the
Apple Neural Engine (flexible shapes force CPU fallback). At runtime the Swift package
pads each input to the smallest bucket that fits; inputs longer than the largest fixed
bucket fall through to the `dyn*-int8/` flexible GPU catch-all. The encoder uses native
RMSNorm and a single fixed RoPE table — the ANE-fastest path on M4 Max / macOS 26.

## Buckets in this repo

| Subfolder | Variant | Bucket (L) | Kind | Size |
|---|---|---|---|---|
{table}

The encoder `weight.bin` is **byte-identical across every bucket** (a single fixed-size
RoPE table makes the weights independent of bucket length). So HF stores the weight blob
**once**, and the HF content-addressed cache fetches it **once by etag** on download —
pulling several buckets costs ~1.15 GB total, not ~1.15 GB × N.

## Use it

Via the [CoreML-LLM Swift package](https://github.com/john-rocky/CoreML-LLM). It uses the
HF Swift Hub client, so only the buckets you request are downloaded and the shared weight
is fetched once into the content-addressed cache:

```swift
import CoreMLLLM
let embedder = try await PplxEmbed.load(
    repo: "{repo}",
    buckets: [512, 1024, 2048])       // shared HF cache; weight fetched once by etag
let vecs = try embedder.embed(["On-device embeddings", "Bonjour le monde"])  // [[Int8]]
```

Each bucket is published in both `.mlpackage` and precompiled `.mlmodelc`; pass
`preferCompiled: false` for the portable package. Or download the bundle directory
yourself and load it with `load(bundleDir:)`.

## I/O contract (per bucket `model_config.json`)

- `input_ids (1, L) int32`, `attention_mask (1, L) fp16` (1.0 valid, 0.0 pad)
- `embedding (1, 1024) int8` — `clamp(round(tanh(x)*127), -128, 127)`; derive
  `binary`/`ubinary` from the int8 sign (see `PplxEmbed`).

## License

Inherits the base model's [license](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b).
"""


def _stage_repo_tree(buckets: list[tuple[str, Path]], stage_dir: Path,
                     manifest: dict, repo: str) -> None:
    """Build a clean tree mirroring the repo layout via symlinks (no copy).

    `hf upload-large-folder` mirrors a local folder to the repo root, so we stage one:
    each shippable file is **hardlinked** to its real source under
    `stage_dir/<subfolder>/<rel>`, plus manifest.json + README.md. Hardlinks are
    indistinguishable from real files to the uploader (no symlink-following caveat) and
    cost no extra disk (same inode); we fall back to symlink then copy if hardlinking
    fails (e.g. cross-filesystem). Re-running rebuilds the tree from scratch.
    """
    import shutil
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for subfolder, bucket_dir in buckets:
        for rel in _bucket_files(bucket_dir):
            src = (bucket_dir / rel).resolve()
            dst = stage_dir / subfolder / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)            # hardlink — zero copy, real-file semantics
            except OSError:
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
    (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (stage_dir / "README.md").write_text(_readme(repo, manifest))


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload pplx-embed CoreML buckets to HF")
    ap.add_argument("--repo", required=True, help="Target HF repo id (e.g. acct/pplx-embed-coreml)")
    ap.add_argument("--plain-dir", default="output/pplx-embed",
                    help="Local dir of plain buckets (Lxxxx-int8/, dynNNNN-int8/)")
    ap.add_argument("--context-dir", default=None,
                    help="Local dir of context buckets (uploaded under context/)")
    ap.add_argument("--buckets", nargs="*", default=None,
                    help="Restrict to these bucket dir names (e.g. L512-int8 L1024-int8)")
    ap.add_argument("--compile", action="store_true",
                    help="Compile each bucket's encoder.mlpackage → encoder.mlmodelc and ship "
                         "BOTH formats (no on-device compile for consumers; .mlmodelc+.mlpackage "
                         "share a bucket's weight.bin so this does not double the payload).")
    ap.add_argument("--stage-dir", default=None,
                    help="Where to build the upload tree (default: <plain-dir>/../pplx-embed-coreml-stage)")
    ap.add_argument("--no-create-repo", action="store_true",
                    help="Don't create the HF repo (the upload command will).")
    args = ap.parse_args()

    plain_dir = Path(args.plain_dir).resolve() if args.plain_dir else None
    context_dir = Path(args.context_dir).resolve() if args.context_dir else None
    only = set(args.buckets) if args.buckets else None

    buckets = _discover_buckets(plain_dir, context_dir, only)
    if not buckets:
        print("No shippable buckets found (need int8-output, fp16-weight model_config.json).")
        return 1

    if args.compile:
        print("Compiling buckets → encoder.mlmodelc …")
        for _subfolder, d in buckets:
            _compile_bucket(d)

    print(f"Discovered {len(buckets)} bucket(s) for {args.repo}:")
    for subfolder, d in buckets:
        n = len(_bucket_files(d))
        print(f"  {subfolder}  ({n} files, {_model_dir_name(d)})  ← {d}")

    manifest = _build_manifest(buckets, args.repo)
    total_gb = manifest["total_size"] / 1e9
    print(f"\nTotal payload (pre-LFS-dedup): {total_gb:.2f} GB across "
          f"{sum(len(b['files']) for b in manifest['buckets'])} files")

    # Stage a clean tree (symlinks) mirroring the repo + manifest.json + README.md.
    out_base = plain_dir.parent if plain_dir else Path.cwd()
    stage_dir = Path(args.stage_dir).resolve() if args.stage_dir \
        else (out_base / "pplx-embed-coreml-stage")
    _stage_repo_tree(buckets, stage_dir, manifest, args.repo)
    print(f"\nStaged repo tree → {stage_dir}  (symlinks + manifest.json + README.md)")

    # Ensure the repo exists so the resumable uploader can push straight to it.
    token = os.environ.get("HF_TOKEN") or None
    from huggingface_hub import HfApi, create_repo
    if token is None:
        try:
            print(f"Using cached HF login: {HfApi().whoami().get('name')}")
        except Exception:
            print("NOTE: no HF_TOKEN and no cached login — run `huggingface-cli login` "
                  "before uploading.")
    if not args.no_create_repo:
        try:
            create_repo(args.repo, repo_type="model", exist_ok=True, token=token)
            print(f"Repo ready: https://huggingface.co/{args.repo}")
        except Exception as e:
            print(f"create_repo skipped ({str(e)[:80]}) — the upload command will create it.")

    print("\nNow run this to upload — resumable, parallel, xet-accelerated, realtime "
          "progress (re-run the SAME command to resume if interrupted):\n")
    print(f"  hf upload-large-folder {args.repo} {stage_dir} --repo-type=model\n")
    print("Weights dedupe across buckets: the encoder uses a single fixed RoPE table, so "
          "every plain bucket (and its .mlmodelc+.mlpackage) shares ONE ~1.15 GB weight.bin; "
          "the context variant is a second blob. HF LFS stores each unique blob once, so the "
          "real upload is ~2 weight blobs regardless of how many buckets you ship. Restrict "
          "buckets by re-running this script with e.g. --buckets L512-int8 L1024-int8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
