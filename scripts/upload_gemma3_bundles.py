#!/usr/bin/env python3
"""Upload FunctionGemma + EmbeddingGemma CoreML bundles to HuggingFace.

Uploads only the files that the runtime needs: compiled .mlmodelc, tokenizer
files (json + chat template), model_config.json, RoPE tables. Skips the
multi-GB HF safetensors snapshot under hf_model/ — those are only needed for
re-conversion, not inference.

Usage:
    python scripts/upload_gemma3_bundles.py
        --bundle output/functiongemma-270m/bundle
        --repo  mlboydaisuke/functiongemma-270m-coreml
        --kind  functiongemma

    python scripts/upload_gemma3_bundles.py
        --bundle output/embeddinggemma-300m/bundle
        --repo  mlboydaisuke/embeddinggemma-300m-coreml
        --kind  embeddinggemma

The file lists below match the canonical layout consumed by
`Sources/CoreMLLLM/Gemma3BundleDownloader.swift`.
"""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo


# Files (relative to the bundle dir) that are required for inference and
# small enough to ship. Anything not listed here is excluded from the upload.
FUNCTIONGEMMA_FILES = [
    "model.mlmodelc/weights/weight.bin",
    "model.mlmodelc/coremldata.bin",
    "model.mlmodelc/model.mil",
    "model.mlmodelc/metadata.json",
    "model.mlmodelc/analytics/coremldata.bin",
    # Batched prefill T=32 — ~10× faster prompt ingestion on ANE.
    "prefill_t32.mlmodelc/weights/weight.bin",
    "prefill_t32.mlmodelc/coremldata.bin",
    "prefill_t32.mlmodelc/model.mil",
    "prefill_t32.mlmodelc/metadata.json",
    "prefill_t32.mlmodelc/analytics/coremldata.bin",
    "model_config.json",
    "hf_model/tokenizer.json",
    "hf_model/tokenizer_config.json",
    "hf_model/config.json",
    "hf_model/special_tokens_map.json",
    "hf_model/chat_template.jinja",
    "hf_model/added_tokens.json",
    "cos_sliding.npy", "sin_sliding.npy",
    "cos_full.npy",    "sin_full.npy",
]

EMBEDDINGGEMMA_FILES = [
    "encoder.mlmodelc/weights/weight.bin",
    "encoder.mlmodelc/coremldata.bin",
    "encoder.mlmodelc/model.mil",
    "encoder.mlmodelc/metadata.json",
    "encoder.mlmodelc/analytics/coremldata.bin",
    "model_config.json",
    "hf_model/tokenizer.json",
    "hf_model/tokenizer_config.json",
    "hf_model/config.json",
    "hf_model/special_tokens_map.json",
]

README_TEMPLATES = {
    "functiongemma": """\
---
language: en
license: gemma
base_model: google/functiongemma-270m-it
tags:
  - coreml
  - apple-neural-engine
  - gemma3
  - function-calling
  - on-device
library_name: coreml
---

# FunctionGemma-270M for Apple CoreML (ANE-optimized)

CoreML conversion of `google/functiongemma-270m-it` produced with the
[CoreML-LLM](https://github.com/john-rocky/CoreML-LLM) pipeline. Targets
iOS 26 / macOS 26.

## What's in this repo

| File | Notes |
|---|---|
| `model.mlmodelc/` | Compiled stateful decoder (fp16, 840 MB). Drop-in for `MLModel(contentsOf:)` |
| `model_config.json` | Bundle metadata (architecture, dims, function-call markers) |
| `hf_model/` | Tokenizer + chat template (function-calling format) |
| `cos_*.npy`, `sin_*.npy` | Pre-computed RoPE tables (optional) |

## ANE residency

**99.42% on Apple Neural Engine** (1893/1904 dispatched ops, verified via
`MLComputePlan` on macOS 26). The 11 CPU-only ops are unavoidable
input-boundary ops (token gather, argmax, scalar squeeze).

## Use it

Via the [CoreML-LLM Swift package](https://github.com/john-rocky/CoreML-LLM):

```swift
import CoreMLLLM
let bundleURL = try await Gemma3BundleDownloader.download(
    .functionGemma270m, into: appSupportDir)
let fg = try await FunctionGemma.load(bundleURL: bundleURL)
let text = try fg.generate(prompt: "Turn on the flashlight",
                           maxNewTokens: 64)
```

For raw Core ML usage, the model expects the same I/O contract as Gemma 4:
`input_ids (1,1) int32`, `position_ids (1,) int32`, `causal_mask (1,1,1,ctx) fp16`,
`update_mask (1,1,ctx,1) fp16`, with a stateful `kv_cache_0` MLState
(2*L, kv_heads, ctx, head_dim).

## License

Inherits Google's [Gemma terms of use](https://ai.google.dev/gemma/terms).
""",
    "embeddinggemma": """\
---
language: multilingual
license: gemma
base_model: google/embeddinggemma-300m
tags:
  - coreml
  - apple-neural-engine
  - gemma3
  - sentence-embedding
  - on-device
  - matryoshka
library_name: coreml
---

# EmbeddingGemma-300M for Apple CoreML (ANE-optimized)

CoreML conversion of `google/embeddinggemma-300m` produced with the
[CoreML-LLM](https://github.com/john-rocky/CoreML-LLM) pipeline. Targets
iOS 26 / macOS 26.

## What's in this repo

| File | Notes |
|---|---|
| `encoder.mlmodelc/` | Compiled stateless bidirectional encoder (fp16, 588 MB) |
| `model_config.json` | I/O contract, Matryoshka dims, task prefixes |
| `hf_model/` | Tokenizer files |

## ANE residency

**99.80% on Apple Neural Engine** (1950/1954 dispatched ops, verified via
`MLComputePlan` on macOS 26). Achieved by:
- residual-stream rescaling (semantic-preserving fp16 fit)
- fp16-safe L2 normalize (divide by max-abs first to keep `sum(x²)` bounded)
- iOS 26 deployment target

## Use it

Via the [CoreML-LLM Swift package](https://github.com/john-rocky/CoreML-LLM):

```swift
import CoreMLLLM
let bundleURL = try await Gemma3BundleDownloader.download(
    .embeddingGemma300m, into: appSupportDir)
let eg = try await EmbeddingGemma.load(bundleURL: bundleURL)
let vec = try eg.encode(text: "On-device embeddings",
                        task: .retrievalQuery,
                        dim: 768)  // or 512 / 256 / 128 (Matryoshka)
```

I/O contract:
- `input_ids (1, 128) int32`, `attention_mask (1, 128) fp16` (1.0 valid, 0.0 pad)
- `embedding (1, 768) fp16` — L2 unit norm; truncate the trailing dim and
  re-normalize for Matryoshka 512 / 256 / 128

The bundle in this repo is built for `max_seq_len=128`. For longer inputs,
re-run `python conversion/build_embeddinggemma_bundle.py --max-seq-len 2048`.

## Sanity check

```
cosine("cat sat on mat", "feline rested on rug") = 0.7345  (high — similar)
cosine("cat sat on mat", "quantum mechanics")   = 0.4650  (low — different)
```

## License

Inherits Google's [Gemma terms of use](https://ai.google.dev/gemma/terms).
""",
}


def upload_bundle(bundle_dir: str, repo_id: str, kind: str, dry_run: bool) -> None:
    files = FUNCTIONGEMMA_FILES if kind == "functiongemma" else EMBEDDINGGEMMA_FILES
    api = HfApi()

    if not dry_run:
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
            print(f"Repo ready: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"create_repo: {e}")
            sys.exit(1)

    # README first (so the repo card renders before the heavy uploads).
    readme_path = os.path.join(bundle_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(README_TEMPLATES[kind])
    files = ["README.md"] + files

    total_bytes = 0
    for rel in files:
        p = os.path.join(bundle_dir, rel)
        if not os.path.exists(p):
            print(f"  SKIP missing {rel}")
            continue
        sz = os.path.getsize(p)
        total_bytes += sz
        print(f"  {rel}  ({sz / 1024 / 1024:.1f} MB)")

    print(f"Total to upload: {total_bytes / 1024 / 1024:.1f} MB")

    if dry_run:
        print("--dry-run: not uploading.")
        return

    # Stream uploads file-by-file rather than upload_folder so we get progress
    # and can resume on interruption.
    for rel in files:
        p = os.path.join(bundle_dir, rel)
        if not os.path.exists(p):
            continue
        print(f"Uploading {rel}...")
        api.upload_file(
            path_or_fileobj=p,
            path_in_repo=rel,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"upload {rel}",
        )

    print(f"\n✅ Done. https://huggingface.co/{repo_id}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="Local bundle directory")
    p.add_argument("--repo", required=True, help="HF repo id (org/name)")
    p.add_argument("--kind", choices=["functiongemma", "embeddinggemma"], required=True)
    p.add_argument("--dry-run", action="store_true", help="Print plan without uploading")
    args = p.parse_args()

    if not os.path.isdir(args.bundle):
        sys.exit(f"bundle dir not found: {args.bundle}")
    upload_bundle(args.bundle, args.repo, args.kind, args.dry_run)


if __name__ == "__main__":
    main()
