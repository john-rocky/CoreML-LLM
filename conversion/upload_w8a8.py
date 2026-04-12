#!/usr/bin/env python3
"""Upload W8A8 8K models to HuggingFace.

Uploads to a subfolder `w8a8-8k/` within the existing repo,
keeping the original INT4 models untouched.

Post-upload, this script HEAD-checks each file the Swift
`ModelDownloader.buildW8A8FileList()` expects and fails loudly if any
are missing. An earlier revision of this script silently skipped files
<1024 bytes and left the HF repo missing chunkN.mlmodelc/coremldata.bin
(required by CoreML), causing the iOS download to 404 mid-stream; see
the commit message history for the incident.
"""
import os
import shutil
import sys
import urllib.request
from huggingface_hub import HfApi

REPO_ID = "mlboydaisuke/gemma-4-E2B-coreml"
LOCAL_DIR = "/tmp/w8a8-all-compiled"
HF_PREFIX = "w8a8-8k"  # subfolder in repo

# Files the Swift client downloads per chunk. Must stay in sync with
# Sources/CoreMLLLM/ModelDownloader.swift :: buildW8A8FileList().
REQUIRED_PER_CHUNK = [
    "coremldata.bin",             # REQUIRED by CoreML; was the historically-missing one
    "model.mil",
    "metadata.json",
    "weights/weight.bin",
    "analytics/coremldata.bin",   # missing file still 404s the Swift download
]
REQUIRED_ROOT = ["model_config.json"]

# Also need: model_config.json, embeddings, RoPE, tokenizer
# These are shared with the base model — upload a model_config.json that marks W8A8
SHARED_DIR = os.path.join(os.path.dirname(__file__), "output/gemma4-mobile")


def main():
    api = HfApi()
    print(f"Uploading W8A8 8K models to {REPO_ID}/{HF_PREFIX}/")

    # Upload chunk mlmodelc files
    for name in ["chunk1", "chunk2", "chunk3", "chunk4"]:
        local_path = os.path.join(LOCAL_DIR, f"{name}.mlmodelc")
        if not os.path.exists(local_path):
            print(f"  SKIP {name} (not found)")
            continue

        # Upload each file in the mlmodelc directory
        for root, dirs, files in os.walk(local_path):
            for f in files:
                local_file = os.path.join(root, f)
                rel_path = os.path.relpath(local_file, LOCAL_DIR)
                repo_path = f"{HF_PREFIX}/{rel_path}"
                sz = os.path.getsize(local_file)
                # Do NOT skip small files. coremldata.bin is 900-1100 bytes but
                # REQUIRED by CoreML to load the mlmodelc. Earlier version of
                # this script skipped files <1024 bytes and produced an HF repo
                # that 404'd during app download; see
                # docs/EAGLE3_INTEGRATION_STATE.md / POST_BENCH_PRIORITIES.md
                # for the incident.
                print(f"  Uploading {repo_path} ({sz/1e6:.2f} MB)")
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=repo_path,
                    repo_id=REPO_ID,
                    repo_type="model",
                )

    # Upload model_config.json with W8A8 marker
    import json
    config_path = os.path.join(SHARED_DIR, "model_config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["quantization"] = "w4a8"
    config["context_length"] = 8192
    config["full_window"] = 0  # 0 = no WFA, standard 8K
    config["variant"] = "w8a8-8k"

    tmp_config = "/tmp/w8a8_model_config.json"
    with open(tmp_config, "w") as f:
        json.dump(config, f, indent=2)

    api.upload_file(
        path_or_fileobj=tmp_config,
        path_in_repo=f"{HF_PREFIX}/model_config.json",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"  Uploaded model_config.json")

    print(f"\nDone! Models at: https://huggingface.co/{REPO_ID}/tree/main/{HF_PREFIX}")
    print(f"\nNote: Shared files (embeddings, RoPE, tokenizer) are in the repo root.")
    print(f"The app should download {HF_PREFIX}/ chunks + root shared files.")

    print("\nVerifying uploaded files with HEAD requests...")
    missing = verify_hf_state()
    if missing:
        print(f"\nFAIL: {len(missing)} required file(s) missing on HF:")
        for m in missing:
            print(f"  - {m}")
        print("\nFix: re-upload the missing files (check the filter in upload_w8a8.py).")
        sys.exit(1)
    print("OK — all files the Swift client expects are reachable on HF.")


def verify_hf_state() -> list[str]:
    """HEAD-check every file buildW8A8FileList expects. Returns list of paths
    that returned 4xx (empty list = all OK)."""
    def head_ok(rel_path: str) -> bool:
        url = f"https://huggingface.co/{REPO_ID}/resolve/main/{HF_PREFIX}/{rel_path}"
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status < 400
        except urllib.error.HTTPError as e:
            return e.code < 400
        except Exception:
            return False

    missing = []
    for chunk in ["chunk1", "chunk2", "chunk3", "chunk4"]:
        for f in REQUIRED_PER_CHUNK:
            p = f"{chunk}.mlmodelc/{f}"
            if not head_ok(p):
                missing.append(p)
    for f in REQUIRED_ROOT:
        if not head_ok(f):
            missing.append(f)
    return missing


if __name__ == "__main__":
    # `python upload_w8a8.py verify` runs the HEAD check in isolation
    # (skips the actual upload). Useful for incident triage.
    if len(sys.argv) == 2 and sys.argv[1] == "verify":
        print(f"Verifying {REPO_ID}/{HF_PREFIX}/ ...")
        missing = verify_hf_state()
        if missing:
            print(f"\nMISSING ({len(missing)}):")
            for m in missing:
                print(f"  {m}")
            sys.exit(1)
        print("OK — all required files present.")
        sys.exit(0)
    main()
