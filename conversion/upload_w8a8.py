#!/usr/bin/env python3
"""Upload W8A8 8K models to HuggingFace.

Uploads to a subfolder `w8a8-8k/` within the existing repo,
keeping the original INT4 models untouched.
"""
import os
import shutil
from huggingface_hub import HfApi

REPO_ID = "mlboydaisuke/gemma-4-E2B-coreml"
LOCAL_DIR = "/tmp/w8a8-all-compiled"
HF_PREFIX = "w8a8-8k"  # subfolder in repo

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
                if sz < 1024:
                    continue  # skip tiny metadata files, upload only weights + model.mil
                print(f"  Uploading {repo_path} ({sz/1e6:.0f} MB)")
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


if __name__ == "__main__":
    main()
