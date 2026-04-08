#!/usr/bin/env python3
"""CoreML-LLM model conversion CLI.

Single entry point to download a HuggingFace model and convert it to
CoreML .mlpackage files optimized for Apple Neural Engine.

Usage:
    python convert.py --model qwen2.5-0.5b --output ./output/
    python convert.py --model Qwen/Qwen2.5-0.5B-Instruct --context-length 1024 --quantize int4
    python convert.py --list

Output:
    output/
    ├── embed.mlpackage         # Token embeddings
    ├── transformer.mlpackage   # Transformer blocks (stateful KV cache)
    ├── lmhead.mlpackage        # LM head + argmax
    └── model_config.json       # Metadata for Swift inference engine
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def download_model(repo_id: str, output_dir: str) -> str:
    """Download model from HuggingFace Hub.

    Returns path to the downloaded model directory.
    """
    from huggingface_hub import snapshot_download

    model_dir = os.path.join(output_dir, "hf_model")
    print(f"Downloading {repo_id}...")
    path = snapshot_download(
        repo_id,
        local_dir=model_dir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model"],
    )
    print(f"Downloaded to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace LLMs to CoreML for Apple Neural Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., 'qwen2.5-0.5b') or HuggingFace repo ID",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to a local HuggingFace model directory (skip download)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Max context length (default: model's default)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["int4", "int8", "none"],
        default="int4",
        help="Quantization mode (default: int4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available pre-configured models",
    )

    args = parser.parse_args()

    if args.list:
        from config import list_models
        list_models()
        return

    if not args.model and not args.model_path:
        parser.error("Either --model or --model-path is required")

    # Resolve model config
    from config import MODEL_REGISTRY

    quantize = None if args.quantize == "none" else args.quantize

    if args.model and args.model in MODEL_REGISTRY:
        # Pre-configured model
        reg = MODEL_REGISTRY[args.model]
        hf_repo = reg.hf_repo
        architecture = reg.architecture
        context_length = args.context_length or reg.default_context_length
    elif args.model:
        # Assume it's a HuggingFace repo ID
        hf_repo = args.model
        architecture = _detect_architecture(args.model)
        context_length = args.context_length or 2048
    else:
        hf_repo = None
        architecture = _detect_architecture_from_path(args.model_path)
        context_length = args.context_length or 2048

    # Download or use local path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = download_model(hf_repo, args.output)

    # Load model
    print(f"\nLoading {architecture} model from {model_path}...")
    model_class = _get_model_class(architecture)
    model = model_class.from_pretrained(model_path, context_length=context_length)
    model.eval()

    # Export
    from exporter import CoreMLExporter

    exporter = CoreMLExporter(model)
    exporter.export(args.output, quantize=quantize)

    # Update config with model info
    config_path = os.path.join(args.output, "model_config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["model_name"] = args.model or os.path.basename(model_path)
    config["architecture"] = architecture
    config["tokenizer_repo"] = hf_repo or model_path
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone! Next steps:")
    print(f"  1. Copy {args.output}/*.mlpackage to your Xcode project")
    print(f"  2. Copy {args.output}/model_config.json to your app bundle")
    print(f"  3. Use CoreMLLLM Swift package to run inference")


def _detect_architecture(repo_id: str) -> str:
    """Detect architecture from HuggingFace repo name."""
    repo_lower = repo_id.lower()
    if "qwen2" in repo_lower or "qwen-2" in repo_lower:
        return "qwen2"
    if "qwen3" in repo_lower or "qwen-3" in repo_lower:
        return "qwen3"
    if "gemma-4" in repo_lower or "gemma4" in repo_lower:
        return "gemma4"
    if "llama" in repo_lower or "smollm" in repo_lower:
        return "llama"
    raise ValueError(
        f"Cannot detect architecture for '{repo_id}'. "
        "Use a pre-configured model name (--list) or specify --architecture"
    )


def _detect_architecture_from_path(path: str) -> str:
    """Detect architecture from config.json in model directory."""
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {path}")

    with open(config_path) as f:
        config = json.load(f)

    arch_list = config.get("architectures", [])
    if not arch_list:
        raise ValueError(f"No 'architectures' field in {config_path}")

    arch = arch_list[0].lower()
    if "qwen2" in arch:
        return "qwen2"
    if "qwen3" in arch:
        return "qwen3"
    if "llama" in arch:
        return "llama"

    raise ValueError(f"Unsupported architecture: {arch_list[0]}")


def _get_model_class(architecture: str):
    """Import and return the model class for the given architecture."""
    if architecture == "qwen2":
        from models.qwen2 import Qwen2Model
        return Qwen2Model
    if architecture == "gemma4":
        from models.gemma4 import Gemma4Model
        return Gemma4Model
    raise ValueError(f"Unsupported architecture: {architecture}")


if __name__ == "__main__":
    main()
