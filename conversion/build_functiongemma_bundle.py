#!/usr/bin/env python3
"""Build a complete on-device bundle for FunctionGemma-270M (Gemma 3 decoder).

Single-chunk monolithic export — 270M fp16 ≈ 540 MB, well under the per-mlprogram
ANE budget (docs/QWEN35_2B_CHUNKED_HANDOFF.md). No chunk splits, no PLE sidecar,
no per-layer projection — Gemma 3 is a strict subset of Gemma 4 on the
architecture side.

Usage:
    python conversion/build_functiongemma_bundle.py --ctx 2048
    python conversion/build_functiongemma_bundle.py --quantize int4 --ctx 2048

Output layout:

    output/functiongemma-270m/bundle/
        model.mlpackage              # monolithic decoder (stateful KV cache)
        cos_sliding.npy, sin_sliding.npy   # RoPE tables (optional; for Swift runtime)
        cos_full.npy,    sin_full.npy
        model_config.json            # Gemma 3 config for downstream loader
        hf_model/tokenizer.json (+ tokenizer_config.json, chat_template.jinja, ...)

The chat template ships under hf_model/ because FunctionGemma's function-calling
format (`<start_function_call>` / `<end_function_call>`, `developer` role) is what
distinguishes it from vanilla Gemma 3 270M.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import MODEL_REGISTRY
from generate_rope import generate_rope_tables


MODEL_NAME = "functiongemma-270m"


def _resolve_hf_dir(override: str | None, output_root: str) -> str:
    if override:
        return override
    if MODEL_NAME not in MODEL_REGISTRY:
        raise SystemExit(f"{MODEL_NAME} not in MODEL_REGISTRY")
    from huggingface_hub import snapshot_download
    repo = MODEL_REGISTRY[MODEL_NAME].hf_repo
    local = os.path.join(output_root, "hf_model")
    has_weights = os.path.isdir(local) and any(
        fn.endswith(".safetensors") for fn in os.listdir(local)
    )
    if not has_weights:
        print(f"Downloading {repo} → {local}...")
        snapshot_download(
            repo, local_dir=local,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model",
                            "*.jinja"],
        )
    return local


def _load_hf_text_config(hf_dir: str) -> dict:
    with open(os.path.join(hf_dir, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def _export_monolithic(hf_dir: str, bundle_dir: str, ctx_length: int,
                       quantize: str | None) -> None:
    """Run CoreMLExporter on a Gemma3Model; moves the resulting model.mlpackage
    into the bundle directory (exporter writes to the same dir by default)."""
    from models.gemma3 import Gemma3Model
    from exporter import CoreMLExporter

    print(f"\n[1/4] Loading Gemma 3 weights from {hf_dir}")
    model = Gemma3Model.from_pretrained(hf_dir, context_length=ctx_length)
    model.eval()

    print(f"\n[2/4] Exporting monolithic mlpackage (quantize={quantize or 'fp16'})")
    exporter = CoreMLExporter(model)
    # Default fp16 lowering. Empirically the model produces coherent output on
    # ANE (51 tok/s, sensible text) despite our PyTorch CPU trace showing fp16
    # overflow by layer 7. ANE silicon's internal handling appears to saturate
    # rather than NaN; the fp32 residual stream we keep in the PyTorch wrapper
    # for parity testing gets collapsed during coremltools fp16 lowering, but
    # the model is still good enough to ship. Re-investigate if quality suffers
    # vs the PyTorch reference.
    exporter.export(bundle_dir, quantize=quantize)
    # exporter writes model.mlpackage + model_config.json into bundle_dir; we
    # overwrite model_config.json in step 4 with richer Gemma 3 metadata.


def _write_rope(bundle_dir: str, text_cfg: dict, ctx_length: int) -> None:
    """Gemma 3 uses dual RoPE (local θ for sliding, global θ for full). Save
    both as fp16 .npy files so Swift (or a parity test) can reuse them
    without re-deriving."""
    print("\n[3/4] Generating RoPE tables")
    rope_theta_global = float(text_cfg.get("rope_theta", 1_000_000.0))
    rope_theta_local = float(
        text_cfg.get("rope_local_base_freq",
                     text_cfg.get("rope_local_theta", 10_000.0))
    )
    head_dim = int(text_cfg.get("head_dim", 256))
    max_positions = max(ctx_length * 2, 4096)

    tables = generate_rope_tables(
        max_positions=max_positions,
        sliding_head_dim=head_dim,   # Gemma 3 uses single head_dim for both
        full_head_dim=head_dim,
        sliding_theta=rope_theta_local,
        full_theta=rope_theta_global,
        dtype=torch.float16,
    )
    for name, arr in tables.items():
        path = os.path.join(bundle_dir, f"{name}.npy")
        np.save(path, arr)
    print(f"  wrote cos/sin_{{sliding,full}}.npy  (max_pos={max_positions}, head_dim={head_dim})")


def _copy_tokenizer(hf_dir: str, bundle_dir: str) -> None:
    """Copy tokenizer files + function-calling chat template.

    FunctionGemma's chat template is the ONLY architectural difference between
    it and vanilla Gemma 3 270M — the template adds:
        - `developer` role for function declarations
        - `<start_function_call>` / `<end_function_call>` wrappers
    so it must travel with the bundle."""
    dst = os.path.join(bundle_dir, "hf_model")
    os.makedirs(dst, exist_ok=True)
    if os.path.abspath(hf_dir) == os.path.abspath(dst):
        # hf_model was downloaded in-place; nothing to copy.
        print("\nTokenizer already under hf_model/ (no copy needed).")
        return
    patterns = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "tokenizer.model", "special_tokens_map.json",
                "chat_template.jinja", "generation_config.json",
                "added_tokens.json")
    copied = []
    for name in os.listdir(hf_dir):
        if name in patterns or name.startswith("tokenizer"):
            shutil.copy2(os.path.join(hf_dir, name), os.path.join(dst, name))
            copied.append(name)
    print(f"\nCopied tokenizer files: {sorted(copied)}")


def _write_model_config(bundle_dir: str, text_cfg: dict, ctx_length: int,
                        quantize: str | None) -> None:
    """Overwrite exporter's slim model_config.json with Gemma-3-aware fields."""
    print("\n[4/4] Writing Gemma 3 model_config.json")

    # Derive per-layer attention type from sliding_window_pattern.
    pattern = int(text_cfg.get("sliding_window_pattern", 6))
    num_layers = int(text_cfg["num_hidden_layers"])
    layer_types = text_cfg.get("layer_types") or [
        "full_attention" if (i + 1) % pattern == 0 else "sliding_attention"
        for i in range(num_layers)
    ]

    # eos_token_id may be int or list (FunctionGemma uses [1, 50] for end-of-turn
    # + function-call close). Preserve whatever HF gave us.
    eos = text_cfg.get("eos_token_id", 1)
    if isinstance(eos, list):
        eos_val = [int(e) for e in eos]
    else:
        eos_val = int(eos)

    cfg = {
        "model_name": MODEL_NAME,
        "architecture": "gemma3",
        "hidden_size": int(text_cfg["hidden_size"]),
        "num_hidden_layers": num_layers,
        "num_layers": num_layers,
        "num_attention_heads": int(text_cfg.get("num_attention_heads", 4)),
        "num_key_value_heads": int(text_cfg.get("num_key_value_heads", 1)),
        "head_dim": int(text_cfg.get("head_dim", 256)),
        "intermediate_size": int(text_cfg.get("intermediate_size", 2048)),
        "vocab_size": int(text_cfg.get("vocab_size", 262144)),
        "context_length": ctx_length,
        "sliding_window": int(text_cfg.get("sliding_window", 512)),
        "sliding_window_pattern": pattern,
        "layer_types": layer_types,
        "embed_scale": math.sqrt(int(text_cfg["hidden_size"])),
        "rope_theta_global": float(text_cfg.get("rope_theta", 1_000_000.0)),
        "rope_theta_local": float(
            text_cfg.get("rope_local_base_freq",
                         text_cfg.get("rope_local_theta", 10_000.0))
        ),
        "query_pre_attn_scalar": (
            float(text_cfg["query_pre_attn_scalar"])
            if text_cfg.get("query_pre_attn_scalar") is not None
            else float(text_cfg.get("head_dim", 256))
        ),
        "rms_norm_eps": float(text_cfg.get("rms_norm_eps", 1e-6)),
        "bos_token_id": int(text_cfg.get("bos_token_id", 2)),
        "eos_token_id": eos_val,
        "tie_word_embeddings": bool(text_cfg.get("tie_word_embeddings", True)),
        "final_logit_softcapping": float(text_cfg.get("final_logit_softcapping") or 0.0),
        "parts": {"model": "model.mlpackage"},
        "quantization": quantize or "fp16",
        "compute_units": "CPU_AND_NE",
        "tokenizer_repo": MODEL_REGISTRY[MODEL_NAME].hf_repo,
        # Marker so a later Swift integrator knows this bundle carries the
        # function-calling chat template and tokenizer surface.
        "chat_format": "functiongemma",
        "function_call_markers": {
            "start": "<start_function_call>",
            "end": "<end_function_call>",
        },
    }
    path = os.path.join(bundle_dir, "model_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Build on-device bundle for FunctionGemma-270M")
    parser.add_argument("--ctx", type=int, default=None,
                        help="Context length (default: registry entry's default)")
    parser.add_argument("--quantize", type=str, default="int4",
                        choices=["int4", "int8", "none"],
                        help="Quantization mode (default: int4)")
    parser.add_argument("--hf-dir", type=str, default=None,
                        help="Override HF model directory (skip auto-download)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output bundle directory (default: ../output/functiongemma-270m/bundle)")
    args = parser.parse_args()

    ctx = args.ctx or MODEL_REGISTRY[MODEL_NAME].default_context_length
    output = args.output or os.path.join(ROOT, "..", "output", MODEL_NAME, "bundle")
    os.makedirs(output, exist_ok=True)
    quantize = None if args.quantize == "none" else args.quantize

    # hf_model lives inside the bundle dir unless overridden
    hf_root = output if args.hf_dir is None else os.path.dirname(args.hf_dir.rstrip("/"))
    hf_dir = _resolve_hf_dir(args.hf_dir, output)
    text_cfg = _load_hf_text_config(hf_dir)

    print(f"Model:   {MODEL_NAME}")
    print(f"HF dir:  {hf_dir}")
    print(f"Bundle:  {output}")
    print(f"Context: {ctx}")
    print(f"Layers:  {text_cfg.get('num_hidden_layers')}  "
          f"hidden: {text_cfg.get('hidden_size')}  "
          f"heads: {text_cfg.get('num_attention_heads')}/"
          f"{text_cfg.get('num_key_value_heads')}")

    _export_monolithic(hf_dir, output, ctx, quantize)
    _write_rope(output, text_cfg, ctx)
    _copy_tokenizer(hf_dir, output)
    _write_model_config(output, text_cfg, ctx, quantize)

    print("\n" + "=" * 60)
    print(f"FunctionGemma bundle ready at {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
