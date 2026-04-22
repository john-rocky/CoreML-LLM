#!/usr/bin/env python3
"""Build a complete on-device bundle for Gemma 4 E2B or E4B.

Produces a directory the iPhone app (Sources/CoreMLLLM/ChunkedEngine.swift,
ModelConfig.swift) can load as-is. Usage:

    python conversion/build_gemma4_bundle.py --model gemma4-e4b --ctx 2048

Output layout (see docs/CONVERSION.md:280 for the canonical list):

    output/<model>/bundle/
        chunk1.mlmodelc/ ... chunk4.mlmodelc/   # compiled on macOS
        embed_tokens_q8.bin + embed_tokens_scales.bin
        embed_tokens_per_layer_q8.bin + embed_tokens_per_layer_scales.bin
        per_layer_projection.bin
        per_layer_norm_weight.bin
        cos_sliding.npy, sin_sliding.npy, cos_full.npy, sin_full.npy
        model_config.json
        hf_model/tokenizer.json (+ tokenizer_config.json, etc.)

The chunks come from build_verify_chunks.py (invoked as a subprocess unless
--skip-chunks). Swift reads all dims from model_config.json — the output is
drop-in for E2B and E4B alike.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import MODEL_REGISTRY
from generate_rope import generate_rope_tables


def _resolve_hf_dir(model_name: str, override: str | None) -> str:
    if override:
        return override
    if model_name not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_name}'. Options: {list(MODEL_REGISTRY)}")
    from huggingface_hub import snapshot_download
    repo = MODEL_REGISTRY[model_name].hf_repo
    local = os.path.join(ROOT, "..", "output", model_name, "hf_model")
    if not os.path.isdir(local) or not any(
        fn.endswith(".safetensors") for fn in os.listdir(local)
    ):
        print(f"Downloading {repo} → {local}...")
        snapshot_download(
            repo, local_dir=local,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model"],
        )
    return local


def _load_hf_text_config(hf_dir: str) -> dict:
    with open(os.path.join(hf_dir, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def _iter_weights(hf_dir: str):
    """Yield (name, tensor) pairs from every safetensors file in hf_dir."""
    files = sorted(f for f in os.listdir(hf_dir) if f.endswith(".safetensors"))
    for fn in files:
        state = safetensors.torch.load_file(os.path.join(hf_dir, fn))
        for name, tensor in state.items():
            yield name, tensor
        del state


LM_PREFIX = "model.language_model."


def _find_weight(hf_dir: str, local_name: str) -> torch.Tensor:
    """Find a named HF tensor by its local (post-prefix) name."""
    target = LM_PREFIX + local_name
    for name, tensor in _iter_weights(hf_dir):
        if name == target:
            return tensor.contiguous()
    raise KeyError(f"weight not found: {target}")


def _quantize_int8_per_row(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric INT8 quantization.

    Returns (int8_data[V, D], scales_fp16[V]).
    Swift dequant (EmbeddingLookup.swift:43): fp = int8 * (scale_fp16 / 127) * embedScale
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight, got {weight.shape}")
    w_fp32 = weight.to(torch.float32)
    row_max = w_fp32.abs().amax(dim=1).clamp_min(1e-8)
    scales = row_max.to(torch.float16)
    # Use the fp16-rounded scale for quant so the stored int8 matches what Swift
    # will dequant with.
    scales_fp32 = scales.to(torch.float32).clamp_min(1e-8)
    q = (w_fp32 * (127.0 / scales_fp32.unsqueeze(1))).round().clamp(-127, 127).to(torch.int8)
    return q.numpy(), scales.numpy().view(np.uint16).view(np.float16)


def _write_int8(path: str, q8: np.ndarray, scales: np.ndarray) -> None:
    # Swift expects: <name>_q8.bin (int8 rows) + <name>_scales.bin (fp16 per row)
    data_path = path + "_q8.bin"
    scales_path = path + "_scales.bin"
    q8.astype(np.int8).tofile(data_path)
    scales.astype(np.float16).tofile(scales_path)
    print(f"  wrote {data_path} ({q8.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  wrote {scales_path} ({scales.nbytes / 1024 / 1024:.2f} MB)")


def _extract_embeddings(hf_dir: str, out_dir: str) -> None:
    """Quantize embed_tokens and embed_tokens_per_layer to INT8 per-row."""
    print("\n[1/4] Extracting INT8 token embeddings")
    w = _find_weight(hf_dir, "embed_tokens.weight")
    print(f"  embed_tokens: {tuple(w.shape)}  (vocab × hidden)")
    q, s = _quantize_int8_per_row(w)
    _write_int8(os.path.join(out_dir, "embed_tokens"), q, s)
    del w, q, s

    print("\n[2/4] Extracting INT8 per-layer embeddings (PLE)")
    w = _find_weight(hf_dir, "embed_tokens_per_layer.weight")
    print(f"  embed_tokens_per_layer: {tuple(w.shape)}  (vocab × nlayers*per_layer_dim)")
    q, s = _quantize_int8_per_row(w)
    _write_int8(os.path.join(out_dir, "embed_tokens_per_layer"), q, s)
    del w, q, s


def _extract_per_layer_projection(hf_dir: str, out_dir: str) -> None:
    """Write per_layer_model_projection.weight and per_layer_projection_norm.weight as fp16 blobs."""
    print("\n[3/4] Extracting per-layer projection + norm weights")
    proj = _find_weight(hf_dir, "per_layer_model_projection.weight")
    proj_fp16 = proj.to(torch.float16).contiguous().numpy()
    proj_path = os.path.join(out_dir, "per_layer_projection.bin")
    proj_fp16.tofile(proj_path)
    print(f"  per_layer_projection: {tuple(proj.shape)}  ({proj_fp16.nbytes / 1024 / 1024:.1f} MB)")

    try:
        norm = _find_weight(hf_dir, "per_layer_projection_norm.weight")
        norm_fp16 = norm.to(torch.float16).contiguous().numpy()
        norm_path = os.path.join(out_dir, "per_layer_norm_weight.bin")
        norm_fp16.tofile(norm_path)
        print(f"  per_layer_norm_weight: {tuple(norm.shape)}  ({norm_fp16.nbytes} B)")
    except KeyError:
        print("  per_layer_projection_norm.weight absent — skipping optional blob")


def _write_rope(out_dir: str, text_cfg: dict, ctx_length: int) -> None:
    print("\n[4/4] Generating RoPE tables")
    rope = text_cfg.get("rope_parameters", {})
    s_rope = rope.get("sliding_attention", {})
    f_rope = rope.get("full_attention", {})
    tables = generate_rope_tables(
        max_positions=max(ctx_length * 2, 4096),
        sliding_head_dim=text_cfg.get("head_dim", 256),
        full_head_dim=text_cfg.get("global_head_dim", 512),
        sliding_theta=float(s_rope.get("rope_theta", 10000.0)),
        full_theta=float(f_rope.get("rope_theta", 1_000_000.0)),
        dtype=torch.float16,
    )
    for name, arr in tables.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)
    print(f"  wrote cos/sin_{{sliding,full}}.npy (max_pos={max(ctx_length * 2, 4096)})")


def _copy_tokenizer(hf_dir: str, out_dir: str) -> None:
    dst = os.path.join(out_dir, "hf_model")
    os.makedirs(dst, exist_ok=True)
    # swift-transformers' AutoTokenizer.from(modelFolder:) requires config.json
    # alongside tokenizer.json to dispatch on model_type; without it the app
    # raises "required configuration file missing: config.json" at load.
    patterns = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "tokenizer.model", "special_tokens_map.json",
                "chat_template.jinja", "generation_config.json")
    copied = []
    for name in os.listdir(hf_dir):
        if name in patterns or name.startswith("tokenizer"):
            shutil.copy2(os.path.join(hf_dir, name), os.path.join(dst, name))
            copied.append(name)
    print(f"\nCopied tokenizer files: {sorted(copied)}")


def _write_model_config(out_dir: str, model_name: str, text_cfg: dict, ctx_length: int) -> None:
    hidden = int(text_cfg["hidden_size"])
    per_layer_dim = int(text_cfg.get("hidden_size_per_layer_input", 256))
    cfg = {
        "model_name": model_name,
        "architecture": "gemma4",
        "hidden_size": hidden,
        "num_hidden_layers": int(text_cfg["num_hidden_layers"]),
        "num_attention_heads": int(text_cfg.get("num_attention_heads", 8)),
        "num_key_value_heads": int(text_cfg.get("num_key_value_heads", 1)),
        "head_dim": int(text_cfg.get("head_dim", 256)),
        "global_head_dim": int(text_cfg.get("global_head_dim", 512)),
        "vocab_size": int(text_cfg.get("vocab_size", 262144)),
        "context_length": ctx_length,
        "sliding_window": int(text_cfg.get("sliding_window", 512)),
        "per_layer_dim": per_layer_dim,
        "num_layers": int(text_cfg["num_hidden_layers"]),
        "embed_scale": math.sqrt(hidden),
        "per_layer_embed_scale": math.sqrt(per_layer_dim),
        "per_layer_model_projection_scale": 1.0 / math.sqrt(hidden),
        "per_layer_input_scale": 1.0 / math.sqrt(2.0),
        "rms_norm_eps": float(text_cfg.get("rms_norm_eps", 1e-6)),
        "bos_token_id": int(text_cfg.get("bos_token_id", 2)),
        "eos_token_id": int(text_cfg.get("eos_token_id", 1)),
        "final_logit_softcapping": float(text_cfg.get("final_logit_softcapping", 30.0)),
        "quantization": "int4",
        "compute_units": "CPU_AND_NE",
        "tokenizer_repo": MODEL_REGISTRY[model_name].hf_repo if model_name in MODEL_REGISTRY else "",
    }
    path = os.path.join(out_dir, "model_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {path}")


def _run_chunks_build(model_name: str, ctx: int, out_dir: str,
                      hf_dir: str | None = None) -> None:
    print(f"\nBuilding 4 chunks via build_verify_chunks.py --model {model_name} --ctx {ctx}")
    cmd = [
        sys.executable, os.path.join(ROOT, "build_verify_chunks.py"),
        "--model", model_name,
        "--ctx", str(ctx),
        "--output", out_dir,
    ]
    if hf_dir:
        cmd.extend(["--hf-dir", hf_dir])
    subprocess.check_call(cmd)


def _compile_mlpackage(pkg: str, mlmodelc: str) -> None:
    """Compile .mlpackage → .mlmodelc via coremltools (macOS-only, text MIL)."""
    import coremltools as ct
    model = ct.models.MLModel(pkg)
    compiled = model.get_compiled_model_path()
    if os.path.exists(mlmodelc):
        shutil.rmtree(mlmodelc)
    shutil.copytree(compiled, mlmodelc)
    del model


def _compile_all_chunks(chunks_dir: str, out_dir: str) -> None:
    print("\nCompiling .mlpackage → .mlmodelc (macOS text-MIL recipe)")
    for name in sorted(os.listdir(chunks_dir)):
        if not name.endswith(".mlpackage"):
            continue
        pkg = os.path.join(chunks_dir, name)
        mlmodelc_name = name.replace(".mlpackage", ".mlmodelc")
        mlmodelc = os.path.join(out_dir, mlmodelc_name)
        t = time.time()
        _compile_mlpackage(pkg, mlmodelc)
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(mlmodelc)
            for f in fns
        ) / 1024 / 1024
        print(f"  {mlmodelc_name}: {size_mb:.1f} MB (compiled in {time.time()-t:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Build on-device bundle for Gemma 4 E2B/E4B")
    parser.add_argument("--model", type=str, default="gemma4-e4b",
                        help="Model name in MODEL_REGISTRY (gemma4-e2b | gemma4-e4b)")
    parser.add_argument("--ctx", type=int, default=None,
                        help="Context length (default: registry entry's default)")
    parser.add_argument("--hf-dir", type=str, default=None,
                        help="Override HF model directory (skip auto-download)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output bundle directory (default: ../output/<model>/bundle)")
    parser.add_argument("--skip-chunks", action="store_true",
                        help="Reuse existing chunks in <output>/chunks/ instead of rebuilding")
    parser.add_argument("--skip-compile", action="store_true",
                        help="Leave .mlpackage uncompiled (Mac-only step)")
    args = parser.parse_args()

    if args.ctx is None:
        args.ctx = (MODEL_REGISTRY[args.model].default_context_length
                    if args.model in MODEL_REGISTRY else 2048)
    if args.output is None:
        args.output = os.path.join(ROOT, "..", "output", args.model, "bundle")
    os.makedirs(args.output, exist_ok=True)

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    text_cfg = _load_hf_text_config(hf_dir)

    print(f"Model:   {args.model}")
    print(f"HF dir:  {hf_dir}")
    print(f"Bundle:  {args.output}")
    print(f"Context: {args.ctx}")
    print(f"Layers:  {text_cfg['num_hidden_layers']}  hidden: {text_cfg['hidden_size']}  "
          f"kv_heads: {text_cfg.get('num_key_value_heads', 1)}")

    chunks_dir = os.path.join(args.output, "chunks")
    if not args.skip_chunks:
        _run_chunks_build(args.model, args.ctx, chunks_dir, hf_dir=hf_dir)
    else:
        if not os.path.isdir(chunks_dir):
            raise SystemExit(f"--skip-chunks set but {chunks_dir} does not exist")

    if not args.skip_compile:
        _compile_all_chunks(chunks_dir, args.output)
    else:
        print("\n--skip-compile set; leaving .mlpackage files in place.")

    _extract_embeddings(hf_dir, args.output)
    _extract_per_layer_projection(hf_dir, args.output)
    _write_rope(args.output, text_cfg, args.ctx)
    _copy_tokenizer(hf_dir, args.output)
    _write_model_config(args.output, args.model, text_cfg, args.ctx)

    print("\n" + "=" * 60)
    print(f"Bundle ready at {args.output}")
    print("=" * 60)
    print("Next: USB sideload to iPhone (see docs/USB_MODEL_SIDELOAD.md):")
    print(
        "  xcrun devicectl device copy to \\\n"
        '    --device "$DEVICE" \\\n'
        "    --domain-type appDataContainer \\\n"
        "    --domain-identifier com.example.CoreMLLLMChat \\\n"
        f"    --source {args.output} \\\n"
        f"    --destination Documents/Models/{args.model} \\\n"
        "    --remove-existing-content true"
    )


if __name__ == "__main__":
    main()
