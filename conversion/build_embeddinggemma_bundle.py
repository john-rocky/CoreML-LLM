#!/usr/bin/env python3
"""Build a CoreML bundle for EmbeddingGemma-300M (bidirectional encoder).

Stateless single-forward export — no KV cache, no causal mask. Takes fixed-
length (`--max-seq-len`, default 512) input + pad mask, returns a unit-norm
768-d embedding. Swift can Matryoshka-truncate to 512/256/128 post-hoc and
renormalize.

Usage:
    python conversion/build_embeddinggemma_bundle.py --max-seq-len 512
    python conversion/build_embeddinggemma_bundle.py --max-seq-len 2048 --quantize int4

Why fixed-length (not RangeDim): variable sequence length via ct.RangeDim
forces GPU fallback on ANE (docs/SPEED_8K.md, docs/ANE_OPTIMIZATION_SURVEY.md).
To handle multiple buckets, run this builder several times with different
--max-seq-len values and select the closest at runtime.

Output layout:
    output/embeddinggemma-300m/bundle/
        encoder.mlpackage            # stateless; fp16 or INT4
        model_config.json
        hf_model/tokenizer.json (+ tokenizer_config.json, special_tokens_map.json, ...)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


MODEL_NAME = "embeddinggemma-300m"
HF_REPO = "google/embeddinggemma-300m"


def _resolve_hf_dir(override: str | None, bundle_dir: str) -> str:
    if override:
        return override
    from huggingface_hub import snapshot_download
    local = os.path.join(bundle_dir, "hf_model")
    has_weights = os.path.isdir(local) and (
        any(fn.endswith(".safetensors") for fn in os.listdir(local))
        or any(fn.endswith(".bin") for fn in os.listdir(local))
    )
    if not has_weights:
        print(f"Downloading {HF_REPO} → {local}...")
        snapshot_download(
            HF_REPO,
            local_dir=local,
            allow_patterns=[
                "*.safetensors", "*.bin", "*.json", "tokenizer*",
                "*.txt", "*.model", "*.jinja",
                "1_Pooling/*", "2_Dense/*", "3_Dense/*", "4_Normalize/*",
            ],
        )
    return local


def _load_transformer_config(hf_dir: str) -> dict:
    with open(os.path.join(hf_dir, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def _iter_state_files(hf_dir: str):
    """Yield paths to every safetensors / bin file under hf_dir and its
    SentenceTransformer sub-module directories (2_Dense/, 3_Dense/, ...)."""
    paths: list[str] = []
    for root, _, files in os.walk(hf_dir):
        for fn in files:
            if fn.endswith(".safetensors") or fn == "model.safetensors":
                paths.append(os.path.join(root, fn))
            elif fn.endswith(".bin") and "pytorch_model" in fn:
                paths.append(os.path.join(root, fn))
    return sorted(set(paths))


def _load_state_dicts(hf_dir: str) -> dict[str, torch.Tensor]:
    """Merge every weight file under hf_dir into a single {name: tensor} dict.
    For SentenceTransformer sub-module files (2_Dense/pytorch_model.bin etc.)
    we prefix the module index so collisions don't shadow each other."""
    import safetensors.torch
    merged: dict[str, torch.Tensor] = {}
    for path in _iter_state_files(hf_dir):
        rel = os.path.relpath(path, hf_dir)
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        # If this came from a sub-module (e.g. 2_Dense/pytorch_model.bin),
        # prefix the module dir name so `linear.weight` becomes `2_Dense.linear.weight`.
        subdir = rel.split(os.sep)[0] if os.sep in rel else ""
        if subdir and subdir not in ("", "hf_model"):
            for k, v in state.items():
                merged[f"{subdir}.{k}"] = v
        else:
            for k, v in state.items():
                merged[k] = v
    print(f"  loaded {len(merged)} tensors from {len(_iter_state_files(hf_dir))} files")
    return merged


def _map_encoder_weight(hf_name: str) -> str | None:
    """Map HF Gemma 3 transformer weight name → local encoder param name.

    EmbeddingGemma's SentenceTransformer snapshot saves the underlying
    Gemma3TextModel weights without the `model.` prefix (e.g., `layers.0.…`,
    `embed_tokens.weight`, `norm.weight`), whereas a standard HF causal LM
    snapshot uses `model.layers.0.…`. Accept both.
    """
    if hf_name.startswith("model."):
        name = hf_name[len("model."):]
    else:
        name = hf_name

    if name == "embed_tokens.weight":
        return "encoder.embed_tokens.weight"
    if name == "norm.weight":
        return "encoder.norm.weight"

    if name.startswith("layers."):
        parts = name.split(".")
        layer_idx = int(parts[1])
        rest = ".".join(parts[2:])

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if rest == f"self_attn.{proj}.weight":
                return f"encoder.layers.{layer_idx}.self_attn.{proj}.weight"
        if rest == "self_attn.q_norm.weight":
            return f"encoder.layers.{layer_idx}.self_attn.q_norm.weight"
        if rest == "self_attn.k_norm.weight":
            return f"encoder.layers.{layer_idx}.self_attn.k_norm.weight"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if rest == f"mlp.{proj}.weight":
                return f"encoder.layers.{layer_idx}.mlp.{proj}.weight"
        for norm_name in (
            "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
        ):
            if rest == f"{norm_name}.weight":
                return f"encoder.layers.{layer_idx}.{norm_name}.weight"

    return None


def _map_dense_weight(hf_name: str) -> tuple[str | None, bool]:
    """Map SentenceTransformer Dense module weights.

    Returns (local_name, is_bias). SentenceTransformer Dense layout:
      2_Dense.linear.weight / 2_Dense.linear.bias  →  dense1.weight / dense1.bias
      3_Dense.linear.weight / 3_Dense.linear.bias  →  dense2.weight / dense2.bias
    """
    for hf_prefix, local in (("2_Dense.linear", "dense1"), ("3_Dense.linear", "dense2")):
        if hf_name == f"{hf_prefix}.weight":
            return f"{local}.weight", False
        if hf_name == f"{hf_prefix}.bias":
            return f"{local}.bias", True
    return None, False


def _copy_into_model(model, weights: dict[str, torch.Tensor]) -> None:
    from ane_ops import MODEL_DTYPE

    loaded = 0
    missing_dense = {"dense1.weight", "dense1.bias", "dense2.weight", "dense2.bias"}

    for hf_name, tensor in weights.items():
        # Try encoder mapping first.
        local = _map_encoder_weight(hf_name)
        if local is None:
            dense_local, _ = _map_dense_weight(hf_name)
            local = dense_local
        if local is None:
            continue

        tensor = tensor.to(MODEL_DTYPE)

        try:
            parts = local.split(".")
            target = model
            for p in parts[:-1]:
                target = getattr(target, p)
            param_name = parts[-1]
            param = getattr(target, param_name)

            # Conv2d stores (out, in, 1, 1) — HF Linear stores (out, in) — unsqueeze.
            if param.dim() == 4 and tensor.dim() == 2:
                tensor = tensor.unsqueeze(-1).unsqueeze(-1)

            if param.shape != tensor.shape:
                print(f"  skip {hf_name}: shape mismatch {param.shape} vs {tensor.shape}")
                continue

            with torch.no_grad():
                param.copy_(tensor)
            loaded += 1
            missing_dense.discard(local)
        except (AttributeError, RuntimeError) as e:
            print(f"  WARN {hf_name} → {local}: {e}")

    if missing_dense:
        print(f"  WARN missing dense weights: {sorted(missing_dense)} — "
              "check that the HF snapshot includes 2_Dense/ and 3_Dense/ subdirectories.")
    print(f"  loaded {loaded} tensors into EmbeddingGemmaModel")


def _detect_dense_dims(weights: dict[str, torch.Tensor]) -> tuple[int, int] | None:
    """Return (dense_intermediate_dim, embed_dim) if dense weights are present."""
    w1 = weights.get("2_Dense.linear.weight")
    w2 = weights.get("3_Dense.linear.weight")
    if w1 is None or w2 is None:
        return None
    # nn.Linear weight: (out, in)
    dense_inter = w1.shape[0]
    embed_dim = w2.shape[0]
    return int(dense_inter), int(embed_dim)


def build_bundle(
    model_path: str,
    output_dir: str,
    quantize: str | None = None,
    max_seq_len: int = 512,
) -> None:
    """Entry point reused from convert.py's dispatch path."""
    import coremltools as ct

    from models.embeddinggemma import EmbeddingGemmaModel
    from models.gemma3_encoder import EncoderConfig

    os.makedirs(output_dir, exist_ok=True)

    # Treat the model_path as either the bundle's hf_model/ or a generic HF dir.
    hf_dir = model_path if os.path.isdir(model_path) else _resolve_hf_dir(None, output_dir)

    print(f"\n[1/4] Reading config + loading weights from {hf_dir}")
    text_cfg = _load_transformer_config(hf_dir)
    text_cfg["max_seq_len"] = max_seq_len
    encoder_config = EncoderConfig(**text_cfg)
    weights = _load_state_dicts(hf_dir)

    # Prefer dims detected from actual weights (handles config drift).
    detected = _detect_dense_dims(weights)
    if detected is not None:
        dense_inter, embed_dim = detected
        print(f"  detected dense dims: {encoder_config.hidden_size} → {dense_inter} → {embed_dim}")
    else:
        dense_inter = EmbeddingGemmaModel.DENSE_INTERMEDIATE_DIM
        embed_dim = 768
        print(f"  dense weights not found in snapshot; using defaults "
              f"{encoder_config.hidden_size} → {dense_inter} → {embed_dim}. "
              "Check that 2_Dense/ and 3_Dense/ were downloaded.")

    model = EmbeddingGemmaModel(
        encoder_config,
        embed_dim=embed_dim,
        dense_intermediate_dim=dense_inter,
    )
    model.eval()
    _copy_into_model(model, weights)

    print("\n[2/4] Tracing model")
    sample_ids = torch.zeros((1, max_seq_len), dtype=torch.int32)
    sample_mask = torch.ones((1, max_seq_len), dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(model, (sample_ids, sample_mask))

    print("\n[3/4] Converting to CoreML (fp32 activations, iOS 18, stateless)")
    # Why fp32 (not the default fp16): EmbeddingGemma's residual stream grows
    # past fp16 max (~65504) by layer 7 (no `layer_scalar` to dampen it like
    # Gemma 4 has). With pure fp16 the encoder produces all-zero embeddings on
    # ANE and NaN on CPU. fp32 activations keep the math correct at the cost
    # of a doubled-size .mlpackage and lower ANE residency. Decoder output
    # (FunctionGemma) tolerates fp16 because argmax discretizes away
    # saturation; an embedding vector cannot.
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_seq_len), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="embedding", dtype=np.float16),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT32,
    )

    if quantize == "int4":
        op_config = ct.optimize.coreml.OpPalettizerConfig(
            nbits=4, granularity="per_grouped_channel", group_size=32,
        )
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config)
        print("  Applied int4 palettization (group_size=32)")
    elif quantize == "int8":
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8",
        )
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)
        print("  Applied int8 linear quantization")

    pkg = os.path.join(output_dir, "encoder.mlpackage")
    if os.path.exists(pkg):
        import shutil
        shutil.rmtree(pkg)
    mlmodel.save(pkg)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(pkg)
        for f in fns
    ) / 1024 / 1024
    print(f"  saved {pkg} ({size_mb:.1f} MB)")

    print("\n[4/4] Writing model_config.json + copying tokenizer")
    _write_model_config(output_dir, encoder_config, embed_dim, dense_inter,
                        max_seq_len, quantize)
    _copy_tokenizer(hf_dir, output_dir)

    print("\n" + "=" * 60)
    print(f"EmbeddingGemma bundle ready at {output_dir}")
    print("=" * 60)


def _write_model_config(bundle_dir: str, cfg: "EncoderConfig",
                        embed_dim: int, dense_inter: int,
                        max_seq_len: int, quantize: str | None) -> None:
    """Canonical I/O contract + metadata for downstream loaders."""
    out = {
        "model_name": MODEL_NAME,
        "architecture": "gemma3-embedding",
        "tokenizer_repo": HF_REPO,
        "parts": {"encoder": "encoder.mlpackage"},
        "io_contract": {
            "inputs": {
                "input_ids": {"shape": [1, max_seq_len], "dtype": "int32"},
                "attention_mask": {"shape": [1, max_seq_len], "dtype": "fp16",
                                    "doc": "1.0 for valid tokens, 0.0 for pad"},
            },
            "outputs": {
                "embedding": {"shape": [1, embed_dim], "dtype": "fp16",
                              "doc": "L2-normalized; Matryoshka-truncate the last dim"},
            },
        },
        # Encoder config.
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "max_seq_len": max_seq_len,
        "sliding_window": cfg.sliding_window,
        "sliding_window_pattern": cfg.sliding_window_pattern,
        "layer_types": cfg.layer_types,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta_global": cfg.rope_theta,
        "rope_theta_local": cfg.rope_local_base_freq,
        # Embedding head.
        "embed_dim": embed_dim,
        "dense_intermediate_dim": dense_inter,
        "pooling": "mean",
        "normalize": "l2",
        "matryoshka_dims": [embed_dim, 512, 256, 128],
        # EmbeddingGemma's published task-prefix convention (Hugging Face model card).
        # Swift runtime should prepend the relevant prefix to the input text
        # before tokenization.
        "task_prefixes": {
            "retrieval_query": "task: search result | query: ",
            "retrieval_document": "title: none | text: ",
            "classification": "task: classification | query: ",
            "clustering": "task: clustering | query: ",
            "similarity": "task: sentence similarity | query: ",
            "code_retrieval": "task: code retrieval | query: ",
            "question_answering": "task: question answering | query: ",
            "fact_verification": "task: fact checking | query: ",
        },
        "quantization": quantize or "fp16",
        "compute_units": "CPU_AND_NE",
    }
    path = os.path.join(bundle_dir, "model_config.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")


def _copy_tokenizer(hf_dir: str, bundle_dir: str) -> None:
    import shutil
    dst = os.path.join(bundle_dir, "hf_model")
    os.makedirs(dst, exist_ok=True)
    if os.path.abspath(hf_dir) == os.path.abspath(dst):
        # hf_model was downloaded in-place; nothing to copy.
        print("  tokenizer already under hf_model/")
        return
    patterns = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "tokenizer.model", "special_tokens_map.json",
                "chat_template.jinja", "generation_config.json")
    copied = []
    for name in os.listdir(hf_dir):
        if name in patterns or name.startswith("tokenizer"):
            shutil.copy2(os.path.join(hf_dir, name), os.path.join(dst, name))
            copied.append(name)
    print(f"  copied tokenizer files: {sorted(copied)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build CoreML bundle for EmbeddingGemma-300M (bidirectional encoder)"
    )
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Fixed trace-time sequence length (default: 512)")
    parser.add_argument("--quantize", type=str, default="none",
                        choices=["int4", "int8", "none"],
                        help="Quantization mode (default: none → fp16)")
    parser.add_argument("--hf-dir", type=str, default=None,
                        help="Override HF model directory (skip auto-download)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output bundle directory (default: ../output/embeddinggemma-300m/bundle)")
    args = parser.parse_args()

    output = args.output or os.path.join(ROOT, "..", "output", MODEL_NAME, "bundle")
    os.makedirs(output, exist_ok=True)
    quantize = None if args.quantize == "none" else args.quantize

    hf_dir = _resolve_hf_dir(args.hf_dir, output)
    build_bundle(
        model_path=hf_dir,
        output_dir=output,
        quantize=quantize,
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
