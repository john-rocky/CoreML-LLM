#!/usr/bin/env python3
"""Convert Gemma 4 E2B multimodal model to CoreML.

Produces 2 CoreML models:
  1. vision.mlpackage  — Vision encoder (pixel_values → image_features)
  2. decoder.mlpackage — Text decoder (inputs_embeds → token predictions)

Plus an embedder that runs in Swift to handle:
  - Token embedding lookup
  - Per-layer embedding computation
  - Image feature injection at placeholder positions

Usage:
    python convert_gemma4_multimodal.py --output ./output/gemma4-multimodal
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import coremltools as ct
import numpy as np
import torch
from transformers import Gemma4ForConditionalGeneration

from models.gemma4 import Gemma4Model
from models.gemma4_decoder import Gemma4DecoderWrapper
from models.gemma4_vision import save_vision_weights


def main():
    parser = argparse.ArgumentParser(description="Convert Gemma 4 E2B multimodal to CoreML")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to local HF model directory")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--quantize", type=str, default="int4", choices=["int4", "int8", "none"])
    parser.add_argument("--output", type=str, default="./output/gemma4-multimodal")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    quantize = None if args.quantize == "none" else args.quantize

    # Download/load HF model
    model_path = args.model_path
    if model_path is None:
        from huggingface_hub import snapshot_download
        model_path = os.path.join(args.output, "hf_model")
        print("Downloading google/gemma-4-E2B-it...")
        snapshot_download(
            "google/gemma-4-E2B-it",
            local_dir=model_path,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model"],
        )

    # ============================================================
    # Part 1: Vision Encoder
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 1/2: Vision Encoder")
    print("=" * 60)

    print("Loading full HF model for vision weights extraction...")
    hf_model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.float16, device_map="cpu"
    )
    hf_model.eval()

    # Save vision weights (vision encoder runs in PyTorch due to dynamic masking)
    save_vision_weights(hf_model, args.output)

    # Free the full model to save memory before decoder conversion
    del hf_model
    import gc
    gc.collect()

    # ============================================================
    # Part 2: Text Decoder (accepts inputs_embeds)
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 2/2: Text Decoder")
    print("=" * 60)

    # Load our custom model for the decoder
    print("Loading text decoder weights...")
    text_model = Gemma4Model.from_pretrained(model_path, context_length=args.context_length)
    text_model.eval()

    wrapper = Gemma4DecoderWrapper(text_model)
    wrapper.eval()

    ctx = args.context_length
    hidden_size = text_model.config.hidden_size
    num_layers = text_model.config.num_hidden_layers
    per_layer_dim = text_model.config.hidden_size_per_layer_input
    total_per_layer = num_layers * per_layer_dim

    # Sample inputs
    sample_embeds = torch.zeros((1, 1, hidden_size), dtype=torch.float16)
    sample_per_layer = torch.zeros((1, 1, total_per_layer), dtype=torch.float16)
    sample_pos = torch.zeros((1,), dtype=torch.int32)
    sample_mask = torch.zeros((1, 1, 1, ctx), dtype=torch.float16)
    sample_umask = torch.zeros((1, 1, ctx, 1), dtype=torch.float16)
    sample_umask[0, 0, 0, 0] = 1.0

    with torch.no_grad():
        wrapper.kv_cache_0.zero_()

    print("Tracing decoder...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (sample_embeds, sample_per_layer, sample_pos, sample_mask, sample_umask),
        )

    cache_shape = tuple(wrapper.kv_cache_0.shape)

    print("Converting decoder to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, 1, hidden_size), dtype=np.float16),
            ct.TensorType(name="per_layer_input", shape=(1, 1, total_per_layer), dtype=np.float16),
            ct.TensorType(name="position_ids", shape=(1,), dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=(1, 1, 1, ctx), dtype=np.float16),
            ct.TensorType(name="update_mask", shape=(1, 1, ctx, 1), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="token_id", dtype=np.int32),
            ct.TensorType(name="token_logit", dtype=np.float16),
        ],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
                name="kv_cache_0",
            ),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )

    if quantize:
        if quantize == "int4":
            op_config = ct.optimize.coreml.OpPalettizerConfig(
                nbits=4, granularity="per_grouped_channel", group_size=32,
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config)
            print("  Applied int4 palettization")
        elif quantize == "int8":
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8",
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)
            print("  Applied int8 quantization")

    path = os.path.join(args.output, "decoder.mlpackage")
    if os.path.exists(path):
        shutil.rmtree(path)
    mlmodel.save(path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path) for f in fns
    ) / 1024 / 1024
    print(f"  Saved {path} ({size_mb:.1f} MB)")

    # ============================================================
    # Export embedding weights for Swift-side processing
    # ============================================================
    print("\nExporting embedding weights...")

    # Save embed_tokens and embed_tokens_per_layer weights as numpy files
    embed_path = os.path.join(args.output, "embeddings")
    os.makedirs(embed_path, exist_ok=True)

    np.save(
        os.path.join(embed_path, "embed_tokens.npy"),
        text_model.embed_tokens.weight.data.cpu().to(torch.float16).numpy()
    )
    np.save(
        os.path.join(embed_path, "embed_tokens_per_layer.npy"),
        text_model.embed_tokens_per_layer.weight.data.cpu().to(torch.float16).numpy()
    )
    np.save(
        os.path.join(embed_path, "per_layer_model_projection.npy"),
        text_model.per_layer_model_projection.weight.data.cpu().to(torch.float16).numpy()
    )
    print(f"  Saved embedding weights to {embed_path}/")

    # Write config
    config_data = {
        "model_name": "gemma4-e2b-multimodal",
        "architecture": "gemma4",
        "multimodal": True,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": text_model.config.num_attention_heads,
        "num_key_value_heads": text_model.config.num_key_value_heads,
        "head_dim": text_model.config.head_dim,
        "global_head_dim": text_model.config.global_head_dim,
        "vocab_size": text_model.config.vocab_size,
        "context_length": ctx,
        "per_layer_dim": per_layer_dim,
        "image_token_id": 258880,
        "image_seq_length": 280,
        "num_patches": 2520,
        "patch_dim": 768,
        "bos_token_id": text_model.config.bos_token_id,
        "eos_token_id": text_model.config.eos_token_id,
        "quantization": quantize or "fp16",
        "parts": {
            "vision": "vision.mlpackage",
            "decoder": "decoder.mlpackage",
            "embeddings": "embeddings/",
        },
        "tokenizer_repo": "google/gemma-4-E2B-it",
        "embed_scale": float(hidden_size ** 0.5),
        "per_layer_model_projection_scale": float(hidden_size ** -0.5),
        "per_layer_input_scale": float(2.0 ** -0.5),
        "per_layer_embed_scale": float(per_layer_dim ** 0.5),
    }

    config_path = os.path.join(args.output, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"  Saved {config_path}")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Vision:  {args.output}/vision.mlpackage")
    print(f"  Decoder: {args.output}/decoder.mlpackage")
    print(f"  Embeds:  {args.output}/embeddings/")
    print(f"  Config:  {args.output}/model_config.json")
    print("=" * 60)

    # Cleanup HF model from memory
    del hf_model, text_model, wrapper
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
