#!/usr/bin/env python3
"""Build a CoreML bundle for pplx-embed (bidirectional Qwen3 encoder).

Stateless single-forward export — no KV cache, no causal mask. Fixed-length
(`--max-seq-len`, the bucket) input + pad mask. Variable length is handled by
padding to the nearest bucket at runtime (fixed shapes keep it on the ANE;
RangeDim/EnumeratedShapes force CPU fallback).

Output modes:
    pooled_fp16  masked-mean → (1, 1024) fp16   — readable from the Python bridge;
                 quantize to int8 downstream. Use for fidelity measurement.
    int8         masked-mean → tanh → int8       — the deliverable (native int8;
                 read via the Swift harness on macOS26).

Usage:
    python conversion/build_pplx_embed_bundle.py --model pplx-embed --max-seq-len 4096
    python conversion/build_pplx_embed_bundle.py --model pplx-embed --max-seq-len 128 \
        --output-mode pooled_fp16
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.qwen3_encoder import (  # noqa: E402
    Qwen3EncoderConfig,
    PplxEmbedModel,
    PplxEmbedContextModel,
    N_MAX_CHUNKS,
    load_encoder_weights,
    apply_fp16_residual_rescale,
)

# Residual rescale factor (see qwen3_encoder.apply_fp16_residual_rescale).
# K=8 is the fidelity↔overflow sweet spot: validated overflow-safe (peak |h| ~37k at
# 455 real tokens, 1.75× under fp16 max) and markedly better than K=16 on short chunks
# (context mean 0.9987 vs 0.9911). K=4 overflows. Bump toward 16 if long inputs NaN.
DEFAULT_RESCALE_K = 8.0


def _snapshot_dir(hf_repo: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(
        hf_repo,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.py", "1_Pooling/*"],
    )


def build_bundle(
    hf_repo: str,
    model_name: str,
    output_dir: str,
    max_seq_len: int = 4096,
    output_mode: str = "int8",
    rescale_k: float = DEFAULT_RESCALE_K,
    quantize: str | None = None,
    variant: str = "plain",
    dynamic_upper: int = 0,
    skip_if_exists: bool = True,
    norm_impl: str = "native",
) -> str:
    """Build a CoreML bundle.

    dynamic_upper > 0 → a **flexible RangeDim** model (seq 1..dynamic_upper) targeting the
    GPU — the non-padded, unbounded-length catch-all for inputs larger than the biggest fixed
    ANE bucket. (Flexible shapes force CPU fallback on ANE and are ~10× slower than fixed
    buckets, so this is GPU-only and reserved for >max-bucket inputs.) Otherwise a fixed-shape
    bucket (the fast ANE path).
    """
    import coremltools as ct

    os.makedirs(output_dir, exist_ok=True)
    pkg = os.path.join(output_dir, "encoder.mlpackage")
    if skip_if_exists and os.path.exists(pkg):
        print(f"  [skip] {pkg} exists")
        return pkg

    dynamic = dynamic_upper > 0
    print(f"[1/4] Loading {hf_repo} (config + weights; variant={variant}"
          + (f", dynamic RangeDim 1..{dynamic_upper} GPU" if dynamic else "") + ")")
    snap = hf_repo if os.path.isdir(hf_repo) else _snapshot_dir(hf_repo)
    # The bucket (input shape). The RoPE table is built once to a fixed length
    # (max_position_embeddings) inside Qwen3Encoder._build_rope and gathered to S at
    # runtime, so it no longer tracks the bucket — that keeps weight.bin byte-identical
    # across buckets (HF LFS stores one blob). For the dynamic RangeDim model max_seq_len
    # is informational only (the input is RangeDim 1..dynamic_upper).
    bucket_len = dynamic_upper if dynamic else max_seq_len
    cfg = Qwen3EncoderConfig.from_json(os.path.join(snap, "config.json"),
                                       max_seq_len=bucket_len, norm_impl=norm_impl)
    if variant == "context":
        model = PplxEmbedContextModel(cfg, output_mode=output_mode).eval()
    else:
        model = PplxEmbedModel(cfg, output_mode=output_mode).eval()
    load_encoder_weights(model.encoder, snap)
    if rescale_k:
        print(f"[1.5/4] fp16 residual rescale 1/K (K={rescale_k})")
        apply_fp16_residual_rescale(model.encoder, rescale_k)

    if dynamic and variant == "context":
        raise ValueError("dynamic (RangeDim) mode supports only the plain variant")

    trace_len = min(512, dynamic_upper) if dynamic else max_seq_len
    print(f"[2/4] Tracing (trace_len={trace_len}, mode={output_mode}, variant={variant}"
          + (f", RangeDim 1..{dynamic_upper}" if dynamic else "") + ")")
    sample_ids = torch.zeros((1, trace_len), dtype=torch.int32)
    sample_mask = torch.ones((1, trace_len), dtype=torch.float16)
    if dynamic:
        seqdim = ct.RangeDim(lower_bound=1, upper_bound=dynamic_upper, default=trace_len)
        inputs = [
            ct.TensorType(name="input_ids", shape=(1, seqdim), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seqdim), dtype=np.float16),
        ]
        trace_args = (sample_ids, sample_mask)
    else:
        inputs = [
            ct.TensorType(name="input_ids", shape=(1, trace_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, trace_len), dtype=np.float16),
        ]
        if variant == "context":
            sample_pool = torch.zeros((N_MAX_CHUNKS, trace_len), dtype=torch.float16)
            sample_pool[0, :] = 1.0 / trace_len
            trace_args = (sample_ids, sample_mask, sample_pool)
            inputs.append(ct.TensorType(name="pool_matrix", shape=(N_MAX_CHUNKS, trace_len), dtype=np.float16))
        else:
            trace_args = (sample_ids, sample_mask)
    with torch.no_grad():
        traced = torch.jit.trace(model, trace_args)

    out_dtype = np.int8 if output_mode == "int8" else np.float16
    # Flexible shapes can't go on ANE (CPU fallback) → GPU; fixed buckets → ANE (ALL picks it).
    compute_units = ct.ComputeUnit.CPU_AND_GPU if dynamic else ct.ComputeUnit.ALL
    print(f"[3/4] Converting to CoreML (fp16, macOS26; out={out_dtype.__name__}; "
          f"units={'CPU_AND_GPU' if dynamic else 'ALL'})")
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="embedding", dtype=out_dtype)],
        minimum_deployment_target=ct.target.macOS26,
        compute_units=compute_units,
    )

    if quantize == "int4":
        op = ct.optimize.coreml.OpPalettizerConfig(nbits=4, granularity="per_grouped_channel", group_size=32)
        mlmodel = ct.optimize.coreml.palettize_weights(
            mlmodel, ct.optimize.coreml.OptimizationConfig(global_config=op))
        print("  applied int4 palettization (group_size=32)")
    elif quantize == "int8":
        op = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        mlmodel = ct.optimize.coreml.linear_quantize_weights(
            mlmodel, ct.optimize.coreml.OptimizationConfig(global_config=op))
        print("  applied int8 weight quantization")

    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    mlmodel.save(pkg)
    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fns in os.walk(pkg) for f in fns) / 1024 / 1024
    print(f"  saved {pkg} ({size_mb:.1f} MB)")

    _write_model_config(output_dir, model_name, hf_repo, cfg, max_seq_len,
                        output_mode, rescale_k, quantize, variant, dynamic_upper, norm_impl)
    _copy_tokenizer(snap, output_dir)
    print(f"[4/4] bundle ready at {output_dir}")
    return pkg


def _write_model_config(output_dir, model_name, hf_repo, cfg, max_seq_len,
                        output_mode, rescale_k, quantize, variant="plain", dynamic_upper=0,
                        norm_impl="native"):
    dynamic = dynamic_upper > 0
    out_dtype = "int8" if output_mode == "int8" else "fp16"
    out_shape = [N_MAX_CHUNKS, 1024] if variant == "context" else [1, 1024]
    seq_shape = [1, f"1..{dynamic_upper}"] if dynamic else [1, max_seq_len]
    inputs = {
        "input_ids": {"shape": seq_shape, "dtype": "int32"},
        "attention_mask": {"shape": seq_shape, "dtype": "fp16",
                           "doc": "1.0 for valid tokens, 0.0 for pad"},
    }
    if variant == "context":
        inputs["pool_matrix"] = {"shape": [N_MAX_CHUNKS, max_seq_len], "dtype": "fp16",
                                 "doc": "row k = 1/n_k over chunk k's span, else 0; unused rows all-zero"}
    cfgd = {
        "model_name": model_name,
        "architecture": "qwen3-encoder",
        "variant": variant,
        "tokenizer_repo": hf_repo,
        "parts": {"encoder": "encoder.mlpackage"},
        "io_contract": {
            "inputs": inputs,
            "outputs": {
                "embedding": {"shape": out_shape, "dtype": out_dtype,
                              "doc": ("per-chunk embeddings; read first N_actual rows (unused rows are 0)"
                                      if variant == "context" else "plain mean-pooled embedding")
                                     + "; int8 = clamp(round(tanh(x)*127),-128,127)"},
            },
        },
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.rms_norm_eps,
        "max_seq_len": max_seq_len,
        "bucket": (f"1..{dynamic_upper}" if dynamic else max_seq_len),
        "dynamic": dynamic,
        "dynamic_upper": dynamic_upper if dynamic else 0,
        "output_mode": output_mode,
        "fp16_residual_rescale_k": rescale_k,
        "norm_impl": norm_impl,
        "pooling": "mean",
        "quantization_weights": quantize or "fp16",
        "matryoshka_dims": [1024, 512, 256, 128],
        # Flexible RangeDim models force CPU fallback on ANE → run on GPU; fixed buckets on ANE.
        "compute_units": "CPU_AND_GPU" if dynamic else "CPU_AND_NE",
    }
    path = os.path.join(output_dir, "model_config.json")
    with open(path, "w") as f:
        json.dump(cfgd, f, indent=2)
    print(f"  wrote {path}")


def _copy_tokenizer(snap, output_dir):
    dst = os.path.join(output_dir, "hf_model")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(snap):
        if name.startswith("tokenizer") or name in (
            "config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
            "added_tokens.json",
        ):
            shutil.copy2(os.path.join(snap, name), os.path.join(dst, name))
    print(f"  copied tokenizer files → {dst}")


def main():
    from config import MODEL_REGISTRY

    ap = argparse.ArgumentParser(description="Build CoreML bundle for pplx-embed")
    ap.add_argument("--model", default="pplx-embed", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--output-mode", default="int8", choices=["int8", "pooled_fp16"])
    ap.add_argument("--rescale-k", type=float, default=DEFAULT_RESCALE_K)
    ap.add_argument("--quantize", default="none", choices=["none", "int8", "int4"])
    ap.add_argument("--variant", default="auto", choices=["auto", "plain", "context"],
                    help="auto → context if the model name contains 'context', else plain")
    ap.add_argument("--dynamic-upper", type=int, default=0,
                    help="If >0, build a flexible RangeDim (1..N) GPU model (the >max-bucket "
                         "catch-all), e.g. 8192. Plain only.")
    ap.add_argument("--hf-dir", default=None, help="Override HF dir (skip download)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-skip", action="store_true", help="Rebuild even if exists")
    ap.add_argument("--norm-impl", default="native", choices=["ane_cat", "native"],
                    help="RMSNorm for the 5 encoder norm sites: native (Qwen3RMSNorm rsqrt, "
                         "default — 12-21%% faster on ANE per experiment_ane_rmsnorm.py) or "
                         "ane_cat (shared cat/chunk LayerNorm trick).")
    args = ap.parse_args()

    reg = MODEL_REGISTRY[args.model]
    hf_repo = args.hf_dir or reg.hf_repo
    variant = ("context" if "context" in args.model else "plain") if args.variant == "auto" else args.variant
    if args.dynamic_upper:
        tag = f"dyn{args.dynamic_upper}-{args.output_mode}"
    else:
        tag = f"L{args.max_seq_len}-{args.output_mode}" + (f"-{args.quantize}" if args.quantize != "none" else "")
    output = args.output or os.path.join(ROOT, "..", "output", args.model, tag)
    quantize = None if args.quantize == "none" else args.quantize
    build_bundle(hf_repo, args.model, output, args.max_seq_len, args.output_mode,
                 args.rescale_k, quantize, variant=variant, dynamic_upper=args.dynamic_upper,
                 skip_if_exists=not args.no_skip, norm_impl=args.norm_impl)


if __name__ == "__main__":
    main()
