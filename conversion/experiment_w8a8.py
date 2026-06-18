#!/usr/bin/env python3
"""W8A8 (int8 weights + int8 ACTIVATIONS) viability probe for pplx-embed.

Milestone B4. WEIGHT-only quant is a dead end for this encoder (int8 linear ~0.42
cosine; int4 palettize ~0.905) *and* only buys 4-8% latency, because the model is
activation/compute-bound (fp16 attention), not weight-bandwidth-bound. The real
bandwidth lever is ACTIVATION quantization. This script answers empirically: can
W8A8 reach acceptable fidelity, or does it hit the attention-family wall (~cos 0.57)?

Pipeline (per the coremltools activation-quant flow):
  1. Build an fp16 pooled_fp16 encoder at a SMALL bucket (L=128/256) — output_mode
     "pooled_fp16" so the CoreML pooled vector is Python-readable on macOS26.
  2. Calibrate activation ranges on a small multilingual corpus (tokenized + padded
     to the bucket) via cto.experimental.linear_quantize_activations.
  3. Quantize weights int8 (linear_symmetric) on top -> W8A8.
  4. Predict on the eval texts, quantize the pooled output with int8_tanh_quant, and
     compute cosine vs the fp32 Reference oracle. Report mean/min.

Two activation-quant modes are exposed because the attention pad-mask uses a large
negative sentinel (Qwen3Encoder.NEG_INF = -1e4; CoreML may lower the mask add to the
fp16 -65504 floor). A SYMMETRIC activation quantizer maps that catastrophically:
scale = 1e4/127 ~= 79, so real attention scores (+-10) round to ~0 and the model
collapses. ASYMMETRIC (mode="linear") lets the range span [-1e4, +score]; when the
span overflows fp16 the scale goes inf and coremltools SKIPS that op (left in fp16) —
which is exactly what we want for the mask add, while every other activation
quantizes normally.

Usage:
  uv run python conversion/experiment_w8a8.py --bucket 128 --mode asymmetric --rescale-k 8
  uv run python conversion/experiment_w8a8.py --bucket 128 --mode symmetric --rescale-k 8
  uv run python conversion/experiment_w8a8.py --bucket 128 --mode asymmetric --rescale-k 0   # no rescale
  uv run python conversion/experiment_w8a8.py --all   # sweep the key variants + baselines
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

import coremltools as ct  # noqa: E402
import coremltools.optimize.coreml as cto  # noqa: E402

from models.qwen3_encoder import (  # noqa: E402
    Qwen3EncoderConfig,
    PplxEmbedModel,
    load_encoder_weights,
    apply_fp16_residual_rescale,
)
from pplx_embed_reference import Reference, int8_tanh_quant, cosine_similarity  # noqa: E402

HF_REPO = "perplexity-ai/pplx-embed-v1-0.6b"
fp16_mil = ct.converters.mil.mil.types.fp16


# --------------------------------------------------------------------------- #
# coremltools compatibility / behaviour patches.
# --------------------------------------------------------------------------- #
def _patch_coremltools_cast() -> None:
    """coremltools 9 _cast() folds const int/bool casts but calls int()/bool() on
    numpy>=2 (1,)-shaped arrays, which raises. Extract the Python scalar first."""
    from coremltools.converters.mil.frontend.torch import ops as _torch_ops
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.mil import Builder as mb

    def _cast_patched(context, node, dtype, dtype_name):
        inputs = _get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            val = x.val
            if isinstance(val, np.ndarray) and val.ndim >= 1:
                val = val.item()
            if not isinstance(val, dtype):
                res = mb.const(val=dtype(val), name=node.name)
            else:
                res = mb.const(val=val, name=node.name)
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    _torch_ops._cast = _cast_patched


def _patch_coremltools_act_quant() -> None:
    """insert_prefix_quantize_dequantize_pair tries to wrap every supported op with a
    quantize/dequantize pair, including ops whose input x is int32 (mask add / expand,
    embedding path). MIL `quantize` requires float input, so int32-input ops crash with
    'scale has dtype fp32 whereas input has dtype int32'. Skip non-float-input ops."""
    from coremltools.optimize.coreml import _quantization_passes
    from coremltools.converters.mil.mil import types as mil_types

    _orig = _quantization_passes.insert_prefix_quantize_dequantize_pair.transform_op

    def _patched(self, op):
        x_var = op.inputs.get("x")
        if x_var is not None and not mil_types.is_float(x_var.dtype):
            return
        return _orig(self, op)

    _quantization_passes.insert_prefix_quantize_dequantize_pair.transform_op = _patched


_patch_coremltools_cast()
_patch_coremltools_act_quant()


# --------------------------------------------------------------------------- #
# Calibration / eval corpus.
# --------------------------------------------------------------------------- #
CALIBRATION_TEXTS = [
    "The transformer architecture has revolutionized natural language processing.",
    "Apple Silicon's Neural Engine achieves high energy efficiency for ML workloads.",
    "Bidirectional attention lets every token attend to every other token.",
    "Retrieval-augmented generation grounds responses in external knowledge.",
    "El procesamiento del lenguaje natural ha avanzado mucho en los ultimos anos.",
    "Le modele encode chaque phrase en un vecteur dense de grande dimension.",
    "Maschinelles Lernen ermoglicht effiziente Inferenz direkt auf dem Geraet.",
    "深層学習はテキストを密なベクトル表現に変換します。",
    "向量检索通过余弦相似度衡量语义相关性。",
    "machine learning",
    "natural language processing",
    "Cosine similarity measures semantic relatedness between embedding vectors.",
    "Late chunking encodes long documents with a single bidirectional forward pass.",
    "The int8 quantization of embeddings reduces memory bandwidth and latency.",
]

EVAL_TEXTS = [
    "Quantum computing leverages superposition and entanglement for computation.",
    "La inteligencia artificial transforma la manera en que trabajamos.",
    "Les reseaux de neurones apprennent des representations hierarchiques.",
    "Neuronale Netze lernen hierarchische Merkmalsrepraesentationen.",
    "気候変動は地球規模で生態系に影響を与えています。",
    "知识图谱将实体和关系组织成结构化的网络。",
    "Vector databases enable fast approximate nearest neighbor search at scale.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The stock market reacted sharply to the central bank's announcement.",
    "Renewable energy sources are critical to mitigating climate change.",
    "embeddings",
    "A short multilingual sentence. Une phrase courte. Ein kurzer Satz.",
]


# --------------------------------------------------------------------------- #
# Tokenization + padding to a fixed bucket.
# --------------------------------------------------------------------------- #
def tokenize_padded(tokenizer, texts: list[str], bucket: int) -> list[dict]:
    """Right-pad each text to `bucket`. Returns list of {input_ids, attention_mask}
    with input_ids int32 [1,L] and attention_mask fp16 [1,L] (1 valid / 0 pad)."""
    pad_id = tokenizer.pad_token_id or 0
    out = []
    for t in texts:
        enc = tokenizer([t], padding=False, truncation=True, max_length=bucket,
                        return_tensors="np")
        ids = enc["input_ids"].astype(np.int32)
        mask = enc["attention_mask"].astype(np.float16)
        L = ids.shape[1]
        if L < bucket:
            pad = bucket - L
            ids = np.concatenate([ids, np.full((1, pad), pad_id, np.int32)], axis=1)
            mask = np.concatenate([mask, np.zeros((1, pad), np.float16)], axis=1)
        out.append({"input_ids": ids, "attention_mask": mask})
    return out


# --------------------------------------------------------------------------- #
# Build.
# --------------------------------------------------------------------------- #
def build_fp16_encoder(snap: str, bucket: int, rescale_k: float) -> ct.models.MLModel:
    cfg = Qwen3EncoderConfig.from_json(os.path.join(snap, "config.json"), max_seq_len=bucket)
    model = PplxEmbedModel(cfg, output_mode="pooled_fp16").eval()
    load_encoder_weights(model.encoder, snap)
    if rescale_k:
        apply_fp16_residual_rescale(model.encoder, rescale_k)

    sample_ids = torch.zeros((1, bucket), dtype=torch.int32)
    sample_mask = torch.ones((1, bucket), dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(model, (sample_ids, sample_mask))

    inputs = [
        ct.TensorType(name="input_ids", shape=(1, bucket), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, bucket), dtype=np.float16),
    ]
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="embedding", dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS26,
        compute_units=ct.ComputeUnit.ALL,
    )
    return mlmodel


def quantize_w8a8(fp16_model: ct.models.MLModel, calib: list[dict], mode: str) -> ct.models.MLModel:
    """mode: 'asymmetric' -> activation mode='linear'; 'symmetric' -> 'linear_symmetric'."""
    act_mode = "linear" if mode == "asymmetric" else "linear_symmetric"
    act_cfg = cto.OptimizationConfig(
        global_config=cto.experimental.OpActivationLinearQuantizerConfig(mode=act_mode),
    )
    model_a8 = cto.experimental.linear_quantize_activations(fp16_model, act_cfg, calib)

    w_cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype=np.int8, weight_threshold=512,
        )
    )
    return cto.linear_quantize_weights(model_a8, w_cfg)


# --------------------------------------------------------------------------- #
# Measure.
# --------------------------------------------------------------------------- #
def predict_pooled(mlmodel: ct.models.MLModel, samples: list[dict]) -> np.ndarray:
    rows = []
    for s in samples:
        out = mlmodel.predict({"input_ids": s["input_ids"], "attention_mask": s["attention_mask"]})
        rows.append(np.asarray(out["embedding"], dtype=np.float32).reshape(-1))
    return np.stack(rows, axis=0)  # [N, 1024]


def fidelity(pooled_fp16: np.ndarray, ref_int8: np.ndarray) -> tuple[float, float, np.ndarray]:
    cm_int8 = int8_tanh_quant(pooled_fp16).astype(np.float32)
    cos = cosine_similarity(cm_int8, ref_int8.astype(np.float32))
    cos = cos[np.isfinite(cos)]
    return float(cos.mean()), float(cos.min()), cos


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def run_variant(snap, tokenizer, ref_int8, bucket, mode, rescale_k, out_root, save=True):
    label = f"w8a8-{mode}-k{int(rescale_k)}-L{bucket}"
    print(f"\n{'='*70}\n{label}\n{'='*70}", flush=True)

    fp16_model = build_fp16_encoder(snap, bucket, rescale_k)
    calib = tokenize_padded(tokenizer, CALIBRATION_TEXTS, bucket)
    eval_samples = tokenize_padded(tokenizer, EVAL_TEXTS, bucket)

    # fp16 baseline fidelity (same graph, no quant) for reference.
    fp16_pooled = predict_pooled(fp16_model, eval_samples)
    fp16_mean, fp16_min, _ = fidelity(fp16_pooled, ref_int8)
    print(f"  fp16 baseline:  mean={fp16_mean:.4f}  min={fp16_min:.4f}", flush=True)

    print(f"  quantizing W8A8 (activation mode={mode}) ...", flush=True)
    w8a8 = quantize_w8a8(fp16_model, calib, mode)
    w8a8_pooled = predict_pooled(w8a8, eval_samples)
    w8a8_mean, w8a8_min, cos = fidelity(w8a8_pooled, ref_int8)
    print(f"  W8A8 {mode:10s}:  mean={w8a8_mean:.4f}  min={w8a8_min:.4f}", flush=True)
    print(f"    per-text cos: {np.round(cos, 3).tolist()}", flush=True)

    pkg = None
    if save:
        pkg = os.path.join(out_root, f"{label}.mlpackage")
        if os.path.exists(pkg):
            shutil.rmtree(pkg)
        os.makedirs(out_root, exist_ok=True)
        w8a8.save(pkg)
        print(f"  saved {pkg}", flush=True)

    return {
        "label": label, "bucket": bucket, "mode": mode, "rescale_k": rescale_k,
        "fp16_mean": fp16_mean, "fp16_min": fp16_min,
        "w8a8_mean": w8a8_mean, "w8a8_min": w8a8_min, "pkg": pkg,
    }


def audit_ane(pkg: str) -> None:
    """Compile a .mlpackage and report ANE/CPU/GPU op residency (no xcrun needed)."""
    from collections import Counter
    from coremltools.models.utils import compile_model
    from coremltools.models.compute_plan import MLComputePlan

    mlc = pkg.rstrip("/") + ".mlmodelc"
    if os.path.exists(mlc):
        shutil.rmtree(mlc)
    compiled = compile_model(pkg, mlc)
    print(f"\n=== ANE audit: {compiled} ===", flush=True)
    plan = MLComputePlan.load_from_path(path=compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
    ms = plan.model_structure
    prog = getattr(ms, "program", None)
    by_dev = Counter()
    by_op_dev = Counter()
    total = 0

    def walk(block, fn):
        for op in block.operations:
            yield fn, op
            for nb in getattr(op, "blocks", ()) or ():
                yield from walk(nb, fn)

    for fn, func in prog.functions.items():
        for fname, op in walk(func.block, fn):
            if op.operator_name == "const":
                continue
            try:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                pref = getattr(usage, "preferred_compute_device", None) or getattr(usage, "preferred", None)
                name = type(pref).__name__ if pref is not None else "unknown"
                dev = "ANE" if ("Neural" in name or "ANE" in name) else ("GPU" if "GPU" in name else ("CPU" if "CPU" in name else name))
            except Exception:
                dev = "unknown"
            by_dev[dev] += 1
            by_op_dev[(op.operator_name, dev)] += 1
            total += 1
    print(f"  total ops: {total}")
    for dev, n in sorted(by_dev.items(), key=lambda kv: -kv[1]):
        print(f"    {dev}: {n}  ({100.0*n/total:.1f}%)")
    non_ane = [(o, d, n) for (o, d), n in by_op_dev.items() if d != "ANE"]
    if non_ane:
        print("  non-ANE ops:")
        for o, d, n in sorted(non_ane, key=lambda t: -t[2])[:15]:
            print(f"    [{d:3s}] {o:<26s} {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", type=int, default=128)
    ap.add_argument("--mode", default="asymmetric", choices=["asymmetric", "symmetric"])
    ap.add_argument("--rescale-k", type=float, default=8.0)
    ap.add_argument("--all", action="store_true", help="Sweep the key variants")
    ap.add_argument("--out", default="/tmp/w8a8-experiment")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--audit", default=None, help="Compile + ANE-audit an existing .mlpackage and exit")
    args = ap.parse_args()

    if args.audit:
        audit_ane(args.audit)
        return

    from huggingface_hub import snapshot_download
    snap = snapshot_download(
        HF_REPO,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.py", "1_Pooling/*"],
    )

    print("Loading fp32 Reference oracle ...", flush=True)
    ref = Reference(HF_REPO)
    tokenizer = ref.tokenizer
    ref_int8 = ref.embed(EVAL_TEXTS)  # [N, 1024] int8

    results = []
    if args.all:
        variants = [
            (args.bucket, "asymmetric", 8.0),
            (args.bucket, "symmetric", 8.0),
            (args.bucket, "asymmetric", 0.0),
            (args.bucket, "asymmetric", 16.0),
        ]
        for bucket, mode, k in variants:
            try:
                results.append(run_variant(snap, tokenizer, ref_int8, bucket, mode, k,
                                           args.out, save=not args.no_save))
            except Exception as e:
                print(f"  VARIANT FAILED ({mode}, k={k}): {e}", flush=True)
                import traceback; traceback.print_exc()
    else:
        results.append(run_variant(snap, tokenizer, ref_int8, args.bucket, args.mode,
                                   args.rescale_k, args.out, save=not args.no_save))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'variant':<26s} {'fp16 mean':>10s} {'W8A8 mean':>10s} {'W8A8 min':>10s}")
    for r in results:
        print(f"{r['label']:<26s} {r['fp16_mean']:>10.4f} {r['w8a8_mean']:>10.4f} {r['w8a8_min']:>10.4f}")
    print("\nReference points: fp16~0.999 | weight-only int8~0.42 | int4~0.905 | wall~0.57 | gate 0.990")
    for r in results:
        if r["pkg"]:
            print(f"  artifact: {r['pkg']}")


if __name__ == "__main__":
    main()
