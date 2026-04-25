#!/usr/bin/env python3
"""Build Gemma 4 E2B stateful chunks (MLState + slice_update KV).

Phase 1 of the Gemma 4 stateful migration — mirrors the Qwen3-VL
v1.5.0 pattern. Produces 4 mlpackages:

  chunk_1.mlpackage: own KV state (sliding + full), computes per_layer_combined
  chunk_2.mlpackage: own KV state, emits kv13_*/kv14_* producer aliases
  chunk_3.mlpackage: stateless, reads kv13/14
  chunk_4.mlpackage: stateless, reads kv13/14, lm_head + argmax

Sliding cache uses ring writes at slot `ring_pos = current_pos % W`,
which Swift precomputes and passes alongside `current_pos`. The
`update_mask` input that the recurrent build needed for ANE-compat
out-of-place full-layer writes is GONE — `ios18.slice_update` does
the in-place write natively.

Sidecars (embed_weight, per-layer projection, RoPE tables, tokenizer,
model_config.json) are produced by the existing
`build_gemma4_bundle.py` pipeline — this script only touches the
chunk mlpackages. After running this, copy the chunk_{1..4}.mlpackage
files alongside an existing E2B sidecar bundle to ship a complete
Gemma 4 E2B stateful build.

T=1 only (decode + slow T=1 prefill). Multifunction prefill_bN is a
follow-up (mirrors Qwen3-VL v1.5.0 → multifunction multifunction).

Usage:
  python conversion/build_gemma4_e2b_stateful_chunks.py \
      --output /tmp/gemma4-e2b-stateful \
      [--ctx 2048] [--nbits 4] [--only-chunk N]

  GEMMA4_HF_DIR or --hf-dir picks the local HF model path.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)

from ane_ops import MODEL_DTYPE
from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import compute_chunk_boundaries
from models.gemma4_swa_stateful_chunks import (
    SWAStatefulChunk1, SWAStatefulChunk2,
    SWAStatefulChunk3, SWAStatefulChunk4,
)


DEFAULT_HF_DIR = os.environ.get(
    "GEMMA4_HF_DIR", f"{ROOT}/../output/gemma4-e2b/hf_model")
fp16 = np.float16


# ============================================================
# HF model loading (mirrors build_verify_chunks.py)
# ============================================================

def _resolve_hf_dir(model_name: str, override: str | None) -> str:
    if override:
        return override
    if model_name in MODEL_REGISTRY:
        from huggingface_hub import snapshot_download
        repo = MODEL_REGISTRY[model_name].hf_repo
        local = os.path.join(ROOT, "..", "output", model_name, "hf_model")
        if not os.path.isdir(local) or not any(
            fn.endswith(".safetensors") for fn in os.listdir(local)
        ):
            print(f"Downloading {repo} to {local}...")
            snapshot_download(
                repo, local_dir=local,
                allow_patterns=["*.safetensors", "*.json", "tokenizer*",
                                "*.txt", "*.model"],
            )
        return local
    return DEFAULT_HF_DIR


# ============================================================
# Conversion helpers
# ============================================================

def _audit_ane(pkg_path: str) -> float:
    try:
        m = ct.models.MLModel(pkg_path,
                              compute_units=ct.ComputeUnit.CPU_AND_NE)
        compiled = m.get_compiled_model_path()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        dev = Counter()
        for fn in plan.model_structure.program.functions.values():
            for op in fn.block.operations:
                a = plan.get_compute_device_usage_for_mlprogram_operation(op)
                d = ("const" if (a is None and op.operator_name == "const")
                     else (a.preferred_compute_device.__class__.__name__
                           if a else "unknown"))
                dev[d] += 1
        total = sum(dev.values())
        compute = total - dev.get("const", 0)
        ane = dev.get("MLNeuralEngineComputeDevice", 0)
        pct = 100 * ane / compute if compute else 0.0
        print(f"    ANE placement: {ane}/{compute} ({pct:.1f}%)")
        return pct
    except Exception as e:
        print(f"    ANE audit skipped: {e}")
        return 0.0


def _trace_and_convert_stateful(
    model, sample_inputs, input_specs, output_specs, state_specs,
    out_path: str, quantize_nbits: int,
):
    """Trace + ct.convert with StateType, save, optionally palettize, audit."""
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_inputs, check_trace=False,
                                 strict=False)
    print(f"    traced in {time.time()-t:.1f}s")

    t = time.time()
    convert_kwargs = dict(
        convert_to="mlprogram",
        inputs=input_specs,
        outputs=output_specs,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    # Only pass states= for stateful chunks. Empty list trips ct.convert
    # on some coremltools versions ("expected at least one state").
    if state_specs:
        convert_kwargs["states"] = state_specs
    mlmodel = ct.convert(traced, **convert_kwargs)
    print(f"    converted in {time.time()-t:.1f}s")

    if quantize_nbits > 0:
        t = time.time()
        op_cfg = OpPalettizerConfig(
            mode="kmeans", nbits=quantize_nbits,
            granularity="per_grouped_channel", group_size=32,
        )
        mlmodel = palettize_weights(mlmodel, OptimizationConfig(global_config=op_cfg))
        print(f"    palettized int{quantize_nbits} in {time.time()-t:.1f}s")

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    mlmodel.save(out_path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(out_path)
        for f in fns
    ) / 1024 / 1024
    print(f"    saved {out_path} ({size_mb:.1f} MB)")
    _audit_ane(out_path)


# ============================================================
# Per-chunk converters
# ============================================================

def convert_chunk1(base, c_start, c_end, ctx, out_path, nbits):
    print("\n" + "=" * 60)
    print(f"CHUNK 1 (L{c_start}-{c_end-1}) — own KV state, computes PLE")
    print("=" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulChunk1(base, c_start, c_end, ctx).eval().to(MODEL_DTYPE)
    ns, nf = max(chunk.num_sliding, 1), max(chunk.num_full, 1)

    sample = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),                  # hidden_states
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),                   # causal_mask_full
        torch.zeros(1, 1, 1, W, dtype=torch.float16),                     # causal_mask_sliding
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),            # per_layer_raw
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),                  # cos_s
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),                  # sin_s
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),                  # cos_f
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),                  # sin_f
        torch.zeros(1, dtype=torch.int32),                                # current_pos
        torch.zeros(1, dtype=torch.int32),                                # ring_pos
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",   shape=sample[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_raw",      shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="current_pos",        shape=(1,),            dtype=np.int32),
        ct.TensorType(name="ring_pos",           shape=(1,),            dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="hidden_states_out",       dtype=fp16),
        ct.TensorType(name="per_layer_combined_out",  dtype=fp16),
    ]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * ns, HKV, W, max_hd), dtype=fp16),
            name="kv_cache_sliding",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * nf, HKV, ctx, max_hd), dtype=fp16),
            name="kv_cache_full",
        ),
    ]
    _trace_and_convert_stateful(
        chunk, sample, inputs, outputs, states, out_path, nbits)


def convert_chunk2(base, c_start, c_end, ctx, out_path, nbits):
    print("\n" + "=" * 60)
    print(f"CHUNK 2 (L{c_start}-{c_end-1}) — own KV state, emits kv13/kv14")
    print("=" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulChunk2(base, c_start, c_end, ctx).eval().to(MODEL_DTYPE)
    ns, nf = max(chunk.num_sliding, 1), max(chunk.num_full, 1)

    sample = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",   shape=sample[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_combined", shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="current_pos",        shape=(1,),            dtype=np.int32),
        ct.TensorType(name="ring_pos",           shape=(1,),            dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="hidden_states_out", dtype=fp16),
        ct.TensorType(name="kv13_k",            dtype=fp16),
        ct.TensorType(name="kv13_v",            dtype=fp16),
        ct.TensorType(name="kv14_k",            dtype=fp16),
        ct.TensorType(name="kv14_v",            dtype=fp16),
    ]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * ns, HKV, W, max_hd), dtype=fp16),
            name="kv_cache_sliding",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * nf, HKV, ctx, max_hd), dtype=fp16),
            name="kv_cache_full",
        ),
    ]
    _trace_and_convert_stateful(
        chunk, sample, inputs, outputs, states, out_path, nbits)


def convert_chunk_shared(chunk_cls, base, c_start, c_end, ctx,
                         out_path, nbits, name, with_lm_head):
    """Stateless chunk (3 or 4). All layers KV-shared from kv13/kv14."""
    print("\n" + "=" * 60)
    print(f"{name} (L{c_start}-{c_end-1}) — stateless, reads kv13/14"
          + (" + lm_head" if with_lm_head else ""))
    print("=" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    HKV = cfg.num_key_value_heads

    chunk = chunk_cls(base, c_start, c_end).eval().to(MODEL_DTYPE)
    sample = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, HKV, W, hd_s, dtype=torch.float16),     # kv13_k
        torch.zeros(1, HKV, W, hd_s, dtype=torch.float16),     # kv13_v
        torch.zeros(1, HKV, ctx, hd_f, dtype=torch.float16),   # kv14_k
        torch.zeros(1, HKV, ctx, hd_f, dtype=torch.float16),   # kv14_v
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",   shape=sample[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_combined", shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="kv13_k",             shape=sample[8].shape, dtype=fp16),
        ct.TensorType(name="kv13_v",             shape=sample[9].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",             shape=sample[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",             shape=sample[11].shape, dtype=fp16),
    ]
    if with_lm_head:
        outputs = [
            ct.TensorType(name="token_id",    dtype=np.int32),
            ct.TensorType(name="token_logit", dtype=fp16),
            ct.TensorType(name="hidden_normed", dtype=fp16),
        ]
    else:
        outputs = [ct.TensorType(name="hidden_states_out", dtype=fp16)]
    _trace_and_convert_stateful(
        chunk, sample, inputs, outputs, state_specs=[],
        out_path=out_path, quantize_nbits=nbits)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b",
                    help="Model name (gemma4-e2b | gemma4-e4b)")
    ap.add_argument("--output", required=True,
                    help="Output directory for stateful mlpackages")
    ap.add_argument("--hf-dir", default=None,
                    help="Override HF model path (skip auto-download)")
    ap.add_argument("--ctx", type=int, default=None,
                    help="Context length (default: registry default)")
    ap.add_argument("--nbits", type=int, default=4, choices=[0, 4, 8],
                    help="Palettization (0 = fp16, 4 = INT4, 8 = INT8)")
    ap.add_argument("--only-chunk", type=int, default=None, choices=[1, 2, 3, 4],
                    help="Smoke test: convert only one chunk and stop. "
                         "Useful for first runs on Mac Studio to validate "
                         "the conversion path before committing to all 4.")
    args = ap.parse_args()

    if args.ctx is None:
        if args.model in MODEL_REGISTRY:
            args.ctx = MODEL_REGISTRY[args.model].default_context_length
        else:
            args.ctx = 2048

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir}...")
    t0 = time.time()
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    cfg = base.config
    boundaries = compute_chunk_boundaries(cfg)
    print(f"\nctx={args.ctx}  W={cfg.sliding_window}  "
          f"hidden={cfg.hidden_size}  pld={cfg.hidden_size_per_layer_input}")
    print(f"layers={cfg.num_hidden_layers}  "
          f"head_dim={cfg.head_dim}  global_head_dim={cfg.global_head_dim}  "
          f"num_kv_heads={cfg.num_key_value_heads}")
    print(f"KV producers: sliding=L{cfg.kv_sliding_producer}, "
          f"full=L{cfg.kv_full_producer}")
    print(f"Chunk boundaries: {boundaries}")
    print(f"Quantize: int{args.nbits}" if args.nbits else "Quantize: fp16")

    do = (lambda n: args.only_chunk is None or args.only_chunk == n)

    if do(1):
        convert_chunk1(base, *boundaries[0], args.ctx,
                       str(out / "chunk_1.mlpackage"), args.nbits)
    if do(2):
        convert_chunk2(base, *boundaries[1], args.ctx,
                       str(out / "chunk_2.mlpackage"), args.nbits)
    if do(3):
        convert_chunk_shared(SWAStatefulChunk3, base, *boundaries[2], args.ctx,
                             str(out / "chunk_3.mlpackage"), args.nbits,
                             name="CHUNK 3", with_lm_head=False)
    if do(4):
        convert_chunk_shared(SWAStatefulChunk4, base, *boundaries[3], args.ctx,
                             str(out / "chunk_4.mlpackage"), args.nbits,
                             name="CHUNK 4", with_lm_head=True)

    print(f"\nartifacts under {out}")
    for p in sorted(out.iterdir()):
        size = (sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                if p.is_dir() else p.stat().st_size) / 1e6
        print(f"  {p.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
