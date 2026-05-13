#!/usr/bin/env python3
"""Build verify_qK variants for the 3-chunk topology (chunk2_3way + chunk3_3way)
and bundle them as multi-function packages with the existing decode_q1
mlpackages from output/<model>/chunks_3way_fp16kv/.

Output layout:
    output/<model>/chunks_3way_fp16kv_mf/
        chunk1.mlpackage              (copy of decode-only chunk1, already
                                       in bundle_diff_logits; 4-chunk and
                                       3-chunk share L0-7)
        chunk2_3way.mlpackage         (multifunction: decode_q1 + verify_qK)
        chunk3_3way.mlpackage         (multifunction: decode_q1 + verify_qK)

Build path:
    1.  Trace MergedChunk23Verify(K) → convert → INT4 palettize w/ keep_fp_kv
    2.  Trace SWAVerifyChunk4(K, start=25, end=35) → convert → palettize
    3.  Bundle decode + verify mlpackages via MultiFunctionDescriptor
    4.  Caller compiles .mlpackage → .mlmodelc with `xcrun coremlcompiler`

Env knobs (mirror build_verify_chunks.py):
    PALETTIZE_KEEP_FP_KV=1       keep self_attn_(k|v)_proj fp16  [default 1]
    PALETTIZE_NBITS=4            INT4 weight quant
    PALETTIZE_GRANULARITY=per_grouped_channel
    PALETTIZE_GROUP_SIZE=32

Usage:
    python conversion/build_verify_chunks_3way.py --model gemma4-e2b --ctx 2048 --k 3
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import SWAVerifyChunk4, compute_chunk_boundaries
from models.gemma4_swa_merged import MergedChunk23Verify


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
                allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model"],
            )
        return local
    raise SystemExit(f"unknown model {model_name}")

fp16 = ct.converters.mil.input_types.dtype_to_str if False else np.float16


def _convert_with_palettize(model, sample, inputs, output_names, *, label: str,
                             quantize: bool = True):
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, check_trace=False)
    print(f"  [{label}] traced in {time.time()-t:.1f}s")

    t = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"  [{label}] converted in {time.time()-t:.1f}s")

    if quantize:
        t = time.time()
        nbits = int(os.environ.get("PALETTIZE_NBITS", "4"))
        granularity = os.environ.get("PALETTIZE_GRANULARITY", "per_grouped_channel")
        group_size = int(os.environ.get("PALETTIZE_GROUP_SIZE", "32"))
        # Default ON: keep K/V proj fp16. Matches the 4-chunk verify build's
        # postnorm_attn flavor that lifts MTP per-slot accept from 0.20 → 0.49
        # on code generation.
        keep_fp_kv = os.environ.get("PALETTIZE_KEEP_FP_KV", "1") == "1"

        cfg_kw = dict(nbits=nbits, granularity=granularity)
        if granularity == "per_grouped_channel":
            cfg_kw["group_size"] = group_size
        global_cfg = ct.optimize.coreml.OpPalettizerConfig(**cfg_kw)

        op_name_configs = {}
        if keep_fp_kv:
            patterns = ["self_attn_k_proj", "self_attn_v_proj"]
            from coremltools.optimize.coreml import get_weights_metadata
            md = get_weights_metadata(mlmodel, weight_threshold=2048)
            for name in md.keys():
                if any(p in name for p in patterns):
                    op_name_configs[name] = None
            print(f"  [{label}] keep_fp_kv → skipping {len(op_name_configs)} K/V proj weights")

        cfg = ct.optimize.coreml.OptimizationConfig(
            global_config=global_cfg,
            op_name_configs=op_name_configs if op_name_configs else None)
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg)
        print(f"  [{label}] palettized INT4/g{group_size} keep_fp_kv={keep_fp_kv} in {time.time()-t:.1f}s")

    return mlmodel


def _save(mlmodel, path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    mlmodel.save(path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path)
        for f in fns
    ) / 1024 / 1024
    print(f"    saved {path}  ({size_mb:.1f} MB)")


# ----------------------------------------------------------------------
# chunk2_3way verify (L8-24, Q=K)
# ----------------------------------------------------------------------

def build_chunk2_3way_verify(base, ctx: int, K: int, out_pkg: str) -> None:
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    max_hd = hd_f
    nkv = cfg.num_key_value_heads

    boundaries = compute_chunk_boundaries(cfg)
    own_range = boundaries[1]
    shared_range = boundaries[2]
    mc = MergedChunk23Verify(base, seq_len=K,
                              own_range=own_range,
                              shared_range=shared_range).eval()
    ns, nf = len(mc.sliding_layer_indices), len(mc.full_layer_indices)
    n_layers = (mc.END_C2 - mc.START_C2) + (mc.END_C3 - mc.START_C3)
    print(f"\n=== chunk2_3way verify_qK (L{mc.START_C2}-{mc.END_C3-1}, "
          f"{n_layers} layers, K={K}) ===")

    sample = (
        torch.zeros(1, K, hidden, dtype=torch.float16),
        torch.zeros(1, 1, K, ctx, dtype=torch.float16),
        torch.zeros(1, 1, K, W, dtype=torch.float16),
        torch.zeros(1, 1, ctx, K, dtype=torch.float16),
        torch.zeros(1, K, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_f, dtype=torch.float16),
        torch.zeros(ns, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(ns, nkv, W, max_hd, dtype=torch.float16),
        torch.zeros(nf, nkv, ctx, max_hd, dtype=torch.float16),
        torch.zeros(nf, nkv, ctx, max_hd, dtype=torch.float16),
    )
    inputs = [
        ct.TensorType(name="hidden_states",       shape=sample[0].shape,  dtype=np.float16),
        ct.TensorType(name="causal_mask_full",    shape=sample[1].shape,  dtype=np.float16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape,  dtype=np.float16),
        ct.TensorType(name="update_indicator",    shape=sample[3].shape,  dtype=np.float16),
        ct.TensorType(name="per_layer_combined",  shape=sample[4].shape,  dtype=np.float16),
        ct.TensorType(name="cos_s",               shape=sample[5].shape,  dtype=np.float16),
        ct.TensorType(name="sin_s",               shape=sample[6].shape,  dtype=np.float16),
        ct.TensorType(name="cos_f",               shape=sample[7].shape,  dtype=np.float16),
        ct.TensorType(name="sin_f",               shape=sample[8].shape,  dtype=np.float16),
        ct.TensorType(name="K_sliding_in",        shape=sample[9].shape,  dtype=np.float16),
        ct.TensorType(name="V_sliding_in",        shape=sample[10].shape, dtype=np.float16),
        ct.TensorType(name="K_full_in",           shape=sample[11].shape, dtype=np.float16),
        ct.TensorType(name="V_full_in",           shape=sample[12].shape, dtype=np.float16),
    ]
    outputs = ["hidden_states_out",
               "new_K_sliding", "new_V_sliding", "new_K_full", "new_V_full",
               "kv13_k", "kv13_v", "kv14_k", "kv14_v"]
    m = _convert_with_palettize(mc, sample, inputs, outputs,
                                 label="chunk2_3way_verify")
    _save(m, out_pkg)


# ----------------------------------------------------------------------
# chunk3_3way verify (L25-34 + LM head, Q=K)
# Reuses SWAVerifyChunk4 with start=25, end=35 (E2B). Boundaries pulled
# from compute_chunk_boundaries so E4B variants also work.
# ----------------------------------------------------------------------

def build_chunk3_3way_verify(base, ctx: int, K: int, out_pkg: str) -> None:
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    nkv = cfg.num_key_value_heads

    boundaries = compute_chunk_boundaries(cfg)
    c4_start, c4_end = boundaries[3]
    print(f"\n=== chunk3_3way verify_qK (L{c4_start}-{c4_end-1} + LM head, K={K}) ===")

    swa = SWAVerifyChunk4(base, seq_len=K, start=c4_start, end=c4_end).eval()

    sample = (
        torch.zeros(1, K, hidden, dtype=torch.float16),
        torch.zeros(1, 1, K, ctx, dtype=torch.float16),
        torch.zeros(1, 1, K, W, dtype=torch.float16),
        torch.zeros(1, K, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, K, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, W, hd_s, dtype=torch.float16),
        torch.zeros(1, nkv, ctx, hd_f, dtype=torch.float16),
        torch.zeros(1, nkv, ctx, hd_f, dtype=torch.float16),
    )
    inputs = [
        ct.TensorType(name="hidden_states",       shape=sample[0].shape,  dtype=np.float16),
        ct.TensorType(name="causal_mask_full",    shape=sample[1].shape,  dtype=np.float16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape,  dtype=np.float16),
        ct.TensorType(name="per_layer_combined",  shape=sample[3].shape,  dtype=np.float16),
        ct.TensorType(name="cos_s",               shape=sample[4].shape,  dtype=np.float16),
        ct.TensorType(name="sin_s",               shape=sample[5].shape,  dtype=np.float16),
        ct.TensorType(name="cos_f",               shape=sample[6].shape,  dtype=np.float16),
        ct.TensorType(name="sin_f",               shape=sample[7].shape,  dtype=np.float16),
        ct.TensorType(name="kv13_k",              shape=sample[8].shape,  dtype=np.float16),
        ct.TensorType(name="kv13_v",              shape=sample[9].shape,  dtype=np.float16),
        ct.TensorType(name="kv14_k",              shape=sample[10].shape, dtype=np.float16),
        ct.TensorType(name="kv14_v",              shape=sample[11].shape, dtype=np.float16),
    ]
    emit_logits = os.environ.get("MTP_EMIT_LOGITS", "0") == "1"
    outputs = ["token_ids", "hidden_states_out"]
    if emit_logits:
        outputs.append("logits_fp16")
    m = _convert_with_palettize(swa, sample, inputs, outputs,
                                 label="chunk3_3way_verify")
    _save(m, out_pkg)


# ----------------------------------------------------------------------
# Multifunction bundle: combine existing decode mlpackage + new verify
# mlpackage into single multifunction mlpackage with decode_q1 / verify_qK
# entry points (matches 4-chunk verify_chunks pattern).
# ----------------------------------------------------------------------

def bundle_multifunction(decode_pkg: str, verify_pkg: str, out_pkg: str) -> None:
    if not os.path.exists(decode_pkg):
        raise FileNotFoundError(f"decode mlpackage not found: {decode_pkg}")
    if not os.path.exists(verify_pkg):
        raise FileNotFoundError(f"verify mlpackage not found: {verify_pkg}")
    desc = MultiFunctionDescriptor()
    desc.add_function(decode_pkg, "main", "decode_q1")
    desc.add_function(verify_pkg, "main", "verify_qK")
    desc.default_function_name = "decode_q1"
    if os.path.exists(out_pkg):
        shutil.rmtree(out_pkg)
    save_multifunction(desc, out_pkg)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(out_pkg)
        for f in fns
    ) / 1024 / 1024
    print(f"  multifunction saved → {out_pkg}  ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--k", type=int, default=3,
                    help="verify_qK batch size (K=3 matches production drafter)")
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--decode-dir", default=None,
                    help="Existing decode-only chunks_3way_fp16kv directory")
    ap.add_argument("--output", default=None,
                    help="Output multifunction bundle directory")
    ap.add_argument("--only", choices=("chunk2", "chunk3"), default=None,
                    help="Only build one of the two verify chunks")
    args = ap.parse_args()

    if args.ctx is None and args.model in MODEL_REGISTRY:
        args.ctx = MODEL_REGISTRY[args.model].default_context_length
    elif args.ctx is None:
        args.ctx = 2048

    args.decode_dir = args.decode_dir or os.path.join(
        ROOT, "..", "output", args.model, "chunks_3way_fp16kv")
    args.output = args.output or os.path.join(
        ROOT, "..", "output", args.model, "chunks_3way_fp16kv_mf")
    os.makedirs(args.output, exist_ok=True)

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir}  ctx={args.ctx}  K={args.k}")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()

    decode_c2 = os.path.join(args.decode_dir, "chunk2_3way.mlpackage")
    decode_c3 = os.path.join(args.decode_dir, "chunk3_3way.mlpackage")
    verify_c2 = os.path.join(args.output, "chunk2_3way_verify.mlpackage")
    verify_c3 = os.path.join(args.output, "chunk3_3way_verify.mlpackage")
    mf_c2 = os.path.join(args.output, "chunk2_3way.mlpackage")
    mf_c3 = os.path.join(args.output, "chunk3_3way.mlpackage")

    if args.only in (None, "chunk2"):
        build_chunk2_3way_verify(base, args.ctx, args.k, verify_c2)
        bundle_multifunction(decode_c2, verify_c2, mf_c2)
    if args.only in (None, "chunk3"):
        build_chunk3_3way_verify(base, args.ctx, args.k, verify_c3)
        bundle_multifunction(decode_c3, verify_c3, mf_c3)

    print("\n" + "=" * 60)
    print(f"3-chunk multifunction bundle written to {args.output}/")
    print(f"Next: compile to .mlmodelc and drop into bundle_diff_logits/:")
    print(f"  xcrun coremlcompiler compile {mf_c2} <out_dir>/")
    print(f"  xcrun coremlcompiler compile {mf_c3} <out_dir>/")


if __name__ == "__main__":
    main()
