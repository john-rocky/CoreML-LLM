#!/usr/bin/env python3
"""Build Gemma 4 prefill chunks as multifunction mlpackages.

For each of the 4 prefill chunks, export one .mlpackage that contains
multiple batch-size variants (e.g. N=64/128/256/512) as separate functions
sharing the same palletized weights via `ct.utils.save_multifunction`.

Weight dedup is validated by `conversion/spikes/multifunction_prefill_spike.py`
(1.00x merged/larger ratio on the stand-in, perfect dedup).

Output layout (matches what the Swift router expects):

    output/prefill_chunk1.mlpackage    # functions: prefill_b64, prefill_b128,
    output/prefill_chunk2.mlpackage    #            prefill_b256, prefill_b512
    output/prefill_chunk3.mlpackage    # default function: prefill_b512
    output/prefill_chunk4.mlpackage    #   (matches single-variant legacy shape)

Backward compat: apps that load the mlpackage without specifying
function_name get the default `prefill_b512`, which has the same shape as
the existing single-variant export → no migration required.

Usage:
    python conversion/build_prefill_multifunction.py \\
        --hf-dir ./output/gemma4-e2b/hf_model \\
        --output ./output/gemma4-e2b/prefill \\
        --sizes 64 128 256 512
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models import gemma4_prefill_chunks as _pc_mod
from models.gemma4_prefill_chunks import (
    PrefillChunk1, PrefillChunk2, PrefillChunk3, PrefillChunk4,
    chunk_kv_layout, chunk_output_names,
)
from models.gemma4_swa_chunks import compute_chunk_boundaries

fp16 = ct.converters.mil.mil.types.fp16

CTX_DEFAULT = 2048


def convert_variant(model, sample_inputs, input_specs, output_names,
                    save_path: str, quantize: bool):
    """Trace + ct.convert + palletize + save one single-function mlpackage."""
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_inputs, check_trace=False)
    print(f"    traced in {time.time()-t:.1f}s")

    t = time.time()
    mlmodel = ct.convert(
        traced, inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"    converted in {time.time()-t:.1f}s")

    if quantize:
        t = time.time()
        cfg = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                nbits=4, granularity="per_grouped_channel", group_size=32))
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg)
        print(f"    palettized in {time.time()-t:.1f}s")

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    mlmodel.save(save_path)


def _rope_shapes(N: int, hd: int, ghd: int):
    """RoPE input shapes common to every chunk."""
    return (
        torch.zeros(1, 1, N, hd,  dtype=torch.float16),  # cos_s
        torch.zeros(1, 1, N, hd,  dtype=torch.float16),  # sin_s
        torch.zeros(1, 1, N, ghd, dtype=torch.float16),  # cos_f
        torch.zeros(1, 1, N, ghd, dtype=torch.float16),  # sin_f
    )


def _chunk1_specs(N: int, hidden: int, total_pld: int, config,
                   start: int, end: int):
    hd = config.head_dim
    ghd = config.global_head_dim
    shapes_core = (
        torch.zeros(1, N, hidden, dtype=torch.float16),
        torch.zeros(1, 1, N, N, dtype=torch.float16),
        torch.zeros(1, N, total_pld, dtype=torch.float16),
    )
    rope = _rope_shapes(N, hd, ghd)
    shapes = shapes_core + rope
    inputs = [
        ct.TensorType(name="hidden_states", shape=shapes[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask",   shape=shapes[1].shape, dtype=fp16),
        ct.TensorType(name="per_layer_raw", shape=shapes[2].shape, dtype=fp16),
        ct.TensorType(name="cos_s",         shape=shapes[3].shape, dtype=fp16),
        ct.TensorType(name="sin_s",         shape=shapes[4].shape, dtype=fp16),
        ct.TensorType(name="cos_f",         shape=shapes[5].shape, dtype=fp16),
        ct.TensorType(name="sin_f",         shape=shapes[6].shape, dtype=fp16),
    ]
    outputs = chunk_output_names(1, start, end, config)
    return shapes, inputs, outputs


def _chunk2_specs(N: int, hidden: int, total_pld: int, config,
                   start: int, end: int):
    hd = config.head_dim
    ghd = config.global_head_dim
    shapes_core = (
        torch.zeros(1, N, hidden, dtype=torch.float16),
        torch.zeros(1, 1, N, N, dtype=torch.float16),
        torch.zeros(1, N, total_pld, dtype=torch.float16),
    )
    rope = _rope_shapes(N, hd, ghd)
    shapes = shapes_core + rope
    inputs = [
        ct.TensorType(name="hidden_states",      shape=shapes[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask",        shape=shapes[1].shape, dtype=fp16),
        ct.TensorType(name="per_layer_combined", shape=shapes[2].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=shapes[3].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=shapes[4].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=shapes[5].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=shapes[6].shape, dtype=fp16),
    ]
    outputs = chunk_output_names(2, start, end, config)
    return shapes, inputs, outputs


def _chunk3_specs(N: int, hidden: int, total_pld: int, config,
                   start: int, end: int):
    hd = config.head_dim
    ghd = config.global_head_dim
    nkv = config.num_key_value_heads
    shapes = (
        torch.zeros(1, N, hidden, dtype=torch.float16),
        torch.zeros(1, 1, N, N, dtype=torch.float16),
        torch.zeros(1, N, total_pld, dtype=torch.float16),
        torch.zeros(1, 1, N, hd,  dtype=torch.float16),  # cos_s
        torch.zeros(1, 1, N, hd,  dtype=torch.float16),  # sin_s
        torch.zeros(1, 1, N, ghd, dtype=torch.float16),  # cos_f
        torch.zeros(1, 1, N, ghd, dtype=torch.float16),  # sin_f
        torch.zeros(1, nkv, N, hd,  dtype=torch.float16),  # kv13_k
        torch.zeros(1, nkv, N, hd,  dtype=torch.float16),  # kv13_v
        torch.zeros(1, nkv, N, ghd, dtype=torch.float16),  # kv14_k
        torch.zeros(1, nkv, N, ghd, dtype=torch.float16),  # kv14_v
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=shapes[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask",        shape=shapes[1].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined", shape=shapes[2].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",              shape=shapes[3].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",              shape=shapes[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",              shape=shapes[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",              shape=shapes[6].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",             shape=shapes[7].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",             shape=shapes[8].shape,  dtype=fp16),
        ct.TensorType(name="kv14_k",             shape=shapes[9].shape,  dtype=fp16),
        ct.TensorType(name="kv14_v",             shape=shapes[10].shape, dtype=fp16),
    ]
    outputs = chunk_output_names(3, start, end, config)
    return shapes, inputs, outputs


def _chunk4_specs(N: int, hidden: int, total_pld: int, config,
                   start: int, end: int):
    hd = config.head_dim
    ghd = config.global_head_dim
    nkv = config.num_key_value_heads
    shapes = (
        torch.zeros(1, N, hidden, dtype=torch.float16),
        torch.zeros(1, 1, N, N, dtype=torch.float16),
        torch.zeros(1, N, total_pld, dtype=torch.float16),
        torch.zeros(1, 1, N, hd,  dtype=torch.float16),  # cos_s
        torch.zeros(1, 1, N, hd,  dtype=torch.float16),  # sin_s
        torch.zeros(1, 1, N, ghd, dtype=torch.float16),  # cos_f
        torch.zeros(1, 1, N, ghd, dtype=torch.float16),  # sin_f
        torch.zeros(1, nkv, N, hd,  dtype=torch.float16),  # kv13_k
        torch.zeros(1, nkv, N, hd,  dtype=torch.float16),  # kv13_v
        torch.zeros(1, nkv, N, ghd, dtype=torch.float16),  # kv14_k
        torch.zeros(1, nkv, N, ghd, dtype=torch.float16),  # kv14_v
        torch.zeros(1, N, 1, dtype=torch.float16),
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=shapes[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask",        shape=shapes[1].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined", shape=shapes[2].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",              shape=shapes[3].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",              shape=shapes[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",              shape=shapes[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",              shape=shapes[6].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",             shape=shapes[7].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",             shape=shapes[8].shape,  dtype=fp16),
        ct.TensorType(name="kv14_k",             shape=shapes[9].shape,  dtype=fp16),
        ct.TensorType(name="kv14_v",             shape=shapes[10].shape, dtype=fp16),
        ct.TensorType(name="last_position_mask", shape=shapes[11].shape, dtype=fp16),
    ]
    outputs = chunk_output_names(4, start, end, config)
    return shapes, inputs, outputs


CHUNK_BUILDERS = {
    1: (PrefillChunk1, _chunk1_specs),
    2: (PrefillChunk2, _chunk2_specs),
    3: (PrefillChunk3, _chunk3_specs),
    4: (PrefillChunk4, _chunk4_specs),
}


def export_chunk_multifunction(base: Gemma4Model, chunk_idx: int, sizes: list[int],
                                out_dir: Path, tmp_dir: Path, quantize: bool,
                                default_size: int):
    """Build one multifunction prefill_chunk{chunk_idx}.mlpackage with
    functions `prefill_b{N}` for each N in `sizes`."""
    cls, spec_fn = CHUNK_BUILDERS[chunk_idx]
    cfg = base.config
    hidden = cfg.hidden_size
    total_pld = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
    boundaries = compute_chunk_boundaries(cfg)
    start, end = boundaries[chunk_idx - 1]

    per_variant_paths: list[tuple[int, str]] = []
    for N in sizes:
        print(f"\n--- chunk{chunk_idx} L{start}-{end-1} N={N} ---")
        # The chunk modules pick up N from the module-level PREFILL_N
        # constant (needed as a Python int so tensor.view()'s dim args
        # stay static across tracing). Flip it per variant.
        _pc_mod.PREFILL_N = N
        module = cls(base, start=start, end=end).eval()
        sample, inputs, outputs = spec_fn(N, hidden, total_pld, cfg,
                                           start=start, end=end)
        tmp_path = tmp_dir / f"_tmp_chunk{chunk_idx}_b{N}.mlpackage"
        convert_variant(module, sample, inputs, outputs, str(tmp_path), quantize)
        per_variant_paths.append((N, str(tmp_path)))

    # Restore default in case any other import references it.
    _pc_mod.PREFILL_N = 512

    # Single-variant short-circuit: skip the multifunction wrapper. The
    # wrapper carries a ~3x cold-load penalty on iPhone ANE even when only
    # one function is active (see docs/experiments/SESSION_2026_04_23.md §multifunction
    # prefill variants), so a one-size build should produce a plain
    # single-function mlpackage instead.
    if len(per_variant_paths) == 1:
        N, tmp_path = per_variant_paths[0]
        out_path = out_dir / f"prefill_chunk{chunk_idx}.mlpackage"
        if out_path.exists():
            shutil.rmtree(out_path)
        shutil.move(tmp_path, out_path)
        size_mb = sum(os.path.getsize(os.path.join(r, f))
                      for r, _, fs in os.walk(out_path) for f in fs) / 1024 / 1024
        print(f"    single-variant (N={N}): skipped save_multifunction wrap")
        print(f"    size: {size_mb:.1f}MB  → {out_path}")
        return

    print(f"\n--- merging chunk{chunk_idx} variants into multifunction mlpackage ---")
    desc = MultiFunctionDescriptor()
    for N, path in per_variant_paths:
        fname = f"prefill_b{N}"
        desc.add_function(path, src_function_name="main", target_function_name=fname)
        print(f"    added function {fname} from {os.path.basename(path)}")
    desc.default_function_name = f"prefill_b{default_size}"
    print(f"    default function: {desc.default_function_name}")

    out_path = out_dir / f"prefill_chunk{chunk_idx}.mlpackage"
    if out_path.exists():
        shutil.rmtree(out_path)
    save_multifunction(desc, str(out_path))

    # Report
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(out_path) for f in fs)
    sum_variants = sum(os.path.getsize(os.path.join(r, f))
                        for _, p in per_variant_paths
                        for r, _, fs in os.walk(p) for f in fs)
    print(f"    merged size: {total/1024/1024:.1f}MB  "
          f"(sum of {len(per_variant_paths)} variants: {sum_variants/1024/1024:.1f}MB, "
          f"saved: {(1 - total/sum_variants)*100:.1f}% via hash dedup)")

    for _, path in per_variant_paths:
        shutil.rmtree(path, ignore_errors=True)
    print(f"    → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True,
                    help="HuggingFace Gemma 4 E2B/E4B checkpoint directory")
    ap.add_argument("--output", required=True,
                    help="Output directory (will hold prefill_chunk{1..4}.mlpackage)")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[64, 128, 256, 512],
                    help="Static batch sizes to export (default: 64 128 256 512)")
    ap.add_argument("--default-size", type=int, default=512,
                    help="Which variant is the default function (default: 512 for backward compat)")
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="Which chunks to build (default: 1 2 3 4)")
    ap.add_argument("--context-length", type=int, default=CTX_DEFAULT,
                    help="Model context length (default 2048)")
    ap.add_argument("--no-quantize", action="store_true",
                    help="Skip palettization (FP16 debug build)")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_multifunction"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sizes = sorted(set(args.sizes))
    if args.default_size not in sizes:
        raise SystemExit(f"--default-size {args.default_size} must be in --sizes {sizes}")

    print(f"Loading Gemma 4 from {args.hf_dir}...")
    base = Gemma4Model.from_pretrained(args.hf_dir, context_length=args.context_length)
    base.eval()

    for idx in args.chunks:
        if idx not in CHUNK_BUILDERS:
            print(f"skipping unknown chunk {idx}")
            continue
        export_chunk_multifunction(
            base, chunk_idx=idx, sizes=sizes,
            out_dir=out_dir, tmp_dir=tmp_dir,
            quantize=not args.no_quantize,
            default_size=args.default_size,
        )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nAll multifunction prefill chunks exported.")
    print(f"Load with configuration.functionName = \"prefill_b<N>\" for N in {sizes}.")
    print(f"Default (no functionName set) resolves to prefill_b{args.default_size}.")


if __name__ == "__main__":
    main()
