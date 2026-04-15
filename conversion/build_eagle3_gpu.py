#!/usr/bin/env python3
"""Build an EAGLE-3 draft mlpackage targeted at A19 Pro GPU (for Mirror SD).

Mirror Speculative Decoding (Apple 2026) runs:
  - Draft on GPU (compute-bound, benefits from tensor cores)
  - Verify on ANE (bandwidth-bound, ANE wins)
in parallel, for +30% over pure-ANE EAGLE-3.

This is the **scaffold** for Approach B of docs/UNEXPLORED_APPROACHES.md.
It produces a second mlpackage for the same trained EAGLE-3 draft, compiled
with `compute_units=.cpuAndGPU` instead of `.cpuAndNeuralEngine`.

The trained checkpoint (eagle3_draft_best.pt from train_eagle3_draft.ipynb)
is reused unchanged. Fusion mlpackage is also reused unchanged (stays on ANE
since its shapes are tiny and it's fine either place).

Usage:
    python conversion/build_eagle3_gpu.py \\
        --ckpt ./eagle3_draft/eagle3_draft_best.pt \\
        --output ./eagle3_draft_gpu.mlpackage

Decode path (verify_chunk*_K3) stays on ANE via its existing builder.
Swift side (see Sources/CoreMLLLM/MirrorSpeculativeLoop.swift) dispatches
draft and verify concurrently on separate DispatchQueues.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--output", type=str, default="./eagle3_draft_gpu.mlpackage")
    ap.add_argument("--palettize-int4", action="store_true",
                    help="INT4 weight palettization — note: GPU tensor cores prefer FP16/INT8 over INT4")
    ap.add_argument("--lm-head", type=str, default=None)
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    args = ap.parse_args()

    # Reuse the architecture + weight-loading logic from build_eagle3.py
    # (they are sibling files in the same directory).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_eagle3", str(Path(__file__).parent / "build_eagle3.py"))
    be3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(be3)

    import torch
    import coremltools as ct

    # Config
    cfg_path = args.config or str(Path(args.ckpt).parent / "eagle3_config.json")
    with open(cfg_path) as f:
        raw = json.load(f)
    cfg = {
        "hidden":     raw["hidden"],
        "num_heads":  raw["num_heads"],
        "num_kv":     raw["num_kv_heads"],
        "head_dim":   raw["head_dim"],
        "ffn":        raw["ffn"],
        "vocab":      raw["vocab"],
        "rms_eps":    raw["rms_eps"],
        "rope_theta": raw["rope_theta"],
        "embed_scale": raw["embed_scale"],
        "fusion_layers": raw["fusion_layers"],
    }
    print(f"config: hidden={cfg['hidden']} heads={cfg['num_heads']} head_dim={cfg['head_dim']}")

    # Load ckpt
    print(f"loading {args.ckpt}...")
    state = torch.load(args.ckpt, map_location="cpu")

    # lm_head
    if args.lm_head:
        lm_head_weight = torch.load(args.lm_head, map_location="cpu")
    else:
        print(f"fetching lm_head from {args.model_id}")
        try:
            from transformers import Gemma4ForConditionalGeneration as TCls
        except Exception:
            from transformers import AutoModelForCausalLM as TCls
        tgt = TCls.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="cpu")
        lm_head_weight = tgt.lm_head.weight.data.detach().clone().to(torch.float16)
        del tgt

    # Build ANE-friendly model (same class; GPU runs the same graph fine)
    ane = be3.EAGLE3DraftANE(cfg).to(torch.float16).eval()
    be3.load_into_ane_model(ane, state, lm_head_weight)

    H = cfg["hidden"]
    dummy_h = torch.zeros((1, 1, H), dtype=torch.float16)
    dummy_e = torch.zeros((1, 1, H), dtype=torch.float16)
    with torch.no_grad():
        _ = ane(dummy_h, dummy_e)
    print("forward OK")

    # Convert with compute_units=CPU_AND_GPU
    print("\nconverting (compute_units=cpuAndGPU)...")
    traced = torch.jit.trace(ane, (dummy_h, dummy_e), strict=False)
    mlm = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="h_prev", shape=(1, 1, H), dtype=ct.converters.mil.mil.types.fp16),
            ct.TensorType(name="e_next", shape=(1, 1, H), dtype=ct.converters.mil.mil.types.fp16),
        ],
        outputs=[
            ct.TensorType(name="h_out"),
            ct.TensorType(name="token"),
            ct.TensorType(name="logit"),
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_GPU,              # <- KEY DIFF from build_eagle3.py
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS26,
    )
    if args.palettize_int4:
        print("  palettizing weights INT4 (group_size=32) — note: may regress GPU perf vs FP16")
        import coremltools.optimize.coreml as cto
        cfg_q = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=4,
                                                  granularity="per_grouped_channel", group_size=32)
        )
        mlm = cto.palettize_weights(mlm, cfg_q)

    mlm.save(args.output)
    size_mb = sum(f.stat().st_size for f in Path(args.output).rglob("*") if f.is_file()) / 1e6
    print(f"\nsaved: {args.output} ({size_mb:.1f} MB)")

    # Sidecar marker (Swift reads this to confirm compute preference)
    (Path(args.output) / "compute_preference.json").write_text(json.dumps({
        "preferred_compute_units": "cpuAndGPU",
        "for_approach": "Mirror Speculative Decoding (docs/UNEXPLORED_APPROACHES.md §B)",
        "companion_ane_path": "eagle3_draft.mlpackage (built by build_eagle3.py)",
    }, indent=2))

    print("\nSwift load-time expectation:")
    print("  let cfg = MLModelConfiguration()")
    print("  cfg.computeUnits = .cpuAndGPU  // draft on GPU tensor cores")
    print("  let draftGPU = try MLModel(contentsOf: eagle3_draft_gpu.mlpackage, configuration: cfg)")
    print("\nPair with ANE verify chunks for parallel Mirror SD dispatch.")


if __name__ == "__main__":
    main()
