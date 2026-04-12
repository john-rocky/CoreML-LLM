#!/usr/bin/env python3
"""W8A8 quantization with proper calibration data.

Strategy:
1. Build FP16 (unquantized) model from HF weights
2. Run existing INT4 models to collect realistic intermediate states
3. Use those states as calibration data for activation quantization
4. Apply weight INT8 quantization on top
5. Benchmark

Starts with chunk2 (the 8K bottleneck) as proof of concept.
"""
from __future__ import annotations
import argparse, os, sys, shutil, time
import torch
import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4
from ane_ops import MODEL_DTYPE

HF_DIR = f"{ROOT}/output/gemma4-e2b-final/hf_model"
CTX = 8192
W = 512
fp16 = ct.converters.mil.mil.types.fp16


def generate_calibration_data(existing_chunks_dir, num_samples=32):
    """Run existing INT4 models to collect realistic activation data for each chunk."""
    hidden = 1536; nlayers = 35; pld = 256; max_hd = 512; total_pld = nlayers * pld

    print(f"Generating {num_samples} calibration samples from existing models...")
    cu = ct.ComputeUnit.CPU_AND_NE

    c1 = ct.models.MLModel(os.path.join(existing_chunks_dir, "chunk1.mlpackage"), compute_units=cu)
    c2_ref = ct.models.MLModel(os.path.join(existing_chunks_dir, "chunk2.mlpackage"), compute_units=cu)

    def make_fp16(shape): return np.zeros(shape, dtype=np.float16)

    # Collect chunk2 inputs by running chunk1 with various positions
    cal_chunk2 = []

    # Init KV buffers
    kS1 = make_fp16((7, 1, W, max_hd)); vS1 = make_fp16((7, 1, W, max_hd))
    kF1 = make_fp16((1, 1, CTX, max_hd)); vF1 = make_fp16((1, 1, CTX, max_hd))
    kS2 = make_fp16((5, 1, W, max_hd)); vS2 = make_fp16((5, 1, W, max_hd))
    kF2 = make_fp16((2, 1, CTX, max_hd)); vF2 = make_fp16((2, 1, CTX, max_hd))

    np.random.seed(42)
    for i in range(num_samples):
        pos = i
        # Realistic hidden states: random with model-like scale
        h_in = (np.random.randn(1, 1, hidden) * 0.5).astype(np.float16)
        plr = (np.random.randn(1, 1, total_pld) * 0.1).astype(np.float16)

        mask_full = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask_full[0, 0, 0, :pos+1] = 0
        mask_sliding = np.full((1, 1, 1, W), -65504.0, dtype=np.float16)
        valid = min(pos + 1, W); mask_sliding[0, 0, 0, W - valid:] = 0
        umask = make_fp16((1, 1, CTX, 1))
        umask[0, 0, min(pos, CTX - 1), 0] = 1.0

        cos_s = (np.random.randn(1, 1, 1, 256) * 0.5).astype(np.float16)
        sin_s = (np.random.randn(1, 1, 1, 256) * 0.5).astype(np.float16)
        cos_f = (np.random.randn(1, 1, 1, 512) * 0.5).astype(np.float16)
        sin_f = (np.random.randn(1, 1, 1, 512) * 0.5).astype(np.float16)

        # Run chunk1 to get realistic outputs
        out1 = c1.predict({
            "hidden_states": h_in,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "update_mask": umask,
            "per_layer_raw": plr,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "K_sliding_in": kS1, "V_sliding_in": vS1,
            "K_full_in": kF1, "V_full_in": vF1,
        })

        h1 = out1["hidden_states_out"]
        plc = out1["per_layer_combined_out"]
        kS1 = out1["K_sliding_out"]; vS1 = out1["V_sliding_out"]
        kF1 = out1["K_full_out"]; vF1 = out1["V_full_out"]

        # Collect chunk2 input — this is realistic calibration data
        cal_chunk2.append({
            "hidden_states": h1,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "update_mask": umask,
            "per_layer_combined": plc,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "K_sliding_in": kS2.copy(), "V_sliding_in": vS2.copy(),
            "K_full_in": kF2.copy(), "V_full_in": vF2.copy(),
        })

        # Also run chunk2 to update its KV state
        out2 = c2_ref.predict({
            "hidden_states": h1,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "update_mask": umask,
            "per_layer_combined": plc,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "K_sliding_in": kS2, "V_sliding_in": vS2,
            "K_full_in": kF2, "V_full_in": vF2,
        })
        kS2 = out2["K_sliding_out"]; vS2 = out2["V_sliding_out"]
        kF2 = out2["K_full_out"]; vF2 = out2["V_full_out"]

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{num_samples} samples")

    print(f"  Done! {len(cal_chunk2)} calibration samples for chunk2")
    return cal_chunk2


def build_w8a8_chunk2(cal_data, output_dir):
    """Build W8A8 chunk2 with proper calibration."""
    hidden = 1536; pld = 256; nlayers = 35; max_hd = 512

    print("\nBuilding FP16 chunk2 from HF weights...")
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=CTX)
    base.eval()
    c2 = SWAChunk2(base).eval()

    s2 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, CTX, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, CTX, max_hd, dtype=torch.float16),
    )
    in2 = [
        ct.TensorType(name="hidden_states", shape=s2[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full", shape=s2[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s2[2].shape, dtype=fp16),
        ct.TensorType(name="update_mask", shape=s2[3].shape, dtype=fp16),
        ct.TensorType(name="per_layer_combined", shape=s2[4].shape, dtype=fp16),
        ct.TensorType(name="cos_s", shape=s2[5].shape, dtype=fp16),
        ct.TensorType(name="sin_s", shape=s2[6].shape, dtype=fp16),
        ct.TensorType(name="cos_f", shape=s2[7].shape, dtype=fp16),
        ct.TensorType(name="sin_f", shape=s2[8].shape, dtype=fp16),
        ct.TensorType(name="K_sliding_in", shape=s2[9].shape, dtype=fp16),
        ct.TensorType(name="V_sliding_in", shape=s2[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in", shape=s2[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in", shape=s2[12].shape, dtype=fp16),
    ]
    out2 = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
            "K_full_out", "V_full_out", "kv13_k", "kv13_v", "kv14_k", "kv14_v"]

    # 1. Trace and convert to FP16 CoreML
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(c2, s2, check_trace=False)
    print(f"  Traced in {time.time()-t:.1f}s")

    t = time.time()
    mlmodel = ct.convert(
        traced, inputs=in2,
        outputs=[ct.TensorType(name=n) for n in out2],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"  Converted FP16 in {time.time()-t:.1f}s")

    # 2. Activation quantization with proper calibration
    print(f"\n  Applying activation quantization ({len(cal_data)} samples)...")
    act_cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8,
        ))
    t = time.time()
    try:
        mlmodel_a8 = cto.linear_quantize_activations(
            mlmodel, act_cfg, cal_data,
            calibration_op_group_size=10  # process in groups to avoid memory issues
        )
        print(f"  Activation quantization done in {time.time()-t:.1f}s")
    except Exception as e:
        print(f"  Activation quantization FAILED: {e}")
        print("  Falling back to weight-only INT8...")
        mlmodel_a8 = mlmodel

    # 3. Try W4A8 (INT4 palette + A8) — more stable than W8A8
    t = time.time()
    try:
        w_cfg = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                nbits=4, granularity="per_grouped_channel", group_size=32))
        mlmodel_final = cto.palettize_weights(mlmodel_a8, w_cfg)
        print(f"  W4A8 (INT4 palette + A8) done in {time.time()-t:.1f}s")
    except Exception as e:
        print(f"  W4A8 failed ({e}), saving A8-only...")
        mlmodel_final = mlmodel_a8

    # Save
    save_path = os.path.join(output_dir, "chunk2.mlpackage")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    mlmodel_final.save(save_path)
    print(f"  Saved to {save_path}")
    return save_path


def benchmark_chunk2(model_path, label):
    """Quick benchmark of chunk2."""
    hidden = 1536; nlayers = 35; pld = 256; max_hd = 512
    m = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    def make_fp16(shape): return np.zeros(shape, dtype=np.float16)

    inp = {
        "hidden_states": make_fp16((1, 1, hidden)),
        "causal_mask_full": np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16),
        "causal_mask_sliding": np.full((1, 1, 1, W), -65504.0, dtype=np.float16),
        "update_mask": make_fp16((1, 1, CTX, 1)),
        "per_layer_combined": make_fp16((1, 1, nlayers * pld)),
        "cos_s": make_fp16((1, 1, 1, 256)), "sin_s": make_fp16((1, 1, 1, 256)),
        "cos_f": make_fp16((1, 1, 1, 512)), "sin_f": make_fp16((1, 1, 1, 512)),
        "K_sliding_in": make_fp16((5, 1, W, max_hd)), "V_sliding_in": make_fp16((5, 1, W, max_hd)),
        "K_full_in": make_fp16((2, 1, CTX, max_hd)), "V_full_in": make_fp16((2, 1, CTX, max_hd)),
    }
    # Warmup
    for _ in range(5): m.predict(inp)
    # Bench
    times = []
    for _ in range(25):
        t0 = time.time(); m.predict(inp); times.append(time.time() - t0)
    times = times[5:]
    mean = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"  {label}: {mean:.1f}ms ±{std:.1f}")
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-dir", type=str, default="output/all_chunks_8k",
                        help="Directory with existing INT4 models for calibration")
    parser.add_argument("--output", type=str, default="/tmp/w8a8-proper")
    parser.add_argument("--cal-samples", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Generate calibration data
    cal_data = generate_calibration_data(args.existing_dir, args.cal_samples)

    # Build W8A8 chunk2
    model_path = build_w8a8_chunk2(cal_data, args.output)

    # Benchmark comparison
    print("\n=== Benchmark ===")
    baseline = benchmark_chunk2(
        os.path.join(args.existing_dir, "chunk2.mlpackage"), "INT4 baseline")
    w8a8 = benchmark_chunk2(model_path, "W8A8")
    print(f"\n  Speedup: {baseline/w8a8:.2f}x ({(1-w8a8/baseline)*100:.1f}%)")


if __name__ == "__main__":
    main()
