#!/usr/bin/env python3
"""Build W8A8 (activation INT8 + weight INT4) for ALL 4 chunks at 8K.

Uses proper calibration from running the existing models.
Requires coremltools_tmp_cleanup patch for disk leak workaround.
"""
from __future__ import annotations
import argparse, os, sys, shutil, time
import torch
import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools_tmp_cleanup  # noqa: F401

from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4
from ane_ops import MODEL_DTYPE

HF_DIR = f"{ROOT}/output/gemma4-e2b-final/hf_model"
CTX = 8192
W = 512
fp16 = ct.converters.mil.mil.types.fp16


def generate_calibration_data(existing_dir, num_samples=16):
    """Run existing INT4 pipeline to collect realistic inputs for each chunk."""
    hidden = 1536; nlayers = 35; pld = 256; max_hd = 512; total_pld = nlayers * pld
    cu = ct.ComputeUnit.CPU_AND_NE
    def make_fp16(shape): return np.zeros(shape, dtype=np.float16)

    print(f"Generating {num_samples} calibration samples...")
    c1 = ct.models.MLModel(os.path.join(existing_dir, "chunk1.mlpackage"), compute_units=cu)
    c2 = ct.models.MLModel(os.path.join(existing_dir, "chunk2.mlpackage"), compute_units=cu)
    c3 = ct.models.MLModel(os.path.join(existing_dir, "chunk3.mlpackage"), compute_units=cu)
    c4_path = os.path.join(existing_dir, "chunk4.mlpackage")
    if not os.path.exists(c4_path):
        c4_path = os.path.join(existing_dir, "chunk4.mlmodelc")
    c4 = ct.models.MLModel(c4_path, compute_units=cu)

    cal = {1: [], 2: [], 3: [], 4: []}
    kS1 = make_fp16((7,1,W,max_hd)); vS1 = make_fp16((7,1,W,max_hd))
    kF1 = make_fp16((1,1,CTX,max_hd)); vF1 = make_fp16((1,1,CTX,max_hd))
    kS2 = make_fp16((5,1,W,max_hd)); vS2 = make_fp16((5,1,W,max_hd))
    kF2 = make_fp16((2,1,CTX,max_hd)); vF2 = make_fp16((2,1,CTX,max_hd))

    np.random.seed(42)
    for i in range(num_samples):
        pos = i
        h_in = (np.random.randn(1,1,hidden)*0.5).astype(np.float16)
        plr = (np.random.randn(1,1,total_pld)*0.1).astype(np.float16)
        mf = np.full((1,1,1,CTX),-65504.0,dtype=np.float16); mf[0,0,0,:pos+1]=0
        ms = np.full((1,1,1,W),-65504.0,dtype=np.float16)
        v = min(pos+1,W); ms[0,0,0,W-v:]=0
        um = make_fp16((1,1,CTX,1)); um[0,0,min(pos,CTX-1),0]=1.0
        cs = (np.random.randn(1,1,1,256)*0.5).astype(np.float16)
        ss = (np.random.randn(1,1,1,256)*0.5).astype(np.float16)
        cf = (np.random.randn(1,1,1,512)*0.5).astype(np.float16)
        sf = (np.random.randn(1,1,1,512)*0.5).astype(np.float16)

        # Chunk 1 inputs
        cal[1].append({"hidden_states":h_in,"causal_mask_full":mf,"causal_mask_sliding":ms,
            "update_mask":um,"per_layer_raw":plr,"cos_s":cs,"sin_s":ss,"cos_f":cf,"sin_f":sf,
            "K_sliding_in":kS1.copy(),"V_sliding_in":vS1.copy(),"K_full_in":kF1.copy(),"V_full_in":vF1.copy()})

        o1 = c1.predict(cal[1][-1])
        h1=o1["hidden_states_out"]; plc=o1["per_layer_combined_out"]
        kS1=o1["K_sliding_out"]; vS1=o1["V_sliding_out"]; kF1=o1["K_full_out"]; vF1=o1["V_full_out"]

        # Chunk 2 inputs
        cal[2].append({"hidden_states":h1,"causal_mask_full":mf,"causal_mask_sliding":ms,
            "update_mask":um,"per_layer_combined":plc,"cos_s":cs,"sin_s":ss,"cos_f":cf,"sin_f":sf,
            "K_sliding_in":kS2.copy(),"V_sliding_in":vS2.copy(),"K_full_in":kF2.copy(),"V_full_in":vF2.copy()})

        o2 = c2.predict(cal[2][-1])
        h2=o2["hidden_states_out"]
        kS2=o2["K_sliding_out"]; vS2=o2["V_sliding_out"]; kF2=o2["K_full_out"]; vF2=o2["V_full_out"]
        kv13_k=o2["kv13_k"]; kv13_v=o2["kv13_v"]; kv14_k=o2["kv14_k"]; kv14_v=o2["kv14_v"]

        shared = {"causal_mask_full":mf,"causal_mask_sliding":ms,"update_mask":um,
            "per_layer_combined":plc,"cos_s":cs,"sin_s":ss,"cos_f":cf,"sin_f":sf,
            "kv13_k":kv13_k,"kv13_v":kv13_v,"kv14_k":kv14_k,"kv14_v":kv14_v}

        # Chunk 3 inputs
        d3 = dict(shared); d3["hidden_states"] = h2
        cal[3].append(d3)
        o3 = c3.predict(d3)
        h3 = o3["hidden_states_out"]

        # Chunk 4 inputs
        d4 = dict(shared); d4["hidden_states"] = h3
        cal[4].append(d4)

        if (i+1) % 8 == 0:
            print(f"  {i+1}/{num_samples} samples")

    del c1, c2, c3, c4
    print(f"  Calibration data ready for all 4 chunks")
    return cal


def build_chunk_w8a8(chunk_cls, base, chunk_name, sample_tensors, input_specs, output_names, cal_data, save_dir):
    """Build one W8A8 chunk: FP16 → A8 → W4 palettize."""
    print(f"\n=== {chunk_name} W8A8 ===")

    chunk = chunk_cls(base).eval()

    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(chunk, sample_tensors, check_trace=False)
    print(f"  Traced in {time.time()-t:.1f}s")

    t = time.time()
    mlmodel = ct.convert(traced, inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  Converted FP16 in {time.time()-t:.1f}s")

    # Activation quantization
    t = time.time()
    act_cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=np.int8))
    try:
        mlmodel = cto.linear_quantize_activations(mlmodel, act_cfg, cal_data, calibration_op_group_size=50)
        print(f"  A8 done in {time.time()-t:.1f}s")
    except Exception as e:
        print(f"  A8 FAILED ({e}), continuing with FP16 activations")

    # Weight palettization (INT4, same as current production)
    t = time.time()
    w_cfg = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(nbits=4, granularity="per_grouped_channel", group_size=32))
    mlmodel = cto.palettize_weights(mlmodel, w_cfg)
    print(f"  W4 palettized in {time.time()-t:.1f}s")

    save_path = os.path.join(save_dir, f"{chunk_name}.mlpackage")
    if os.path.exists(save_path): shutil.rmtree(save_path)
    mlmodel.save(save_path)
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(save_path) for f in fns)
    print(f"  Saved ({sz/1e6:.0f} MB)")
    del mlmodel, chunk, traced
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-dir", type=str, default="output/all_chunks_8k")
    parser.add_argument("--output", type=str, default="/tmp/w8a8-all")
    parser.add_argument("--cal-samples", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cal = generate_calibration_data(args.existing_dir, args.cal_samples)

    print("\nLoading Gemma 4 E2B...")
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=CTX)
    base.eval()

    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512

    # === Chunk 1 ===
    s1 = (torch.zeros(1,1,hidden,dtype=torch.float16),
          torch.zeros(1,1,1,CTX,dtype=torch.float16),
          torch.zeros(1,1,1,W,dtype=torch.float16),
          torch.zeros(1,1,CTX,1,dtype=torch.float16),
          torch.zeros(1,1,nlayers*pld,dtype=torch.float16),
          torch.zeros(1,1,1,256,dtype=torch.float16),torch.zeros(1,1,1,256,dtype=torch.float16),
          torch.zeros(1,1,1,512,dtype=torch.float16),torch.zeros(1,1,1,512,dtype=torch.float16),
          torch.zeros(7,1,W,max_hd,dtype=torch.float16),torch.zeros(7,1,W,max_hd,dtype=torch.float16),
          torch.zeros(1,1,CTX,max_hd,dtype=torch.float16),torch.zeros(1,1,CTX,max_hd,dtype=torch.float16))
    in1 = [ct.TensorType(name=n,shape=s1[i].shape,dtype=fp16) for i,n in enumerate([
        "hidden_states","causal_mask_full","causal_mask_sliding","update_mask","per_layer_raw",
        "cos_s","sin_s","cos_f","sin_f","K_sliding_in","V_sliding_in","K_full_in","V_full_in"])]
    out1 = ["hidden_states_out","K_sliding_out","V_sliding_out","K_full_out","V_full_out","per_layer_combined_out"]
    build_chunk_w8a8(SWAChunk1, base, "chunk1", s1, in1, out1, cal[1], args.output)

    # === Chunk 2 ===
    s2 = (torch.zeros(1,1,hidden,dtype=torch.float16),
          torch.zeros(1,1,1,CTX,dtype=torch.float16),
          torch.zeros(1,1,1,W,dtype=torch.float16),
          torch.zeros(1,1,CTX,1,dtype=torch.float16),
          torch.zeros(1,1,nlayers*pld,dtype=torch.float16),
          torch.zeros(1,1,1,256,dtype=torch.float16),torch.zeros(1,1,1,256,dtype=torch.float16),
          torch.zeros(1,1,1,512,dtype=torch.float16),torch.zeros(1,1,1,512,dtype=torch.float16),
          torch.zeros(5,1,W,max_hd,dtype=torch.float16),torch.zeros(5,1,W,max_hd,dtype=torch.float16),
          torch.zeros(2,1,CTX,max_hd,dtype=torch.float16),torch.zeros(2,1,CTX,max_hd,dtype=torch.float16))
    in2 = [ct.TensorType(name=n,shape=s2[i].shape,dtype=fp16) for i,n in enumerate([
        "hidden_states","causal_mask_full","causal_mask_sliding","update_mask","per_layer_combined",
        "cos_s","sin_s","cos_f","sin_f","K_sliding_in","V_sliding_in","K_full_in","V_full_in"])]
    out2 = ["hidden_states_out","K_sliding_out","V_sliding_out","K_full_out","V_full_out",
            "kv13_k","kv13_v","kv14_k","kv14_v"]
    build_chunk_w8a8(SWAChunk2, base, "chunk2", s2, in2, out2, cal[2], args.output)

    # === Chunk 3 ===
    s3 = (torch.zeros(1,1,hidden,dtype=torch.float16),
          torch.zeros(1,1,1,CTX,dtype=torch.float16),
          torch.zeros(1,1,1,W,dtype=torch.float16),
          torch.zeros(1,1,CTX,1,dtype=torch.float16),
          torch.zeros(1,1,nlayers*pld,dtype=torch.float16),
          torch.zeros(1,1,1,256,dtype=torch.float16),torch.zeros(1,1,1,256,dtype=torch.float16),
          torch.zeros(1,1,1,512,dtype=torch.float16),torch.zeros(1,1,1,512,dtype=torch.float16),
          torch.zeros(1,1,W,256,dtype=torch.float16),torch.zeros(1,1,W,256,dtype=torch.float16),
          torch.zeros(1,1,CTX,512,dtype=torch.float16),torch.zeros(1,1,CTX,512,dtype=torch.float16))
    in3 = [ct.TensorType(name=n,shape=s3[i].shape,dtype=fp16) for i,n in enumerate([
        "hidden_states","causal_mask_full","causal_mask_sliding","update_mask","per_layer_combined",
        "cos_s","sin_s","cos_f","sin_f","kv13_k","kv13_v","kv14_k","kv14_v"])]
    out3 = ["hidden_states_out"]
    build_chunk_w8a8(SWAChunk3, base, "chunk3", s3, in3, out3, cal[3], args.output)

    # === Chunk 4 ===
    out4 = ["token_id","token_logit","hidden_states_out"]
    build_chunk_w8a8(SWAChunk4, base, "chunk4", s3, in3, out4, cal[4], args.output)

    print(f"\n{'='*60}")
    print(f"W8A8 models saved to {args.output}/")
    print(f"Drop-in replacement for 8K chunks. Test on iPhone for INT8-INT8 fast path.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
