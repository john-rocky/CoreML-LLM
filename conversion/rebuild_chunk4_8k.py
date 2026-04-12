#!/usr/bin/env python3
"""Rebuild chunk4 with CTX=8192 (fix: was compiled at 2048).

Usage:
    cd conversion && source .venv/bin/activate
    python rebuild_chunk4_8k.py
"""
import os, sys, shutil, time
import torch
import coremltools as ct

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import SWAChunk4
from ane_ops import MODEL_DTYPE

HF_DIR = f"{ROOT}/output/gemma4-e2b-final/hf_model"
CTX = 8192
W = 512
fp16 = ct.converters.mil.mil.types.fp16
OUTPUT = f"{ROOT}/output/chunk4_8k"


def main():
    os.makedirs(OUTPUT, exist_ok=True)

    print("Loading Gemma 4 E2B...")
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=CTX)
    base.eval()

    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers

    print(f"\n=== Chunk4 (decode, L25-34 + LM head) CTX={CTX} ===")
    swa4 = SWAChunk4(base).eval()

    s4 = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),      # causal_mask_full
        torch.zeros(1, 1, 1, W, dtype=torch.float16),         # causal_mask_sliding
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),       # update_mask
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),       # cos_s
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),       # sin_s
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),       # cos_f
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),       # sin_f
        torch.zeros(1, 1, W, 256, dtype=torch.float16),       # kv13_k
        torch.zeros(1, 1, W, 256, dtype=torch.float16),       # kv13_v
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),     # kv14_k
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),     # kv14_v
    )
    in4 = [
        ct.TensorType(name="hidden_states",       shape=s4[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s4[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s4[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s4[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s4[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s4[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s4[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s4[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s4[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=s4[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=s4[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=s4[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=s4[12].shape, dtype=fp16),
    ]
    out4 = ["token_id", "token_logit"]

    # Check if SWAChunk4 outputs hidden_states_out
    with torch.no_grad():
        test_out = swa4(*s4)
    if isinstance(test_out, tuple) and len(test_out) >= 3:
        out4.append("hidden_states_out")
        print("  (includes hidden_states_out)")

    print("Tracing...")
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(swa4, s4, check_trace=False)
    print(f"  traced in {time.time()-t:.1f}s")

    print("Converting to CoreML...")
    t = time.time()
    mlmodel = ct.convert(
        traced, inputs=in4,
        outputs=[ct.TensorType(name=n) for n in out4],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"  converted in {time.time()-t:.1f}s")

    print("Palettizing (INT4)...")
    t = time.time()
    cfg = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpPalettizerConfig(
            nbits=4, granularity="per_grouped_channel", group_size=32))
    mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg)
    print(f"  palettized in {time.time()-t:.1f}s")

    save_path = f"{OUTPUT}/chunk4.mlpackage"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    mlmodel.save(save_path)

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(save_path)
        for f in fns
    ) / 1024 / 1024
    print(f"\nSaved {save_path} ({size_mb:.1f} MB)")

    # Compile to .mlmodelc
    print("Compiling to .mlmodelc...")
    compiled_path = ct.models.MLModel(save_path).get_compiled_model_path()
    dest = f"{OUTPUT}/chunk4.mlmodelc"
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(compiled_path, dest)
    print(f"Compiled to {dest}")

    # Verify shape
    import coremltools
    m = coremltools.models.MLModel(save_path)
    for name, desc in m.input_description.items():
        if 'causal_mask_full' in name:
            print(f"\nVerification: {name} shape = {desc.type}")

    print("\nDone! Upload chunk4.mlmodelc to HF sdpa-8k/swa/")


if __name__ == "__main__":
    main()
