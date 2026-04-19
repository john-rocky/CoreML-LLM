#!/usr/bin/env python3
"""Build EAGLE-3 target decode chunks for Gemma 4 E2B.

Adds hidden-state taps at FUSION_LAYERS = [8, 17, 34] so the EAGLE-3 draft's
fusion layer can read multi-layer hiddens from the target's decode step.

Outputs:
  chunk1.mlpackage  — L0-7, unchanged (no fusion layer in this chunk)
  chunk2.mlpackage  — L8-14, + hidden_at_L8 extra output (post-L8, pre-L9)
  chunk3.mlpackage  — L15-24, + hidden_at_L17 extra output
  chunk4.mlpackage  — L25-34, + hidden_at_L34 extra output (post-L34, PRE final-norm)

All extra hidden outputs are (1, 1, hidden) fp16. Swift side reads them from
the decode step just committed and passes them to fusion(h_low, h_mid, h_high).

Verify chunks (seq_dim=3) are built by a separate script because
_run_layer_swa is hardcoded for T=1 and needs a prefill-based rewrite for T>1.

Usage:
    python conversion/build_eagle3_chunks.py --output ./output/eagle3-chunks
    python conversion/build_eagle3_chunks.py --only chunk2  # build a single chunk
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
import coremltools as ct

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import (
    SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4, _run_layer_swa,
)
from ane_ops import MODEL_DTYPE

HF_DIR = os.environ.get(
    "GEMMA4_HF_DIR",
    f"{ROOT}/../output/gemma4-e2b/hf_model",
)
CTX = 2048
W = 512
FUSION_LAYERS = (8, 17, 34)
fp16 = ct.converters.mil.mil.types.fp16


def do_convert(model, sample_inputs, input_specs, output_names, save_path, quantize=True):
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
    return mlmodel


# ============================================================
# EagleChunk2: SWAChunk2 + hidden_at_L8 (captured after layers[0] = L8)
# ============================================================
class EagleChunk2(nn.Module):
    def __init__(self, base: SWAChunk2):
        super().__init__()
        self.base = base

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.base.config
        kv13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        K_sliding_outs = []
        V_sliding_outs = []
        K_full_outs = []
        V_full_outs = []
        hidden_at_L8 = hidden_states  # placeholder; overwritten below

        for local_idx in range(self.base.end - self.base.start):
            layer_idx = self.base.start + local_idx  # 8..14
            is_full = config.is_full_attention(layer_idx)
            if is_full:
                fi = self.base.full_map[layer_idx]
                K_full_slot = K_full_in[fi].unsqueeze(0)
                V_full_slot = V_full_in[fi].unsqueeze(0)
                K_sliding_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
                V_sliding_slot = K_sliding_slot
            else:
                si = self.base.sliding_map[layer_idx]
                K_sliding_slot = K_sliding_in[si].unsqueeze(0)
                V_sliding_slot = V_sliding_in[si].unsqueeze(0)
                K_full_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
                V_full_slot = K_full_slot

            (hidden_states, Kso, Vso, Kfo, Vfo,
             kv13_k, kv13_v, kv14_k, kv14_v) = _run_layer_swa(
                self.base.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
            if is_full:
                K_full_outs.append(Kfo.squeeze(0))
                V_full_outs.append(Vfo.squeeze(0))
            else:
                K_sliding_outs.append(Kso.squeeze(0))
                V_sliding_outs.append(Vso.squeeze(0))

            if layer_idx == 8:
                hidden_at_L8 = hidden_states

        K_sliding_out = torch.stack(K_sliding_outs, dim=0)
        V_sliding_out = torch.stack(V_sliding_outs, dim=0)
        K_full_out = torch.stack(K_full_outs, dim=0)
        V_full_out = torch.stack(V_full_outs, dim=0)
        return (hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out,
                kv13_k, kv13_v, kv14_k, kv14_v, hidden_at_L8)


# ============================================================
# EagleChunk3: SWAChunk3 + hidden_at_L17 (captured after layers[2] = L17)
# ============================================================
class EagleChunk3(nn.Module):
    def __init__(self, base: SWAChunk3):
        super().__init__()
        self.base = base

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.base.config
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K
        hidden_at_L17 = hidden_states
        for local_idx in range(self.base.end - self.base.start):
            layer_idx = self.base.start + local_idx  # 15..24
            hidden_states, *_ = _run_layer_swa(
                self.base.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                dummy_K, dummy_V, dummy_K, dummy_V,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
            if layer_idx == 17:
                hidden_at_L17 = hidden_states
        return hidden_states, hidden_at_L17


# ============================================================
# EagleChunk4: SWAChunk4 emits pre-norm post-L34 hidden instead of post-norm
# ============================================================
class EagleChunk4(nn.Module):
    def __init__(self, base: SWAChunk4):
        super().__init__()
        self.base = base

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.base.config
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K

        for local_idx in range(self.base.end - self.base.start):
            layer_idx = self.base.start + local_idx  # 25..34
            hidden_states, *_ = _run_layer_swa(
                self.base.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                dummy_K, dummy_V, dummy_K, dummy_V,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
        # post-final-norm hidden — matches `out.hidden_states[-1]` semantics
        # from HF's Gemma4TextModel, which is what the training collector
        # used as `h_high`. Earlier this captured the pre-norm hidden
        # (norm ~40), whereas training saw the post-norm tensor (norm ~235):
        # that 6× magnitude + distribution shift crushes the draft's
        # accept rate to ~0% on device. Capture it AFTER `self.base.norm`.
        normed = self.base.norm(hidden_states)
        hidden_at_L34 = normed
        x = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        logits = self.base.lm_head(x).squeeze(2).permute(0, 2, 1)
        if self.base.softcap > 0:
            logits = torch.tanh(logits / self.base.softcap) * self.base.softcap
        token_id, token_logit = self.base.argmax(logits.squeeze(0))
        return token_id, token_logit, hidden_at_L34


def build_chunk1(base, out_dir):
    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512
    print("\n=== chunk1 (L0-7, unchanged) ===")
    c1 = SWAChunk1(base).eval()
    s = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(1, 1, CTX, max_hd, dtype=torch.float16),
        torch.zeros(1, 1, CTX, max_hd, dtype=torch.float16),
    )
    ins = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",       shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=s[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=s[12].shape, dtype=fp16),
    ]
    outs = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
            "K_full_out", "V_full_out", "per_layer_combined_out"]
    do_convert(c1, s, ins, outs, f"{out_dir}/chunk1.mlpackage")


def build_chunk2(base, out_dir):
    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512
    print("\n=== chunk2 (L8-14, + hidden_at_L8) ===")
    c2 = EagleChunk2(SWAChunk2(base).eval()).eval()
    s = (
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
    ins = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=s[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=s[12].shape, dtype=fp16),
    ]
    outs = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
            "K_full_out", "V_full_out", "kv13_k", "kv13_v", "kv14_k", "kv14_v",
            "hidden_at_L8"]
    do_convert(c2, s, ins, outs, f"{out_dir}/chunk2.mlpackage")


def build_chunk3(base, out_dir):
    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    print("\n=== chunk3 (L15-24, + hidden_at_L17) ===")
    c3 = EagleChunk3(SWAChunk3(base).eval()).eval()
    s = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
    )
    ins = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=s[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=s[12].shape, dtype=fp16),
    ]
    outs = ["hidden_states_out", "hidden_at_L17"]
    do_convert(c3, s, ins, outs, f"{out_dir}/chunk3.mlpackage")


def build_chunk4(base, out_dir):
    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    print("\n=== chunk4 (L25-34 + LM + hidden_at_L34 pre-norm) ===")
    c4 = EagleChunk4(SWAChunk4(base).eval()).eval()
    s = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, CTX, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 1, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 256, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, 1, 512, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
    )
    ins = [
        ct.TensorType(name="hidden_states",       shape=s[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=s[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=s[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=s[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=s[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=s[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=s[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=s[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=s[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=s[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=s[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=s[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=s[12].shape, dtype=fp16),
    ]
    outs = ["token_id", "token_logit", "hidden_at_L34"]
    do_convert(c4, s, ins, outs, f"{out_dir}/chunk4.mlpackage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="./output/eagle3-chunks")
    ap.add_argument("--only", type=str, default=None,
                    choices=[None, "chunk1", "chunk2", "chunk3", "chunk4"],
                    help="Build only the named chunk")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading Gemma 4 E2B from {HF_DIR}...")
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=CTX)
    base.eval()

    targets = [args.only] if args.only else ["chunk1", "chunk2", "chunk3", "chunk4"]
    dispatch = {
        "chunk1": build_chunk1,
        "chunk2": build_chunk2,
        "chunk3": build_chunk3,
        "chunk4": build_chunk4,
    }
    for name in targets:
        dispatch[name](base, args.output)

    print(f"\nDone. Files in {args.output}/")


if __name__ == "__main__":
    main()
