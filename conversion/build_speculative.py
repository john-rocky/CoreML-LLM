#!/usr/bin/env python3
"""Build CoreML models for Medusa speculative decoding.

Produces:
  1. chunk4 with hidden_states_out (decode, N=1) — replaces existing chunk4
  2. verify_chunk1-4 (N=4) — batched verification chunks
  3. medusa.mlpackage — 3 Medusa draft heads + shared lm_head

Usage:
    # After training Medusa heads:
    python build_speculative.py \
        --medusa-path ./output/medusa_heads/medusa_heads.pt \
        --output /tmp/gemma4-speculative

Requires the base HF model for weight extraction.
"""
from __future__ import annotations
import argparse, os, sys, shutil, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4
from ane_ops import MODEL_DTYPE, InModelArgmax

HF_DIR = os.environ.get("GEMMA4_HF_DIR", f"{ROOT}/../output/gemma4-e2b/hf_model")
CTX = 2048
W = 512
N_VERIFY = 4
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
        minimum_deployment_target=ct.target.iOS26,
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
# Medusa head wrapper for CoreML export
# ============================================================
class MedusaHeadBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x))) + x


class MedusaCoreML(nn.Module):
    """Takes normed hidden state, runs K heads, applies shared lm_head, returns K token IDs."""
    def __init__(self, heads: nn.ModuleList, lm_head: nn.Conv2d, softcap: float):
        super().__init__()
        self.heads = heads
        self.lm_head = lm_head
        self.softcap = softcap
        self.argmax = InModelArgmax()

    def forward(self, hidden_states):
        # hidden_states: (1, 1, hidden_size)
        draft_ids = []
        for head in self.heads:
            h = head(hidden_states)  # (1, 1, hidden)
            x = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (1, hidden, 1, 1)
            logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)  # (1, 1, vocab)
            if self.softcap > 0:
                logits = torch.tanh(logits / self.softcap) * self.softcap
            tid, _ = self.argmax(logits.squeeze(0))
            draft_ids.append(tid)
        return tuple(draft_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medusa-path", type=str, default=None,
                        help="Path to medusa_heads.pt (skip if not yet trained)")
    parser.add_argument("--output", type=str, default="/tmp/gemma4-speculative")
    parser.add_argument("--chunk4-only", action="store_true",
                        help="Only rebuild chunk4 with hidden_states_out")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading Gemma 4 E2B...")
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=CTX)
    base.eval()

    hidden = base.config.hidden_size
    pld = base.config.hidden_size_per_layer_input
    nlayers = base.config.num_hidden_layers
    max_hd = 512

    # ============================================================
    # 1. Rebuild chunk4 with hidden_states_out
    # ============================================================
    print("\n=== Chunk4 (decode, L25-34 + LM head + hidden_out) ===")
    swa4 = SWAChunk4(base).eval()
    s4 = (
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
    # Now outputs: token_id, token_logit, AND normed hidden state
    out4 = ["token_id", "token_logit", "hidden_states_out"]
    do_convert(swa4, s4, in4, out4, f"{args.output}/chunk4.mlpackage")

    if args.chunk4_only:
        print("\n--chunk4-only: done.")
        return

    # ============================================================
    # 2. Verify chunks (N=4)
    # ============================================================
    NV = N_VERIFY
    print(f"\n=== Verify chunks (N={NV}) ===")

    # Verify chunk 1
    print(f"\n  verify_chunk1 (L0-7, N={NV})")
    vc1 = SWAChunk1(base).eval()
    vs1 = (
        torch.zeros(1, NV, hidden, dtype=torch.float16),
        torch.zeros(1, 1, NV, CTX, dtype=torch.float16),
        torch.zeros(1, 1, NV, W, dtype=torch.float16),
        torch.zeros(1, NV, CTX, 1, dtype=torch.float16),
        torch.zeros(1, NV, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, NV, 256, dtype=torch.float16),
        torch.zeros(1, 1, NV, 256, dtype=torch.float16),
        torch.zeros(1, 1, NV, 512, dtype=torch.float16),
        torch.zeros(1, 1, NV, 512, dtype=torch.float16),
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(7, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(1, 1, CTX, max_hd, dtype=torch.float16),
        torch.zeros(1, 1, CTX, max_hd, dtype=torch.float16),
    )
    vin1 = [
        ct.TensorType(name="hidden_states",       shape=vs1[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=vs1[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=vs1[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=vs1[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_raw",       shape=vs1[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=vs1[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=vs1[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=vs1[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=vs1[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=vs1[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=vs1[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=vs1[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=vs1[12].shape, dtype=fp16),
    ]
    vout1 = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
             "K_full_out", "V_full_out", "per_layer_combined_out"]
    do_convert(vc1, vs1, vin1, vout1, f"{args.output}/verify_chunk1.mlpackage")

    # Verify chunk 2
    print(f"\n  verify_chunk2 (L8-14, N={NV})")
    vc2 = SWAChunk2(base).eval()
    vs2 = (
        torch.zeros(1, NV, hidden, dtype=torch.float16),
        torch.zeros(1, 1, NV, CTX, dtype=torch.float16),
        torch.zeros(1, 1, NV, W, dtype=torch.float16),
        torch.zeros(1, NV, CTX, 1, dtype=torch.float16),
        torch.zeros(1, NV, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, NV, 256, dtype=torch.float16),
        torch.zeros(1, 1, NV, 256, dtype=torch.float16),
        torch.zeros(1, 1, NV, 512, dtype=torch.float16),
        torch.zeros(1, 1, NV, 512, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(5, 1, W, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, CTX, max_hd, dtype=torch.float16),
        torch.zeros(2, 1, CTX, max_hd, dtype=torch.float16),
    )
    vin2 = [
        ct.TensorType(name="hidden_states",       shape=vs2[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=vs2[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=vs2[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=vs2[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=vs2[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=vs2[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=vs2[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=vs2[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=vs2[8].shape,  dtype=fp16),
        ct.TensorType(name="K_sliding_in",        shape=vs2[9].shape,  dtype=fp16),
        ct.TensorType(name="V_sliding_in",        shape=vs2[10].shape, dtype=fp16),
        ct.TensorType(name="K_full_in",           shape=vs2[11].shape, dtype=fp16),
        ct.TensorType(name="V_full_in",           shape=vs2[12].shape, dtype=fp16),
    ]
    vout2 = ["hidden_states_out", "K_sliding_out", "V_sliding_out",
             "K_full_out", "V_full_out", "kv13_k", "kv13_v", "kv14_k", "kv14_v"]
    do_convert(vc2, vs2, vin2, vout2, f"{args.output}/verify_chunk2.mlpackage")

    # Verify chunk 3
    print(f"\n  verify_chunk3 (L15-24 KV-shared, N={NV})")
    vc3 = SWAChunk3(base).eval()
    vs3 = (
        torch.zeros(1, NV, hidden, dtype=torch.float16),
        torch.zeros(1, 1, NV, CTX, dtype=torch.float16),
        torch.zeros(1, 1, NV, W, dtype=torch.float16),
        torch.zeros(1, NV, CTX, 1, dtype=torch.float16),
        torch.zeros(1, NV, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, NV, 256, dtype=torch.float16),
        torch.zeros(1, 1, NV, 256, dtype=torch.float16),
        torch.zeros(1, 1, NV, 512, dtype=torch.float16),
        torch.zeros(1, 1, NV, 512, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, W, 256, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
        torch.zeros(1, 1, CTX, 512, dtype=torch.float16),
    )
    vin3 = [
        ct.TensorType(name="hidden_states",       shape=vs3[0].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_full",    shape=vs3[1].shape,  dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=vs3[2].shape,  dtype=fp16),
        ct.TensorType(name="update_mask",         shape=vs3[3].shape,  dtype=fp16),
        ct.TensorType(name="per_layer_combined",  shape=vs3[4].shape,  dtype=fp16),
        ct.TensorType(name="cos_s",               shape=vs3[5].shape,  dtype=fp16),
        ct.TensorType(name="sin_s",               shape=vs3[6].shape,  dtype=fp16),
        ct.TensorType(name="cos_f",               shape=vs3[7].shape,  dtype=fp16),
        ct.TensorType(name="sin_f",               shape=vs3[8].shape,  dtype=fp16),
        ct.TensorType(name="kv13_k",              shape=vs3[9].shape,  dtype=fp16),
        ct.TensorType(name="kv13_v",              shape=vs3[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",              shape=vs3[11].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",              shape=vs3[12].shape, dtype=fp16),
    ]
    vout3 = ["hidden_states_out"]
    do_convert(vc3, vs3, vin3, vout3, f"{args.output}/verify_chunk3.mlpackage")

    # Verify chunk 4 — outputs ALL N token IDs (not just last)
    print(f"\n  verify_chunk4 (L25-34 + LM, N={NV})")
    # SWAChunk4 now returns (token_id, token_logit, normed)
    # For verify, we need all N positions' token IDs.
    # We'll build a custom wrapper.
    class VerifyChunk4(nn.Module):
        """Like SWAChunk4 but outputs token_id per position (N=4)."""
        def __init__(self, chunk4: SWAChunk4):
            super().__init__()
            self.chunk4 = chunk4

        def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                    update_mask, per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                    kv13_k, kv13_v, kv14_k, kv14_v):
            # Run chunk4's layers + norm (but not its single-position argmax)
            config = self.chunk4.config
            dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
            dummy_V = dummy_K
            h = hidden_states
            for local_idx in range(self.chunk4.END - self.chunk4.START):
                layer_idx = self.chunk4.START + local_idx
                from models.gemma4_swa_chunks import _run_layer_swa
                h, *_ = _run_layer_swa(
                    self.chunk4.layers[local_idx], layer_idx, h,
                    cos_s, sin_s, cos_f, sin_f,
                    causal_mask_full, causal_mask_sliding, update_mask,
                    dummy_K, dummy_V, dummy_K, dummy_V,
                    config, per_layer_combined,
                    kv13_k, kv13_v, kv14_k, kv14_v,
                )
            normed = self.chunk4.norm(h)
            x = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
            logits = self.chunk4.lm_head(x).squeeze(2).permute(0, 2, 1)  # (1, N, vocab)
            if self.chunk4.softcap > 0:
                logits = torch.tanh(logits / self.chunk4.softcap) * self.chunk4.softcap
            # Per-position argmax
            token_ids = torch.argmax(logits, dim=-1).to(torch.int32)  # (1, N)
            return token_ids

    vc4 = VerifyChunk4(swa4).eval()
    vs4 = vs3  # same input shapes
    vin4 = vin3
    vout4 = ["token_ids"]
    do_convert(vc4, vs4, vin4, vout4, f"{args.output}/verify_chunk4.mlpackage")

    # ============================================================
    # 3. Medusa heads → CoreML
    # ============================================================
    if args.medusa_path and os.path.exists(args.medusa_path):
        print(f"\n=== Medusa heads ({args.medusa_path}) ===")
        data = torch.load(args.medusa_path, weights_only=True)
        num_heads = data["num_heads"]
        hs = data["hidden_size"]

        heads = nn.ModuleList()
        for k in range(num_heads):
            head = MedusaHeadBlock(hs)
            head.fc1.weight.data = data[f"head_{k}_fc1_weight"].float()
            head.fc2.weight.data = data[f"head_{k}_fc2_weight"].float()
            heads.append(head)

        lm_head = nn.Conv2d(base.lm_head.in_channels, base.lm_head.out_channels,
                            kernel_size=1, bias=False)
        lm_head.weight.data = base.lm_head.weight.data.clone()

        medusa = MedusaCoreML(heads, lm_head, float(base.softcap)).eval()
        sample = (torch.zeros(1, 1, hs, dtype=torch.float16),)
        in_medusa = [ct.TensorType(name="hidden_states", shape=(1, 1, hs), dtype=fp16)]
        out_medusa = [f"draft_token_{k}" for k in range(num_heads)]
        do_convert(medusa, sample, in_medusa, out_medusa,
                   f"{args.output}/medusa.mlpackage", quantize=False)
    else:
        print("\n  (Skipping Medusa conversion — no --medusa-path or file not found)")

    print(f"\n{'='*60}")
    print(f"All models saved to {args.output}/")
    print(f"  chunk4.mlpackage          — decode with hidden_states_out")
    print(f"  verify_chunk1-4.mlpackage — N={NV} batched verification")
    if args.medusa_path:
        print(f"  medusa.mlpackage          — {num_heads} draft heads")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
