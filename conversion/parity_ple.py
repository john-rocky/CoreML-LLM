#!/usr/bin/env python3
"""Verify that our _compute_ple math == HF's project_per_layer_inputs.

Compare per_layer_combined for token=BOS at position 0.
"""
from __future__ import annotations
import sys, os
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    target = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=torch.float32, low_cpu_mem_usage=True).eval()
    text_model = target.model.language_model

    ids = torch.tensor([[2, 818]])  # 2-token prompt
    inputs_embeds = text_model.embed_tokens(ids).float()
    per_layer_inputs_token = text_model.get_per_layer_inputs(ids, inputs_embeds).float()
    print(f"inputs_embeds shape: {tuple(inputs_embeds.shape)}")
    print(f"per_layer_inputs_token shape: {tuple(per_layer_inputs_token.shape)}")

    # HF's project_per_layer_inputs
    hf_pl_combined = text_model.project_per_layer_inputs(inputs_embeds, per_layer_inputs_token).float()
    print(f"HF per_layer_combined shape: {tuple(hf_pl_combined.shape)}")
    # Reshape: HF returns (B, L, num_layers, ple_dim) — flatten last two for our format.
    if hf_pl_combined.ndim == 4:
        hf_pl_combined_flat = hf_pl_combined.reshape(
            hf_pl_combined.shape[0], hf_pl_combined.shape[1], -1)
    else:
        hf_pl_combined_flat = hf_pl_combined

    # Our _compute_ple — manually replicate
    from models.gemma4_swa_chunks import SWAChunk1
    cfg = text_model.config
    if not hasattr(cfg, "is_full_attention"):
        cfg.is_full_attention = lambda i: cfg.layer_types[i] == "full_attention"
    if not hasattr(cfg, "is_kv_shared"):
        first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers
        cfg.is_kv_shared = lambda i, fs=first_shared: i >= fs
    if not hasattr(cfg, "get_head_dim"):
        cfg.get_head_dim = lambda i: cfg.global_head_dim if cfg.is_full_attention(i) else cfg.head_dim
    chunk1 = SWAChunk1(text_model, 0, 8)

    # Need per_layer_inputs_token in the same format SWAChunk1 expects:
    # (1, 1, num_layers * ple_dim) — flatten the (num_layers, ple_dim) pair.
    if per_layer_inputs_token.ndim == 4:
        per_layer_raw_flat = per_layer_inputs_token.reshape(
            per_layer_inputs_token.shape[0], per_layer_inputs_token.shape[1], -1)
    else:
        per_layer_raw_flat = per_layer_inputs_token

    # Test position 0
    h0 = inputs_embeds[:, 0:1, :].contiguous()
    plr0 = per_layer_raw_flat[:, 0:1, :].contiguous()
    ours_pl_combined = chunk1._compute_ple(h0, plr0).float()
    print(f"\nPosition 0:")
    print(f"  our shape: {tuple(ours_pl_combined.shape)}")

    a = hf_pl_combined_flat[:, 0:1, :].flatten()
    b = ours_pl_combined.flatten()
    cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))
    diff = (a - b).abs()
    print(f"  PLE cos={cos:.6f}  |hf|={a.norm():.3f}  |our|={b.norm():.3f}  "
          f"max_diff={diff.max():.4f}  mean_diff={diff.mean():.6f}")


if __name__ == "__main__":
    main()
