#!/usr/bin/env python3
"""AWQ-style smoothing v2 — uses real SWAChunk1..4 forward for calibration.

Improves over v1 by:
  * Driving the actual chunked path with full residual + attention
  * Multi-position forward (16-32 tokens per prompt)
  * KV cache accumulation across positions (real attention spread)

Same scale formula as v1 (per AWQ paper, alpha=0.5):
  s_i = (act_max[i] ** alpha) / (weight_max[i] ** (1 - alpha))
  norm.weight /= s ; weight *= s (input-dim)
"""
from __future__ import annotations
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4

# Reuse v1 helpers
from awq_smooth_gemma4 import (
    CALIB_PROMPTS, _hook_activation_stats, _attention_layer_pairs,
    _compute_smooth_scales, _apply_smooth,
)


def _build_inputs(model, cfg, model_dtype, ctx, ids):
    """Build chunk-friendly inputs from token id sequence at position 0..len-1."""
    seq_len = ids.shape[1]
    embed = model.embed_tokens(ids).to(model_dtype) * (cfg.hidden_size ** 0.5)
    pl = model.embed_tokens_per_layer(ids).to(model_dtype) * (cfg.hidden_size_per_layer_input ** 0.5)
    pl_grouped = pl.view(1, seq_len, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input)
    return embed, pl_grouped


def _drive_token(model, chunks, cfg, model_dtype, ctx, embed_t, pl_t, position,
                 K_full_in, V_full_in, K_swa_in, V_swa_in):
    """Run a single token forward through chunks 1-4. Returns updated KV caches."""
    # cos/sin at this position (use model's prebuilt buffers).
    cos_s = model.cos_sliding[position:position+1].view(1, 1, 1, cfg.head_dim).to(model_dtype)
    sin_s = model.sin_sliding[position:position+1].view(1, 1, 1, cfg.head_dim).to(model_dtype)
    cos_f = model.cos_full[position:position+1].view(1, 1, 1, cfg.global_head_dim).to(model_dtype)
    sin_f = model.sin_full[position:position+1].view(1, 1, 1, cfg.global_head_dim).to(model_dtype)
    # Full causal mask: zeros for past+current positions, -inf for future.
    mask_full = torch.zeros(1, 1, 1, ctx, dtype=model_dtype)
    mask_full[..., position+1:] = -1e4
    # Sliding mask (W positions)
    W = cfg.sliding_window
    mask_swa = torch.zeros(1, 1, 1, W, dtype=model_dtype)
    # update_mask: 1 at current position, 0 elsewhere (for full cache write).
    update_mask = torch.zeros(1, 1, ctx, 1, dtype=model_dtype)
    update_mask[:, :, position, :] = 1.0

    chunk1, chunk2, chunk3, chunk4 = chunks

    h, K_swa_out1, V_swa_out1, K_full_out1, V_full_out1, plc = chunk1(
        embed_t, mask_full, mask_swa, update_mask, pl_t,
        cos_s, sin_s, cos_f, sin_f,
        K_swa_in[:chunk1.num_sliding], V_swa_in[:chunk1.num_sliding],
        K_full_in[:chunk1.num_full], V_full_in[:chunk1.num_full],
    )
    h, K_swa_out2, V_swa_out2, K_full_out2, V_full_out2, kv13_k, kv13_v, kv14_k, kv14_v = chunk2(
        h, mask_full, mask_swa, update_mask, plc,
        cos_s, sin_s, cos_f, sin_f,
        K_swa_in[chunk1.num_sliding:chunk1.num_sliding + chunk2.num_sliding],
        V_swa_in[chunk1.num_sliding:chunk1.num_sliding + chunk2.num_sliding],
        K_full_in[chunk1.num_full:chunk1.num_full + chunk2.num_full],
        V_full_in[chunk1.num_full:chunk1.num_full + chunk2.num_full],
    )
    # chunk3 (KV-shared, no own K/V outputs)
    h = chunk3(h, mask_full, mask_swa, plc, cos_s, sin_s, cos_f, sin_f,
               kv13_k, kv13_v, kv14_k, kv14_v)
    # chunk4 (KV-shared)
    _ = chunk4(h, mask_full, mask_swa, plc, cos_s, sin_s, cos_f, sin_f,
               kv13_k, kv13_v, kv14_k, kv14_v)

    # Update K/V caches: stack chunk1 + chunk2 outputs.
    K_swa_new = torch.cat([K_swa_out1, K_swa_out2], dim=0)
    V_swa_new = torch.cat([V_swa_out1, V_swa_out2], dim=0)
    K_full_new = torch.cat([K_full_out1, K_full_out2], dim=0)
    V_full_new = torch.cat([V_full_out1, V_full_out2], dim=0)
    return K_swa_new, V_swa_new, K_full_new, V_full_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default=os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/"
        "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"))
    ap.add_argument("--out-state", required=True)
    ap.add_argument("--n-calib", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=32, help="Max prompt length to drive.")
    args = ap.parse_args()

    print(f"Loading Gemma4Model from {args.hf_dir}")
    model = Gemma4Model.from_pretrained(args.hf_dir, context_length=args.ctx).eval()
    print("Snapshotting fp16 weights, casting to fp32 for calibration")
    fp16_weights = {n: p.data.clone() for n, p in model.named_parameters()
                    if p.dtype == torch.float16}
    fp16_buffers = {n: b.data.clone() for n, b in model.named_buffers()
                    if b.dtype == torch.float16}
    for p in model.parameters():
        p.data = p.data.float()
    for n, b in list(model.named_buffers()):
        if b.dtype == torch.float16:
            b.data = b.data.float()
    model_dtype = torch.float32

    cfg = model.config
    if not hasattr(cfg, "is_full_attention"):
        cfg.is_full_attention = lambda i: cfg.layer_types[i] == "full_attention"
    if not hasattr(cfg, "is_kv_shared"):
        first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers
        cfg.is_kv_shared = lambda i, fs=first_shared: i >= fs
    if not hasattr(cfg, "get_head_dim"):
        cfg.get_head_dim = lambda i: cfg.global_head_dim if cfg.is_full_attention(i) else cfg.head_dim

    # Build chunks. compute_chunk_boundaries gives [(0, 8), (8, 15), (15, 25), (25, 35)] for E2B.
    from models.gemma4_swa_chunks import compute_chunk_boundaries
    boundaries = compute_chunk_boundaries(cfg)
    chunk1 = SWAChunk1(model, *boundaries[0]).eval()
    chunk2 = SWAChunk2(model, *boundaries[1]).eval()
    chunk3 = SWAChunk3(model, *boundaries[2]).eval()
    chunk4 = SWAChunk4(model, *boundaries[3]).eval()
    chunks = (chunk1, chunk2, chunk3, chunk4)

    nkv = cfg.num_key_value_heads
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window

    # Each layer has its own KV cache slot in chunk1/chunk2's input.
    # Total sliding slots: chunk1.num_sliding + chunk2.num_sliding
    # Total full slots: chunk1.num_full + chunk2.num_full
    n_swa_slots = chunk1.num_sliding + chunk2.num_sliding
    n_full_slots = chunk1.num_full + chunk2.num_full
    K_swa = torch.zeros(n_swa_slots, nkv, W, cfg.global_head_dim, dtype=model_dtype)
    V_swa = torch.zeros(n_swa_slots, nkv, W, cfg.global_head_dim, dtype=model_dtype)
    K_full = torch.zeros(n_full_slots, nkv, args.ctx, cfg.global_head_dim, dtype=model_dtype)
    V_full = torch.zeros(n_full_slots, nkv, args.ctx, cfg.global_head_dim, dtype=model_dtype)

    print("Tokenizing calibration prompts")
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    prompts = CALIB_PROMPTS[: args.n_calib]
    prompt_ids = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, : args.max_len]
        prompt_ids.append(ids)

    print("Hooking + driving real chunk forward (multi-position, real attention)")
    act_max, handles = _hook_activation_stats(model)

    with torch.no_grad():
        for pi, ids in enumerate(prompt_ids):
            # Reset KV caches per prompt.
            K_swa.zero_(); V_swa.zero_(); K_full.zero_(); V_full.zero_()
            seq_len = ids.shape[1]
            embed, pl_grouped = _build_inputs(model, cfg, model_dtype, args.ctx, ids)
            for pos in range(seq_len):
                emb_t = embed[:, pos:pos+1, :].contiguous()
                pl_t = pl_grouped[:, pos:pos+1, :].contiguous()
                K_swa, V_swa, K_full, V_full = _drive_token(
                    model, chunks, cfg, model_dtype, args.ctx, emb_t, pl_t, pos,
                    K_full, V_full, K_swa, V_swa)
            print(f"  prompt {pi+1}/{len(prompt_ids)}: {seq_len} tokens driven")

    for h in handles:
        h.remove()
    populated = {k: v for k, v in act_max.items() if v is not None}
    print(f"\nCollected stats for {len(populated)} modules")

    print(f"Computing AWQ scales (alpha={args.alpha})")
    scales = _compute_smooth_scales(populated, model, alpha=args.alpha)
    print(f"Computed scales for {len(scales)} (norm, linears) groups")
    _apply_smooth(scales)

    # Restore fp16 weights/buffers, save.
    print("Casting smoothed weights back to fp16")
    for n, p in model.named_parameters():
        if n in fp16_weights:
            p.data = p.data.half()
    for n, b in list(model.named_buffers()):
        if n in fp16_buffers:
            b.data = b.data.half()

    SKIP_PREFIXES = ("kv_cache_", "cos_", "sin_")
    state = {k: v.clone() for k, v in model.state_dict().items()
             if not any(k.startswith(p) for p in SKIP_PREFIXES)}
    os.makedirs(os.path.dirname(args.out_state) or ".", exist_ok=True)
    torch.save(state, args.out_state)
    print(f"\nSaved smoothed state to {args.out_state} ({len(state)} tensors)")


if __name__ == "__main__":
    main()
