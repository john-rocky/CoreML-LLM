#!/usr/bin/env python3
"""Parity test: our PyTorch port vs HuggingFace reference.

Runs the same random `inputs_embeds` + `shared_kv_states` through:
  - transformers.models.gemma4_assistant.Gemma4AssistantForCausalLM (reference)
  - conversion.mtp_drafter_model.MtpDrafterModel             (our port)

Pass criteria:
  - logits cosine similarity > 0.999
  - top-1 argmax match
  - post_projection / last_hidden_state cosine > 0.999

Requires `transformers >= 5.7.0` for `Gemma4AssistantForCausalLM`.

Usage:
    python conversion/test_mtp_parity.py \\
        --hf-repo google/gemma-4-E2B-it-assistant \\
        --ckpt   output/mtp_probe/mtp_drafter.pt
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_drafter_model import MtpDrafterModel, MtpDrafterConfig


def _build_shared_kv(B: int, ctx: int, swa_head_dim: int, full_head_dim: int,
                     seed: int = 1234):
    """Random but reproducible (K, V) for both layer types.

    Layout matches transformers' shared_kv_states (post-norm, post-RoPE, transposed):
      K: (B, num_kv, ctx, head_dim) where num_kv=1 here (drafter num_key_value_heads=1).
      V: (B, num_kv, ctx, head_dim)
    """
    g = torch.Generator().manual_seed(seed)
    swa_k = torch.randn(B, 1, ctx, swa_head_dim, generator=g) * 0.5
    swa_v = torch.randn(B, 1, ctx, swa_head_dim, generator=g) * 0.5
    full_k = torch.randn(B, 1, ctx, full_head_dim, generator=g) * 0.5
    full_v = torch.randn(B, 1, ctx, full_head_dim, generator=g) * 0.5
    return (swa_k, swa_v, full_k, full_v)


def _run_hf_reference(hf_repo: str, dtype, inputs_embeds, position_ids,
                      swa_k, swa_v, full_k, full_v, use_full_lm_head: bool):
    """Run the HF reference; returns (logits, last_hidden_state).

    If `use_full_lm_head=True`, swap the masked_embedding path off and run a
    full lm_head — produces dense logits comparable to our port. Otherwise
    the model uses its trained masked-centroid path (sparse logits).
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo, dtype=dtype, low_cpu_mem_usage=True).eval()

    if use_full_lm_head:
        # Disable the centroid lookup so we can compare apples-to-apples.
        model.config.use_ordered_embeddings = False
        model.masked_embedding = None

    shared_kv = {
        "sliding_attention": (swa_k.to(dtype), swa_v.to(dtype)),
        "full_attention":    (full_k.to(dtype), full_v.to(dtype)),
    }
    # attention_mask=None lets transformers create a full bidirectional mask.
    with torch.no_grad():
        out = model(
            inputs_embeds=inputs_embeds.to(dtype),
            position_ids=position_ids,
            shared_kv_states=shared_kv,
            attention_mask=None,
        )
    return out.logits.float(), out.last_hidden_state.float()


def _run_port(ckpt: str, inputs_embeds, position_ids,
              swa_k, swa_v, full_k, full_v, mask_swa, mask_full):
    """Run our PyTorch port; returns (logits, post_projection)."""
    cfg = MtpDrafterConfig()
    model = MtpDrafterModel(cfg).float().eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Known: RoPE buffers (swa_*, full_*) live in checkpoint? They're persistent=False,
    # so they are NOT saved — model recomputes them on init. Should be empty.
    if unexpected:
        print(f"  unexpected keys ignored: {unexpected[:6]}{'...' if len(unexpected) > 6 else ''}")

    # Our model expects V pre-transposed: (B, 1, head_dim, ctx).
    swa_v_t = swa_v.transpose(-2, -1).contiguous()
    full_v_t = full_v.transpose(-2, -1).contiguous()

    pos = torch.tensor([int(position_ids[0, 0].item())], dtype=torch.int32)
    with torch.no_grad():
        logits, proj_act = model(
            inputs_embeds.float(), pos,
            swa_k.float(), swa_v_t.float(),
            full_k.float(), full_v_t.float(),
            mask_swa.float(), mask_full.float())
    return logits.float(), proj_act.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="google/gemma-4-E2B-it-assistant")
    ap.add_argument("--ckpt", default="output/mtp_probe/mtp_drafter.pt")
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--pos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                    help="HF reference dtype. fp32 gives the tightest parity bound.")
    ap.add_argument("--full-lm-head", action="store_true",
                    help="Disable HF masked_embedding so logits use the full "
                         "lm_head — direct apples-to-apples vs the port. "
                         "Default (off) preserves HF's trained inference path.")
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        print(f"ERROR: ckpt not found: {args.ckpt}")
        print("Run `python conversion/mtp_drafter_model.py` first.")
        sys.exit(1)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    cfg = MtpDrafterConfig()
    B, L = 1, 1
    ctx = args.ctx
    pos = args.pos

    g = torch.Generator().manual_seed(args.seed)
    inputs_embeds = torch.randn(B, L, cfg.input_size, generator=g) * 0.05
    position_ids = torch.full((B, L), pos, dtype=torch.long)

    swa_k, swa_v, full_k, full_v = _build_shared_kv(
        B, ctx, cfg.swa_head_dim, cfg.full_head_dim, seed=args.seed + 1)

    # HF reference uses bidirectional masks; for q_len=1 every K position is
    # valid regardless of `pos`. So we pass an all-zeros (no-op) additive mask
    # to the port to mirror that behaviour for the parity bound.
    mask_swa = torch.zeros(B, 1, 1, ctx)
    mask_full = torch.zeros(B, 1, 1, ctx)

    print(f"=== Inputs: ctx={ctx} pos={pos} dtype(HF)={args.dtype} ===")

    print(f"\n[HF reference] loading + running (full-lm-head={args.full_lm_head}) ...")
    hf_logits, hf_hidden = _run_hf_reference(
        args.hf_repo, dtype, inputs_embeds, position_ids,
        swa_k, swa_v, full_k, full_v, use_full_lm_head=args.full_lm_head)
    print(f"  logits {tuple(hf_logits.shape)} max={hf_logits.max().item():.3f}"
          f" argmax={int(hf_logits.argmax(-1).item())}")

    print("\n[Port]         loading + running ...")
    port_logits, port_proj = _run_port(
        args.ckpt, inputs_embeds, position_ids,
        swa_k, swa_v, full_k, full_v, mask_swa, mask_full)
    print(f"  logits {tuple(port_logits.shape)} max={port_logits.max().item():.3f}"
          f" argmax={int(port_logits.argmax(-1).item())}")

    # post_projection is what our port returns directly. The HF reference's
    # `last_hidden_state` is ALSO post_projection-output (1536-dim) per
    # the source code — its name is misleading (model's actual last hidden
    # is 256-dim, but Gemma4AssistantOutput stores the projected state under
    # `last_hidden_state` so downstream MTP loops can feed it back).
    print(f"  hf_hidden {tuple(hf_hidden.shape)}  port_proj {tuple(port_proj.shape)}")

    print("\n=== Parity ===")
    a, b = hf_logits.flatten(), port_logits.flatten()
    cos_logits = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))
    a_h, b_h = hf_hidden.flatten(), port_proj.flatten()
    cos_hidden = float(torch.dot(a_h, b_h) / (a_h.norm() * b_h.norm() + 1e-8))

    hf_top5 = torch.topk(hf_logits.flatten(), 5).indices.tolist()
    port_top5 = torch.topk(port_logits.flatten(), 5).indices.tolist()
    overlap = len(set(hf_top5) & set(port_top5))

    print(f"  logits  cosine = {cos_logits:.6f}")
    print(f"  hidden  cosine = {cos_hidden:.6f}")
    print(f"  argmax: HF={hf_logits.argmax().item()}  port={port_logits.argmax().item()}"
          f"  match={hf_logits.argmax() == port_logits.argmax()}")
    print(f"  top-5 overlap = {overlap}/5  HF={hf_top5}  port={port_top5}")

    ok = cos_logits > 0.999 and cos_hidden > 0.999 and (
        hf_logits.argmax() == port_logits.argmax())
    print("\n  " + ("PASS" if ok else "WARN: divergence — investigate before CoreML build"))


if __name__ == "__main__":
    main()
