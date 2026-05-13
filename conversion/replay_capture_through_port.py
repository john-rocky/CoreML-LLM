#!/usr/bin/env python3
"""Replay HF capture through our PyTorch port + show divergence point.

Loads `output/mtp_probe/hf_capture.pt`, feeds the captured inputs to
`MtpDrafterModel` (our port), and compares logits + top-1 against the
HF reference values stored in the capture.

If port matches HF on REAL K/V inputs → port is fine; bug is in
CoreML mlpackage or Swift.
If port differs → drafter port has a real-data sensitivity that the
random-K parity test missed.
"""
from __future__ import annotations
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_drafter_model import MtpDrafterModel, MtpDrafterConfig


def main():
    cap_path = "output/mtp_probe/hf_capture.pt"
    cap = torch.load(cap_path, map_location="cpu", weights_only=False)
    print(f"[replay] loaded {cap_path}")
    print(f"[replay] keys: {list(cap.keys())}")

    cfg = MtpDrafterConfig()
    model = MtpDrafterModel(cfg).float().eval()
    ckpt = "output/mtp_probe/mtp_drafter.pt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)

    # Build the activations input: cat([last_token_embedding, last_hidden_state]) — exactly how HF feeds it.
    inputs_embeds = cap["inputs_embeds"].float()                 # (1, 1, 3072)
    position_ids = cap["position_ids"].long()                    # (1, 1)
    pos_int = int(position_ids[0, 0].item())

    # K and V from HF: K (1, 1, ctx, hd), V (1, 1, ctx, hd) -- our port expects V pre-transposed.
    swa_k = cap["sliding_k"].float()                              # (1, 1, ctx, 256)
    swa_v = cap["sliding_v"].float().transpose(-2, -1).contiguous()  # (1, 1, 256, ctx)
    full_k = cap["full_k"].float()                                # (1, 1, ctx, 512)
    full_v = cap["full_v"].float().transpose(-2, -1).contiguous() # (1, 1, 512, ctx)
    ctx = swa_k.shape[2]

    # Drafter expects bidirectional mask with shape (B, 1, 1, ctx). HF uses
    # `create_bidirectional_mask` + sliding-window flip. The simplest faithful
    # mask: all positions allowed (all-zero additive mask). This matches HF's
    # behavior when input_ids[-1] is the only "seen" token at decode time.
    mask_swa = torch.zeros(1, 1, 1, ctx)
    mask_full = torch.zeros(1, 1, 1, ctx)

    pos_t = torch.tensor([pos_int], dtype=torch.int32)
    with torch.no_grad():
        port_logits, port_proj = model(
            inputs_embeds, pos_t,
            swa_k, swa_v, full_k, full_v,
            mask_swa, mask_full)
    port_argmax = int(port_logits.argmax(-1).item())
    hf_argmax = int(cap["drafter_token"][0, 0].item())
    print(f"[replay] HF drafter top-1 = {hf_argmax}")
    print(f"[replay] PORT top-1       = {port_argmax}")
    print(f"[replay] match            = {hf_argmax == port_argmax}")

    # Logit-level cosine.
    hf_logits = cap["drafter_logits"].float().flatten()
    pt_logits = port_logits.float().flatten()
    cos = torch.dot(hf_logits, pt_logits) / (hf_logits.norm() * pt_logits.norm() + 1e-8)
    print(f"[replay] logits cosine   = {cos:.6f}")

    # last_hidden_state cosine.
    hf_lh = cap["drafter_last_hidden"].float().flatten()
    pt_lh = port_proj.float().flatten()
    cos_lh = torch.dot(hf_lh, pt_lh) / (hf_lh.norm() * pt_lh.norm() + 1e-8)
    print(f"[replay] hidden cosine   = {cos_lh:.6f}")

    # HF top-5 vs port top-5.
    hf_top5 = torch.topk(hf_logits, 5).indices.tolist()
    pt_top5 = torch.topk(pt_logits, 5).indices.tolist()
    print(f"[replay] HF top-5  = {hf_top5}")
    print(f"[replay] PORT top-5 = {pt_top5}")


if __name__ == "__main__":
    main()
