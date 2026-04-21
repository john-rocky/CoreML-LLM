#!/usr/bin/env python3
"""Layer-by-layer comparison between PyTorch drafter and TFLite drafter.

Uses TFLite interpreter and PyTorch model with IDENTICAL zero-KV inputs.
Dumps intermediate hidden state norms at each layer's output to find
where they diverge.
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_drafter_model import MtpDrafterModel, MtpDrafterConfig

# --- Load PyTorch model ---
cfg = MtpDrafterConfig()
pt = MtpDrafterModel(cfg).float().eval()
sd = torch.load("output/mtp_probe/mtp_drafter.pt", map_location="cpu")
pt.load_state_dict(sd)

# --- Test inputs ---
np.random.seed(42)
ctx = 512
activations = np.random.randn(1, 1, 3072).astype(np.float32) * 0.1
pos = 10

# Zero KV (no attention contribution)
kv13_k = np.zeros((1, 1, ctx, 256), dtype=np.float32)
kv13_v = np.zeros((1, 1, 256, ctx), dtype=np.float32)
kv14_k = np.zeros((1, 1, ctx, 512), dtype=np.float32)
kv14_v = np.zeros((1, 1, 512, ctx), dtype=np.float32)
mask_swa = np.full((1, 1, 1, ctx), -1e9, dtype=np.float32)
mask_swa[0, 0, 0, :pos+1] = 0
mask_full = np.full((1, 1, 1, ctx), -1e9, dtype=np.float32)
mask_full[0, 0, 0, :pos+1] = 0

a_t = torch.from_numpy(activations).float()
pos_t = torch.tensor([pos], dtype=torch.int32)
kv13_k_t = torch.from_numpy(kv13_k).float()
kv13_v_t = torch.from_numpy(kv13_v).float()
kv14_k_t = torch.from_numpy(kv14_k).float()
kv14_v_t = torch.from_numpy(kv14_v).float()
ms_t = torch.from_numpy(mask_swa).float()
mf_t = torch.from_numpy(mask_full).float()

# --- Dump intermediate activations layer-by-layer ---
with torch.no_grad():
    x = pt.mtp_pre_proj(a_t)
    print(f"post pre_proj: norm={x.norm().item():.4f} mean={x.mean().item():.4f} std={x.std().item():.4f}")

    swa_cos = pt.swa_cos[pos:pos+1].unsqueeze(0)
    swa_sin = pt.swa_sin[pos:pos+1].unsqueeze(0)
    full_cos = pt.full_cos[pos:pos+1].unsqueeze(0)
    full_sin = pt.full_sin[pos:pos+1].unsqueeze(0)

    for i, layer in enumerate(pt.layers):
        if i < 3:
            x = layer(x, kv13_k_t, kv13_v_t, swa_cos, swa_sin, ms_t)
        else:
            x = layer(x, kv14_k_t, kv14_v_t, full_cos, full_sin, mf_t)
        print(f"post L{i}: norm={x.norm().item():.4f} mean={x.mean().item():.4f} std={x.std().item():.4f}")

    h = pt.final_norm(x)
    print(f"post final_norm: norm={h.norm().item():.4f}")
    logits = pt.lm_head(h)
    print(f"post lm_head: norm={logits.norm().item():.4f}")
    logits = torch.tanh(logits / pt.softcap_factor) * pt.softcap_factor
    print(f"post softcap: argmax={logits.argmax(-1).item()}")

    # Compare lm_head_weight with ref
    lm_head_norm = pt.lm_head.weight.norm().item()
    final_norm_w_norm = pt.final_norm.weight.norm().item()
    pre_proj_norm = pt.mtp_pre_proj.weight.norm().item()
    print(f"\nWeight norms:")
    print(f"  mtp_pre_proj.weight: {pre_proj_norm:.2f}")
    print(f"  final_norm.weight:   {final_norm_w_norm:.2f}")
    print(f"  lm_head.weight:      {lm_head_norm:.2f}")

    # Check layer 3 (full attn) q_norm
    for i, layer in enumerate(pt.layers):
        qn = layer.attn.q_norm.weight
        print(f"  L{i}.attn.q_norm: shape={tuple(qn.shape)} mean={qn.mean().item():.4f} std={qn.std().item():.4f}")
