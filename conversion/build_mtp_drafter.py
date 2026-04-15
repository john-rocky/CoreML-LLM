#!/usr/bin/env python3
"""Convert MTP drafter to ANE-optimized CoreML mlpackage.

Input:   output/mtp_probe/mtp_drafter.pt  (from mtp_drafter_model.py)
Output:  mtp_drafter.mlpackage            (ANE-targeted, fp16, INT4 palettized)

Design choices for ANE:
  - All Linear → Conv2d(1,1) for 3× ANE throughput
  - RMSNorm → cat([x,-x]) + LayerNorm + slice trick (ANERMSNorm)
  - RoPE precomputed tables (same theta as target model)
  - In-model top-k(8) on logits to avoid shipping 262k floats from ANE
  - Logit softcapping: tanh(logits/30) * 30
  - T=1 per call: Swift calls K times for K-step drafting
  - No self-KV cache: reads target's kv13/kv14 directly

Usage:
    python conversion/build_mtp_drafter.py \
        --ckpt output/mtp_probe/mtp_drafter.pt \
        --output mtp_drafter.mlpackage \
        --palettize-int4

I/O contract:
  Inputs:
    embed_token    (1, 1, 1536) fp16  — raw embedding of current token
    proj_act       (1, 1, 1536) fp16  — projected_activations from prev step
                                        (or target's last hidden for step 0)
    kv13_k         (1, 1, W, 256) fp16  — target's sliding K
    kv13_v         (1, 1, 256, W) fp16  — target's sliding V
    kv14_k         (1, 1, C, 512) fp16  — target's full K
    kv14_v         (1, 1, 512, C) fp16  — target's full V
    cos_swa        (1, 128) fp16       — precomputed cos for this position (SWA)
    sin_swa        (1, 128) fp16
    cos_full       (1, 256) fp16       — precomputed cos for this position (full)
    sin_full       (1, 256) fp16
    mask_swa       (1, 1, 1, W) fp16   — sliding window causal mask
    mask_full      (1, 1, 1, C) fp16   — full context causal mask

  Outputs:
    top_k_indices  (8,) int32          — top-8 token indices
    top_k_values   (8,) fp16           — top-8 logit values (softcapped)
    proj_act_out   (1, 1, 1536) fp16   — projected_activations for next step
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ane_ops import MODEL_DTYPE, ANERMSNorm


# ---------------------------------------------------------------------------
# ANE-friendly MTP drafter
# ---------------------------------------------------------------------------

class MtpDrafterANE(nn.Module):
    """MTP drafter optimized for ANE inference.

    Key differences from the reference MtpDrafterModel:
    - Linear → Conv2d(1,1) for ANE throughput
    - ANERMSNorm for efficient normalization
    - In-model top-k instead of full logit output
    - Separate embed_token and proj_act inputs (no concat needed)
    - Pre-computed RoPE tables passed as inputs
    """

    def __init__(self):
        super().__init__()
        H = 256       # internal hidden
        TH = 1536     # target hidden
        FFN = 2048
        V = 262144    # vocab
        eps = 1e-6

        # Input projection: [embed_token || proj_act] → hidden
        # Decomposed to avoid concat: in_e(embed) + in_p(proj_act)
        self.in_e = nn.Conv2d(TH, H, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_p = nn.Conv2d(TH, H, 1, bias=False, dtype=MODEL_DTYPE)

        # 4 transformer layers
        self.layers = nn.ModuleList()
        for i in range(4):
            hd = 512 if i == 3 else 256
            nh = 4
            self.layers.append(MtpLayerANE(H, nh, hd, FFN, eps, i))

        # Output
        self.final_norm = ANERMSNorm(H, eps)
        # lm_head weight as buffer (not trainable)
        self.register_buffer("lm_head_weight",
                             torch.zeros(V, H, dtype=MODEL_DTYPE), persistent=False)
        self.mtp_post_proj = nn.Conv2d(H, TH, 1, bias=False, dtype=MODEL_DTYPE)

        self.softcap = 30.0

    def forward(
        self,
        embed_token,    # (1, 1, TH)  BSH format
        proj_act,       # (1, 1, TH)
        kv13_k,         # (1, 1, W, 256)
        kv13_v,         # (1, 1, 256, W)
        kv14_k,         # (1, 1, C, 512)
        kv14_v,         # (1, 1, 512, C)
        cos_swa,        # (1, 128)
        sin_swa,        # (1, 128)
        cos_full,       # (1, 256)
        sin_full,       # (1, 256)
        mask_swa,       # (1, 1, 1, W)
        mask_full,      # (1, 1, 1, C)
    ):
        # BSH → NCHW for conv
        e_nchw = embed_token.permute(0, 2, 1).unsqueeze(2)  # (1, TH, 1, 1)
        p_nchw = proj_act.permute(0, 2, 1).unsqueeze(2)

        # Input projection (decomposed concat)
        x_nchw = self.in_e(e_nchw) + self.in_p(p_nchw)  # (1, H, 1, 1)
        # NCHW → BSH
        x = x_nchw.squeeze(2).permute(0, 2, 1)  # (1, 1, H)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            if i < 3:
                x = layer(x, kv13_k, kv13_v, cos_swa, sin_swa, mask_swa)
            else:
                x = layer(x, kv14_k, kv14_v, cos_full, sin_full, mask_full)

        # Output
        h = self.final_norm(x)  # (1, 1, H) BSH

        # lm_head + softcap + top-k
        logits = F.linear(h.float(), self.lm_head_weight.float())  # (1, 1, V)
        logits = torch.tanh(logits / self.softcap) * self.softcap
        logits = logits.squeeze(0).squeeze(0)  # (V,)

        top_k_vals, top_k_ids = torch.topk(logits, k=8)

        # projected_activations
        h_nchw = h.permute(0, 2, 1).unsqueeze(2)  # (1, H, 1, 1)
        proj_out_nchw = self.mtp_post_proj(h_nchw)  # (1, TH, 1, 1)
        proj_out = proj_out_nchw.squeeze(2).permute(0, 2, 1)  # (1, 1, TH)

        return top_k_ids.to(torch.int32), top_k_vals.to(MODEL_DTYPE), proj_out


class MtpLayerANE(nn.Module):
    """Single drafter layer with sandwich norms, ANE-optimized."""

    def __init__(self, H: int, nh: int, hd: int, ffn: int, eps: float, layer_idx: int):
        super().__init__()
        self.nh = nh
        self.hd = hd
        self.layer_idx = layer_idx

        # Sandwich norms
        self.input_layernorm = ANERMSNorm(H, eps)
        self.post_attention_layernorm = ANERMSNorm(H, eps)
        self.pre_feedforward_layernorm = ANERMSNorm(H, eps)
        self.post_feedforward_layernorm = ANERMSNorm(H, eps)

        # Q-only attention
        self.q_proj = nn.Conv2d(H, nh * hd, 1, bias=False, dtype=MODEL_DTYPE)
        self.q_norm = ANERMSNorm(hd, eps)
        self.o_proj = nn.Conv2d(nh * hd, H, 1, bias=False, dtype=MODEL_DTYPE)

        # GeGLU MLP
        self.gate1 = nn.Conv2d(H, ffn, 1, bias=False, dtype=MODEL_DTYPE)
        self.gate2 = nn.Conv2d(H, ffn, 1, bias=False, dtype=MODEL_DTYPE)
        self.down = nn.Conv2d(ffn, H, 1, bias=False, dtype=MODEL_DTYPE)

    def forward(self, x, kv_k, kv_v, cos, sin, mask):
        # x: (1, 1, H) BSH format

        # Attention
        residual = x
        h = self.input_layernorm(x)  # (1, 1, H) BSH

        # BSH → NCHW for Q projection
        h_nchw = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (1, H, 1, 1)
        q = self.q_proj(h_nchw)  # (1, nh*hd, 1, 1)

        # Per-head Q norm: reshape to (1, nh, 1, hd), norm each head
        q = q.view(1, self.nh, self.hd, 1).permute(0, 1, 3, 2)  # (1, nh, 1, hd)
        # Apply q_norm per-head via reshape: (nh, 1, hd) → norm on hd
        q_flat = q.view(self.nh, 1, self.hd)  # (nh, 1, hd)
        q_normed = self.q_norm(q_flat)  # (nh, 1, hd) — ANERMSNorm on last dim
        q = q_normed.view(1, self.nh, 1, self.hd)  # (1, nh, 1, hd)

        # Apply RoPE to Q
        half = self.hd // 2
        q1, q2 = q[..., :half], q[..., half:]
        cos_r = cos.view(1, 1, 1, half)
        sin_r = sin.view(1, 1, 1, half)
        q = torch.cat([q1 * cos_r - q2 * sin_r, q2 * cos_r + q1 * sin_r], dim=-1)

        # Attention: Q @ K^T
        q = q.to(MODEL_DTYPE)  # (1, nh, 1, hd)
        k_t = kv_k.transpose(-2, -1).to(MODEL_DTYPE)  # (1, 1, hd, ctx)
        attn = torch.matmul(q.float(), k_t.float())  # (1, nh, 1, ctx)
        attn = attn + mask.float()
        attn = F.softmax(attn, dim=-1).to(MODEL_DTYPE)

        # Attn @ V
        v_t = kv_v.transpose(-2, -1).to(MODEL_DTYPE)  # (1, 1, ctx, hd)
        out = torch.matmul(attn.float(), v_t.float()).to(MODEL_DTYPE)  # (1, nh, 1, hd)

        # Reshape → NCHW → o_proj → BSH
        out_nchw = out.reshape(1, self.nh * self.hd, 1, 1)
        h_attn_nchw = self.o_proj(out_nchw)  # (1, H, 1, 1)
        h_attn = h_attn_nchw.squeeze(2).permute(0, 2, 1)  # (1, 1, H) BSH
        h_attn = self.post_attention_layernorm(h_attn)
        x = residual + h_attn

        # MLP with sandwich norm
        residual = x
        h = self.pre_feedforward_layernorm(x)  # (1, 1, H) BSH
        h_nchw = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (1, H, 1, 1)
        g1 = self.gate1(h_nchw)
        g2 = self.gate2(h_nchw)
        mlp_nchw = self.down(F.gelu(g1) * g2)  # (1, H, 1, 1)
        mlp_out = mlp_nchw.squeeze(2).permute(0, 2, 1)  # (1, 1, H) BSH
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        x = residual + mlp_out

        return x


# ---------------------------------------------------------------------------
# Weight loading from reference model
# ---------------------------------------------------------------------------

def load_ane_weights(ane: MtpDrafterANE, ref_state: dict, lm_head_weight: torch.Tensor):
    """Load weights from reference MtpDrafterModel into ANE model."""

    def to_conv(w):
        """Convert Linear weight (out, in) → Conv2d weight (out, in, 1, 1)."""
        return w.unsqueeze(-1).unsqueeze(-1).to(MODEL_DTYPE)

    sd = ane.state_dict()

    # Input projection: split mtp_pre_proj (3072→256) into in_e(1536→256) + in_p(1536→256)
    W_pre = ref_state["mtp_pre_proj.weight"]  # (256, 3072)
    sd["in_e.weight"] = to_conv(W_pre[:, :1536])  # first half = embed
    sd["in_p.weight"] = to_conv(W_pre[:, 1536:])  # second half = proj_act

    # Layers
    for i in range(4):
        pfx = f"layers.{i}"
        ref_pfx = f"layers.{i}"

        # Sandwich norms
        for norm_name in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
            sd[f"{pfx}.{norm_name}.weight"] = ref_state[f"{ref_pfx}.{norm_name}.weight"].to(MODEL_DTYPE)

        # Attention
        sd[f"{pfx}.q_proj.weight"] = to_conv(ref_state[f"{ref_pfx}.attn.q_proj.weight"])
        sd[f"{pfx}.q_norm.weight"] = ref_state[f"{ref_pfx}.attn.q_norm.weight"].to(MODEL_DTYPE)
        sd[f"{pfx}.o_proj.weight"] = to_conv(ref_state[f"{ref_pfx}.attn.o_proj.weight"])

        # MLP
        sd[f"{pfx}.gate1.weight"] = to_conv(ref_state[f"{ref_pfx}.mlp.gate1.weight"])
        sd[f"{pfx}.gate2.weight"] = to_conv(ref_state[f"{ref_pfx}.mlp.gate2.weight"])
        sd[f"{pfx}.down.weight"] = to_conv(ref_state[f"{ref_pfx}.mlp.down.weight"])

    # Output
    sd["final_norm.weight"] = ref_state["final_norm.weight"].to(MODEL_DTYPE)
    sd["mtp_post_proj.weight"] = to_conv(ref_state["mtp_post_proj.weight"])

    # lm_head
    with torch.no_grad():
        ane.lm_head_weight.copy_(lm_head_weight.to(MODEL_DTYPE))

    ane.load_state_dict(sd, strict=False)
    print(f"  Loaded {len(sd)} tensors into ANE model")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="output/mtp_probe/mtp_drafter.pt")
    ap.add_argument("--output", type=str, default="mtp_drafter.mlpackage")
    ap.add_argument("--palettize-int4", action="store_true",
                    help="Apply INT4 palettization (group_size=32)")
    ap.add_argument("--sliding-window", type=int, default=512,
                    help="Sliding window size for SWA layers")
    ap.add_argument("--context-length", type=int, default=8192,
                    help="Max context length for full-attention layer")
    args = ap.parse_args()

    W = args.sliding_window
    C = args.context_length
    H = 256
    TH = 1536

    print(f"Building MTP drafter ANE model (W={W}, C={C})...")
    ane = MtpDrafterANE().to(MODEL_DTYPE).eval()
    print(f"  Params: {sum(p.numel() for p in ane.parameters()):,}")

    # Load weights
    print(f"Loading checkpoint: {args.ckpt}")
    ref_state = torch.load(args.ckpt, map_location="cpu")
    lm_head_weight = ref_state["lm_head.weight"]
    load_ane_weights(ane, ref_state, lm_head_weight)

    # Forward test
    print("\nForward test...")
    with torch.no_grad():
        embed = torch.zeros(1, 1, TH, dtype=MODEL_DTYPE)
        proj = torch.zeros(1, 1, TH, dtype=MODEL_DTYPE)
        kv13_k = torch.zeros(1, 1, W, 256, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 256, W, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, C, 512, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 512, C, dtype=MODEL_DTYPE)
        cos_s = torch.ones(1, 128, dtype=MODEL_DTYPE)
        sin_s = torch.zeros(1, 128, dtype=MODEL_DTYPE)
        cos_f = torch.ones(1, 256, dtype=MODEL_DTYPE)
        sin_f = torch.zeros(1, 256, dtype=MODEL_DTYPE)
        mask_s = torch.zeros(1, 1, 1, W, dtype=MODEL_DTYPE)
        mask_f = torch.zeros(1, 1, 1, C, dtype=MODEL_DTYPE)

        top_ids, top_vals, proj_out = ane(
            embed, proj, kv13_k, kv13_v, kv14_k, kv14_v,
            cos_s, sin_s, cos_f, sin_f, mask_s, mask_f
        )
    print(f"  top_ids:  {tuple(top_ids.shape)} {top_ids.dtype}")
    print(f"  top_vals: {tuple(top_vals.shape)} {top_vals.dtype}")
    print(f"  proj_out: {tuple(proj_out.shape)} {proj_out.dtype}")
    print("  Forward OK")

    # CoreML conversion
    print("\nConverting to CoreML...")
    import coremltools as ct

    # Trace
    traced = torch.jit.trace(ane, (
        embed, proj, kv13_k, kv13_v, kv14_k, kv14_v,
        cos_s, sin_s, cos_f, sin_f, mask_s, mask_f
    ), strict=False)

    fp16_type = ct.converters.mil.mil.types.fp16

    mlm = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="embed_token",  shape=(1, 1, TH), dtype=fp16_type),
            ct.TensorType(name="proj_act",     shape=(1, 1, TH), dtype=fp16_type),
            ct.TensorType(name="kv13_k",       shape=(1, 1, W, 256), dtype=fp16_type),
            ct.TensorType(name="kv13_v",       shape=(1, 1, 256, W), dtype=fp16_type),
            ct.TensorType(name="kv14_k",       shape=(1, 1, C, 512), dtype=fp16_type),
            ct.TensorType(name="kv14_v",       shape=(1, 1, 512, C), dtype=fp16_type),
            ct.TensorType(name="cos_swa",      shape=(1, 128), dtype=fp16_type),
            ct.TensorType(name="sin_swa",      shape=(1, 128), dtype=fp16_type),
            ct.TensorType(name="cos_full",     shape=(1, 256), dtype=fp16_type),
            ct.TensorType(name="sin_full",     shape=(1, 256), dtype=fp16_type),
            ct.TensorType(name="mask_swa",     shape=(1, 1, 1, W), dtype=fp16_type),
            ct.TensorType(name="mask_full",    shape=(1, 1, 1, C), dtype=fp16_type),
        ],
        outputs=[
            ct.TensorType(name="top_k_indices"),
            ct.TensorType(name="top_k_values"),
            ct.TensorType(name="proj_act_out"),
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS26,
    )

    if args.palettize_int4:
        print("  Palettizing weights INT4 (group_size=32)...")
        import coremltools.optimize.coreml as cto
        pal_cfg = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                mode="kmeans", nbits=4,
                granularity="per_grouped_channel", group_size=32
            )
        )
        mlm = cto.palettize_weights(mlm, pal_cfg)

    mlm.save(args.output)
    size_mb = sum(f.stat().st_size for f in Path(args.output).rglob("*") if f.is_file()) / 1e6
    print(f"\n  Saved: {args.output} ({size_mb:.1f} MB)")

    print("\nNext steps:")
    print(f"  1. Copy {args.output} to device")
    print("  2. Wire into MtpDraftSource.swift")
    print("  3. Feed hidden_at_L34 from ChunkedEngine as proj_act for step 0")
    print("  4. Feed target embedder output as embed_token")
    print("  5. K-step loop: draft token → embed(token) → next call")


if __name__ == "__main__":
    main()
