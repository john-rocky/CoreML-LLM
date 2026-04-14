#!/usr/bin/env python3
"""Convert a single MTP module to CoreML mlpackage (ANE-targeted, fp16, INT4).

Used BOTH for:
  (a) ANE latency measurement (dummy random weights)  — Gate 1 in Path C
  (b) Production deployment (trained weights)          — Phase 3

Single-module forward contract (T=1 inference):
  Inputs:
    hidden_prev   (1, 1, 1536) fp16  — L34 hidden from trunk OR previous module
    embed_token   (1, 1, 1536) fp16  — scaled embedding (× sqrt(H))
    kv_k_in       (1, nkv, W, hd) fp16  — module's own K cache (W=128)
    kv_v_in       (1, nkv, W, hd) fp16  — module's own V cache (pre-transposed
                                          (hd, W) saves per-call transpose;
                                          keeping (W, hd) for v1 simplicity)
    cos           (1, hd/2) fp16
    sin           (1, hd/2) fp16
    mask          (1, 1, 1, W) fp16
    update_idx    (1, 1, W, 1) fp16  — one-hot position for KV scatter

  Outputs:
    top_k_indices (8,) int32
    top_k_values  (8,) fp16
    hidden_out    (1, 1, 1536) fp16  — feeds next module
    kv_k_out      (1, nkv, W, hd) fp16  — updated cache
    kv_v_out      (1, nkv, W, hd) fp16

Design:
  - Linear → Conv2d(1,1) for ANE throughput
  - ANERMSNorm (cat+LayerNorm+slice trick)
  - In-model top-k(8) to avoid 262144-dim output transfer
  - INT4 palettization (group=32) post-conversion
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ane_ops import MODEL_DTYPE, ANERMSNorm

from train_mtp_modules.mtp_modules import MtpModuleConfig


# ---------------------------------------------------------------------------
# ANE-friendly single module (Conv2d everywhere, T=1)
# ---------------------------------------------------------------------------

class MtpModuleANE(nn.Module):
    """ANE-optimized single MTP module. T=1 inference only.

    CoreML entry point: forward(...) returns (top_k_indices, top_k_values,
    hidden_out, kv_k_out, kv_v_out).
    """

    def __init__(self, cfg: MtpModuleConfig, include_lm_head: bool = True):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        I = cfg.intermediate_size
        NH = cfg.num_attention_heads
        NKV = cfg.num_kv_heads
        HD = cfg.head_dim
        W = cfg.kv_window

        # Input fusion (Conv1x1 = 2H → H)
        # Decomposed concat: in_h(hidden_prev) + in_e(embed)
        self.in_h = nn.Conv2d(H, H, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_e = nn.Conv2d(H, H, 1, bias=False, dtype=MODEL_DTYPE)

        # Gemma sandwich norms
        self.input_layernorm = ANERMSNorm(H, cfg.rms_norm_eps)
        self.post_attention_layernorm = ANERMSNorm(H, cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = ANERMSNorm(H, cfg.rms_norm_eps)
        self.post_feedforward_layernorm = ANERMSNorm(H, cfg.rms_norm_eps)
        self.final_norm = ANERMSNorm(H, cfg.rms_norm_eps)

        # Attention (Q over T=1, KV read from cache)
        self.q_proj = nn.Conv2d(H, NH * HD, 1, bias=False, dtype=MODEL_DTYPE)
        self.k_proj = nn.Conv2d(H, NKV * HD, 1, bias=False, dtype=MODEL_DTYPE)
        self.v_proj = nn.Conv2d(H, NKV * HD, 1, bias=False, dtype=MODEL_DTYPE)
        self.o_proj = nn.Conv2d(NH * HD, H, 1, bias=False, dtype=MODEL_DTYPE)
        self.q_norm = ANERMSNorm(HD, cfg.rms_norm_eps)
        self.k_norm = ANERMSNorm(HD, cfg.rms_norm_eps)

        # GeGLU MLP
        self.gate_proj = nn.Conv2d(H, I, 1, bias=False, dtype=MODEL_DTYPE)
        self.up_proj = nn.Conv2d(H, I, 1, bias=False, dtype=MODEL_DTYPE)
        self.down_proj = nn.Conv2d(I, H, 1, bias=False, dtype=MODEL_DTYPE)

        # Tied LM head (optional at build time — for ANE latency test w/ random
        # weights we can skip the huge 262144-dim lm_head)
        self.include_lm_head = include_lm_head
        if include_lm_head:
            self.register_buffer(
                "lm_head_weight",
                torch.zeros(cfg.vocab_size, H, dtype=MODEL_DTYPE),
                persistent=False,
            )

        self.softcap = cfg.logit_softcap
        self.nh = NH
        self.nkv = NKV
        self.hd = HD
        self.W = W

    def forward(
        self,
        hidden_prev: torch.Tensor,  # (1, 1, H) BSH
        embed_token: torch.Tensor,  # (1, 1, H)
        kv_k_in: torch.Tensor,      # (1, NKV, W, HD)
        kv_v_in: torch.Tensor,      # (1, NKV, W, HD)
        cos: torch.Tensor,          # (1, HD/2)
        sin: torch.Tensor,          # (1, HD/2)
        mask: torch.Tensor,         # (1, 1, 1, W)
        update_idx: torch.Tensor,   # (1, 1, W, 1) one-hot, indicates write slot
    ):
        H = self.cfg.hidden_size
        NKV = self.nkv
        HD = self.hd
        W = self.W
        NH = self.nh

        # BSH → NCHW (1, H, 1, 1)
        h_nchw = hidden_prev.permute(0, 2, 1).unsqueeze(2)
        e_nchw = embed_token.permute(0, 2, 1).unsqueeze(2)

        # Input fusion
        x_nchw = self.in_h(h_nchw) + self.in_e(e_nchw)  # (1, H, 1, 1)
        x = x_nchw.squeeze(2).permute(0, 2, 1)  # (1, 1, H) BSH

        # Attention branch
        residual = x
        h_norm = self.input_layernorm(x)
        h_norm_nchw = h_norm.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)

        q = self.q_proj(h_norm_nchw)  # (1, NH*HD, 1, 1)
        k_new = self.k_proj(h_norm_nchw)  # (1, NKV*HD, 1, 1)
        v_new = self.v_proj(h_norm_nchw)  # (1, NKV*HD, 1, 1)

        # Reshape + q_norm/k_norm
        q = q.view(1, NH, HD, 1).permute(0, 1, 3, 2)  # (1, NH, 1, HD)
        q_flat = q.view(NH, 1, HD)
        q_normed = self.q_norm(q_flat).view(1, NH, 1, HD)
        q = q_normed

        k_new = k_new.view(1, NKV, HD, 1).permute(0, 1, 3, 2)  # (1, NKV, 1, HD)
        k_flat = k_new.view(NKV, 1, HD)
        k_new_normed = self.k_norm(k_flat).view(1, NKV, 1, HD)

        v_new_reshaped = v_new.view(1, NKV, HD, 1).permute(0, 1, 3, 2)  # (1, NKV, 1, HD)

        # Apply RoPE to Q and K_new
        half = HD // 2
        q1, q2 = q[..., :half], q[..., half:]
        cos_r = cos.view(1, 1, 1, half)
        sin_r = sin.view(1, 1, 1, half)
        q = torch.cat([q1 * cos_r - q2 * sin_r, q2 * cos_r + q1 * sin_r], dim=-1)

        k1, k2 = k_new_normed[..., :half], k_new_normed[..., half:]
        k_new_rot = torch.cat([k1 * cos_r - k2 * sin_r, k2 * cos_r + k1 * sin_r], dim=-1)

        # Scatter K_new and V_new into cache at update_idx position
        # update_idx: (1, 1, W, 1) — broadcast to (1, NKV, W, HD)
        # Multiply old cache by (1 - update_idx) and add K_new broadcast by update_idx
        # We pad k_new to full window then masked-add
        k_new_broadcast = k_new_rot.expand(1, NKV, W, HD) * update_idx
        v_new_broadcast = v_new_reshaped.expand(1, NKV, W, HD) * update_idx
        keep_mask = 1.0 - update_idx
        kv_k_out = kv_k_in * keep_mask + k_new_broadcast
        kv_v_out = kv_v_in * keep_mask + v_new_broadcast

        # Q @ K^T over W positions
        # Q: (1, NH, 1, HD), K: (1, NKV, W, HD) — broadcast NKV→NH
        if NH != NKV:
            n_rep = NH // NKV
            k_expanded = kv_k_out.repeat_interleave(n_rep, dim=1)
            v_expanded = kv_v_out.repeat_interleave(n_rep, dim=1)
        else:
            k_expanded = kv_k_out
            v_expanded = kv_v_out

        scale = 1.0 / math.sqrt(HD)
        attn = torch.matmul(q.float(), k_expanded.transpose(-2, -1).float()) * scale
        attn = attn + mask.float()
        attn = F.softmax(attn, dim=-1).to(MODEL_DTYPE)

        # attn @ V: (1, NH, 1, W) @ (1, NH, W, HD) = (1, NH, 1, HD)
        out = torch.matmul(attn.float(), v_expanded.float()).to(MODEL_DTYPE)
        out_nchw = out.reshape(1, NH * HD, 1, 1)
        h_attn = self.o_proj(out_nchw)
        h_attn = h_attn.squeeze(2).permute(0, 2, 1)  # (1, 1, H)

        h_attn = self.post_attention_layernorm(h_attn)
        x = residual + h_attn

        # MLP branch
        residual = x
        h_ffn = self.pre_feedforward_layernorm(x)
        h_ffn_nchw = h_ffn.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        g = self.gate_proj(h_ffn_nchw)
        u = self.up_proj(h_ffn_nchw)
        d = self.down_proj(F.gelu(g) * u)
        h_ffn_out = d.squeeze(2).permute(0, 2, 1)
        h_ffn_out = self.post_feedforward_layernorm(h_ffn_out)
        x = residual + h_ffn_out

        # Final norm → hidden_out
        hidden_out = self.final_norm(x)  # (1, 1, H)

        if self.include_lm_head:
            # LM head + softcap + argmax (top-1).
            #
            # NOTE: we used to emit torch.topk(k=8) to carry candidates for
            # speculative sampling. An equivalence test on coremltools 9.0
            # showed topk output being truncated to the lower 16 bits of the
            # vocab index for the large V=262144 case (indices off by k*65536),
            # but ONLY when topk followed the full MtpModule graph (squeeze +
            # softcap); isolated topk was fine. Switching to argmax side-steps
            # the bug and returns exactly what the Swift consumer reads (first
            # element of top_k_indices). Kept output name for Swift compatibility.
            logits = F.linear(hidden_out.float(), self.lm_head_weight.float())
            logits = torch.tanh(logits / self.softcap) * self.softcap  # (1, 1, V)
            argmax_id = logits.argmax(dim=-1)  # (1, 1)
            top_k_indices = argmax_id.reshape(1).to(torch.int32)  # (1,)
            # Return value at the argmax for introspection (not used by Swift).
            top_k_values = logits.gather(-1, argmax_id.unsqueeze(-1)).reshape(1).to(MODEL_DTYPE)
            return (
                top_k_indices,
                top_k_values,
                hidden_out,
                kv_k_out,
                kv_v_out,
            )
        else:
            return hidden_out, kv_k_out, kv_v_out


# ---------------------------------------------------------------------------
# Main: build and convert
# ---------------------------------------------------------------------------

def load_trained_module_weights(ane_model: MtpModuleANE, ckpt_path: str,
                                module_idx: int):
    """Extract module[module_idx] weights from a trained MtpStack checkpoint
    and load into an MtpModuleANE (ANE Conv2d layout).

    Training MtpModule uses nn.Linear; inference MtpModuleANE uses nn.Conv2d(1,1).
    Weight shape conversion: Linear (out, in) → Conv2d (out, in, 1, 1).

    The training module's `input_proj` takes concat(hidden_prev, token_embed)
    of shape (..., 2H). Split into two separate Conv1x1s: in_h and in_e.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    prefix = f"modules_list.{module_idx}."
    module_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not module_sd:
        raise RuntimeError(f"No keys with prefix {prefix!r} found in checkpoint")
    print(f"  Found {len(module_sd)} tensors for module_{module_idx}")

    H = ane_model.cfg.hidden_size

    def to_conv(w):
        return w.unsqueeze(-1).unsqueeze(-1).to(MODEL_DTYPE)

    ane_sd = {}

    # Input fusion: split input_proj (H, 2H) → in_h (H, H, 1, 1) + in_e (H, H, 1, 1)
    # input_proj was called with cat([hidden_prev, token_embed], dim=-1):
    # - weight[:, :H] multiplies hidden_prev → in_h
    # - weight[:, H:] multiplies token_embed → in_e
    W_in = module_sd["input_proj.weight"]  # (H, 2H)
    ane_sd["in_h.weight"] = to_conv(W_in[:, :H])
    ane_sd["in_e.weight"] = to_conv(W_in[:, H:])

    # Sandwich norms + final norm (same names, direct copy)
    for norm_key in ["input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm",
                     "final_norm"]:
        ane_sd[f"{norm_key}.weight"] = module_sd[f"{norm_key}.weight"].to(MODEL_DTYPE)

    # Attention projections: Linear → Conv2d
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        ane_sd[f"{name}.weight"] = to_conv(module_sd[f"attn.{name}.weight"])
    # Attention head norms
    ane_sd["q_norm.weight"] = module_sd["attn.q_norm.weight"].to(MODEL_DTYPE)
    ane_sd["k_norm.weight"] = module_sd["attn.k_norm.weight"].to(MODEL_DTYPE)

    # MLP projections (gate/up/down)
    for name in ["gate_proj", "up_proj", "down_proj"]:
        ane_sd[f"{name}.weight"] = to_conv(module_sd[f"mlp.{name}.weight"])

    # Load into ANE model (strict=False because lm_head_weight is a buffer
    # loaded separately from the HF trunk)
    missing, unexpected = ane_model.load_state_dict(ane_sd, strict=False)
    # lm_head_weight (buffer) is expected-missing; warn on others
    real_missing = [k for k in missing if "lm_head_weight" not in k]
    if real_missing:
        print(f"  WARNING: missing keys after load: {real_missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")

    print(f"  Loaded {len(ane_sd)} weights from module_{module_idx}")


def load_lm_head_from_hf(ane_model: MtpModuleANE, hf_dir: str):
    """Load the tied LM head weight from the HF trunk (= embed_tokens.weight)."""
    from transformers import Gemma4ForConditionalGeneration
    print(f"  Loading HF trunk for tied lm_head from {hf_dir}")
    # device_map requires `accelerate`; omit to work on vanilla envs (e.g. local Mac).
    hf = Gemma4ForConditionalGeneration.from_pretrained(
        hf_dir, torch_dtype=torch.float32,
    )
    lm = hf.model.language_model
    embed = lm.embed_tokens.weight.detach().clone()
    with torch.no_grad():
        ane_model.lm_head_weight.copy_(embed.to(MODEL_DTYPE))
    print(f"  lm_head_weight loaded: {tuple(embed.shape)} norm={embed.norm().item():.2f}")
    del hf, lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Trained MtpStack checkpoint (.pt). If None, random weights.")
    ap.add_argument("--module-idx", type=int, default=0,
                    help="Which module from the trained stack to convert (0 or 1 for K=2).")
    ap.add_argument("--hf-dir", type=str, default=None,
                    help="HF Gemma 4 dir for loading tied lm_head. Required with --ckpt.")
    ap.add_argument("--output", type=str, default="/tmp/mtp_module.mlpackage")
    ap.add_argument("--palettize-int4", action="store_true")
    ap.add_argument("--include-lm-head", action="store_true", default=True,
                    help="Include tied LM head (adds ~100MB to model size). "
                         "Omit for fastest ANE latency test without full top-k.")
    ap.add_argument("--no-lm-head", dest="include_lm_head", action="store_false")
    args = ap.parse_args()

    cfg = MtpModuleConfig()
    print(f"Config: H={cfg.hidden_size}, I={cfg.intermediate_size}, "
          f"NH={cfg.num_attention_heads}, HD={cfg.head_dim}, W={cfg.kv_window}")

    model = MtpModuleANE(cfg, include_lm_head=args.include_lm_head).to(MODEL_DTYPE).eval()
    trainable = sum(p.numel() for p in model.parameters())
    print(f"Module params: {trainable:,}")

    if args.ckpt:
        print(f"Loading module_{args.module_idx} from {args.ckpt}")
        load_trained_module_weights(model, args.ckpt, args.module_idx)
        if args.include_lm_head:
            if not args.hf_dir:
                raise ValueError("--hf-dir is required when --ckpt is provided "
                                 "with --include-lm-head")
            load_lm_head_from_hf(model, args.hf_dir)
    else:
        print("Using RANDOM weights (for latency test only)")
        if args.include_lm_head:
            with torch.no_grad():
                model.lm_head_weight.copy_(
                    torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=MODEL_DTYPE) * 0.02
                )

    # Dummy inputs for tracing
    H = cfg.hidden_size
    NKV = cfg.num_kv_heads
    HD = cfg.head_dim
    W = cfg.kv_window
    half = HD // 2

    hidden_prev = torch.zeros(1, 1, H, dtype=MODEL_DTYPE)
    embed = torch.zeros(1, 1, H, dtype=MODEL_DTYPE)
    kv_k = torch.zeros(1, NKV, W, HD, dtype=MODEL_DTYPE)
    kv_v = torch.zeros(1, NKV, W, HD, dtype=MODEL_DTYPE)
    cos = torch.ones(1, half, dtype=MODEL_DTYPE)
    sin = torch.zeros(1, half, dtype=MODEL_DTYPE)
    mask = torch.zeros(1, 1, 1, W, dtype=MODEL_DTYPE)
    update_idx = torch.zeros(1, 1, W, 1, dtype=MODEL_DTYPE)
    update_idx[0, 0, 0, 0] = 1.0  # write at slot 0 for dummy test

    # Forward smoke
    print("\nForward smoke test...")
    with torch.no_grad():
        if args.include_lm_head:
            top_ids, top_vals, hidden_out, kv_k_out, kv_v_out = model(
                hidden_prev, embed, kv_k, kv_v, cos, sin, mask, update_idx
            )
            print(f"  top_ids: {tuple(top_ids.shape)} {top_ids.dtype}")
            print(f"  top_vals: {tuple(top_vals.shape)} {top_vals.dtype}")
        else:
            hidden_out, kv_k_out, kv_v_out = model(
                hidden_prev, embed, kv_k, kv_v, cos, sin, mask, update_idx
            )
        print(f"  hidden_out: {tuple(hidden_out.shape)}")
        print(f"  kv_k_out: {tuple(kv_k_out.shape)}  kv_v_out: {tuple(kv_v_out.shape)}")

    # CoreML conversion
    print("\nConverting to CoreML...")
    import coremltools as ct

    traced = torch.jit.trace(
        model,
        (hidden_prev, embed, kv_k, kv_v, cos, sin, mask, update_idx),
        strict=False,
    )

    fp16_type = ct.converters.mil.mil.types.fp16
    inputs = [
        ct.TensorType(name="hidden_prev", shape=(1, 1, H), dtype=fp16_type),
        ct.TensorType(name="embed_token", shape=(1, 1, H), dtype=fp16_type),
        ct.TensorType(name="kv_k_in", shape=(1, NKV, W, HD), dtype=fp16_type),
        ct.TensorType(name="kv_v_in", shape=(1, NKV, W, HD), dtype=fp16_type),
        ct.TensorType(name="cos", shape=(1, half), dtype=fp16_type),
        ct.TensorType(name="sin", shape=(1, half), dtype=fp16_type),
        ct.TensorType(name="mask", shape=(1, 1, 1, W), dtype=fp16_type),
        ct.TensorType(name="update_idx", shape=(1, 1, W, 1), dtype=fp16_type),
    ]

    if args.include_lm_head:
        outputs = [
            ct.TensorType(name="top_k_indices"),
            ct.TensorType(name="top_k_values"),
            ct.TensorType(name="hidden_out"),
            ct.TensorType(name="kv_k_out"),
            ct.TensorType(name="kv_v_out"),
        ]
    else:
        outputs = [
            ct.TensorType(name="hidden_out"),
            ct.TensorType(name="kv_k_out"),
            ct.TensorType(name="kv_v_out"),
        ]

    mlm = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    if args.palettize_int4:
        print("  Palettizing INT4 (group=32)...")
        import coremltools.optimize.coreml as cto
        cfg_opt = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                mode="kmeans", nbits=4,
                granularity="per_grouped_channel", group_size=32
            )
        )
        mlm = cto.palettize_weights(mlm, cfg_opt)

    mlm.save(args.output)
    size_mb = sum(f.stat().st_size for f in Path(args.output).rglob("*") if f.is_file()) / 1e6
    print(f"\n  Saved: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
