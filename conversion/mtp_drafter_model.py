#!/usr/bin/env python3
"""PyTorch reimplementation of Google's MTP drafter from Gemma 4 E2B LiteRT.

Architecture:
  mtp_pre_proj  Linear(3072 → 256)
  layer_0-2     SWA (q=1024, FFN=2048, reads kv13)
  layer_3       Full-attn (q=2048, FFN=2048, reads kv14)
  final_norm    RMSNorm(256)
  lm_head       Linear(256 → 262144)  [logits]
  mtp_post_proj Linear(256 → 1536)    [projected_activations]

Key: drafter has Q-only attention (no K/V projections). It reads K/V directly
from the target model's kv_cache_13 (sliding) and kv_cache_14 (full attention).

Usage:
    python conversion/mtp_drafter_model.py \
        --tflite output/mtp_probe/section_10.tflite \
        --output output/mtp_probe/mtp_drafter.pt
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class MtpDrafterConfig:
    hidden_size: int = 256          # internal dim
    input_size: int = 3072          # 2 × target hidden (1536)
    target_hidden: int = 1536       # target model hidden_size
    num_layers: int = 4
    # SWA layers (0-2)
    swa_num_heads: int = 4
    swa_head_dim: int = 256         # head_dim for sliding layers
    swa_q_dim: int = 1024           # 4 × 256
    # Full layer (3)
    full_num_heads: int = 4
    full_head_dim: int = 512        # head_dim for full-attn layer
    full_q_dim: int = 2048          # 4 × 512
    # MLP
    ffn_dim: int = 2048
    # Vocab
    vocab_size: int = 262144
    # RoPE
    rope_theta: float = 10000.0
    # Norm
    rms_eps: float = 1e-6


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


def _precompute_rope(head_dim: int, max_pos: int, theta: float = 10000.0):
    """Precompute cos/sin tables for RoPE."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos_table = torch.cos(freqs)  # (max_pos, half)
    sin_table = torch.sin(freqs)
    return cos_table, sin_table


def _apply_rope_q(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to Q only. q: (B, H, 1, D), cos/sin: (1, D//2)."""
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)


class DrafterAttention(nn.Module):
    """Q-only attention that reads external K/V caches."""

    def __init__(self, hidden: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(
        self,
        x: torch.Tensor,           # (B, 1, hidden)
        kv_k: torch.Tensor,        # (B, 1, ctx, head_dim) int8 → dequant to fp
        kv_v: torch.Tensor,        # (B, 1, head_dim, ctx) int8 → dequant to fp
        cos: torch.Tensor,         # (1, head_dim//2) for this position
        sin: torch.Tensor,
        mask: torch.Tensor,        # (B, 1, 1, ctx) causal mask
    ) -> torch.Tensor:
        B = x.shape[0]

        # Q projection
        q = self.q_proj(x)  # (B, 1, num_heads * head_dim)
        q = q.view(B, 1, self.num_heads, self.head_dim)
        # Per-head Q norm
        q = self.q_norm(q)
        q = q.permute(0, 2, 1, 3)  # (B, H, 1, D)

        # Apply RoPE to Q
        q = _apply_rope_q(q, cos, sin)

        # External K/V (already RoPE'd and normed in target model)
        # K: (B, 1, ctx, D) → (B, 1, D, ctx) for matmul
        # Actually K in TFLite is stored as (B, num_kv_heads, ctx, head_dim)
        # and V as (B, num_kv_heads, head_dim, ctx)
        # For the drafter: num_kv_heads=1 for SWA, and we broadcast to num_heads
        k = kv_k.float()  # (B, 1, ctx, D)
        v = kv_v.float()  # (B, 1, D, ctx)

        # Attention: Q @ K^T
        # Q: (B, H, 1, D), K: (B, 1, ctx, D) → K^T: (B, 1, D, ctx)
        # Broadcasting: K^T will broadcast from kv_heads=1 to H
        attn = torch.matmul(q.float(), k.transpose(-2, -1)) * self.scale  # (B, H, 1, ctx)
        attn = attn + mask.float()
        attn = F.softmax(attn, dim=-1).to(x.dtype)

        # attn @ V: V is (B, 1, D, ctx) → need (B, 1, ctx, D)
        v_t = v.transpose(-2, -1)  # (B, 1, ctx, D)
        out = torch.matmul(attn.float(), v_t.float()).to(x.dtype)  # (B, H, 1, D)
        out = out.permute(0, 2, 1, 3).reshape(B, 1, self.num_heads * self.head_dim)

        return self.o_proj(out)


class DrafterMLP(nn.Module):
    """GeGLU MLP: gelu(gate1(x)) * gate2(x), then down."""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.gate1 = nn.Linear(hidden, ffn, bias=False)
        self.gate2 = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.gate1(x)) * self.gate2(x))


class DrafterLayer(nn.Module):
    """Drafter layer with Gemma 4 sandwich norms (4 RMSNorms per layer)."""

    def __init__(self, hidden: int, num_heads: int, head_dim: int, ffn: int, eps: float):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps)
        self.attn = DrafterAttention(hidden, num_heads, head_dim)
        self.mlp = DrafterMLP(hidden, ffn)

    def forward(self, x, kv_k, kv_v, cos, sin, mask):
        # Attention with sandwich norm
        residual = x
        h = self.input_layernorm(x)
        h = self.attn(h, kv_k, kv_v, cos, sin, mask)
        h = self.post_attention_layernorm(h)
        x = residual + h

        # MLP with sandwich norm
        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = residual + h
        return x


class MtpDrafterModel(nn.Module):
    """Google's MTP drafter reimplemented in PyTorch."""

    def __init__(self, cfg: MtpDrafterConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = MtpDrafterConfig()
        self.cfg = cfg

        # Input projection: concat(embed, projected_activations) → hidden
        self.mtp_pre_proj = nn.Linear(cfg.input_size, cfg.hidden_size, bias=False)

        # 4 transformer layers
        self.layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            if i < 3:  # SWA layers
                self.layers.append(DrafterLayer(
                    cfg.hidden_size, cfg.swa_num_heads, cfg.swa_head_dim,
                    cfg.ffn_dim, cfg.rms_eps
                ))
            else:  # Full attention layer
                self.layers.append(DrafterLayer(
                    cfg.hidden_size, cfg.full_num_heads, cfg.full_head_dim,
                    cfg.ffn_dim, cfg.rms_eps
                ))

        # Output heads
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.mtp_post_proj = nn.Linear(cfg.hidden_size, cfg.target_hidden, bias=False)

        # Logit softcapping (same as Gemma 4 target: tanh(x/cap) * cap)
        self.softcap_factor = 30.0

        # Precompute RoPE tables
        swa_cos, swa_sin = _precompute_rope(cfg.swa_head_dim, 33000, cfg.rope_theta)
        full_cos, full_sin = _precompute_rope(cfg.full_head_dim, 33000, cfg.rope_theta)
        self.register_buffer("swa_cos", swa_cos, persistent=False)
        self.register_buffer("swa_sin", swa_sin, persistent=False)
        self.register_buffer("full_cos", full_cos, persistent=False)
        self.register_buffer("full_sin", full_sin, persistent=False)

    def forward(
        self,
        activations: torch.Tensor,      # (B, 1, 3072) = concat(embed, proj_act)
        input_pos: torch.Tensor,         # (B,) int position
        kv13_k: torch.Tensor,            # (B, 1, ctx, 256) target's sliding K
        kv13_v: torch.Tensor,            # (B, 1, 256, ctx) target's sliding V
        kv14_k: torch.Tensor,            # (B, 1, ctx, 512) target's full K
        kv14_v: torch.Tensor,            # (B, 1, 512, ctx) target's full V
        mask_swa: torch.Tensor,          # (B, 1, 1, ctx) sliding mask
        mask_full: torch.Tensor,         # (B, 1, 1, ctx) full mask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, 1, vocab_size)
            projected_activations: (B, 1, target_hidden)
        """
        # Input projection
        x = self.mtp_pre_proj(activations)  # (B, 1, hidden)

        # Get RoPE for this position
        pos = input_pos[0].item()
        swa_cos = self.swa_cos[pos:pos+1].unsqueeze(0)   # (1, 1, D//2)
        swa_sin = self.swa_sin[pos:pos+1].unsqueeze(0)
        full_cos = self.full_cos[pos:pos+1].unsqueeze(0)
        full_sin = self.full_sin[pos:pos+1].unsqueeze(0)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            if i < 3:  # SWA
                x = layer(x, kv13_k, kv13_v, swa_cos, swa_sin, mask_swa)
            else:  # Full
                x = layer(x, kv14_k, kv14_v, full_cos, full_sin, mask_full)

        # Output
        h = self.final_norm(x)
        logits = self.lm_head(h)
        # Logit softcapping: tanh(logits / factor) * factor
        logits = torch.tanh(logits / self.softcap_factor) * self.softcap_factor
        proj_act = self.mtp_post_proj(h)

        return logits, proj_act


# ---------------------------------------------------------------------------
# TFLite weight extraction
# ---------------------------------------------------------------------------

def _extract_tflite_weights(tflite_path: str) -> dict[str, np.ndarray]:
    """Extract all weight tensors from TFLite file, dequantizing as needed."""
    import tflite

    with open(tflite_path, "rb") as f:
        buf = f.read()

    model = tflite.Model.GetRootAs(buf, 0)
    sg = model.Subgraphs(0)

    weights = {}
    for ti in range(sg.TensorsLength()):
        t = sg.Tensors(ti)
        name = t.Name().decode() if t.Name() else f"tensor_{ti}"
        buf_idx = t.Buffer()
        if buf_idx == 0 or buf_idx >= model.BuffersLength():
            continue
        b = model.Buffers(buf_idx)
        if b.DataLength() == 0:
            continue

        shape = tuple(t.Shape(d) for d in range(t.ShapeLength()))
        if len(shape) == 0:
            # Scalar
            shape = ()

        raw = b.DataAsNumpy()
        dtype_code = t.Type()

        # Dequantize based on type
        qparams = t.Quantization()

        if dtype_code == 0:  # FLOAT32
            arr = np.frombuffer(raw, dtype=np.float32).copy()
            if shape:
                arr = arr.reshape(shape)
        elif dtype_code == 9:  # INT8
            arr_int8 = np.frombuffer(raw, dtype=np.int8).copy()
            if qparams and qparams.ScaleLength() > 0:
                scales = np.array([qparams.Scale(i) for i in range(qparams.ScaleLength())])
                zeros = np.array([qparams.ZeroPoint(i) for i in range(qparams.ZeroPointLength())])
                if shape:
                    arr_int8 = arr_int8.reshape(shape)
                if len(scales) == 1:
                    arr = (arr_int8.astype(np.float32) - zeros[0]) * scales[0]
                else:
                    # Per-channel: scales along first dim
                    arr = (arr_int8.astype(np.float32) - zeros.reshape(-1, *([1]*(len(shape)-1)))) * \
                          scales.reshape(-1, *([1]*(len(shape)-1)))
            else:
                arr = arr_int8.astype(np.float32)
                if shape:
                    arr = arr.reshape(shape)
        elif dtype_code == 17:  # INT4
            # INT4: each byte packs two 4-bit values (low nibble first)
            raw_bytes = np.frombuffer(raw, dtype=np.uint8).copy()
            numel = 1
            for s in shape:
                numel *= s
            # Vectorized INT4 unpacking
            lo = (raw_bytes & 0x0F).astype(np.int8)
            hi = ((raw_bytes >> 4) & 0x0F).astype(np.int8)
            # Sign-extend: values 8-15 become -8 to -1
            lo[lo >= 8] -= 16
            hi[hi >= 8] -= 16
            # Interleave: [lo0, hi0, lo1, hi1, ...]
            arr_int4 = np.empty(len(raw_bytes) * 2, dtype=np.int8)
            arr_int4[0::2] = lo
            arr_int4[1::2] = hi
            arr_int4 = arr_int4[:numel]

            if qparams and qparams.ScaleLength() > 0:
                scales = np.array([qparams.Scale(i) for i in range(qparams.ScaleLength())])
                zeros = np.array([qparams.ZeroPoint(i) for i in range(qparams.ZeroPointLength())])
                if shape:
                    arr_int4 = arr_int4.reshape(shape)
                if len(scales) == 1:
                    arr = (arr_int4.astype(np.float32) - zeros[0]) * scales[0]
                else:
                    arr = (arr_int4.astype(np.float32) - zeros.reshape(-1, *([1]*(len(shape)-1)))) * \
                          scales.reshape(-1, *([1]*(len(shape)-1)))
            else:
                arr = arr_int4.astype(np.float32)
                if shape:
                    arr = arr.reshape(shape)
        elif dtype_code == 2:  # INT32
            arr = np.frombuffer(raw, dtype=np.int32).copy()
            if shape:
                arr = arr.reshape(shape)
        elif dtype_code == 6:  # BOOL
            arr = np.frombuffer(raw, dtype=np.bool_).copy()
            if shape:
                arr = arr.reshape(shape)
        else:
            continue

        weights[name] = arr

    return weights


def _build_rename_map() -> dict[str, str]:
    """Build mapping from TFLite tensor names to PyTorch state_dict keys.

    TFLite naming pattern (from extraction):
      mtp_pre_proj:   MtpDrafterModel.mtp_pre_project/mtp_pre_proj/btm,md->btd/dot_general
      layer_i q_proj: layer_{i}/layer_{i}.pre_q/.../q_einsum/.../dot_general
      layer_i o_proj: layer_{i}/layer_{i}.post_qkv/.../attn_vec_einsum/.../dot_general
      layer_i gate1:  layer_{i}/layer_{i}.post_qkv/mlp/gating_einsum1/.../dot_general
      layer_i gate2:  layer_{i}/layer_{i}.post_qkv/mlp/gating_einsum2/.../dot_general
      layer_i down:   layer_{i}/layer_{i}.post_qkv/mlp/linear/.../dot_general
      lm_head:        MtpDrafterModel.decode_softmax/.../embedder.decode/composite
      mtp_post_proj:  MtpDrafterModel.mtp_post_project/.../dot_general
    """
    rename = {}

    # Projection layers (match by substring)
    rename["mtp_pre_project/mtp_pre_proj"] = "mtp_pre_proj.weight"
    rename["mtp_post_project/mtp_post_proj"] = "mtp_post_proj.weight"
    rename["embedder.decode"] = "lm_head.weight"

    for i in range(4):
        rename[f"layer_{i}/layer_{i}.pre_q"] = f"layers.{i}.attn.q_proj.weight"
        rename[f"layer_{i}/layer_{i}.post_qkv/attn.post_qkv/attn_vec_einsum"] = f"layers.{i}.attn.o_proj.weight"
        rename[f"layer_{i}/layer_{i}.post_qkv/mlp/gating_einsum1"] = f"layers.{i}.mlp.gate1.weight"
        rename[f"layer_{i}/layer_{i}.post_qkv/mlp/gating_einsum2"] = f"layers.{i}.mlp.gate2.weight"
        rename[f"layer_{i}/layer_{i}.post_qkv/mlp/linear"] = f"layers.{i}.mlp.down.weight"

    return rename


def load_from_tflite_auto(model: MtpDrafterModel, tflite_path: str) -> list[str]:
    """Auto-match TFLite weights to PyTorch model by shape + graph position."""
    print(f"Extracting weights from {tflite_path}...")
    tfl_weights = _extract_tflite_weights(tflite_path)
    print(f"  Found {len(tfl_weights)} tensors total")

    sd = model.state_dict()
    matched_pt = set()
    matched_tfl = set()

    # 1. Match named linear weights by substring patterns
    linear_patterns = [
        ("mtp_pre_project/mtp_pre_proj", "mtp_pre_proj.weight"),
        ("mtp_post_project/mtp_post_proj", "mtp_post_proj.weight"),
        ("embedder.decode", "lm_head.weight"),
    ]
    for i in range(4):
        linear_patterns.extend([
            (f"layer_{i}/layer_{i}.pre_q", f"layers.{i}.attn.q_proj.weight"),
            (f"layer_{i}/layer_{i}.post_qkv/attn.post_qkv/attn_vec_einsum",
             f"layers.{i}.attn.o_proj.weight"),
            (f"layer_{i}/layer_{i}.post_qkv/mlp/gating_einsum1",
             f"layers.{i}.mlp.gate1.weight"),
            (f"layer_{i}/layer_{i}.post_qkv/mlp/gating_einsum2",
             f"layers.{i}.mlp.gate2.weight"),
            (f"layer_{i}/layer_{i}.post_qkv/mlp/linear",
             f"layers.{i}.mlp.down.weight"),
        ])

    for tfl_name, arr in tfl_weights.items():
        for pattern, pt_key in linear_patterns:
            if pattern in tfl_name and pt_key in sd and pt_key not in matched_pt:
                t = torch.from_numpy(arr).float()
                if t.shape == sd[pt_key].shape:
                    sd[pt_key] = t
                    matched_pt.add(pt_key)
                    matched_tfl.add(tfl_name)
                    print(f"  {pt_key:45s} ← {tuple(t.shape)}")
                else:
                    print(f"  SHAPE MISMATCH: {pt_key} expects {tuple(sd[pt_key].shape)}, "
                          f"got {tuple(t.shape)} from {tfl_name[:50]}")
                break

    # 2. Match norm weights — definitive mapping from TFLite graph analysis.
    # Verified by tracing operator inputs in parse_tflite_graph.py.
    norm_args = {}
    for tfl_name, arr in tfl_weights.items():
        if "jax2tf_arg_" in tfl_name and tfl_name not in matched_tfl:
            parts = tfl_name.split("jax2tf_arg_")[1].split("/")[0]
            try:
                arg_num = int(parts)
                norm_args[arg_num] = (tfl_name, arr)
            except ValueError:
                pass

    # Graph-verified norm mapping (from parse_tflite_graph.py output):
    # arg → graph output name → PyTorch key
    norm_direct = {
        # Layer 0
        12: "layers.0.input_layernorm.weight",           # pre_attention_norm
        10: "layers.0.post_attention_layernorm.weight",   # post_attention_norm
        13: "layers.0.pre_feedforward_layernorm.weight",  # pre_ffw_norm
        11: "layers.0.post_feedforward_layernorm.weight", # post_ffw_norm
        # Layer 1 (arg_23 is layer_1 input_layernorm, from cross-layer trace)
        23: "layers.1.input_layernorm.weight",
        21: "layers.1.post_attention_layernorm.weight",
        24: "layers.1.pre_feedforward_layernorm.weight",
        22: "layers.1.post_feedforward_layernorm.weight",
        # Layer 2 (arg_34 is layer_2 input_layernorm)
        34: "layers.2.input_layernorm.weight",
        32: "layers.2.post_attention_layernorm.weight",
        35: "layers.2.pre_feedforward_layernorm.weight",
        33: "layers.2.post_feedforward_layernorm.weight",
        # Layer 3 (arg_45 is layer_3 input_layernorm)
        45: "layers.3.input_layernorm.weight",
        39: "layers.3.attn.q_norm.weight",               # query_norm [512]
        43: "layers.3.post_attention_layernorm.weight",
        46: "layers.3.pre_feedforward_layernorm.weight",
        44: "layers.3.post_feedforward_layernorm.weight",
        # Final norm
        0:  "final_norm.weight",
    }

    # Layers 0-2 q_norm uses arith.constant3 (shared, all 0.9916)
    for tfl_name, arr in tfl_weights.items():
        if tfl_name == "arith.constant3" or (tfl_name.startswith("arith.constant3") and arr.shape == (256,)):
            for i in range(3):
                pt_key = f"layers.{i}.attn.q_norm.weight"
                if pt_key in sd and pt_key not in matched_pt:
                    sd[pt_key] = torch.from_numpy(arr).float()
                    matched_pt.add(pt_key)
                    print(f"  {pt_key:45s} ← arith.constant3 {tuple(arr.shape)}")
            break

    for arg_num, pt_key in sorted(norm_direct.items()):
        if arg_num in norm_args and pt_key in sd and pt_key not in matched_pt:
            tfl_name, arr = norm_args[arg_num]
            t = torch.from_numpy(arr).float()
            if t.shape == sd[pt_key].shape:
                sd[pt_key] = t
                matched_pt.add(pt_key)
                matched_tfl.add(tfl_name)
                print(f"  {pt_key:45s} ← arg_{arg_num} {tuple(t.shape)}")
            else:
                print(f"  NORM MISMATCH: {pt_key} expects {tuple(sd[pt_key].shape)}, "
                      f"got {tuple(t.shape)} from arg_{arg_num}")

    # Report
    unmatched = [k for k in sd
                 if k not in matched_pt
                 and not k.startswith("swa_") and not k.startswith("full_")]
    if unmatched:
        print(f"\n  UNMATCHED PyTorch keys ({len(unmatched)}):")
        for k in unmatched:
            print(f"    {k} {tuple(sd[k].shape)}")

    print(f"\n  Matched: {len(matched_pt)}/{len(sd)} PyTorch params")

    model.load_state_dict(sd, strict=False)
    return unmatched


# ---------------------------------------------------------------------------
# Main: extract + validate
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", type=str, default="output/mtp_probe/section_10.tflite")
    ap.add_argument("--output", type=str, default="output/mtp_probe/mtp_drafter.pt")
    args = ap.parse_args()

    cfg = MtpDrafterConfig()
    model = MtpDrafterModel(cfg).float().eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    unmatched = load_from_tflite_auto(model, args.tflite)

    if unmatched:
        print(f"\nWARNING: {len(unmatched)} unmatched keys — norm mapping may need adjustment")
        print("Attempting shape-based fallback for remaining norms...")
        # Fallback: try matching remaining [256] norms by proximity
        tfl_weights = _extract_tflite_weights(args.tflite)
        sd = model.state_dict()
        for key in unmatched:
            target_shape = tuple(sd[key].shape)
            for tfl_name, arr in tfl_weights.items():
                if "jax2tf_arg_" in tfl_name and tuple(arr.shape) == target_shape:
                    # Check if this tfl tensor is already used
                    t = torch.from_numpy(arr).float()
                    sd[key] = t
                    print(f"  FALLBACK: {key} ← {tfl_name} {tuple(t.shape)}")
                    break
        model.load_state_dict(sd, strict=False)

    # Save checkpoint
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\nSaved: {args.output}")
    print(f"Size: {Path(args.output).stat().st_size / 1e6:.1f} MB")

    # Quick forward test
    print("\nForward test...")
    with torch.no_grad():
        B, ctx = 1, 512
        act = torch.randn(B, 1, 3072)
        pos = torch.tensor([10], dtype=torch.int32)
        kv13_k = torch.randn(B, 1, ctx, 256)
        kv13_v = torch.randn(B, 1, 256, ctx)
        kv14_k = torch.randn(B, 1, ctx, 512)
        kv14_v = torch.randn(B, 1, 512, ctx)
        mask_swa = torch.zeros(B, 1, 1, ctx)
        mask_full = torch.zeros(B, 1, 1, ctx)

        logits, proj_act = model(act, pos, kv13_k, kv13_v, kv14_k, kv14_v, mask_swa, mask_full)
        print(f"  logits:    {tuple(logits.shape)} (argmax={logits.argmax(-1).item()})")
        print(f"  proj_act:  {tuple(proj_act.shape)} (norm={proj_act.norm().item():.4f})")
        print("  Forward OK")


if __name__ == "__main__":
    main()
