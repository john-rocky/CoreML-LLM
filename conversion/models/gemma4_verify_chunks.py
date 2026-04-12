"""Gemma 4 EAGLE-3 Verify chunks: batch-verify T candidate tokens per decode step.

Used by the speculative decoding loop to score K=T draft proposals in one ANE
call instead of T sequential T=1 decode steps. The KV cache is READ-ONLY here —
the Swift side re-runs the T=1 decode path (`commitAccepted`) for the accepted
prefix, so verify chunks never shift/write cache tensors.

Shape conventions (T=3 is the deploy target, but T is a constructor arg so the
same module can be reconverted for other K values):

  hidden_states            (1, T, hidden)                         fp16
  causal_mask_full         (1, 1, T, ctx + T)                     fp16   — Swift-built
  causal_mask_sliding      (1, 1, T, W + T)                       fp16   — Swift-built
  cos_s, sin_s             (1, 1, T, 256)                         fp16
  cos_f, sin_f             (1, 1, T, 512)                         fp16
  per_layer_raw            (1, T, num_layers * per_layer_dim)     fp16   — chunk1 only
  per_layer_combined       (1, T, num_layers * per_layer_dim)     fp16   — chunks 2-4
  K_sliding_in (chunks 1,2)                                       fp16
  V_sliding_in             —— same shape as SWA decode —— read-only
  K_full_in, V_full_in                                            fp16
  kv13_k, kv13_v (chunks 3,4) shape (1, 1, W + T, 256)            fp16   — Swift-built
  kv14_k, kv14_v             shape (1, 1, ctx + T, 512)           fp16   — Swift-built

Outputs:
  chunk1: hidden_states_out, per_layer_combined_out
  chunk2: hidden_states_out, kv13_k_out (1,1,W+T,256), kv13_v_out,
                              kv14_k_out (1,1,ctx+T,512), kv14_v_out
  chunk3: hidden_states_out
  chunk4: token_ids (T,) int32, token_logits (T,) fp16

kv13/kv14 in chunk2's output are the existing cache concatenated with the
newly computed K/V for L13 (sliding) and L14 (full). Swift supplies the cache
portion directly via the `kv13_k/...` inputs to chunks 3/4 (same as decode)
but with length W+T / ctx+T — chunks 3/4 don't know about the ±T shift, only
that the length is whatever it is. The shared-layer attention path treats
these as the full K/V tensor to attend against (no further concat needed).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, ane_softmax

from .gemma4 import Gemma4Model


def v_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean_sq = x.pow(2).mean(-1, keepdim=True) + eps
    return x * torch.rsqrt(mean_sq)


def _run_layer_verify(
    layer, layer_idx, hidden_states,  # (1, T, hidden)
    cos_s, sin_s, cos_f, sin_f,       # (1, 1, T, hd)
    causal_mask_full,                 # (1, 1, T, ctx + T)
    causal_mask_sliding,              # (1, 1, T, W  + T)
    K_sliding_slot, V_sliding_slot,   # (1, 1, W, max_hd) — own-cache, read-only
    K_full_slot, V_full_slot,         # (1, 1, ctx, max_hd)
    config, per_layer_combined,       # (1, T, total_pld)
    kv_store_13_k, kv_store_13_v,     # (1, 1, W + T, 256)   — shared by L15..34 sliding
    kv_store_14_k, kv_store_14_v,     # (1, 1, ctx + T, 512) — shared by L15..34 full
    T: int,
):
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    is_full = config.is_full_attention(layer_idx)
    hd = config.get_head_dim(layer_idx)
    is_kv_shared = config.is_kv_shared(layer_idx)

    residual = hidden_states
    h = layer.input_layernorm(hidden_states)
    # (1, T, hidden) → Conv2d layout (1, hidden, 1, T)
    x = h.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

    # Q projection: (1, hidden, 1, T) → (1, num_heads*hd, 1, T)
    q_raw = layer.self_attn["q_proj"](x)
    # → (1, num_heads, hd, T) → (1, num_heads, T, hd)
    q = q_raw.view(1, num_heads, hd, T).permute(0, 1, 3, 2).to(MODEL_DTYPE)
    # Per-token q_norm: reshape to (T, num_heads, hd)
    q = q.permute(0, 2, 1, 3).contiguous().view(T, num_heads, hd)
    q = layer.self_attn["q_norm"](q)
    q = q.view(1, T, num_heads, hd).permute(0, 2, 1, 3)  # (1, num_heads, T, hd)
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    # k_new_out / v_new_out: per-position K/V for the T new positions, for
    # Swift-side direct-write commit. Shape (1, num_kv_heads, T, hd). For
    # KV-shared layers these stay None (the shared kv13/kv14 update is
    # emitted as extended kv_store_13/14 already).
    k_new_out = None
    v_new_out = None
    if not is_kv_shared:
        k_raw = layer.self_attn["k_proj"](x)  # (1, num_kv_heads*hd, 1, T)
        v_raw = layer.self_attn["v_proj"](x)
        k_new = k_raw.view(1, num_kv_heads, hd, T).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v_new = v_raw.view(1, num_kv_heads, hd, T).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        # Per-token k_norm
        k_new = k_new.permute(0, 2, 1, 3).contiguous().view(T, num_kv_heads, hd)
        k_new = layer.self_attn["k_norm"](k_new)
        k_new = k_new.view(1, T, num_kv_heads, hd).permute(0, 2, 1, 3)
        v_new = v_norm(v_new)

        if is_full:
            _, k_new = apply_rotary_pos_emb(k_new, k_new, cos_f, sin_f)
        else:
            _, k_new = apply_rotary_pos_emb(k_new, k_new, cos_s, sin_s)

        k_new_out = k_new  # (1, num_kv_heads, T, hd) — post-RoPE/k_norm
        v_new_out = v_new

        if is_full:
            # K_full_slot is (1, 1, ctx, max_hd=512). Slice to hd then concat.
            K_cache = K_full_slot[..., :hd]  # (1, 1, ctx, hd=512)
            V_cache = V_full_slot[..., :hd]
            K_for_attn = torch.cat([K_cache, k_new], dim=2)  # (1, 1, ctx+T, hd)
            V_for_attn = torch.cat([V_cache, v_new], dim=2)
        else:
            # K_sliding_slot is (1, 1, W, max_hd=512). hd=256 for sliding.
            K_cache = K_sliding_slot[..., :hd]  # (1, 1, W, 256)
            V_cache = V_sliding_slot[..., :hd]
            K_for_attn = torch.cat([K_cache, k_new], dim=2)  # (1, 1, W+T, 256)
            V_for_attn = torch.cat([V_cache, v_new], dim=2)

        # Stash for shared layers downstream.
        if layer_idx == 13:
            kv_store_13_k = K_for_attn  # (1, 1, W+T, 256)
            kv_store_13_v = V_for_attn
        elif layer_idx == 14:
            kv_store_14_k = K_for_attn  # (1, 1, ctx+T, 512)
            kv_store_14_v = V_for_attn
    else:
        # Shared: read from extended kv13/kv14 stores.
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)  # (1, num_heads, *, hd)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

    mask = causal_mask_full if is_full else causal_mask_sliding
    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))  # (1, H, T, KV_len)
    attn_weights = attn_weights + mask
    attn_weights = ane_softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)  # (1, H, T, hd)

    # (1, H, T, hd) → (1, T, H*hd) → Conv2d (1, H*hd, 1, T) → o_proj → (1, hidden, 1, T)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, T, num_heads * hd)
    attn_output = attn_output.permute(0, 2, 1).unsqueeze(2)
    attn_output = layer.self_attn["o_proj"](attn_output)
    attn_output = attn_output.squeeze(2).permute(0, 2, 1)  # (1, T, hidden)
    attn_output = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + attn_output

    # MLP
    residual = hidden_states
    h = layer.pre_feedforward_layernorm(hidden_states)
    x_mlp = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (1, hidden, 1, T)
    gate = layer.mlp["gate_proj"](x_mlp)
    up = layer.mlp["up_proj"](x_mlp)
    gate = F.gelu(gate, approximate="tanh")
    mlp_out = layer.mlp["down_proj"](gate * up)  # (1, hidden, 1, T)
    hidden_states_mlp = mlp_out.squeeze(2).permute(0, 2, 1)  # (1, T, hidden)
    hidden_states_mlp = layer.post_feedforward_layernorm(hidden_states_mlp)
    hidden_states = residual + hidden_states_mlp

    # Per-layer input
    residual_pl = hidden_states
    s = layer_idx * config.hidden_size_per_layer_input
    e = s + config.hidden_size_per_layer_input
    per_layer_slice = per_layer_combined[:, :, s:e]  # (1, T, pld)
    hs_conv = hidden_states.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (1, hidden, 1, T)
    gated = layer.per_layer_input_gate(hs_conv)  # (1, pld, 1, T)
    gated = F.gelu(gated, approximate="tanh")
    per_layer_slice_conv = per_layer_slice.permute(0, 2, 1).unsqueeze(2)  # (1, pld, 1, T)
    gated = gated * per_layer_slice_conv
    gated = layer.per_layer_projection(gated)  # (1, hidden, 1, T)
    gated = gated.squeeze(2).permute(0, 2, 1)  # (1, T, hidden)
    hidden_states = layer.post_per_layer_input_norm(gated)
    hidden_states = residual_pl + hidden_states
    hidden_states = hidden_states * layer.layer_scalar.to(MODEL_DTYPE)

    return (hidden_states, kv_store_13_k, kv_store_13_v,
            kv_store_14_k, kv_store_14_v, k_new_out, v_new_out)


def _layer_kv_map(start: int, end: int, config):
    sliding_map = {}
    full_map = {}
    si = 0
    fi = 0
    for i in range(start, end):
        if config.is_full_attention(i):
            full_map[i] = fi
            fi += 1
        else:
            sliding_map[i] = si
            si += 1
    return sliding_map, full_map


class VerifyChunk1(nn.Module):
    """Layers 0-7 verify. Computes PLE like SWAChunk1 (PLE is position-independent,
    reused across all T tokens)."""
    START, END = 0, 8

    def __init__(self, model: Gemma4Model, T: int = 3):
        super().__init__()
        self.config = model.config
        self.T = T
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        self.sliding_map, self.full_map = _layer_kv_map(self.START, self.END, model.config)
        self.per_layer_model_projection = model.per_layer_model_projection
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_model_projection_scale = model.per_layer_model_projection_scale
        self.per_layer_input_scale = model.per_layer_input_scale
        self.per_layer_dim = model.config.hidden_size_per_layer_input
        self.num_layers_total = model.config.num_hidden_layers

    def _compute_ple(self, hidden_states, per_layer_raw):
        """Same ANE cat-trick norm as SWAChunk1, but over the T-dim."""
        T = self.T
        h_conv = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)  # (1, hidden, 1, T)
        proj = self.per_layer_model_projection(h_conv) * self.per_layer_model_projection_scale
        proj = proj.squeeze(2).permute(0, 2, 1)  # (1, T, total_pld)
        proj_grouped = proj.view(1, T, self.num_layers_total, self.per_layer_dim)
        # Apply cat-trick norm over last dim (per-layer-dim).
        norm_w = self.per_layer_projection_norm.weight
        eps = float(self.per_layer_projection_norm.eps)
        doubled = torch.cat([proj_grouped, -proj_grouped], dim=-1)
        normed = F.layer_norm(doubled, normalized_shape=(2 * self.per_layer_dim,),
                              weight=None, bias=None, eps=eps)
        normed, _ = torch.chunk(normed, 2, dim=-1)
        proj_normed = (normed * norm_w).view(1, T, self.num_layers_total * self.per_layer_dim)
        return (proj_normed + per_layer_raw) * self.per_layer_input_scale

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        T = self.T
        per_layer_combined = self._compute_ple(hidden_states, per_layer_raw)
        # Dummies for L13/L14 shared stores — never read in chunk1 (layers 0..7 all own KV).
        dummy_13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        dummy_13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        dummy_14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        dummy_14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        # Collect per-layer K_new/V_new for the T new positions, ordered by
        # the existing cache's slot layout (sliding slots 0..6, full slot 0).
        K_sliding_new_list = [None] * len(self.sliding_map)
        V_sliding_new_list = [None] * len(self.sliding_map)
        K_full_new_list = [None] * len(self.full_map)
        V_full_new_list = [None] * len(self.full_map)

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            is_full = config.is_full_attention(layer_idx)
            if is_full:
                fi = self.full_map[layer_idx]
                K_full_slot = K_full_in[fi].unsqueeze(0)
                V_full_slot = V_full_in[fi].unsqueeze(0)
                K_sliding_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
                V_sliding_slot = K_sliding_slot
            else:
                si = self.sliding_map[layer_idx]
                K_sliding_slot = K_sliding_in[si].unsqueeze(0)
                V_sliding_slot = V_sliding_in[si].unsqueeze(0)
                K_full_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
                V_full_slot = K_full_slot

            (hidden_states, _, _, _, _, k_new_out, v_new_out) = _run_layer_verify(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
                config, per_layer_combined,
                dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v,
                T,
            )
            # k_new_out/v_new_out shape (1, num_kv_heads=1, T, hd).
            # Pad sliding layers (hd=256) to max_hd=512 so Swift sees uniform shape.
            if is_full:
                K_full_new_list[self.full_map[layer_idx]] = k_new_out.squeeze(0)
                V_full_new_list[self.full_map[layer_idx]] = v_new_out.squeeze(0)
            else:
                pad_k = F.pad(k_new_out, (0, 512 - k_new_out.shape[-1])).squeeze(0)
                pad_v = F.pad(v_new_out, (0, 512 - v_new_out.shape[-1])).squeeze(0)
                K_sliding_new_list[self.sliding_map[layer_idx]] = pad_k
                V_sliding_new_list[self.sliding_map[layer_idx]] = pad_v

        K_sliding_new = torch.stack(K_sliding_new_list, dim=0)  # (7, 1, T, 512)
        V_sliding_new = torch.stack(V_sliding_new_list, dim=0)
        K_full_new = torch.stack(K_full_new_list, dim=0)        # (1, 1, T, 512)
        V_full_new = torch.stack(V_full_new_list, dim=0)
        return (hidden_states, per_layer_combined,
                K_sliding_new, V_sliding_new, K_full_new, V_full_new)


class VerifyChunk2(nn.Module):
    """Layers 8-14 verify. Builds extended kv13/kv14 stores (W+T / ctx+T)
    for downstream shared layers in chunks 3,4."""
    START, END = 8, 15

    def __init__(self, model: Gemma4Model, T: int = 3):
        super().__init__()
        self.config = model.config
        self.T = T
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        self.sliding_map, self.full_map = _layer_kv_map(self.START, self.END, model.config)

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        T = self.T
        # Placeholders; overwritten when we hit L13/L14.
        kv13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        K_sliding_new_list = [None] * len(self.sliding_map)  # 5 slots
        V_sliding_new_list = [None] * len(self.sliding_map)
        K_full_new_list = [None] * len(self.full_map)  # 2 slots
        V_full_new_list = [None] * len(self.full_map)
        hidden_at_L8 = hidden_states  # placeholder

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            is_full = config.is_full_attention(layer_idx)
            if is_full:
                fi = self.full_map[layer_idx]
                K_full_slot = K_full_in[fi].unsqueeze(0)
                V_full_slot = V_full_in[fi].unsqueeze(0)
                K_sliding_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
                V_sliding_slot = K_sliding_slot
            else:
                si = self.sliding_map[layer_idx]
                K_sliding_slot = K_sliding_in[si].unsqueeze(0)
                V_sliding_slot = V_sliding_in[si].unsqueeze(0)
                K_full_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
                V_full_slot = K_full_slot

            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v,
             k_new_out, v_new_out) = _run_layer_verify(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )
            # Capture hidden_at_L8 after L8 is processed (local_idx == 0).
            if layer_idx == 8:
                hidden_at_L8 = hidden_states

            if is_full:
                K_full_new_list[self.full_map[layer_idx]] = k_new_out.squeeze(0)
                V_full_new_list[self.full_map[layer_idx]] = v_new_out.squeeze(0)
            else:
                pad_k = F.pad(k_new_out, (0, 512 - k_new_out.shape[-1])).squeeze(0)
                pad_v = F.pad(v_new_out, (0, 512 - v_new_out.shape[-1])).squeeze(0)
                K_sliding_new_list[self.sliding_map[layer_idx]] = pad_k
                V_sliding_new_list[self.sliding_map[layer_idx]] = pad_v

        K_sliding_new = torch.stack(K_sliding_new_list, dim=0)  # (5, 1, T, 512)
        V_sliding_new = torch.stack(V_sliding_new_list, dim=0)
        K_full_new = torch.stack(K_full_new_list, dim=0)        # (2, 1, T, 512)
        V_full_new = torch.stack(V_full_new_list, dim=0)
        return (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v,
                K_sliding_new, V_sliding_new, K_full_new, V_full_new,
                hidden_at_L8)


class VerifyChunk3(nn.Module):
    """Layers 15-24 verify. All KV-shared via kv13 (W+T) and kv14 (ctx+T)."""
    START, END = 15, 25

    def __init__(self, model: Gemma4Model, T: int = 3):
        super().__init__()
        self.config = model.config
        self.T = T
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        T = self.T
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K
        hidden_at_L17 = hidden_states
        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            hidden_states, *_ = _run_layer_verify(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                dummy_K, dummy_V, dummy_K, dummy_V,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )
            if layer_idx == 17:
                hidden_at_L17 = hidden_states
        return hidden_states, hidden_at_L17


class VerifyChunk4(nn.Module):
    """Layers 25-34 + norm + lm_head. Emits per-position argmax (token_ids (T,))."""
    START, END = 25, 35

    def __init__(self, model: Gemma4Model, T: int = 3):
        super().__init__()
        self.config = model.config
        self.T = T
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        self.norm = model.norm
        self.lm_head = nn.Conv2d(model.lm_head.in_channels, model.lm_head.out_channels,
                                 kernel_size=1, bias=False)
        self.lm_head.weight.data = model.lm_head.weight.data.clone()
        self.softcap = model.softcap

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        T = self.T
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K
        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            hidden_states, *_ = _run_layer_verify(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                dummy_K, dummy_V, dummy_K, dummy_V,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )
        # pre-norm hidden after L34 — matches decode chunk4's hidden_at_L34
        # (1, T, hidden).
        hidden_at_L34 = hidden_states

        normed = self.norm(hidden_states)  # (1, T, hidden)
        # lm_head via Conv2d: (1, T, hidden) → (1, hidden, 1, T) → (1, vocab, 1, T) → (1, T, vocab)
        x = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)  # (1, T, vocab)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        # Per-position argmax: treat T as batch → (T, vocab) argmax → (T,) int32
        logits2d = logits.squeeze(0)  # (T, vocab)
        token_ids = torch.argmax(logits2d, dim=-1).to(torch.int32)  # (T,)
        token_logits = logits2d.gather(-1, token_ids.long().unsqueeze(-1)).squeeze(-1)
        return token_ids, token_logits, hidden_at_L34
