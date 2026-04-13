"""Gemma 4 SWA chunks with Flash Decoding for full-attention layers.

Full-attention layers split Q@K^T into K-dim chunks with online softmax.
Mathematically EXACT — produces identical output to standard attention
(within fp16 rounding: cosine > 0.999999).

Key insight: Instead of one huge matmul (1,8,1,8192)×(1,8,8192,512)^T that
spills ANE SRAM, do 8 smaller matmuls (1,8,1,1024)×(1,8,1024,512)^T that
each fit in SRAM. Online softmax combines results exactly.

Changes from gemma4_swa_chunks.py:
- _run_layer_swa → _run_layer_flash: flash decoding for full-attention
- Sliding layers unchanged (already fit in SRAM at W=512)
- No architecture change, no quality loss
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, _LOG2_E, apply_rotary_pos_emb, ane_softmax

from .gemma4 import Gemma4Model


# Attention chunk size for flash decoding. Each chunk's KV = chunk_size × head_dim × 2 bytes.
# 1024 × 512 × 2 = 1 MB per K or V chunk → fits easily in 32 MB SRAM.
ATTN_CHUNK_SIZE = 1024

# Pre-computed at build time. Set by build_flash.py before tracing.
_FULL_ATTN_CHUNKS = 8  # CTX / ATTN_CHUNK_SIZE = 8192 / 1024


def v_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean_sq = x.pow(2).mean(-1, keepdim=True) + eps
    return x * torch.rsqrt(mean_sq)


def _flash_one_chunk(Q, K_c, V_c, mask_c, m_prev, l_prev, o_prev):
    """Process one chunk of flash decoding. All ops are trace-friendly."""
    s_c = torch.matmul(Q, K_c.transpose(-1, -2)).to(MODEL_DTYPE)
    s_c = s_c + mask_c

    m_c = s_c.max(dim=-1, keepdim=True).values.to(MODEL_DTYPE)
    m_new = torch.maximum(m_prev, m_c)

    exp_prev = torch.exp2(((m_prev - m_new) * _LOG2_E).to(MODEL_DTYPE)).to(MODEL_DTYPE)
    exp_c = torch.exp2(((s_c - m_new) * _LOG2_E).to(MODEL_DTYPE)).to(MODEL_DTYPE)

    l_c = exp_c.sum(dim=-1, keepdim=True).to(MODEL_DTYPE)
    l_new = (l_prev * exp_prev + l_c).to(MODEL_DTYPE)

    # Safe division: add epsilon to avoid div-by-zero
    inv_l = (1.0 / (l_new + 1e-7)).to(MODEL_DTYPE)
    scale_prev = (l_prev * exp_prev * inv_l).to(MODEL_DTYPE)

    o_new = (o_prev * scale_prev + torch.matmul(exp_c, V_c) * inv_l).to(MODEL_DTYPE)
    return m_new, l_new, o_new


def flash_decode_attention(Q_grouped, K_b, V_b, mask, num_chunks, num_kv_heads, n_rep, head_dim):
    """Flash Decoding with broadcast GQA.

    Uses 5D tensors so matmul broadcasts over the GQA rep dim instead of
    materializing the full num_heads expansion.

    Q_grouped: (1, num_kv, n_rep, 1, head_dim)
    K_b:       (1, num_kv, 1, S, head_dim)  — broadcast on rep dim
    V_b:       (1, num_kv, 1, S, head_dim)
    mask:      (1, 1, 1, S)  — auto-broadcasts to 5D

    num_chunks, num_kv_heads, n_rep, head_dim must be Python integer constants.
    """
    K_chunks = torch.chunk(K_b, num_chunks, dim=3)   # split S along dim=3
    V_chunks = torch.chunk(V_b, num_chunks, dim=3)
    mask_chunks = torch.chunk(mask, num_chunks, dim=-1)

    m = torch.full((1, num_kv_heads, n_rep, 1, 1), -65504.0, dtype=MODEL_DTYPE)
    l = torch.zeros(1, num_kv_heads, n_rep, 1, 1, dtype=MODEL_DTYPE)
    o = torch.zeros(1, num_kv_heads, n_rep, 1, head_dim, dtype=MODEL_DTYPE)

    for K_c, V_c, mask_c in zip(K_chunks, V_chunks, mask_chunks):
        m, l, o = _flash_one_chunk(Q_grouped, K_c, V_c, mask_c, m, l, o)

    return o.view(1, num_kv_heads * n_rep, 1, head_dim)


def _run_layer_flash(
    layer, layer_idx, hidden_states,
    cos_s, sin_s, cos_f, sin_f,
    causal_mask_full, causal_mask_sliding,
    update_mask,
    K_sliding_slot, V_sliding_slot,
    K_full_slot, V_full_slot,
    config, per_layer_combined,
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
    full_attn_chunks,
):
    """Run one layer. Full-attention uses flash decoding; sliding unchanged."""
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    max_hd = config.global_head_dim
    is_full = config.is_full_attention(layer_idx)
    hd = config.get_head_dim(layer_idx)
    is_kv_shared = config.is_kv_shared(layer_idx)

    residual = hidden_states
    h = layer.input_layernorm(hidden_states)
    x = h.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

    q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
    q = layer.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    K_sliding_out = K_sliding_slot
    V_sliding_out = V_sliding_slot
    K_full_out = K_full_slot
    V_full_out = V_full_slot

    if not is_kv_shared:
        k = layer.self_attn["k_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = layer.self_attn["v_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        k = layer.self_attn["k_norm"](k.reshape(1, num_kv_heads, hd)).view(1, num_kv_heads, 1, hd)
        v = v_norm(v)
        if is_full:
            _, k = apply_rotary_pos_emb(k, k, cos_f, sin_f)
        else:
            _, k = apply_rotary_pos_emb(k, k, cos_s, sin_s)

        if hd < max_hd:
            k_padded = F.pad(k, (0, max_hd - hd))
            v_padded = F.pad(v, (0, max_hd - hd))
        else:
            k_padded, v_padded = k, v

        if is_full:
            K_full_out = K_full_slot * (1 - update_mask) + k_padded.expand_as(K_full_slot) * update_mask
            V_full_out = V_full_slot * (1 - update_mask) + v_padded.expand_as(V_full_slot) * update_mask
            K_for_attn = K_full_out[..., :hd]
            V_for_attn = V_full_out[..., :hd]
        else:
            K_sliding_out = torch.cat([K_sliding_slot[:, :, 1:, :], k_padded], dim=2)
            V_sliding_out = torch.cat([V_sliding_slot[:, :, 1:, :], v_padded], dim=2)
            K_for_attn = K_sliding_out[..., :hd]
            V_for_attn = V_sliding_out[..., :hd]

        if layer_idx == 13:
            kv_store_13_k = K_sliding_out[..., :256]
            kv_store_13_v = V_sliding_out[..., :256]
        elif layer_idx == 14:
            kv_store_14_k = K_full_out[..., :512]
            kv_store_14_v = V_full_out[..., :512]
    else:
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    # GQA via broadcast matmul — avoids materializing the n_rep expansion.
    q_grouped = q.view(1, num_kv_heads, n_rep, 1, hd)
    K_b = K_for_attn.unsqueeze(2)   # (1, kv, 1, S, hd)
    V_b = V_for_attn.unsqueeze(2)   # (1, kv, 1, S, hd)

    # ATTENTION: use flash decoding for full-attention, standard for sliding
    mask = causal_mask_full if is_full else causal_mask_sliding
    if is_full and full_attn_chunks > 1:
        # Flash Decoding with broadcast GQA
        attn_output = flash_decode_attention(q_grouped, K_b, V_b, mask,
                                             full_attn_chunks, num_kv_heads, n_rep, hd)
    else:
        # Standard attention with broadcast GQA
        attn_weights = torch.matmul(q_grouped, K_b.transpose(-1, -2))  # (1,kv,rep,1,S)
        attn_weights = attn_weights + mask
        attn_weights = ane_softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V_b)  # (1, kv, rep, 1, hd)
        attn_output = attn_output.view(1, num_heads, 1, hd)

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
    attn_output = layer.self_attn["o_proj"](
        attn_output.permute(0, 2, 1).unsqueeze(2)
    ).squeeze(2).permute(0, 2, 1)
    attn_output = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + attn_output

    # MLP — tile (B,C,1,1) → (B,C,8,8) for ANE PE utilization
    residual = hidden_states
    h = layer.pre_feedforward_layernorm(hidden_states)
    x_mlp = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (1, C, 1, 1)
    x_mlp = x_mlp.expand(1, -1, 8, 8)  # (1, C, 8, 8) — fills ANE spatial PEs
    gate = layer.mlp["gate_proj"](x_mlp)
    up = layer.mlp["up_proj"](x_mlp)
    gate = F.gelu(gate, approximate="tanh")
    mlp_out = layer.mlp["down_proj"](gate * up)  # (1, C, 8, 8)
    mlp_out = mlp_out[:, :, :1, :1]  # back to (1, C, 1, 1)
    hidden_states = mlp_out.squeeze(2).permute(0, 2, 1)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    # Per-layer input (unchanged)
    residual_pl = hidden_states
    s = layer_idx * config.hidden_size_per_layer_input
    e = s + config.hidden_size_per_layer_input
    per_layer_slice = per_layer_combined[:, :, s:e]
    hs_conv = hidden_states.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
    gated = layer.per_layer_input_gate(hs_conv)
    gated = F.gelu(gated, approximate="tanh")
    per_layer_slice_conv = per_layer_slice.permute(0, 2, 1).unsqueeze(2)
    gated = gated * per_layer_slice_conv
    gated = layer.per_layer_projection(gated)
    gated = gated.squeeze(2).permute(0, 2, 1)
    hidden_states = layer.post_per_layer_input_norm(gated)
    hidden_states = residual_pl + hidden_states
    hidden_states = hidden_states * layer.layer_scalar.to(MODEL_DTYPE)

    return (hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out,
            kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v)


# Import the chunk helper and chunk classes from swa_chunks, then override _run_layer_swa
from .gemma4_swa_chunks import _layer_kv_map, SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4


class FlashChunk1(SWAChunk1):
    """Chunk1 with flash decoding for full-attention layers."""

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        per_layer_combined = self._compute_ple(hidden_states, per_layer_raw)
        dummy_13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        dummy_13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        dummy_14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        dummy_14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        full_chunks = _FULL_ATTN_CHUNKS

        K_sliding_outs = []; V_sliding_outs = []
        K_full_outs = []; V_full_outs = []

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

            (hidden_states, Kso, Vso, Kfo, Vfo, *_) = _run_layer_flash(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
                config, per_layer_combined,
                dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v,
                full_chunks,
            )
            if is_full:
                K_full_outs.append(Kfo.squeeze(0))
                V_full_outs.append(Vfo.squeeze(0))
            else:
                K_sliding_outs.append(Kso.squeeze(0))
                V_sliding_outs.append(Vso.squeeze(0))

        K_sliding_out = torch.stack(K_sliding_outs, dim=0)
        V_sliding_out = torch.stack(V_sliding_outs, dim=0)
        K_full_out = torch.stack(K_full_outs, dim=0)
        V_full_out = torch.stack(V_full_outs, dim=0)
        return hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out, per_layer_combined


class FlashChunk2(SWAChunk2):
    """Chunk2 with flash decoding for full-attention layers."""

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        full_chunks = _FULL_ATTN_CHUNKS
        kv13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        K_sliding_outs = []; V_sliding_outs = []
        K_full_outs = []; V_full_outs = []

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

            (hidden_states, Kso, Vso, Kfo, Vfo,
             kv13_k, kv13_v, kv14_k, kv14_v) = _run_layer_flash(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                full_chunks,
            )
            if is_full:
                K_full_outs.append(Kfo.squeeze(0))
                V_full_outs.append(Vfo.squeeze(0))
            else:
                K_sliding_outs.append(Kso.squeeze(0))
                V_sliding_outs.append(Vso.squeeze(0))

        K_sliding_out = torch.stack(K_sliding_outs, dim=0)
        V_sliding_out = torch.stack(V_sliding_outs, dim=0)
        K_full_out = torch.stack(K_full_outs, dim=0)
        V_full_out = torch.stack(V_full_outs, dim=0)
        return (hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out,
                kv13_k, kv13_v, kv14_k, kv14_v)


class FlashChunk3(SWAChunk3):
    """Chunk3 (KV-shared) with flash decoding for shared full-attention."""

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        full_chunks = _FULL_ATTN_CHUNKS
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            hidden_states, *_ = _run_layer_flash(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                dummy_K, dummy_V, dummy_K, dummy_V,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                full_chunks,
            )
        return hidden_states


class FlashChunk4(SWAChunk4):
    """Chunk4 (KV-shared + LM head) with flash decoding."""

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        full_chunks = _FULL_ATTN_CHUNKS
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            hidden_states, *_ = _run_layer_flash(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                dummy_K, dummy_V, dummy_K, dummy_V,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                full_chunks,
            )

        normed = self.norm(hidden_states)
        x = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        return token_id, token_logit, normed
