"""Gemma 4 Sliding Window Attention (SWA) chunks.

Key optimization: exploit Gemma 4's native sliding window attention
(W=512) to make 28/35 layers O(W) instead of O(ctx).

Architecture:
- 35 layers = 28 sliding (W=512) + 7 full (ctx)
- Layers 0-14: own KV caches
- Layers 15-34: all KV-shared (read from L13/L14)
- L13 is sliding → kv13 is W-sized
- L14 is full → kv14 is ctx-sized

KV tensor shapes:
- Sliding K/V cache: (num_sliding_in_chunk, 1, W, max_hd)
  - Shift-based update: cat([K[:, :, 1:], new_k], dim=2)
- Full K/V cache: (num_full_in_chunk, 1, ctx, max_hd)
  - Mask-based update (same as before)

Chunk layer breakdown:
  chunk1 (L0-7): 7 sliding (L0-3,5-7) + 1 full (L4)
  chunk2 (L8-14): 5 sliding (L8,10-13) + 2 full (L9,14)
  chunk3 (L15-24): all shared, reads kv13 (W-sized) and kv14 (ctx-sized)
  chunk4 (L25-34): all shared + norm + lm_head

Two causal masks:
- causal_mask_full: (1, 1, 1, ctx) — for full attention layers
- causal_mask_sliding: (1, 1, 1, W) — for sliding layers
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, ane_softmax

from .gemma4 import Gemma4Model


def v_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # RMSNorm without learnable scale (HF Gemma 4 v_norm has with_scale=False).
    # Uses the cat-trick (layer_norm over [x, -x]) instead of rsqrt so it maps
    # to ANE's native LayerNorm kernel. Math is identical: first half of
    # layer_norm([x, -x]) == x / sqrt(mean(x^2) + eps).
    hd = x.size(-1)
    doubled = torch.cat([x, -x], dim=-1)
    normed = F.layer_norm(
        doubled,
        normalized_shape=(2 * hd,),
        weight=None,
        bias=None,
        eps=float(eps),
    )
    normed, _ = torch.chunk(normed, 2, dim=-1)
    return normed


def _run_layer_swa(
    layer, layer_idx, hidden_states,
    cos_s, sin_s, cos_f, sin_f,
    causal_mask_full, causal_mask_sliding,
    update_mask,  # for full layers only: (1, 1, ctx, 1)
    K_sliding_slot, V_sliding_slot,  # (1, 1, W, max_hd) or None
    K_full_slot, V_full_slot,  # (1, 1, ctx, max_hd) or None
    config, per_layer_combined,
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
):
    """Run one layer. Returns hidden_states and updated K/V for the layer's cache type."""
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

    # Q
    q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
    q = layer.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    # K/V: compute if not shared
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
            # Full attention: mask-based update on (1, num_kv_heads, ctx, max_hd)
            K_full_out = K_full_slot * (1 - update_mask) + k_padded.expand_as(K_full_slot) * update_mask
            V_full_out = V_full_slot * (1 - update_mask) + v_padded.expand_as(V_full_slot) * update_mask
            K_for_attn = K_full_out[..., :hd]
            V_for_attn = V_full_out[..., :hd]
        else:
            # Sliding: shift-based on (1, num_kv_heads, W, max_hd)
            # cat([K[:, :, 1:, :], k_padded], dim=2) where k_padded is (1, nkv, 1, max_hd)
            K_sliding_out = torch.cat([K_sliding_slot[:, :, 1:, :], k_padded], dim=2)
            V_sliding_out = torch.cat([V_sliding_slot[:, :, 1:, :], v_padded], dim=2)
            K_for_attn = K_sliding_out[..., :hd]
            V_for_attn = V_sliding_out[..., :hd]

        # Store kv13/kv14 for sharing
        if layer_idx == 13:
            # L13 is sliding → kv13 is W-sized (1, 1, W, 256)
            kv_store_13_k = K_sliding_out[..., :256]
            kv_store_13_v = V_sliding_out[..., :256]
        elif layer_idx == 14:
            # L14 is full → kv14 is ctx-sized (1, 1, ctx, 512)
            kv_store_14_k = K_full_out[..., :512]
            kv_store_14_v = V_full_out[..., :512]
    else:
        # Shared: read from kv13 or kv14
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k  # W-sized now!
            V_for_attn = kv_store_13_v

    # GQA
    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

    # Manual attention with scale=1.0 (Gemma 4's effective scale after q_norm/k_norm).
    # SDPA fusion was attempted with d^(1/4) pre-scaling but CoreML's SDPA
    # decomposition produces slightly different results from manual attention,
    # causing wrong token predictions. Keeping manual attention for correctness.
    mask = causal_mask_full if is_full else causal_mask_sliding
    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + mask
    attn_weights = ane_softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
    attn_output = layer.self_attn["o_proj"](
        attn_output.permute(0, 2, 1).unsqueeze(2)
    ).squeeze(2).permute(0, 2, 1)
    attn_output = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + attn_output

    # MLP
    residual = hidden_states
    h = layer.pre_feedforward_layernorm(hidden_states)
    x_mlp = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
    gate = layer.mlp["gate_proj"](x_mlp)
    up = layer.mlp["up_proj"](x_mlp)
    gate = F.gelu(gate, approximate="tanh")
    mlp_out = layer.mlp["down_proj"](gate * up)
    hidden_states = mlp_out.squeeze(2).permute(0, 2, 1)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    # Per-layer input (Conv2d-based, NCHW internally)
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
    if getattr(layer, "use_layer_scalar", True):
        hidden_states = hidden_states * layer.layer_scalar.to(MODEL_DTYPE)

    return (hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out,
            kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v)


def _layer_kv_map(start: int, end: int, config):
    """Return (sliding_indices, full_indices) within [start, end) layers.
    sliding_indices[layer_idx] = local sliding slot index
    full_indices[layer_idx] = local full slot index
    """
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


class SWAChunk1(nn.Module):
    """Layers 0-7: 7 sliding (L0-3, L5-7) + 1 full (L4). Own KV cache.
    Computes PLE (per_layer_combined) internally from per_layer_raw input.
    """
    START, END = 0, 8

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        self.sliding_map, self.full_map = _layer_kv_map(self.START, self.END, model.config)
        self.num_sliding = len(self.sliding_map)  # 7
        self.num_full = len(self.full_map)  # 1
        # PLE computation modules (moved from Swift → ANE)
        self.per_layer_model_projection = model.per_layer_model_projection
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_model_projection_scale = model.per_layer_model_projection_scale
        self.per_layer_input_scale = model.per_layer_input_scale
        self.per_layer_dim = model.config.hidden_size_per_layer_input
        self.num_layers_total = model.config.num_hidden_layers

    def _compute_ple(self, hidden_states, per_layer_raw):
        """Compute per_layer_combined from hidden_states and raw per-layer embedding.

        hidden_states: (1, 1, hidden)
        per_layer_raw: (1, 1, num_layers * per_layer_dim) — already scaled by per_layer_embed_scale
        Returns: (1, 1, num_layers * per_layer_dim)

        The per-layer norm has identical weights across all 35 layer slices
        (it's a single ANERMSNorm reused), so instead of 35 separate norms +
        34 concats (~100 MIL ops), we reshape to (1, 35, 256) and apply ONE
        layer_norm over the last dim. ~70 ops eliminated per forward pass.
        """
        import torch.nn.functional as F
        # Conv2d layout: (1, 1, hidden) → (1, hidden, 1, 1)
        h_conv = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        proj = self.per_layer_model_projection(h_conv) * self.per_layer_model_projection_scale
        # (1, total_pld, 1, 1) → (1, 1, total_pld) → (1, num_layers, per_layer_dim)
        proj = proj.squeeze(2).permute(0, 2, 1)
        proj_grouped = proj.view(1, self.num_layers_total, self.per_layer_dim)

        # ANE cat-trick RMSNorm: layer_norm([x, -x]) then drop mirror.
        norm_w = self.per_layer_projection_norm.weight  # (per_layer_dim,)
        eps = float(self.per_layer_projection_norm.eps)
        doubled = torch.cat([proj_grouped, -proj_grouped], dim=-1)
        normed = F.layer_norm(doubled, normalized_shape=(2 * self.per_layer_dim,),
                              weight=None, bias=None, eps=eps)
        normed, _ = torch.chunk(normed, 2, dim=-1)
        proj_normed = (normed * norm_w).view(1, 1, self.num_layers_total * self.per_layer_dim)

        # Combine: (normed_proj + raw) * input_scale
        return (proj_normed + per_layer_raw) * self.per_layer_input_scale

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        # Compute PLE internally (8ms savings vs Swift BLAS)
        per_layer_combined = self._compute_ple(hidden_states, per_layer_raw)
        dummy_13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        dummy_13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        dummy_14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        dummy_14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        K_sliding_outs = []
        V_sliding_outs = []
        K_full_outs = []
        V_full_outs = []

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            is_full = config.is_full_attention(layer_idx)
            if is_full:
                fi = self.full_map[layer_idx]
                K_full_slot = K_full_in[fi].unsqueeze(0)
                V_full_slot = V_full_in[fi].unsqueeze(0)
                K_sliding_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)  # dummy
                V_sliding_slot = K_sliding_slot
            else:
                si = self.sliding_map[layer_idx]
                K_sliding_slot = K_sliding_in[si].unsqueeze(0)
                V_sliding_slot = V_sliding_in[si].unsqueeze(0)
                K_full_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)  # dummy
                V_full_slot = K_full_slot

            (hidden_states, Kso, Vso, Kfo, Vfo, *_) = _run_layer_swa(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
                config, per_layer_combined,
                dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v,
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
        # Return per_layer_combined as output → passed to chunks 2-4
        return hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out, per_layer_combined


class SWAChunk2(nn.Module):
    """Layers 8-14: 5 sliding (L8, L10-13) + 2 full (L9, L14). Own KV cache.
    L13 is sliding → outputs kv13 (W-sized sliding).
    L14 is full → outputs kv14 (ctx-sized full).
    """
    START, END = 8, 15

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        self.sliding_map, self.full_map = _layer_kv_map(self.START, self.END, model.config)
        self.num_sliding = len(self.sliding_map)  # 5
        self.num_full = len(self.full_map)  # 2

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        kv13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        K_sliding_outs = []
        V_sliding_outs = []
        K_full_outs = []
        V_full_outs = []

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
             kv13_k, kv13_v, kv14_k, kv14_v) = _run_layer_swa(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
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


class SWAChunk3(nn.Module):
    """Layers 15-24: all KV-shared. Reads kv13 (W-sized) and kv14 (ctx-sized)."""
    START, END = 15, 25

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            hidden_states, *_ = _run_layer_swa(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                dummy_K, dummy_V, dummy_K, dummy_V,  # unused (shared)
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
        return hidden_states


class SWAChunk4(nn.Module):
    """Layers 25-34: all KV-shared. + norm + lm_head."""
    START, END = 25, 35

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        self.norm = model.norm
        self.lm_head = nn.Conv2d(model.lm_head.in_channels, model.lm_head.out_channels,
                                  kernel_size=1, bias=False)
        self.lm_head.weight.data = model.lm_head.weight.data.clone()
        self.argmax = model.argmax
        self.softcap = model.softcap

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            hidden_states, *_ = _run_layer_swa(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                dummy_K, dummy_V, dummy_K, dummy_V,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )

        normed = self.norm(hidden_states)
        x = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        # Softcap (tanh(logits/c)*c) dropped: it is monotonic in logits, so
        # argmax(softcap(logits)) == argmax(logits). For sampling paths the
        # Swift runtime should apply softcap to the single selected scalar.
        token_id, token_logit = self.argmax(logits.squeeze(0))
        # Output normed hidden state for Medusa speculative decoding heads.
        # Shape: (1, 1, hidden_size) — the last hidden state before lm_head.
        return token_id, token_logit, normed


# ============================================================
# Verify mode: Q=K batched speculative verification (read-only KV)
# ============================================================

def _run_layer_verify(
    layer, layer_idx, hidden_states, seq_len,
    cos_s, sin_s, cos_f, sin_f,
    causal_mask_full, causal_mask_sliding,
    update_indicator,  # (1, 1, ctx, K) for full-attn KV scatter; None for shared layers
    K_sliding_slot, V_sliding_slot,  # (num_slots, 1, W, max_hd) or None
    K_full_slot, V_full_slot,  # (num_slots, 1, ctx, max_hd) or None
    config, per_layer_combined,
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
    sliding_map, full_map,
):
    """Run one layer in verify mode (Q=seq_len) WITH KV write-through.

    For non-shared layers (L0-14): computes K/V projections and writes
    to the cache. Sliding layers shift by K; full layers scatter via
    update_indicator. For shared layers (L15-34): reads kv13/kv14 only.

    After verification, rejected tokens' KV entries remain in the cache
    but are masked out by the causal mask in future decode steps — matching
    Google's LiteRT approach (see docs/LITERT_RUNTIME_ANALYSIS.md §B1.3).

    Returns:
        hidden_states, K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
        kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v
    """
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    max_hd = config.global_head_dim
    hd = config.get_head_dim(layer_idx)
    is_full = config.is_full_attention(layer_idx)
    is_kv_shared = config.is_kv_shared(layer_idx)

    residual = hidden_states
    h = layer.input_layernorm(hidden_states)
    x = h.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)  # (1, hidden, 1, seq_len)

    # Q projection: (1, num_heads*hd, 1, seq_len) -> (1, num_heads, seq_len, hd)
    q = layer.self_attn["q_proj"](x)
    q = q.view(1, num_heads, hd, seq_len).permute(0, 1, 3, 2).to(MODEL_DTYPE)

    # Q norm: merge seq_len into batch dim for per-position normalization
    q = q.permute(0, 2, 1, 3).contiguous().view(seq_len, num_heads, hd)
    q = layer.self_attn["q_norm"](q)
    q = q.view(1, seq_len, num_heads, hd).permute(0, 2, 1, 3)

    # RoPE on Q
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    K_sliding_out = K_sliding_slot
    V_sliding_out = V_sliding_slot
    K_full_out = K_full_slot
    V_full_out = V_full_slot

    if not is_kv_shared:
        # Compute K/V for all K tokens
        k = layer.self_attn["k_proj"](x)
        k = k.view(1, num_kv_heads, hd, seq_len).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = layer.self_attn["v_proj"](x)
        v = v.view(1, num_kv_heads, hd, seq_len).permute(0, 1, 3, 2).to(MODEL_DTYPE)

        # K norm, V norm (per-token via batch dim merge)
        k = k.permute(0, 2, 1, 3).contiguous().view(seq_len, num_kv_heads, hd)
        k = layer.self_attn["k_norm"](k)
        k = k.view(1, seq_len, num_kv_heads, hd).permute(0, 2, 1, 3)
        v = v_norm(v)

        # RoPE on K
        if is_full:
            _, k = apply_rotary_pos_emb(k, k, cos_f, sin_f)
        else:
            _, k = apply_rotary_pos_emb(k, k, cos_s, sin_s)

        # Pad to max_hd if needed: (1, kv, K, hd) -> (1, kv, K, max_hd)
        if hd < max_hd:
            k_padded = F.pad(k, (0, max_hd - hd))
            v_padded = F.pad(v, (0, max_hd - hd))
        else:
            k_padded, v_padded = k, v

        if is_full:
            fi = full_map[layer_idx]
            # Scatter K entries into ctx positions via indicator matmul
            # update_indicator: (1, 1, ctx, K), k_padded: (1, kv, K, max_hd)
            k_scattered = torch.matmul(
                update_indicator.expand(1, num_kv_heads, -1, -1),
                k_padded)  # (1, kv, ctx, max_hd)
            v_scattered = torch.matmul(
                update_indicator.expand(1, num_kv_heads, -1, -1),
                v_padded)
            combined_mask = update_indicator.sum(dim=-1, keepdim=True)  # (1, 1, ctx, 1)
            slot_k = K_full_slot[fi:fi+1]
            slot_v = V_full_slot[fi:fi+1]
            new_k = slot_k * (1 - combined_mask) + k_scattered
            new_v = slot_v * (1 - combined_mask) + v_scattered
            K_full_out = torch.cat([K_full_slot[:fi], new_k, K_full_slot[fi+1:]], dim=0)
            V_full_out = torch.cat([V_full_slot[:fi], new_v, V_full_slot[fi+1:]], dim=0)
            K_for_attn = K_full_out[fi:fi+1][..., :hd]
            V_for_attn = V_full_out[fi:fi+1][..., :hd]
        else:
            si = sliding_map[layer_idx]
            # Shift by K and append K new entries
            slot_k = K_sliding_slot[si:si+1]
            slot_v = V_sliding_slot[si:si+1]
            new_k = torch.cat([slot_k[:, :, seq_len:, :], k_padded], dim=2)
            new_v = torch.cat([slot_v[:, :, seq_len:, :], v_padded], dim=2)
            K_sliding_out = torch.cat([K_sliding_slot[:si], new_k, K_sliding_slot[si+1:]], dim=0)
            V_sliding_out = torch.cat([V_sliding_slot[:si], new_v, V_sliding_slot[si+1:]], dim=0)
            K_for_attn = K_sliding_out[si:si+1][..., :hd]
            V_for_attn = V_sliding_out[si:si+1][..., :hd]

        # Store kv13/kv14 for sharing
        if layer_idx == 13:
            kv_store_13_k = K_sliding_out[si:si+1][..., :256]
            kv_store_13_v = V_sliding_out[si:si+1][..., :256]
        elif layer_idx == 14:
            kv_store_14_k = K_full_out[fi:fi+1][..., :512]
            kv_store_14_v = V_full_out[fi:fi+1][..., :512]
    else:
        # Shared: read from kv13 or kv14
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    # GQA expansion
    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

    # Attention: (1, heads, seq_len, hd) @ (1, heads, hd, cache_len)
    mask = causal_mask_full if is_full else causal_mask_sliding
    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + mask
    attn_weights = ane_softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)

    # Output projection: (1, heads, seq_len, hd) -> (1, seq_len, hidden)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, seq_len, -1)
    attn_output = layer.self_attn["o_proj"](
        attn_output.permute(0, 2, 1).unsqueeze(2)
    ).squeeze(2).permute(0, 2, 1)
    attn_output = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + attn_output

    # MLP (Conv2d-based, operates per-token — handles any seq_len naturally)
    residual = hidden_states
    h = layer.pre_feedforward_layernorm(hidden_states)
    x_mlp = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
    gate = layer.mlp["gate_proj"](x_mlp)
    up = layer.mlp["up_proj"](x_mlp)
    gate = F.gelu(gate, approximate="tanh")
    mlp_out = layer.mlp["down_proj"](gate * up)
    hidden_states = mlp_out.squeeze(2).permute(0, 2, 1)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    # Per-layer input (Conv2d-based, handles any seq_len)
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
    if getattr(layer, "use_layer_scalar", True):
        hidden_states = hidden_states * layer.layer_scalar.to(MODEL_DTYPE)

    return (hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out,
            kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v)


class SWAVerifyChunk1(nn.Module):
    """Verify version of chunk1 (L0-7): Q=K with KV write-through.

    Computes K/V projections and writes them to the cache (shift for sliding,
    scatter for full-attn). Rejected entries are masked by causal mask later.
    Outputs hidden_states, updated KV caches, and per_layer_combined.
    """
    START, END = 0, 8

    def __init__(self, model: Gemma4Model, seq_len: int):
        super().__init__()
        self.config = model.config
        self.seq_len = seq_len
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        self.sliding_map, self.full_map = _layer_kv_map(self.START, self.END, model.config)
        # PLE modules (same weights as decode chunk1)
        self.per_layer_model_projection = model.per_layer_model_projection
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_model_projection_scale = model.per_layer_model_projection_scale
        self.per_layer_input_scale = model.per_layer_input_scale
        self.per_layer_dim = model.config.hidden_size_per_layer_input
        self.num_layers_total = model.config.num_hidden_layers

    def _compute_ple(self, hidden_states, per_layer_raw):
        """PLE computation for K tokens.

        hidden_states: (1, K, hidden)
        per_layer_raw: (1, K, nlayers * pld)
        Returns: (1, K, nlayers * pld)
        """
        K = self.seq_len
        h_conv = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        proj = self.per_layer_model_projection(h_conv) * self.per_layer_model_projection_scale
        proj = proj.squeeze(2).permute(0, 2, 1)  # (1, K, total_pld)
        proj_grouped = proj.contiguous().view(K, self.num_layers_total, self.per_layer_dim)

        norm_w = self.per_layer_projection_norm.weight
        eps = float(self.per_layer_projection_norm.eps)
        doubled = torch.cat([proj_grouped, -proj_grouped], dim=-1)
        normed = F.layer_norm(doubled, normalized_shape=(2 * self.per_layer_dim,),
                              weight=None, bias=None, eps=eps)
        normed, _ = torch.chunk(normed, 2, dim=-1)
        proj_normed = (normed * norm_w).view(1, K, self.num_layers_total * self.per_layer_dim)

        return (proj_normed + per_layer_raw) * self.per_layer_input_scale

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                update_indicator, per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        K = self.seq_len
        per_layer_combined = self._compute_ple(hidden_states, per_layer_raw)

        kv13_k = kv13_v = kv14_k = kv14_v = None

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            (hidden_states, K_sliding_in, V_sliding_in, K_full_in, V_full_in,
             kv13_k, kv13_v, kv14_k, kv14_v) = _run_layer_verify(
                self.layers[local_idx], layer_idx, hidden_states, K,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                update_indicator,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                self.sliding_map, self.full_map,
            )

        return (hidden_states, K_sliding_in, V_sliding_in,
                K_full_in, V_full_in, per_layer_combined)


class SWAVerifyChunk2(nn.Module):
    """Verify version of chunk2 (L8-14): Q=K with KV write-through.

    Writes K/V for all K positions. Extracts updated kv13/kv14 for chunks 3-4.
    """
    START, END = 8, 15

    def __init__(self, model: Gemma4Model, seq_len: int):
        super().__init__()
        self.config = model.config
        self.seq_len = seq_len
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        self.sliding_map, self.full_map = _layer_kv_map(self.START, self.END, model.config)

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                update_indicator, per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        K = self.seq_len

        kv13_k = kv13_v = kv14_k = kv14_v = None

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            (hidden_states, K_sliding_in, V_sliding_in, K_full_in, V_full_in,
             kv13_k, kv13_v, kv14_k, kv14_v) = _run_layer_verify(
                self.layers[local_idx], layer_idx, hidden_states, K,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                update_indicator,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                self.sliding_map, self.full_map,
            )

        return (hidden_states, K_sliding_in, V_sliding_in,
                K_full_in, V_full_in,
                kv13_k, kv13_v, kv14_k, kv14_v)


class SWAVerifyChunk3(nn.Module):
    """Verify version of chunk3 (L15-24): Q=K, shared KV from kv13/kv14.

    All layers are KV-shared — no cache writes. Just reads kv13/kv14.
    """
    START, END = 15, 25

    def __init__(self, model: Gemma4Model, seq_len: int):
        super().__init__()
        self.config = model.config
        self.seq_len = seq_len
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        K = self.seq_len

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            (hidden_states, _, _, _, _,
             _, _, _, _) = _run_layer_verify(
                self.layers[local_idx], layer_idx, hidden_states, K,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                None,  # no update_indicator for shared layers
                None, None, None, None,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                {}, {},  # empty maps — shared layers don't index into slots
            )

        return hidden_states


class SWAVerifyChunk4(nn.Module):
    """Verify version of chunk4 (L25-34): Q=K, shared KV + norm + lm_head.

    Outputs per-position token IDs (1, K) and hidden_states for MTP carry state.
    """
    START, END = 25, 35

    def __init__(self, model: Gemma4Model, seq_len: int):
        super().__init__()
        self.config = model.config
        self.seq_len = seq_len
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
        K = self.seq_len

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            (hidden_states, _, _, _, _,
             _, _, _, _) = _run_layer_verify(
                self.layers[local_idx], layer_idx, hidden_states, K,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                None, None, None, None, None,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                {}, {},
            )

        # Final norm + LM head: operates per-token (1, K, hidden)
        normed = self.norm(hidden_states)
        x = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (1, hidden, 1, K)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)  # (1, K, vocab)
        # Softcap dropped (monotonic; argmax-invariant). See SWAChunk4.forward.
        # Per-position argmax
        token_ids = torch.argmax(logits, dim=-1).to(torch.int32)  # (1, K)
        # Return hidden_states for MTP drafter carry state
        return token_ids, hidden_states
