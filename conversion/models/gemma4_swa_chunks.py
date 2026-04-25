"""Gemma 4 Sliding Window Attention (SWA) chunks — E2B and E4B.

Key optimization: exploit Gemma 4's native sliding window attention
(W=512) to make most layers O(W) instead of O(ctx).

Architecture (both variants):
- own-KV region 0..kv_full_producer; shared region kv_full_producer+1..N-1
- Shared sliding layers read from `kv_sliding_producer`'s cache (W-sized)
- Shared full layers read from `kv_full_producer`'s cache (ctx-sized)
- E2B: N=35, producers L13/L14; E4B: N=42, producers L22/L23

The output tensor names kv13_k/kv13_v/kv14_k/kv14_v are opaque aliases
for the sliding/full producer KVs (preserved across variants so
Sources/CoreMLLLM/ChunkedEngine.swift needs no edit).

KV tensor shapes:
- Sliding K/V cache: (num_sliding_in_chunk, num_kv_heads, W, max_hd)
  - Shift-based update: cat([K[:, :, 1:], new_k], dim=2)
- Full K/V cache: (num_full_in_chunk, num_kv_heads, ctx, max_hd)
  - Mask-based update

Chunk layout is derived from config via `compute_chunk_boundaries(config)`.

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
    mean_sq = x.pow(2).mean(-1, keepdim=True) + eps
    return x * torch.rsqrt(mean_sq)


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

        # Store producer-layer KV for sharing. Output names stay kv13/kv14 (aliases)
        # to avoid churn in Sources/CoreMLLLM/ChunkedEngine.swift — they are opaque
        # feature labels, not literal layer-13/14 references. Actual producers come
        # from config (E2B: L13/L14, E4B: L22/L23).
        if layer_idx == config.kv_sliding_producer:
            kv_store_13_k = K_sliding_out[..., :config.head_dim]
            kv_store_13_v = V_sliding_out[..., :config.head_dim]
        elif layer_idx == config.kv_full_producer:
            kv_store_14_k = K_full_out[..., :config.global_head_dim]
            kv_store_14_v = V_full_out[..., :config.global_head_dim]
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


def compute_chunk_boundaries(config) -> list[tuple[int, int]]:
    """Derive 4 decode-chunk boundaries from the Gemma4 config.

    chunk2 must end at kv_full_producer+1 so it can emit the shared KV.
    chunk1 splits the own-KV region roughly in half; chunks 3 and 4 split
    the remaining shared region in half.

    For E2B (35 layers, producers L13/L14) returns [(0,8),(8,15),(15,25),(25,35)]
    — matches the pre-E4B hardcoded layout.
    For E4B (42 layers, producers L22/L23) returns [(0,12),(12,24),(24,33),(33,42)].
    """
    n = config.num_hidden_layers
    own_end = config.kv_full_producer + 1
    c1_end = (own_end + 1) // 2
    c3_end = own_end + (n - own_end) // 2
    return [(0, c1_end), (c1_end, own_end), (own_end, c3_end), (c3_end, n)]


class SWAChunk1(nn.Module):
    """First decode chunk. Own KV cache. Computes PLE (per_layer_combined) from per_layer_raw.
    Boundaries are config-driven; for E2B (L0-7) it is 7 sliding + 1 full, for E4B (L0-11)
    it is 10 sliding + 2 full.
    """

    def __init__(self, model: Gemma4Model, start: int = 0, end: int = 8):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        self.sliding_map, self.full_map = _layer_kv_map(start, end, model.config)
        self.num_sliding = len(self.sliding_map)
        self.num_full = len(self.full_map)
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
        dummy_13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        dummy_13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        dummy_14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        dummy_14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)

        K_sliding_outs = []
        V_sliding_outs = []
        K_full_outs = []
        V_full_outs = []

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
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
    """Second decode chunk. Own KV cache. Ends at kv_full_producer+1 so it emits
    the sliding (kv13_*) and full (kv14_*) producer caches for chunks 3-4.
    For E2B (L8-14): 5 sliding + 2 full. For E4B (L12-23): 10 sliding + 2 full.
    """

    def __init__(self, model: Gemma4Model, start: int = 8, end: int = 15):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        self.sliding_map, self.full_map = _layer_kv_map(start, end, model.config)
        self.num_sliding = len(self.sliding_map)
        self.num_full = len(self.full_map)

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        kv13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)

        K_sliding_outs = []
        V_sliding_outs = []
        K_full_outs = []
        V_full_outs = []

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
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
    """Third decode chunk. All layers are KV-shared; reads kv13 (W-sized) and kv14 (ctx-sized).
    For E2B: L15-24 (10 shared). For E4B: L24-32 (9 shared).
    """

    def __init__(self, model: Gemma4Model, start: int = 15, end: int = 25):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
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
    """Final decode chunk. All KV-shared + final norm + lm_head + argmax.
    For E2B: L25-34 (10 shared). For E4B: L33-41 (9 shared).
    """

    def __init__(self, model: Gemma4Model, start: int = 25, end: int = 35):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
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

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
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
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        # Output normed hidden state for Medusa speculative decoding heads.
        # Shape: (1, 1, hidden_size) — the last hidden state before lm_head.
        return token_id, token_logit, normed


class SWAChunk4_LMSplit(SWAChunk4):
    """Variant of SWAChunk4 with the lm_head Conv2d split into N parallel heads.

    Per ANEMLL's `qwen_model.py:1006-1124`: for vocab=128k Qwen3 they split the
    final projection into 16 Conv2d kernels because a single Conv2d(hidden,
    vocab, 1) is too large to tile efficiently in ANE SRAM. Our vocab=262144
    is 2× larger, so the same motivation applies more strongly.

    `n_splits` must divide `vocab_size`. Common choices: 2 (matches anemll's
    ENABLE_VACAB_SPLIT), 8, 16 (anemll's ENABLE_VACAB_SPLIT16).

    All other behavior — softcap, in-model argmax, output names — is identical
    to SWAChunk4 so the runtime contract does not change.
    """

    def __init__(self, model: Gemma4Model, start: int = 25, end: int = 35,
                 n_splits: int = 16):
        super().__init__(model, start, end)
        vocab = model.lm_head.out_channels
        hidden = model.lm_head.in_channels
        if vocab % n_splits != 0:
            raise ValueError(
                f"vocab_size {vocab} not divisible by n_splits {n_splits}")
        split = vocab // n_splits
        full_w = model.lm_head.weight.data  # (V, H, 1, 1)
        for i in range(n_splits):
            head = nn.Conv2d(hidden, split, 1, bias=False, dtype=MODEL_DTYPE)
            head.weight.data = full_w[i * split:(i + 1) * split].clone()
            setattr(self, f"lm_head{n_splits}_{i + 1}", head)
        self.n_splits = n_splits
        # Drop the unsplit head so it isn't traced.
        del self.lm_head

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
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
        parts = []
        for i in range(self.n_splits):
            head = getattr(self, f"lm_head{self.n_splits}_{i + 1}")
            parts.append(head(x).squeeze(2).permute(0, 2, 1))
        logits = torch.cat(parts, dim=2)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
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
    """Run one layer in verify mode (Q=seq_len) under the 11c protocol.

    KV cache buffers (K_sliding_slot/K_full_slot) are updated LOCALLY only —
    used to derive K_for_attn within this verify call, and to produce the
    extended kv13/kv14 outputs for shared layers in chunks 3/4. They are
    NOT returned from the chunk forward, so persistent storage is untouched.

    Per-position raw new K/V slices (`new_k_slice`, `new_v_slice`) are
    additionally returned so Swift can selectively commit only the accepted
    prefix into persistent storage after acceptance is decided.

    Returns:
        hidden_states, K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
        kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
        new_k_slice, new_v_slice
        - new_k_slice / new_v_slice: (1, num_kv_heads=1, seq_len, hd) for non-shared layers,
          None for shared layers (L15-34).
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
    new_k_slice = None  # (1, num_kv_heads=1, seq_len, hd) for non-shared layers
    new_v_slice = None

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

        # 11c: raw per-T slice output (at hd, not max_hd) — Swift commits to persistent cache
        new_k_slice = k  # (1, 1, seq_len, hd)
        new_v_slice = v

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
            # Shift by K and append K new entries (LOCAL only — not returned by chunk)
            slot_k = K_sliding_slot[si:si+1]
            slot_v = V_sliding_slot[si:si+1]
            new_k = torch.cat([slot_k[:, :, seq_len:, :], k_padded], dim=2)
            new_v = torch.cat([slot_v[:, :, seq_len:, :], v_padded], dim=2)
            K_sliding_out = torch.cat([K_sliding_slot[:si], new_k, K_sliding_slot[si+1:]], dim=0)
            V_sliding_out = torch.cat([V_sliding_slot[:si], new_v, V_sliding_slot[si+1:]], dim=0)
            K_for_attn = K_sliding_out[si:si+1][..., :hd]
            V_for_attn = V_sliding_out[si:si+1][..., :hd]

        # Store producer-layer KV for sharing (same alias convention as decode path).
        # Extended within-verify form consumed by chunks 3/4 in this verify call —
        # NOT written to persistent storage by Swift under the 11c protocol.
        if layer_idx == config.kv_sliding_producer:
            kv_store_13_k = K_sliding_out[si:si+1][..., :config.head_dim]
            kv_store_13_v = V_sliding_out[si:si+1][..., :config.head_dim]
        elif layer_idx == config.kv_full_producer:
            kv_store_14_k = K_full_out[fi:fi+1][..., :config.global_head_dim]
            kv_store_14_v = V_full_out[fi:fi+1][..., :config.global_head_dim]
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
    hidden_states = hidden_states * layer.layer_scalar.to(MODEL_DTYPE)

    return (hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out,
            kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
            new_k_slice, new_v_slice)


class SWAVerifyChunk1(nn.Module):
    """Verify version of chunk1 (L0-7) under the 11c protocol.

    Computes attention with full K/V context (input cache + new K/V at K positions)
    but does NOT return the updated cache. Instead returns per-T raw new K/V slices
    so Swift can selectively commit only accepted positions to persistent storage.
    """

    def __init__(self, model: Gemma4Model, seq_len: int, start: int = 0, end: int = 8):
        super().__init__()
        self.config = model.config
        self.seq_len = seq_len
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        self.sliding_map, self.full_map = _layer_kv_map(start, end, model.config)
        # Precompute (sliding_layer_idx -> stack_position) and same for full
        # so chunk output ordering matches the persistent cache slot ordering.
        self.sliding_layer_indices = sorted(self.sliding_map.keys(),
                                            key=lambda k: self.sliding_map[k])
        self.full_layer_indices = sorted(self.full_map.keys(),
                                         key=lambda k: self.full_map[k])
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
        sliding_slices_k = [None] * len(self.sliding_layer_indices)
        sliding_slices_v = [None] * len(self.sliding_layer_indices)
        full_slices_k = [None] * len(self.full_layer_indices)
        full_slices_v = [None] * len(self.full_layer_indices)

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, K_sliding_in, V_sliding_in, K_full_in, V_full_in,
             kv13_k, kv13_v, kv14_k, kv14_v,
             new_k_slice, new_v_slice) = _run_layer_verify(
                self.layers[local_idx], layer_idx, hidden_states, K,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                update_indicator,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                self.sliding_map, self.full_map,
            )
            if layer_idx in self.sliding_map:
                sliding_slices_k[self.sliding_map[layer_idx]] = new_k_slice
                sliding_slices_v[self.sliding_map[layer_idx]] = new_v_slice
            elif layer_idx in self.full_map:
                full_slices_k[self.full_map[layer_idx]] = new_k_slice
                full_slices_v[self.full_map[layer_idx]] = new_v_slice

        # Stack slices in slot order: (num_slots, 1, K, hd)
        new_K_sliding = torch.cat(sliding_slices_k, dim=0)
        new_V_sliding = torch.cat(sliding_slices_v, dim=0)
        new_K_full = torch.cat(full_slices_k, dim=0)
        new_V_full = torch.cat(full_slices_v, dim=0)

        return (hidden_states, per_layer_combined,
                new_K_sliding, new_V_sliding, new_K_full, new_V_full)


class SWAVerifyChunk2(nn.Module):
    """Verify version of chunk2 (L8-14) under the 11c protocol.

    Same write-after-accept pattern as chunk1: returns per-T raw K/V slices
    for selective Swift commit. Also emits the extended within-verify kv13/kv14
    for chunks 3/4 to consume during this verify call (these are NOT persisted).
    """

    def __init__(self, model: Gemma4Model, seq_len: int, start: int = 8, end: int = 15):
        super().__init__()
        self.config = model.config
        self.seq_len = seq_len
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        self.sliding_map, self.full_map = _layer_kv_map(start, end, model.config)
        self.sliding_layer_indices = sorted(self.sliding_map.keys(),
                                            key=lambda k: self.sliding_map[k])
        self.full_layer_indices = sorted(self.full_map.keys(),
                                         key=lambda k: self.full_map[k])

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                update_indicator, per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        K = self.seq_len

        kv13_k = kv13_v = kv14_k = kv14_v = None
        sliding_slices_k = [None] * len(self.sliding_layer_indices)
        sliding_slices_v = [None] * len(self.sliding_layer_indices)
        full_slices_k = [None] * len(self.full_layer_indices)
        full_slices_v = [None] * len(self.full_layer_indices)

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, K_sliding_in, V_sliding_in, K_full_in, V_full_in,
             kv13_k, kv13_v, kv14_k, kv14_v,
             new_k_slice, new_v_slice) = _run_layer_verify(
                self.layers[local_idx], layer_idx, hidden_states, K,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                update_indicator,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                self.sliding_map, self.full_map,
            )
            if layer_idx in self.sliding_map:
                sliding_slices_k[self.sliding_map[layer_idx]] = new_k_slice
                sliding_slices_v[self.sliding_map[layer_idx]] = new_v_slice
            elif layer_idx in self.full_map:
                full_slices_k[self.full_map[layer_idx]] = new_k_slice
                full_slices_v[self.full_map[layer_idx]] = new_v_slice

        new_K_sliding = torch.cat(sliding_slices_k, dim=0)
        new_V_sliding = torch.cat(sliding_slices_v, dim=0)
        new_K_full = torch.cat(full_slices_k, dim=0)
        new_V_full = torch.cat(full_slices_v, dim=0)

        return (hidden_states,
                new_K_sliding, new_V_sliding, new_K_full, new_V_full,
                kv13_k, kv13_v, kv14_k, kv14_v)


class SWAVerifyChunk3(nn.Module):
    """Verify version of chunk3: Q=K, shared KV from kv13/kv14.

    All layers are KV-shared — no cache writes. Reads kv13/kv14 only.
    """

    def __init__(self, model: Gemma4Model, seq_len: int, start: int = 15, end: int = 25):
        super().__init__()
        self.config = model.config
        self.seq_len = seq_len
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        K = self.seq_len

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, _, _, _, _,
             _, _, _, _, _, _) = _run_layer_verify(
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
    """Verify version of chunk4: Q=K, shared KV + final norm + lm_head.

    Outputs per-position token IDs (1, K) and hidden_states for MTP carry state.
    """

    def __init__(self, model: Gemma4Model, seq_len: int, start: int = 25, end: int = 35):
        super().__init__()
        self.config = model.config
        self.seq_len = seq_len
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
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

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, _, _, _, _,
             _, _, _, _, _, _) = _run_layer_verify(
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
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        # Per-position argmax
        token_ids = torch.argmax(logits, dim=-1).to(torch.int32)  # (1, K)
        # Return hidden_states for MTP drafter carry state
        return token_ids, hidden_states
