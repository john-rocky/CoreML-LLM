"""Gemma 4 STATELESS chunks for ANE — explicit KV I/O, no MLState.

Research findings (ANE constraints on iPhone):
- MLState + ANE is fragile / unsupported in many cases.
- 15+ layers per chunk causes ANE compile planning to hang.
- ANEMLL (which successfully runs on iPhone ANE) uses explicit KV I/O.

Strategy:
- 4 chunks with ~8-9 layers each (smaller graphs for ANE compile).
- Each chunk takes KV cache as input, returns updated KV cache.
- chunk2 contains layers 13, 14 (KV sources) → outputs kv13/kv14.
- chunk3 and chunk4 take kv13/kv14 as input (all their layers are shared).

Chunking:
  chunk1: layers 0-7 (8 layers, own KV cache, no sharing)
  chunk2: layers 8-14 (7 layers, contains KV sources 13/14, no sharing for these 7)
  chunk3: layers 15-24 (10 layers, ALL shared from kv13/kv14)
  chunk4: layers 25-34 (10 layers, ALL shared from kv13/kv14) + norm + lm_head
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, ane_softmax, ane_fused_softmax, repeat_kv_ane

from .gemma4 import Gemma4Model


def v_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm without learnable scale, fp16 throughout (ANE-friendly)."""
    mean_sq = x.pow(2).mean(-1, keepdim=True) + eps
    return x * torch.rsqrt(mean_sq)


def _run_layer_stateless(
    layer, layer_idx, hidden_states,
    cos_s, sin_s, cos_f, sin_f,  # pre-computed, passed as inputs (ANE-friendly)
    causal_mask, update_mask,
    K_in, V_in,  # explicit KV cache input for this layer (None if KV-shared)
    config, per_layer_combined,
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
):
    """Run one layer statelessly. Returns hidden_states, updated K, updated V, kv stores."""
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    max_hd = config.global_head_dim
    is_full = config.is_full_attention(layer_idx)
    hd = config.get_head_dim(layer_idx)

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

    q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
    q = layer.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    is_kv_shared = config.is_kv_shared(layer_idx)

    K_out, V_out = K_in, V_in  # passthrough if shared

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

        K_out = K_in * (1 - update_mask) + k_padded.expand_as(K_in) * update_mask
        V_out = V_in * (1 - update_mask) + v_padded.expand_as(V_in) * update_mask

        K_for_attn = K_out[..., :hd]
        V_for_attn = V_out[..., :hd]

        if layer_idx == 13:
            kv_store_13_k = K_out[..., :256]
            kv_store_13_v = V_out[..., :256]
        elif layer_idx == 14:
            kv_store_14_k = K_out[..., :512]
            kv_store_14_v = V_out[..., :512]
    else:
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    # GQA: repeat_interleave (verified working on Mac)
    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

    # All fp16 ANE-friendly attention (manual softmax, no float32)
    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + causal_mask
    attn_weights = ane_fused_softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
    attn_output = layer.self_attn["o_proj"](
        attn_output.permute(0, 2, 1).unsqueeze(2)
    ).squeeze(2).permute(0, 2, 1)
    attn_output = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    x_mlp = hidden_states.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
    gate = layer.mlp["gate_proj"](x_mlp)
    up = layer.mlp["up_proj"](x_mlp)
    gate = F.gelu(gate, approximate="tanh")
    mlp_out = layer.mlp["down_proj"](gate * up)
    hidden_states = mlp_out.squeeze(2).permute(0, 2, 1)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

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
    hidden_states = hidden_states * layer.layer_scalar

    return hidden_states, K_out, V_out, kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v


# ============================================================================
# CHUNK 1: layers 0-7 + embedding (8 layers, own KV cache, no KV sharing)
# ============================================================================
class StatelessChunk1(nn.Module):
    START, END = 0, 8

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        # NO embed_tokens buffer — hidden_states passed as input (text or image
        # embedding already selected externally). Eliminates gather op.
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])

    def forward(self, hidden_states, causal_mask, update_mask,
                per_layer_combined,
                cos_s, sin_s, cos_f, sin_f,
                K_in, V_in):
        config = self.config

        # Dummy kv stores (not used in chunk1)
        dummy_13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        dummy_13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        dummy_14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        dummy_14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        K_outs = []
        V_outs = []
        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            K_slot = K_in[local_idx].unsqueeze(0)
            V_slot = V_in[local_idx].unsqueeze(0)
            hidden_states, K_new, V_new, *_ = _run_layer_stateless(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f, causal_mask, update_mask,
                K_slot, V_slot, config, per_layer_combined,
                dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v,
            )
            K_outs.append(K_new.squeeze(0))
            V_outs.append(V_new.squeeze(0))

        K_out = torch.stack(K_outs, dim=0)
        V_out = torch.stack(V_outs, dim=0)
        return hidden_states, K_out, V_out


# ============================================================================
# CHUNK 2: layers 8-14 (7 layers, contains KV sources 13/14)
# ============================================================================
class StatelessChunk2(nn.Module):
    START, END = 8, 15

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])

    def forward(self, hidden_states, causal_mask, update_mask,
                per_layer_combined,
                cos_s, sin_s, cos_f, sin_f,
                K_in, V_in):
        config = self.config

        kv_store_13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv_store_13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv_store_14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv_store_14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        K_outs = []
        V_outs = []
        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            K_slot = K_in[local_idx].unsqueeze(0)
            V_slot = V_in[local_idx].unsqueeze(0)
            (hidden_states, K_new, V_new,
             kv_store_13_k, kv_store_13_v,
             kv_store_14_k, kv_store_14_v) = _run_layer_stateless(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f, causal_mask, update_mask,
                K_slot, V_slot, config, per_layer_combined,
                kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
            )
            K_outs.append(K_new.squeeze(0))
            V_outs.append(V_new.squeeze(0))

        K_out = torch.stack(K_outs, dim=0)
        V_out = torch.stack(V_outs, dim=0)
        return (hidden_states, K_out, V_out,
                kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v)


# ============================================================================
# CHUNK 3: layers 15-24 (10 layers, ALL KV-shared)
# ============================================================================
class StatelessChunk3(nn.Module):
    START, END = 15, 25

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])

    def forward(self, hidden_states, causal_mask, update_mask,
                per_layer_combined,
                cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config

        # All layers shared, no KV cache needed
        dummy_K = torch.zeros(1, 1, self.config.context_length,
                              self.config.global_head_dim, dtype=MODEL_DTYPE)
        dummy_V = torch.zeros(1, 1, self.config.context_length,
                              self.config.global_head_dim, dtype=MODEL_DTYPE)

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            hidden_states, _, _, *_ = _run_layer_stateless(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f, causal_mask, update_mask,
                dummy_K, dummy_V, config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )

        return hidden_states


# ============================================================================
# CHUNK 4: layers 25-34 + norm + lm_head + argmax (10 layers, ALL KV-shared)
# ============================================================================
class StatelessChunk4(nn.Module):
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

    def forward(self, hidden_states, causal_mask, update_mask,
                per_layer_combined,
                cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config

        dummy_K = torch.zeros(1, 1, self.config.context_length,
                              self.config.global_head_dim, dtype=MODEL_DTYPE)
        dummy_V = torch.zeros(1, 1, self.config.context_length,
                              self.config.global_head_dim, dtype=MODEL_DTYPE)

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            hidden_states, _, _, *_ = _run_layer_stateless(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f, causal_mask, update_mask,
                dummy_K, dummy_V, config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )

        hidden_states = self.norm(hidden_states)
        x = hidden_states.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        return token_id, token_logit
