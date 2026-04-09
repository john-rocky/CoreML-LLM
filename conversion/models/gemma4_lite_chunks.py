"""Gemma 4 lite chunked: 2 chunks for ANE compile.

Chunk1: layers 0-14 (incl KV sources 13, 14) + embedding
Chunk2: layers 15-34 (all shared KV from 13/14) + norm + lm_head

Reduces graph size per chunk → ANE compiler can plan.
All fp16, external PLE.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb


def v_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean_sq = x.float().pow(2).mean(-1, keepdim=True) + eps
    return (x.float() * torch.pow(mean_sq, -0.5)).to(x.dtype)

from .gemma4 import Gemma4Model


def _run_layer(layer, layer_idx, hidden_states, cos_s, sin_s, cos_f, sin_f,
               causal_mask, update_mask, kv_cache, kv_k_idx, kv_v_idx,
               config, per_layer_combined,
               kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v):
    """Run a single transformer layer. Returns updated hidden_states and kv stores."""
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

    if not is_kv_shared:
        k = layer.self_attn["k_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = layer.self_attn["v_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        k = layer.self_attn["k_norm"](k.reshape(1, num_kv_heads, hd)).view(1, num_kv_heads, 1, hd)
        v = v_norm(v)
        if is_full:
            _, k = apply_rotary_pos_emb(k, k, cos_f, sin_f)
        else:
            _, k = apply_rotary_pos_emb(k, k, cos_s, sin_s)

        K_cache = kv_cache[kv_k_idx].unsqueeze(0)
        V_cache = kv_cache[kv_v_idx].unsqueeze(0)
        if hd < max_hd:
            k_padded = F.pad(k, (0, max_hd - hd))
            v_padded = F.pad(v, (0, max_hd - hd))
        else:
            k_padded, v_padded = k, v
        K_new = K_cache * (1 - update_mask) + k_padded.expand_as(K_cache) * update_mask
        V_new = V_cache * (1 - update_mask) + v_padded.expand_as(V_cache) * update_mask
        kv_cache[kv_k_idx] = K_new.squeeze(0)
        kv_cache[kv_v_idx] = V_new.squeeze(0)
        K_for_attn = K_new[..., :hd]
        V_for_attn = V_new[..., :hd]

        if layer_idx == 13:
            kv_store_13_k = K_new[..., :256]
            kv_store_13_v = V_new[..., :256]
        elif layer_idx == 14:
            kv_store_14_k = K_new[..., :512]
            kv_store_14_v = V_new[..., :512]
    else:
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + causal_mask
    attn_weights = torch.softmax(attn_weights, dim=-1)
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
    gated = layer.per_layer_input_gate(hidden_states.to(MODEL_DTYPE))
    gated = F.gelu(gated, approximate="tanh")
    gated = gated * per_layer_slice
    gated = layer.per_layer_projection(gated)
    hidden_states = layer.post_per_layer_input_norm(gated)
    hidden_states = residual_pl + hidden_states
    hidden_states = hidden_states * layer.layer_scalar

    return hidden_states, kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v


class LiteChunk1(nn.Module):
    """Layers 0-14 + embedding. Outputs hidden_states and kv13/14 stores."""

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.embed_tokens = model.embed_tokens
        self.layers = nn.ModuleList([model.layers[i] for i in range(15)])
        self.register_buffer("cos_sliding", model.cos_sliding)
        self.register_buffer("sin_sliding", model.sin_sliding)
        self.register_buffer("cos_full", model.cos_full)
        self.register_buffer("sin_full", model.sin_full)
        ctx = self.config.context_length
        max_hd = self.config.global_head_dim
        # KV cache for 15 layers × 2 (K, V) = 30 entries
        self.register_buffer("kv_cache_0", torch.zeros(30, 1, ctx, max_hd, dtype=MODEL_DTYPE))

    def forward(self, input_ids, position_ids, causal_mask, update_mask,
                per_layer_combined, image_embedding):
        config = self.config
        text_emb = self.embed_tokens(input_ids).to(MODEL_DTYPE)
        text_emb = text_emb * torch.tensor(config.hidden_size ** 0.5, dtype=MODEL_DTYPE)
        is_image = (image_embedding.abs().sum(dim=-1, keepdim=True) > 0).to(MODEL_DTYPE)
        hidden_states = text_emb * (1 - is_image) + image_embedding * is_image

        cos_s = torch.index_select(self.cos_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_s = torch.index_select(self.sin_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        cos_f = torch.index_select(self.cos_full, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_f = torch.index_select(self.sin_full, 0, position_ids).unsqueeze(0).unsqueeze(0)

        kv_store_13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv_store_13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv_store_14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv_store_14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        for local_idx in range(15):
            layer_idx = local_idx
            hidden_states, kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v = _run_layer(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f, causal_mask, update_mask,
                self.kv_cache_0, local_idx, 15 + local_idx,
                config, per_layer_combined,
                kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
            )

        return hidden_states, kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v


class LiteChunk2(nn.Module):
    """Layers 15-34 + norm + lm_head + argmax. Takes kv13/14 as input."""

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(15, 35)])
        self.norm = model.norm
        self.lm_head = nn.Conv2d(model.lm_head.in_channels, model.lm_head.out_channels,
                                  kernel_size=1, bias=False)
        self.lm_head.weight.data = model.lm_head.weight.data.clone()
        self.argmax = model.argmax
        self.softcap = model.softcap
        self.register_buffer("cos_sliding", model.cos_sliding)
        self.register_buffer("sin_sliding", model.sin_sliding)
        self.register_buffer("cos_full", model.cos_full)
        self.register_buffer("sin_full", model.sin_full)
        ctx = self.config.context_length
        max_hd = self.config.global_head_dim
        # These are all shared layers so no KV cache needed
        self.register_buffer("kv_cache_dummy", torch.zeros(1, 1, ctx, max_hd, dtype=MODEL_DTYPE))

    def forward(self, hidden_states, position_ids, causal_mask, update_mask,
                per_layer_combined, kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config

        cos_s = torch.index_select(self.cos_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_s = torch.index_select(self.sin_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        cos_f = torch.index_select(self.cos_full, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_f = torch.index_select(self.sin_full, 0, position_ids).unsqueeze(0).unsqueeze(0)

        # All layers in chunk2 are KV-shared (from 13 or 14)
        for local_idx in range(20):
            layer_idx = 15 + local_idx
            hidden_states, _, _, _, _ = _run_layer(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f, causal_mask, update_mask,
                self.kv_cache_dummy, 0, 0,  # unused (all KV shared)
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )

        hidden_states = self.norm(hidden_states)
        x = hidden_states.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        return token_id, token_logit
