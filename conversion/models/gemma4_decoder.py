"""Gemma 4 text decoder wrapper for multimodal CoreML conversion.

This wrapper accepts inputs_embeds directly (instead of input_ids),
allowing the calling code to inject vision features at image token positions
before passing to the decoder.

Inputs:
  inputs_embeds: (1, 1, hidden_size) — pre-embedded token (text or vision)
  per_layer_input: (1, 1, num_layers * per_layer_dim) — per-layer embedding info
  position_ids: (1,)
  causal_mask: (1, 1, 1, context_length)
  update_mask: (1, 1, context_length, 1)

Outputs:
  token_id: (1,)
  token_logit: (1,)

State:
  kv_cache_0: (2*num_layers, num_kv_heads, context_length, max_head_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb


def v_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm without learnable scale."""
    mean_sq = x.float().pow(2).mean(-1, keepdim=True) + eps
    return (x.float() * torch.pow(mean_sq, -0.5)).to(x.dtype)
from .gemma4 import Gemma4Model, Gemma4Config


class Gemma4DecoderWrapper(nn.Module):
    """Decoder that accepts pre-embedded inputs for multimodal support."""

    def __init__(self, model: Gemma4Model) -> None:
        super().__init__()
        self.layers = model.layers
        self.norm = model.norm
        self.lm_head = model.lm_head
        self.argmax = model.argmax
        self.config = model.config
        self.softcap = model.softcap

        # KV cache state
        self.register_buffer("kv_cache_0", model.kv_cache_0.clone())

        # Per-layer projection norm
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_dim = model.config.hidden_size_per_layer_input

        # RoPE caches
        self.register_buffer("cos_sliding", model.cos_sliding)
        self.register_buffer("sin_sliding", model.sin_sliding)
        self.register_buffer("cos_full", model.cos_full)
        self.register_buffer("sin_full", model.sin_full)
        self.full_rotary_dim = model.full_rotary_dim

    def forward(
        self,
        inputs_embeds: torch.Tensor,     # (1, 1, hidden_size) — already embedded
        per_layer_input: torch.Tensor,   # (1, 1, num_layers * 256) — combined per-layer info
        position_ids: torch.Tensor,      # (1,)
        causal_mask: torch.Tensor,       # (1, 1, 1, context_length)
        update_mask: torch.Tensor,       # (1, 1, context_length, 1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        num_layers = config.num_hidden_layers

        hidden_states = inputs_embeds.to(MODEL_DTYPE)

        # RoPE for both attention types
        cos_s = torch.index_select(self.cos_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_s = torch.index_select(self.sin_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        cos_f = torch.index_select(self.cos_full, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_f = torch.index_select(self.sin_full, 0, position_ids).unsqueeze(0).unsqueeze(0)

        # Transformer layers
        for layer_idx in range(num_layers):
            layer = self.layers[layer_idx]
            is_full = config.is_full_attention(layer_idx)
            hd = config.get_head_dim(layer_idx)
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            n_rep = num_heads // num_kv_heads
            scale = 1.0 / (hd ** 0.5)

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            # QKV projection
            x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
            q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
            k = layer.self_attn["k_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
            v = layer.self_attn["v_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)

            # QK normalization
            q = layer.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
            k = layer.self_attn["k_norm"](k.reshape(1, num_kv_heads, hd)).view(1, num_kv_heads, 1, hd)
            v = v_norm(v)  # RMSNorm without scale on values

            # RoPE (cos/sin already padded to full head_dim)
            if is_full:
                q, k = apply_rotary_pos_emb(q, k, cos_f, sin_f)
            else:
                q, k = apply_rotary_pos_emb(q, k, cos_s, sin_s)

            # KV cache update
            k_idx = layer_idx
            v_idx = num_layers + layer_idx
            max_hd = config.global_head_dim

            K_cache = self.kv_cache_0[k_idx].unsqueeze(0)
            V_cache = self.kv_cache_0[v_idx].unsqueeze(0)

            if hd < max_hd:
                pad_size = max_hd - hd
                k_padded = F.pad(k, (0, pad_size))
                v_padded = F.pad(v, (0, pad_size))
            else:
                k_padded = k
                v_padded = v

            K_new = K_cache * (1 - update_mask) + k_padded.expand_as(K_cache) * update_mask
            V_new = V_cache * (1 - update_mask) + v_padded.expand_as(V_cache) * update_mask

            self.kv_cache_0[k_idx] = K_new.squeeze(0)
            self.kv_cache_0[v_idx] = V_new.squeeze(0)

            K_for_attn = K_new[..., :hd]
            V_for_attn = V_new[..., :hd]
            K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
            V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

            # Attention
            q_f = q.to(torch.float32)
            k_f = K_expanded.to(torch.float32)
            attn_weights = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale
            attn_weights = attn_weights + causal_mask.to(torch.float32)
            attn_weights = torch.softmax(attn_weights, dim=-1).to(MODEL_DTYPE)
            attn_output = torch.matmul(
                attn_weights.to(torch.float32), V_expanded.to(torch.float32)
            ).to(MODEL_DTYPE)

            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
            attn_output = layer.self_attn["o_proj"](
                attn_output.permute(0, 2, 1).unsqueeze(2)
            ).squeeze(2).permute(0, 2, 1)

            attn_output = layer.post_attention_layernorm(attn_output)
            hidden_states = residual + attn_output

            # MLP
            residual = hidden_states
            hidden_states = layer.pre_feedforward_layernorm(hidden_states)
            x = hidden_states.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
            gate = layer.mlp["gate_proj"](x)
            up = layer.mlp["up_proj"](x)
            gate = F.gelu(gate, approximate="tanh")
            mlp_out = layer.mlp["down_proj"](gate * up)
            hidden_states = mlp_out.squeeze(2).permute(0, 2, 1)
            hidden_states = layer.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            # Per-layer input (matches HF: gate → act_fn → multiply → project → norm → residual)
            residual_pl = hidden_states
            start = layer_idx * self.per_layer_dim
            end = start + self.per_layer_dim
            per_layer_slice = per_layer_input[:, :, start:end]
            gated = layer.per_layer_input_gate(hidden_states.to(MODEL_DTYPE))
            gated = F.gelu(gated, approximate="tanh")  # act_fn after gate
            gated = gated * per_layer_slice
            gated = layer.per_layer_projection(gated)
            hidden_states = layer.post_per_layer_input_norm(gated)
            hidden_states = residual_pl + hidden_states

            hidden_states = hidden_states * layer.layer_scalar

        # Final norm + LM head
        hidden_states = self.norm(hidden_states)
        x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))

        return token_id, token_logit
