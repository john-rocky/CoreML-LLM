"""Gemma 4 monolithic wrapper for CoreML tracing.

Handles Gemma 4's complex architecture:
- Dual attention (sliding/full with different head_dims)
- QK normalization
- Sandwich norm (4 norms per layer)
- GELU activation
- Logit softcapping
- Partial rotary embeddings for full attention
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb


def v_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm without learnable scale (matches Gemma4RMSNorm with_scale=False)."""
    mean_sq = x.float().pow(2).mean(-1, keepdim=True) + eps
    return (x.float() * torch.pow(mean_sq, -0.5)).to(x.dtype)
from .gemma4 import Gemma4Model, Gemma4Config


class Gemma4MonolithicWrapper(nn.Module):
    """Monolithic wrapper for Gemma 4 E2B text decoder."""

    def __init__(self, model: Gemma4Model) -> None:
        super().__init__()
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self.norm = model.norm
        self.lm_head = model.lm_head
        self.argmax = model.argmax
        self.config = model.config
        self.softcap = model.softcap

        # KV cache state
        self.register_buffer("kv_cache_0", model.kv_cache_0.clone())

        # Per-layer embeddings
        self.embed_tokens_per_layer = model.embed_tokens_per_layer
        self.per_layer_model_projection = model.per_layer_model_projection
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_dim = model.config.hidden_size_per_layer_input  # 256
        self.num_layers = model.config.num_hidden_layers
        self.per_layer_model_projection_scale = model.per_layer_model_projection_scale
        self.per_layer_input_scale = model.per_layer_input_scale
        self.per_layer_embed_scale = model.per_layer_embed_scale

        # RoPE caches
        self.register_buffer("cos_sliding", model.cos_sliding)
        self.register_buffer("sin_sliding", model.sin_sliding)
        self.register_buffer("cos_full", model.cos_full)
        self.register_buffer("sin_full", model.sin_full)
        self.full_rotary_dim = model.full_rotary_dim

        # KV sharing map: kv_shared_source[layer_idx] = source_layer_idx or -1 if not shared
        self.kv_shared_source = {}
        self.kv_store_layers = set()
        # Layer 13 (sliding) stores KV, shared by layers 15-18, 20-23, 25-28, 30-33
        # Layer 14 (full) stores KV, shared by layers 19, 24, 29, 34
        for i in range(self.config.num_hidden_layers):
            if i >= self.config.num_hidden_layers - self.config.num_kv_shared_layers:
                # KV-shared layer: find source
                if self.config.is_full_attention(i):
                    self.kv_shared_source[i] = 14  # full_attention parent
                else:
                    self.kv_shared_source[i] = 13  # sliding_attention parent
        self.kv_store_layers = {13, 14}  # layers that store KV for sharing

    def forward(
        self,
        input_ids: torch.Tensor,     # (1, 1)
        position_ids: torch.Tensor,  # (1,)
        causal_mask: torch.Tensor,   # (1, 1, 1, context_length)
        update_mask: torch.Tensor,   # (1, 1, context_length, 1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        num_layers = config.num_hidden_layers

        # --- Embedding ---
        # HF ScaledWordEmbedding has embed_scale=sqrt(hidden_size) built-in.
        # Our plain nn.Embedding doesn't, so we apply it here.
        hidden_states = self.embed_tokens(input_ids).to(MODEL_DTYPE)
        hidden_states = hidden_states * (config.hidden_size ** 0.5)

        # --- Per-layer embeddings (matches HF project_per_layer_inputs) ---
        # 1. Look up per-layer token embeddings and scale
        per_layer_raw = self.embed_tokens_per_layer(input_ids).to(MODEL_DTYPE) * self.per_layer_embed_scale
        # Shape: (1, 1, num_layers * per_layer_dim) = (1, 1, 8960)

        # 2. Project main embeddings to per-layer space
        per_layer_proj = self.per_layer_model_projection(hidden_states.float()).to(MODEL_DTYPE) * self.per_layer_model_projection_scale

        # 3. Apply norm to projection per-layer slices (HF does this BEFORE combining)
        # We apply norm to each 256-dim slice of the 8960-dim projection
        normed_slices = []
        for li in range(self.num_layers):
            s = li * self.per_layer_dim
            e = s + self.per_layer_dim
            normed_slices.append(self.per_layer_projection_norm(per_layer_proj[:, :, s:e]))
        per_layer_proj_normed = torch.cat(normed_slices, dim=-1)

        # 4. Combine: (normed_projection + raw) * input_scale
        per_layer_combined = (per_layer_proj_normed + per_layer_raw) * self.per_layer_input_scale

        # --- Get RoPE for both types ---
        cos_s = torch.index_select(self.cos_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_s = torch.index_select(self.sin_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        cos_f = torch.index_select(self.cos_full, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_f = torch.index_select(self.sin_full, 0, position_ids).unsqueeze(0).unsqueeze(0)

        # --- Transformer layers ---
        # KV sharing: layers 13 and 14 store KV; layers 15+ read from them
        # We use dedicated buffers instead of a dict (trace-compatible)
        kv_store_13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv_store_13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv_store_14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv_store_14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

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

            x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

            # Q projection + norm + RoPE (always computed)
            q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
            q = layer.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
            if is_full:
                q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
            else:
                q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

            # K/V: compute for non-shared layers, read from cache for shared
            is_kv_shared = config.is_kv_shared(layer_idx)
            k_idx = layer_idx
            v_idx = num_layers + layer_idx
            max_hd = config.global_head_dim

            if not is_kv_shared:
                # Compute own K/V
                k = layer.self_attn["k_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
                v = layer.self_attn["v_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
                k = layer.self_attn["k_norm"](k.reshape(1, num_kv_heads, hd)).view(1, num_kv_heads, 1, hd)
                v = v_norm(v)
                if is_full:
                    _, k = apply_rotary_pos_emb(k, k, cos_f, sin_f)
                else:
                    _, k = apply_rotary_pos_emb(k, k, cos_s, sin_s)

                # Write to KV cache
                K_cache = self.kv_cache_0[k_idx].unsqueeze(0)
                V_cache = self.kv_cache_0[v_idx].unsqueeze(0)
                if hd < max_hd:
                    k_padded = F.pad(k, (0, max_hd - hd))
                    v_padded = F.pad(v, (0, max_hd - hd))
                else:
                    k_padded = k
                    v_padded = v
                K_new = K_cache * (1 - update_mask) + k_padded.expand_as(K_cache) * update_mask
                V_new = V_cache * (1 - update_mask) + v_padded.expand_as(V_cache) * update_mask
                self.kv_cache_0[k_idx] = K_new.squeeze(0)
                self.kv_cache_0[v_idx] = V_new.squeeze(0)
                K_for_attn = K_new[..., :hd]
                V_for_attn = V_new[..., :hd]

                # Store for KV sharing (layers 13 and 14)
                if layer_idx == 13:
                    kv_store_13_k = K_new[..., :256]
                    kv_store_13_v = V_new[..., :256]
                elif layer_idx == 14:
                    kv_store_14_k = K_new[..., :512]
                    kv_store_14_v = V_new[..., :512]
            else:
                # KV-shared: read from parent's KV cache
                if is_full:
                    # Full attention layers (19, 24, 29, 34) share from layer 14
                    K_for_attn = kv_store_14_k
                    V_for_attn = kv_store_14_v
                else:
                    # Sliding attention layers share from layer 13
                    K_for_attn = kv_store_13_k
                    V_for_attn = kv_store_13_v

            # Expand KV for GQA
            K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
            V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

            # Attention in fp32
            q_f = q.to(torch.float32)
            k_f = K_expanded.to(torch.float32)
            attn_weights = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale
            attn_weights = attn_weights + causal_mask.to(torch.float32)
            attn_weights = torch.softmax(attn_weights, dim=-1).to(MODEL_DTYPE)
            attn_output = torch.matmul(
                attn_weights.to(torch.float32), V_expanded.to(torch.float32)
            ).to(MODEL_DTYPE)

            # Output projection
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
            attn_output = layer.self_attn["o_proj"](
                attn_output.permute(0, 2, 1).unsqueeze(2)
            ).squeeze(2).permute(0, 2, 1)

            # Post-attention norm (sandwich)
            attn_output = layer.post_attention_layernorm(attn_output)

            hidden_states = residual + attn_output

            # MLP with sandwich norm
            residual = hidden_states
            hidden_states = layer.pre_feedforward_layernorm(hidden_states)

            # GELU MLP
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
            per_layer_slice = per_layer_combined[:, :, start:end]  # (1, 1, 256)
            gated = layer.per_layer_input_gate(hidden_states.to(MODEL_DTYPE))
            gated = F.gelu(gated, approximate="tanh")  # act_fn after gate
            gated = gated * per_layer_slice
            gated = layer.per_layer_projection(gated)
            hidden_states = layer.post_per_layer_input_norm(gated)
            hidden_states = residual_pl + hidden_states

            # Apply layer scalar (scales entire layer output including residual)
            hidden_states = hidden_states * layer.layer_scalar

        # --- Final norm ---
        hidden_states = self.norm(hidden_states)

        # --- LM Head + Softcapping + Argmax ---
        x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)

        # Logit softcapping: tanh(logits / cap) * cap
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap

        token_id, token_logit = self.argmax(logits.squeeze(0))

        return token_id, token_logit
