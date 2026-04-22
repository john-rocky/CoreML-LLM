"""Gemma 3 batched-prefill wrapper for CoreML (ANE-optimized).

Processes T prompt tokens per forward instead of 1. Sharing weights with the
T=1 decode wrapper, this compresses a 300-token prompt from ~300 sequential
forward passes into ~10 batched ones — ANE is memory-bandwidth-bound, so the
per-step cost barely grows with T.

Bridge to decode: both wrappers produce a stateful KV cache of the same shape
(`kv_cache_0`, 2·L × kv_heads × ctx × head_dim). Swift runs prefill through
this wrapper to populate the cache, then copies the state buffer byte-wise
into the decode wrapper's `MLState` and continues with the existing T=1
decode path.

KV-cache write under batching:
  update_mask : (1, 1, ctx, T) — column t has a single 1.0 at position p+t
                 (p = chunk-start position). Zero elsewhere.
  K_new       : (1, kv_heads, T, hd) — the T newly-computed keys
  We want K_cache[:, :, p+t, :] = K_new[:, :, t, :]. Written as a mask
  multiply:
      k_increment = matmul(update_mask, K_new) → (1, kv_heads, ctx, hd)
      write_any   = update_mask.sum(dim=-1)    → (1, 1, ctx, 1)  [0 or 1]
      K_cache_new = K_cache * (1 - write_any) + k_increment
  ANE-safe (single matmul + two elementwise ops, no scatter_nd).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, repeat_kv_ane

from .gemma3 import Gemma3Model


class Gemma3PrefillWrapper(nn.Module):
    """Traced Gemma 3 prefill: processes T tokens in one forward pass.

    Outputs the argmax token for the final position so the caller can fold
    the prefill directly into a generation loop (the token after the last
    prompt token is the first generated token).
    """

    def __init__(self, model: Gemma3Model, T: int = 32) -> None:
        super().__init__()
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self.norm = model.norm
        self.lm_head = model.lm_head
        self.argmax = model.argmax
        self.config = model.config
        self.softcap = float(model.softcap or 0.0)
        self.T = T

        self.register_buffer("kv_cache_0", model.kv_cache_0.clone())
        self.register_buffer("cos_sliding", model.cos_sliding)
        self.register_buffer("sin_sliding", model.sin_sliding)
        self.register_buffer("cos_full", model.cos_full)
        self.register_buffer("sin_full", model.sin_full)

        qpas = model.config.query_pre_attn_scalar
        if qpas is None:
            qpas = float(model.config.head_dim)
        self.attn_scale = float(qpas) ** -0.5

    def forward(
        self,
        input_ids: torch.Tensor,        # (1, T) int32
        position_ids: torch.Tensor,     # (T,)   int32
        causal_mask: torch.Tensor,      # (1, 1, T, state_len) fp16
        update_mask: torch.Tensor,      # (1, 1, state_len, T)  fp16  (col t one-hot at pos+t)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        T = self.T
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv = config.num_key_value_heads
        hd = config.head_dim
        n_rep = num_heads // num_kv
        ctx = config.state_length

        # Embedding + √hidden, fp32 residual.
        h = self.embed_tokens(input_ids).to(torch.float32)
        h = h * (config.hidden_size ** 0.5)

        # RoPE: per-token via index_select.
        cos_s = torch.index_select(self.cos_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)  # (1,1,T,hd)
        sin_s = torch.index_select(self.sin_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        cos_f = torch.index_select(self.cos_full, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_f = torch.index_select(self.sin_full, 0, position_ids).unsqueeze(0).unsqueeze(0)

        # Scatter-mask for KV cache (same for every layer).
        write_any = update_mask.sum(dim=-1, keepdim=True)  # (1, 1, ctx, 1)

        for layer_idx in range(num_layers):
            layer = self.layers[layer_idx]
            is_full = config.is_full_attention(layer_idx)

            # ---- Attention block ----
            residual = h
            normed = layer.input_layernorm(h)

            # Conv2d layout: (1, H, 1, T).
            x = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)

            # Q/K/V: (1, q_dim, 1, T) → (1, heads, T, hd).
            q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, T).permute(0, 1, 3, 2)
            q = layer.self_attn["q_norm"](q.reshape(1, num_heads, T, hd))
            k = layer.self_attn["k_proj"](x).view(1, num_kv, hd, T).permute(0, 1, 3, 2)
            k = layer.self_attn["k_norm"](k.reshape(1, num_kv, T, hd))
            v = layer.self_attn["v_proj"](x).view(1, num_kv, hd, T).permute(0, 1, 3, 2)

            if is_full:
                q, k = apply_rotary_pos_emb(q, k, cos_f, sin_f)
            else:
                q, k = apply_rotary_pos_emb(q, k, cos_s, sin_s)

            # KV-cache scatter via matmul.
            k_idx = layer_idx
            v_idx = num_layers + layer_idx
            K_cache = self.kv_cache_0[k_idx].unsqueeze(0)          # (1, kv, ctx, hd)
            V_cache = self.kv_cache_0[v_idx].unsqueeze(0)

            # update_mask broadcasts against batch/kv dims: result (1, kv, ctx, hd).
            k_increment = torch.matmul(update_mask.to(MODEL_DTYPE), k)
            v_increment = torch.matmul(update_mask.to(MODEL_DTYPE), v)

            K_new = K_cache * (1 - write_any) + k_increment
            V_new = V_cache * (1 - write_any) + v_increment

            self.kv_cache_0[k_idx] = K_new.squeeze(0)
            self.kv_cache_0[v_idx] = V_new.squeeze(0)

            # GQA expansion.
            K_expanded = repeat_kv_ane(K_new, n_rep, num_kv, ctx, hd)
            V_expanded = repeat_kv_ane(V_new, n_rep, num_kv, ctx, hd)

            # Attention in fp32.
            q_f = q.to(torch.float32)
            k_f = K_expanded.to(torch.float32)
            attn_weights = torch.matmul(q_f, k_f.transpose(-1, -2)) * self.attn_scale
            attn_weights = attn_weights + causal_mask.to(torch.float32)
            attn_weights = torch.softmax(attn_weights, dim=-1).to(MODEL_DTYPE)
            attn_output = torch.matmul(
                attn_weights.to(torch.float32), V_expanded.to(torch.float32)
            ).to(MODEL_DTYPE)

            # (1, heads, T, hd) → (1, T, heads*hd) → Conv2d o_proj.
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, T, num_heads * hd)
            attn_output = layer.self_attn["o_proj"](
                attn_output.permute(0, 2, 1).unsqueeze(2)
            ).squeeze(2).permute(0, 2, 1)

            attn_output = layer.post_attention_layernorm(attn_output)
            h = residual + attn_output.to(torch.float32)

            # ---- MLP block ----
            residual = h
            normed = layer.pre_feedforward_layernorm(h)
            x_mlp = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
            gate = layer.mlp["gate_proj"](x_mlp)
            up = layer.mlp["up_proj"](x_mlp)
            gate = F.gelu(gate, approximate="tanh")
            mlp_out = layer.mlp["down_proj"](gate * up).squeeze(2).permute(0, 2, 1)
            mlp_out = layer.post_feedforward_layernorm(mlp_out)
            h = residual + mlp_out.to(torch.float32)

        # Final norm + lm_head ONLY on the last T position (we don't need
        # per-token logits for prefill — just the argmax that kicks off decode).
        h = self.norm(h)  # fp32
        last_h = h[:, -1:, :].to(MODEL_DTYPE)  # (1, 1, H)
        x = last_h.permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        return token_id, token_logit
