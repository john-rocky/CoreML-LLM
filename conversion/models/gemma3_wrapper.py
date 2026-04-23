"""Gemma 3 monolithic wrapper for CoreML tracing (ANE-optimized).

Single-token decode wrapper used by exporter.py. Mirrors Gemma4MonolithicWrapper
but without Gemma 4's extras: no PLE, no KV sharing, no dual head_dim, no
v_norm, no layer_scalar. Keeps:
- sandwich norms (4 per layer)
- QK norm
- GeGLU with tanh-approx GELU
- dual RoPE (global θ for full-attn layers, local θ for sliding-attn layers)
- mask-based KV-cache write (ANE-resident alternative to index_copy_)
- GQA expansion via repeat_kv_ane (reshape+repeat+view)

Sliding-window masking: we rely on the caller to pass a pre-baked `causal_mask`
shaped (1,1,1,state_length) that already has -1e4 outside the window for
sliding-attn positions. Dual masks would double the input surface; the Swift
runtime composes the per-layer mask upstream the same way Gemma 4 does for its
sliding layers today (see Sources/CoreMLLLM/*.swift).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, repeat_kv_ane

from .gemma3 import Gemma3Model


class Gemma3MonolithicWrapper(nn.Module):
    """Traced Gemma 3 decoder: (input_ids, position_ids, causal_mask, update_mask) → (token_id, token_logit)."""

    def __init__(self, model: Gemma3Model) -> None:
        super().__init__()
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self.norm = model.norm
        self.lm_head = model.lm_head
        self.argmax = model.argmax
        self.config = model.config
        self.softcap = float(model.softcap or 0.0)

        # Stateful KV cache buffer (registered so CoreML maps it to an MLState).
        self.register_buffer("kv_cache_0", model.kv_cache_0.clone())

        # Dual RoPE tables.
        self.register_buffer("cos_sliding", model.cos_sliding)
        self.register_buffer("sin_sliding", model.sin_sliding)
        self.register_buffer("cos_full", model.cos_full)
        self.register_buffer("sin_full", model.sin_full)

        # Gemma 3 applies query_pre_attn_scalar in addition to QK norm
        # (transformers/models/gemma3/modeling_gemma3.py Gemma3Attention):
        #     scale = query_pre_attn_scalar ** -0.5
        # For Gemma 3 270M this is 256, giving scale = 1/16. Note Gemma 4
        # uses scale=1.0 despite QK norm — do NOT copy that here.
        qpas = model.config.query_pre_attn_scalar
        if qpas is None:
            qpas = float(model.config.head_dim)
        self.attn_scale = float(qpas) ** -0.5

    def forward(
        self,
        input_ids: torch.Tensor,        # (1, 1) int32
        position_ids: torch.Tensor,     # (1,)   int32
        causal_mask: torch.Tensor,      # (1, 1, 1, state_length) float16, additive -1e4 outside
        update_mask: torch.Tensor,      # (1, 1, state_length, 1) float16, 1.0 at write pos
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        n_rep = num_heads // num_kv_heads
        state_len = config.state_length

        # Embedding + √hidden scaling (Gemma 3 matches Gemma 4 here).
        # Gemma 3 was trained in bf16; converting to fp16 overflows the residual
        # stream (values hit +inf by layer 7) because there is no `layer_scalar`
        # to dampen each layer's contribution like Gemma 4 has. We keep the
        # residual stream in fp32 and only cast down to fp16 inside the sublayer
        # compute where Conv2d is fp16 for ANE efficiency. HF does the same
        # (internally upcasts for RMSNorm and residual adds).
        hidden_states = self.embed_tokens(input_ids).to(torch.float32)
        hidden_states = hidden_states * (config.hidden_size ** 0.5)

        # RoPE lookups via index_select (ANE-safe).
        cos_s = torch.index_select(self.cos_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_s = torch.index_select(self.sin_sliding, 0, position_ids).unsqueeze(0).unsqueeze(0)
        cos_f = torch.index_select(self.cos_full, 0, position_ids).unsqueeze(0).unsqueeze(0)
        sin_f = torch.index_select(self.sin_full, 0, position_ids).unsqueeze(0).unsqueeze(0)

        for layer_idx in range(num_layers):
            layer = self.layers[layer_idx]
            is_full = config.is_full_attention(layer_idx)

            # ---- Attention block (sandwich: input_ln → attn → post_attention_ln) ----
            # Residual kept in fp32; pass fp32 directly into the norm — GemmaRMSNorm
            # does its math in fp32 regardless of input dtype and returns the same
            # dtype it was given. Casting h to fp16 here would overflow after a
            # few layers since the residual stream reaches ~100k.
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)

            # Conv2d layout: (B, hidden, 1, seq=1). normed is fp32; the sublayer
            # projections are fp16, so cast here where values are bounded (post-norm).
            x = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)

            # Q projection + QK norm + RoPE.
            q = layer.self_attn["q_proj"](x).view(1, num_heads, head_dim, 1).permute(0, 1, 3, 2)
            q = layer.self_attn["q_norm"](q.reshape(1, num_heads, head_dim)).view(1, num_heads, 1, head_dim)
            k = layer.self_attn["k_proj"](x).view(1, num_kv_heads, head_dim, 1).permute(0, 1, 3, 2)
            k = layer.self_attn["k_norm"](k.reshape(1, num_kv_heads, head_dim)).view(1, num_kv_heads, 1, head_dim)
            v = layer.self_attn["v_proj"](x).view(1, num_kv_heads, head_dim, 1).permute(0, 1, 3, 2)

            if is_full:
                q, k = apply_rotary_pos_emb(q, k, cos_f, sin_f)
            else:
                q, k = apply_rotary_pos_emb(q, k, cos_s, sin_s)

            # Mask-based KV-cache write (ANE-friendly; avoids index_copy_).
            k_idx = layer_idx
            v_idx = num_layers + layer_idx
            K_cache = self.kv_cache_0[k_idx].unsqueeze(0)
            V_cache = self.kv_cache_0[v_idx].unsqueeze(0)

            k_broadcast = k.expand_as(K_cache)
            v_broadcast = v.expand_as(V_cache)
            K_new = K_cache * (1 - update_mask) + k_broadcast * update_mask
            V_new = V_cache * (1 - update_mask) + v_broadcast * update_mask

            self.kv_cache_0[k_idx] = K_new.squeeze(0)
            self.kv_cache_0[v_idx] = V_new.squeeze(0)

            # GQA expansion via reshape+repeat+view (not repeat_interleave).
            K_expanded = repeat_kv_ane(K_new, n_rep, num_kv_heads, state_len, head_dim)
            V_expanded = repeat_kv_ane(V_new, n_rep, num_kv_heads, state_len, head_dim)

            # Attention in fp32 for numerical stability. Gemma 3 scales by
            # query_pre_attn_scalar ** -0.5 even though QK norm is applied.
            q_f = q.to(torch.float32)
            k_f = K_expanded.to(torch.float32)
            attn_weights = torch.matmul(q_f, k_f.transpose(-1, -2)) * self.attn_scale
            attn_weights = attn_weights + causal_mask.to(torch.float32)
            attn_weights = torch.softmax(attn_weights, dim=-1).to(MODEL_DTYPE)
            attn_output = torch.matmul(
                attn_weights.to(torch.float32), V_expanded.to(torch.float32)
            ).to(MODEL_DTYPE)

            # Output projection.
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
            attn_output = layer.self_attn["o_proj"](
                attn_output.permute(0, 2, 1).unsqueeze(2)
            ).squeeze(2).permute(0, 2, 1)

            attn_output = layer.post_attention_layernorm(attn_output)
            # fp32 residual add (attn_output is fp16 → upcast).
            hidden_states = residual + attn_output.to(torch.float32)

            # ---- MLP block (sandwich: pre_ff_ln → GeGLU → post_ff_ln) ----
            residual = hidden_states
            normed = layer.pre_feedforward_layernorm(hidden_states)

            x_mlp = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
            gate = layer.mlp["gate_proj"](x_mlp)
            up = layer.mlp["up_proj"](x_mlp)
            gate = F.gelu(gate, approximate="tanh")
            mlp_out = layer.mlp["down_proj"](gate * up)
            mlp_out = mlp_out.squeeze(2).permute(0, 2, 1)

            mlp_out = layer.post_feedforward_layernorm(mlp_out)
            hidden_states = residual + mlp_out.to(torch.float32)

        # Final norm (fp32 in, fp32 out; GemmaRMSNorm handles the precision).
        hidden_states = self.norm(hidden_states)

        # LM head → (optional) softcap → argmax.
        x = hidden_states.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap

        token_id, token_logit = self.argmax(logits.squeeze(0))
        return token_id, token_logit
