"""Gemma 4 STATEFUL chunks — KV cache as MLState (iOS 18+).

Re-evaluation of MLState on iOS 26 / coremltools 9.0. The stateless variant
passes KV as explicit I/O tensors, incurring per-dispatch IOSurface overhead.
Per Orion (arXiv 2603.06728), this overhead — not compute — dominates decode.

MLState keeps KV resident in ANE-managed memory. Each chunk's KV is a
register_buffer with index-assignment mutations (captured by tracer).

Chunking:
  chunk2: layers 8-14 (5 sliding + 2 full-attn, KV as state)
  chunk3/4: unchanged (no own KV, read kv13/kv14 as regular inputs)
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


# ============================================================================
# CHUNK 2 (stateful): layers 8-14, KV as internal state
# ============================================================================
class StatefulChunk2(nn.Module):
    """Layers 8-14 with KV cache as register_buffer (MLState).

    Two state buffers (different seq-dim sizes):
    - kv_sliding: (10, 1, W, max_hd) — 5 K slots [0-4] + 5 V slots [5-9]
    - kv_full: (4, 1, CTX, max_hd) — 2 K slots [0-1] + 2 V slots [2-3]

    Sliding layers: shift-and-append (cat).
    Full layers: mask-based update (same as stateless).
    Outputs kv13/kv14 as regular tensors for chunk3/4.
    """
    START, END = 8, 15
    N_SLIDING = 5  # L8-12
    N_FULL = 2     # L13-14

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])

        ctx = model.config.context_length
        W = model.config.sliding_window
        max_hd = model.config.global_head_dim

        # State buffers: K and V stacked on dim 0
        self.register_buffer("kv_sliding",
            torch.zeros(self.N_SLIDING * 2, 1, W, max_hd, dtype=MODEL_DTYPE))
        self.register_buffer("kv_full",
            torch.zeros(self.N_FULL * 2, 1, ctx, max_hd, dtype=MODEL_DTYPE))

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                update_mask, per_layer_combined,
                cos_s, sin_s, cos_f, sin_f):
        config = self.config
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        n_rep = num_heads // num_kv_heads
        max_hd = config.global_head_dim

        kv13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        sliding_idx = 0
        full_idx = 0

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            layer = self.layers[local_idx]
            is_full = config.is_full_attention(layer_idx)
            hd = config.get_head_dim(layer_idx)

            # --- Pre-norm + QKV ---
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

            q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
            q = layer.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
            if is_full:
                q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
            else:
                q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

            # --- KV compute + state update ---
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
                # Full-attention: mask-based update on kv_full state
                ki, vi = full_idx, self.N_FULL + full_idx
                K_cur = self.kv_full[ki].unsqueeze(0)  # (1, 1, CTX, hd)
                V_cur = self.kv_full[vi].unsqueeze(0)
                K_new = K_cur * (1 - update_mask) + k_padded.expand_as(K_cur) * update_mask
                V_new = V_cur * (1 - update_mask) + v_padded.expand_as(V_cur) * update_mask
                self.kv_full[ki] = K_new.squeeze(0)
                self.kv_full[vi] = V_new.squeeze(0)
                K_for_attn = K_new[..., :hd]
                V_for_attn = V_new[..., :hd]

                if layer_idx == 14:
                    kv14_k = K_new[..., :512]
                    kv14_v = V_new[..., :512]
                full_idx += 1
            else:
                # Sliding: shift-and-append on kv_sliding state
                ki, vi = sliding_idx, self.N_SLIDING + sliding_idx
                K_cur = self.kv_sliding[ki].unsqueeze(0)  # (1, 1, W, hd)
                V_cur = self.kv_sliding[vi].unsqueeze(0)
                K_new = torch.cat([K_cur[:, :, 1:, :], k_padded], dim=2)
                V_new = torch.cat([V_cur[:, :, 1:, :], v_padded], dim=2)
                self.kv_sliding[ki] = K_new.squeeze(0)
                self.kv_sliding[vi] = V_new.squeeze(0)
                K_for_attn = K_new[..., :hd]
                V_for_attn = V_new[..., :hd]

                if layer_idx == 13:
                    kv13_k = K_new[..., :256]
                    kv13_v = V_new[..., :256]
                sliding_idx += 1

            # --- Attention ---
            K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
            V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)
            causal_mask = causal_mask_full if is_full else causal_mask_sliding
            attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
            attn_weights = attn_weights + causal_mask
            attn_weights = ane_softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, V_expanded)

            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
            attn_output = layer.self_attn["o_proj"](
                attn_output.permute(0, 2, 1).unsqueeze(2)
            ).squeeze(2).permute(0, 2, 1)
            attn_output = layer.post_attention_layernorm(attn_output)
            hidden_states = residual + attn_output

            # --- MLP ---
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

            # --- Per-layer input ---
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

        return (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v)
