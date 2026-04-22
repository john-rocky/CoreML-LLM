"""Gemma 3 bidirectional encoder (ANE-optimized) for EmbeddingGemma.

EmbeddingGemma-300M uses a Gemma 3 transformer backbone modified to use
**bidirectional attention** instead of causal (arXiv:2509.20354 §3). The
attention block structure is otherwise identical to the causal decoder in
models/gemma3.py:
  - sandwich norms (4 per layer)
  - q_norm / k_norm (RMSNorm per head on Q and K)
  - GeGLU MLP with tanh-approx GELU
  - GQA (3 heads / 1 kv head by default)
  - sliding_window_pattern alternation with dual RoPE (local θ vs global θ)

Differences from the causal wrapper:
  - No causal mask. Attention is over the whole sequence; we apply only a
    pad-mask built from `attention_mask` (fp16 additive, −1e4 on pad positions).
  - No KV cache. One full-sequence forward.
  - Fixed trace-time sequence length (see build_embeddinggemma_bundle.py);
    variable-length is handled upstream by padding to the nearest bucket.

ANE layout follows docs/ANE_OPTIMIZATION_SURVEY.md + conversion/ane_ops.py:
all projections are Conv2d(1×1) with (B, C, 1, S) layout, RMSNorm uses the
cat([x, −x])→LayerNorm trick, softmax is fp16 on dim=−1, and GQA expansion
uses reshape+repeat+view (repeat_kv_ane) instead of repeat_interleave.
"""

from __future__ import annotations

import json
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ane_ops import (
    MODEL_DTYPE,
    ANERMSNorm,
    apply_rotary_pos_emb,
    repeat_kv_ane,
)


class EncoderConfig:
    """Gemma 3 encoder config (read from HF config.json for EmbeddingGemma)."""

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 24)
        self.num_attention_heads = kwargs.get("num_attention_heads", 3)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 1)
        self.head_dim = kwargs.get("head_dim", 256)
        self.intermediate_size = kwargs.get("intermediate_size", 1152)
        self.vocab_size = kwargs.get("vocab_size", 262144)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.attention_bias = kwargs.get("attention_bias", False)
        self.sliding_window = kwargs.get("sliding_window", 512)
        self.sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 2048)

        self.rope_theta = kwargs.get("rope_theta", 1_000_000.0)
        self.rope_local_base_freq = kwargs.get(
            "rope_local_base_freq", kwargs.get("rope_local_theta", 10_000.0)
        )

        self.layer_types = kwargs.get("layer_types", [])
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if (i + 1) % self.sliding_window_pattern == 0
                else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]

        # Trace-time fixed sequence length.
        self.max_seq_len = kwargs.get("max_seq_len", 512)

    @classmethod
    def from_json(cls, path: str, max_seq_len: int = 512) -> EncoderConfig:
        with open(path) as f:
            d = json.load(f)
        if "text_config" in d:
            d = d["text_config"]
        d["max_seq_len"] = max_seq_len
        return cls(**d)

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"


class EncoderLayer(nn.Module):
    """One bidirectional Gemma-3 block (ANE layout)."""

    def __init__(self, config: EncoderConfig, layer_idx: int):
        super().__init__()
        self.is_full = config.is_full_attention(layer_idx)
        self.sliding_window = config.sliding_window

        hidden = config.hidden_size
        head_dim = config.head_dim
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        inter = config.intermediate_size
        eps = config.rms_norm_eps
        has_bias = config.attention_bias

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Conv2d(hidden, q_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "k_proj": nn.Conv2d(hidden, kv_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "v_proj": nn.Conv2d(hidden, kv_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "o_proj": nn.Conv2d(q_dim, hidden, 1, bias=False, dtype=MODEL_DTYPE),
            "q_norm": ANERMSNorm(head_dim, eps=eps),
            "k_norm": ANERMSNorm(head_dim, eps=eps),
        })
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Conv2d(hidden, inter, 1, bias=False, dtype=MODEL_DTYPE),
            "up_proj": nn.Conv2d(hidden, inter, 1, bias=False, dtype=MODEL_DTYPE),
            "down_proj": nn.Conv2d(inter, hidden, 1, bias=False, dtype=MODEL_DTYPE),
        })
        self.input_layernorm = ANERMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = ANERMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = ANERMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = ANERMSNorm(hidden, eps=eps)

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,        # (1, L, H)
        cos: torch.Tensor,                  # (1, 1, L, head_dim)
        sin: torch.Tensor,                  # (1, 1, L, head_dim)
        attention_mask: torch.Tensor,       # (1, 1, L, L) fp16 additive (0 or −1e4)
        seq_len: int,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        # (B, H, 1, L) layout for Conv2d.
        x = normed.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        n_rep = num_heads // num_kv_heads

        # Q/K/V: (1, q_dim, 1, L) → (1, heads, L, head_dim).
        q = self.self_attn["q_proj"](x).view(1, num_heads, head_dim, seq_len).permute(0, 1, 3, 2)
        k = self.self_attn["k_proj"](x).view(1, num_kv_heads, head_dim, seq_len).permute(0, 1, 3, 2)
        v = self.self_attn["v_proj"](x).view(1, num_kv_heads, head_dim, seq_len).permute(0, 1, 3, 2)

        # QK norm per head.
        q = self.self_attn["q_norm"](q.reshape(1, num_heads, seq_len, head_dim))
        k = self.self_attn["k_norm"](k.reshape(1, num_kv_heads, seq_len, head_dim))

        # RoPE.
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA expansion (ANE-safe reshape+repeat+view; no repeat_interleave).
        k = repeat_kv_ane(k, n_rep, num_kv_heads, seq_len, head_dim)
        v = repeat_kv_ane(v, n_rep, num_kv_heads, seq_len, head_dim)

        # Attention in fp32 for numerical stability. No causal mask (encoder is
        # bidirectional); only the pad-mask + optional sliding-window mask.
        q_f = q.to(torch.float32)
        k_f = k.to(torch.float32)
        attn_weights = torch.matmul(q_f, k_f.transpose(-1, -2))  # scale=1.0 (QK norm)
        attn_weights = attn_weights + attention_mask.to(torch.float32)
        attn_weights = torch.softmax(attn_weights, dim=-1).to(MODEL_DTYPE)
        attn_out = torch.matmul(
            attn_weights.to(torch.float32), v.to(torch.float32)
        ).to(MODEL_DTYPE)

        # (1, heads, L, head_dim) → (1, L, heads*head_dim) → Conv2d o_proj.
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(1, seq_len, num_heads * head_dim)
        attn_out = self.self_attn["o_proj"](
            attn_out.permute(0, 2, 1).unsqueeze(2)
        ).squeeze(2).permute(0, 2, 1)

        attn_out = self.post_attention_layernorm(attn_out)
        hidden_states = residual + attn_out

        # MLP sandwich: pre_ff_ln → GeGLU → post_ff_ln.
        residual = hidden_states
        normed = self.pre_feedforward_layernorm(hidden_states)
        x_mlp = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        gate = self.mlp["gate_proj"](x_mlp)
        up = self.mlp["up_proj"](x_mlp)
        gate = F.gelu(gate, approximate="tanh")
        mlp_out = self.mlp["down_proj"](gate * up).squeeze(2).permute(0, 2, 1)
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        hidden_states = residual + mlp_out

        return hidden_states


class Gemma3Encoder(nn.Module):
    """Bidirectional Gemma 3 encoder backbone (no pooling, no projections).

    Input:
        input_ids      (1, L) int32
        attention_mask (1, L) fp16 — 1.0 for valid tokens, 0.0 for pad
    Output:
        hidden_states  (1, L, hidden_size) fp16
    """

    NEG_INF = -1.0e4  # ANE-safe additive-mask value (see METAL_ANE_KERNELS_CATALOG.md)

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = ANERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Precompute RoPE for max_seq_len (dual tables: local and global).
        self._build_rope(config)

        # Precompute a (L, L) sliding-window mask (bidirectional band of width
        # 2*sliding_window+1 centered on each position). 0 inside, NEG_INF outside.
        L = config.max_seq_len
        w = config.sliding_window
        idx = torch.arange(L)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (L, L)
        window_mask = torch.where(
            dist <= w,
            torch.zeros(L, L, dtype=MODEL_DTYPE),
            torch.full((L, L), self.NEG_INF, dtype=MODEL_DTYPE),
        )
        self.register_buffer("sliding_window_mask", window_mask)

    def _build_rope(self, config: EncoderConfig):
        head_dim = config.head_dim
        L = config.max_seq_len
        t = torch.arange(L).float()

        inv_local = 1.0 / (
            config.rope_local_base_freq
            ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        freqs_local = torch.einsum("i,j->ij", t, inv_local)
        emb_local = torch.cat((freqs_local, freqs_local), dim=-1)
        self.register_buffer("cos_local", emb_local.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_local", emb_local.sin().to(MODEL_DTYPE))

        inv_global = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        freqs_global = torch.einsum("i,j->ij", t, inv_global)
        emb_global = torch.cat((freqs_global, freqs_global), dim=-1)
        self.register_buffer("cos_global", emb_global.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_global", emb_global.sin().to(MODEL_DTYPE))

    def _make_attention_masks(self, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-token pad-masks combined with optional sliding-window masks.

        Returns (pad_mask_4d, sliding_mask_4d), each (1, 1, L, L) additive fp16.
        """
        L = self.config.max_seq_len
        # attention_mask: (1, L) where 1.0 = valid, 0.0 = pad.
        # Build (1, 1, 1, L) key-side pad mask: 0 on valid, −1e4 on pad.
        key_pad = (1.0 - attention_mask).to(MODEL_DTYPE) * self.NEG_INF
        key_pad = key_pad.view(1, 1, 1, L)  # broadcast over queries

        full_mask = key_pad.expand(1, 1, L, L)
        sliding_mask = full_mask + self.sliding_window_mask.view(1, 1, L, L)
        return full_mask, sliding_mask

    def forward(
        self,
        input_ids: torch.Tensor,          # (1, L) int32
        attention_mask: torch.Tensor,     # (1, L) fp16
    ) -> torch.Tensor:
        config = self.config
        L = config.max_seq_len

        hidden = self.embed_tokens(input_ids).to(MODEL_DTYPE)
        hidden = hidden * torch.tensor(config.hidden_size ** 0.5, dtype=MODEL_DTYPE)

        # RoPE tables are precomputed for positions 0..L-1 at trace time; view
        # them directly (index_select would be a no-op identity here).
        cos_local = self.cos_local.view(1, 1, L, config.head_dim)
        sin_local = self.sin_local.view(1, 1, L, config.head_dim)
        cos_global = self.cos_global.view(1, 1, L, config.head_dim)
        sin_global = self.sin_global.view(1, 1, L, config.head_dim)

        full_mask, sliding_mask = self._make_attention_masks(attention_mask)

        for layer in self.layers:
            if layer.is_full:
                hidden = layer(hidden, cos_global, sin_global, full_mask, L)
            else:
                hidden = layer(hidden, cos_local, sin_local, sliding_mask, L)

        return self.norm(hidden)
