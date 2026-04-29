"""Abstract base classes for ANE-optimized Transformer models.

Provides reusable building blocks that concrete model implementations
(Qwen2, Qwen3, LLaMA, etc.) extend with architecture-specific details.

All layers use Conv2d for ANE optimization and ANERMSNorm for normalization.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ane_ops import (
    MODEL_DTYPE,
    ANERMSNorm,
    Conv2dLinear,
    InModelArgmax,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
    stable_attention,
)


@dataclass
class ModelConfig:
    """Configuration for a Transformer LLM."""

    architectures: list[str] = field(default_factory=lambda: ["LlamaForCausalLM"])
    hidden_size: int = 896
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864
    vocab_size: int = 151936
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True
    attention_bias: bool = True
    hidden_act: str = "silu"
    # Qwen3 / some newer Qwen variants apply per-head RMSNorm to Q and K
    # before RoPE. Leave off by default to keep existing Qwen2 / Gemma conversions unchanged.
    has_qk_norm: bool = False
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    # Conversion settings (not from HF config)
    context_length: int = 2048
    state_length: int = 2048

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        cfg = cls(**filtered)
        # Compute head_dim if not explicitly provided
        if "head_dim" not in d:
            cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
        return cfg

    @classmethod
    def from_json(cls, path: str) -> ModelConfig:
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dim = config.head_dim
        base = config.rope_theta

        if config.rope_scaling is not None:
            original_max = config.rope_scaling.get("original_max_position_embeddings")
            if original_max is not None and config.context_length > original_max:
                import warnings
                warnings.warn(
                    f"rope_scaling={config.rope_scaling} ignored (YaRN not implemented). "
                    f"context_length={config.context_length} exceeds original {original_max}; "
                    f"RoPE will extrapolate and quality may degrade.",
                    stacklevel=2,
                )

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        max_len = max(config.context_length, config.state_length) * 2
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0))

    def forward_single(self, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin for a single position. Shape: (1, 1, 1, head_dim)."""
        cos = self.cos_cached[:, pos].view(1, 1, 1, -1).to(MODEL_DTYPE)
        sin = self.sin_cached[:, pos].view(1, 1, 1, -1).to(MODEL_DTYPE)
        return cos, sin

    def forward_range(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin for a range of positions.

        Args:
            positions: (seq_len,) tensor of position indices
        Returns:
            cos, sin: (1, seq_len, 1, head_dim)
        """
        seq_len = positions.size(0)
        cos = self.cos_cached[:, positions].view(1, seq_len, 1, self.dim).to(MODEL_DTYPE)
        sin = self.sin_cached[:, positions].view(1, seq_len, 1, self.dim).to(MODEL_DTYPE)
        return cos, sin


class ANEAttention(nn.Module):
    """Multi-head attention with GQA support, optimized for ANE via Conv2d."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        has_bias = config.attention_bias

        self.q_proj = nn.Conv2d(config.hidden_size, q_dim, 1, bias=has_bias, dtype=MODEL_DTYPE)
        self.k_proj = nn.Conv2d(config.hidden_size, kv_dim, 1, bias=has_bias, dtype=MODEL_DTYPE)
        self.v_proj = nn.Conv2d(config.hidden_size, kv_dim, 1, bias=has_bias, dtype=MODEL_DTYPE)
        self.o_proj = nn.Conv2d(q_dim, config.hidden_size, 1, bias=False, dtype=MODEL_DTYPE)

        self.has_qk_norm = config.has_qk_norm
        if self.has_qk_norm:
            self.q_norm = ANERMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = ANERMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(config)

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to Q, K, V using Conv2d.

        Input: (batch, seq, hidden)
        Returns Q: (batch, num_heads, seq, head_dim)
                K: (batch, num_kv_heads, seq, head_dim)
                V: (batch, num_kv_heads, seq, head_dim)
        """
        batch, seq_len, _ = hidden_states.shape
        # (batch, hidden, 1, seq) for Conv2d
        x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        q = self.q_proj(x).view(batch, self.num_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        k = self.k_proj(x).view(batch, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        v = self.v_proj(x).view(batch, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)

        if self.has_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        return q.to(MODEL_DTYPE), k.to(MODEL_DTYPE), v.to(MODEL_DTYPE)

    def _output_proj(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Project attention output back to hidden_size.

        Input: (batch, num_heads, seq, head_dim)
        Output: (batch, seq, hidden)
        """
        batch = attn_output.shape[0]
        seq_len = attn_output.shape[2]
        # (batch, heads, seq, dim) -> (batch, seq, heads*dim)
        x = attn_output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, -1)
        # Conv2d projection
        x = self.o_proj(x.permute(0, 2, 1).unsqueeze(2))
        return x.squeeze(2).permute(0, 2, 1)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        current_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token decode with KV cache.

        Args:
            hidden_states: (1, 1, hidden_size)
            kv_cache: (K, V) each (1, num_kv_heads, state_length, head_dim)
            current_pos: current position index

        Returns:
            output: (1, 1, hidden_size)
            new_k: (1, num_kv_heads, 1, head_dim)
            new_v: (1, num_kv_heads, 1, head_dim)
        """
        q, k, v = self._project_qkv(hidden_states)

        cos, sin = self.rotary_emb.forward_single(current_pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        K_cache, V_cache = kv_cache
        K_full = K_cache[:, :, : self.config.state_length, :]
        V_full = V_cache[:, :, : self.config.state_length, :]

        K_expanded = repeat_kv(K_full, self.n_rep)
        V_expanded = repeat_kv(V_full, self.n_rep)

        # Build causal mask: attend only to positions <= current_pos
        causal_mask = torch.full(
            (1, 1, 1, self.config.state_length),
            float("-inf"),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        causal_mask[:, :, :, : current_pos + 1] = 0.0

        attn_output = stable_attention(q, K_expanded, V_expanded, self.scale, causal_mask)
        output = self._output_proj(attn_output)

        return output, k, v

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched prefill with KV cache.

        Args:
            hidden_states: (1, seq_len, hidden_size)
            kv_cache: (K, V) each (1, num_kv_heads, state_length, head_dim)
            positions: (seq_len,) position indices
            causal_mask: (1, 1, seq_len, state_length)

        Returns:
            output: (1, seq_len, hidden_size)
            new_k: (1, num_kv_heads, seq_len, head_dim)
            new_v: (1, num_kv_heads, seq_len, head_dim)
        """
        q, k, v = self._project_qkv(hidden_states)

        cos, sin = self.rotary_emb.forward_range(positions)
        # Reshape for attention: (1, seq, 1, dim) -> (1, 1, seq, dim)
        cos = cos.permute(0, 2, 1, 3)
        sin = sin.permute(0, 2, 1, 3)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        K_cache, V_cache = kv_cache
        K_full = K_cache[:, :, : self.config.state_length, :]
        V_full = V_cache[:, :, : self.config.state_length, :]

        K_expanded = repeat_kv(K_full, self.n_rep)
        V_expanded = repeat_kv(V_full, self.n_rep)

        attn_output = stable_attention(q, K_expanded, V_expanded, self.scale, causal_mask)
        output = self._output_proj(attn_output)

        return output, k, v


class ANEMLP(nn.Module):
    """SiLU-gated MLP using Conv2d for ANE optimization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(
            config.hidden_size, config.intermediate_size, 1, bias=False, dtype=MODEL_DTYPE
        )
        self.up_proj = nn.Conv2d(
            config.hidden_size, config.intermediate_size, 1, bias=False, dtype=MODEL_DTYPE
        )
        self.down_proj = nn.Conv2d(
            config.intermediate_size, config.hidden_size, 1, bias=False, dtype=MODEL_DTYPE
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input/Output: (batch, seq, hidden)."""
        x = x.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        a = self.gate_proj(x)
        b = self.up_proj(x)
        c = F.silu(a) * b
        d = self.down_proj(c)
        return d.squeeze(2).permute(0, 2, 1)


class ANEDecoderLayer(nn.Module):
    """Single Transformer decoder layer with pre-norm (ANE-optimized)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = ANEAttention(config)
        self.mlp = ANEMLP(config)
        self.input_layernorm = ANERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ANERMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class ANETransformerModel(nn.Module, ABC):
    """Abstract base for a full decoder-only Transformer with KV cache.

    Subclasses must implement:
    - load_weights(model_path): Load weights from HuggingFace safetensors
    - weight_map(): Return mapping from HF weight names to local parameter names
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [ANEDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final norm
        self.norm = ANERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM Head
        self.lm_head = nn.Conv2d(
            config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE
        )

        # Argmax
        self.argmax = InModelArgmax()

        # Unified KV cache: (2*num_layers, num_kv_heads, state_length, head_dim)
        # First half is K, second half is V (interleaved by layer)
        cache_shape = (
            2 * config.num_hidden_layers,
            config.num_key_value_heads,
            config.state_length,
            config.head_dim,
        )
        self.register_buffer(
            "kv_cache_0", torch.zeros(cache_shape, dtype=MODEL_DTYPE)
        )

    def get_kv_cache_for_layer(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get K and V cache slices for a given layer."""
        k_idx = layer_idx
        v_idx = self.config.num_hidden_layers + layer_idx
        return (
            self.kv_cache_0[k_idx : k_idx + 1],
            self.kv_cache_0[v_idx : v_idx + 1],
        )

    def update_kv_cache(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        pos: int,
        seq_len: int = 1,
    ) -> None:
        """Write new K/V values into the cache at the given position."""
        k_idx = layer_idx
        v_idx = self.config.num_hidden_layers + layer_idx
        self.kv_cache_0[k_idx, :, pos : pos + seq_len, :] = new_k.squeeze(0)
        self.kv_cache_0[v_idx, :, pos : pos + seq_len, :] = new_v.squeeze(0)

    def forward_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Part 1: Token IDs -> hidden states.

        Input: (batch, seq_len) int64
        Output: (batch, seq_len, hidden_size)
        """
        return self.embed_tokens(input_ids).to(MODEL_DTYPE)

    def forward_transformer_decode(
        self,
        hidden_states: torch.Tensor,
        current_pos: int,
    ) -> torch.Tensor:
        """Part 2 (decode): Single token through all layers with KV cache update.

        Input: (1, 1, hidden_size)
        Output: (1, 1, hidden_size)
        """
        for layer_idx, layer in enumerate(self.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            kv_cache = self.get_kv_cache_for_layer(layer_idx)
            hidden_states, new_k, new_v = layer.self_attn.forward_decode(
                hidden_states, kv_cache, current_pos
            )
            self.update_kv_cache(layer_idx, new_k, new_v, current_pos)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return self.norm(hidden_states)

    def forward_transformer_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Part 2 (prefill): Batch of tokens through all layers.

        Input: (1, seq_len, hidden_size)
        Output: (1, seq_len, hidden_size)
        """
        seq_len = hidden_states.shape[1]
        start_pos = positions[0].item()

        for layer_idx, layer in enumerate(self.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            kv_cache = self.get_kv_cache_for_layer(layer_idx)
            hidden_states, new_k, new_v = layer.self_attn.forward_prefill(
                hidden_states, kv_cache, positions, causal_mask
            )
            self.update_kv_cache(layer_idx, new_k, new_v, start_pos, seq_len)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return self.norm(hidden_states)

    def forward_lm_head(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Part 3: Hidden states -> token prediction (argmax).

        Input: (batch, seq_len, hidden_size)
        Output: (token_id, logit_value)
        """
        # Only take the last token
        last = hidden_states[:, -1:, :]
        x = last.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        return self.argmax(logits.squeeze(0))

    @abstractmethod
    def load_weights(self, model_path: str) -> None:
        """Load pretrained weights from a HuggingFace model directory."""
        ...

    @abstractmethod
    def weight_map(self) -> dict[str, str]:
        """Return a mapping from HuggingFace weight names to local param names."""
        ...
