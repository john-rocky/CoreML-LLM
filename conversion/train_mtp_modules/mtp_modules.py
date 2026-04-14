#!/usr/bin/env python3
"""Path C: Self-trained MTP modules (sequential, DeepSeek V3 style).

Each module is a single Gemma-style transformer block trained against OUR
frozen Gemma 4 E2B trunk's L34 hidden state and token embeddings. K=2 depth
for v1 (DeepSeek V3 paper's 2-token MTP, validated 85-90% acc on module_2).

Architecture:
  input_1 = hidden_from_prev  # L34[t] for module_1, h_{k-1} for module_k (k>=2)
  input_2 = embed(tok)        # scaled (× sqrt(hidden_size))
  x = input_proj(concat(input_1, input_2))     # (3072 → 1536)
  # Sandwich norm Gemma block:
  x = x + post_attn_norm(self_attn(input_norm(x)))
  x = x + post_ffw_norm(mlp(pre_ffw_norm(x)))
  h = final_norm(x)
  logits = lm_head(h)  # tied with trunk
  return h, logits  # h → next module's input_1

Design choices for v1:
  - NO target KV reuse (modules have own small KV cache)
  - Top-only input (L34 hidden). Multi-layer fusion deferred to v1.5.
  - Shared embedding + shared LM head (frozen, from trunk).
  - Own KV cache with small window (128 tokens) per module.
  - In-model top-k(8) for logits output (same as Google drafter).

Total trainable per module: ~50M (attn + MLP, 1 layer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MtpModuleConfig:
    hidden_size: int = 1536        # target trunk hidden
    intermediate_size: int = 6144  # Gemma 4 FFN intermediate (SWA layer width)
    num_attention_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256            # SWA head_dim (drafter inherits SWA layer shape)
    vocab_size: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    kv_window: int = 128           # module's own KV cache window
    logit_softcap: float = 30.0    # same as trunk
    num_modules: int = 2           # K=2 for v1 (DeepSeek V3 style)


class RMSNorm(nn.Module):
    """Gemma-style RMSNorm (raw weight, not `1 + w`)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


def _precompute_rope(head_dim: int, max_pos: int, theta: float):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)  # (max_pos, half)


def _apply_rope_q(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """q: (B, H, T, D); cos/sin: (T, D/2). Returns same shape."""
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    cos = cos.view(1, 1, cos.shape[0], half)
    sin = sin.view(1, 1, sin.shape[0], half)
    return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)


class ModuleAttention(nn.Module):
    """Self-attention with own small KV cache. Q-K-V all computed from module input.

    During training: T-position batched, standard attention.
    During inference: T=1 per module call, KV cache updated in place.
    """
    def __init__(self, cfg: MtpModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_kv_heads
        self.hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.scale = 1.0 / math.sqrt(self.hd)

    def forward(
        self,
        x: torch.Tensor,             # (B, T, hidden)
        positions: torch.Tensor,     # (T,) absolute positions for RoPE
        cos: torch.Tensor, sin: torch.Tensor,  # (max_pos, D/2)
        causal_mask: torch.Tensor,   # (B, 1, T, T) fp with -inf for masked
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B, nh, T, hd)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # RoPE — cast cos/sin to q's dtype to avoid auto-promotion mismatch with v
        cos_pos = cos[positions].to(q.dtype)
        sin_pos = sin[positions].to(q.dtype)
        q = _apply_rope_q(q, cos_pos, sin_pos)
        k = _apply_rope_q(k, cos_pos, sin_pos)

        # GQA broadcast
        if self.nh != self.nkv:
            n_rep = self.nh // self.nkv
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Ensure consistent dtype throughout attention
        attn_dtype = q.dtype
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)
        attn = attn + causal_mask.to(attn_dtype)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.to(attn_dtype))  # (B, nh, T, hd)

        out = out.permute(0, 2, 1, 3).reshape(B, T, self.nh * self.hd)
        return self.o_proj(out)


class ModuleMLP(nn.Module):
    """GeGLU MLP — identical to Gemma 4 SWA layer."""
    def __init__(self, cfg: MtpModuleConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class MtpModule(nn.Module):
    """Single MTP module: Gemma transformer block + input fusion."""
    def __init__(self, cfg: MtpModuleConfig):
        super().__init__()
        self.cfg = cfg
        # Input fusion: concat(hidden_prev, embed) → hidden
        # hidden_prev is L34 hidden (for module_1) or module_{k-1} output (for k>=2)
        self.input_proj = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False)

        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        self.attn = ModuleAttention(cfg)
        self.mlp = ModuleMLP(cfg)
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        hidden_prev: torch.Tensor,   # (B, T, hidden)
        token_embed: torch.Tensor,   # (B, T, hidden)
        positions: torch.Tensor,     # (T,)
        cos: torch.Tensor, sin: torch.Tensor,
        causal_mask: torch.Tensor,   # (B, 1, T, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (hidden_out, logits_fp32_pre_softcap).

        hidden_out feeds into next module. logits go through tied LM head + softcap
        at the training/inference driver level.
        """
        x = self.input_proj(torch.cat([hidden_prev, token_embed], dim=-1))

        residual = x
        h = self.input_layernorm(x)
        h = self.attn(h, positions, cos, sin, causal_mask)
        h = self.post_attention_layernorm(h)
        x = residual + h

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = residual + h

        h_out = self.final_norm(x)
        return h_out


class MtpStack(nn.Module):
    """K modules stacked. Shares RoPE tables, shared LM head (tied to trunk).

    Training-time entry point. Inference-time is the CoreML-converted version
    which runs one module per CoreML predict() call.
    """
    def __init__(self, cfg: MtpModuleConfig, lm_head_weight: torch.Tensor,
                 embed_weight: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.modules_list = nn.ModuleList([MtpModule(cfg) for _ in range(cfg.num_modules)])

        # Tied embeddings (frozen — from trunk)
        self.register_buffer("embed_weight", embed_weight, persistent=False)  # (V, H)
        self.register_buffer("lm_head_weight", lm_head_weight, persistent=False)  # (V, H)

        # RoPE
        max_pos = 4096  # drafter's own positional reference; can be small
        cos, sin = _precompute_rope(cfg.head_dim, max_pos, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Embed scale (Gemma convention)
        self.embed_scale = math.sqrt(cfg.hidden_size)

    def _scaled_embed(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.embed_weight) * self.embed_scale

    def _lm_head(self, h: torch.Tensor) -> torch.Tensor:
        """Apply tied LM head + softcap."""
        logits = F.linear(h.float(), self.lm_head_weight.float())
        return torch.tanh(logits / self.cfg.logit_softcap) * self.cfg.logit_softcap

    def forward(
        self,
        l34_hidden: torch.Tensor,      # (B, T, H) from frozen trunk
        token_ids: torch.Tensor,       # (B, T+K) — current + K draft targets
    ) -> list[torch.Tensor]:
        """Training-time forward. Returns list of logits, one per module.

        module_k predicts token at position t+k+1, given:
          - hidden_prev = L34[t] (k=1) or h_{k-1} (k>=2)
          - embed(tok_t+k-1)
        """
        B, T, _ = l34_hidden.shape
        K = self.cfg.num_modules

        assert token_ids.shape[1] >= T + K, \
            f"Need T+K={T+K} tokens for K-depth MTP, got {token_ids.shape[1]}"

        # Cast l34 to match module parameter dtype (modules may be bf16/fp16
        # while precomputed hiddens on disk were fp32-loaded).
        module_dtype = next(self.modules_list.parameters()).dtype
        l34_hidden = l34_hidden.to(module_dtype)

        positions = torch.arange(T, device=l34_hidden.device)
        causal_mask = torch.zeros(B, 1, T, T, device=l34_hidden.device, dtype=torch.float32)
        # Standard causal: mask future positions
        causal_mask = causal_mask + torch.triu(
            torch.full((T, T), float("-inf"), device=l34_hidden.device), diagonal=1
        )

        logits_per_module = []
        h_prev = l34_hidden
        for k in range(K):
            # Input token for module_k: tok_t+k (shifted by k from position t)
            tok_k = token_ids[:, k:k + T]            # (B, T)
            emb_k = self._scaled_embed(tok_k)        # (B, T, H)

            h_out = self.modules_list[k](
                h_prev, emb_k, positions, self.cos, self.sin, causal_mask,
            )
            logits_k = self._lm_head(h_out)
            logits_per_module.append(logits_k)
            h_prev = h_out

        return logits_per_module


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    cfg = MtpModuleConfig()
    print(f"Config: {cfg}")

    # Mock tied weights (normally loaded from trunk)
    embed_weight = torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.02
    lm_head_weight = embed_weight.clone()  # tied

    stack = MtpStack(cfg, lm_head_weight, embed_weight)
    print(f"Trainable params per module: {count_params(stack.modules_list[0]):,}")
    print(f"Total trainable params: {count_params(stack):,}")

    # Forward smoke
    B, T = 2, 16
    l34 = torch.randn(B, T, cfg.hidden_size)
    tokens = torch.randint(0, cfg.vocab_size, (B, T + cfg.num_modules))

    logits_list = stack(l34, tokens)
    for k, logits in enumerate(logits_list):
        print(f"  module_{k+1} logits: {tuple(logits.shape)} "
              f"argmax_sample: {logits[0, 0].argmax().item()}")
    print("Smoke test passed.")
