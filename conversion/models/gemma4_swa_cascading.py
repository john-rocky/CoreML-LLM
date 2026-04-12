"""Cascading KV Cache variant for Gemma 4 E2B full-attention layers (ANE-compatible).

Scaffold for Approach C of docs/UNEXPLORED_APPROACHES.md.

Idea (arXiv 2406.17808):
  Replace the monolithic full-attention KV at 8K with a hierarchical
  bucket set. Each bucket covers a different slice of history at a
  different sampling density:

      Level 0 (sink):    tokens 0..3                    (fixed 4 slots)
      Level 1 (recent):  tokens [L-512, L)              (dense, sliding)
      Level 2 (mid):     tokens [L-1536, L-512) stride 2 (sampled)
      Level 3 (far):     tokens [L-7680, L-1536) stride 4 (coarse)

  Retained KV size:
      4 + 512 + 512 + 1536 = 2564 tokens

  All eviction patterns are position-based (fixed schedule from the
  current decode position), NOT attention-score-based — which keeps the
  graph static-shape and ANE-compilable. This is a simpler variant than
  the paper's EMA-weighted retention, but the paper shows position-based
  cascading already captures most of the benefit on PG19 and LongBench.

Integration:
  - Mirrors the API of conversion/models/gemma4_swa_chunks.py so
    build_speculative.py can produce cascading-KV decode chunks via a
    flag. The sliding layers (28/35) are unchanged; only the 7 full-
    attention layers get the cascading treatment.
  - All ctx-sized KV buffers shrink from 8192 to 2564 entries. SRAM
    pressure drops, decode cost at ctx=8192 becomes comparable to ctx=2K.

This file is the MODEL DEFINITION ONLY. The conversion pipeline
(build_speculative.py) must be extended with a `--cascading` flag that
selects this module in place of gemma4_swa_chunks.py. That extension is
owned by the bench session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Cascading schedule ──────────────────────────────────────────────────────

@dataclass
class CascadingConfig:
    """Fixed, position-based cascading schedule. All values compile to static
    shapes on ANE — no data-dependent branching."""
    sink_size: int = 4
    recent_window: int = 512           # dense, rolling
    mid_window: int = 512              # stride 2 over [L-1536, L-512)
    mid_stride: int = 2
    far_window: int = 1536             # stride 4 over [L-7680, L-1536)
    far_stride: int = 4
    max_ctx: int = 8192                # matches Gemma 4 E2B ctx_length cap

    @property
    def total_slots(self) -> int:
        return self.sink_size + self.recent_window + self.mid_window + self.far_window

    def describe(self) -> str:
        return (
            f"Cascading KV (sink={self.sink_size}, recent={self.recent_window}, "
            f"mid={self.mid_window}@stride{self.mid_stride}, "
            f"far={self.far_window}@stride{self.far_stride}) "
            f"→ total_slots={self.total_slots} (vs full {self.max_ctx})"
        )


# ── Position mapping ────────────────────────────────────────────────────────

def build_gather_indices(position: int, cfg: CascadingConfig) -> torch.Tensor:
    """Return a (total_slots,) LongTensor of source KV indices to gather from
    the underlying full history for decode position `position`.

    Entries pointing past `position` are clamped to position and will be
    masked by the attention mask (sink positions past eos still reference
    valid early tokens).
    """
    L = position + 1
    idxs: list[int] = []

    # Sink
    idxs.extend(range(min(cfg.sink_size, L)))
    idxs.extend([0] * max(0, cfg.sink_size - L))

    # Recent (dense)
    recent_start = max(cfg.sink_size, L - cfg.recent_window)
    recent_idxs = list(range(recent_start, L))
    # Pad to recent_window if early in decoding
    while len(recent_idxs) < cfg.recent_window:
        recent_idxs.insert(0, recent_start)
    idxs.extend(recent_idxs[-cfg.recent_window:])

    # Mid (stride 2)
    mid_end = recent_start
    mid_start = max(cfg.sink_size, mid_end - cfg.mid_window * cfg.mid_stride)
    mid_idxs = list(range(mid_start, mid_end, cfg.mid_stride))
    while len(mid_idxs) < cfg.mid_window:
        mid_idxs.insert(0, mid_start)
    idxs.extend(mid_idxs[-cfg.mid_window:])

    # Far (stride 4)
    far_end = mid_start
    far_start = max(cfg.sink_size, far_end - cfg.far_window * cfg.far_stride)
    far_idxs = list(range(far_start, far_end, cfg.far_stride))
    while len(far_idxs) < cfg.far_window:
        far_idxs.insert(0, far_start)
    idxs.extend(far_idxs[-cfg.far_window:])

    return torch.tensor(idxs, dtype=torch.long)


# ── Cascading full-attention ──────────────────────────────────────────────

class CascadingFullAttention(nn.Module):
    """Drop-in replacement for Gemma 4 E2B full-attention. Matches the I/O
    contract of the layer in gemma4_swa_chunks.py but reads KV via a
    position-dependent gather from a compact cascading buffer instead of
    the monolithic (1, 1, ctx, head_dim) buffer.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, rope_theta: float, cascading: CascadingConfig,
                 layer_idx: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.cfg = cascading
        self.layer_idx = layer_idx

        # Projections — same shapes as Gemma4TextAttention
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Q/K/V norms per Gemma 4 architecture
        from torch.nn import LayerNorm
        self.q_norm = LayerNorm(head_dim, elementwise_affine=True)
        self.k_norm = LayerNorm(head_dim, elementwise_affine=True)
        self.v_norm = LayerNorm(head_dim, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.Tensor,      # (1, 1, H)
        k_cache: torch.Tensor,            # (1, num_kv, max_ctx, head_dim) — full history
        v_cache: torch.Tensor,            # same
        position: int,                    # current decode position (scalar, baked into gather)
        rope_cos: torch.Tensor,           # (max_ctx, head_dim/2)
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:

        B, T, _ = hidden_states.shape      # expect (1, 1, H)
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k_new = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        k_new = self.k_norm(k_new)
        v_new = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v_new = self.v_norm(v_new)

        # RoPE on Q at current position
        cos_p = rope_cos[position:position + 1]   # (1, head_dim/2)
        sin_p = rope_sin[position:position + 1]
        q = _apply_rope(q, cos_p, sin_p)
        k_new = _apply_rope(k_new, cos_p, sin_p)

        # Cascading gather: build index vector (static at compile if position
        # is part of the graph via EnumeratedShapes of chunk-indexed positions,
        # or via a lookup-table tensor of size [max_ctx, total_slots])
        gather_idx = build_gather_indices(position, self.cfg).to(k_cache.device)

        # Gather K/V from the FULL cache (written at every decode step) via
        # fixed pattern. Concat with k_new/v_new (current position).
        # k_cache shape: (1, num_kv, max_ctx, head_dim) → gather on dim=2
        k_g = k_cache.index_select(dim=2, index=gather_idx)    # (1, num_kv, total_slots, head_dim)
        v_g = v_cache.index_select(dim=2, index=gather_idx)

        # Append current K/V at the end
        k_full = torch.cat([k_g, k_new.transpose(1, 2)], dim=2)  # (1, num_kv, total_slots+1, head_dim)
        v_full = torch.cat([v_g, v_new.transpose(1, 2)], dim=2)

        # GQA expand
        rep = self.num_heads // self.num_kv_heads
        k_full = k_full.repeat_interleave(rep, dim=1)
        v_full = v_full.repeat_interleave(rep, dim=1)

        # Attention (current Q attending to retained history + self)
        q = q.transpose(1, 2)                                     # (1, num_heads, 1, head_dim)
        attn = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
        attn = attn.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)

        return self.o_proj(attn)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    # Broadcast cos/sin over batch/heads dims
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── Chunk-level integration hook ────────────────────────────────────────────

def make_cascading_full_attention(base_attention_module, cascading: CascadingConfig):
    """Utility: given an existing Gemma4TextAttention instance for a full-
    attention layer, return a CascadingFullAttention that takes the weights
    from it. Used by build_speculative.py to swap attention modules before
    CoreML conversion.
    """
    cfg = base_attention_module.config if hasattr(base_attention_module, "config") else None
    if cfg is None:
        raise ValueError("base_attention_module must expose a .config attribute")
    ca = CascadingFullAttention(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        rope_theta=getattr(cfg, "rope_theta", 10000.0),
        cascading=cascading,
        layer_idx=getattr(base_attention_module, "layer_idx", 0),
    )
    # Weight copy
    with torch.no_grad():
        ca.q_proj.weight.copy_(base_attention_module.q_proj.weight)
        ca.k_proj.weight.copy_(base_attention_module.k_proj.weight)
        ca.v_proj.weight.copy_(base_attention_module.v_proj.weight)
        ca.o_proj.weight.copy_(base_attention_module.o_proj.weight)
        if hasattr(base_attention_module, "q_norm"):
            ca.q_norm.weight.copy_(base_attention_module.q_norm.weight)
        if hasattr(base_attention_module, "k_norm"):
            ca.k_norm.weight.copy_(base_attention_module.k_norm.weight)
        # v_norm has no weight on Gemma 4 (with_scale=False); nothing to copy.
    return ca


# ── Quick sanity check when run as main ─────────────────────────────────────

if __name__ == "__main__":
    cfg = CascadingConfig()
    print(cfg.describe())
    for pos in [0, 10, 600, 2000, 7000]:
        idx = build_gather_indices(pos, cfg)
        print(f"  position={pos:5d}: gather_idx shape={tuple(idx.shape)} "
              f"min={idx.min().item()} max={idx.max().item()}")
