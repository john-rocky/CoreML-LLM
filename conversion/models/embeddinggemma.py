"""EmbeddingGemma-300M: bidirectional Gemma-3 encoder → mean pool → 2 dense → L2.

Reference: arXiv:2509.20354 §3 — n=24 layers, dₘ=768, dᵤ=3072, d=768. Final
768-d vector can be Matryoshka-truncated to 512/256/128 post-hoc.

Runtime I/O contract (matches the exported mlpackage):
    input_ids       (1, L) int32
    attention_mask  (1, L) fp16 — 1.0 for valid tokens, 0.0 for pad
    embedding       (1, 768) fp16 — unit-norm; truncate the leading dim for
                                     Matryoshka-d ∈ {128, 256, 512, 768}
                                     then renormalize in Swift.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ane_ops import MODEL_DTYPE

from .gemma3_encoder import EncoderConfig, Gemma3Encoder


class EmbeddingGemmaModel(nn.Module):
    """Full EmbeddingGemma forward: tokens → 768-d unit-norm embedding."""

    # Matryoshka intermediate embedding dim (arXiv:2509.20354 §3).
    DENSE_INTERMEDIATE_DIM = 3072

    def __init__(self, encoder_config: EncoderConfig,
                 embed_dim: int = 768,
                 dense_intermediate_dim: int | None = None):
        super().__init__()
        self.encoder = Gemma3Encoder(encoder_config)
        self.embed_dim = embed_dim
        self.dense_intermediate_dim = dense_intermediate_dim or self.DENSE_INTERMEDIATE_DIM

        # Two dense projections (Conv2d(1x1) on (B, C, 1, 1) input = pooled vector).
        # Layer 2 in the SentenceTransformer module list: hidden → dense_inter.
        # Layer 3: dense_inter → embed_dim.
        self.dense1 = nn.Conv2d(
            encoder_config.hidden_size, self.dense_intermediate_dim, 1,
            bias=True, dtype=MODEL_DTYPE,
        )
        self.dense2 = nn.Conv2d(
            self.dense_intermediate_dim, embed_dim, 1,
            bias=True, dtype=MODEL_DTYPE,
        )
        # Conv2d defaults bias to uniform random; if the HF snapshot's Dense
        # modules were saved without bias (common for no-bias Linear layers),
        # we want zero bias, not random. Explicitly zero here so the loader
        # does the right thing whether or not a `*.bias` tensor is present.
        with torch.no_grad():
            self.dense1.bias.zero_()
            self.dense2.bias.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,        # (1, L) int32
        attention_mask: torch.Tensor,   # (1, L) fp16
    ) -> torch.Tensor:
        # 1) Encoder: (1, L, hidden) in fp32 (Gemma 3 residual stream, see
        #    gemma3_encoder.py for why).
        hidden_states = self.encoder(input_ids, attention_mask)

        # 2) Mean pooling over valid (non-pad) positions. Keep pool in fp32
        #    so overflow-safe; cast down to fp16 right before the dense layers.
        mask = attention_mask.to(torch.float32).unsqueeze(-1)
        masked = hidden_states * mask
        summed = masked.sum(dim=1, keepdim=False)                 # (1, hidden)
        denom = mask.sum(dim=1, keepdim=False).clamp_min(1.0)     # (1, 1)
        pooled = summed / denom                                    # (1, hidden)

        # 3) Two dense projections via Conv2d(1x1) on (1, hidden, 1, 1).
        x = pooled.to(MODEL_DTYPE).unsqueeze(-1).unsqueeze(-1)  # (1, hidden, 1, 1)
        x = self.dense1(x)                      # (1, dense_inter, 1, 1)
        x = self.dense2(x)                      # (1, embed_dim,  1, 1)
        pooled = x.view(1, self.embed_dim)

        # 4) L2 normalize (unit-norm embedding).
        #
        # Naïve `sqrt(sum(x²))` overflows fp16 in coremltools' lowering: a
        # 768-d vector with values up to ~150 produces sum-of-squares ~16M,
        # which exceeds fp16 max (65504) → sqrt(inf) = inf → x/inf = 0
        # (even though we cast to fp32 in PyTorch, the cast is collapsed
        # during fp16 lowering). Workaround: divide by max-abs first so
        # values land in [-1, 1] before squaring, making sum-of-squares
        # bounded by D. Mathematically identical to the standard L2 norm.
        eps_abs = torch.tensor(1.0e-6, dtype=MODEL_DTYPE)
        abs_max = pooled.abs().max(dim=-1, keepdim=True).values
        abs_max = torch.maximum(abs_max, eps_abs)
        scaled = pooled / abs_max                                 # in [-1, 1]
        sumsq = (scaled * scaled).sum(dim=-1, keepdim=True)       # ≤ D = 768
        inv_norm = torch.rsqrt(sumsq + 1.0e-12)                   # well-defined
        normalized = scaled * inv_norm                            # unit vector
        return normalized.to(MODEL_DTYPE)
