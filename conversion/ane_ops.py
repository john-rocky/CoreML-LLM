"""ANE-optimized operations for CoreML LLM inference.

Provides drop-in replacements for standard PyTorch operations that are
optimized for Apple Neural Engine execution via CoreML:

- ANERMSNorm: RMSNorm using the [x, -x] concatenation trick
- Conv2dLinear: nn.Linear replacement using nn.Conv2d(kernel_size=1)
- InModelArgmax: Embeds argmax in the CoreML graph

Plus stateless factory helpers for the common builder boilerplate:
- conv_from_linear(lin, dtype=MODEL_DTYPE)
- ane_norm_from_hf(weight, eps, hidden, plus_one_gain=False)

Reference: ANEMLL project (github.com/Anemll/Anemll). Recipe lessons
distilled in `docs/MLSTATE_PATTERN.md` (single ct.StateType per chunk;
slice_update on int32 current_pos; KV-only MLState for hybrid models).

RMSNorm gain convention varies by model:
- Qwen3.5 / Llama-style: y = x * rsqrt(var+eps) * (1 + w)
  → call ane_norm_from_hf(weight, eps, hidden, plus_one_gain=True)
- Qwen3-VL / Gemma 4-style: y = x * rsqrt(var+eps) * w
  → call ane_norm_from_hf(weight, eps, hidden, plus_one_gain=False)
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_DTYPE = torch.float16

# Env toggle for the decomposed-vs-native softmax probe.  `ane_softmax`
# historically avoids the native softmax op (earlier iOS/chip rejected it
# on ANE) and decomposes into max/sub/exp/sum/div.  Set USE_NATIVE_SOFTMAX=1
# at conversion time to emit `F.softmax` instead — four fewer MIL ops per
# attention layer, speed impact is the subject of the A/B probe.
_USE_NATIVE_SOFTMAX = os.environ.get("USE_NATIVE_SOFTMAX") == "1"


class ANERMSNorm(nn.Module):
    """RMSNorm optimized for Apple Neural Engine.

    Standard RMSNorm uses rsqrt(mean(x^2)) which ANE cannot accelerate.
    Instead, we use the identity: RMSNorm(x) ≈ LayerNorm([x, -x])[:hidden_size]
    because cat([x, -x]) has zero mean, making LayerNorm equivalent to RMSNorm.
    ANE has a highly optimized LayerNorm kernel, so this runs fast.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        # fp16 weight to prevent fp32 upcast during multiply
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=MODEL_DTYPE))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # All fp16 for ANE compatibility (no float32 cast)
        # Matches ANEMLL's LlamaRMSNorm pattern.
        # 2026-05-06: tried fp32 cast inside this op to chase per-chunk
        # cosine vs HF; the cast is stripped by `compute_precision=FLOAT16`
        # at conversion time, so it had no effect. Genuine fp32 norm
        # requires either build-time op-selector to keep norm fp32 or
        # FLOAT32 compute_precision (slower / no ANE).
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * self.hidden_size,),
            weight=None,
            bias=None,
            eps=float(self.eps),
        )
        normed, _ = torch.chunk(normed, 2, dim=-1)
        return normed * self.weight


class Conv2dLinear(nn.Module):
    """Linear layer implemented as Conv2d(kernel_size=1) for ANE optimization.

    ANE processes Conv2d operations much more efficiently than linear layers.
    Weight shape: nn.Linear(in, out) uses (out, in)
                  nn.Conv2d(in, out, 1) uses (out, in, 1, 1)

    Input must be reshaped: (batch, seq, hidden) -> (batch, hidden, 1, seq)
    Output is reshaped back: (batch, out, 1, seq) -> (batch, seq, out)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = MODEL_DTYPE,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size=1, bias=bias, dtype=dtype
        )
        self.in_features = in_features
        self.out_features = out_features

    @classmethod
    def from_linear(cls, linear: nn.Linear,
                     dtype: torch.dtype | None = None) -> Conv2dLinear:
        """Convert an existing nn.Linear to Conv2dLinear.

        `dtype` defaults to the source linear's weight dtype. For ANE
        builders, pass `dtype=MODEL_DTYPE` (fp16) so the resulting
        Conv2d weight is fp16 regardless of the HF source precision.
        """
        target_dtype = dtype if dtype is not None else linear.weight.dtype
        has_bias = linear.bias is not None
        conv_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            dtype=target_dtype,
        )
        # Reshape: (out, in) -> (out, in, 1, 1)
        conv_linear.conv.weight.data = (
            linear.weight.detach().to(target_dtype).unsqueeze(-1).unsqueeze(-1)
        )
        if has_bias:
            conv_linear.conv.bias.data = linear.bias.detach().to(target_dtype)
        return conv_linear

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting Conv2d layout: (batch, channels, 1, seq)."""
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic layout conversion.

        Input: (batch, seq, features) -> Output: (batch, seq, features)
        """
        # (batch, seq, in) -> (batch, in, 1, seq)
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = self.conv(x)
        # (batch, out, 1, seq) -> (batch, seq, out)
        return x.squeeze(2).permute(0, 2, 1)


class InModelArgmax(nn.Module):
    """Embeds argmax into the CoreML graph to minimize ANE-to-host data transfer.

    Instead of outputting the full logits tensor (vocab_size can be 150K+),
    this computes argmax on-device and returns only the token index and its logit value.
    This dramatically reduces the data transfer from ANE to CPU.
    """

    def forward(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: (batch, seq, vocab_size) or (batch, vocab_size)

        Returns:
            token_id: (batch,) or (batch, seq) - argmax index
            token_logit: (batch,) or (batch, seq) - logit value at argmax
        """
        token_id = torch.argmax(logits, dim=-1)
        token_logit = logits.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
        return token_id, token_logit


# ---- factory helpers used by every chunked ANE builder ------------------


def conv_from_linear(lin: nn.Linear,
                      dtype: torch.dtype = MODEL_DTYPE) -> Conv2dLinear:
    """Wrap an HF nn.Linear as a Conv2dLinear at the given dtype (fp16
    by default for ANE).

    Equivalent to Conv2dLinear.from_linear with an explicit dtype.
    Provided as a free function so model builders can write a single
    `from ane_ops import conv_from_linear` instead of duplicating the
    boilerplate.
    """
    return Conv2dLinear.from_linear(lin, dtype=dtype)


def ane_norm_from_hf(weight: torch.Tensor, eps: float, hidden: int, *,
                      plus_one_gain: bool = False) -> ANERMSNorm:
    """Build an ANERMSNorm whose output matches an HF RMSNorm exactly.

    HF RMSNorm has two conventions in the wild:
      Qwen3.5 / Llama style:
        y = x * rsqrt(mean(x^2)+eps) * (1 + w)
        → set plus_one_gain=True so the wrapped ANERMSNorm.weight is
          stored as (1 + w).
      Qwen3-VL / Gemma 4 style:
        y = x * rsqrt(mean(x^2)+eps) * w
        → leave plus_one_gain=False (default).

    `hidden` is the size of the LAST dim that the norm operates on —
    i.e., the model's `hidden_size` for layer-level norms, or
    `head_dim` for per-head q_norm / k_norm.
    """
    n = ANERMSNorm(hidden, eps=eps)
    w = weight.detach().float()
    gain = (1.0 + w) if plus_one_gain else w
    n.weight.data = gain.to(MODEL_DTYPE).clone()
    return n


# ---- math primitives ----------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions: [x1, x2] -> [-x2, x1]."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def ane_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax using only ANE-friendly primitives.

    Default: decomposed form (max/sub/exp/sum/div) — avoids the native
    softmax op because earlier iOS/chip combinations rejected it on ANE.

    If USE_NATIVE_SOFTMAX=1, fall back to `F.softmax` — iOS 26 + ctools 9
    accepts the native op; probe whether the ANE runs it efficiently.
    """
    if _USE_NATIVE_SOFTMAX:
        return F.softmax(x.to(MODEL_DTYPE), dim=dim).to(MODEL_DTYPE)
    # Force fp16 throughout; torch.exp auto-upcasts without this.
    x = x.to(MODEL_DTYPE)
    x_max = x.max(dim=dim, keepdim=True).values.to(MODEL_DTYPE)
    x_shifted = (x - x_max).to(MODEL_DTYPE)
    exp_x = torch.exp(x_shifted).to(MODEL_DTYPE)
    exp_sum = exp_x.sum(dim=dim, keepdim=True).to(MODEL_DTYPE)
    return (exp_x / exp_sum).to(MODEL_DTYPE)


def repeat_kv_ane(hidden_states: torch.Tensor, n_rep: int,
                   num_kv_heads: int, seq_len: int, head_dim: int) -> torch.Tensor:
    """GQA repeat using reshape+repeat+view instead of repeat_interleave.

    ANEMLL pattern: shapes passed explicitly (no .shape access for trace).

    Input:  (1, num_kv_heads, seq, head_dim)
    Output: (1, num_kv_heads * n_rep, seq, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    # (1, kv, S, D) → (1, kv, 1, S, D) → (1, kv, rep, S, D) → (1, kv*rep, S, D)
    hidden_states = hidden_states.unsqueeze(2)
    hidden_states = hidden_states.repeat(1, 1, n_rep, 1, 1)
    return hidden_states.view(1, num_kv_heads * n_rep, seq_len, head_dim)


def repeat_kv(
    hidden_states: torch.Tensor, n_rep: int
) -> torch.Tensor:
    """Legacy GQA repeat. Use repeat_kv_ane for ANE."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def stable_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    causal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention in fp32 for numerical stability.

    Args:
        query: (batch, heads, q_len, head_dim)
        key:   (batch, heads, kv_len, head_dim)
        value: (batch, heads, kv_len, head_dim)
        scale: 1/sqrt(head_dim)
        causal_mask: (1, 1, q_len, kv_len) with -inf for masked positions

    Returns:
        (batch, heads, q_len, head_dim)
    """
    # Upcast to fp32 to avoid overflow in fp16 matmul
    q = query.to(torch.float32)
    k = key.to(torch.float32)

    attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scale

    if causal_mask is not None:
        attn_weights = attn_weights + causal_mask.to(torch.float32)

    # Softmax in fp32, then cast back
    attn_weights = torch.softmax(attn_weights, dim=-1).to(query.dtype)

    return torch.matmul(attn_weights.to(torch.float32), value.to(torch.float32)).to(
        query.dtype
    )
