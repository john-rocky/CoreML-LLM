"""ANE-optimized operations for CoreML LLM inference.

Provides drop-in replacements for standard PyTorch operations that are
optimized for Apple Neural Engine execution via CoreML:

- ANERMSNorm: RMSNorm using the [x, -x] concatenation trick
- Conv2dLinear: nn.Linear replacement using nn.Conv2d(kernel_size=1)
- InModelArgmax: Embeds argmax in the CoreML graph

Reference: ANEMLL project (github.com/Anemll/Anemll)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_DTYPE = torch.float16


class ANERMSNorm(nn.Module):
    """RMSNorm optimized for Apple Neural Engine.

    Standard RMSNorm uses rsqrt(mean(x^2)) which ANE cannot accelerate.
    Instead, we use the identity: RMSNorm(x) ≈ LayerNorm([x, -x])[:hidden_size]
    because cat([x, -x]) has zero mean, making LayerNorm equivalent to RMSNorm.
    ANE has a highly optimized LayerNorm kernel, so this runs fast.

    Args:
        hidden_size: Feature dimension being normalized.
        eps: RMSNorm epsilon.
        affine: If True (default), apply the learnable scale `weight` after
            normalization. If False, omit the scale multiply — caller is
            responsible for absorbing `self.weight` into the adjacent Conv2d
            weight (one elementwise op saved per layer; 35 ops model-wide).
            The `weight` parameter still exists (for load_state_dict
            compatibility) but is not applied at forward time.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        affine: bool = True,
    ) -> None:
        super().__init__()
        # fp16 weight to prevent fp32 upcast during multiply
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=MODEL_DTYPE))
        self.eps = eps
        self.hidden_size = hidden_size
        self.affine = affine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # All fp16 for ANE compatibility (no float32 cast)
        # Matches ANEMLL's LlamaRMSNorm pattern.
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * self.hidden_size,),
            weight=None,
            bias=None,
            eps=float(self.eps),
        )
        # Drop mirror half: use chunk to avoid dynamic slice during trace
        normed, _ = torch.chunk(normed, 2, dim=-1)
        if self.affine:
            return normed * self.weight
        return normed


def absorb_rmsnorm_scale_into_conv(
    norm: ANERMSNorm,
    conv: nn.Conv2d,
) -> None:
    """Fold `norm.weight` into `conv.weight` and disable affine on `norm`.

    Mathematically: RMSNorm(x) * w_norm -> Conv(y) is equivalent to
    RMSNorm(x) -> Conv(w_conv * w_norm). This saves one elementwise multiply
    op per RMSNorm-preceding-Conv pair, times N layers in a transformer.

    Only call this on pairs that are architecturally RMSNorm -> Conv2d(1x1)
    with no intermediate ops. The Conv weight shape is (out, in, 1, 1); we
    scale along `in` by broadcasting `w_norm` of shape (in,) to (1, in, 1, 1).

    Idempotent via the `norm.affine` flag: absorbing twice is a no-op because
    the second call sees affine=False.

    Args:
        norm: An ANERMSNorm whose `weight` should be absorbed.
        conv: A 1x1 Conv2d whose input channels equal norm.hidden_size.

    Raises:
        ValueError: if shapes do not match.
    """
    if not isinstance(norm, ANERMSNorm):
        raise ValueError(f"Expected ANERMSNorm, got {type(norm)}")
    if not isinstance(conv, nn.Conv2d) or conv.kernel_size != (1, 1):
        raise ValueError(f"Expected 1x1 Conv2d, got {type(conv)} {getattr(conv, 'kernel_size', None)}")
    if conv.in_channels != norm.hidden_size:
        raise ValueError(
            f"Conv in_channels {conv.in_channels} != RMSNorm hidden_size {norm.hidden_size}"
        )
    if not norm.affine:
        return  # already absorbed; idempotent
    with torch.no_grad():
        # conv.weight: (out, in, 1, 1); norm.weight: (in,) -> (1, in, 1, 1)
        scale = norm.weight.data.view(1, -1, 1, 1).to(conv.weight.dtype)
        conv.weight.data.mul_(scale)
        # Flag the norm as absorbed so forward() skips the multiply.
        norm.affine = False


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
    def from_linear(cls, linear: nn.Linear) -> Conv2dLinear:
        """Convert an existing nn.Linear to Conv2dLinear."""
        has_bias = linear.bias is not None
        conv_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            dtype=linear.weight.dtype,
        )
        # Reshape: (out, in) -> (out, in, 1, 1)
        conv_linear.conv.weight.data = linear.weight.data.unsqueeze(-1).unsqueeze(-1)
        if has_bias:
            conv_linear.conv.bias.data = linear.bias.data
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

    Avoids the softmax op entirely: max, sub, exp, sum, div.
    All casts explicit to prevent PyTorch fp16→fp32 auto-upcast in torch.exp.

    Historical context: in early coremltools, tracing `F.softmax` auto-upcast
    the input to fp32 at the softmax boundary, pushing the ANE compiler to
    emit an fp32-accumulating op. This manual decomposition forced the
    pipeline to stay in fp16.

    For current coremltools 9.0 / iOS 26, the MIL `softmax` op is fused and
    ANE-native when inputs are fp16. The decomposed form below produces
    ~8-12 MIL ops per call instead of 1 fused `softmax` op, and the extra
    ops are not always merged back by the optimizer. Use `ane_fused_softmax`
    below for new attention code; keep this function for backwards
    compatibility with existing chunks that rely on the decomposed form
    (see gemma4.py line 277-281: Q @ K^T fp16 overflow workaround).
    """
    # Force fp16 throughout; torch.exp auto-upcasts without this.
    x = x.to(MODEL_DTYPE)
    x_max = x.max(dim=dim, keepdim=True).values.to(MODEL_DTYPE)
    x_shifted = (x - x_max).to(MODEL_DTYPE)
    exp_x = torch.exp(x_shifted).to(MODEL_DTYPE)
    exp_sum = exp_x.sum(dim=dim, keepdim=True).to(MODEL_DTYPE)
    return (exp_x / exp_sum).to(MODEL_DTYPE)


def ane_fused_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Fused softmax that maps to the MIL `softmax` op on ANE.

    Uses F.softmax directly with explicit fp16 casts to prevent fp32 upcast
    during trace. Produces a single MIL `softmax` op vs `ane_softmax`'s
    ~8-12 decomposed ops. Per mlpackage structure audit, this is what
    mtp_drafter uses and it works on ANE.

    SAFETY NOTE: This cannot be a drop-in replacement for `ane_softmax` in
    all paths. Gemma 4's attention uses effective scale=1.0 (no 1/sqrt(d))
    to avoid fp16 overflow in Q @ K^T. If you swap softmax for the fused
    variant in attention, verify no NaN/Inf escape in the Q @ K^T
    intermediate by running test_merged_parity.py — the decomposed form is
    slightly more forgiving numerically because of the explicit fp16 casts
    at each step.

    Safe usage: logit softcapping, LM head softmax, non-attention reductions.
    Risky usage: attention softmax on long seq (>1024) at fp16.
    """
    # PyTorch's F.softmax upcasts to fp32 by default; force fp16 on input
    # and cast back to ensure the compiler sees a fp16->fp16 softmax op.
    x = x.to(MODEL_DTYPE)
    return F.softmax(x, dim=dim).to(MODEL_DTYPE)


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
