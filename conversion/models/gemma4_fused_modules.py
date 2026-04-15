"""Fused QKV and Gate/Up modules for Gemma 4 ANE optimization.

Drop-in replacements for the split q_proj/k_proj/v_proj and gate_proj/up_proj
projections in Gemma4DecoderLayer. Motivation per docs/GEMMA4_ANE_REWRITES.md
and docs/CONVERSION_AUDIT_2026_04_15.md:

  - Split QKV: 3 Conv2d dispatches per layer -> 1. Expected +8-12% decode.
  - Split Gate/Up: 2 Conv2d dispatches per layer -> 1. Expected +5-8% decode.

Both fuses also improve weight cache locality on ANE (packed tensors stay
co-resident in SRAM for the layer's working set).

These modules are NOT yet wired into the chunk builders (SWAChunk1..4,
MergedChunk12/34, etc.) — those modules directly reference
`layer.self_attn.q_proj.conv`, `layer.mlp.gate_proj.conv` and friends.
Wiring is deferred to a follow-up pass that:

  1. Replaces the ModuleDict construction in Gemma4DecoderLayer.__init__
     when `config.use_fused_projections` is True.
  2. Updates each chunk's _run_layer_* function to call the fused projections
     and split the output.
  3. Updates `_map_weight_name` in Gemma4Model to accept HF weights for
     q/k/v/gate/up and concatenate them into fused layouts at load time.

Until then, these classes serve as a reference implementation and a unit-
testable component.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ane_ops import MODEL_DTYPE


class FusedQKV(nn.Module):
    """Single Conv2d packing Q, K, V projections for GQA attention.

    Output channels = num_q_heads*head_dim + 2*num_kv_heads*head_dim.
    Caller slices along channel dim after forward.

    Layout: output channels are concatenated as [Q | K | V]. Slice boundaries:
        q_end = num_q_heads * head_dim
        k_end = q_end + num_kv_heads * head_dim
        v_end = k_end + num_kv_heads * head_dim
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        q_dim = num_q_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.fused = nn.Conv2d(
            hidden_size,
            q_dim + 2 * kv_dim,
            kernel_size=1,
            bias=bias,
            dtype=MODEL_DTYPE,
        )
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x: (B, hidden, 1, S). Returns (Q, K, V) each (B, ..., 1, S)."""
        packed = self.fused(x)
        q_end = self.q_dim
        k_end = q_end + self.kv_dim
        v_end = k_end + self.kv_dim
        q = packed[:, :q_end, :, :]
        k = packed[:, q_end:k_end, :, :]
        v = packed[:, k_end:v_end, :, :]
        return q, k, v

    @classmethod
    def from_split(
        cls,
        q_proj: nn.Conv2d,
        k_proj: nn.Conv2d,
        v_proj: nn.Conv2d,
    ) -> "FusedQKV":
        """Build a FusedQKV from three already-loaded split projections.

        Concatenates weights (and biases if present) along out_channels.
        """
        hidden_size = q_proj.in_channels
        head_q = q_proj.out_channels
        head_kv = k_proj.out_channels
        if v_proj.out_channels != head_kv:
            raise ValueError(f"k_proj and v_proj must match: {head_kv} vs {v_proj.out_channels}")

        # Build a new fused module with the right shape.
        # head_dim is encoded via num_q_heads; caller should know. We do not
        # require num_q_heads/num_kv_heads/head_dim separately because we
        # only need to know q_dim and kv_dim for slicing.
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        has_bias = q_proj.bias is not None
        m.fused = nn.Conv2d(
            hidden_size,
            head_q + 2 * head_kv,
            kernel_size=1,
            bias=has_bias,
            dtype=q_proj.weight.dtype,
        )
        m.q_dim = head_q
        m.kv_dim = head_kv
        m.hidden_size = hidden_size

        with torch.no_grad():
            w = torch.cat(
                [q_proj.weight.data, k_proj.weight.data, v_proj.weight.data],
                dim=0,
            )
            m.fused.weight.data.copy_(w)
            if has_bias:
                if k_proj.bias is None or v_proj.bias is None:
                    raise ValueError("bias mismatch across q/k/v")
                b = torch.cat(
                    [q_proj.bias.data, k_proj.bias.data, v_proj.bias.data], dim=0
                )
                m.fused.bias.data.copy_(b)
        return m


class FusedGateUp(nn.Module):
    """Single Conv2d packing gate and up projections for SwiGLU/GeGLU MLP.

    Output channels = 2 * intermediate_size. Caller splits to (gate, up) and
    applies activation to gate before elementwise mul with up.

    Gemma 4 uses GELU-tanh (hidden_activation="gelu_pytorch_tanh"), NOT SiLU.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.fused = nn.Conv2d(
            hidden_size,
            2 * intermediate_size,
            kernel_size=1,
            bias=bias,
            dtype=MODEL_DTYPE,
        )
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Input x: (B, hidden, 1, S). Returns (gate, up) each (B, intermediate, 1, S).

        Caller should apply activation to gate (e.g. F.gelu(gate, approximate='tanh')
        for Gemma 4) and multiply with up, then feed to down_proj.
        """
        packed = self.fused(x)
        gate = packed[:, : self.intermediate_size, :, :]
        up = packed[:, self.intermediate_size :, :, :]
        return gate, up

    @classmethod
    def from_split(
        cls,
        gate_proj: nn.Conv2d,
        up_proj: nn.Conv2d,
    ) -> "FusedGateUp":
        """Build a FusedGateUp from already-loaded split projections."""
        if gate_proj.in_channels != up_proj.in_channels:
            raise ValueError("gate and up must share in_channels")
        if gate_proj.out_channels != up_proj.out_channels:
            raise ValueError("gate and up must share out_channels (both = intermediate)")

        m = cls.__new__(cls)
        nn.Module.__init__(m)
        has_bias = gate_proj.bias is not None
        m.fused = nn.Conv2d(
            gate_proj.in_channels,
            2 * gate_proj.out_channels,
            kernel_size=1,
            bias=has_bias,
            dtype=gate_proj.weight.dtype,
        )
        m.intermediate_size = gate_proj.out_channels
        m.hidden_size = gate_proj.in_channels

        with torch.no_grad():
            w = torch.cat([gate_proj.weight.data, up_proj.weight.data], dim=0)
            m.fused.weight.data.copy_(w)
            if has_bias:
                if up_proj.bias is None:
                    raise ValueError("bias mismatch across gate/up")
                b = torch.cat([gate_proj.bias.data, up_proj.bias.data], dim=0)
                m.fused.bias.data.copy_(b)
        return m


def fuse_layer_projections(layer) -> None:
    """In-place add fused qkv and gate_up projections to a Gemma4DecoderLayer.

    Gemma4DecoderLayer stores projections directly as nn.Conv2d inside
    nn.ModuleDict (see gemma4.py:378-400), so we read the Conv2d modules
    directly (no `.conv` attribute indirection like Conv2dLinear would have).

    After calling, the layer gains attributes:
        layer.self_attn['qkv_fused']   (FusedQKV)
        layer.mlp['gate_up_fused']      (FusedGateUp)
    and still retains the split projections for backwards compat during
    migration. Chunk forward code can be updated incrementally.

    Usage:
        for layer in model.layers:
            fuse_layer_projections(layer)
        # Then convert. Chunk forwards must be updated to call qkv_fused /
        # gate_up_fused instead of split projections to realize the win.
    """
    q = layer.self_attn["q_proj"]
    k = layer.self_attn["k_proj"]
    v = layer.self_attn["v_proj"]
    layer.self_attn["qkv_fused"] = FusedQKV.from_split(q, k, v)

    gate = layer.mlp["gate_proj"]
    up = layer.mlp["up_proj"]
    layer.mlp["gate_up_fused"] = FusedGateUp.from_split(gate, up)
