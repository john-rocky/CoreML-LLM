"""ANE-optimized decode layer for Qwen3.5 hybrid SSM + full_attention stack.

Drop-in alternative to `test_qwen3_5_full_decode_trace.DecoderDecodeLayer`
with the Gemma 4 ANE recipe applied (proven 11→31 tok/s on Qwen3-VL 4B
chunked path; previously reverted on Qwen3.5 monolithic build at 5be231b
because the 24-layer graph hit the iOS 26.1 BNNS/ANEF compiler ceiling
"No space left on device". Per-chunk graphs are small enough to fit).

Recipe applied:
  1. Conv2dLinear for every big projection (in_proj_qkv/z/b/a/out_proj
     in SSM; q/k/v/o + gate/up/down + lm_head in full_attn / MLP).
  2. ANERMSNorm (cat([x,-x]) → LayerNorm → slice) for the layer-level
     input/post_attn norms and the attention head norms (q_norm, k_norm).
     The SSM RMSNormGated keeps the explicit fp32 path because the SiLU
     z-gate doesn't fit the [x,-x] identity.
  3. ane_softmax (max/sub/exp/sum/div fp16) for full_attention.
  4. repeat_kv_ane (reshape+repeat+view) instead of repeat_interleave.
  5. KV cache update via where(range == position) — scatter-free.

Layout: layers receive hidden as (1, 1, hidden) at the chunk boundary.
Internally, projections use Conv2dLinear's auto-permute path so the
per-projection op picks up the (1, hidden, 1, 1) ANE form. This keeps
the chunk's I/O byte-compatible with the existing v1.0.3 / v1.1.0 Swift
loader (no MLMultiArray shape change).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import (
    MODEL_DTYPE,
    ANERMSNorm,
    Conv2dLinear,
    ane_softmax,
    repeat_kv_ane,
)


def _conv_from_linear(lin: nn.Linear) -> Conv2dLinear:
    """Wrap an HF nn.Linear into a Conv2dLinear (kernel_size=1) at fp16."""
    c = Conv2dLinear(
        lin.in_features, lin.out_features,
        bias=lin.bias is not None, dtype=MODEL_DTYPE,
    )
    c.conv.weight.data = lin.weight.data.detach().to(MODEL_DTYPE).unsqueeze(-1).unsqueeze(-1)
    if lin.bias is not None:
        c.conv.bias.data = lin.bias.data.detach().to(MODEL_DTYPE)
    return c


def _norm_from_hf(weight: torch.Tensor, eps: float, hidden: int) -> ANERMSNorm:
    """Build ANERMSNorm whose weight matches an HF RMSNorm.weight exactly.

    HF RMSNorm: y = x * rsqrt(mean(x^2)+eps) * (1 + w)
    ANERMSNorm: y = layer_norm(cat([x,-x])) * w'
    Identity: with cat([x,-x]) the LN reduces to RMSNorm with weight=1, so
    setting w' = (1 + w) reproduces HF semantics exactly.
    """
    n = ANERMSNorm(hidden, eps=eps)
    n.weight.data = (1.0 + weight.detach().float()).to(MODEL_DTYPE).clone()
    return n


def _norm_from_hf_head(weight: torch.Tensor, eps: float, head_dim: int) -> ANERMSNorm:
    """Same as _norm_from_hf but for per-head norms (q_norm/k_norm)."""
    n = ANERMSNorm(head_dim, eps=eps)
    n.weight.data = (1.0 + weight.detach().float()).to(MODEL_DTYPE).clone()
    return n


# ---- SSM (Gated DeltaNet) decode step --------------------------------------


class ANELinearAttnDecodeStep(nn.Module):
    """Single-token SSM decode step with Conv2d projections.

    The recurrence math (state * key, state * query reductions, RMSNormGated)
    runs in fp32 just like the baseline LinearAttnDecodeStep — those are
    rank-1 reductions that don't have an ANE-specific replacement and
    benefit from fp32 stability for the per-layer state evolution.

    Only the four input projections + out_proj are swapped to Conv2d. They
    dominate the per-step bandwidth for this layer type.
    """
    def __init__(self, cfg, hf_layer):
        super().__init__()
        self.num_v = cfg.linear_num_value_heads
        self.num_k = cfg.linear_num_key_heads
        self.Dk = cfg.linear_key_head_dim
        self.Dv = cfg.linear_value_head_dim
        self.key_dim = self.Dk * self.num_k
        self.value_dim = self.Dv * self.num_v
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.K = cfg.linear_conv_kernel_dim
        self.eps = cfg.rms_norm_eps
        self.v_per_k = self.num_v // self.num_k

        lin = hf_layer.linear_attn
        self.in_proj_qkv = _conv_from_linear(lin.in_proj_qkv)
        self.in_proj_z   = _conv_from_linear(lin.in_proj_z)
        self.in_proj_b   = _conv_from_linear(lin.in_proj_b)
        self.in_proj_a   = _conv_from_linear(lin.in_proj_a)
        self.out_proj    = _conv_from_linear(lin.out_proj)
        # conv1d kernel (depthwise) and SSM-only scalars stay native.
        self.conv_w  = nn.Parameter(lin.conv1d.weight.detach().to(MODEL_DTYPE).clone(),
                                     requires_grad=False)
        self.dt_bias = nn.Parameter(lin.dt_bias.detach().to(MODEL_DTYPE).clone(),
                                     requires_grad=False)
        self.A_log   = nn.Parameter(lin.A_log.detach().to(MODEL_DTYPE).clone(),
                                     requires_grad=False)
        self.norm_w  = nn.Parameter(lin.norm.weight.detach().to(MODEL_DTYPE).clone(),
                                     requires_grad=False)

    @staticmethod
    def _l2norm(x, eps: float = 1e-6) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)

    def _rmsnorm_gated(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = self.norm_w.float() * x
        x = x * F.silu(z.float())
        return x.to(in_dtype)

    def forward(self, hidden_in: torch.Tensor,
                conv_state: torch.Tensor,
                rec_state: torch.Tensor):
        """hidden_in: (1, 1, H). conv_state: (1, C, K). rec_state: (1, Hv, Dk, Dv)."""
        # Conv2dLinear.forward auto-permutes (1,1,H) ↔ (1,H,1,1) for ANE.
        mixed_qkv = self.in_proj_qkv(hidden_in)   # (1, 1, conv_dim)
        z = self.in_proj_z(hidden_in)
        b = self.in_proj_b(hidden_in)
        a = self.in_proj_a(hidden_in)

        # Depthwise conv1d over (conv_state | new) — (1, C, K+1).
        mixed_qkv_t = mixed_qkv.transpose(1, 2)                       # (1, C, 1)
        w = torch.cat([conv_state, mixed_qkv_t], dim=-1)               # (1, C, K+1)
        kw = self.conv_w.squeeze(1)                                     # (C, K)
        conv_out = (w[:, :, 1:] * kw.unsqueeze(0)).sum(dim=-1, keepdim=True)
        conv_out = F.silu(conv_out)
        new_conv_state = w[:, :, 1:]

        mixed_qkv = conv_out.transpose(1, 2)                            # (1, 1, C)
        q, k, v = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, 1, -1, self.Dk)
        k = k.reshape(1, 1, -1, self.Dk)
        v = v.reshape(1, 1, -1, self.Dv)

        beta = torch.sigmoid(b)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        if self.v_per_k > 1:
            q = q.repeat_interleave(self.v_per_k, dim=2)
            k = k.repeat_interleave(self.v_per_k, dim=2)

        q_ = q.transpose(1, 2).contiguous().float()[:, :, 0]
        k_ = k.transpose(1, 2).contiguous().float()[:, :, 0]
        v_ = v.transpose(1, 2).contiguous().float()[:, :, 0]
        q_ = self._l2norm(q_) * (1.0 / math.sqrt(self.Dk))
        k_ = self._l2norm(k_)
        g_scalar = g.float().transpose(1, 2).exp().squeeze(-1)
        g_ = g_scalar.unsqueeze(-1).unsqueeze(-1)
        beta_ = beta.float().transpose(1, 2).squeeze(-1)

        rec = rec_state.float() * g_
        kv_mem = (rec * k_.unsqueeze(-1)).sum(dim=-2)
        delta = (v_ - kv_mem) * beta_.unsqueeze(-1)
        rec = rec + k_.unsqueeze(-1) * delta.unsqueeze(-2)
        core_out = (rec * q_.unsqueeze(-1)).sum(dim=-2)

        core_flat = core_out.reshape(-1, self.Dv).to(hidden_in.dtype)
        z_flat = z.reshape(-1, self.Dv)
        out_flat = self._rmsnorm_gated(core_flat, z_flat)
        out = out_flat.reshape(1, 1, self.value_dim)
        hidden_out = self.out_proj(out)

        return hidden_out, new_conv_state, rec.to(rec_state.dtype)


# ---- Full attention decode step --------------------------------------------


class ANEFullAttnDecodeStep(nn.Module):
    """Single-token GQA full attention with Conv2d projections, ANERMSNorm
    head norms, ane_softmax, and ANE-friendly KV cache update.

    Mirrors VL 4B `_ane` attention block but keeps the Qwen3.5 quirks:
      - q_proj output is double-width (gate fold): (1, 1, H * D * 2).
      - partial RoPE (rotary_factor = 0.25, only the first head_dim/4 of q/k).
      - sigmoid gate is multiplied into the attention output before o_proj.
    """
    def __init__(self, cfg, hf_attn, max_seq: int):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        self.eps = cfg.rms_norm_eps
        self.max_seq = max_seq
        self.rotary_dim = int(self.head_dim * 0.25)  # partial_rotary_factor=0.25

        self.q_proj = _conv_from_linear(hf_attn.q_proj)
        self.k_proj = _conv_from_linear(hf_attn.k_proj)
        self.v_proj = _conv_from_linear(hf_attn.v_proj)
        self.o_proj = _conv_from_linear(hf_attn.o_proj)
        self.q_norm = _norm_from_hf_head(hf_attn.q_norm.weight, self.eps, self.head_dim)
        self.k_norm = _norm_from_hf_head(hf_attn.k_norm.weight, self.eps, self.head_dim)

        # range_buf for KV write mask + causal mask. Stored once, used twice.
        self.register_buffer(
            "range_buf",
            torch.arange(max_seq, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor):
        rd = self.rotary_dim
        cos = cos.unsqueeze(1)  # (1, 1, 1, rd)
        sin = sin.unsqueeze(1)
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_out = q_rot * cos + self._rotate_half(q_rot) * sin
        k_out = k_rot * cos + self._rotate_half(k_rot) * sin
        return torch.cat([q_out, q_pass], dim=-1), torch.cat([k_out, k_pass], dim=-1)

    def forward(self, hidden_in: torch.Tensor,
                position: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                k_cache: torch.Tensor, v_cache: torch.Tensor):
        """
        hidden_in : (1, 1, H)
        position  : (1,) fp32
        cos, sin  : (1, 1, rotary_dim)
        k_cache   : (1, num_kv_heads, max_seq, head_dim)
        v_cache   : same
        """
        H, HKV, D = self.num_heads, self.num_kv_heads, self.head_dim
        max_seq = self.max_seq

        qg = self.q_proj(hidden_in)      # (1, 1, H * D * 2)
        k = self.k_proj(hidden_in)        # (1, 1, HKV * D)
        v = self.v_proj(hidden_in)

        qg = qg.reshape(1, 1, H, D * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(1, 1, H * D)

        # q_norm/k_norm operate on the head dim. ANERMSNorm normalizes over
        # the LAST dim, which is exactly head_dim here.
        q = self.q_norm(q).transpose(1, 2)               # (1, H, 1, D)
        k = k.reshape(1, 1, HKV, D)
        k = self.k_norm(k).transpose(1, 2)               # (1, HKV, 1, D)
        v = v.reshape(1, 1, HKV, D).transpose(1, 2)      # (1, HKV, 1, D)

        q, k = self._apply_rope(q, k, cos, sin)

        # Scatter-free KV write: broadcast new K/V over max_seq, then
        # where(range == position).
        write_mask = (self.range_buf == position).view(1, 1, max_seq, 1)
        new_k_b = k.expand(1, HKV, max_seq, D)
        new_v_b = v.expand(1, HKV, max_seq, D)
        new_k = torch.where(write_mask, new_k_b, k_cache)
        new_v = torch.where(write_mask, new_v_b, v_cache)

        # GQA repeat — ANE-resident (no repeat_interleave).
        if self.num_kv_groups > 1:
            k_r = repeat_kv_ane(new_k, self.num_kv_groups, HKV, max_seq, D)
            v_r = repeat_kv_ane(new_v, self.num_kv_groups, HKV, max_seq, D)
        else:
            k_r = new_k
            v_r = new_v

        # Attention: scores (1, H, 1, max_seq).
        scores = (q @ k_r.transpose(-1, -2)) * self.scale
        causal = torch.where(
            self.range_buf <= position,
            torch.zeros_like(self.range_buf),
            torch.full_like(self.range_buf, -1e4),
        ).view(1, 1, 1, max_seq)
        scores = scores + causal

        attn = ane_softmax(scores, dim=-1)
        attn_out = attn @ v_r                             # (1, H, 1, D)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(1, 1, H * D)

        attn_out = attn_out * torch.sigmoid(gate)
        hidden_out = self.o_proj(attn_out)
        return hidden_out, new_k, new_v


# ---- ANE-form MLP + dispatcher --------------------------------------------


class ANEDecodeMLP(nn.Module):
    """SwiGLU MLP with Conv2dLinear gate/up/down projections."""
    def __init__(self, hf_mlp):
        super().__init__()
        self.gate_proj = _conv_from_linear(hf_mlp.gate_proj)
        self.up_proj   = _conv_from_linear(hf_mlp.up_proj)
        self.down_proj = _conv_from_linear(hf_mlp.down_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.silu(self.gate_proj(x))
        u = self.up_proj(x)
        return self.down_proj(g * u)


class ANEDecoderDecodeLayer(nn.Module):
    """Layer-type aware dispatcher matching the baseline interface.

    Same forward signature as test_qwen3_5_full_decode_trace.DecoderDecodeLayer
    so chunk builders can swap between baseline and ANE variants without
    touching the chunk wiring.
    """
    def __init__(self, cfg, hf_layer, max_seq: int):
        super().__init__()
        self.layer_type = hf_layer.layer_type
        self.input_norm = _norm_from_hf(
            hf_layer.input_layernorm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        self.post_attn_norm = _norm_from_hf(
            hf_layer.post_attention_layernorm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        self.mlp = ANEDecodeMLP(hf_layer.mlp)
        if self.layer_type == "linear_attention":
            self.mixer = ANELinearAttnDecodeStep(cfg, hf_layer)
        else:
            self.mixer = ANEFullAttnDecodeStep(cfg, hf_layer.self_attn, max_seq)

    def forward(self, hidden, position, cos, sin, state_a, state_b):
        residual = hidden
        h = self.input_norm(hidden)
        if self.layer_type == "linear_attention":
            h, ns_a, ns_b = self.mixer(h, state_a, state_b)
        else:
            h, ns_a, ns_b = self.mixer(h, position, cos, sin, state_a, state_b)
        hidden = residual + h
        residual = hidden
        h = self.post_attn_norm(hidden)
        h = self.mlp(h)
        return residual + h, ns_a, ns_b
