"""ANE decode layer for Qwen3.5 — KV cache via MLState (slice_update),
SSM state via classic input/output I/O. Single ct.StateType per chunk
matches VL Phase 1's verified iPhone-ANE-runnable pattern.

Why this split:
- The full multi-StateType MLState port (kv_cache + conv_state +
  rec_state per chunk, see qwen3_5_decode_layer_mlstate.py) compiles
  and runs on Mac GPU but blows ANE error 11 + CPU runtime miscompiles.
  iPhone ANE shows the same pattern in VL Phase 1's experience —
  single StateType only.
- KV cache is the biggest per-step state (12 MB on 0.8B). Putting it
  in MLState saves the marshaling cost. SSM state is smaller (~1-10 MB)
  and stays as conventional I/O. This is the "safe" subset that VL has
  proven runs on iPhone ANE.
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
    ane_softmax,
    ane_norm_from_hf,
    conv_from_linear,
    repeat_kv_ane,
)


# Local convenience aliases — Qwen3.5's RMSNorm uses the (1+w) gain
# convention, so plus_one_gain=True. Per-head norms (q_norm, k_norm)
# normalize over head_dim instead of hidden_size; the only difference
# is the `hidden` argument value passed in.
def _conv_from_linear(lin: nn.Linear):
    return conv_from_linear(lin, dtype=MODEL_DTYPE)


def _norm_from_hf(weight: torch.Tensor, eps: float, hidden: int):
    return ane_norm_from_hf(weight, eps, hidden, plus_one_gain=True)


def _norm_from_hf_head(weight: torch.Tensor, eps: float, head_dim: int):
    return ane_norm_from_hf(weight, eps, head_dim, plus_one_gain=True)


# ---- SSM step (stateless I/O) ---------------------------------------------


class MLKVLinearAttnDecodeStep(nn.Module):
    """Stateless SSM step — same conv_state/rec_state in/out interface as
    the original stateless ANE layer. Only the KV in the chunk lives in
    MLState; SSM is unchanged from qwen3_5_decode_layer_ane.py."""
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
        """conv_state: (1, C, K). rec_state: (1, num_v, Dk, Dv)."""
        mixed_qkv = self.in_proj_qkv(hidden_in)
        z = self.in_proj_z(hidden_in)
        b = self.in_proj_b(hidden_in)
        a = self.in_proj_a(hidden_in)

        mixed_qkv_t = mixed_qkv.transpose(1, 2)
        w = torch.cat([conv_state, mixed_qkv_t], dim=-1)
        kw = self.conv_w.squeeze(1)
        conv_out = (w[:, :, 1:] * kw.unsqueeze(0)).sum(dim=-1, keepdim=True)
        conv_out = F.silu(conv_out)
        new_conv_state = w[:, :, 1:]

        mixed_qkv = conv_out.transpose(1, 2)
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


# ---- Full attention step (MLState) ----------------------------------------


class MLKVFullAttnDecodeStep(nn.Module):
    """Single-token GQA. KV in chunk-owned MLState; per-step writes via
    slice_update at current_pos."""
    def __init__(self, cfg, hf_attn, max_seq: int, layer_idx_in_chunk_full: int):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        self.eps = cfg.rms_norm_eps
        self.max_seq = max_seq
        self.rotary_dim = int(self.head_dim * 0.25)
        self.k_idx = 2 * layer_idx_in_chunk_full
        self.v_idx = 2 * layer_idx_in_chunk_full + 1

        self.q_proj = _conv_from_linear(hf_attn.q_proj)
        self.k_proj = _conv_from_linear(hf_attn.k_proj)
        self.v_proj = _conv_from_linear(hf_attn.v_proj)
        self.o_proj = _conv_from_linear(hf_attn.o_proj)
        self.q_norm = _norm_from_hf_head(hf_attn.q_norm.weight, self.eps, self.head_dim)
        self.k_norm = _norm_from_hf_head(hf_attn.k_norm.weight, self.eps, self.head_dim)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, q, k, cos, sin):
        rd = self.rotary_dim
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_out = q_rot * cos + self._rotate_half(q_rot) * sin
        k_out = k_rot * cos + self._rotate_half(k_rot) * sin
        return torch.cat([q_out, q_pass], dim=-1), torch.cat([k_out, k_pass], dim=-1)

    def forward(self, hidden_in, cos, sin, causal_mask, current_pos, kv_cache):
        H, HKV, D = self.num_heads, self.num_kv_heads, self.head_dim
        max_seq = self.max_seq

        qg = self.q_proj(hidden_in)
        k = self.k_proj(hidden_in)
        v = self.v_proj(hidden_in)

        qg = qg.reshape(1, 1, H, D * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(1, 1, H * D)

        q = self.q_norm(q).transpose(1, 2)
        k = k.reshape(1, 1, HKV, D)
        k = self.k_norm(k).transpose(1, 2)
        v = v.reshape(1, 1, HKV, D).transpose(1, 2)

        q, k = self._apply_rope(q, k, cos, sin)

        # slice-update writes — same pattern as VL Phase 1.
        k_write = k.squeeze(0).to(kv_cache.dtype)
        v_write = v.squeeze(0).to(kv_cache.dtype)
        kv_cache[self.k_idx:self.k_idx + 1, :, current_pos:current_pos + 1, :] = \
            k_write.unsqueeze(0)
        kv_cache[self.v_idx:self.v_idx + 1, :, current_pos:current_pos + 1, :] = \
            v_write.unsqueeze(0)

        k_full = kv_cache[self.k_idx:self.k_idx + 1, :, :, :]
        v_full = kv_cache[self.v_idx:self.v_idx + 1, :, :, :]

        if self.num_kv_groups > 1:
            k_r = repeat_kv_ane(k_full, self.num_kv_groups, HKV, max_seq, D)
            v_r = repeat_kv_ane(v_full, self.num_kv_groups, HKV, max_seq, D)
        else:
            k_r = k_full
            v_r = v_full

        scores = (q @ k_r.transpose(-1, -2)) * self.scale
        scores = scores + causal_mask
        attn = ane_softmax(scores, dim=-1)
        attn_out = attn @ v_r
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(1, 1, H * D)
        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)


class MLKVDecodeMLP(nn.Module):
    def __init__(self, hf_mlp):
        super().__init__()
        self.gate_proj = _conv_from_linear(hf_mlp.gate_proj)
        self.up_proj   = _conv_from_linear(hf_mlp.up_proj)
        self.down_proj = _conv_from_linear(hf_mlp.down_proj)

    def forward(self, x):
        g = F.silu(self.gate_proj(x))
        u = self.up_proj(x)
        return self.down_proj(g * u)


def is_full_attn(layer_idx: int) -> bool:
    return layer_idx % 4 == 3


class MLKVDecoderLayer(nn.Module):
    """Routes to SSM (stateless) or full_attention (MLState KV) step."""
    def __init__(self, cfg, hf_layer, max_seq: int, full_idx_in_chunk: int = 0):
        super().__init__()
        self.layer_type = hf_layer.layer_type
        self.input_norm = _norm_from_hf(
            hf_layer.input_layernorm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        self.post_attn_norm = _norm_from_hf(
            hf_layer.post_attention_layernorm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        self.mlp = MLKVDecodeMLP(hf_layer.mlp)
        if self.layer_type == "linear_attention":
            self.mixer = MLKVLinearAttnDecodeStep(cfg, hf_layer)
        else:
            self.mixer = MLKVFullAttnDecodeStep(
                cfg, hf_layer.self_attn, max_seq, full_idx_in_chunk)

    def forward_lin(self, hidden, state_a, state_b):
        residual = hidden
        h = self.input_norm(hidden)
        h, ns_a, ns_b = self.mixer(h, state_a, state_b)
        hidden = residual + h
        residual = hidden
        h = self.post_attn_norm(hidden)
        h = self.mlp(h)
        return residual + h, ns_a, ns_b

    def forward_full(self, hidden, cos, sin, causal_mask, current_pos, kv_cache):
        residual = hidden
        h = self.input_norm(hidden)
        h = self.mixer(h, cos, sin, causal_mask, current_pos, kv_cache)
        hidden = residual + h
        residual = hidden
        h = self.post_attn_norm(hidden)
        h = self.mlp(h)
        return residual + h
