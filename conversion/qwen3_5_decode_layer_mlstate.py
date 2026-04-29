"""ANE + MLState decode layer for Qwen3.5 hybrid SSM + full_attention.

VL Phase 1's MLState recipe (10→24.4 tok/s = 2.4× on Qwen3-VL 2B,
phys_footprint 1.7 GB → 264 MB) ported to Qwen3.5's hybrid stack:

  - **Full attention layers**: KV cache lives in a unified MLState
    `kv_cache_<chunk>` of shape (2*L_full, num_kv_heads, max_seq, head_dim).
    Per-step write uses `kv[k_idx:k_idx+1, :, p:p+1, :] = k_new` —
    coremltools lowers this to ios18.slice_update inside the graph.

  - **Linear (Gated DeltaNet) layers**: conv_state and rec_state both
    fit in MLStates packed across the linear layers in this chunk:
      conv_state_<chunk> : (L_lin, conv_dim, K=4)
      rec_state_<chunk>  : (L_lin, num_v, Dk, Dv)
    Per-step both states are fully overwritten — slice_update on the
    leading dim (which layer in the chunk) writes the new value.

Compared to the stateless (state_X_a / new_state_X_a) port in
qwen3_5_decode_layer_ane.py, this saves the per-step Python/Swift→
CoreML state marshaling on ~50 tensors. VL Phase 1 measured the win at
2.4× decode tok/s.
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
    c = Conv2dLinear(lin.in_features, lin.out_features,
                     bias=lin.bias is not None, dtype=MODEL_DTYPE)
    c.conv.weight.data = lin.weight.data.detach().to(MODEL_DTYPE).unsqueeze(-1).unsqueeze(-1)
    if lin.bias is not None:
        c.conv.bias.data = lin.bias.data.detach().to(MODEL_DTYPE)
    return c


def _norm_from_hf(weight: torch.Tensor, eps: float, hidden: int) -> ANERMSNorm:
    """HF Qwen3.5 RMSNorm = x * rsqrt(mean(x^2)+eps) * (1+w).
    ANERMSNorm with weight=(1+w) is bit-identical."""
    n = ANERMSNorm(hidden, eps=eps)
    n.weight.data = (1.0 + weight.detach().float()).to(MODEL_DTYPE).clone()
    return n


def _norm_from_hf_head(weight: torch.Tensor, eps: float, head_dim: int) -> ANERMSNorm:
    n = ANERMSNorm(head_dim, eps=eps)
    n.weight.data = (1.0 + weight.detach().float()).to(MODEL_DTYPE).clone()
    return n


# ---- SSM step (stateful) --------------------------------------------------


class MLStateLinearAttnDecodeStep(nn.Module):
    """Single-token Gated DeltaNet decode step. conv_state / rec_state
    live in chunk-owned MLStates; this layer reads/writes its slice."""
    def __init__(self, cfg, hf_layer, layer_idx_in_chunk_lin: int):
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
        self.li = layer_idx_in_chunk_lin

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
                conv_states: torch.Tensor,
                rec_states: torch.Tensor):
        """
        hidden_in  : (1, 1, H)
        conv_states: (L_lin, 1, conv_dim, K) — 4D MLState; ANE prefers
                      rank-4 states. Slice extracts a 4D block then
                      squeezes the singleton dim for compute.
        rec_states : (L_lin, num_v, Dk, Dv) — already 4D.
        """
        # Read this layer's state slices.
        conv_state = conv_states[self.li:self.li + 1, :, :, :].squeeze(1)  # (1, C, K)
        rec_state  = rec_states[self.li:self.li + 1, :, :, :]  # (1, num_v, Dk, Dv)

        mixed_qkv = self.in_proj_qkv(hidden_in)   # (1, 1, conv_dim)
        z = self.in_proj_z(hidden_in)
        b = self.in_proj_b(hidden_in)
        a = self.in_proj_a(hidden_in)

        mixed_qkv_t = mixed_qkv.transpose(1, 2)                       # (1, C, 1)
        w = torch.cat([conv_state, mixed_qkv_t], dim=-1)               # (1, C, K+1)
        kw = self.conv_w.squeeze(1)                                     # (C, K)
        conv_out = (w[:, :, 1:] * kw.unsqueeze(0)).sum(dim=-1, keepdim=True)
        conv_out = F.silu(conv_out)
        new_conv_state = w[:, :, 1:]   # (1, C, K)

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

        # In-place slice writes back into the chunk-owned MLStates.
        # conv_states is 4D (L_lin, 1, C, K); unsqueeze new_conv_state to match.
        conv_states[self.li:self.li + 1, :, :, :] = \
            new_conv_state.unsqueeze(1).to(conv_states.dtype)
        rec_states[self.li:self.li + 1, :, :, :] = rec.to(rec_states.dtype)

        return hidden_out


# ---- Full attention step (stateful) ---------------------------------------


class MLStateFullAttnDecodeStep(nn.Module):
    """Single-token GQA attention. KV cache lives in chunk-owned
    `kv_cache` MLState; per-step writes via slice_update at current_pos."""
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
        self.rotary_dim = int(self.head_dim * 0.25)  # partial_rotary_factor=0.25
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

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor):
        rd = self.rotary_dim
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_out = q_rot * cos + self._rotate_half(q_rot) * sin
        k_out = k_rot * cos + self._rotate_half(k_rot) * sin
        return torch.cat([q_out, q_pass], dim=-1), torch.cat([k_out, k_pass], dim=-1)

    def forward(self, hidden_in: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                causal_mask: torch.Tensor,
                current_pos: torch.Tensor,
                kv_cache: torch.Tensor):
        """
        hidden_in  : (1, 1, H)
        cos, sin   : (1, 1, rotary_dim)
        causal_mask: (1, 1, 1, max_seq) fp16 — -1e4 for positions > current_pos
        current_pos: (1,) int32
        kv_cache   : (2*L_full, num_kv_heads, max_seq, head_dim) MLState
        """
        H, HKV, D = self.num_heads, self.num_kv_heads, self.head_dim
        max_seq = self.max_seq

        qg = self.q_proj(hidden_in)      # (1, 1, H * D * 2)
        k = self.k_proj(hidden_in)        # (1, 1, HKV * D)
        v = self.v_proj(hidden_in)

        qg = qg.reshape(1, 1, H, D * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(1, 1, H * D)

        q = self.q_norm(q).transpose(1, 2)               # (1, H, 1, D)
        k = k.reshape(1, 1, HKV, D)
        k = self.k_norm(k).transpose(1, 2)               # (1, HKV, 1, D)
        v = v.reshape(1, 1, HKV, D).transpose(1, 2)      # (1, HKV, 1, D)

        q, k = self._apply_rope(q, k, cos, sin)

        # slice-assign writes — coremltools lowers to ios18.slice_update.
        # k/v shape (1, HKV, 1, D) → unsqueeze to (1, HKV, 1, D) target slice
        # in kv_cache[k_idx:k_idx+1, :, p:p+1, :].
        k_write = k.squeeze(0).to(kv_cache.dtype)        # (HKV, 1, D)
        v_write = v.squeeze(0).to(kv_cache.dtype)
        kv_cache[self.k_idx:self.k_idx + 1, :, current_pos:current_pos + 1, :] = \
            k_write.unsqueeze(0)
        kv_cache[self.v_idx:self.v_idx + 1, :, current_pos:current_pos + 1, :] = \
            v_write.unsqueeze(0)

        # Re-slice the full layer K/V (post-write).
        k_full = kv_cache[self.k_idx:self.k_idx + 1, :, :, :]   # (1, HKV, max_seq, D)
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
        hidden_out = self.o_proj(attn_out)
        return hidden_out


# ---- ANE-form MLP --------------------------------------------------------


class MLStateDecodeMLP(nn.Module):
    def __init__(self, hf_mlp):
        super().__init__()
        self.gate_proj = _conv_from_linear(hf_mlp.gate_proj)
        self.up_proj   = _conv_from_linear(hf_mlp.up_proj)
        self.down_proj = _conv_from_linear(hf_mlp.down_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.silu(self.gate_proj(x))
        u = self.up_proj(x)
        return self.down_proj(g * u)


# ---- Layer dispatcher (stateful) ------------------------------------------


def is_full_attn(layer_idx: int) -> bool:
    """Qwen3.5 hybrid pattern [L,L,L,F]×6 — full at every 4th index."""
    return layer_idx % 4 == 3


class MLStateDecoderLayer(nn.Module):
    """Routes to SSM or full_attention step. Holds the per-layer norm
    weights + MLP. State is owned by the enclosing chunk and passed in.
    """
    def __init__(self, cfg, hf_layer, max_seq: int,
                 lin_idx_in_chunk: int = 0,
                 full_idx_in_chunk: int = 0):
        super().__init__()
        self.layer_type = hf_layer.layer_type
        self.input_norm = _norm_from_hf(
            hf_layer.input_layernorm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        self.post_attn_norm = _norm_from_hf(
            hf_layer.post_attention_layernorm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        self.mlp = MLStateDecodeMLP(hf_layer.mlp)
        if self.layer_type == "linear_attention":
            self.mixer = MLStateLinearAttnDecodeStep(cfg, hf_layer, lin_idx_in_chunk)
        else:
            self.mixer = MLStateFullAttnDecodeStep(
                cfg, hf_layer.self_attn, max_seq, full_idx_in_chunk)

    def forward(self, hidden, cos, sin, causal_mask, current_pos,
                conv_states, rec_states, kv_cache):
        """Reads/writes state buffers in-place; returns updated hidden."""
        residual = hidden
        h = self.input_norm(hidden)
        if self.layer_type == "linear_attention":
            h = self.mixer(h, conv_states, rec_states)
        else:
            h = self.mixer(h, cos, sin, causal_mask, current_pos, kv_cache)
        hidden = residual + h
        residual = hidden
        h = self.post_attn_norm(hidden)
        h = self.mlp(h)
        return residual + h
