"""Decode with all nn.Linear-equivalent matmuls replaced by 1x1 Conv2d.

Apple's ml-ane-transformers pattern: ANE treats nn.Conv2d(kernel_size=1)
differently from nn.Linear. The conv kernel reportedly uses higher-precision
accumulation headers on ANE, giving fp32-like accuracy on the output while
keeping fp16 storage for weights/activations. This is the Gemma 4 approach
(see conversion/ane_ops.py).

For decode specifically, the biggest precision loss on ANE is the
1024 -> 248320 lm_head matmul (top-1 drops to 40% because the reduction
tips rank orderings). If Conv2d 1x1 is indeed the "fp32 accumulate on ANE"
escape hatch, replacing linear ops here should recover top-1 without
leaving ANE.

This module builds decode the same as test_qwen3_5_full_decode_trace.py
but swaps every F.linear call for a (B,T,C) -> (B,C,1,T) Conv2d path.
Everything else (RMSNorm, Neumann-free recurrent SSM, KV cache via
where-eq) is unchanged.
"""
from pathlib import Path
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from test_qwen3_5_full_decode_trace import (
    DecodeRMSNorm, MAX_SEQ, make_zero_states, make_example_inputs,
    cos_sim, convert_and_audit,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
ORACLE = Path(__file__).parent / "qwen3_5_reference_logits.pt"


def conv2d_linear(x, weight, bias=None):
    """nn.Linear equivalent as Conv2d 1x1. Layout matches ane_ops.py:
    (B, T, in) -> (B, in, 1, T) conv -> (B, out, 1, T) -> (B, T, out)."""
    x4 = x.permute(0, 2, 1).unsqueeze(2)
    w4 = weight.unsqueeze(-1).unsqueeze(-1)
    y4 = F.conv2d(x4, w4, bias=bias)
    return y4.squeeze(2).permute(0, 2, 1)


class ConvMLP(nn.Module):
    def __init__(self, gate_w, up_w, down_w):
        super().__init__()
        self.gate_w = nn.Parameter(gate_w.detach().clone(), requires_grad=False)
        self.up_w = nn.Parameter(up_w.detach().clone(), requires_grad=False)
        self.down_w = nn.Parameter(down_w.detach().clone(), requires_grad=False)

    def forward(self, x):
        g = F.silu(conv2d_linear(x, self.gate_w))
        u = conv2d_linear(x, self.up_w)
        return conv2d_linear(g * u, self.down_w)


class ConvLinearAttnDecodeStep(nn.Module):
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
        self.conv_w = nn.Parameter(lin.conv1d.weight.detach().clone(), requires_grad=False)
        self.in_proj_qkv_w = nn.Parameter(lin.in_proj_qkv.weight.detach().clone(), requires_grad=False)
        self.in_proj_z_w   = nn.Parameter(lin.in_proj_z.weight.detach().clone(),   requires_grad=False)
        self.in_proj_b_w   = nn.Parameter(lin.in_proj_b.weight.detach().clone(),   requires_grad=False)
        self.in_proj_a_w   = nn.Parameter(lin.in_proj_a.weight.detach().clone(),   requires_grad=False)
        self.out_proj_w    = nn.Parameter(lin.out_proj.weight.detach().clone(),    requires_grad=False)
        self.dt_bias = nn.Parameter(lin.dt_bias.detach().clone(), requires_grad=False)
        self.A_log = nn.Parameter(lin.A_log.detach().clone(), requires_grad=False)
        self.norm_w = nn.Parameter(lin.norm.weight.detach().clone(), requires_grad=False)

    @staticmethod
    def _l2norm(x, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)

    def _rmsnorm_gated(self, x, z):
        in_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = self.norm_w * x.to(in_dtype)
        x = x * F.silu(z.float())
        return x.to(in_dtype)

    def forward(self, hidden_in, conv_state, rec_state):
        mixed_qkv = conv2d_linear(hidden_in, self.in_proj_qkv_w)
        z = conv2d_linear(hidden_in, self.in_proj_z_w)
        b = conv2d_linear(hidden_in, self.in_proj_b_w)
        a = conv2d_linear(hidden_in, self.in_proj_a_w)
        mixed_qkv_t = mixed_qkv.transpose(1, 2)
        w = torch.cat([conv_state, mixed_qkv_t], dim=-1)
        kw = self.conv_w.squeeze(1)
        conv_out = (w[:, :, 1:] * kw.unsqueeze(0)).sum(dim=-1, keepdim=True)
        conv_out = F.silu(conv_out)
        new_conv_state = w[:, :, 1:]
        mixed_qkv = conv_out.transpose(1, 2)
        q, k, v = torch.split(mixed_qkv,
                               [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, 1, -1, self.Dk)
        k = k.reshape(1, 1, -1, self.Dk)
        v = v.reshape(1, 1, -1, self.Dv)
        beta = torch.sigmoid(b)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
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
        hidden_out = conv2d_linear(out, self.out_proj_w)
        return hidden_out, new_conv_state, rec.to(rec_state.dtype)


class ConvFullAttnDecodeStep(nn.Module):
    def __init__(self, cfg, hf_attn, max_seq):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        self.eps = cfg.rms_norm_eps
        self.max_seq = max_seq
        self.q_proj_w = nn.Parameter(hf_attn.q_proj.weight.detach().clone(), requires_grad=False)
        self.k_proj_w = nn.Parameter(hf_attn.k_proj.weight.detach().clone(), requires_grad=False)
        self.v_proj_w = nn.Parameter(hf_attn.v_proj.weight.detach().clone(), requires_grad=False)
        self.o_proj_w = nn.Parameter(hf_attn.o_proj.weight.detach().clone(), requires_grad=False)
        self.q_norm_w = nn.Parameter(hf_attn.q_norm.weight.detach().clone(), requires_grad=False)
        self.k_norm_w = nn.Parameter(hf_attn.k_norm.weight.detach().clone(), requires_grad=False)
        self.register_buffer("range_buf",
                              torch.arange(max_seq, dtype=torch.float32),
                              persistent=False)

    def _head_rmsnorm(self, x, w):
        in_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x * (1.0 + w.float())
        return x.to(in_dtype)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, q, k, cos, sin):
        rd = int(self.head_dim * 0.25)
        cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_out = q_rot * cos + self._rotate_half(q_rot) * sin
        k_out = k_rot * cos + self._rotate_half(k_rot) * sin
        return torch.cat([q_out, q_pass], dim=-1), torch.cat([k_out, k_pass], dim=-1)

    def forward(self, hidden_in, position, cos, sin, k_cache, v_cache):
        H = self.hidden_size
        qg = conv2d_linear(hidden_in, self.q_proj_w)
        k = conv2d_linear(hidden_in, self.k_proj_w)
        v = conv2d_linear(hidden_in, self.v_proj_w)
        qg = qg.reshape(1, 1, self.num_heads, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(1, 1, self.num_heads * self.head_dim)
        q = self._head_rmsnorm(q, self.q_norm_w).transpose(1, 2)
        k = k.reshape(1, 1, self.num_kv_heads, self.head_dim)
        k = self._head_rmsnorm(k, self.k_norm_w).transpose(1, 2)
        v = v.reshape(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self._apply_rope(q, k, cos, sin)
        write_mask = (self.range_buf == position).view(1, 1, self.max_seq, 1)
        new_k_broadcast = k.expand(1, self.num_kv_heads, self.max_seq, self.head_dim)
        new_v_broadcast = v.expand(1, self.num_kv_heads, self.max_seq, self.head_dim)
        new_k_cache = torch.where(write_mask, new_k_broadcast, k_cache)
        new_v_cache = torch.where(write_mask, new_v_broadcast, v_cache)
        if self.num_kv_groups > 1:
            k_r = new_k_cache.repeat_interleave(self.num_kv_groups, dim=1)
            v_r = new_v_cache.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_r = new_k_cache; v_r = new_v_cache
        attn_scores = q @ k_r.transpose(-1, -2) * self.scale
        attn_mask = torch.where(self.range_buf <= position,
                                 torch.zeros_like(self.range_buf),
                                 torch.full_like(self.range_buf, -1e4))
        attn_scores = attn_scores + attn_mask.view(1, 1, 1, self.max_seq)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ v_r
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(
            1, 1, self.num_heads * self.head_dim)
        attn_out = attn_out * torch.sigmoid(gate)
        hidden_out = conv2d_linear(attn_out, self.o_proj_w)
        return hidden_out, new_k_cache, new_v_cache


class ConvDecoderLayer(nn.Module):
    def __init__(self, cfg, hf_layer, max_seq):
        super().__init__()
        self.layer_type = hf_layer.layer_type
        self.input_norm = DecodeRMSNorm(cfg.rms_norm_eps, hf_layer.input_layernorm.weight)
        self.post_attn_norm = DecodeRMSNorm(cfg.rms_norm_eps, hf_layer.post_attention_layernorm.weight)
        self.mlp = ConvMLP(hf_layer.mlp.gate_proj.weight,
                            hf_layer.mlp.up_proj.weight,
                            hf_layer.mlp.down_proj.weight)
        if self.layer_type == "linear_attention":
            self.mixer = ConvLinearAttnDecodeStep(cfg, hf_layer)
        else:
            self.mixer = ConvFullAttnDecodeStep(cfg, hf_layer.self_attn, max_seq)

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


class ConvFullDecodeModel(nn.Module):
    def __init__(self, cfg, hf_model, max_seq):
        super().__init__()
        self.max_seq = max_seq
        self.num_layers = cfg.num_hidden_layers
        self.embed_w = nn.Parameter(hf_model.model.embed_tokens.weight.detach().clone(),
                                     requires_grad=False)
        self.final_norm = DecodeRMSNorm(cfg.rms_norm_eps, hf_model.model.norm.weight)
        self.lm_head_w = nn.Parameter(hf_model.lm_head.weight.detach().clone(),
                                       requires_grad=False)
        self.layers = nn.ModuleList([
            ConvDecoderLayer(cfg, hf_model.model.layers[i], max_seq)
            for i in range(self.num_layers)
        ])

    def forward(self, input_token, position, cos, sin, *states):
        hidden = F.embedding(input_token.to(torch.long), self.embed_w)
        new_states = []
        for i, layer in enumerate(self.layers):
            sa, sb = states[2 * i], states[2 * i + 1]
            hidden, ns_a, ns_b = layer(hidden, position, cos, sin, sa, sb)
            new_states.append(ns_a); new_states.append(ns_b)
        hidden = self.final_norm(hidden)
        logits = conv2d_linear(hidden, self.lm_head_w)
        return (logits, *new_states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    hf = Qwen3_5ForCausalLM.from_pretrained(MODEL_ID, config=cfg, torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    model = ConvFullDecodeModel(cfg, hf, args.max_seq).eval().float()
    del hf

    oracle = torch.load(str(ORACLE), map_location="cpu", weights_only=False)
    # torch sanity: feed oracle tokens, confirm conv2d math matches fp32 path
    print(f"\n=== torch fp32 sanity (Conv2d path) ===")
    for rec in oracle["records"][:3]:
        ids = rec["input_ids"]
        S = ids.shape[1]
        states = make_zero_states(cfg, args.max_seq)
        last_logits = None
        with torch.no_grad():
            for t in range(S):
                tok = ids[:, t:t+1].to(torch.int32)
                pos = torch.tensor([float(t)], dtype=torch.float32)
                pos_ids = torch.tensor([[t]], dtype=torch.long)
                dummy = torch.zeros(1, 1, cfg.hidden_size)
                c_t, s_t = rot(dummy, pos_ids)
                out = model(tok, pos, c_t.float(), s_t.float(),
                             *[s.float() for s in states])
                logits, *new_states = out
                states = list(new_states)
                if t == S - 1:
                    last_logits = logits[0, 0].float()
        ref = rec["logits_recurrent"][-1].float()
        c = cos_sim(last_logits, ref)
        top1 = int(torch.argmax(last_logits).item())
        match = top1 == int(rec["top10_last_ids"][0].item())
        print(f"  S={S:3d}  cos={c:.6f}  top1={match}  {rec['prompt'][:30]!r}")

    if args.skip_convert:
        return

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qwen3_5_0_8b_decode_conv2d_fp16_mseq{args.max_seq}.mlpackage"
    convert_and_audit(model, cfg, rot, args.max_seq, out_path)


if __name__ == "__main__":
    main()
