"""Phase 4e-1: 24-layer stateful decode converter for Qwen3.5-0.8B.

Builds a FullDecodeModel that does a single-token auto-regressive step:
  embed_tokens -> 24 decoder layers (18 linear_attention + 6 full_attention)
  -> final RMSNorm -> lm_head.

Inputs:
  input_token  : (1, 1) int32
  position     : (1,)   int32   (current token position, 0..max_seq-1)
  cos, sin     : (1, 1, rotary_dim)  precomputed RoPE for this position
  states       : 18 × (conv_state, rec_state) for linear_attention layers,
                  6 × (k_cache,     v_cache)   for full_attention layers.
                  Shapes fixed to max_seq=128 (plenty of headroom).

Outputs:
  logits       : (1, 1, V)
  states'      : updated states, same shapes as inputs.

Parity is verified by running the module token-by-token from all-zero
states and comparing to oracle['records'][i]['logits_recurrent'] (HF's
token-by-token recurrent decode).

The linear_attention decode step is ported from Phase 2a (single-layer,
proven 100% ANE). The full_attention decode step is a GQA softmax attention
with a fixed-max_seq KV cache updated via `torch.where(range == position)`
— scatter-free so it stays on ANE.
"""
from collections import Counter
from pathlib import Path
import argparse
import math
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding


MODEL_ID = "Qwen/Qwen3.5-0.8B"
ORACLE = Path(__file__).parent / "qwen3_5_reference_logits.pt"
MAX_SEQ = 128  # max prefill+decode length for this converter


# ---- layer submodules ------------------------------------------------------


class DecodeRMSNorm(nn.Module):
    """Qwen3_5RMSNorm: (x * rsqrt(mean(x²) + eps)) * (1 + w)."""
    def __init__(self, eps, weight):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(weight.detach().clone(), requires_grad=False)

    def forward(self, x):
        in_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x * (1.0 + self.w.float())
        return x.to(in_dtype)


class DecodeMLP(nn.Module):
    def __init__(self, gate_w, up_w, down_w):
        super().__init__()
        self.gate_w = nn.Parameter(gate_w.detach().clone(), requires_grad=False)
        self.up_w = nn.Parameter(up_w.detach().clone(), requires_grad=False)
        self.down_w = nn.Parameter(down_w.detach().clone(), requires_grad=False)

    def forward(self, x):
        g = F.silu(F.linear(x, self.gate_w))
        u = F.linear(x, self.up_w)
        return F.linear(g * u, self.down_w)


class LinearAttnDecodeStep(nn.Module):
    """Single-token Gated-DeltaNet decode step. Ported from Phase 2a proof."""
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
        self.A_log   = nn.Parameter(lin.A_log.detach().clone(),   requires_grad=False)
        self.norm_w  = nn.Parameter(lin.norm.weight.detach().clone(), requires_grad=False)

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
        # hidden_in: (1, 1, H).  conv_state: (1, C, K).  rec_state: (1, Hv, Dk, Dv).
        mixed_qkv = F.linear(hidden_in, self.in_proj_qkv_w)
        z = F.linear(hidden_in, self.in_proj_z_w)
        b = F.linear(hidden_in, self.in_proj_b_w)
        a = F.linear(hidden_in, self.in_proj_a_w)

        mixed_qkv_t = mixed_qkv.transpose(1, 2)                      # (1, C, 1)
        w = torch.cat([conv_state, mixed_qkv_t], dim=-1)             # (1, C, K+1)
        kw = self.conv_w.squeeze(1)                                   # (C, K)
        conv_out = (w[:, :, 1:] * kw.unsqueeze(0)).sum(dim=-1, keepdim=True)
        conv_out = F.silu(conv_out)
        new_conv_state = w[:, :, 1:]

        mixed_qkv = conv_out.transpose(1, 2)                          # (1, 1, C)
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
        hidden_out = F.linear(out, self.out_proj_w)

        return hidden_out, new_conv_state, rec.to(rec_state.dtype)


class FullAttnDecodeStep(nn.Module):
    """Single-token full_attention decode step with fixed-max_seq KV cache.

    Cache is updated by `where(range_buf == position)` — no scatter op, stays
    on ANE. Attention mask is `where(range_buf <= position)` so positions past
    the current token get -inf (plus the max-1 unused tail).
    """
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

        # range buffer (max_seq,) used for both KV write and attention mask
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
        rd = int(self.head_dim * 0.25)  # partial_rotary_factor = 0.25
        cos = cos.unsqueeze(1)  # (1, 1, 1, rd)
        sin = sin.unsqueeze(1)
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_out = q_rot * cos + self._rotate_half(q_rot) * sin
        k_out = k_rot * cos + self._rotate_half(k_rot) * sin
        return torch.cat([q_out, q_pass], dim=-1), torch.cat([k_out, k_pass], dim=-1)

    def forward(self, hidden_in, position, cos, sin, k_cache, v_cache):
        """
        hidden_in : (1, 1, H)
        position  : (1,) float (scalar position as rank-1 for trace-stability)
        cos, sin  : (1, 1, rd)
        k_cache   : (1, num_kv, max_seq, head_dim)
        v_cache   : same
        """
        H = self.hidden_size
        # q_proj double output for gate fold
        qg = F.linear(hidden_in, self.q_proj_w)       # (1, 1, num_heads * head_dim * 2)
        k = F.linear(hidden_in, self.k_proj_w)         # (1, 1, num_kv * head_dim)
        v = F.linear(hidden_in, self.v_proj_w)

        qg = qg.reshape(1, 1, self.num_heads, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)                  # each (1, 1, num_heads, head_dim)
        gate = gate.reshape(1, 1, self.num_heads * self.head_dim)

        q = self._head_rmsnorm(q, self.q_norm_w).transpose(1, 2)  # (1, num_heads, 1, head_dim)
        k = k.reshape(1, 1, self.num_kv_heads, self.head_dim)
        k = self._head_rmsnorm(k, self.k_norm_w).transpose(1, 2)  # (1, num_kv, 1, head_dim)
        v = v.reshape(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self._apply_rope(q, k, cos, sin)

        # KV cache write at `position` via where(range_buf == position)
        # position is a (1,) tensor; broadcast against range_buf (max_seq,).
        write_mask = (self.range_buf == position).view(1, 1, self.max_seq, 1)  # bool
        new_k_broadcast = k.expand(1, self.num_kv_heads, self.max_seq, self.head_dim)
        new_v_broadcast = v.expand(1, self.num_kv_heads, self.max_seq, self.head_dim)
        new_k_cache = torch.where(write_mask, new_k_broadcast, k_cache)
        new_v_cache = torch.where(write_mask, new_v_broadcast, v_cache)

        # GQA: repeat kv heads if needed
        if self.num_kv_groups > 1:
            k_r = new_k_cache.repeat_interleave(self.num_kv_groups, dim=1)
            v_r = new_v_cache.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_r = new_k_cache
            v_r = new_v_cache

        # Attention: scores = q @ k^T * scale, (1, num_heads, 1, max_seq)
        attn_scores = q @ k_r.transpose(-1, -2) * self.scale

        # Additive mask: -1e4 for positions > current, 0 for positions <= current
        attn_mask = torch.where(self.range_buf <= position,
                                 torch.zeros_like(self.range_buf),
                                 torch.full_like(self.range_buf, -1e4))
        attn_scores = attn_scores + attn_mask.view(1, 1, 1, self.max_seq)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ v_r                  # (1, num_heads, 1, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(
            1, 1, self.num_heads * self.head_dim
        )
        attn_out = attn_out * torch.sigmoid(gate)
        hidden_out = F.linear(attn_out, self.o_proj_w)  # (1, 1, H)

        return hidden_out, new_k_cache, new_v_cache


class DecoderDecodeLayer(nn.Module):
    """Dispatches to LinearAttnDecodeStep or FullAttnDecodeStep per layer_type."""
    def __init__(self, cfg, hf_layer, max_seq):
        super().__init__()
        self.layer_type = hf_layer.layer_type
        self.input_norm = DecodeRMSNorm(cfg.rms_norm_eps, hf_layer.input_layernorm.weight)
        self.post_attn_norm = DecodeRMSNorm(cfg.rms_norm_eps, hf_layer.post_attention_layernorm.weight)
        self.mlp = DecodeMLP(hf_layer.mlp.gate_proj.weight,
                              hf_layer.mlp.up_proj.weight,
                              hf_layer.mlp.down_proj.weight)
        if self.layer_type == "linear_attention":
            self.mixer = LinearAttnDecodeStep(cfg, hf_layer)
        else:
            self.mixer = FullAttnDecodeStep(cfg, hf_layer.self_attn, max_seq)

    def forward(self, hidden, position, cos, sin, state_a, state_b):
        """state_a/b = conv_state/rec_state  for linear,
                       k_cache/v_cache       for full."""
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


class FullDecodeModel(nn.Module):
    """24-layer stateful decode. State tensors are passed as positional args."""
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
            DecoderDecodeLayer(cfg, hf_model.model.layers[i], max_seq)
            for i in range(self.num_layers)
        ])

    def forward(self, input_token, position, cos, sin, *states):
        """input_token: (1,1) int. position: (1,) float. cos/sin: (1,1,rd).
        states: flat list of 2 * num_layers tensors, ordered layer-by-layer."""
        hidden = F.embedding(input_token.to(torch.long), self.embed_w)
        new_states = []
        for i, layer in enumerate(self.layers):
            sa, sb = states[2 * i], states[2 * i + 1]
            hidden, ns_a, ns_b = layer(hidden, position, cos, sin, sa, sb)
            new_states.append(ns_a)
            new_states.append(ns_b)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.lm_head_w)
        return (logits, *new_states)


# ---- state init + example input construction ------------------------------


def make_zero_states(cfg, max_seq):
    states = []
    for i in range(cfg.num_hidden_layers):
        lt = "linear_attention" if i % 4 != 3 else "full_attention"  # 0.8B pattern
        if lt == "linear_attention":
            conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 + \
                        cfg.linear_value_head_dim * cfg.linear_num_value_heads
            states.append(torch.zeros(1, conv_dim, cfg.linear_conv_kernel_dim))
            states.append(torch.zeros(1, cfg.linear_num_value_heads,
                                        cfg.linear_key_head_dim, cfg.linear_value_head_dim))
        else:
            states.append(torch.zeros(1, cfg.num_key_value_heads, max_seq, cfg.head_dim))
            states.append(torch.zeros(1, cfg.num_key_value_heads, max_seq, cfg.head_dim))
    return states


def make_example_inputs(cfg, max_seq, rot):
    input_token = torch.zeros(1, 1, dtype=torch.int32)
    position = torch.zeros(1, dtype=torch.float32)
    dummy = torch.zeros(1, 1, cfg.hidden_size)
    pos_ids = torch.zeros(1, 1, dtype=torch.long)
    with torch.no_grad():
        cos, sin = rot(dummy, pos_ids)
    states = make_zero_states(cfg, max_seq)
    return (input_token, position, cos, sin, *states)


# ---- parity check ---------------------------------------------------------


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def torch_parity(model, oracle, cfg, rot, max_seq, n_prompts=3):
    """Run token-by-token, compare last-position logits vs oracle's recurrent."""
    print(f"\n=== torch decode parity (first {n_prompts} prompts) ===")
    for pi, rec in enumerate(oracle["records"][:n_prompts]):
        ids = rec["input_ids"]
        S = ids.shape[1]
        # Build states fresh
        states = make_zero_states(cfg, max_seq)
        last_logits = None
        with torch.no_grad():
            for t in range(S):
                tok = ids[:, t:t+1].to(torch.int32)
                pos = torch.tensor([float(t)], dtype=torch.float32)
                pos_ids = torch.tensor([[t]], dtype=torch.long)
                dummy = torch.zeros(1, 1, cfg.hidden_size)
                cos_t, sin_t = rot(dummy, pos_ids)
                out = model(tok, pos, cos_t.float(), sin_t.float(), *[s.float() for s in states])
                logits, *new_states = out
                states = list(new_states)
                if t == S - 1:
                    last_logits = logits[0, 0].float()
        ref = rec["logits_recurrent"][-1].float()   # (V,)
        c = cos_sim(last_logits, ref)
        top1 = int(torch.argmax(last_logits).item())
        ref_top1 = int(rec["top10_last_ids"][0].item())
        print(f"  prompt[{pi}] S={S}  cos={c:.6f}  top1_match={top1 == ref_top1}  "
              f"{rec['prompt'][:30]!r}")


# ---- conversion -----------------------------------------------------------


def convert_and_audit(model, cfg, rot, max_seq, out_path):
    print(f"\n=== CoreML conversion (max_seq={max_seq}) ===")
    example = make_example_inputs(cfg, max_seq, rot)
    traced = torch.jit.trace(model, example, strict=False)
    print("  trace OK")

    # Build CoreML input spec
    n_layers = cfg.num_hidden_layers
    ct_inputs = [
        ct.TensorType(name="input_token", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="position", shape=(1,), dtype=np.float32),
        ct.TensorType(name="cos", shape=example[2].shape, dtype=np.float16),
        ct.TensorType(name="sin", shape=example[3].shape, dtype=np.float16),
    ]
    ct_outputs = [
        ct.TensorType(name="logits", dtype=np.float32),
    ]
    state_offset = 4
    for i in range(n_layers):
        sa = example[state_offset + 2 * i]
        sb = example[state_offset + 2 * i + 1]
        ct_inputs.append(ct.TensorType(name=f"state_{i}_a", shape=sa.shape, dtype=np.float16))
        ct_inputs.append(ct.TensorType(name=f"state_{i}_b", shape=sb.shape, dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_state_{i}_a", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_state_{i}_b", dtype=np.float16))

    # Keep the lm_head matmul in fp32. Reason: ANE fp16 hardware accumulates
    # over the 1024-dim reduction of this projection to a 248320-entry output,
    # which is where per-step top-1 drift comes from (40% match when all fp16).
    # Forcing fp32 makes coremltools route THIS op to CPU while the 24 decoder
    # layers (much smaller matmuls, fp16-safe) stay on ANE. Hybrid
    # ANE-body + CPU-head aims to preserve ANE speed while keeping lm_head
    # accuracy.
    from coremltools.converters.mil.mil.passes.defs.quantization import (
        FP16ComputePrecision,
    )
    VOCAB = cfg.vocab_size
    def _keep_lm_head_fp32(op):
        # Return True → cast this op to fp16. False → keep fp32.
        if op.op_type == "linear":
            for out in op.outputs:
                # MIL Operation.outputs[i].shape may be a tuple of ints or a
                # shape with dynamic dims; guard by length and last-dim equality.
                shape = getattr(out, "shape", None)
                if shape is not None and len(shape) >= 1:
                    last = shape[-1]
                    if isinstance(last, int) and last == VOCAB:
                        return False
        return True
    ct_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_precision=FP16ComputePrecision(op_selector=_keep_lm_head_fp32),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print("  convert OK")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved {out_path} ({size_mb:.1f} MB)")

    print("\n=== placement audit ===")
    reloaded = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    dev_counts: Counter = Counter()
    op_type_by_dev: dict[str, Counter] = {}
    for func_name, func in program.functions.items():
        for op in func.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if a is None:
                dev = "const" if op.operator_name == "const" else "unknown"
            else:
                dev = a.preferred_compute_device.__class__.__name__
            dev_counts[dev] += 1
            op_type_by_dev.setdefault(dev, Counter())[op.operator_name] += 1
    total = sum(dev_counts.values())
    ane = dev_counts.get("MLNeuralEngineComputeDevice", 0)
    cpu = dev_counts.get("MLCPUComputeDevice", 0)
    const = dev_counts.get("const", 0)
    compute = total - const
    print(f"  total: {total}  compute: {compute}  const: {const}")
    print(f"    ANE: {ane} ({100*ane/compute:.2f}% of compute)")
    print(f"    CPU: {cpu}")
    for dev in ("MLCPUComputeDevice", "MLNeuralEngineComputeDevice"):
        c = op_type_by_dev.get(dev, Counter())
        if not c: continue
        print(f"  === {dev} top ops ===")
        for op_type, n in c.most_common(12):
            print(f"    {op_type}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    args = ap.parse_args()

    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    hf = Qwen3_5ForCausalLM.from_pretrained(MODEL_ID, config=cfg, torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    model = FullDecodeModel(cfg, hf, args.max_seq).eval().float()
    del hf

    oracle = torch.load(str(ORACLE), map_location="cpu", weights_only=False)
    torch_parity(model, oracle, cfg, rot, args.max_seq, n_prompts=3)

    if args.skip_convert:
        return

    out_dir = Path(args.out_dir or tempfile.mkdtemp(prefix="qwen35_decode_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qwen3_5_0_8b_decode_fp16_mseq{args.max_seq}.mlpackage"
    convert_and_audit(model, cfg, rot, args.max_seq, out_path)


if __name__ == "__main__":
    main()
