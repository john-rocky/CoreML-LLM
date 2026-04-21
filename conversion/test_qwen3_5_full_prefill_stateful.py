"""Phase 4e-2: stateful prefill converter.

Extends the Phase 4a FullModel so prefill also outputs the initial decode
states, letting Swift chain prefill -> decode without zero-init warmup.

State outputs per linear_attention layer (18 layers):
  conv_state (1, 6144, K=4)        last K positions of pre-silu conv input
  rec_state  (1, 16, Dk=128, Dv=128)  final last_state from chunked rule
State outputs per full_attention layer (6 layers), padded to max_seq=128:
  k_cache (1, num_kv=2, max_seq, head_dim=256)
  v_cache (1, num_kv=2, max_seq, head_dim=256)

After prefill at seq=S_real, positions 0..S_real-1 hold real K/V and
positions S_real..max_seq-1 are zero. Decode then starts at position S_real.

Parity: outputs logits identical to Phase 4a fwdsub version (same math
for the hidden path; only extra outputs added).
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

# Reuse math constants / the fwdsub-patched PrefillLinearAttnLayer math.
from test_qwen3_5_prefill_trace import CHUNK_SIZE


MODEL_ID = "Qwen/Qwen3.5-0.8B"
ORACLE = Path(__file__).parent / "qwen3_5_reference_logits.pt"
MAX_SEQ = 128


class StatefulRMSNorm(nn.Module):
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


class StatefulMLP(nn.Module):
    def __init__(self, gate_w, up_w, down_w):
        super().__init__()
        self.gate_w = nn.Parameter(gate_w.detach().clone(), requires_grad=False)
        self.up_w = nn.Parameter(up_w.detach().clone(), requires_grad=False)
        self.down_w = nn.Parameter(down_w.detach().clone(), requires_grad=False)

    def forward(self, x):
        g = F.silu(F.linear(x, self.gate_w))
        u = F.linear(x, self.up_w)
        return F.linear(g * u, self.down_w)


class StatefulPrefillLinearAttn(nn.Module):
    """Prefill linear_attention with forward-substitution (I-L)^-1 and state
    outputs. Math mirrors test_qwen3_5_prefill_trace.PrefillLinearAttnLayer
    exactly — only adds conv_state + rec_state outputs."""

    def __init__(self, cfg, hf_layer, seq_len: int):
        super().__init__()
        assert seq_len % CHUNK_SIZE == 0
        self.S = seq_len
        self.num_chunks = seq_len // CHUNK_SIZE
        self.CS = CHUNK_SIZE
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

        self.register_buffer("chunk_eye",
                              torch.eye(CHUNK_SIZE, dtype=torch.float32),
                              persistent=False)

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

    def _chunk_gated_delta_rule(self, q, k, v, g, beta):
        B, H, S, CS, NC, Dk, Dv = 1, self.num_v, self.S, self.CS, self.num_chunks, self.Dk, self.Dv
        BN = B * H * NC
        scale = 1.0 / math.sqrt(Dk)
        q = q * scale
        v_beta = v * beta.unsqueeze(-1)
        k_beta = k * beta.unsqueeze(-1)
        q = q.reshape(B, H, NC, CS, Dk)
        k = k.reshape(B, H, NC, CS, Dk)
        v = v.reshape(B, H, NC, CS, Dv)
        k_beta = k_beta.reshape(B, H, NC, CS, Dk)
        v_beta = v_beta.reshape(B, H, NC, CS, Dv)
        g = g.reshape(B, H, NC, CS).cumsum(dim=-1)

        g_diff = g.unsqueeze(-1) - g.unsqueeze(-2)
        g_diff = g_diff.clamp(max=10.0)
        tril = torch.tril(torch.ones(CS, CS, dtype=torch.bool, device=q.device))
        decay_mask = torch.where(tril, g_diff.exp(), torch.zeros_like(g_diff))

        upper = torch.triu(torch.ones(CS, CS, dtype=torch.bool, device=q.device), diagonal=0)
        k3 = k.reshape(BN, CS, Dk)
        k_beta3 = k_beta.reshape(BN, CS, Dk)
        v_beta3 = v_beta.reshape(BN, CS, Dv)
        L3 = -torch.bmm(k_beta3, k3.transpose(-1, -2)) * decay_mask.reshape(BN, CS, CS)
        L3 = torch.where(upper, torch.zeros_like(L3), L3)

        # Forward substitution (see Phase 4b notes)
        eye = self.chunk_eye
        rows = [eye[0, :].unsqueeze(0).unsqueeze(0).expand(BN, 1, CS)]
        for i in range(1, CS):
            prev = torch.cat(rows, dim=1)
            L_row = L3[:, i:i+1, :i]
            delta = torch.bmm(L_row, prev)
            row_i = eye[i, :].unsqueeze(0).unsqueeze(0).expand(BN, 1, CS) + delta
            rows.append(row_i)
        attn3 = torch.cat(rows, dim=1)

        value = torch.bmm(attn3, v_beta3).reshape(B, H, NC, CS, Dv)
        k_decay = (k_beta * g.exp().unsqueeze(-1)).reshape(BN, CS, Dk)
        k_cumdecay = torch.bmm(attn3, k_decay).reshape(B, H, NC, CS, Dk)

        last_state = torch.zeros(B, H, Dk, Dv, device=q.device, dtype=q.dtype)
        strict_upper = torch.triu(torch.ones(CS, CS, dtype=torch.bool, device=q.device), diagonal=1)
        core_chunks = []
        for i in range(NC):
            q_i = q[:, :, i]; k_i = k[:, :, i]
            v_i = value[:, :, i]
            attn_i = (q_i @ k_i.transpose(-1, -2)) * decay_mask[:, :, i]
            attn_i = torch.where(strict_upper, torch.zeros_like(attn_i), attn_i)
            v_prime = k_cumdecay[:, :, i] @ last_state
            v_new = v_i - v_prime
            g_i = g[:, :, i]
            attn_inter = (q_i * g_i.unsqueeze(-1).exp()) @ last_state
            out_i = attn_inter + attn_i @ v_new
            core_chunks.append(out_i)
            g_last = g_i[..., -1:].unsqueeze(-1)
            decay_k = (g_last.squeeze(-1) - g_i).exp().unsqueeze(-1)
            last_state = last_state * g_last.exp() + (k_i * decay_k).transpose(-1, -2) @ v_new

        core_attn_out = torch.stack(core_chunks, dim=2).reshape(B, H, S, Dv)
        return core_attn_out, last_state

    def forward(self, hidden_in):
        mixed_qkv = F.linear(hidden_in, self.in_proj_qkv_w)
        z = F.linear(hidden_in, self.in_proj_z_w)
        b = F.linear(hidden_in, self.in_proj_b_w)
        a = F.linear(hidden_in, self.in_proj_a_w)

        mixed_qkv_t = mixed_qkv.transpose(1, 2)                      # (1,C,S)
        # conv_state for decode handoff: last K positions of pre-silu conv input
        conv_state_out = mixed_qkv_t[:, :, -self.K:].contiguous()    # (1, C, K)

        conv_out = F.conv1d(mixed_qkv_t, self.conv_w, bias=None,
                             groups=self.conv_dim, padding=self.K - 1)
        conv_out = F.silu(conv_out[:, :, :self.S])
        mixed_qkv = conv_out.transpose(1, 2)

        q, k, v = torch.split(mixed_qkv,
                               [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, self.S, self.num_k, self.Dk)
        k = k.reshape(1, self.S, self.num_k, self.Dk)
        v = v.reshape(1, self.S, self.num_v, self.Dv)

        beta = torch.sigmoid(b).float()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        q_ = self._l2norm(q.float().transpose(1, 2).contiguous())
        k_ = self._l2norm(k.float().transpose(1, 2).contiguous())
        v_ = v.float().transpose(1, 2).contiguous()
        beta_ = beta.transpose(1, 2).contiguous()
        g_ = g.transpose(1, 2).contiguous()

        core_out, rec_state_out = self._chunk_gated_delta_rule(q_, k_, v_, g_, beta_)
        core_out = core_out.transpose(1, 2).contiguous().to(hidden_in.dtype)

        core_flat = core_out.reshape(-1, self.Dv)
        z_flat = z.reshape(-1, self.Dv)
        out_flat = self._rmsnorm_gated(core_flat, z_flat)
        out = out_flat.reshape(1, self.S, self.value_dim)
        hidden_out = F.linear(out, self.out_proj_w)

        return hidden_out, conv_state_out.to(hidden_in.dtype), rec_state_out.to(hidden_in.dtype)


class StatefulPrefillFullAttn(nn.Module):
    """Prefill full_attention emitting K/V cache padded to max_seq."""

    def __init__(self, cfg, hf_attn, seq_len: int, max_seq: int):
        super().__init__()
        self.S = seq_len
        self.max_seq = max_seq
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        self.eps = cfg.rms_norm_eps

        self.q_proj_w = nn.Parameter(hf_attn.q_proj.weight.detach().clone(), requires_grad=False)
        self.k_proj_w = nn.Parameter(hf_attn.k_proj.weight.detach().clone(), requires_grad=False)
        self.v_proj_w = nn.Parameter(hf_attn.v_proj.weight.detach().clone(), requires_grad=False)
        self.o_proj_w = nn.Parameter(hf_attn.o_proj.weight.detach().clone(), requires_grad=False)
        self.q_norm_w = nn.Parameter(hf_attn.q_norm.weight.detach().clone(), requires_grad=False)
        self.k_norm_w = nn.Parameter(hf_attn.k_norm.weight.detach().clone(), requires_grad=False)

        causal = torch.full((seq_len, seq_len), -1e4, dtype=torch.float32)
        causal = torch.triu(causal, diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def _rmsnorm(self, x, w):
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

    def forward(self, hidden_in, cos, sin):
        S = self.S
        qg = F.linear(hidden_in, self.q_proj_w)
        k = F.linear(hidden_in, self.k_proj_w)
        v = F.linear(hidden_in, self.v_proj_w)

        qg = qg.reshape(1, S, self.num_heads, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(1, S, self.num_heads * self.head_dim)

        q = self._rmsnorm(q, self.q_norm_w).transpose(1, 2)
        k = k.reshape(1, S, self.num_kv_heads, self.head_dim)
        k = self._rmsnorm(k, self.k_norm_w).transpose(1, 2)       # (1, num_kv, S, head_dim)
        v = v.reshape(1, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self._apply_rope(q, k, cos, sin)                   # k: (1, num_kv, S, head_dim)

        # Export K/V cache: pad to max_seq with zeros at the tail
        pad_len = self.max_seq - S
        if pad_len > 0:
            pad_shape = (1, self.num_kv_heads, pad_len, self.head_dim)
            zpad = torch.zeros(pad_shape, dtype=k.dtype, device=k.device)
            k_cache_out = torch.cat([k, zpad], dim=2)
            v_cache_out = torch.cat([v, zpad], dim=2)
        else:
            k_cache_out = k
            v_cache_out = v

        if self.num_kv_groups > 1:
            k_r = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_r = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_r = k
            v_r = v

        attn_scores = q @ k_r.transpose(-1, -2) * self.scale
        attn_scores = attn_scores + self.causal_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ v_r

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(
            1, S, self.num_heads * self.head_dim
        )
        attn_out = attn_out * torch.sigmoid(gate)
        hidden_out = F.linear(attn_out, self.o_proj_w)

        return (hidden_out,
                k_cache_out.to(hidden_in.dtype),
                v_cache_out.to(hidden_in.dtype))


class StatefulDecoderLayer(nn.Module):
    def __init__(self, cfg, hf_layer, seq_len: int, max_seq: int):
        super().__init__()
        self.layer_type = hf_layer.layer_type
        self.input_norm = StatefulRMSNorm(cfg.rms_norm_eps, hf_layer.input_layernorm.weight)
        self.post_attn_norm = StatefulRMSNorm(cfg.rms_norm_eps, hf_layer.post_attention_layernorm.weight)
        self.mlp = StatefulMLP(hf_layer.mlp.gate_proj.weight,
                                hf_layer.mlp.up_proj.weight,
                                hf_layer.mlp.down_proj.weight)
        if self.layer_type == "linear_attention":
            self.mixer = StatefulPrefillLinearAttn(cfg, hf_layer, seq_len)
        else:
            self.mixer = StatefulPrefillFullAttn(cfg, hf_layer.self_attn, seq_len, max_seq)

    def forward(self, hidden, cos, sin):
        residual = hidden
        h = self.input_norm(hidden)
        if self.layer_type == "linear_attention":
            h, s_a, s_b = self.mixer(h)
        else:
            h, s_a, s_b = self.mixer(h, cos, sin)
        hidden = residual + h
        residual = hidden
        h = self.post_attn_norm(hidden)
        h = self.mlp(h)
        return residual + h, s_a, s_b


class FullPrefillModel(nn.Module):
    def __init__(self, cfg, hf_model, seq_len: int, max_seq: int):
        super().__init__()
        self.S = seq_len
        self.max_seq = max_seq
        self.eps = cfg.rms_norm_eps
        self.embed_w = nn.Parameter(hf_model.model.embed_tokens.weight.detach().clone(),
                                     requires_grad=False)
        self.final_norm = StatefulRMSNorm(cfg.rms_norm_eps, hf_model.model.norm.weight)
        self.lm_head_w = nn.Parameter(hf_model.lm_head.weight.detach().clone(),
                                       requires_grad=False)
        self.layers = nn.ModuleList([
            StatefulDecoderLayer(cfg, hf_model.model.layers[i], seq_len, max_seq)
            for i in range(cfg.num_hidden_layers)
        ])
        rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        dummy = torch.zeros(1, seq_len, cfg.hidden_size)
        with torch.no_grad():
            cos, sin = rot(dummy, pos)
        self.register_buffer("cos", cos.detach().clone(), persistent=False)
        self.register_buffer("sin", sin.detach().clone(), persistent=False)

    def forward(self, input_ids):
        hidden = F.embedding(input_ids.to(torch.long), self.embed_w)
        states = []
        for layer in self.layers:
            hidden, s_a, s_b = layer(hidden, self.cos, self.sin)
            states.append(s_a); states.append(s_b)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.lm_head_w)
        return (logits, *states)


# ---- parity against Phase 1 oracle (fp32) --------------------------------


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def parity(model, oracle, seq_len):
    print(f"\n=== parity vs oracle (seq={seq_len}) ===")
    for rec in oracle["records"]:
        ids = rec["input_ids"]; S = ids.shape[1]
        if S > seq_len: continue
        padded = torch.zeros(1, seq_len, dtype=ids.dtype)
        padded[:, :S] = ids
        with torch.no_grad():
            out = model(padded)
            logits = out[0]
        per_pos = torch.tensor([cos_sim(logits[0, i], rec["logits_prefill"][i])
                                 for i in range(S)])
        top1 = int(torch.argmax(logits[0, S-1]).item())
        match = top1 == int(rec["top10_last_ids"][0].item())
        print(f"  S={S:3d}  mean={per_pos.mean():.6f}  worst={per_pos.min():.6f}  "
              f"top1_match={match}  {rec['prompt'][:30]!r}")


# ---- convert ---------------------------------------------------------------


def convert(model, cfg, seq_len, max_seq, out_path):
    print(f"\n=== convert (seq={seq_len}, max_seq={max_seq}) ===")
    example = (torch.zeros(1, seq_len, dtype=torch.int32),)
    traced = torch.jit.trace(model, example, strict=False)
    print("  trace OK")

    N = cfg.num_hidden_layers
    ct_inputs = [ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32)]
    ct_outputs = [ct.TensorType(name="logits", dtype=np.float32)]

    # Run a sample forward to learn per-layer state shapes
    with torch.no_grad():
        sample = model(example[0])
    # sample = (logits, s0_a, s0_b, s1_a, s1_b, ...)
    for i in range(N):
        s_a = sample[1 + 2*i]
        s_b = sample[2 + 2*i]
        ct_outputs.append(ct.TensorType(name=f"state_{i}_a", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"state_{i}_b", dtype=np.float16))

    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print("  convert OK")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved {out_path} ({size_mb:.1f} MB)")

    reloaded = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    dev_counts: Counter = Counter()
    for func_name, func in program.functions.items():
        for op in func.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = "const" if (a is None and op.operator_name == "const") \
                  else (a.preferred_compute_device.__class__.__name__ if a else "unknown")
            dev_counts[dev] += 1
    total = sum(dev_counts.values())
    ane = dev_counts.get("MLNeuralEngineComputeDevice", 0)
    cpu = dev_counts.get("MLCPUComputeDevice", 0)
    const = dev_counts.get("const", 0)
    compute = total - const
    print(f"  total={total} compute={compute} const={const}")
    print(f"  ANE: {ane} ({100*ane/compute:.2f}% of compute)  CPU: {cpu}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    hf = Qwen3_5ForCausalLM.from_pretrained(MODEL_ID, config=cfg, torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    model = FullPrefillModel(cfg, hf, args.seq_len, args.max_seq).eval().float()
    del hf
    print(f"  params={sum(p.numel() for p in model.parameters())/1e9:.3f}B")

    oracle = torch.load(str(ORACLE), map_location="cpu", weights_only=False)
    parity(model, oracle, args.seq_len)

    if args.skip_convert: return
    out_dir = Path(args.out_dir or tempfile.mkdtemp(prefix="qwen35_prefill_stateful_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qwen3_5_0_8b_prefill_stateful_fp16_seq{args.seq_len}.mlpackage"
    convert(model, cfg, args.seq_len, args.max_seq, out_path)


if __name__ == "__main__":
    main()
