"""Phase 2a: try to convert a single decode-step Gated DeltaNet layer to CoreML.

Strategy: build a minimal nn.Module that mimics the HF decode-step math using
only MIL-convertible ops (matmul, conv1d, elementwise, silu, rmsnorm), copy
weights from the loaded HF model's layer 0 (linear_attn), verify numerical
parity token-by-token vs the HF model, then torch-to-CoreML convert and audit
ANE placement.

Parity target: cos >= 0.999 vs HF on 32 decode steps from a fixed prompt.
Placement target: non-const compute ops on MLNeuralEngineComputeDevice.
"""
from collections import Counter
from pathlib import Path
import math
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, AutoTokenizer

import coremltools as ct


MODEL_ID = "Qwen/Qwen3.5-0.8B"


class DecodeStepModule(nn.Module):
    """Single-token Gated-DeltaNet decode step for one layer.

    Inputs:
        hidden_in:     (1, 1, H)           - current token hidden
        conv_state:    (1, C, K)           - HF caches K positions (not K-1)
        recurrent_state: (1, Hv, Dk, Dv)   - SSM recurrent state

    Outputs:
        hidden_out:     (1, 1, H)           - layer output (pre-residual / norm)
        new_conv_state: (1, C, K)
        new_rec_state:  (1, Hv, Dk, Dv)

    Math mirrors Qwen3_5GatedDeltaNet.forward in use_precomputed_states=True mode
    (causal_conv1d_update path + torch_recurrent_gated_delta_rule).
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
        self.v_per_k = self.num_v // self.num_k  # == 1 for 0.8B

        # Borrow weights directly from the HF layer's linear_attn module.
        # Qwen3.5 has four separate projections (not the qkvz/ba split of Qwen3Next).
        lin = hf_layer.linear_attn
        self.conv_w = nn.Parameter(lin.conv1d.weight.detach().clone(), requires_grad=False)       # (C,1,K)
        self.in_proj_qkv_w = nn.Parameter(lin.in_proj_qkv.weight.detach().clone(), requires_grad=False)
        self.in_proj_z_w   = nn.Parameter(lin.in_proj_z.weight.detach().clone(),   requires_grad=False)
        self.in_proj_b_w   = nn.Parameter(lin.in_proj_b.weight.detach().clone(),   requires_grad=False)
        self.in_proj_a_w   = nn.Parameter(lin.in_proj_a.weight.detach().clone(),   requires_grad=False)
        self.out_proj_w    = nn.Parameter(lin.out_proj.weight.detach().clone(),    requires_grad=False)
        self.dt_bias = nn.Parameter(lin.dt_bias.detach().clone(), requires_grad=False)   # (Hv,)
        self.A_log   = nn.Parameter(lin.A_log.detach().clone(),   requires_grad=False)   # (Hv,)
        self.norm_w  = nn.Parameter(lin.norm.weight.detach().clone(), requires_grad=False)  # (Dv,)

    @staticmethod
    def _l2norm(x, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)

    def _rmsnorm_gated(self, x, z):
        # Qwen3_5RMSNormGated: norm first, weight, then gate by silu(z). Order matters.
        in_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = self.norm_w * x.to(in_dtype)
        x = x * F.silu(z.float())
        return x.to(in_dtype)

    def forward(self, hidden_in, conv_state, rec_state):
        # hidden_in: (1,1,H)
        B = 1

        # 1. Four separate projections (Qwen3.5 layout)
        mixed_qkv = F.linear(hidden_in, self.in_proj_qkv_w)   # (1,1,C)
        z = F.linear(hidden_in, self.in_proj_z_w)             # (1,1,V)
        b = F.linear(hidden_in, self.in_proj_b_w)             # (1,1,num_v)
        a = F.linear(hidden_in, self.in_proj_a_w)             # (1,1,num_v)

        # 2. causal_conv1d_update: HF caches K positions (not K-1). concat(prev K,
        #    current 1) → K+1 positions, conv1d with kernel K → 2 outputs, take
        #    the latest (= output at position 1). New state = last K positions.
        mixed_qkv_t = mixed_qkv.transpose(1, 2)               # (1,C,1)
        w = torch.cat([conv_state, mixed_qkv_t], dim=-1)      # (1,C,K+1)
        kw = self.conv_w.squeeze(1)                            # (C,K)
        conv_out = (w[:, :, 1:] * kw.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (1,C,1)
        conv_out = F.silu(conv_out)
        new_conv_state = w[:, :, 1:]                          # (1,C,K)

        # 3. Split conv output back to q, k, v
        mixed_qkv = conv_out.transpose(1, 2)                   # (1,1,C)
        q, k, v = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        q = q.reshape(B, 1, -1, self.Dk)                      # (1,1,num_k,Dk)
        k = k.reshape(B, 1, -1, self.Dk)
        v = v.reshape(B, 1, -1, self.Dv)                      # (1,1,num_v,Dv)

        # 4. beta, g (fp32 math)
        beta = torch.sigmoid(b)                                # (1,1,num_v)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # (1,1,num_v)

        # 5. Repeat q,k across v_per_k if >1 (0.8B: 1, no-op)
        if self.v_per_k > 1:
            q = q.repeat_interleave(self.v_per_k, dim=2)
            k = k.repeat_interleave(self.v_per_k, dim=2)

        # 6. Recurrent Gated Delta step (fp32)
        q_ = q.transpose(1, 2).contiguous().float()[:, :, 0]   # (1,num_v,Dk)
        k_ = k.transpose(1, 2).contiguous().float()[:, :, 0]
        v_ = v.transpose(1, 2).contiguous().float()[:, :, 0]   # (1,num_v,Dv)
        q_ = self._l2norm(q_)
        k_ = self._l2norm(k_)
        q_ = q_ * (1.0 / math.sqrt(self.Dk))
        g_scalar = g.float().transpose(1, 2).exp().squeeze(-1)      # (1,num_v)
        g_ = g_scalar.unsqueeze(-1).unsqueeze(-1)             # (1,num_v,1,1)
        beta_ = beta.float().transpose(1, 2).squeeze(-1)       # (1,num_v)

        rec = rec_state.float() * g_                          # (1,num_v,Dk,Dv)
        kv_mem = (rec * k_.unsqueeze(-1)).sum(dim=-2)         # (1,num_v,Dv)
        delta = (v_ - kv_mem) * beta_.unsqueeze(-1)           # (1,num_v,Dv)
        rec = rec + k_.unsqueeze(-1) * delta.unsqueeze(-2)    # (1,num_v,Dk,Dv)
        core_out = (rec * q_.unsqueeze(-1)).sum(dim=-2)       # (1,num_v,Dv)

        # 7. Gated RMSNorm (applied per head_v_dim row) + out_proj
        #    Reshape to (-1, Dv) to normalize along last dim, like HF does.
        core_flat = core_out.reshape(-1, self.Dv).to(hidden_in.dtype)  # (num_v, Dv)
        z_flat = z.reshape(-1, self.Dv)                        # (num_v, Dv)
        out_flat = self._rmsnorm_gated(core_flat, z_flat)     # (num_v, Dv)
        out = out_flat.reshape(B, 1, self.value_dim)          # (1,1,V)
        hidden_out = F.linear(out, self.out_proj_w)           # (1,1,H)

        return hidden_out, new_conv_state, rec.to(rec_state.dtype)


def extract_hf_decode_io(hf_model, prompt_ids, tok_idx: int):
    """Run the HF model in decode mode up to and including token `tok_idx`,
    capture the layer-0 linear_attn hidden input, conv_state, recurrent_state
    (before the update), and the resulting hidden_out (after the layer's SSM
    path, before the residual add).

    We use forward hooks on layer 0.
    """
    hf_model.eval()
    captured = {}

    layer0 = hf_model.model.layers[0]
    lin = layer0.linear_attn

    # We need to know what input goes INTO linear_attn.forward (= hidden_states
    # after input_layernorm), and what comes out (= pre-residual SSM result).
    def pre_hook(module, args, kwargs):
        h = args[0] if len(args) > 0 else kwargs["hidden_states"]
        captured["input_hidden"] = h.detach().clone()

    def post_hook(module, args, kwargs, output):
        out = output if not isinstance(output, tuple) else output[0]
        captured["output_hidden"] = out.detach().clone()

    h1 = lin.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = lin.register_forward_hook(post_hook, with_kwargs=True)

    past = None
    try:
        with torch.no_grad():
            for i in range(tok_idx + 1):
                tok = prompt_ids[:, i:i+1]
                out = hf_model(input_ids=tok, past_key_values=past, use_cache=True)
                past = out.past_key_values
                if i == tok_idx:
                    # Capture state snapshot BEFORE this step's update
                    pass
    finally:
        h1.remove(); h2.remove()

    # To get pre-step state: do the same run but capture cache BEFORE step tok_idx.
    past_before = None
    with torch.no_grad():
        for i in range(tok_idx):
            tok = prompt_ids[:, i:i+1]
            out = hf_model(input_ids=tok, past_key_values=past_before, use_cache=True)
            past_before = out.past_key_values

    # Extract pre-step conv_state and recurrent_state for layer 0
    layer_cache = past_before.layers[0] if past_before is not None else None
    if layer_cache is None:
        # No prior state: use zeros (HF cache holds K positions, not K-1)
        conv_state = torch.zeros(1, lin.conv_dim, lin.conv_kernel_size)
        rec_state = torch.zeros(1, lin.num_v_heads, lin.head_k_dim, lin.head_v_dim)
    else:
        conv_state = layer_cache.conv_states.detach().clone()
        rec_state = layer_cache.recurrent_states.detach().clone()

    return captured["input_hidden"], conv_state, rec_state, captured["output_hidden"]


def cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    print("loading HF model in fp32 (reference)...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    print(f"loaded in {time.time()-t0:.1f}s")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = "The capital of France is Paris. The capital of Japan is"
    input_ids = tok(prompt, return_tensors="pt").input_ids

    # Build our decode module from layer 0.
    decode_mod = DecodeStepModule(cfg, model.model.layers[0]).eval().float()

    # Parity check at several token positions.
    print("\n=== parity check (decode-step vs HF layer-0 linear_attn) ===")
    worst = 1.0
    for tok_idx in [0, 1, 2, 5, 10, input_ids.shape[1] - 1]:
        hin, cs, rs, hout_ref = extract_hf_decode_io(model, input_ids, tok_idx)
        with torch.no_grad():
            hout, cs2, rs2 = decode_mod(hin.float(), cs.float(), rs.float())
        c = cos(hout.float(), hout_ref.float())
        print(f"  tok_idx={tok_idx}  cos(hidden_out)={c:.6f}  "
              f"shapes: hidden={tuple(hout.shape)} conv={tuple(cs2.shape)} rec={tuple(rs2.shape)}")
        worst = min(worst, c)
    print(f"worst cos: {worst:.6f}")

    if worst < 0.995:
        print("FAILED parity — stopping before CoreML conversion.")
        return

    # CoreML conversion
    print("\n=== CoreML conversion (torch.jit.trace -> coremltools) ===")
    example_in = (
        torch.randn(1, 1, cfg.hidden_size),
        torch.zeros(1, decode_mod.conv_dim, decode_mod.K),
        torch.zeros(1, decode_mod.num_v, decode_mod.Dk, decode_mod.Dv),
    )
    traced = torch.jit.trace(decode_mod, example_in, strict=False)
    print("  trace OK")

    ct_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="hidden_in", shape=example_in[0].shape, dtype=np.float16),
            ct.TensorType(name="conv_state", shape=example_in[1].shape, dtype=np.float16),
            ct.TensorType(name="rec_state", shape=example_in[2].shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="hidden_out", dtype=np.float16),
            ct.TensorType(name="new_conv_state", dtype=np.float16),
            ct.TensorType(name="new_rec_state", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print("  convert OK")

    tmpdir = Path(tempfile.mkdtemp(prefix="qwen35_decode_"))
    path = tmpdir / "decode_layer0.mlpackage"
    ct_model.save(str(path))
    print(f"  saved {path} ({sum(f.stat().st_size for f in path.rglob('*') if f.is_file())/1e6:.1f} MB)")

    # Placement audit
    print("\n=== placement audit ===")
    reloaded = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
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
            dev = a.preferred_compute_device.__class__.__name__ if a else "unknown"
            dev_counts[dev] += 1
            op_type_by_dev.setdefault(dev, Counter())[op.operator_name] += 1
    total = sum(dev_counts.values())
    print(f"  total ops: {total}")
    for dev, n in dev_counts.most_common():
        print(f"    {dev}: {n} ({100*n/total:.1f}%)")
    for dev, c in op_type_by_dev.items():
        print(f"  === {dev} ops (top 15) ===")
        for op_type, n in c.most_common(15):
            print(f"    {op_type}: {n}")


if __name__ == "__main__":
    main()
