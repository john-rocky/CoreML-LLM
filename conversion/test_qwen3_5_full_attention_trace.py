"""Phase 3 step 1: verify the full_attention layer (Qwen3_5Attention) converts
to CoreML with high ANE placement.

This is the 6-of-24 layer type that isn't Gated DeltaNet. It is:
- q_proj (double-width for gate-fold) → chunk → q, gate
- k_proj, v_proj
- q_norm, k_norm (per-head RMSNorm)
- partial RoPE on first 64 of 256 head dims (rope_theta=10M)
- causal softmax attention (eager) or SDPA iOS18
- output gate: attn * sigmoid(gate)
- o_proj

For text-only prefill, MRoPE collapses to plain RoPE — we precompute cos/sin
via the HF rotary module with identical T/H/W position_ids.
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
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import coremltools as ct


MODEL_ID = "Qwen/Qwen3.5-0.8B"


class FullAttentionLayer(nn.Module):
    """Qwen3_5Attention prefill forward at fixed seq_len.
    Takes (hidden_in, cos, sin) → hidden_out. cos/sin are precomputed from the
    rotary module with text-only position_ids."""

    def __init__(self, cfg, hf_attn, seq_len: int):
        super().__init__()
        self.S = seq_len
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

        # Precomputed causal mask (additive). Buffer avoids trace-time int/cast
        # weirdness and stays as a constexpr in the MIL graph.
        causal = torch.full((seq_len, seq_len), -1e4, dtype=torch.float32)
        causal = torch.triu(causal, diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def _rmsnorm(self, x, w):
        # x: (..., head_dim). Qwen3_5RMSNorm uses (1 + w).
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
        # q, k shape: (B, H, S, head_dim). cos, sin shape: (B, S, rotary_dim).
        # Use the fixed rotary_dim from __init__ to avoid shape() ops during trace.
        rd = int(self.head_dim * 0.25)   # partial_rotary_factor = 0.25
        cos = cos.unsqueeze(1)  # (B, 1, S, rotary_dim)
        sin = sin.unsqueeze(1)
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_out = q_rot * cos + self._rotate_half(q_rot) * sin
        k_out = k_rot * cos + self._rotate_half(k_rot) * sin
        q = torch.cat([q_out, q_pass], dim=-1)
        k = torch.cat([k_out, k_pass], dim=-1)
        return q, k

    def forward(self, hidden_in, cos, sin):
        """hidden_in: (1, S, H), cos/sin: (1, S, rotary_dim=64). S fixed at init."""
        S = self.S

        # 1. Projections
        qg = F.linear(hidden_in, self.q_proj_w)              # (1, S, num_heads*head_dim*2)
        k = F.linear(hidden_in, self.k_proj_w)                # (1, S, num_kv_heads*head_dim)
        v = F.linear(hidden_in, self.v_proj_w)                # (1, S, num_kv_heads*head_dim)

        # 2. Split q from gate (second half of each head's doubled output)
        qg = qg.reshape(1, S, self.num_heads, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)                         # (1, S, num_heads, head_dim) each
        gate = gate.reshape(1, S, self.num_heads * self.head_dim)

        # 3. q_norm / k_norm on head_dim, transpose to (B, H, S, head_dim)
        q = self._rmsnorm(q, self.q_norm_w).transpose(1, 2)   # (1, num_heads, S, head_dim)
        k = k.reshape(1, S, self.num_kv_heads, self.head_dim)
        k = self._rmsnorm(k, self.k_norm_w).transpose(1, 2)   # (1, num_kv_heads, S, head_dim)
        v = v.reshape(1, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 4. RoPE
        q, k = self._apply_rope(q, k, cos, sin)

        # 5. Repeat-kv if GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # 6. Softmax attention (eager). Uses a precomputed additive causal mask.
        # Avoid tensor.dtype / to(dtype) reads during trace — causal_mask is
        # already stored in the model dtype (float at trace time).
        attn_scores = q @ k.transpose(-1, -2) * self.scale                    # (B, num_heads, S, S)
        attn_scores = attn_scores + self.causal_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ v                                            # (B, num_heads, S, head_dim)

        # 7. Reshape, output gate, o_proj
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(1, S, self.num_heads * self.head_dim)
        attn_out = attn_out * torch.sigmoid(gate)
        hidden_out = F.linear(attn_out, self.o_proj_w)                         # (1, S, hidden)
        return hidden_out


def precompute_cos_sin(cfg, seq_len: int):
    """Use HF rotary module to make cos/sin for text-only position_ids."""
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    rot.eval()
    # text-only: position_ids shape (1, S). The module expands to 3 grids,
    # all identical, so MRoPE collapses to plain RoPE.
    pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(1, seq_len, cfg.hidden_size)
    with torch.no_grad():
        cos, sin = rot(dummy, pos)
    # cos/sin shape: depending on version, typically (B, S, rotary_dim).
    # Debug:
    return cos, sin


def extract_hf_full_attn_io(hf_model, cfg, seq_len: int, layer_idx: int = 3):
    """Run HF prefill, capture layer-`layer_idx` Qwen3_5Attention in/out."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    text = "The capital of France is Paris. The capital of Japan is Tokyo. " \
           "Machine learning systems benefit from on-device inference."
    ids = tok(text, return_tensors="pt").input_ids
    if ids.shape[1] < seq_len:
        pad_id = tok.eos_token_id or 0
        ids = torch.cat([ids, torch.full((1, seq_len - ids.shape[1]), pad_id, dtype=ids.dtype)], dim=1)
    else:
        ids = ids[:, :seq_len]

    attn_layer = hf_model.model.layers[layer_idx].self_attn  # full_attention at idx 3,7,11,...
    captured = {}

    def pre_hook(module, args, kwargs):
        h = args[0] if len(args) > 0 else kwargs["hidden_states"]
        captured["input"] = h.detach().clone()
        # position_embeddings is also passed
        pe = kwargs.get("position_embeddings")
        if pe is None and len(args) > 1:
            pe = args[1]
        captured["cos"] = pe[0].detach().clone()
        captured["sin"] = pe[1].detach().clone()

    def post_hook(module, args, kwargs, output):
        out = output if not isinstance(output, tuple) else output[0]
        captured["output"] = out.detach().clone()

    h1 = attn_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = attn_layer.register_forward_hook(post_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            hf_model(input_ids=ids, use_cache=False)
    finally:
        h1.remove(); h2.remove()
    return captured


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def parity_check(hf_model, cfg, seq_lens):
    print("\n=== parity check ===")
    # Pick layer idx 3 (first full_attention after 3 linear_attention layers).
    layer_idx = 3
    hf_attn = hf_model.model.layers[layer_idx].self_attn
    for seq_len in seq_lens:
        cap = extract_hf_full_attn_io(hf_model, cfg, seq_len, layer_idx=layer_idx)
        mod = FullAttentionLayer(cfg, hf_attn, seq_len).eval().float()
        with torch.no_grad():
            hout = mod(cap["input"].float(), cap["cos"].float(), cap["sin"].float())
        c = cos_sim(hout.float(), cap["output"].float())
        print(f"  seq={seq_len:4d}  cos={c:.6f}  hin={tuple(cap['input'].shape)} "
              f"cos_shape={tuple(cap['cos'].shape)}")


def convert_and_audit(hf_model, cfg, seq_len=64):
    print(f"\n=== CoreML conversion (seq={seq_len}) ===")
    layer_idx = 3
    hf_attn = hf_model.model.layers[layer_idx].self_attn
    mod = FullAttentionLayer(cfg, hf_attn, seq_len).eval().float()

    cap = extract_hf_full_attn_io(hf_model, cfg, seq_len, layer_idx=layer_idx)
    example = (
        torch.randn(1, seq_len, cfg.hidden_size),
        cap["cos"].float(),
        cap["sin"].float(),
    )
    traced = torch.jit.trace(mod, example, strict=False)
    print("  trace OK")

    ct_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="hidden_in", shape=example[0].shape, dtype=np.float16),
            ct.TensorType(name="cos", shape=example[1].shape, dtype=np.float16),
            ct.TensorType(name="sin", shape=example[2].shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="hidden_out", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print("  convert OK")

    tmpdir = Path(tempfile.mkdtemp(prefix="qwen35_full_attn_"))
    path = tmpdir / f"full_attn_seq{seq_len}.mlpackage"
    ct_model.save(str(path))
    size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved {path} ({size_mb:.1f} MB)")

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
        if dev == "unknown":
            continue
        print(f"  === {dev} op types ===")
        for op_type, n in c.most_common(20):
            print(f"    {op_type}: {n}")


def main():
    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    parity_check(model, cfg, seq_lens=[64, 128, 256])

    for seq_len in [64, 256, 2048]:
        convert_and_audit(model, cfg, seq_len=seq_len)


if __name__ == "__main__":
    main()
