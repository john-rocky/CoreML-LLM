"""Phase 3b: stack 4 Qwen3.5 decoder layers (3 linear_attention + 1 full_attention)
end-to-end, verify parity vs HF, audit ANE placement.

This validates the decoder-layer wrapping pattern: input_layernorm, residuals,
MLP (SwiGLU), and the per-layer type dispatch. If a 4-layer mini-stack converts
with high ANE placement, scaling to 24 layers is applied engineering — no
further research risk.
"""
from collections import Counter
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, AutoTokenizer

import coremltools as ct

# Reuse the per-layer forward modules from Phase 2b / 3a.
from test_qwen3_5_prefill_trace import PrefillLinearAttnLayer
from test_qwen3_5_full_attention_trace import FullAttentionLayer


MODEL_ID = "Qwen/Qwen3.5-0.8B"
NUM_LAYERS = 4        # 3 linear + 1 full_attention (pattern matches layers 0..3 of 0.8B)
SEQ_LEN = 64


class RMSNorm(nn.Module):
    """Qwen3_5RMSNorm: (x * rsqrt(mean(x²) + eps)) * (1 + w).

    Uses the ANE-friendly `[x, -x] concat → LayerNorm` identity (cat has zero
    mean so LayerNorm ≡ RMSNorm). ANE has a fast LayerNorm kernel and no
    rsqrt kernel; this is the Apple-blessed pattern also used by Gemma 4
    (see conversion/ane_ops.py ANERMSNorm)."""
    def __init__(self, hidden_size, eps, weight):
        super().__init__()
        self.eps = float(eps)
        self.hidden = hidden_size
        # Qwen3_5's learned scale is applied as (1 + w). Pre-fold into a
        # single scale parameter so the forward has one multiply.
        self.w = nn.Parameter((1.0 + weight.detach().clone()), requires_grad=False)

    def forward(self, x):
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(
            doubled, normalized_shape=(2 * self.hidden,),
            weight=None, bias=None, eps=self.eps,
        )
        normed, _ = torch.chunk(normed, 2, dim=-1)
        return normed * self.w


class MLP(nn.Module):
    """Qwen3_5MLP = down(silu(gate(x)) * up(x))."""
    def __init__(self, gate_w, up_w, down_w):
        super().__init__()
        self.gate_w = nn.Parameter(gate_w.detach().clone(), requires_grad=False)
        self.up_w = nn.Parameter(up_w.detach().clone(), requires_grad=False)
        self.down_w = nn.Parameter(down_w.detach().clone(), requires_grad=False)

    def forward(self, x):
        # Conv2d 1x1 path: Apple's ANE-transformers recipe. Gives better
        # fp16 accumulation precision on Neural Engine than the nn.Linear
        # matmul kernel.
        from test_qwen3_5_prefill_trace import _linear_as_conv2d
        g = F.silu(_linear_as_conv2d(x, self.gate_w))
        u = _linear_as_conv2d(x, self.up_w)
        return _linear_as_conv2d(g * u, self.down_w)


class DecoderLayer(nn.Module):
    """Mirror of Qwen3_5DecoderLayer for prefill. Uses the Phase 2b/3a modules
    as the token mixer depending on layer_type."""
    def __init__(self, cfg, hf_layer, seq_len: int):
        super().__init__()
        self.layer_type = hf_layer.layer_type
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                  hf_layer.input_layernorm.weight)
        self.post_attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                       hf_layer.post_attention_layernorm.weight)
        self.mlp = MLP(hf_layer.mlp.gate_proj.weight,
                       hf_layer.mlp.up_proj.weight,
                       hf_layer.mlp.down_proj.weight)
        if self.layer_type == "linear_attention":
            self.mixer = PrefillLinearAttnLayer(cfg, hf_layer, seq_len)
        else:
            self.mixer = FullAttentionLayer(cfg, hf_layer.self_attn, seq_len)

    def forward(self, hidden, cos, sin):
        residual = hidden
        h = self.input_norm(hidden)
        if self.layer_type == "linear_attention":
            h = self.mixer(h)
        else:
            h = self.mixer(h, cos, sin)
        hidden = residual + h
        residual = hidden
        h = self.post_attn_norm(hidden)
        h = self.mlp(h)
        return residual + h


class MiniStack(nn.Module):
    """N consecutive decoder layers. Input: hidden_in, cos, sin. Output: hidden_out."""
    def __init__(self, cfg, hf_layers, seq_len: int):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(cfg, hf_layer, seq_len) for hf_layer in hf_layers
        ])

    def forward(self, hidden, cos, sin):
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
        return hidden


def extract_hf_stack_io(hf_model, cfg, seq_len: int, num_layers: int):
    """Capture the input to layer 0 (post-embed) and the output of layer
    num_layers-1 (post-residuals, before layer num_layers's input_layernorm)."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    text = "The capital of France is Paris. The capital of Japan is Tokyo. " \
           "Machine learning systems benefit from on-device inference."
    ids = tok(text, return_tensors="pt").input_ids
    if ids.shape[1] < seq_len:
        pad_id = tok.eos_token_id or 0
        ids = torch.cat([ids, torch.full((1, seq_len - ids.shape[1]), pad_id, dtype=ids.dtype)], dim=1)
    else:
        ids = ids[:, :seq_len]

    captured = {}

    # Hook layer 0: capture input
    def layer0_pre(module, args, kwargs):
        h = args[0] if len(args) > 0 else kwargs["hidden_states"]
        captured["layer0_in"] = h.detach().clone()
        pe = kwargs.get("position_embeddings")
        if pe is not None:
            captured["cos"] = pe[0].detach().clone()
            captured["sin"] = pe[1].detach().clone()

    # Hook last layer: capture output
    def last_post(module, args, kwargs, output):
        out = output if not isinstance(output, tuple) else output[0]
        captured["last_out"] = out.detach().clone()

    h1 = hf_model.model.layers[0].register_forward_pre_hook(layer0_pre, with_kwargs=True)
    h2 = hf_model.model.layers[num_layers - 1].register_forward_hook(last_post, with_kwargs=True)
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


def main():
    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Sanity: what are layer types of layers 0..3?
    types = [model.model.layers[i].layer_type for i in range(NUM_LAYERS)]
    print(f"  layer_types[0..{NUM_LAYERS-1}]: {types}")

    # Capture
    cap = extract_hf_stack_io(model, cfg, SEQ_LEN, NUM_LAYERS)
    print(f"  layer0_in: {tuple(cap['layer0_in'].shape)}  "
          f"last_out: {tuple(cap['last_out'].shape)}  "
          f"cos: {tuple(cap['cos'].shape)}")

    # Build stack
    hf_layers = [model.model.layers[i] for i in range(NUM_LAYERS)]
    stack = MiniStack(cfg, hf_layers, SEQ_LEN).eval().float()

    with torch.no_grad():
        hout = stack(cap["layer0_in"].float(), cap["cos"].float(), cap["sin"].float())
    c = cos_sim(hout.float(), cap["last_out"].float())
    print(f"\n=== parity ===")
    print(f"  {NUM_LAYERS}-layer stack cos = {c:.6f}")
    if c < 0.995:
        print("  FAILED parity — stopping before conversion.")
        return

    # CoreML conversion
    print(f"\n=== CoreML conversion ({NUM_LAYERS} layers, seq={SEQ_LEN}) ===")
    example = (
        torch.randn(1, SEQ_LEN, cfg.hidden_size),
        cap["cos"].float(),
        cap["sin"].float(),
    )
    traced = torch.jit.trace(stack, example, strict=False)
    print("  trace OK")

    ct_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="hidden_in", shape=example[0].shape, dtype=np.float16),
            ct.TensorType(name="cos", shape=example[1].shape, dtype=np.float16),
            ct.TensorType(name="sin", shape=example[2].shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="hidden_out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print("  convert OK")

    tmpdir = Path(tempfile.mkdtemp(prefix="qwen35_stack_"))
    path = tmpdir / f"stack_{NUM_LAYERS}layers_seq{SEQ_LEN}.mlpackage"
    ct_model.save(str(path))
    size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved {path} ({size_mb:.1f} MB)")

    print(f"\n=== placement audit ===")
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
        print(f"  === {dev} ops (top 25) ===")
        for op_type, n in c.most_common(25):
            print(f"    {op_type}: {n}")


if __name__ == "__main__":
    main()
