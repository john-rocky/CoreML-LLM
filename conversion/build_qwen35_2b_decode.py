"""Convert Qwen3.5-2B (VL) text backbone to CoreML decode mlpackage.

2B is a vision-language model (Qwen3_5ForConditionalGeneration). The text
backbone has the same architecture as 0.8B — 24 layers, same layer_types
[L,L,L,F]x6, same head dims — just wider (hidden 1024→2048, intermediate
3072→6144). Disregards the vision tower; decode graph only.

Downstream:
  python build_qwen35_2b_decode.py --out-dir /tmp/qwen35_2b
  # then INT8 palettize via build_qwen35_decode_int4.py --nbits 8
"""
import argparse, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

from transformers import (
    AutoModelForCausalLM, Qwen3_5TextConfig, Qwen3_5ForCausalLM,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from test_qwen3_5_full_decode_trace import (
    DecoderDecodeLayer, DecodeRMSNorm, MAX_SEQ, make_zero_states, cos_sim,
)

MODEL_ID = "Qwen/Qwen3.5-2B"
ORACLE = Path(__file__).parent / "qwen3_5_reference_logits.pt"  # 0.8B oracle


def load_text_backbone(model_id: str):
    """Load a Qwen3.5 checkpoint. Probed behavior: transformers 4.57+
    returns `Qwen3_5ForCausalLM` (text backbone only) when using
    `AutoModelForCausalLM.from_pretrained` even if the repo's config
    declares `Qwen3_5ForConditionalGeneration` — the VL tower is simply
    absent from the returned module. Same .model.layers /
    .model.embed_tokens / .lm_head layout as 0.8B."""
    full = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    assert hasattr(full, "model") and hasattr(full.model, "layers"), \
        f"expected .model.layers on {type(full).__name__}"
    return full


# The FullDecodeModel is identical for 2B — same architecture, just wider.
# Reuse by composition.
class FullDecodeModel2B(nn.Module):
    def __init__(self, cfg, hf_model, max_seq):
        super().__init__()
        self.num_layers = cfg.num_hidden_layers
        self.embed_w = nn.Parameter(hf_model.model.embed_tokens.weight.detach().clone(),
                                     requires_grad=False)
        self.final_norm = DecodeRMSNorm(cfg.rms_norm_eps, hf_model.model.norm.weight)
        # tie_word_embeddings=True → lm_head shares embed_tokens weight
        lm_w = (hf_model.lm_head.weight if not cfg.tie_word_embeddings
                else hf_model.model.embed_tokens.weight)
        self.lm_head_w = nn.Parameter(lm_w.detach().clone(), requires_grad=False)
        self.layers = nn.ModuleList([
            DecoderDecodeLayer(cfg, hf_model.model.layers[i], max_seq)
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
        logits = F.linear(hidden, self.lm_head_w)
        return (logits, *new_states)


def _layer_state_shapes(cfg, i, max_seq):
    lt = "linear_attention" if i % 4 != 3 else "full_attention"
    if lt == "linear_attention":
        conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 + \
                    cfg.linear_value_head_dim * cfg.linear_num_value_heads
        a = (1, conv_dim, cfg.linear_conv_kernel_dim)
        b = (1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim)
    else:
        a = (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
        b = (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
    return a, b


def convert(model, cfg, max_seq, out_path):
    print(f"\n=== convert Qwen3.5-2B decode ===")
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    pos_ids = torch.zeros(1, 1, dtype=torch.long)
    dummy = torch.zeros(1, 1, cfg.hidden_size)
    with torch.no_grad():
        cos_t, sin_t = rot(dummy, pos_ids)

    inputs = [torch.zeros(1, 1, dtype=torch.int32),
              torch.zeros(1, dtype=torch.float32),
              cos_t.float(), sin_t.float()]
    for i in range(cfg.num_hidden_layers):
        sa, sb = _layer_state_shapes(cfg, i, max_seq)
        inputs.append(torch.zeros(*sa)); inputs.append(torch.zeros(*sb))
    traced = torch.jit.trace(model, tuple(inputs), strict=False)
    print("  trace OK")

    ct_in = [
        ct.TensorType(name="input_token", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="position", shape=(1,), dtype=np.float32),
        ct.TensorType(name="cos", shape=cos_t.shape, dtype=np.float16),
        ct.TensorType(name="sin", shape=sin_t.shape, dtype=np.float16),
    ]
    ct_out = [ct.TensorType(name="logits", dtype=np.float32)]
    for i in range(cfg.num_hidden_layers):
        sa, sb = _layer_state_shapes(cfg, i, max_seq)
        ct_in.append(ct.TensorType(name=f"state_{i}_a", shape=sa, dtype=np.float16))
        ct_in.append(ct.TensorType(name=f"state_{i}_b", shape=sb, dtype=np.float16))
        ct_out.append(ct.TensorType(name=f"new_state_{i}_a", dtype=np.float16))
        ct_out.append(ct.TensorType(name=f"new_state_{i}_b", dtype=np.float16))

    m = ct.convert(
        traced, convert_to="mlprogram", inputs=ct_in, outputs=ct_out,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    m.save(str(out_path))
    print(f"  saved: {out_path}")

    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  bundle size: {size_mb:.0f} MB (fp16)")

    reloaded = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    dev = Counter()
    for fn in plan.model_structure.program.functions.values():
        for op in fn.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            d = "const" if (a is None and op.operator_name == "const") \
                else (a.preferred_compute_device.__class__.__name__ if a else "unknown")
            dev[d] += 1
    total = sum(dev.values()); const = dev.get("const", 0); compute = total - const
    ane = dev.get("MLNeuralEngineComputeDevice", 0)
    print(f"  ANE placement: {ane}/{compute} ({100*ane/compute:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    args = ap.parse_args()

    print(f"loading {MODEL_ID}...")
    t0 = time.time()
    # Text-config extraction from the VL config
    from transformers import AutoConfig
    full_cfg = AutoConfig.from_pretrained(MODEL_ID)
    if hasattr(full_cfg, 'text_config'):
        text_cfg_dict = full_cfg.text_config.to_dict()
    else:
        text_cfg_dict = full_cfg.to_dict()
    cfg = Qwen3_5TextConfig.from_dict(text_cfg_dict)
    print(f"  text config: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
          f"vocab={cfg.vocab_size}")

    text_lm = load_text_backbone(MODEL_ID)
    print(f"  loaded in {time.time()-t0:.1f}s, class={type(text_lm).__name__}")

    model = FullDecodeModel2B(cfg, text_lm, args.max_seq).eval().float()
    del text_lm

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qwen3_5_2b_decode_fp16_mseq128.mlpackage"
    convert(model, cfg, args.max_seq, out_path)


if __name__ == "__main__":
    main()
