"""Decode model with in-graph argmax — avoids 248K-vocab CPU transfer per step.

Rationale: on iPhone the per-decode-step cost is dominated by transferring
logits (248K fp16 ≈ 500 KB) from ANE/GPU to CPU, then argmax-ing in Swift.
Moving argmax into the mlpackage reduces output to a single int32 token ID.
Expected end-to-end decode speedup: 13 → 20+ tok/s.

Output: `qwen3_5_0_8b_decode_argmax_fp16_mseq128.mlpackage`

For sampling-mode generation, keep using the original decode model.
"""
import argparse, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from test_qwen3_5_full_decode_trace import (
    DecoderDecodeLayer, DecodeRMSNorm, MAX_SEQ, make_zero_states, cos_sim,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
ORACLE = Path(__file__).parent / "qwen3_5_reference_logits.pt"


class FullDecodeArgmaxModel(nn.Module):
    """Same 24-layer stateful decode, but final output is argmax token ID
    instead of full logit vector. 248K-vocab logit tensor never leaves ANE."""
    def __init__(self, cfg, hf_model, max_seq):
        super().__init__()
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
        hidden = F.embedding(input_token.to(torch.long), self.embed_w)
        new_states = []
        for i, layer in enumerate(self.layers):
            sa, sb = states[2 * i], states[2 * i + 1]
            hidden, ns_a, ns_b = layer(hidden, position, cos, sin, sa, sb)
            new_states.append(ns_a); new_states.append(ns_b)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.lm_head_w)        # (1, 1, V) fp16
        next_token = torch.argmax(logits, dim=-1)        # (1, 1)      int64
        next_token = next_token.to(torch.int32)          # CoreML friendly
        return (next_token, *new_states)


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
    print(f"\n=== convert decode-with-argmax ===")
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
    ct_out = [ct.TensorType(name="next_token", dtype=np.int32)]
    for i in range(cfg.num_hidden_layers):
        sa, sb = _layer_state_shapes(cfg, i, max_seq)
        ct_in.append(ct.TensorType(name=f"state_{i}_a", shape=sa, dtype=np.float16))
        ct_in.append(ct.TensorType(name=f"state_{i}_b", shape=sb, dtype=np.float16))
        ct_out.append(ct.TensorType(name=f"new_state_{i}_a", dtype=np.float16))
        ct_out.append(ct.TensorType(name=f"new_state_{i}_b", dtype=np.float16))

    m = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_in, outputs=ct_out,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    m.save(str(out_path))
    print(f"  saved {out_path}")

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
    print(f"  ANE {ane}/{compute} = {100*ane/compute:.1f}%")


def predict_parity(path, oracle, cfg, max_seq):
    """Run on ANE across oracle prompts, confirm argmax matches HF fp32 top-1."""
    print("\n=== parity check: argmax output vs oracle top-1 ===")
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    mlm = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    hits = 0
    for pi, rec in enumerate(oracle["records"][:5]):
        ids = rec["input_ids"]; S = ids.shape[1]
        states = make_zero_states(cfg, max_seq)
        state_d = {f"state_{i}_a": states[2*i].numpy().astype(np.float16)
                   for i in range(cfg.num_hidden_layers)}
        state_d.update({f"state_{i}_b": states[2*i+1].numpy().astype(np.float16)
                        for i in range(cfg.num_hidden_layers)})
        last_tok = -1
        for t in range(S):
            pos_ids = torch.tensor([[t]], dtype=torch.long)
            dummy = torch.zeros(1, 1, cfg.hidden_size)
            with torch.no_grad():
                c_t, s_t = rot(dummy, pos_ids)
            inp = {
                "input_token": ids[:, t:t+1].numpy().astype(np.int32),
                "position": np.array([float(t)], dtype=np.float32),
                "cos": c_t.numpy().astype(np.float16),
                "sin": s_t.numpy().astype(np.float16),
                **state_d,
            }
            out = mlm.predict(inp)
            for i in range(cfg.num_hidden_layers):
                state_d[f"state_{i}_a"] = out[f"new_state_{i}_a"]
                state_d[f"state_{i}_b"] = out[f"new_state_{i}_b"]
            if t == S - 1:
                last_tok = int(out["next_token"].flatten()[0])
        oracle_top1 = int(rec["top10_last_ids"][0].item())
        match = last_tok == oracle_top1
        if match: hits += 1
        print(f"  prompt[{pi}] S={S}  model_argmax={last_tok}  oracle={oracle_top1}  match={match}")
    print(f"  argmax parity: {hits}/5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    args = ap.parse_args()

    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    hf = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    model = FullDecodeArgmaxModel(cfg, hf, args.max_seq).eval().float()
    del hf

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qwen3_5_0_8b_decode_argmax_fp16_mseq128.mlpackage"
    convert(model, cfg, args.max_seq, path)

    oracle = torch.load(str(ORACLE), map_location="cpu", weights_only=False)
    predict_parity(path, oracle, cfg, args.max_seq)


if __name__ == "__main__":
    main()
