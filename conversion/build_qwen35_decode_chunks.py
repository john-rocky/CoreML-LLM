"""Decode split into two stateful chunks (layers 0-11 + layers 12-23 + head).
Both chunks declared fp16 compute, fp32 hidden handoff between them.

Hypothesis: if ANE fp16 per-layer drift compounds non-linearly across 24
layers, forcing a fp32 boundary mid-stack resets the accumulation and
brings top-1 back up. If drift is linear (uniform per-layer), chunking
alone won't help.

Each chunk has its own 24-state slice (12 layers × 2 states each).
Swift threads hidden + states across chunks per token."""
from pathlib import Path
import argparse
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from test_qwen3_5_full_decode_trace import (
    DecodeRMSNorm, DecoderDecodeLayer, MAX_SEQ,
    make_zero_states, cos_sim,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
ORACLE = Path(__file__).parent / "qwen3_5_reference_logits.pt"


class DecodeChunkA(nn.Module):
    """embed + layers [0, split). Emits hidden after layer split-1 + updated
    states for those layers."""
    def __init__(self, cfg, hf_model, split, max_seq):
        super().__init__()
        self.split = split
        self.embed_w = nn.Parameter(hf_model.model.embed_tokens.weight.detach().clone(),
                                     requires_grad=False)
        self.layers = nn.ModuleList([
            DecoderDecodeLayer(cfg, hf_model.model.layers[i], max_seq)
            for i in range(split)
        ])

    def forward(self, input_token, position, cos, sin, *states):
        hidden = F.embedding(input_token.to(torch.long), self.embed_w)
        new_states = []
        for i, layer in enumerate(self.layers):
            sa, sb = states[2 * i], states[2 * i + 1]
            hidden, ns_a, ns_b = layer(hidden, position, cos, sin, sa, sb)
            new_states.append(ns_a); new_states.append(ns_b)
        return (hidden, *new_states)


class DecodeChunkB(nn.Module):
    """layers [split, num_layers) + final_norm + lm_head. Reads hidden from
    chunk A (possibly fp32) and the remaining state slice."""
    def __init__(self, cfg, hf_model, split, max_seq):
        super().__init__()
        self.split = split
        self.num_layers = cfg.num_hidden_layers
        self.layers = nn.ModuleList([
            DecoderDecodeLayer(cfg, hf_model.model.layers[i], max_seq)
            for i in range(split, self.num_layers)
        ])
        self.final_norm = DecodeRMSNorm(cfg.rms_norm_eps, hf_model.model.norm.weight)
        self.lm_head_w = nn.Parameter(hf_model.lm_head.weight.detach().clone(),
                                       requires_grad=False)

    def forward(self, hidden_in, position, cos, sin, *states):
        hidden = hidden_in
        new_states = []
        for i, layer in enumerate(self.layers):
            sa, sb = states[2 * i], states[2 * i + 1]
            hidden, ns_a, ns_b = layer(hidden, position, cos, sin, sa, sb)
            new_states.append(ns_a); new_states.append(ns_b)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.lm_head_w)
        return (logits, *new_states)


def _layer_state_shapes(cfg, layer_idx, max_seq):
    lt = "linear_attention" if layer_idx % 4 != 3 else "full_attention"
    if lt == "linear_attention":
        conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 + \
                    cfg.linear_value_head_dim * cfg.linear_num_value_heads
        a = (1, conv_dim, cfg.linear_conv_kernel_dim)
        b = (1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim)
    else:
        a = (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
        b = (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
    return a, b


def convert_chunk(chunk, cfg, start_layer, end_layer, max_seq, out_path,
                   hidden_dtype_input=None):
    print(f"\n=== convert decode chunk layers [{start_layer}, {end_layer}) ===")
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    pos_ids = torch.zeros(1, 1, dtype=torch.long)
    dummy = torch.zeros(1, 1, cfg.hidden_size)
    with torch.no_grad():
        cos_t, sin_t = rot(dummy, pos_ids)

    # Build example inputs
    inputs = []
    if hidden_dtype_input is None:
        # Chunk A: input_token
        inputs.append(torch.zeros(1, 1, dtype=torch.int32))
    else:
        # Chunk B: hidden_in
        inputs.append(torch.zeros(1, 1, cfg.hidden_size, dtype=torch.float32))
    inputs.append(torch.zeros(1, dtype=torch.float32))  # position
    inputs.append(cos_t.float())
    inputs.append(sin_t.float())
    for i in range(start_layer, end_layer):
        sa_shape, sb_shape = _layer_state_shapes(cfg, i, max_seq)
        inputs.append(torch.zeros(*sa_shape))
        inputs.append(torch.zeros(*sb_shape))
    example = tuple(inputs)

    traced = torch.jit.trace(chunk, example, strict=False)
    print("  trace OK")

    # Build coremltools spec
    ct_inputs = []
    if hidden_dtype_input is None:
        ct_inputs.append(ct.TensorType(name="input_token", shape=(1, 1), dtype=np.int32))
    else:
        ct_inputs.append(ct.TensorType(name="hidden_in",
                                        shape=(1, 1, cfg.hidden_size),
                                        dtype=hidden_dtype_input))
    ct_inputs.append(ct.TensorType(name="position", shape=(1,), dtype=np.float32))
    ct_inputs.append(ct.TensorType(name="cos", shape=cos_t.shape, dtype=np.float16))
    ct_inputs.append(ct.TensorType(name="sin", shape=sin_t.shape, dtype=np.float16))

    ct_outputs = []
    if hidden_dtype_input is None:
        # Chunk A emits hidden
        ct_outputs.append(ct.TensorType(name="hidden", dtype=np.float32))
    else:
        ct_outputs.append(ct.TensorType(name="logits", dtype=np.float32))

    for local_i, i in enumerate(range(start_layer, end_layer)):
        sa_shape, sb_shape = _layer_state_shapes(cfg, i, max_seq)
        ct_inputs.append(ct.TensorType(name=f"state_{i}_a", shape=sa_shape, dtype=np.float16))
        ct_inputs.append(ct.TensorType(name=f"state_{i}_b", shape=sb_shape, dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_state_{i}_a", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_state_{i}_b", dtype=np.float16))

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
    dev = Counter()
    for fn_name, fn in plan.model_structure.program.functions.items():
        for op in fn.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            d = "const" if (a is None and op.operator_name == "const") \
                else (a.preferred_compute_device.__class__.__name__ if a else "unknown")
            dev[d] += 1
    total = sum(dev.values()); const = dev.get("const", 0); compute = total - const
    ane = dev.get("MLNeuralEngineComputeDevice", 0)
    print(f"  total={total} compute={compute}  ANE={ane} ({100*ane/compute:.1f}% of compute)")


def predict_parity(body_path, head_path, oracle, cfg, split, max_seq,
                    body_units=ct.ComputeUnit.CPU_AND_NE,
                    head_units=ct.ComputeUnit.CPU_AND_NE):
    print(f"\n=== Mac predict parity (body={body_units}, head={head_units}) ===")
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    mlm_a = ct.models.MLModel(str(body_path), compute_units=body_units)
    mlm_b = ct.models.MLModel(str(head_path), compute_units=head_units)

    for pi, rec in enumerate(oracle["records"][:5]):
        ids = rec["input_ids"]; S = ids.shape[1]
        states = make_zero_states(cfg, max_seq)
        a_dict = {f"state_{i}_a": states[2*i].numpy().astype(np.float16)
                  for i in range(split)}
        a_dict.update({f"state_{i}_b": states[2*i+1].numpy().astype(np.float16)
                       for i in range(split)})
        b_dict = {f"state_{i}_a": states[2*i].numpy().astype(np.float16)
                  for i in range(split, cfg.num_hidden_layers)}
        b_dict.update({f"state_{i}_b": states[2*i+1].numpy().astype(np.float16)
                       for i in range(split, cfg.num_hidden_layers)})

        last_logits = None
        for t in range(S):
            pos_ids = torch.tensor([[t]], dtype=torch.long)
            dummy = torch.zeros(1, 1, cfg.hidden_size)
            with torch.no_grad():
                c_t, s_t = rot(dummy, pos_ids)
            a_in = {
                "input_token": ids[:, t:t+1].numpy().astype(np.int32),
                "position": np.array([float(t)], dtype=np.float32),
                "cos": c_t.numpy().astype(np.float16),
                "sin": s_t.numpy().astype(np.float16),
                **a_dict,
            }
            a_out = mlm_a.predict(a_in)
            for i in range(split):
                a_dict[f"state_{i}_a"] = a_out[f"new_state_{i}_a"]
                a_dict[f"state_{i}_b"] = a_out[f"new_state_{i}_b"]
            hidden = a_out["hidden"].astype(np.float32)
            b_in = {
                "hidden_in": hidden,
                "position": np.array([float(t)], dtype=np.float32),
                "cos": c_t.numpy().astype(np.float16),
                "sin": s_t.numpy().astype(np.float16),
                **b_dict,
            }
            b_out = mlm_b.predict(b_in)
            for i in range(split, cfg.num_hidden_layers):
                b_dict[f"state_{i}_a"] = b_out[f"new_state_{i}_a"]
                b_dict[f"state_{i}_b"] = b_out[f"new_state_{i}_b"]
            if t == S - 1:
                last_logits = b_out["logits"][0, 0]
        ref = rec["logits_recurrent"][-1].float().numpy()
        c = cos_sim(torch.from_numpy(last_logits), torch.from_numpy(ref))
        top1 = int(np.argmax(last_logits))
        match = top1 == int(rec["top10_last_ids"][0].item())
        print(f"  prompt[{pi}] S={S}  cos={c:.6f}  top1={match}  {rec['prompt'][:30]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", type=int, default=12)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    hf = Qwen3_5ForCausalLM.from_pretrained(MODEL_ID, config=cfg, torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    a = DecodeChunkA(cfg, hf, args.split, args.max_seq).eval().float()
    b = DecodeChunkB(cfg, hf, args.split, args.max_seq).eval().float()
    del hf

    if args.skip_convert:
        # torch fp32 sanity
        print(f"\n=== torch fp32 sanity (chunks A + B) ===")
        oracle = torch.load(str(ORACLE), map_location="cpu", weights_only=False)
        rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
        for rec in oracle["records"][:3]:
            ids = rec["input_ids"]; S = ids.shape[1]
            states = make_zero_states(cfg, args.max_seq)
            with torch.no_grad():
                for t in range(S):
                    tok = ids[:, t:t+1].to(torch.int32)
                    pos = torch.tensor([float(t)], dtype=torch.float32)
                    pos_ids = torch.tensor([[t]], dtype=torch.long)
                    dummy = torch.zeros(1, 1, cfg.hidden_size)
                    c_t, s_t = rot(dummy, pos_ids)
                    a_states = [s.float() for s in states[:2*args.split]]
                    a_out = a(tok, pos, c_t.float(), s_t.float(), *a_states)
                    hidden, *new_a_states = a_out
                    states[:2*args.split] = list(new_a_states)
                    b_states = [s.float() for s in states[2*args.split:]]
                    b_out = b(hidden, pos, c_t.float(), s_t.float(), *b_states)
                    logits, *new_b_states = b_out
                    states[2*args.split:] = list(new_b_states)
                    if t == S - 1:
                        last_logits = logits[0, 0].float()
            ref = rec["logits_recurrent"][-1].float()
            c = cos_sim(last_logits, ref)
            top1 = int(torch.argmax(last_logits).item())
            match = top1 == int(rec["top10_last_ids"][0].item())
            print(f"  S={S}  cos={c:.6f}  top1={match}")
        return

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    body_path = out_dir / f"qwen3_5_decode_chunk_a_split{args.split}.mlpackage"
    head_path = out_dir / f"qwen3_5_decode_chunk_b_split{args.split}.mlpackage"

    convert_chunk(a, cfg, 0, args.split, args.max_seq, body_path,
                   hidden_dtype_input=None)  # int32 input
    convert_chunk(b, cfg, args.split, cfg.num_hidden_layers, args.max_seq, head_path,
                   hidden_dtype_input=np.float32)  # fp32 hidden input

    oracle = torch.load(str(ORACLE), map_location="cpu", weights_only=False)
    predict_parity(body_path, head_path, oracle, cfg, args.split, args.max_seq)


if __name__ == "__main__":
    main()
