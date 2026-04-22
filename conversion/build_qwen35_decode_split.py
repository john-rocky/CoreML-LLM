"""Phase 4g: split decode into body (fp16 on ANE) + head (fp32 on CPU/GPU).

Why split: the entire fp16 decode graph on ANE gives top-1 40% because the
final 1024 -> 248320 lm_head matmul accumulates in fp16 and tips the ranks.
Just marking lm_head as fp32 inside a single mlprogram makes coremltools'
compute_precision pass downgrade every connected op out of ANE, defeating
the purpose.

Split-mlpackage approach (same pattern Gemma 4 uses for per-chunk compute
unit choice): two independent mlpackages, Swift drives them in sequence:

  decode_body : input_token + position + cos + sin + 48 states
                -> hidden (fp16) + 48 new_states
                Runs with compute_units=.cpuAndNeuralEngine, body stays
                on ANE at full speed.
  decode_head : hidden (fp32, cast on Swift side) -> logits (fp32)
                Just final_norm + lm_head. Runs with compute_units
                of user's choice (GPU gives full precision).

Parity expectation: body ANE contributes same per-layer fp16 drift as
today's single-graph decode, but the lm_head precision is fully recovered
so top-1 match rate should climb from 40% back to ~100%.
"""
from collections import Counter
from pathlib import Path
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from test_qwen3_5_full_decode_trace import (
    DecodeRMSNorm, DecoderDecodeLayer, FullDecodeModel, MAX_SEQ,
    make_zero_states, make_example_inputs, cos_sim, torch_parity,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
ORACLE = Path(__file__).parent / "qwen3_5_reference_logits.pt"


class DecodeBody(nn.Module):
    """24-layer stateful decode without the final_norm + lm_head. Emits the
    post-residual hidden for the head module."""
    def __init__(self, cfg, hf_model, max_seq):
        super().__init__()
        self.max_seq = max_seq
        self.num_layers = cfg.num_hidden_layers
        self.embed_w = nn.Parameter(hf_model.model.embed_tokens.weight.detach().clone(),
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
        return (hidden, *new_states)


class DecodeHead(nn.Module):
    """final_norm + lm_head. Kept as fp32 so the 1024->248320 reduction
    uses full precision even when the body is fp16 on ANE."""
    def __init__(self, cfg, hf_model):
        super().__init__()
        self.final_norm = DecodeRMSNorm(cfg.rms_norm_eps, hf_model.model.norm.weight)
        self.lm_head_w = nn.Parameter(hf_model.lm_head.weight.detach().clone(),
                                       requires_grad=False)

    def forward(self, hidden_in):
        h = self.final_norm(hidden_in)
        return F.linear(h, self.lm_head_w)


def convert_body(body, cfg, rot, max_seq, out_path):
    print(f"\n=== convert body (ANE, fp16) ===")
    example = make_example_inputs(cfg, max_seq, rot)
    traced = torch.jit.trace(body, example, strict=False)
    print("  trace OK")
    ct_inputs = [
        ct.TensorType(name="input_token", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="position", shape=(1,), dtype=np.float32),
        ct.TensorType(name="cos", shape=example[2].shape, dtype=np.float16),
        ct.TensorType(name="sin", shape=example[3].shape, dtype=np.float16),
    ]
    ct_outputs = [
        ct.TensorType(name="hidden", dtype=np.float16),
    ]
    for i in range(cfg.num_hidden_layers):
        sa = example[4 + 2 * i]; sb = example[4 + 2 * i + 1]
        ct_inputs.append(ct.TensorType(name=f"state_{i}_a", shape=sa.shape, dtype=np.float16))
        ct_inputs.append(ct.TensorType(name=f"state_{i}_b", shape=sb.shape, dtype=np.float16))
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
    return _audit(out_path, ct.ComputeUnit.CPU_AND_NE)


def convert_head(head, cfg, out_path):
    print(f"\n=== convert head (fp32) ===")
    example = (torch.zeros(1, 1, cfg.hidden_size, dtype=torch.float32),)
    traced = torch.jit.trace(head, example, strict=False)
    print("  trace OK")
    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, 1, cfg.hidden_size), dtype=np.float32),
    ]
    ct_outputs = [
        ct.TensorType(name="logits", dtype=np.float32),
    ]
    # compute_precision=FLOAT32 so the lm_head reduction keeps full precision.
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )
    print("  convert OK")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved {out_path} ({size_mb:.1f} MB)")
    return _audit(out_path, ct.ComputeUnit.ALL)


def _audit(path, units):
    reloaded = ct.models.MLModel(str(path), compute_units=units)
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=units,
    )
    prog = plan.model_structure.program
    dev = Counter()
    for fn_name, fn in prog.functions.items():
        for op in fn.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if a is None:
                d = "const" if op.operator_name == "const" else "unknown"
            else:
                d = a.preferred_compute_device.__class__.__name__
            dev[d] += 1
    total = sum(dev.values()); const = dev.get("const", 0); compute = total - const
    ane = dev.get("MLNeuralEngineComputeDevice", 0)
    gpu = dev.get("MLGPUComputeDevice", 0)
    cpu = dev.get("MLCPUComputeDevice", 0)
    print(f"  total={total} compute={compute}  ANE={ane} GPU={gpu} CPU={cpu}")
    return dev


def predict_parity(body_path, head_path, oracle):
    """Mac CPU+ANE on body, CPU+GPU on head — token by token, compare to
    HF recurrent oracle."""
    print(f"\n=== Mac predict parity (body=ANE, head=GPU) ===")
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()

    mlm_body = ct.models.MLModel(str(body_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    mlm_head = ct.models.MLModel(str(head_path), compute_units=ct.ComputeUnit.ALL)

    n_layers = cfg.num_hidden_layers
    for pi, rec in enumerate(oracle["records"][:5]):
        ids = rec["input_ids"]
        S = ids.shape[1]
        states_t = make_zero_states(cfg, MAX_SEQ)
        state_dict = {f"state_{i}_a": states_t[2*i].numpy().astype(np.float16)
                      for i in range(n_layers)}
        state_dict.update({f"state_{i}_b": states_t[2*i+1].numpy().astype(np.float16)
                           for i in range(n_layers)})

        last_logits = None
        for t in range(S):
            pos_ids = torch.tensor([[t]], dtype=torch.long)
            dummy = torch.zeros(1, 1, cfg.hidden_size)
            with torch.no_grad():
                c_t, s_t = rot(dummy, pos_ids)
            body_in = {
                "input_token": ids[:, t:t+1].numpy().astype(np.int32),
                "position": np.array([float(t)], dtype=np.float32),
                "cos": c_t.numpy().astype(np.float16),
                "sin": s_t.numpy().astype(np.float16),
                **state_dict,
            }
            body_out = mlm_body.predict(body_in)
            for i in range(n_layers):
                state_dict[f"state_{i}_a"] = body_out[f"new_state_{i}_a"]
                state_dict[f"state_{i}_b"] = body_out[f"new_state_{i}_b"]
            hidden = body_out["hidden"].astype(np.float32)
            head_out = mlm_head.predict({"hidden_in": hidden})
            if t == S - 1:
                last_logits = head_out["logits"][0, 0]
        ref = rec["logits_recurrent"][-1].float().numpy()
        c = cos_sim(torch.from_numpy(last_logits), torch.from_numpy(ref))
        top1 = int(np.argmax(last_logits))
        match = top1 == int(rec["top10_last_ids"][0].item())
        print(f"  prompt[{pi}] S={S}  cos={c:.6f}  top1={match}  {rec['prompt'][:30]!r}")


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

    body = DecodeBody(cfg, hf, args.max_seq).eval().float()
    head = DecodeHead(cfg, hf).eval().float()
    del hf

    # torch fp32 sanity: body + head should match the oracle
    oracle = torch.load(str(ORACLE), map_location="cpu", weights_only=False)
    print(f"\n=== torch fp32 sanity (body + head chain) ===")
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
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
                out = body(tok, pos, c_t.float(), s_t.float(), *[s.float() for s in states])
                hidden, *new_states = out
                states = list(new_states)
                if t == S - 1:
                    logits = head(hidden)
                    last_logits = logits[0, 0].float()
        ref = rec["logits_recurrent"][-1].float()
        c = cos_sim(last_logits, ref)
        top1 = int(torch.argmax(last_logits).item())
        match = top1 == int(rec["top10_last_ids"][0].item())
        print(f"  S={S:3d}  cos={c:.6f}  top1={match}  {rec['prompt'][:30]!r}")

    if args.skip_convert:
        return

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    body_path = out_dir / "qwen3_5_decode_body.mlpackage"
    head_path = out_dir / "qwen3_5_decode_head.mlpackage"

    convert_body(body, cfg, rot, args.max_seq, body_path)
    convert_head(head, cfg, head_path)
    predict_parity(body_path, head_path, oracle)


if __name__ == "__main__":
    main()
