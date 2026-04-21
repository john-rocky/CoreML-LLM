"""Phase 4f: split Qwen3.5-0.8B 24-layer prefill into chunks.

iPhone Core ML 26.1 BNNS/ANEF compilers can't handle the full 24-layer
single-graph mlpackage when ANE-specific tricks (Conv2d, ANERMSNorm) are
applied; they hit "No space left on device" / "Couldn't communicate with
a helper application". Gemma 4 solves this by shipping 4 chunks of 7-10
layers each. Do the same for Qwen3.5 with 2 chunks of 12 layers.

Produces:
  chunk_a.mlpackage :  input_ids (1, seq_len) int32
                       -> hidden (1, seq_len, hidden_size) fp16
                       embed + decoder layers [0..N/2)
  chunk_b.mlpackage :  hidden_in (1, seq_len, hidden_size), cos, sin
                       -> logits (1, seq_len, vocab_size) fp32
                       decoder layers [N/2..N) + final_norm + lm_head

Swift side threads the hidden state from chunk_a.output to chunk_b.input.
cos/sin are precomputed inside chunk_a (buffer) and chunk_b (buffer).
"""
from collections import Counter
from pathlib import Path
import argparse
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from test_qwen3_5_stack_trace import DecoderLayer, RMSNorm


MODEL_ID = "Qwen/Qwen3.5-0.8B"
ORACLE_PATH = Path(__file__).parent / "qwen3_5_reference_logits.pt"


class ChunkA(nn.Module):
    """Embed tokens + run decoder layers [0, split). Emits hidden for chunk B."""
    def __init__(self, cfg, hf_model, seq_len: int, split: int):
        super().__init__()
        self.S = seq_len
        self.split = split
        self.embed_w = nn.Parameter(hf_model.model.embed_tokens.weight.detach().clone(),
                                     requires_grad=False)
        self.layers = nn.ModuleList([
            DecoderLayer(cfg, hf_model.model.layers[i], seq_len)
            for i in range(split)
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
        for layer in self.layers:
            hidden = layer(hidden, self.cos, self.sin)
        return hidden


class ChunkB(nn.Module):
    """Run decoder layers [split, num_layers) + final_norm + lm_head. Emits logits."""
    def __init__(self, cfg, hf_model, seq_len: int, split: int):
        super().__init__()
        self.S = seq_len
        self.split = split
        self.layers = nn.ModuleList([
            DecoderLayer(cfg, hf_model.model.layers[i], seq_len)
            for i in range(split, cfg.num_hidden_layers)
        ])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                   hf_model.model.norm.weight)
        self.lm_head_w = nn.Parameter(hf_model.lm_head.weight.detach().clone(),
                                       requires_grad=False)
        rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        dummy = torch.zeros(1, seq_len, cfg.hidden_size)
        with torch.no_grad():
            cos, sin = rot(dummy, pos)
        self.register_buffer("cos", cos.detach().clone(), persistent=False)
        self.register_buffer("sin", sin.detach().clone(), persistent=False)

    def forward(self, hidden_in):
        hidden = hidden_in
        for layer in self.layers:
            hidden = layer(hidden, self.cos, self.sin)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.lm_head_w)
        return logits


def cos_sim(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def parity(chunk_a, chunk_b, oracle, seq_len):
    print("\n=== torch fp32 parity (chunk_a -> chunk_b) ===")
    pad_id = 0
    for rec in oracle["records"]:
        ids = rec["input_ids"]; S = ids.shape[1]
        if S > seq_len: continue
        padded = torch.full((1, seq_len), pad_id, dtype=ids.dtype)
        padded[:, :S] = ids
        with torch.no_grad():
            hidden = chunk_a(padded)
            logits = chunk_b(hidden)
        per_pos = torch.tensor([
            cos_sim(logits[0, i], rec["logits_prefill"][i]) for i in range(S)
        ])
        top1 = int(torch.argmax(logits[0, S-1]).item())
        match = top1 == int(rec["top10_last_ids"][0].item())
        print(f"  S={S:3d}  mean={per_pos.mean():.6f}  worst={per_pos.min():.6f}  "
              f"top1={match}  {rec['prompt'][:30]!r}")


def convert_chunk(model, input_specs, output_specs, out_path, label):
    print(f"\n=== convert {label} ===")
    example = tuple(torch.zeros(s["shape"],
                                 dtype=torch.int32 if s["dtype"] == np.int32 else torch.float32)
                    for s in input_specs)
    traced = torch.jit.trace(model, example, strict=False)
    print("  trace OK")

    ct_inputs = [
        ct.TensorType(name=s["name"], shape=s["shape"], dtype=s["dtype"])
        for s in input_specs
    ]
    ct_outputs = [
        ct.TensorType(name=s["name"], dtype=s["dtype"]) for s in output_specs
    ]
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

    # Placement audit
    reloaded = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    dev_counts: Counter = Counter()
    for fn_name, fn in program.functions.items():
        for op in fn.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = "const" if (a is None and op.operator_name == "const") \
                  else (a.preferred_compute_device.__class__.__name__ if a else "unknown")
            dev_counts[dev] += 1
    total = sum(dev_counts.values())
    ane = dev_counts.get("MLNeuralEngineComputeDevice", 0)
    cpu = dev_counts.get("MLCPUComputeDevice", 0)
    const = dev_counts.get("const", 0)
    compute = total - const
    print(f"  total={total}  compute={compute}  const={const}")
    print(f"  ANE: {ane} ({100*ane/compute:.2f}% of compute)  CPU: {cpu}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--split", type=int, default=12, help="number of layers in chunk_a (rest go to chunk_b)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    hf = Qwen3_5ForCausalLM.from_pretrained(MODEL_ID, config=cfg, torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    chunk_a = ChunkA(cfg, hf, args.seq_len, args.split).eval().float()
    chunk_b = ChunkB(cfg, hf, args.seq_len, args.split).eval().float()
    del hf

    oracle = torch.load(str(ORACLE_PATH), map_location="cpu", weights_only=False)
    parity(chunk_a, chunk_b, oracle, args.seq_len)

    if args.skip_convert:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    convert_chunk(
        chunk_a,
        input_specs=[{"name": "input_ids", "shape": (1, args.seq_len), "dtype": np.int32}],
        output_specs=[{"name": "hidden", "dtype": np.float16}],
        out_path=out_dir / "qwen3_5_chunk_a.mlpackage",
        label=f"chunk_a (embed + layers [0, {args.split}))",
    )
    convert_chunk(
        chunk_b,
        input_specs=[{"name": "hidden_in",
                      "shape": (1, args.seq_len, cfg.hidden_size),
                      "dtype": np.float16}],
        output_specs=[{"name": "logits", "dtype": np.float32}],
        out_path=out_dir / "qwen3_5_chunk_b.mlpackage",
        label=f"chunk_b (layers [{args.split}, {cfg.num_hidden_layers}) + lm_head)",
    )


if __name__ == "__main__":
    main()
