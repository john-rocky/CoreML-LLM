"""Phase 4a: end-to-end 24-layer Qwen3.5-0.8B converter (fp16, text-only).

Builds the full model: embed_tokens -> 24 decoder layers (18 linear_attention +
6 full_attention) -> final RMSNorm -> lm_head (tied). RoPE cos/sin are
precomputed at init for the fixed seq_len and stored as buffers so the CoreML
graph takes just input_ids.

Gates:
  (1) parity cos >= 0.998 vs Phase 1 oracle (qwen3_5_reference_logits.pt),
      measured on the real-token positions of every oracle prompt whose
      length <= SEQ_LEN.
  (2) ANE placement >= 99% of compute ops via MLComputePlan audit.

Reuses DecoderLayer / RMSNorm / MLP primitives from Phase 3b.
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

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import coremltools as ct

from test_qwen3_5_stack_trace import DecoderLayer, RMSNorm


MODEL_ID = "Qwen/Qwen3.5-0.8B"
ORACLE_PATH = Path(__file__).parent / "qwen3_5_reference_logits.pt"
DEFAULT_SEQ_LEN = 64


class FullModel(nn.Module):
    """Qwen3.5-0.8B text-only, fixed seq_len. Input: input_ids (int32, 1xS).
    Output: logits (1, S, vocab_size)."""

    def __init__(self, cfg, hf_model, seq_len: int):
        super().__init__()
        assert seq_len % 64 == 0, "seq_len must be divisible by chunk_size=64"
        self.S = seq_len
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.num_layers = cfg.num_hidden_layers
        self.eps = cfg.rms_norm_eps

        # Embedding + final norm + lm_head. tie_word_embeddings=True, but we
        # still read lm_head.weight explicitly (HF materializes the tie, so the
        # tensor identity is already shared and torch.save/load won't duplicate).
        self.embed_w = nn.Parameter(
            hf_model.model.embed_tokens.weight.detach().clone(), requires_grad=False
        )
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                   hf_model.model.norm.weight)
        self.lm_head_w = nn.Parameter(
            hf_model.lm_head.weight.detach().clone(), requires_grad=False
        )

        self.layers = nn.ModuleList([
            DecoderLayer(cfg, hf_model.model.layers[i], seq_len)
            for i in range(self.num_layers)
        ])

        # Precompute RoPE cos/sin for fixed seq_len. For text-only, the 3
        # MRoPE grids collapse to the same plain RoPE.
        rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        dummy = torch.zeros(1, seq_len, cfg.hidden_size)
        with torch.no_grad():
            cos, sin = rot(dummy, pos)
        self.register_buffer("cos", cos.detach().clone(), persistent=False)
        self.register_buffer("sin", sin.detach().clone(), persistent=False)

    def forward(self, input_ids):
        # (1, S) -> (1, S, H). F.embedding requires long indices; the explicit
        # cast becomes a single dtype-change op in MIL.
        hidden = F.embedding(input_ids.to(torch.long), self.embed_w)
        for layer in self.layers:
            hidden = layer(hidden, self.cos, self.sin)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.lm_head_w)
        return logits


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def parity_vs_oracle(model: FullModel, oracle: dict, seq_len: int):
    """Compare fp32 torch FullModel logits to Phase 1 oracle for every prompt
    whose real length <= seq_len. Compare logits[:, :S_real, :] vs oracle
    logits_prefill (S_real, V)."""
    pad_id = 0  # eos_token_id default in oracle was eos or 0; right-pad tokens
                # don't affect left positions (causal), exact pad id is moot
    per_prompt = []
    print(f"\n=== parity vs oracle (seq_len={seq_len}) ===")
    for rec in oracle["records"]:
        ids = rec["input_ids"]  # (1, S_real)
        S_real = ids.shape[1]
        if S_real > seq_len:
            print(f"  SKIP (S_real={S_real} > {seq_len}): {rec['prompt'][:40]!r}")
            continue
        padded = torch.full((1, seq_len), pad_id, dtype=ids.dtype)
        padded[:, :S_real] = ids
        with torch.no_grad():
            logits = model(padded)             # (1, seq_len, V) fp32
        logits_real = logits[0, :S_real, :].float()
        ref = rec["logits_prefill"].float()    # (S_real, V) fp16 cast to fp32

        per_pos = torch.tensor([
            cos_sim(logits_real[i], ref[i]) for i in range(S_real)
        ])
        last_pos_cos = per_pos[-1].item()
        mean_cos = per_pos.mean().item()
        min_cos = per_pos.min().item()

        # Top-1 next-token match at last position.
        pred_top1 = int(torch.argmax(logits_real[-1]).item())
        ref_top1 = int(rec["top10_last_ids"][0].item())
        top1_match = pred_top1 == ref_top1

        print(f"  S={S_real:3d}  mean={mean_cos:.6f}  min={min_cos:.6f}  "
              f"last={last_pos_cos:.6f}  top1_match={top1_match}  "
              f"prompt={rec['prompt'][:35]!r}")
        per_prompt.append({
            "S_real": S_real, "mean": mean_cos, "min": min_cos,
            "last": last_pos_cos, "top1": top1_match, "prompt": rec["prompt"],
        })

    overall_min = min(p["min"] for p in per_prompt)
    overall_mean = sum(p["mean"] for p in per_prompt) / len(per_prompt)
    top1_rate = sum(p["top1"] for p in per_prompt) / len(per_prompt)
    print(f"\n  overall: mean={overall_mean:.6f}  worst_pos={overall_min:.6f}  "
          f"top1_match={top1_rate*100:.0f}% ({sum(p['top1'] for p in per_prompt)}"
          f"/{len(per_prompt)})")
    return overall_min, overall_mean, top1_rate


def convert_to_coreml(model: FullModel, seq_len: int, out_path: Path):
    print(f"\n=== CoreML conversion (seq={seq_len}) ===")
    example = (torch.zeros(1, seq_len, dtype=torch.int32),)
    traced = torch.jit.trace(model, example, strict=False)
    print("  trace OK")

    ct_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print("  convert OK")

    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved {out_path} ({size_mb:.1f} MB)")
    return out_path, size_mb


def audit_placement(path: Path):
    """Audit device placement. coremltools 8.3's compute-plan API returns None
    for static `const` ops (9.0 classified them); we treat those as 'const'
    instead of 'unknown' and report ANE% against compute ops, matching the
    Phase 3b reporting semantics."""
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
            if a is None:
                dev = "const" if op.operator_name == "const" else "unknown"
            else:
                dev = a.preferred_compute_device.__class__.__name__
            dev_counts[dev] += 1
            op_type_by_dev.setdefault(dev, Counter())[op.operator_name] += 1

    total = sum(dev_counts.values())
    ane = dev_counts.get("MLNeuralEngineComputeDevice", 0)
    cpu = dev_counts.get("MLCPUComputeDevice", 0)
    gpu = dev_counts.get("MLGPUComputeDevice", 0)
    const = dev_counts.get("const", 0)
    unknown = dev_counts.get("unknown", 0)
    compute_total = ane + cpu + gpu + unknown
    ane_pct_compute = 100 * ane / compute_total if compute_total else 0.0

    print(f"  total ops: {total}  (compute={compute_total}, const={const})")
    print(f"    ANE:     {ane} ({100*ane/total:.2f}% total, "
          f"{ane_pct_compute:.2f}% of compute)")
    print(f"    CPU:     {cpu} ({100*cpu/total:.2f}%)")
    if gpu:
        print(f"    GPU:     {gpu} ({100*gpu/total:.2f}%)")
    if unknown:
        print(f"    unknown: {unknown} ({100*unknown/total:.2f}%)")
    for dev in ("MLCPUComputeDevice", "MLNeuralEngineComputeDevice"):
        c = op_type_by_dev.get(dev, Counter())
        if not c:
            continue
        print(f"  === {dev} top ops ===")
        for op_type, n in c.most_common(15):
            print(f"    {op_type}: {n}")
    return ane_pct_compute, dev_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--skip-convert", action="store_true",
                    help="parity only, skip CoreML conversion")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="if set, save mlpackage here; else use a tempdir")
    args = ap.parse_args()

    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    hf_model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    print(f"  num_hidden_layers={cfg.num_hidden_layers}  "
          f"hidden_size={cfg.hidden_size}  vocab={cfg.vocab_size}  "
          f"tie_emb={cfg.tie_word_embeddings}")

    # Sanity: layer_type pattern
    types = [hf_model.model.layers[i].layer_type for i in range(cfg.num_hidden_layers)]
    n_lin = sum(t == "linear_attention" for t in types)
    n_full = sum(t == "full_attention" for t in types)
    print(f"  layer_types: {n_lin} linear_attention + {n_full} full_attention")

    oracle = torch.load(ORACLE_PATH, map_location="cpu", weights_only=False)
    assert oracle["model_id"] == MODEL_ID
    assert oracle["config"]["num_hidden_layers"] == cfg.num_hidden_layers

    print(f"\nbuilding FullModel (seq_len={args.seq_len})...")
    t1 = time.time()
    model = FullModel(cfg, hf_model, args.seq_len).eval().float()
    print(f"  built in {time.time()-t1:.1f}s  "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.3f}B")

    # Free HF model; FullModel has its own parameter copies.
    del hf_model

    overall_min, overall_mean, top1_rate = parity_vs_oracle(model, oracle, args.seq_len)
    if overall_min < 0.998:
        print(f"\nFAILED parity gate (worst_pos={overall_min:.6f} < 0.998). Stopping.")
        return

    if args.skip_convert:
        print("\n--skip-convert set; done.")
        return

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="qwen35_full_"))
    out_path = out_dir / f"qwen3_5_0_8b_fp16_seq{args.seq_len}.mlpackage"
    convert_to_coreml(model, args.seq_len, out_path)
    ane_pct, dev_counts = audit_placement(out_path)

    print(f"\n=== Phase 4a summary ===")
    print(f"  parity: mean={overall_mean:.6f}  worst_pos={overall_min:.6f}  "
          f"top1_match={top1_rate*100:.0f}%")
    print(f"  ANE placement: {ane_pct:.2f}%")
    print(f"  mlpackage: {out_path}")
    if overall_min >= 0.998 and ane_pct >= 99.0:
        print("  GATE PASSED")
    elif overall_min >= 0.998 and ane_pct >= 95.0:
        print(f"  GATE SOFT-PASS (ANE {ane_pct:.2f}% in 95-99% band)")
    else:
        print("  GATE FAILED")


if __name__ == "__main__":
    main()
