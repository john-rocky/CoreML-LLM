"""Layer-by-layer ANE vs CPU fp16 drift analysis. Builds a FullModel that
emits hidden states after layers {0, 5, 11, 17, 23} + final logits. Runs
the compiled mlpackage on ANE and CPU and compares cosine similarity at
every tap. Pinpoints where ANE drift enters.
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
ORACLE = Path(__file__).parent / "qwen3_5_reference_logits.pt"
TAP_LAYERS = [0, 5, 11, 17, 23]


class TappedModel(nn.Module):
    """Full model that also emits hidden states after selected layers."""
    def __init__(self, cfg, hf_model, seq_len):
        super().__init__()
        assert seq_len % 64 == 0
        self.S = seq_len
        self.eps = cfg.rms_norm_eps
        self.embed_w = nn.Parameter(hf_model.model.embed_tokens.weight.detach().clone(),
                                     requires_grad=False)
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                   hf_model.model.norm.weight)
        self.lm_head_w = nn.Parameter(hf_model.lm_head.weight.detach().clone(),
                                       requires_grad=False)
        self.layers = nn.ModuleList([
            DecoderLayer(cfg, hf_model.model.layers[i], seq_len)
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
        taps = {}
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, self.cos, self.sin)
            if i in TAP_LAYERS:
                taps[i] = hidden
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.lm_head_w)
        return (logits,
                taps[0], taps[5], taps[11], taps[17], taps[23])


def cos_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-12))


def run_and_diff(path):
    print(f"\n=== layer-tap drift (ANE vs CPU) ===")
    mlm_ane = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    mlm_cpu = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)

    oracle = torch.load(str(ORACLE), map_location="cpu", weights_only=False)

    per_prompt = []
    for rec in oracle["records"][:5]:  # 5 prompts for speed
        ids = rec["input_ids"].numpy()
        S_real = ids.shape[1]
        padded = np.zeros((1, 64), dtype=np.int32)
        padded[:, :S_real] = ids.astype(np.int32)
        inp = {"input_ids": padded}

        out_ane = mlm_ane.predict(inp)
        out_cpu = mlm_cpu.predict(inp)

        row = {"S": S_real, "prompt": rec["prompt"][:30]}
        for i, tap in enumerate(TAP_LAYERS):
            key = f"tap_{tap}"
            c = cos_sim(out_ane[key][0, :S_real], out_cpu[key][0, :S_real])
            row[key] = c
        c_logits = cos_sim(out_ane["logits"][0, :S_real], out_cpu["logits"][0, :S_real])
        row["logits"] = c_logits
        per_prompt.append(row)
        print(f"  S={S_real:3d}  " +
              "  ".join(f"L{t}={row[f'tap_{t}']:.4f}" for t in TAP_LAYERS) +
              f"  logits={row['logits']:.4f}  prompt={row['prompt']!r}")

    print(f"\n  mean cos (across 5 prompts):")
    for tap in TAP_LAYERS:
        key = f"tap_{tap}"
        m = sum(p[key] for p in per_prompt) / len(per_prompt)
        print(f"    layer_{tap:2d}: {m:.4f}")
    m_logits = sum(p["logits"] for p in per_prompt) / len(per_prompt)
    print(f"    logits:   {m_logits:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--reuse", type=str, default=None,
                    help="if given, skip convert and reuse an existing mlpackage")
    args = ap.parse_args()

    if args.reuse:
        run_and_diff(Path(args.reuse))
        return

    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    hf = Qwen3_5ForCausalLM.from_pretrained(MODEL_ID, config=cfg, torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    model = TappedModel(cfg, hf, 64).eval().float()
    del hf
    print(f"  model built, params={sum(p.numel() for p in model.parameters())/1e9:.3f}B")

    example = (torch.zeros(1, 64, dtype=torch.int32),)
    traced = torch.jit.trace(model, example, strict=False)
    print("  trace OK")

    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=[ct.TensorType(name="input_ids", shape=(1, 64), dtype=np.int32)],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="tap_0", dtype=np.float32),
            ct.TensorType(name="tap_5", dtype=np.float32),
            ct.TensorType(name="tap_11", dtype=np.float32),
            ct.TensorType(name="tap_17", dtype=np.float32),
            ct.TensorType(name="tap_23", dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print("  convert OK")

    out_dir = Path(args.out_dir or tempfile.mkdtemp(prefix="qwen35_taps_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qwen3_5_tapped.mlpackage"
    ct_model.save(str(out_path))
    print(f"  saved {out_path}")

    run_and_diff(out_path)


if __name__ == "__main__":
    main()
