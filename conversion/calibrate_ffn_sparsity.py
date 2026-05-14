#!/usr/bin/env python3
"""Calibrate FFN activation sparsity on Gemma 4 E2B (and other models).

Phase α-1 of the sparse-activation roadmap. Discovers which intermediate
neurons in each FFN layer carry the bulk of the activation magnitude
across a calibration corpus. Outputs:
  * per-layer ranked neuron indices (top-K at multiple K thresholds)
  * per-layer cumulative magnitude curve
  * per-token hit rate (fraction of neurons fired above threshold)

Result tells us whether post-training sparsification is worth pursuing:
  * if top-30% covers 95%+ of magnitude → ~3.3× FFN bandwidth win
  * if top-50% only covers 80% → marginal, likely lossy without retrain
  * if top-10% covers 95% → Apple-FM-class sparsity, big breakthrough

Usage:
  pyenv shell lama-cml
  python conversion/calibrate_ffn_sparsity.py \
    --model output/gemma4-e2b/hf_model \
    --tokens 4096 \
    --out output/sparsity_gemma4_e2b.json

Optionally run on another model (SmolLM, Gemma 3n, OpenELM) to compare.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Optional

import sys as _sys
import types as _types
import importlib.machinery as _machinery

# Stub wandb out — its actual install in this venv has a broken
# protobuf chain that gets triggered by `accelerate` / `timm` paths
# during Gemma 3n model loading. Accelerate calls
# `importlib.util.find_spec("wandb")` so we need a proper __spec__.
if "wandb" not in _sys.modules:
    _w = _types.ModuleType("wandb")
    _w.__path__ = []  # type: ignore[attr-defined]
    _w.__spec__ = _machinery.ModuleSpec("wandb", loader=None,
                                          is_package=True)
    _sys.modules["wandb"] = _w

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CALIBRATION_TEXT = """
The history of computing is filled with paradigm shifts that redefined
what machines could do. Early mechanical calculators gave way to vacuum
tube computers, then transistors, then integrated circuits. Each
revolution brought orders of magnitude in speed, miniaturisation, and
accessibility. Today the dominant trend is parallel computing,
specifically the kind of massively parallel computation that GPUs and
neural processing units enable. Large language models have become the
flagship workload of this era, with billions of parameters trained on
trillions of tokens to capture the statistics of human language.

Yet running these models on personal devices remains a challenge. A
modern smartphone has perhaps eight gigabytes of unified memory and a
power budget measured in single-digit watts. The model weights alone
can occupy a significant fraction of that memory. Bandwidth between
DRAM and the compute units is often the binding constraint rather
than compute throughput itself. Optimisations that reduce the amount
of data moved per token — quantisation, sparsity, speculative
decoding, prefix caching — have therefore taken centre stage in the
last few years.

Apple's Neural Engine occupies an interesting niche in this landscape.
It was originally designed for small computer-vision models that fit
comfortably in its on-chip SRAM. The introduction of the Foundation
Models framework in iOS 26 pushed it into territory it was not built
for, and developers have been discovering the boundaries ever since.
Recent reverse engineering work like the Orion paper has illuminated
much of the previously opaque architecture, revealing both
constraints and opportunities.

Sparse activation patterns are one such opportunity. When a feed
forward network applies its first linear layer to a token's hidden
state, only a small fraction of the resulting intermediate neurons
fire strongly. The rest contribute negligibly to the output. If we
could predict which neurons will fire before computing the layer, we
would avoid reading the weights of the dormant ones, saving bandwidth
proportional to the dormancy rate. Studies on dense models have shown
typical dormancy rates of fifty to ninety percent depending on the
architecture, training data, and downstream task.

Code generation tasks tend to exhibit different sparsity patterns
than natural language tasks. The token distribution is heavily skewed
toward a small vocabulary of punctuation, keywords, and identifiers.
Repeated patterns are common. The intermediate activations follow
suit, with certain neurons firing reliably on opening braces, closing
brackets, semicolons, and indentation tokens. By contrast, narrative
text exercises a broader swath of the network because the next token
is less predictable from local context alone.

Mathematical reasoning sits somewhere in between. Equations and
formulas have repeated structure, but the values within them vary
freely. Models trained with chain-of-thought traces often develop
specialised "reasoning circuits" — small clusters of neurons that
activate together on multi-step problems. Identifying these clusters
and routing tokens through only the relevant subset is one path
toward more efficient inference on resource-constrained devices.

Multimodal models add yet another dimension. When an image or audio
spectrogram is processed alongside text, the intermediate
representations occupy regions of activation space that pure text
prompts never visit. Calibration corpora for sparsification must
therefore include multimodal data if the resulting sparse model is to
serve multimodal queries without degradation.
""".strip()


def get_device() -> torch.device:
    """Prefer MPS (Apple Silicon GPU) for speed; fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_ffn_modules(model) -> dict[int, dict[str, torch.nn.Module]]:
    """Locate the FFN gate/up/down projections in each decoder layer.

    Gemma 4 layers use SwiGLU: `down_proj(act(gate_proj(x)) * up_proj(x))`.
    The "intermediate activations" we want to inspect are the elementwise
    product `act(gate) * up` which is the input to `down_proj`. We hook
    the input of `down_proj` to capture this.
    """
    out: dict[int, dict[str, torch.nn.Module]] = {}
    # Gemma 4 text model lives at `model.language_model` or `model.model`
    # depending on the loader. Probe both.
    root = getattr(model, "model", None) or model
    text_model = getattr(root, "language_model", None) or root
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise SystemExit(f"cannot find decoder layers under {type(model).__name__}")
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        out[i] = {
            "down_proj": mlp.down_proj,
            "gate_proj": getattr(mlp, "gate_proj", None),
            "up_proj": getattr(mlp, "up_proj", None),
        }
    return out


def calibrate(model_path: str, num_tokens: int, out_path: str,
              corpus_path: Optional[str] = None) -> None:
    device = get_device()
    print(f"[calib] device={device} model={model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    # Load in fp16 for memory; activations also fp16. MPS supports fp16.
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    print(f"[calib] model loaded, {sum(p.numel() for p in model.parameters()):,} params")

    ffn_mods = find_ffn_modules(model)
    print(f"[calib] {len(ffn_mods)} FFN layers found")

    # Statistics accumulators per layer.
    # `mag_sum[i]` = running sum of |activation| for each intermediate neuron
    # `hit_count[i]` = number of tokens where neuron's |activation| > threshold
    # `token_count[i]` = total tokens passed through this layer
    THRESHOLD = 0.01  # |activation| > this counts as a "hit"
    mag_sum: dict[int, torch.Tensor] = {}
    hit_count: dict[int, torch.Tensor] = {}
    token_count: dict[int, int] = defaultdict(int)
    inter_size_per_layer: dict[int, int] = {}

    handles = []
    for layer_idx, mods in ffn_mods.items():
        def make_hook(li: int):
            def hook(_module, inputs):
                # forward_pre_hook signature: (module, inputs)
                # `inputs` is a tuple; the down_proj input is the SwiGLU
                # intermediate activation. Shape: (batch, seq, intermediate)
                x = inputs[0]
                if x.dim() == 3:
                    flat = x.reshape(-1, x.shape[-1])  # (B*S, intermediate)
                else:
                    flat = x.reshape(-1, x.shape[-1])
                # MPS only supports up to float32; move to CPU first then cast.
                mag = flat.detach().abs().to(device="cpu", dtype=torch.float32)
                bs_tokens = mag.shape[0]
                inter = mag.shape[1]
                if li not in mag_sum:
                    mag_sum[li] = torch.zeros(inter, dtype=torch.float64, device="cpu")
                    hit_count[li] = torch.zeros(inter, dtype=torch.int64, device="cpu")
                    inter_size_per_layer[li] = inter
                mag_sum[li] += mag.sum(dim=0).to(torch.float64)
                hit_count[li] += (mag > THRESHOLD).sum(dim=0)
                token_count[li] += bs_tokens
            return hook
        h = mods["down_proj"].register_forward_pre_hook(make_hook(layer_idx))
        handles.append(h)

    # Build calibration tokens.
    if corpus_path and os.path.exists(corpus_path):
        with open(corpus_path) as f:
            text = f.read()
    else:
        text = CALIBRATION_TEXT
    enc = tok(text, return_tensors="pt", truncation=True, max_length=num_tokens)
    input_ids = enc["input_ids"].to(device)
    print(f"[calib] calibration corpus: {input_ids.shape[1]} tokens")

    t0 = time.time()
    with torch.no_grad():
        # Single forward through the whole corpus. Activations are
        # captured per token by hooks.
        _ = model(input_ids=input_ids)
    dt = time.time() - t0
    print(f"[calib] forward complete in {dt:.1f}s")

    for h in handles:
        h.remove()

    # Build per-layer report.
    report = {
        "model": model_path,
        "num_calibration_tokens": int(input_ids.shape[1]),
        "threshold": THRESHOLD,
        "layers": {},
    }
    cum_targets = [0.50, 0.80, 0.90, 0.95, 0.99]
    K_pcts = [10, 20, 30, 50, 70]
    for li in sorted(mag_sum.keys()):
        msum = mag_sum[li].numpy()  # (intermediate,)
        hits = hit_count[li].numpy()
        toks = token_count[li]
        inter = inter_size_per_layer[li]
        # Sort by magnitude desc.
        order = np.argsort(-msum)
        sorted_mag = msum[order]
        total = float(sorted_mag.sum())
        cum = np.cumsum(sorted_mag) / max(total, 1e-9)
        cov = {}
        for tgt in cum_targets:
            idx = int(np.searchsorted(cum, tgt))
            cov[f"cum_{int(tgt*100)}pct_neurons"] = int(idx + 1)
            cov[f"cum_{int(tgt*100)}pct_ratio"] = round((idx + 1) / inter, 4)
        top_k = {}
        for K in K_pcts:
            n = max(1, int(inter * K / 100))
            top_k[f"top_{K}pct"] = order[:n].tolist()
            top_k[f"top_{K}pct_coverage"] = round(float(cum[n - 1]), 4)
        hit_rate = (hits.astype(np.float64) / max(toks, 1)).mean()
        report["layers"][str(li)] = {
            "intermediate_size": inter,
            "tokens_passed": toks,
            "avg_hit_rate_per_neuron": round(float(hit_rate), 4),
            "coverage": cov,
            "top_k": top_k,
        }

    with open(out_path, "w") as f:
        json.dump(report, f)
    print(f"[calib] wrote {out_path}")

    # Print a compact summary.
    print("\n=== Per-layer coverage summary ===")
    print(f"{'layer':>5} | "
          + " | ".join([f"top{p}%cov" for p in K_pcts])
          + " | hits/tok")
    for li in sorted(report["layers"].keys(), key=int):
        L = report["layers"][li]
        cov_row = [
            f"{L['top_k'][f'top_{p}pct_coverage']:.2f}" for p in K_pcts
        ]
        print(f"{li:>5} | " + " | ".join(cov_row)
              + f" | {L['avg_hit_rate_per_neuron']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model path or HF hub ID")
    p.add_argument("--tokens", type=int, default=2048,
                   help="num calibration tokens (truncate corpus)")
    p.add_argument("--out", required=True,
                   help="output JSON path")
    p.add_argument("--corpus",
                   help="optional path to calibration text (else builtin)")
    args = p.parse_args()
    calibrate(args.model, args.tokens, args.out, args.corpus)


if __name__ == "__main__":
    main()
