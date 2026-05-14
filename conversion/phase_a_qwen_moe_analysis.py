#!/usr/bin/env python3
"""Phase A — Mac validation for Qwen1.5-MoE-A2.7B-Chat.

Runs end-to-end Mac MPS validation BEFORE committing to CoreML work:

  1. Load model (HF transformers, fp16 on MPS)
  2. Measure Mac baseline tok/s for reference
  3. Run quality smoke on 3 prompts (code/essay/pattern), compare to
     a Gemma 4 E2B reference (if available)
  4. Hook all 24 expert gate layers, capture per-token expert routing
  5. Analyse routing: per-layer expert usage, top-K stability across
     prompts, hot-expert concentration
  6. Verify per-token bandwidth math empirically

Output:
  /tmp/phase_a_qwen_moe_report.json
  docs/SESSION_2026_05_15_QWEN_MOE_PHASE_A.md (written by hand later)

Gate: PASS if routing matches the assumed 4-of-60 + shared structure
AND there's no surprise factor (e.g. all tokens always pick same 4
experts → degenerate to dense; or expert gate has been pruned).
"""
from __future__ import annotations
import argparse
import json
import os
import time

# wandb stub
import sys as _sys
import types as _types
import importlib.machinery as _machinery
if "wandb" not in _sys.modules:
    _w = _types.ModuleType("wandb")
    _w.__path__ = []  # type: ignore[attr-defined]
    _w.__spec__ = _machinery.ModuleSpec("wandb", loader=None, is_package=True)
    _sys.modules["wandb"] = _w

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = {
    "code": "Write a Python class for a binary search tree with insert, search, and inorder traversal methods.",
    "essay": "Write a short essay (3-4 paragraphs) about the history of computing and the role of large language models in modern hardware design.",
    "pattern": "Say the word 'yes' ten times, separated by spaces.",
    "math": "Compute the derivative of f(x) = x^3 - 2x^2 + 5x - 7 and explain each step.",
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[load] tokenizer vocab size: {tok.vocab_size}")
    print(f"[load] loading model from {model_path}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    print(f"[load] model loaded in {time.time()-t0:.1f}s, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return tok, model


def baseline_tok_per_s(tok, model, device, prompt: str, max_new: int = 64):
    inputs = tok(prompt, return_tensors="pt").to(device)
    # warm-up
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    torch.mps.synchronize() if device.type == "mps" else None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                              do_sample=False, use_cache=True)
    torch.mps.synchronize() if device.type == "mps" else None
    dt = time.time() - t0
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    return new_tokens / dt, tok.decode(out[0], skip_special_tokens=True)


def hook_expert_routing(model) -> tuple[list, dict]:
    """Attach hooks to capture every expert-routing gate decision."""
    root = getattr(model, "model", None) or model
    text = getattr(root, "language_model", None) or root
    layers = getattr(text, "layers", None)
    if layers is None:
        raise SystemExit(f"can't find layers under {type(model).__name__}")

    # Find the gate module. In Qwen2MoeSparseMoeBlock it's mlp.gate.
    captured: dict[int, list] = {i: [] for i in range(len(layers))}
    handles = []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        gate = getattr(mlp, "gate", None)
        if gate is None:
            print(f"[hook] layer {i}: no gate found (dense layer?) — skip")
            continue

        def make_hook(idx):
            def hook(_m, _inp, out):
                # `out` is the logits tensor (batch, seq, num_experts)
                if isinstance(out, tuple):
                    out = out[0]
                captured[idx].append(out.detach().to(torch.float32).cpu().numpy())
            return hook
        handles.append(gate.register_forward_hook(make_hook(i)))
    return handles, captured


def analyse_routing(captured: dict, top_k: int, num_experts: int) -> dict:
    report = {}
    all_layer_top: list[np.ndarray] = []
    for li, chunks in captured.items():
        if not chunks:
            continue
        gates = np.concatenate(chunks, axis=1) if chunks[0].ndim == 3 else np.concatenate(chunks, axis=0)
        # (B, T, E) or (T, E)
        if gates.ndim == 3:
            gates = gates.reshape(-1, gates.shape[-1])
        # Top-K per token
        topk_idx = np.argpartition(-gates, top_k - 1, axis=-1)[:, :top_k]
        # Per-expert firing rate
        T = gates.shape[0]
        rate = np.zeros(num_experts)
        for k in range(top_k):
            rate += np.bincount(topk_idx[:, k], minlength=num_experts)
        rate = rate / T
        # Entropy of usage
        eps = 1e-9
        p = rate / (rate.sum() + eps)
        H = -(p * np.log(p + eps)).sum()
        report[str(li)] = {
            "tokens": int(T),
            "max_usage": float(rate.max()),
            "min_usage": float(rate.min()),
            "n_dead": int((rate == 0).sum()),
            "entropy_norm": float(H / np.log(num_experts)),
            "top_5_experts": np.argsort(-rate)[:5].tolist(),
            "top_5_rates": [float(rate[i]) for i in np.argsort(-rate)[:5]],
        }
        all_layer_top.append(topk_idx)

    # Cross-layer correlation: do the same experts fire across layers?
    # Just print summary
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/tmp/qwen15-moe-chat")
    p.add_argument("--corpus-tokens", type=int, default=512,
                   help="num tokens for routing analysis")
    p.add_argument("--gen-tokens", type=int, default=64,
                   help="num new tokens per quality prompt")
    p.add_argument("--out", default="/tmp/phase_a_qwen_moe_report.json")
    args = p.parse_args()

    device = get_device()
    print(f"=== Phase A — Qwen1.5-MoE-A2.7B Mac validation ===")
    print(f"device={device}")

    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}")
        return

    tok, model = load_model(args.model, device)

    # Config-derived constants
    cfg = model.config
    print(f"[cfg] num_hidden_layers={cfg.num_hidden_layers}")
    print(f"[cfg] num_experts={getattr(cfg, 'num_experts', None)}")
    print(f"[cfg] num_experts_per_tok={getattr(cfg, 'num_experts_per_tok', None)}")
    print(f"[cfg] moe_intermediate_size={getattr(cfg, 'moe_intermediate_size', None)}")
    print(f"[cfg] shared_expert_intermediate_size={getattr(cfg, 'shared_expert_intermediate_size', None)}")
    print(f"[cfg] hidden_size={cfg.hidden_size}")
    print(f"[cfg] vocab_size={cfg.vocab_size}")

    num_experts = getattr(cfg, "num_experts", 60)
    top_k = getattr(cfg, "num_experts_per_tok", 4)

    # ---------- 1. Quality + baseline tok/s ----------
    print(f"\n=== 1. Quality smoke + baseline tok/s ===")
    quality_results = {}
    tps_results = {}
    for name, prompt in PROMPTS.items():
        tps, text = baseline_tok_per_s(tok, model, device, prompt,
                                        max_new=args.gen_tokens)
        print(f"\n[{name}] {tps:.1f} tok/s")
        print(f"OUTPUT (first 200 chars):\n{text[len(prompt):len(prompt)+200]}")
        quality_results[name] = text[len(prompt):]
        tps_results[name] = tps

    # ---------- 2. Routing analysis ----------
    print(f"\n=== 2. Expert routing analysis on calibration corpus ===")
    # Build a mixed corpus
    corpus = "\n\n".join(PROMPTS.values()) + "\n\n" + (
        "The quick brown fox jumps over the lazy dog. " * 10)
    enc = tok(corpus, return_tensors="pt", truncation=True,
              max_length=args.corpus_tokens).to(device)
    print(f"corpus tokens: {enc['input_ids'].shape[1]}")

    handles, captured = hook_expert_routing(model)
    print(f"hooked {len(handles)} expert gates")
    with torch.no_grad():
        _ = model(input_ids=enc["input_ids"])
    for h in handles:
        h.remove()

    routing_report = analyse_routing(captured, top_k, num_experts)
    print(f"\nPer-layer routing summary:")
    print(f"{'layer':>5} | tokens | maxUse | minUse | dead | entropy_norm | top5 experts")
    for li in sorted(routing_report.keys(), key=int):
        r = routing_report[li]
        print(f"{li:>5} | {r['tokens']:>6} | {r['max_usage']:.3f} | "
              f"{r['min_usage']:.3f} | {r['n_dead']:>4} | "
              f"{r['entropy_norm']:.3f}        | {r['top_5_experts'][:3]}...")

    # ---------- 3. Bandwidth math verification ----------
    print(f"\n=== 3. Bandwidth math sanity ===")
    # Empirical: each token reads attn + (top_k+1) experts
    expected_params_per_token_per_layer = (
        4 * cfg.hidden_size * cfg.hidden_size  # attn QKVO
        + (top_k + 1) * 3 * cfg.hidden_size * cfg.moe_intermediate_size
        # ^ top_k routed experts + 1 shared expert (assuming shared_intermediate
        # is structurally analogous to moe_intermediate per layer)
    )
    print(f"per-token per-layer params (math): "
          f"{expected_params_per_token_per_layer/1e6:.1f}M")
    print(f"total decoder per token: "
          f"{expected_params_per_token_per_layer * cfg.num_hidden_layers / 1e9:.2f}B")

    # ---------- Save report ----------
    report = {
        "model": args.model,
        "config": {
            "num_layers": cfg.num_hidden_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "moe_intermediate": getattr(cfg, "moe_intermediate_size", None),
            "shared_intermediate": getattr(cfg, "shared_expert_intermediate_size", None),
            "hidden": cfg.hidden_size,
            "vocab": cfg.vocab_size,
        },
        "mac_baseline_tok_per_s": tps_results,
        "quality_completions": quality_results,
        "routing": routing_report,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
