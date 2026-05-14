#!/usr/bin/env python3
"""Free recursive-tie experiment on Gemma 4 E2B.

Question: if we tie Gemma 4 E2B's decoder layers in pairs (the free,
zero-training starting point for a recursive/weight-shared model that
would run ~2x faster on ANE), how much quality do we lose?

A tied model has N/2 unique decoder blocks, each applied twice. ANE
loves this — static graph, no gather, half the per-token weight
bandwidth. The only question is the quality cost, which this measures.

Tying strategies tested (all training-free):
  * average : block = mean(layer_2i, layer_2i+1)
  * first   : block = layer_2i  (use the earlier layer for both)
  * second  : block = layer_2i+1
  * mid_only_average : tie only the middle 60% of layers (early/late
                       layers kept unique — they tend to matter more)

Quality metrics vs the untied original, on a mixed corpus:
  * next-token top-1 agreement (the headline number)
  * logit cosine similarity
  * KL divergence (untied || tied)
  * generation smoke on 3 prompts

Usage:
  pyenv shell lama-cml
  python conversion/recursive_tie_experiment.py \
    --model output/gemma4-e2b/hf_model --out /tmp/tie_experiment.json
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time

# wandb stub (Gemma loads accelerate/timm chains)
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


CORPUS = (
    "The history of computing is filled with paradigm shifts. Apple's "
    "Neural Engine occupies an interesting niche. Sparse activation "
    "patterns are an opportunity.\n\n"
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n"
    "    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n"
    "Compute the derivative of f(x) = x^3 - 2x^2 + 5x - 7. By the power "
    "rule the answer is 3x^2 - 4x + 5.\n\n"
    "User: Summarise speculative decoding. Assistant: A small drafter "
    "proposes tokens, the large target verifies them in parallel; the "
    "speedup depends on the drafter's accept rate.\n\n"
    "The transformer architecture replaced recurrence with self-attention. "
    "Each layer applies multi-head attention then a feed forward network. "
    "Residual connections and normalisation stabilise training."
)

GEN_PROMPTS = [
    "Write a Python function that reverses a linked list.",
    "Explain in two sentences why the sky is blue.",
    "List three causes of the fall of the Roman Empire.",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_layers(model):
    """Return the decoder layer ModuleList for Gemma 4."""
    root = getattr(model, "model", None) or model
    text = getattr(root, "language_model", None) or root
    layers = getattr(text, "layers", None)
    if layers is None:
        raise SystemExit(f"can't find layers under {type(model).__name__}")
    return layers


def layer_signature(layer) -> tuple:
    """Shape signature so we only tie layers that are tie-compatible.
    Gemma 4 E2B is non-uniform: FFN size changes mid-stack and every
    5th layer is global-attention with a wider q_proj."""
    sd = layer.state_dict()
    sig = []
    for k in sorted(sd):
        if "weight" in k and sd[k].dim() == 2:
            sig.append((k.split(".")[-2], tuple(sd[k].shape)))
    return tuple(sig)


def tie_layers(model, strategy: str):
    """Mutate `model` in place so decoder layers are tied in pairs,
    pairing only CONSECUTIVE layers with matching shape signatures.

    Returns (num_unique_blocks, num_total_layers, detail).
    """
    layers = find_layers(model)
    n = len(layers)
    sigs = [layer_signature(layers[i]) for i in range(n)]

    def avg_into(dst, srcs):
        with torch.no_grad():
            dsd = dst.state_dict()
            for k in dsd:
                stacked = torch.stack([s.state_dict()[k].float() for s in srcs])
                dsd[k].copy_(stacked.mean(0).to(dsd[k].dtype))

    def copy_into(dst, src):
        with torch.no_grad():
            dsd, ssd = dst.state_dict(), src.state_dict()
            for k in dsd:
                dsd[k].copy_(ssd[k])

    # Decide which consecutive pairs are eligible.
    # mid_only: skip the first/last ~20% of layers (keep them unique).
    lo, hi = 0, n
    if strategy.startswith("mid_only"):
        lo, hi = int(n * 0.2), int(n * 0.8)
    base_strategy = strategy.replace("mid_only_", "")

    tied_pairs = []
    i = lo
    while i < hi - 1:
        if sigs[i] == sigs[i + 1]:
            a, b = layers[i], layers[i + 1]
            if base_strategy == "average":
                snap_a = copy.deepcopy(a)
                avg_into(a, [snap_a, b])
                copy_into(b, a)
            elif base_strategy == "first":
                copy_into(b, a)
            elif base_strategy == "second":
                copy_into(a, b)
            else:
                raise SystemExit(f"unknown base strategy {base_strategy}")
            tied_pairs.append((i, i + 1))
            i += 2  # consume the pair
        else:
            i += 1  # signature mismatch (e.g. global-attn layer) — skip
    unique = n - len(tied_pairs)
    detail = {"tied_pairs": tied_pairs, "n_tied_pairs": len(tied_pairs)}
    return unique, n, detail


@torch.no_grad()
def logits_on_corpus(model, input_ids):
    out = model(input_ids=input_ids)
    return out.logits[0].float().cpu()  # (T, vocab)


def compare(ref_logits, tied_logits):
    """next-token top-1 agreement, logit cos sim, KL(ref||tied)."""
    # shift: predict token t+1 from position t
    ref = ref_logits[:-1]
    tied = tied_logits[:-1]
    ref_top1 = ref.argmax(-1)
    tied_top1 = tied.argmax(-1)
    top1_agree = (ref_top1 == tied_top1).float().mean().item()
    # logit cos sim per position
    num = (ref * tied).sum(-1)
    den = ref.norm(dim=-1) * tied.norm(dim=-1) + 1e-9
    cos = (num / den).mean().item()
    # KL(ref || tied)
    ref_lp = torch.log_softmax(ref, dim=-1)
    tied_lp = torch.log_softmax(tied, dim=-1)
    kl = (ref_lp.exp() * (ref_lp - tied_lp)).sum(-1).mean().item()
    return {"top1_agreement": top1_agree, "logit_cos_sim": cos,
            "kl_ref_tied": kl}


@torch.no_grad()
def gen_smoke(model, tok, device, prompt, max_new=48):
    inp = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="output/gemma4-e2b/hf_model")
    p.add_argument("--strategies", default="average,first,mid_only_average")
    p.add_argument("--out", default="/tmp/tie_experiment.json")
    args = p.parse_args()

    device = get_device()
    print(f"[tie] device={device} model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = tok(CORPUS, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    print(f"[tie] corpus {input_ids.shape[1]} tokens")

    # Reference (untied) logits + generations.
    print("[tie] loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()
    n_layers = len(find_layers(ref_model))
    print(f"[tie] {n_layers} decoder layers")
    t0 = time.time()
    ref_logits = logits_on_corpus(ref_model, input_ids)
    print(f"[tie] reference forward {time.time()-t0:.1f}s")
    ref_gens = [gen_smoke(ref_model, tok, device, pr) for pr in GEN_PROMPTS]
    del ref_model
    if device.type == "mps":
        torch.mps.empty_cache()

    report = {"model": args.model, "num_layers": n_layers,
              "reference_generations": ref_gens, "strategies": {}}

    for strat in args.strategies.split(","):
        strat = strat.strip()
        print(f"\n=== strategy: {strat} ===")
        m = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device).eval()
        unique, total, detail = tie_layers(m, strat)
        share_ratio = total / unique
        print(f"[{strat}] {unique} unique blocks / {total} layers "
              f"= {share_ratio:.2f}x weight reuse "
              f"({detail['n_tied_pairs']} pairs tied)")
        tied_logits = logits_on_corpus(m, input_ids)
        metrics = compare(ref_logits, tied_logits)
        gens = [gen_smoke(m, tok, device, pr) for pr in GEN_PROMPTS]
        print(f"[{strat}] top-1 agreement: {metrics['top1_agreement']:.4f}")
        print(f"[{strat}] logit cos sim:   {metrics['logit_cos_sim']:.4f}")
        print(f"[{strat}] KL(ref||tied):   {metrics['kl_ref_tied']:.4f}")
        for i, (pr, g) in enumerate(zip(GEN_PROMPTS, gens)):
            print(f"  gen[{i}]: {g[:120]!r}")
        report["strategies"][strat] = {
            "unique_blocks": unique, "total_layers": total,
            "share_ratio": share_ratio, "n_tied_pairs": detail["n_tied_pairs"],
            **metrics, "generations": gens,
        }
        del m
        if device.type == "mps":
            torch.mps.empty_cache()

    # Verdict
    print("\n=== Verdict ===")
    print(f"{'strategy':>18} | {'reuse':>5} | top1-agree | cos-sim | KL")
    for strat, r in report["strategies"].items():
        print(f"{strat:>18} | {r['share_ratio']:.2f}x | "
              f"{r['top1_agreement']:.3f}      | {r['logit_cos_sim']:.3f}   | "
              f"{r['kl_ref_tied']:.3f}")
    best = max(report["strategies"].items(),
              key=lambda kv: kv[1]["top1_agreement"])
    print(f"\nBest free-tie: {best[0]} — top-1 agreement "
          f"{best[1]['top1_agreement']:.3f} at {best[1]['share_ratio']:.2f}x reuse")
    if best[1]["top1_agreement"] >= 0.90:
        print("STRONG — free tie keeps >90% next-token agreement. The "
              "recursive 2x path may need little or no training.")
    elif best[1]["top1_agreement"] >= 0.75:
        print("MODERATE — meaningful drop. LoRA-relax training would be "
              "needed to recover; measure how much it recovers next.")
    else:
        print("WEAK — free tie breaks the model. Full uptrain (not just "
              "LoRA) would be needed; recursive path is training-heavy.")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
