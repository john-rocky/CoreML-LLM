#!/usr/bin/env python3
"""RRT-style recursive Gemma 4 E2B — shared base + per-position low-rank
delta, no-training quality sweep.

Naive tie (recursive_tie_experiment.py) broke the model: setting both
layers of a pair to the average destroys per-layer specialisation. The
RRT recipe keeps a shared base BUT adds back a low-rank per-position
delta — that's what makes the recursive model trainable to near-lossless.

For each tie-eligible consecutive pair (A,B) and each 2D weight matrix:
    base   = (W_A + W_B) / 2          # the shared block — read once
    delta  = (W_A - W_B) / 2          # W_A = base+delta, W_B = base-delta
    delta_r = truncated_SVD(delta, rank=r)
    W_A_eff = base + delta_r          # per-position LoRA = +delta_r
    W_B_eff = base - delta_r          # per-position LoRA = -delta_r
(1D params — RMSNorm weights — stay per-layer; negligible bandwidth.)

r = full  → exact original (lossless, no speedup)
r = 0     → naive average tie (already known broken)
r = mid   → the operating point: how much quality survives BEFORE any
            training. This sizes the gap uptraining must close.

Bandwidth: a tied pair stores base + 2 low-rank deltas instead of 2 full
matrices. For an (out,in) matrix, the pair costs out*in + 2*r*(out+in)
vs 2*out*in — a saving whenever r < out*in / (2*(out+in)).

Usage:
  pyenv shell lama-cml
  python conversion/recursive_lowrank_experiment.py \
    --model output/gemma4-e2b/hf_model --ranks 0,64,128,256,512 \
    --out /tmp/recursive_lowrank.json
"""
from __future__ import annotations
import argparse
import json
import time

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

# Reuse corpus + helpers from the naive-tie experiment.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from recursive_tie_experiment import (
    CORPUS, GEN_PROMPTS, get_device, find_layers, layer_signature,
    logits_on_corpus, compare, gen_smoke,
)


def eligible_pairs(layers) -> list[tuple[int, int]]:
    """Consecutive layers with matching shape signatures."""
    n = len(layers)
    sigs = [layer_signature(layers[i]) for i in range(n)]
    pairs = []
    i = 0
    while i < n - 1:
        if sigs[i] == sigs[i + 1]:
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    return pairs


def truncated_svd_delta(delta: torch.Tensor, rank: int) -> torch.Tensor:
    """Rank-r approximation of a 2D delta matrix. rank<=0 → zeros;
    rank>=min(shape) → exact."""
    if rank <= 0:
        return torch.zeros_like(delta)
    out_d, in_d = delta.shape
    if rank >= min(out_d, in_d):
        return delta.clone()
    # SVD in fp32 for stability.
    U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
    Ur, Sr, Vhr = U[:, :rank], S[:rank], Vh[:rank, :]
    return (Ur * Sr) @ Vhr


def apply_recursive_lowrank(model, pairs, rank: int) -> dict:
    """Mutate model: each tied pair → shared base + ±rank-r delta.
    Returns bandwidth bookkeeping."""
    layers = find_layers(model)
    full_params = 0      # baseline: sum of all 2D weights in tied layers
    recursive_params = 0  # base (once) + 2 low-rank deltas per pair
    with torch.no_grad():
        for (ia, ib) in pairs:
            la, lb = layers[ia], layers[ib]
            sda, sdb = la.state_dict(), lb.state_dict()
            for k in sda:
                wa, wb = sda[k], sdb[k]
                if wa.dim() != 2:
                    continue  # norms etc. — keep per-layer
                base = (wa.float() + wb.float()) / 2.0
                delta = (wa.float() - wb.float()) / 2.0
                delta_r = truncated_svd_delta(delta, rank)
                wa.copy_((base + delta_r).to(wa.dtype))
                wb.copy_((base - delta_r).to(wb.dtype))
                out_d, in_d = wa.shape
                full_params += 2 * out_d * in_d
                eff_rank = (0 if rank <= 0
                            else min(rank, min(out_d, in_d)))
                recursive_params += out_d * in_d + 2 * eff_rank * (out_d + in_d)
    return {
        "full_params": full_params,
        "recursive_params": recursive_params,
        "bandwidth_ratio": (full_params / max(recursive_params, 1)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="output/gemma4-e2b/hf_model")
    p.add_argument("--ranks", default="0,64,128,256,512")
    p.add_argument("--out", default="/tmp/recursive_lowrank.json")
    args = p.parse_args()

    device = get_device()
    print(f"[rlr] device={device} model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = tok(CORPUS, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    print(f"[rlr] corpus {input_ids.shape[1]} tokens")

    print("[rlr] loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()
    layers = find_layers(ref_model)
    pairs = eligible_pairs(layers)
    n = len(layers)
    print(f"[rlr] {n} layers, {len(pairs)} tie-eligible pairs "
          f"→ {n - len(pairs)} unique blocks at full sharing")
    ref_logits = logits_on_corpus(ref_model, input_ids)
    ref_gens = [gen_smoke(ref_model, tok, device, pr) for pr in GEN_PROMPTS]
    del ref_model
    if device.type == "mps":
        torch.mps.empty_cache()

    report = {"model": args.model, "num_layers": n,
              "n_eligible_pairs": len(pairs),
              "unique_blocks_full_share": n - len(pairs),
              "reference_generations": ref_gens, "ranks": {}}

    ranks = [int(r) for r in args.ranks.split(",")]
    for r in ranks:
        print(f"\n=== rank {r} ===")
        m = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device).eval()
        t0 = time.time()
        bw = apply_recursive_lowrank(m, pairs, r)
        tied_logits = logits_on_corpus(m, input_ids)
        metrics = compare(ref_logits, tied_logits)
        gens = [gen_smoke(m, tok, device, pr) for pr in GEN_PROMPTS]
        print(f"[r={r}] applied in {time.time()-t0:.1f}s  "
              f"tied-pair bandwidth ratio {bw['bandwidth_ratio']:.2f}x")
        print(f"[r={r}] top-1 agreement: {metrics['top1_agreement']:.4f}")
        print(f"[r={r}] logit cos sim:   {metrics['logit_cos_sim']:.4f}")
        print(f"[r={r}] KL(ref||tied):   {metrics['kl_ref_tied']:.4f}")
        for i, g in enumerate(gens):
            print(f"  gen[{i}]: {g[:110]!r}")
        report["ranks"][str(r)] = {
            **metrics,
            "tied_pair_bandwidth_ratio": bw["bandwidth_ratio"],
            "full_params": bw["full_params"],
            "recursive_params": bw["recursive_params"],
            "generations": gens,
        }
        del m
        if device.type == "mps":
            torch.mps.empty_cache()

    print("\n=== Verdict ===")
    print(f"{'rank':>6} | bw-ratio | top1-agree | cos-sim |   KL")
    for r in ranks:
        d = report["ranks"][str(r)]
        print(f"{r:>6} | {d['tied_pair_bandwidth_ratio']:.2f}x   | "
              f"{d['top1_agreement']:.3f}      | {d['logit_cos_sim']:.3f}   | "
              f"{d['kl_ref_tied']:.2f}")
    # Find smallest rank with usable quality (top1 >= 0.5 AND coherent-ish)
    usable = [r for r in ranks
              if report["ranks"][str(r)]["top1_agreement"] >= 0.5]
    if usable:
        rmin = min(usable)
        d = report["ranks"][str(rmin)]
        print(f"\nSmallest rank with top-1 ≥ 0.5: r={rmin} "
              f"({d['tied_pair_bandwidth_ratio']:.2f}x bandwidth, "
              f"top-1 {d['top1_agreement']:.3f})")
        print("→ uptraining starts from a COHERENT model — the gap to close "
              "is the (1 - top1) at this rank, not from-scratch.")
        report["verdict"] = {"min_usable_rank": rmin,
                             "starting_top1": d["top1_agreement"]}
    else:
        print("\nNo rank reaches top-1 ≥ 0.5 without training. The shared-"
              "base+low-rank-delta init alone is insufficient — uptraining "
              "must do more, OR a smarter init (true RRT SVD) is needed.")
        report["verdict"] = {"min_usable_rank": None}

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
