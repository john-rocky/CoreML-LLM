#!/usr/bin/env python3
"""Verify the slice-structure finding holds across multiple sparse layers.

Runs activation capture on layers 0, 4, 9 (all marked 95% sparse in
Gemma 3n config) and reports oracle K-sweep + slice entropy for each.
If all three layers show uniform per-slice firing, the negative result
is structural to the model, not to layer 0.

Also reports raw magnitude statistics on Y so we can distinguish
"true zeros from trained sparsity" vs "small but non-zero noise".
"""
from __future__ import annotations
import argparse
import json
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


CORPUS = (
    """The history of computing is filled with paradigm shifts.
Apple's Neural Engine occupies an interesting niche. Sparse activation
patterns are an opportunity. Code generation tasks exhibit different
patterns than narrative.

class BinarySearchTree:
    def __init__(self):
        self.root = None
    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
            return

Say yes ten times. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes."""
).strip()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_layer(model, idx):
    root = getattr(model, "model", None) or model
    text = getattr(root, "language_model", None) or root
    return text.layers[idx]


def slice_entropy(Y: np.ndarray, num_slices: int) -> float:
    T, inter = Y.shape
    per = inter // num_slices
    abs_y = np.abs(Y)
    per_tok_thresh = abs_y.max(axis=-1, keepdims=True) * 0.01
    firing = (abs_y > per_tok_thresh).astype(np.float32)
    firing_sliced = firing.reshape(T, num_slices, per).sum(axis=-1)
    firings_per_token = firing.sum(axis=-1)
    fs_norm = firing_sliced / (firings_per_token[:, None] + 1e-9)
    ent = -(fs_norm * np.log(fs_norm + 1e-12)).sum(axis=-1)
    return float(ent.mean())


def magnitude_stats(Y: np.ndarray) -> dict:
    """Distribution of |Y| values to distinguish 'trained zeros' from noise."""
    abs_y = np.abs(Y).flatten()
    return {
        "n": int(abs_y.size),
        "frac_eq_zero": float((abs_y == 0).mean()),
        "frac_below_1e-4": float((abs_y < 1e-4).mean()),
        "frac_below_1e-3": float((abs_y < 1e-3).mean()),
        "frac_below_1e-2": float((abs_y < 1e-2).mean()),
        "frac_below_1e-1": float((abs_y < 1e-1).mean()),
        "frac_above_1": float((abs_y > 1.0).mean()),
        "frac_above_10": float((abs_y > 10).mean()),
        "max": float(abs_y.max()),
        "p99": float(np.percentile(abs_y, 99)),
        "p90": float(np.percentile(abs_y, 90)),
        "p50": float(np.percentile(abs_y, 50)),
    }


def oracle_K_curve(Y: np.ndarray, W: np.ndarray, num_slices: int,
                   K_list: list[int]) -> list[dict]:
    T, inter = Y.shape
    per = inter // num_slices
    Y2 = (Y * Y).reshape(T, num_slices, per).sum(axis=-1)
    out_dense = Y @ W.T
    res = []
    for K in K_list:
        topk = np.argpartition(-Y2, kth=K - 1, axis=-1)[:, :K]
        mask = np.zeros((T, num_slices), dtype=np.float32)
        rows = np.arange(T)[:, None]
        mask[rows, topk] = 1.0
        nm = np.repeat(mask, per, axis=1)
        out = (Y * nm) @ W.T
        n = (out_dense * out).sum(axis=-1)
        d = (np.linalg.norm(out_dense, axis=-1) *
             np.linalg.norm(out, axis=-1) + 1e-9)
        cs = (n / d)
        res.append({"K": K, "cos_mean": float(cs.mean()),
                    "cos_p10": float(np.percentile(cs, 10))})
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/tmp/gemma3n-e2b")
    p.add_argument("--layers", default="0,4,9,14")
    p.add_argument("--num-slices", type=int, default=16)
    p.add_argument("--K-list", default="2,4,8,12")
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--out", default="/tmp/stage1_multilayer.json")
    args = p.parse_args()

    layer_ids = [int(x) for x in args.layers.split(",")]
    K_list = [int(x) for x in args.K_list.split(",")]

    device = get_device()
    print(f"device={device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()

    enc = tok(CORPUS, return_tensors="pt", truncation=True,
              max_length=args.tokens).to(device)
    input_ids = enc["input_ids"]
    print(f"corpus tokens: {input_ids.shape[1]}")

    captured: dict[int, dict] = {}
    handles = []
    for li in layer_ids:
        layer = find_layer(model, li)
        captured[li] = {"Y": [], "W": None}
        captured[li]["W"] = layer.mlp.down_proj.weight.detach().to(torch.float32).cpu().numpy()

        def make_hook(idx):
            def hook(_m, inputs):
                y = inputs[0].detach()
                flat = y.reshape(-1, y.shape[-1]).to(torch.float32).cpu().numpy()
                captured[idx]["Y"].append(flat)
            return hook
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(li)))

    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    print(f"forward {time.time()-t0:.1f}s")
    for h in handles:
        h.remove()

    report = {"layers": {}}
    for li in layer_ids:
        Y = np.concatenate(captured[li]["Y"], axis=0)
        # Drop non-finite rows
        finite = np.isfinite(Y).all(axis=-1)
        Y = Y[finite]
        W = captured[li]["W"]
        if Y.shape[0] == 0:
            print(f"layer {li}: NO finite rows, skip")
            continue

        ent = slice_entropy(Y, args.num_slices)
        mag = magnitude_stats(Y)
        curve = oracle_K_curve(Y, W, args.num_slices, K_list)
        print(f"\n=== Layer {li} ===")
        print(f"  Y shape after finite-filter: {Y.shape}")
        print(f"  slice entropy = {ent:.3f} (max {np.log(args.num_slices):.3f})")
        print(f"  |Y| frac == 0:          {mag['frac_eq_zero']*100:.2f}%")
        print(f"  |Y| frac < 1e-4:        {mag['frac_below_1e-4']*100:.2f}%")
        print(f"  |Y| frac < 1e-3:        {mag['frac_below_1e-3']*100:.2f}%")
        print(f"  |Y| frac < 1e-2:        {mag['frac_below_1e-2']*100:.2f}%")
        print(f"  |Y| frac < 1e-1:        {mag['frac_below_1e-1']*100:.2f}%")
        print(f"  |Y| max = {mag['max']:.2g}, p99 = {mag['p99']:.3g}, p50 = {mag['p50']:.3g}")
        print(f"  Oracle K-sweep at N={args.num_slices}:")
        for r in curve:
            print(f"    K={r['K']:>2} cos mean {r['cos_mean']:.3f}  p10 {r['cos_p10']:.3f}")

        report["layers"][str(li)] = {
            "n_tokens": int(Y.shape[0]),
            "slice_entropy": ent,
            "max_entropy": float(np.log(args.num_slices)),
            "magnitude_stats": mag,
            "oracle_K_curve": curve,
        }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
