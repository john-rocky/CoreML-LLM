#!/usr/bin/env python3
"""Diagnostic for Stage 1 negative result.

Loads cached activations (X, Y, W_down) from /tmp/l0_activations.npz and
answers structural questions:

  1. Is the captured intermediate Y finite? (drops NaN/inf tokens)
  2. Sparsity per token: what fraction of 8192 neurons fire above threshold?
  3. Spatial structure: are firing neurons clustered into a few slices or
     scattered uniformly?
  4. Oracle slice-K curve: at K=1..16, what fraction of |Y|^2 energy and
     what cos sim does picking the true top-K slices give?

This decides whether ANY slice-routing scheme on Gemma 3n L0 can work
before we invest more in router training.
"""
from __future__ import annotations
import argparse
import json

import numpy as np


def load(path: str):
    z = np.load(path)
    return z["X"], z["Y"], z["W_down"]


def filter_finite(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop rows where Y has non-finite values."""
    finite = np.isfinite(Y).all(axis=-1) & np.isfinite(X).all(axis=-1)
    dropped = int((~finite).sum())
    return X[finite], Y[finite], dropped


def slice_stats(Y: np.ndarray, num_slices: int) -> dict:
    T, inter = Y.shape
    per = inter // num_slices
    Y2 = (Y * Y).reshape(T, num_slices, per).sum(axis=-1)  # (T, S) slice energy
    total_e = Y2.sum(axis=-1, keepdims=True) + 1e-12  # (T,1)
    frac = Y2 / total_e  # per-token slice-energy share
    return {
        "mean_frac_per_slice": frac.mean(axis=0).tolist(),
        "std_frac_per_slice": frac.std(axis=0).tolist(),
        "min_frac_per_slice": frac.min(axis=0).tolist(),
        "max_frac_per_slice": frac.max(axis=0).tolist(),
        "uniform_baseline": 1.0 / num_slices,
    }


def neuron_firing_distribution(Y: np.ndarray, num_slices: int) -> dict:
    """For each token, find which neurons are firing (|Y_t| > thresh) and
    measure how concentrated they are across slices.
    """
    T, inter = Y.shape
    per = inter // num_slices
    abs_y = np.abs(Y)
    # Per-token threshold = 1% of that token's max activation.
    per_tok_thresh = abs_y.max(axis=-1, keepdims=True) * 0.01
    firing = (abs_y > per_tok_thresh).astype(np.float32)  # (T, inter)
    firings_per_token = firing.sum(axis=-1)  # (T,) num firing neurons
    firing_sliced = firing.reshape(T, num_slices, per).sum(axis=-1)  # (T, S)
    # Entropy of firing distribution across slices (high = uniform, low = concentrated)
    fs_norm = firing_sliced / (firings_per_token[:, None] + 1e-9)
    ent = -(fs_norm * np.log(fs_norm + 1e-12)).sum(axis=-1)
    # Top slice's share
    top_slice_share = firing_sliced.max(axis=-1) / (firings_per_token + 1e-9)
    return {
        "num_tokens": int(T),
        "mean_neurons_firing": float(firings_per_token.mean()),
        "median_neurons_firing": float(np.median(firings_per_token)),
        "frac_neurons_firing_mean": float(firings_per_token.mean() / inter),
        "slice_entropy_mean": float(ent.mean()),
        "slice_entropy_max_possible": float(np.log(num_slices)),
        "top_slice_share_mean": float(top_slice_share.mean()),
    }


def oracle_K_sweep(Y: np.ndarray, W_down: np.ndarray, num_slices: int,
                   K_list: list[int]) -> list[dict]:
    T, inter = Y.shape
    per = inter // num_slices
    Y2_slice = (Y * Y).reshape(T, num_slices, per).sum(axis=-1)  # (T, S)
    out_dense = Y @ W_down.T
    den_dense = np.linalg.norm(out_dense, axis=-1) + 1e-9
    results = []
    for K in K_list:
        # Per-token top-K slice indices.
        topk = np.argpartition(-Y2_slice, kth=K - 1, axis=-1)[:, :K]
        mask = np.zeros((T, num_slices), dtype=np.float32)
        rows = np.arange(T)[:, None]
        mask[rows, topk] = 1.0
        neuron_mask = np.repeat(mask, per, axis=1)
        Y_masked = Y * neuron_mask
        out_sparse = Y_masked @ W_down.T
        num = (out_dense * out_sparse).sum(axis=-1)
        den_sparse = np.linalg.norm(out_sparse, axis=-1) + 1e-9
        cs = num / (den_dense * den_sparse)
        # Energy coverage
        total_e = (Y * Y).sum(axis=-1) + 1e-12
        selected_e = (Y_masked * Y_masked).sum(axis=-1)
        e_cov = selected_e / total_e
        results.append({
            "K": K,
            "frac_retained": K / num_slices,
            "cos_sim_mean": float(cs.mean()),
            "cos_sim_p10": float(np.percentile(cs, 10)),
            "cos_sim_p1": float(np.percentile(cs, 1)),
            "energy_cov_mean": float(e_cov.mean()),
            "energy_cov_p10": float(np.percentile(e_cov, 10)),
        })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/l0_activations.npz")
    p.add_argument("--num-slices-list", default="8,16,32")
    p.add_argument("--K-list", default="1,2,3,4,6,8,10,12")
    p.add_argument("--out", default="/tmp/stage1_diagnostic.json")
    args = p.parse_args()

    X, Y, W = load(args.cache)
    X, Y, dropped = filter_finite(X, Y)
    print(f"loaded X{X.shape} Y{Y.shape} W{W.shape}, dropped {dropped} non-finite rows")

    K_list = [int(k) for k in args.K_list.split(",")]
    slice_counts = [int(s) for s in args.num_slices_list.split(",")]

    report = {"dropped_rows": dropped, "num_finite_tokens": int(X.shape[0]),
              "per_N": {}}

    for N in slice_counts:
        print(f"\n=== N = {N} slices ===")
        firing = neuron_firing_distribution(Y, N)
        print(f"  mean neurons firing per token: {firing['mean_neurons_firing']:.1f} "
              f"of 8192 ({firing['frac_neurons_firing_mean']*100:.1f}%)")
        print(f"  slice entropy: {firing['slice_entropy_mean']:.2f} "
              f"(max uniform = {firing['slice_entropy_max_possible']:.2f})")
        print(f"  top-slice share of firing neurons: "
              f"{firing['top_slice_share_mean']*100:.1f}%")

        valid_K = [k for k in K_list if k <= N]
        sweep = oracle_K_sweep(Y, W, N, valid_K)
        print(f"  Oracle K-sweep at N={N}:")
        print(f"   {'K':>3} | retain | cos mean | cos p10 | cos p1 | E cov")
        for r in sweep:
            print(f"   {r['K']:>3} | {r['frac_retained']:.2f}   | "
                  f"{r['cos_sim_mean']:.3f}    | "
                  f"{r['cos_sim_p10']:.3f}   | "
                  f"{r['cos_sim_p1']:.3f}  | "
                  f"{r['energy_cov_mean']:.3f}")
        report["per_N"][str(N)] = {"firing": firing, "sweep": sweep}

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
