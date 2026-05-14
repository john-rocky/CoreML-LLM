#!/usr/bin/env python3
"""Final-chance test for Phase β-1 Strategy A.

Hypothesis: Gemma 3n's trained sparsity makes individual neurons fire
~uniformly across slices when slices are contiguous index ranges. But
many models have NEURON CO-FIRING structure — sets of neurons that
tend to fire together for certain input categories. If we PERMUTE
neurons by co-firing similarity before slicing, the firing pattern
might cluster into blocks → slice-level top-K routing becomes viable.

Procedure:
  1. Load cached activations Y (T, 8192) from Stage 1 capture.
  2. Build per-token binary firing matrix F (T, 8192).
  3. Compute neuron-neuron co-firing matrix C (8192, 8192) =
     normalised cosine similarity of F columns.
  4. Cluster neurons into N groups via spectral clustering (or
     agglomerative on the C matrix).
  5. Permute Y and W_down according to the clustering.
  6. Re-run the oracle K-sweep on the permuted Y.
  7. Compare cos sim vs the contiguous-slice baseline.

If the permuted oracle cos sim at K=2 of 16 jumps from ~0.77 to ≥0.95,
we have a path (build a permutation pass into the conversion). If it
stays under 0.85, the firing is truly random per token → Phase β-1 is
structurally dead.
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np


def load(path):
    z = np.load(path)
    return z["X"], z["Y"], z["W_down"]


def build_firing(Y: np.ndarray, rel_thresh: float = 0.01) -> np.ndarray:
    abs_y = np.abs(Y)
    thr = abs_y.max(axis=-1, keepdims=True) * rel_thresh
    return (abs_y > thr).astype(np.float32)


def cofiring(F: np.ndarray) -> np.ndarray:
    """Cosine similarity between neuron firing vectors. F: (T, N). C: (N, N)."""
    # Normalise columns
    norms = np.linalg.norm(F, axis=0) + 1e-9
    Fn = F / norms[None, :]
    C = Fn.T @ Fn
    np.fill_diagonal(C, 0.0)
    return C


def cluster_neurons(C: np.ndarray, num_groups: int, seed: int = 0) -> np.ndarray:
    """Spectral clustering on the cofiring similarity matrix."""
    from sklearn.cluster import SpectralClustering
    print(f"[cluster] spectral clustering {C.shape} -> {num_groups} groups")
    t0 = time.time()
    sc = SpectralClustering(n_clusters=num_groups, affinity="precomputed",
                            assign_labels="kmeans", random_state=seed,
                            n_init=5)
    # SpectralClustering wants non-negative similarity. Clamp.
    Cn = np.clip(C, 0.0, None)
    labels = sc.fit_predict(Cn)
    print(f"[cluster] done in {time.time()-t0:.1f}s, group sizes: "
          f"{np.bincount(labels).tolist()}")
    return labels


def permutation_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return a permutation that sorts neurons by cluster label."""
    return np.argsort(labels, kind="stable")


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
        # Also compute the variance of slice energy share per token, after
        # this permutation. Higher variance => more concentrated => better
        # for slice-routing.
        slice_share = Y2 / (Y2.sum(axis=-1, keepdims=True) + 1e-9)
        gini = (slice_share.max(axis=-1) - slice_share.min(axis=-1)).mean()
        res.append({"K": K, "cos_mean": float(cs.mean()),
                    "cos_p10": float(np.percentile(cs, 10)),
                    "gini": float(gini)})
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/l0_activations.npz")
    p.add_argument("--num-groups", type=int, default=16)
    p.add_argument("--K-list", default="1,2,3,4,6,8,12")
    p.add_argument("--out", default="/tmp/stage1_cofiring.json")
    args = p.parse_args()

    X, Y, W = load(args.cache)
    finite = np.isfinite(Y).all(axis=-1)
    Y = Y[finite]
    print(f"loaded Y {Y.shape}, W {W.shape}")

    K_list = [int(k) for k in args.K_list.split(",")]

    # Build firing matrix
    F = build_firing(Y, rel_thresh=0.01)  # (T, 8192)
    print(f"firing: mean per-tok {F.sum(axis=-1).mean():.1f}, per-neuron "
          f"fire rate {F.mean()*100:.2f}%")

    # Only cluster neurons that fire at all (skip permanently-dead ones)
    fire_count = F.sum(axis=0)
    active = fire_count > 0
    active_idx = np.where(active)[0]
    print(f"active neurons (fired at least once): {active.sum()} of "
          f"{F.shape[1]}")

    if active.sum() < 100:
        print("ERROR: too few active neurons for clustering")
        return

    Fa = F[:, active]
    Ca = cofiring(Fa)
    print(f"cofiring stats: mean {Ca.mean():.3f} max {Ca.max():.3f} "
          f"p99 {np.percentile(Ca, 99):.3f}")

    labels_active = cluster_neurons(Ca, args.num_groups, seed=0)

    # Map back to full 8192-neuron index space. Inactive neurons get
    # placed into a dedicated "dead" cluster, then permuted to the end.
    full_labels = np.full(F.shape[1], args.num_groups, dtype=np.int64)
    full_labels[active_idx] = labels_active

    perm = permutation_from_labels(full_labels)
    Y_perm = Y[:, perm]
    W_perm = W[:, perm]

    # Compare contiguous slicing on permuted Y vs original Y
    print("\n=== Original (contiguous-index) slicing ===")
    base = oracle_K_curve(Y, W, args.num_groups, K_list)
    for r in base:
        print(f"  K={r['K']:>2}  cos_mean {r['cos_mean']:.3f}  "
              f"p10 {r['cos_p10']:.3f}  gini {r['gini']:.3f}")
    print("\n=== Permuted (cofiring-clustered) slicing ===")
    permuted = oracle_K_curve(Y_perm, W_perm, args.num_groups, K_list)
    for r in permuted:
        print(f"  K={r['K']:>2}  cos_mean {r['cos_mean']:.3f}  "
              f"p10 {r['cos_p10']:.3f}  gini {r['gini']:.3f}")

    # Verdict
    print("\n=== Verdict ===")
    for r_base, r_perm in zip(base, permuted):
        delta = r_perm["cos_mean"] - r_base["cos_mean"]
        print(f"  K={r_base['K']:>2}: base {r_base['cos_mean']:.3f} -> "
              f"permuted {r_perm['cos_mean']:.3f}  Δ={delta:+.3f}")

    report = {"baseline": base, "permuted": permuted,
              "num_active_neurons": int(active.sum()),
              "cofiring_mean": float(Ca.mean()),
              "cofiring_max": float(Ca.max()),
              "cofiring_p99": float(np.percentile(Ca, 99))}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
