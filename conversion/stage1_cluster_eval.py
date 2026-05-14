#!/usr/bin/env python3
"""Full clustering + router evaluation on Gemma 3n L0.

Builds on stage1_cofiring_cluster.py. Differences:
  * Uses the larger 2k-token cache from stage1_capture_large_corpus.py
  * Tests BOTH spectral (imbalanced) and balanced k-means clustering
  * Splits corpus 70/15/15 by token index for train/val/test so we can
    measure generalisation
  * Trains a linear router on the permuted firing pattern
  * Reports oracle vs router cos sim at several K
  * Reports the bandwidth save fraction implied by each (K, N) operating point

Decision gate at end: at K such that router cos sim >= 0.98, what is the
implied bandwidth save? If >= 40%, the approach is worth Stage 2. Else,
report and stop.
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering


def load(path):
    z = np.load(path)
    return z["X"], z["Y"], z["W_down"]


def build_firing(Y, rel_thresh=0.01):
    abs_y = np.abs(Y)
    thr = abs_y.max(axis=-1, keepdims=True) * rel_thresh
    return (abs_y > thr).astype(np.float32)


def cofiring(F):
    norms = np.linalg.norm(F, axis=0) + 1e-9
    Fn = F / norms[None, :]
    C = Fn.T @ Fn
    np.fill_diagonal(C, 0.0)
    return C


def balanced_kmeans(features, num_groups, seed=0, max_iter=100):
    """K-means with a hard balance constraint via Hungarian assignment.

    Iterates: (1) k-means assign; (2) rebalance via cost-matrix bipartite
    matching when groups exceed (n/num_groups)*1.25.
    A simple greedy approximation good enough for our 5k-neuron problem.
    """
    n = features.shape[0]
    target_size = n // num_groups
    print(f"[balanced-km] {n} points -> {num_groups} groups, target size "
          f"{target_size}")
    # initial k-means
    km = KMeans(n_clusters=num_groups, n_init=5, random_state=seed)
    labels = km.fit_predict(features)
    centers = km.cluster_centers_

    for it in range(max_iter):
        sizes = np.bincount(labels, minlength=num_groups)
        if sizes.max() <= int(target_size * 1.1) and sizes.min() >= int(target_size * 0.9):
            break
        # Reassign: greedy move from biggest cluster's farthest members to
        # closest under-filled cluster
        biggest = int(np.argmax(sizes))
        smallest = int(np.argmin(sizes))
        if sizes[biggest] - sizes[smallest] < 2:
            break
        in_big = np.where(labels == biggest)[0]
        # Distance of those points to smallest center
        d_to_small = np.linalg.norm(features[in_big] - centers[smallest], axis=-1)
        d_to_big = np.linalg.norm(features[in_big] - centers[biggest], axis=-1)
        # Move the one whose ratio (d_to_small / d_to_big) is smallest
        score = d_to_small / (d_to_big + 1e-9)
        move_idx = in_big[np.argmin(score)]
        labels[move_idx] = smallest
        # Recompute centers (cheap)
        for c in (biggest, smallest):
            mem = features[labels == c]
            if len(mem) > 0:
                centers[c] = mem.mean(axis=0)
    final_sizes = np.bincount(labels, minlength=num_groups)
    print(f"[balanced-km] final sizes: min {final_sizes.min()} "
          f"max {final_sizes.max()} mean {final_sizes.mean():.0f}")
    return labels


def perm_from_labels(labels):
    return np.argsort(labels, kind="stable")


def oracle_K_curve(Y, W, num_slices, K_list, per_slice_widths=None):
    """If per_slice_widths is None, uses uniform slices. Else uses
    irregular contiguous slices per the widths list."""
    T, inter = Y.shape
    if per_slice_widths is None:
        per = inter // num_slices
        Y2 = (Y * Y).reshape(T, num_slices, per).sum(axis=-1)
        slice_neuron_counts = np.full(num_slices, per)
        # neuron->slice mapping for masking
        slice_starts = np.arange(num_slices) * per
        slice_ends = slice_starts + per
    else:
        slice_neuron_counts = np.array(per_slice_widths)
        assert slice_neuron_counts.sum() == inter
        slice_starts = np.concatenate(([0], np.cumsum(slice_neuron_counts)[:-1]))
        slice_ends = slice_starts + slice_neuron_counts
        Y2 = np.zeros((T, num_slices), dtype=np.float32)
        for s in range(num_slices):
            chunk = Y[:, slice_starts[s]:slice_ends[s]]
            Y2[:, s] = (chunk * chunk).sum(axis=-1)
    out_dense = Y @ W.T
    den_dense = np.linalg.norm(out_dense, axis=-1) + 1e-9
    res = []
    for K in K_list:
        topk = np.argpartition(-Y2, kth=K - 1, axis=-1)[:, :K]
        mask = np.zeros((T, num_slices), dtype=np.float32)
        rows = np.arange(T)[:, None]
        mask[rows, topk] = 1.0
        # build neuron-level mask
        nm = np.zeros((T, inter), dtype=np.float32)
        for s in range(num_slices):
            nm[:, slice_starts[s]:slice_ends[s]] = mask[:, s:s+1]
        out = (Y * nm) @ W.T
        n = (out_dense * out).sum(axis=-1)
        d = den_dense * (np.linalg.norm(out, axis=-1) + 1e-9)
        cs = n / d
        # bandwidth save = (1 - selected_neurons / total_neurons) avg over tokens
        selected_neurons = (mask * slice_neuron_counts[None, :]).sum(axis=-1)
        avg_kept_frac = selected_neurons.mean() / inter
        res.append({"K": K, "cos_mean": float(cs.mean()),
                    "cos_p10": float(np.percentile(cs, 10)),
                    "cos_p1": float(np.percentile(cs, 1)),
                    "avg_kept_frac": float(avg_kept_frac),
                    "bw_save_frac": float(1 - avg_kept_frac)})
    return res


def train_linear_router(X_train, T_train, X_test, T_test, epochs=200, lr=1e-2):
    import torch
    device = torch.device("cpu")
    Xt = torch.from_numpy(X_train).to(device)
    Yt = torch.from_numpy(T_train).to(device)
    Xv = torch.from_numpy(X_test).to(device)
    Yv = torch.from_numpy(T_test).to(device)
    router = torch.nn.Linear(X_train.shape[1], T_train.shape[1])
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    best_overlap = -1.0
    best_state = None
    K = int(T_train.sum(axis=-1).max())
    for ep in range(epochs):
        router.train()
        logits = router(Xt)
        loss = bce(logits, Yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0 or ep == epochs - 1:
            router.eval()
            with torch.no_grad():
                vl = router(Xv)
                pred_top = vl.topk(K, dim=-1).indices.numpy()
                true_top = T_test.argsort(axis=-1)[:, -K:]
                overlap = 0.0
                for i in range(X_test.shape[0]):
                    overlap += len(set(pred_top[i].tolist()) & set(true_top[i].tolist())) / K
                overlap /= X_test.shape[0]
            if overlap > best_overlap:
                best_overlap = overlap
                best_state = {k: v.detach().clone() for k, v in router.state_dict().items()}
            print(f"   ep {ep+1:>3}  train {loss.item():.4f}  val overlap {overlap:.3f}")
    if best_state is not None:
        router.load_state_dict(best_state)
    return router, best_overlap


def predict_router_mask(router, X, K):
    import torch
    router.eval()
    with torch.no_grad():
        logits = router(torch.from_numpy(X))
        top = logits.topk(K, dim=-1).indices.numpy()
    mask = np.zeros((X.shape[0], logits.shape[-1]), dtype=np.float32)
    rows = np.arange(X.shape[0])[:, None]
    mask[rows, top] = 1.0
    return mask


def evaluate_with_router(X, Y, W, perm, num_slices, K, per_slice_widths,
                         router):
    """Compute router cos sim at given K on permuted-Y."""
    inter = Y.shape[1]
    T = X.shape[0]
    if per_slice_widths is None:
        per = inter // num_slices
        slice_starts = np.arange(num_slices) * per
        slice_ends = slice_starts + per
        per_slice_widths = [per] * num_slices
    else:
        slice_starts = np.concatenate(([0], np.cumsum(per_slice_widths)[:-1]))
        slice_ends = slice_starts + np.array(per_slice_widths)
    Y_perm = Y[:, perm]
    W_perm = W[:, perm]
    out_dense = Y_perm @ W_perm.T
    den_dense = np.linalg.norm(out_dense, axis=-1) + 1e-9

    mask = predict_router_mask(router, X, K)
    nm = np.zeros((T, inter), dtype=np.float32)
    for s in range(num_slices):
        nm[:, slice_starts[s]:slice_ends[s]] = mask[:, s:s+1]
    out = (Y_perm * nm) @ W_perm.T
    n = (out_dense * out).sum(axis=-1)
    d = den_dense * (np.linalg.norm(out, axis=-1) + 1e-9)
    cs = n / d
    return {"cos_mean": float(cs.mean()),
            "cos_p10": float(np.percentile(cs, 10)),
            "cos_p1": float(np.percentile(cs, 1))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/l0_activations_large.npz")
    p.add_argument("--num-slices", type=int, default=16)
    p.add_argument("--K-list", default="2,3,4,6,8,12")
    p.add_argument("--out", default="/tmp/stage1_cluster_eval.json")
    args = p.parse_args()

    X, Y, W = load(args.cache)
    print(f"loaded X{X.shape} Y{Y.shape} W{W.shape}")
    K_list = [int(k) for k in args.K_list.split(",")]
    N = args.num_slices

    # Token split 70/15/15
    T = X.shape[0]
    rng = np.random.default_rng(0)
    perm_idx = rng.permutation(T)
    n_train = int(T * 0.7)
    n_val = int(T * 0.15)
    train_i = perm_idx[:n_train]
    val_i = perm_idx[n_train:n_train+n_val]
    test_i = perm_idx[n_train+n_val:]
    print(f"split: train {len(train_i)} val {len(val_i)} test {len(test_i)}")

    # Build firing matrix on training tokens (so clustering doesn't peek at test)
    F_train = build_firing(Y[train_i], 0.01)
    fire_count = F_train.sum(axis=0)
    active_mask = fire_count > 0
    active_idx = np.where(active_mask)[0]
    print(f"active neurons on train: {active_mask.sum()} of {Y.shape[1]}")

    # Co-firing similarity
    Ca = cofiring(F_train[:, active_mask])
    print(f"cofiring: mean {Ca.mean():.3f} max {Ca.max():.3f} "
          f"p99 {np.percentile(Ca, 99):.3f}")

    report = {"args": vars(args), "num_active": int(active_mask.sum()),
              "configs": {}}

    # ----- Config A: spectral (imbalanced) -----
    print("\n=== Config A: spectral (imbalanced) ===")
    sc = SpectralClustering(n_clusters=N, affinity="precomputed",
                            assign_labels="kmeans", random_state=0, n_init=5)
    Cn = np.clip(Ca, 0.0, None)
    labels_a = sc.fit_predict(Cn)
    full_a = np.full(Y.shape[1], N, dtype=np.int64)
    full_a[active_idx] = labels_a
    perm_a = perm_from_labels(full_a)
    Y_a = Y[:, perm_a]
    W_a = W[:, perm_a]
    sizes_a = np.bincount(full_a, minlength=N+1).tolist()  # includes dead cluster
    print(f"group sizes (incl dead at end): {sizes_a}")
    # Use the FIRST N slices (the clusters), drop the dead trailing one
    # Actually we want all 8192 neurons in slices. Let's keep the layout:
    # contiguous in perm order, where neurons get assigned to slice 0..N-1
    # based on their label. Dead neurons (label=N) form an N+1'th "trash"
    # zone after the active ones.
    # For oracle eval, just take slices of equal width inter/N over Y_a.
    print("Uniform-slice oracle (split 8192 into N equal chunks):")
    sweep_a_uniform = oracle_K_curve(Y_a, W_a, N, K_list)
    for r in sweep_a_uniform:
        print(f"  K={r['K']:>2}  cos {r['cos_mean']:.3f} p10 {r['cos_p10']:.3f}"
              f"  bw_save {r['bw_save_frac']*100:.1f}%")
    # Variable-slice oracle: slice widths follow cluster sizes (dead cluster
    # absorbed into last slice). This is the layout we'd actually ship.
    sizes_for_slices = sizes_a[:N]  # cluster widths
    sizes_for_slices[-1] += sizes_a[N]  # absorb dead into last slice
    print(f"Variable-slice widths: {sizes_for_slices}")
    sweep_a_var = oracle_K_curve(Y_a, W_a, N, K_list, per_slice_widths=sizes_for_slices)
    for r in sweep_a_var:
        print(f"  K={r['K']:>2}  cos {r['cos_mean']:.3f} p10 {r['cos_p10']:.3f}"
              f"  bw_save {r['bw_save_frac']*100:.1f}%")

    # ----- Config B: balanced k-means -----
    print("\n=== Config B: balanced k-means on active neurons ===")
    feats_active = F_train[:, active_mask].T  # (n_active, T_train) — rows=neurons
    labels_b = balanced_kmeans(feats_active, N, seed=0)
    full_b = np.full(Y.shape[1], N, dtype=np.int64)
    full_b[active_idx] = labels_b
    perm_b = perm_from_labels(full_b)
    Y_b = Y[:, perm_b]
    W_b = W[:, perm_b]
    sizes_b = np.bincount(full_b, minlength=N+1).tolist()
    print(f"group sizes (incl dead): {sizes_b}")
    # Uniform-slice oracle (8192/N each):
    sweep_b_uniform = oracle_K_curve(Y_b, W_b, N, K_list)
    print("Uniform-slice oracle:")
    for r in sweep_b_uniform:
        print(f"  K={r['K']:>2}  cos {r['cos_mean']:.3f} p10 {r['cos_p10']:.3f}"
              f"  bw_save {r['bw_save_frac']*100:.1f}%")

    report["configs"]["spectral_uniform"] = sweep_a_uniform
    report["configs"]["spectral_variable"] = sweep_a_var
    report["configs"]["balanced_uniform"] = sweep_b_uniform
    report["spectral_group_sizes"] = sizes_a
    report["balanced_group_sizes"] = sizes_b

    # ----- Train router on best config + measure router cos sim -----
    # Pick config: prefer balanced if it gives within 0.02 of spectral at
    # K=4 (regular slice sizes are easier for ANE). Otherwise spectral.
    print("\n=== Picking best config for router training ===")
    chosen = "balanced_uniform"
    a_at_k4 = next(r["cos_mean"] for r in sweep_a_uniform if r["K"] == 4)
    b_at_k4 = next(r["cos_mean"] for r in sweep_b_uniform if r["K"] == 4)
    if b_at_k4 < a_at_k4 - 0.02:
        chosen = "spectral_uniform"
    print(f"chosen: {chosen}  (balanced K=4 {b_at_k4:.3f} vs spectral K=4 {a_at_k4:.3f})")
    Y_use = Y_b if chosen.startswith("balanced") else Y_a
    perm_use = perm_b if chosen.startswith("balanced") else perm_a
    sizes_use = sizes_b if chosen.startswith("balanced") else sizes_a
    # router training on permuted data: build targets per token
    inter = Y.shape[1]
    per = inter // N
    Y_p_train = Y_use[train_i]
    Y_p_val = Y_use[val_i]
    Y2_train = (Y_p_train * Y_p_train).reshape(len(train_i), N, per).sum(axis=-1)
    Y2_val = (Y_p_val * Y_p_val).reshape(len(val_i), N, per).sum(axis=-1)
    # Target = top-K binary mask. Train with K=4 target (biggest K tested at gate).
    K_target = max(K_list)
    targ_train = np.zeros_like(Y2_train)
    rows = np.arange(len(train_i))[:, None]
    topk = np.argpartition(-Y2_train, K_target - 1, axis=-1)[:, :K_target]
    targ_train[rows, topk] = 1.0
    targ_val = np.zeros_like(Y2_val)
    rows_v = np.arange(len(val_i))[:, None]
    topk_v = np.argpartition(-Y2_val, K_target - 1, axis=-1)[:, :K_target]
    targ_val[rows_v, topk_v] = 1.0
    print(f"training router (linear 2048 -> {N}, K_target={K_target})")
    router, best_overlap = train_linear_router(
        X[train_i], targ_train, X[val_i], targ_val,
        epochs=300, lr=5e-3)
    print(f"best val overlap (top-{K_target}): {best_overlap:.3f}")

    print("\n=== Router test-set cos sim ===")
    router_results = {}
    for K in K_list:
        r = evaluate_with_router(
            X[test_i], Y[test_i], W,  # NOT pre-permuted: function permutes
            perm_use, N, K, None,  # uniform slice widths
            router)
        # Oracle at same K on test set for comparison
        oracle_at_K = oracle_K_curve(Y[test_i][:, perm_use], W[:, perm_use],
                                     N, [K])[0]
        print(f"  K={K:>2}  oracle cos {oracle_at_K['cos_mean']:.3f}  "
              f"router cos {r['cos_mean']:.3f}  "
              f"gap {oracle_at_K['cos_mean'] - r['cos_mean']:+.3f}  "
              f"bw_save {oracle_at_K['bw_save_frac']*100:.1f}%")
        router_results[K] = {**r, "oracle_cos_mean": oracle_at_K["cos_mean"],
                              "bw_save_frac": oracle_at_K["bw_save_frac"]}

    report["chosen_config"] = chosen
    report["router_test_results"] = {str(K): v for K, v in router_results.items()}
    report["best_router_val_overlap"] = float(best_overlap)

    # ----- Decision -----
    print("\n=== Decision gate ===")
    # Smallest K where router cos >= 0.98
    passing = [K for K in K_list
               if router_results[K]["cos_mean"] >= 0.98]
    if passing:
        Kmin = min(passing)
        bw = router_results[Kmin]["bw_save_frac"]
        print(f"router cos>=0.98 achieved at K={Kmin}, bw_save={bw*100:.1f}%")
        if bw >= 0.40:
            print("PASS — proceed to Stage 2 CoreML feasibility")
            report["decision"] = "PASS"
            report["pass_K"] = Kmin
            report["pass_bw_save"] = bw
        else:
            print("FAIL — bandwidth save under 40%, not worth CoreML effort")
            report["decision"] = "FAIL_BW"
    else:
        print("FAIL — router never reaches 0.98 cos sim at any tested K")
        report["decision"] = "FAIL_QUALITY"

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
