#!/usr/bin/env python3
"""ANE route, angle 4 — is the Qwen MoE expert bank low-rank?

User's idea: restructure the model PyTorch-side to fit ANE. The most
promising restructuring is expert-bank factorization. MoE experts are
trained from a common initialisation, so the 60 experts of a layer may
be low-rank in expert-space:

  Option A (basis):   expert_i ≈ Σ_r U[i,r] · basis_r        (R ≪ 60)
  Option B (shared+Δ): expert_i = W_shared + Δ_i, Δ_i low-rank

If R is small, the shared part becomes a static ANE matmul (fuseable
into chunks, 1 dispatch) and per-token routing collapses to cheap
coefficient combination — no runtime gather.

This script dequantizes the MLX 3-bit Qwen MoE expert weights and runs
two SVD analyses per layer:
  1. SVD across the expert axis of the raw (60, out*in) weight matrix
  2. SVD of the deltas (expert_i − mean) — the shared+Δ decomposition

Reports how many components capture 90/95/99% of the energy. If a
handful do, the factored ANE design is viable.

Usage:
  pyenv shell lama-cml
  python conversion/ane4_expert_lowrank.py --model /tmp/qwen_moe_3bit
"""
from __future__ import annotations
import argparse
import json

import numpy as np


def dequant_layer_experts(sm, proj: str):
    """Dequantize one projection's (60, out, in) weight tensor from a
    SwitchGLU submodule. Returns a numpy fp32 array."""
    import mlx.core as mx
    qp = sm[proj]
    w, scales, biases = qp["weight"], qp["scales"], qp["biases"]
    # Infer group_size: packed last dim vs scales last dim.
    # 3-bit: each uint32 packs 32/3 ≈ 10.67 → mlx uses 32-value groups.
    # group_size = (true_in) / scales.shape[-1]. true_in we know per proj.
    deq = mx.dequantize(w, scales, biases, group_size=64, bits=3)
    # mlx bf16 -> fp32 inside mlx first; numpy can't buffer-convert bf16.
    return np.array(deq.astype(mx.float32))  # (60, out, in)


def svd_energy(mat: np.ndarray) -> dict:
    """SVD of (n_experts, features) matrix; report energy-capture ranks."""
    # Center? No — we want raw rank for Option A.
    s = np.linalg.svd(mat, compute_uv=False)
    energy = (s ** 2)
    cum = np.cumsum(energy) / max(energy.sum(), 1e-12)
    out = {}
    for tgt in (0.90, 0.95, 0.99):
        r = int(np.searchsorted(cum, tgt)) + 1
        out[f"rank_{int(tgt*100)}"] = r
    out["top1_share"] = float(energy[0] / max(energy.sum(), 1e-12))
    out["top8_share"] = float(energy[:8].sum() / max(energy.sum(), 1e-12))
    out["n_singular"] = int(len(s))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/tmp/qwen_moe_3bit")
    p.add_argument("--layers", default="0,6,12,18,23",
                   help="comma-sep layer indices to analyse")
    p.add_argument("--out", default="/tmp/qwen_moe_expert_lowrank.json")
    args = p.parse_args()

    import mlx_lm
    print(f"[load] {args.model}")
    model, tok = mlx_lm.load(args.model)
    layer_ids = [int(x) for x in args.layers.split(",")]
    num_experts = model.model.layers[0].mlp.num_experts
    print(f"[cfg] {num_experts} experts/layer, analysing layers {layer_ids}")

    report = {"model": args.model, "num_experts": num_experts, "layers": {}}

    for li in layer_ids:
        sm = model.model.layers[li].mlp["switch_mlp"]
        layer_rep = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            try:
                W = dequant_layer_experts(sm, proj)  # (60, out, in)
            except Exception as e:
                print(f"  L{li} {proj}: dequant failed: {e}")
                continue
            E = W.shape[0]
            flat = W.reshape(E, -1)  # (60, out*in)

            # Option A: raw SVD across expert axis
            optA = svd_energy(flat)

            # Option B: shared + delta
            shared = flat.mean(axis=0, keepdims=True)
            delta = flat - shared
            optB = svd_energy(delta)
            # how much energy is in the shared component vs deltas
            shared_e = float((shared ** 2).sum() * E)
            delta_e = float((delta ** 2).sum())
            shared_frac = shared_e / max(shared_e + delta_e, 1e-12)

            layer_rep[proj] = {
                "shape": list(W.shape),
                "optA_raw_svd": optA,
                "optB_delta_svd": optB,
                "shared_energy_frac": round(shared_frac, 4),
            }
            print(f"  L{li:>2} {proj:<10}: "
                  f"raw rank95={optA['rank_95']:>2}/{E}  "
                  f"delta rank95={optB['rank_95']:>2}/{E}  "
                  f"shared_frac={shared_frac:.3f}  "
                  f"top8_share(raw)={optA['top8_share']:.2f}")
        report["layers"][str(li)] = layer_rep

    # Aggregate verdict
    print(f"\n=== Verdict ===")
    all_raw95 = []
    all_delta95 = []
    all_shared = []
    for lr in report["layers"].values():
        for pr in lr.values():
            all_raw95.append(pr["optA_raw_svd"]["rank_95"])
            all_delta95.append(pr["optB_delta_svd"]["rank_95"])
            all_shared.append(pr["shared_energy_frac"])
    if all_raw95:
        mean_raw = np.mean(all_raw95)
        mean_delta = np.mean(all_delta95)
        mean_shared = np.mean(all_shared)
        print(f"mean raw-SVD rank for 95% energy:   {mean_raw:.1f} / {num_experts}")
        print(f"mean delta-SVD rank for 95% energy: {mean_delta:.1f} / {num_experts}")
        print(f"mean shared-component energy frac:  {mean_shared:.3f}")
        report["verdict"] = {
            "mean_raw_rank95": float(mean_raw),
            "mean_delta_rank95": float(mean_delta),
            "mean_shared_frac": float(mean_shared),
        }
        if mean_raw <= 16 or (mean_shared > 0.5 and mean_delta <= 16):
            print("PROMISING — expert bank is low-rank. A factored design "
                  "(shared static basis + small per-expert combine) is "
                  "ANE-viable. Worth building + measuring.")
            report["verdict"]["status"] = "PROMISING"
        elif mean_raw <= 30:
            print("PARTIAL — moderate low-rank structure. Factoring would "
                  "halve dispatch cost at best; marginal.")
            report["verdict"]["status"] = "PARTIAL"
        else:
            print("FULL-RANK — experts are near-independent. Factorization "
                  "won't help; this ANE angle is closed.")
            report["verdict"]["status"] = "FULL_RANK"

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
