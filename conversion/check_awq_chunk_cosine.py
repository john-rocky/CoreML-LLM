#!/usr/bin/env python3
"""Verify AWQ chunk1 vs baseline chunk1 produce similar outputs on dummy input.

If AWQ helps, INT4-palettized AWQ chunks should produce *closer* outputs to
the fp16 reference than INT4-palettized baseline chunks. The expectation:

  cos(awq_chunk1, fp_chunk1) > cos(baseline_chunk1, fp_chunk1)
"""
from __future__ import annotations
import argparse
import numpy as np
import coremltools as ct


def _decode_inputs(seq_len, ctx, hd_full, hd_swa, hidden, ple, num_full, num_swa, W):
    return {
        "K_full_in":      np.zeros((num_full, 1, ctx, hd_full), dtype=np.float16),
        "K_sliding_in":   np.zeros((num_swa, 1, W, hd_full), dtype=np.float16),
        "V_full_in":      np.zeros((num_full, 1, ctx, hd_full), dtype=np.float16),
        "V_sliding_in":   np.zeros((num_swa, 1, W, hd_full), dtype=np.float16),
        "causal_mask_full":     np.zeros((1, 1, seq_len, ctx), dtype=np.float16),
        "causal_mask_sliding":  np.zeros((1, 1, seq_len, W), dtype=np.float16),
        "cos_f": np.zeros((1, 1, seq_len, hd_full), dtype=np.float16),
        "sin_f": np.zeros((1, 1, seq_len, hd_full), dtype=np.float16),
        "cos_s": np.zeros((1, 1, seq_len, hd_swa), dtype=np.float16),
        "sin_s": np.zeros((1, 1, seq_len, hd_swa), dtype=np.float16),
        "hidden_states":  np.random.randn(1, seq_len, hidden).astype(np.float16) * 0.5,
        "per_layer_combined":  np.random.randn(1, seq_len, ple).astype(np.float16) * 0.5,
        "update_mask":    np.zeros((1, 1, ctx, 1), dtype=np.float16),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="/tmp/gemma4_chunks_K3/chunk1.mlpackage",
                    help="Baseline (no AWQ) chunk1.mlpackage")
    ap.add_argument("--awq", default="/tmp/gemma4_chunks_K3_awq/chunk1.mlpackage",
                    help="AWQ chunk1.mlpackage")
    ap.add_argument("--n-trials", type=int, default=5)
    args = ap.parse_args()

    np.random.seed(0)

    print(f"Loading baseline: {args.baseline}")
    m_base = ct.models.MLModel(args.baseline, function_name="decode_q1",
                                compute_units=ct.ComputeUnit.CPU_ONLY)
    print(f"Loading AWQ: {args.awq}")
    m_awq = ct.models.MLModel(args.awq, function_name="decode_q1",
                               compute_units=ct.ComputeUnit.CPU_ONLY)

    # Constants for chunk1 (E2B): L0-7, 7 sliding + 1 full
    HIDDEN = 1536
    PLE = 35 * 256
    HD_FULL = 512
    HD_SWA = 256
    CTX = 2048
    W = 512
    NUM_FULL = 1
    NUM_SWA = 7

    print(f"\nRunning {args.n_trials} trials on random inputs...")
    cos_sims = []
    for t in range(args.n_trials):
        feed = _decode_inputs(1, CTX, HD_FULL, HD_SWA, HIDDEN, PLE, NUM_FULL, NUM_SWA, W)
        out_base = m_base.predict(feed)
        out_awq = m_awq.predict(feed)

        # Compare hidden_states_out
        for k in out_base:
            if k == "hidden_states_out":
                a = np.asarray(out_base[k]).flatten().astype(np.float32)
                b = np.asarray(out_awq[k]).flatten().astype(np.float32)
                cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                cos_sims.append(cos)
                print(f"  trial {t}: cos(hidden_states_out)={cos:.6f}")
                break

    print(f"\nMean cos similarity: {np.mean(cos_sims):.6f}  (over {len(cos_sims)} trials)")
    print("Note: equal weights would give cos=1.0; AWQ smoothing is structurally")
    print("different so cos != 1.0. The bench (tok/s + accept rate) is the real test.")


if __name__ == "__main__":
    main()
