#!/usr/bin/env python3
"""Time chunk2 decode_q1 (K=1) and verify_qK on ANE / GPU / CPU separately.

Goal: split "verify time on ANE" into compute-bound (K-linear) vs
fixed-cost components by profiling at multiple K and multiple compute units.

Currently we have a K=3 verify build. We'll time decode_q1 (K=1) and
verify_qK (K=3) on each compute unit, repeated for stable timing.

Bundle has chunk2.mlpackage already built.
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import coremltools as ct


HEAD_DIM_FULL = 512
HEAD_DIM_SWA = 256
HIDDEN = 1536
PER_LAYER_DIM = 256
NUM_LAYERS = 35
PLE_TOTAL = NUM_LAYERS * PER_LAYER_DIM  # 8960
CTX = 2048
W = 512


def _decode_inputs(num_sliding=5, num_full=2):
    return {
        "K_full_in":      np.zeros((num_full, 1, CTX, HEAD_DIM_FULL), dtype=np.float16),
        "K_sliding_in":   np.zeros((num_sliding, 1, W, HEAD_DIM_FULL), dtype=np.float16),
        "V_full_in":      np.zeros((num_full, 1, CTX, HEAD_DIM_FULL), dtype=np.float16),
        "V_sliding_in":   np.zeros((num_sliding, 1, W, HEAD_DIM_FULL), dtype=np.float16),
        "causal_mask_full":     np.zeros((1, 1, 1, CTX), dtype=np.float16),
        "causal_mask_sliding":  np.zeros((1, 1, 1, W), dtype=np.float16),
        "cos_f": np.zeros((1, 1, 1, HEAD_DIM_FULL), dtype=np.float16),
        "sin_f": np.zeros((1, 1, 1, HEAD_DIM_FULL), dtype=np.float16),
        "cos_s": np.zeros((1, 1, 1, HEAD_DIM_SWA), dtype=np.float16),
        "sin_s": np.zeros((1, 1, 1, HEAD_DIM_SWA), dtype=np.float16),
        "hidden_states":  np.zeros((1, 1, HIDDEN), dtype=np.float16),
        "per_layer_combined":  np.zeros((1, 1, PLE_TOTAL), dtype=np.float16),
        "update_mask":    np.zeros((1, 1, CTX, 1), dtype=np.float16),
    }


def _verify_inputs(K: int, num_sliding=5, num_full=2):
    return {
        "K_full_in":      np.zeros((num_full, 1, CTX, HEAD_DIM_FULL), dtype=np.float16),
        "K_sliding_in":   np.zeros((num_sliding, 1, W, HEAD_DIM_FULL), dtype=np.float16),
        "V_full_in":      np.zeros((num_full, 1, CTX, HEAD_DIM_FULL), dtype=np.float16),
        "V_sliding_in":   np.zeros((num_sliding, 1, W, HEAD_DIM_FULL), dtype=np.float16),
        "causal_mask_full":     np.zeros((1, 1, K, CTX), dtype=np.float16),
        "causal_mask_sliding":  np.zeros((1, 1, K, W), dtype=np.float16),
        "cos_f": np.zeros((1, 1, K, HEAD_DIM_FULL), dtype=np.float16),
        "sin_f": np.zeros((1, 1, K, HEAD_DIM_FULL), dtype=np.float16),
        "cos_s": np.zeros((1, 1, K, HEAD_DIM_SWA), dtype=np.float16),
        "sin_s": np.zeros((1, 1, K, HEAD_DIM_SWA), dtype=np.float16),
        "hidden_states":  np.zeros((1, K, HIDDEN), dtype=np.float16),
        "per_layer_combined":  np.zeros((1, K, PLE_TOTAL), dtype=np.float16),
        "update_indicator": np.zeros((1, 1, CTX, K), dtype=np.float16),
    }


def _bench(label, model, feed, n_iter=20, n_warm=3):
    times = []
    for i in range(n_warm + n_iter):
        t0 = time.perf_counter()
        _ = model.predict(feed)
        dt = time.perf_counter() - t0
        if i >= n_warm:
            times.append(dt)
    arr = np.array(times)
    return arr.mean() * 1000, arr.std() * 1000, arr.min() * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="/tmp/gemma4_chunks_K3/chunk2.mlpackage",
                    help="Path to chunk2.mlpackage with verify_qK function")
    ap.add_argument("--K", type=int, default=3,
                    help="K value baked into verify_qK in this package")
    ap.add_argument("--n-iter", type=int, default=20)
    args = ap.parse_args()

    print(f"Package: {args.package}  K (verify): {args.K}  n_iter: {args.n_iter}")

    for unit_label, unit in [
        ("ANE_only",     ct.ComputeUnit.CPU_AND_NE),
        ("GPU_only",     ct.ComputeUnit.CPU_AND_GPU),
        ("CPU_only",     ct.ComputeUnit.CPU_ONLY),
        ("ALL",          ct.ComputeUnit.ALL),
    ]:
        print(f"\n=== compute_units = {unit_label} ===")
        # decode_q1 (K=1)
        try:
            spec_dec = ct.utils.load_spec(args.package)
            model_dec = ct.models.MLModel(args.package, compute_units=unit,
                                           function_name="decode_q1")
            mean, std, mn = _bench("decode_q1", model_dec, _decode_inputs(), args.n_iter)
            print(f"  decode_q1   (K=1):       mean={mean:7.2f}ms  std={std:5.2f}  min={mn:7.2f}")
        except Exception as e:
            print(f"  decode_q1 FAILED: {e}")
        # verify_qK
        try:
            model_ver = ct.models.MLModel(args.package, compute_units=unit,
                                           function_name="verify_qK")
            mean, std, mn = _bench("verify_qK", model_ver, _verify_inputs(args.K), args.n_iter)
            print(f"  verify_qK   (K={args.K}):       mean={mean:7.2f}ms  std={std:5.2f}  min={mn:7.2f}")
        except Exception as e:
            print(f"  verify_qK FAILED: {e}")


if __name__ == "__main__":
    main()
