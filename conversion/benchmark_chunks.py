#!/usr/bin/env python3
"""Benchmark CoreML chunk inference on Mac Studio.

Measures per-chunk and total decode latency for 2K and 8K context models.
"""
import time
import numpy as np
import coremltools as ct
import argparse
import os

def load_model(path, compute_units):
    """Load a CoreML model."""
    t0 = time.time()
    m = ct.models.MLModel(path, compute_units=compute_units)
    dt = time.time() - t0
    print(f"  Loaded {os.path.basename(path)} in {dt:.1f}s")
    return m

def make_fp16(shape):
    return np.zeros(shape, dtype=np.float16)

def benchmark_decode(chunks_dir, ctx, num_steps=50, compute_units=ct.ComputeUnit.CPU_AND_NE,
                     wfa_mode=False, fw=2048):
    """Run decode benchmark with per-chunk timing.

    wfa_mode: If True, use WFA input format (no update_mask, FW-sized full KV).
    """
    W = 512
    hidden = 1536
    nlayers = 35
    pld = 256
    max_hd = 512
    total_pld = nlayers * pld

    print(f"\n{'='*60}")
    print(f"Benchmark: ctx={ctx}, chunks_dir={os.path.basename(chunks_dir)}")
    print(f"Compute units: {compute_units}")
    print(f"{'='*60}")

    # Load models
    print("Loading models...")
    c1 = load_model(os.path.join(chunks_dir, "chunk1.mlpackage"), compute_units)
    c2 = load_model(os.path.join(chunks_dir, "chunk2.mlpackage"), compute_units)
    c3 = load_model(os.path.join(chunks_dir, "chunk3.mlpackage"), compute_units)
    # chunk4: try mlpackage first, fallback to mlmodelc
    c4_path = os.path.join(chunks_dir, "chunk4.mlpackage")
    if not os.path.exists(c4_path):
        c4_path = os.path.join(chunks_dir, "chunk4.mlmodelc")
    c4 = load_model(c4_path, compute_units)

    # Persistent KV cache buffers
    kSliding1 = make_fp16((7, 1, W, max_hd))
    vSliding1 = make_fp16((7, 1, W, max_hd))
    kFull1 = make_fp16((1, 1, ctx, max_hd))
    vFull1 = make_fp16((1, 1, ctx, max_hd))
    kSliding2 = make_fp16((5, 1, W, max_hd))
    vSliding2 = make_fp16((5, 1, W, max_hd))
    kFull2 = make_fp16((2, 1, ctx, max_hd))
    vFull2 = make_fp16((2, 1, ctx, max_hd))

    # Timing accumulators
    times_c1 = []
    times_c2 = []
    times_c3 = []
    times_c4 = []
    times_prep = []
    times_total = []

    print(f"\nRunning {num_steps} decode steps...")

    for step in range(num_steps):
        pos = step
        t_total_start = time.time()

        # Prepare inputs
        t_prep = time.time()
        h_in = make_fp16((1, 1, hidden))
        plr = make_fp16((1, 1, total_pld))
        mask_full = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float16)
        mask_full[0, 0, 0, :pos+1] = 0
        mask_sliding = np.full((1, 1, 1, W), -65504.0, dtype=np.float16)
        valid = min(pos + 1, W)
        mask_sliding[0, 0, 0, W-valid:] = 0
        umask = make_fp16((1, 1, ctx, 1))
        umask[0, 0, min(pos, ctx-1), 0] = 1.0
        cos_s = make_fp16((1, 1, 1, 256))
        sin_s = make_fp16((1, 1, 1, 256))
        cos_f = make_fp16((1, 1, 1, 512))
        sin_f = make_fp16((1, 1, 1, 512))
        dt_prep = time.time() - t_prep
        times_prep.append(dt_prep)

        # Chunk 1
        t1 = time.time()
        out1 = c1.predict({
            "hidden_states": h_in,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "update_mask": umask,
            "per_layer_raw": plr,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "K_sliding_in": kSliding1, "V_sliding_in": vSliding1,
            "K_full_in": kFull1, "V_full_in": vFull1,
        })
        dt_c1 = time.time() - t1
        times_c1.append(dt_c1)

        h1 = out1["hidden_states_out"]
        plc = out1["per_layer_combined_out"]
        kSliding1 = out1["K_sliding_out"]
        vSliding1 = out1["V_sliding_out"]
        kFull1 = out1["K_full_out"]
        vFull1 = out1["V_full_out"]

        # Chunk 2
        t2 = time.time()
        out2 = c2.predict({
            "hidden_states": h1,
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "update_mask": umask,
            "per_layer_combined": plc,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "K_sliding_in": kSliding2, "V_sliding_in": vSliding2,
            "K_full_in": kFull2, "V_full_in": vFull2,
        })
        dt_c2 = time.time() - t2
        times_c2.append(dt_c2)

        h2 = out2["hidden_states_out"]
        kSliding2 = out2["K_sliding_out"]
        vSliding2 = out2["V_sliding_out"]
        kFull2 = out2["K_full_out"]
        vFull2 = out2["V_full_out"]
        kv13_k = out2["kv13_k"]
        kv13_v = out2["kv13_v"]
        kv14_k = out2["kv14_k"]
        kv14_v = out2["kv14_v"]

        shared = {
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "update_mask": umask,
            "per_layer_combined": plc,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
            "kv13_k": kv13_k, "kv13_v": kv13_v,
            "kv14_k": kv14_k, "kv14_v": kv14_v,
        }

        # Chunk 3
        t3 = time.time()
        d3 = dict(shared)
        d3["hidden_states"] = h2
        out3 = c3.predict(d3)
        dt_c3 = time.time() - t3
        times_c3.append(dt_c3)

        h3 = out3["hidden_states_out"]

        # Chunk 4
        t4 = time.time()
        d4 = dict(shared)
        d4["hidden_states"] = h3
        out4 = c4.predict(d4)
        dt_c4 = time.time() - t4
        times_c4.append(dt_c4)

        dt_total = time.time() - t_total_start
        times_total.append(dt_total)

        if step == 0 or (step + 1) % 10 == 0:
            print(f"  Step {step+1}: total={dt_total*1000:.1f}ms "
                  f"[c1={dt_c1*1000:.1f} c2={dt_c2*1000:.1f} "
                  f"c3={dt_c3*1000:.1f} c4={dt_c4*1000:.1f}]")

    # Skip first 5 warm-up steps
    skip = 5
    def stats(arr):
        a = arr[skip:]
        return np.mean(a)*1000, np.std(a)*1000, np.min(a)*1000, np.max(a)*1000

    print(f"\n{'='*60}")
    print(f"Results (ctx={ctx}, steps {skip+1}-{num_steps}):")
    print(f"{'='*60}")
    for name, arr in [("prep", times_prep), ("chunk1", times_c1),
                       ("chunk2", times_c2), ("chunk3", times_c3),
                       ("chunk4", times_c4), ("TOTAL", times_total)]:
        mean, std, mn, mx = stats(arr)
        print(f"  {name:8s}: {mean:6.1f}ms ±{std:4.1f} (min={mn:.1f}, max={mx:.1f})")

    mean_total = np.mean(times_total[skip:]) * 1000
    print(f"\n  Throughput: {1000/mean_total:.1f} tok/s")
    return mean_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=str, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cpu-only", action="store_true",
                        help="Use CPU only (no ANE)")
    args = parser.parse_args()

    cu = ct.ComputeUnit.CPU_ONLY if args.cpu_only else ct.ComputeUnit.CPU_AND_NE
    benchmark_decode(args.chunks_dir, args.ctx, args.steps, cu)


if __name__ == "__main__":
    main()
