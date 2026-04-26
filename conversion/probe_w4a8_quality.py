#!/usr/bin/env python3
"""W4 vs W4A8 quality probe (Stage 1, cml9 PR #2577 final form).

Compares chunk_1.mlpackage outputs from two builds:
  A: --linear-projections                    (W4 LUT, fp16 activations)
  B: --linear-projections --activation-quant (W4 LUT + INT8 activations)

For N synthetic inputs (N=32 default), runs both models, captures
hidden_states_out, computes:
  - per-tensor max_abs_diff
  - cosine similarity
  - top-1 token disagreement % (after lm_head — only chunk_4 emits
    token_id, so we compare hidden_states_out at chunk_1 and stop)

Run from repo root:
  python conversion/probe_w4a8_quality.py \
      --w4   /tmp/g4_w4a8/w4_linear/chunk_1.mlpackage \
      --w4a8 /tmp/g4_w4a8/w4a8_linear/chunk_1.mlpackage \
      --samples 32

Verdict: target < 1% top-1 disagreement and max_abs_diff < ~0.5
(hidden_states scale is ~O(1) post-RMSNorm).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct


def _make_inputs(input_specs, num_samples, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(num_samples):
        d = {}
        for spec in input_specs:
            shape = tuple(int(s) for s in spec.shape)
            np_dtype = spec.dtype
            name = spec.name
            if np_dtype == np.int32:
                # current_pos / ring_pos: monotone valid positions
                d[name] = np.full(shape, i, dtype=np.int32)
            elif "mask" in name:
                d[name] = np.zeros(shape, dtype=np.float16)
            elif name.startswith(("cos_", "sin_")):
                d[name] = rng.uniform(-1.0, 1.0, shape).astype(np.float16)
            else:
                d[name] = rng.normal(0.0, 0.5, shape).astype(np.float16)
        samples.append(d)
    return samples


def _load_real_inputs(npz_path, input_names, max_samples):
    """Load real-prompt inputs produced by gen_calib_data_real.py."""
    data = np.load(npz_path)
    n = int(data["_meta_num_samples"][0])
    samples = []
    for i in range(min(n, max_samples)):
        d = {}
        ok = True
        for name in input_names:
            key = f"sample_{i:03d}__{name}"
            if key in data.files:
                d[name] = data[key]
            else:
                ok = False
                break
        if ok:
            samples.append(d)
    return samples


def _input_specs_from_model(mlmodel):
    """Extract (name, shape, dtype) from an mlpackage's input description.

    MLMultiArrayDataType enum:
      0 INVALID, 65552 FLOAT32, 65568 FLOAT16, 65600 DOUBLE, 131104 INT32
    Our converter writes FLOAT32 for fp16-declared TensorType inputs in the
    description proto (CoreML auto-converts at predict time), so map both
    65552 and 65568 to numpy float16 — the runtime accepts fp16 ndarrays
    against fp32-declared multiarray inputs.
    """
    spec = mlmodel.get_spec()
    inputs = []
    for desc in spec.description.input:
        name = desc.name
        if desc.type.WhichOneof("Type") != "multiArrayType":
            continue
        marr = desc.type.multiArrayType
        shape = tuple(marr.shape)
        dt_code = marr.dataType
        if dt_code == 131104:
            np_dtype = np.int32
        else:
            np_dtype = np.float16
        class _S:
            pass
        s = _S()
        s.name = name
        s.shape = shape
        s.dtype = np_dtype
        inputs.append(s)
    return inputs


def _is_stateful(mlmodel):
    spec = mlmodel.get_spec()
    if spec.HasField("description"):
        if hasattr(spec.description, "state"):
            return len(spec.description.state) > 0
    return False


def _run_pair(w4_path, w4a8_path, samples, real_data_path=None):
    print(f"loading W4   from {w4_path}")
    m4 = ct.models.MLModel(w4_path,
                           compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"loading W4A8 from {w4a8_path}")
    m48 = ct.models.MLModel(w4a8_path,
                            compute_units=ct.ComputeUnit.CPU_AND_NE)

    input_specs = _input_specs_from_model(m4)
    print(f"input specs: {len(input_specs)} inputs")
    for s in input_specs:
        print(f"  {s.name}: {s.shape} {s.dtype}")

    if real_data_path:
        names = {s.name for s in input_specs}
        inputs = _load_real_inputs(real_data_path, names, samples)
        print(f"using {len(inputs)} REAL-prompt samples from {real_data_path}")
    else:
        inputs = _make_inputs(input_specs, samples, seed=0)
        print(f"using {len(inputs)} synthetic N(0, 0.5) samples")

    stats = {"max_abs": [], "cos": [], "rel": []}
    for i, x in enumerate(inputs):
        st4 = m4.make_state() if _is_stateful(m4) else None
        st48 = m48.make_state() if _is_stateful(m48) else None
        y4 = m4.predict(x, state=st4) if st4 else m4.predict(x)
        y48 = m48.predict(x, state=st48) if st48 else m48.predict(x)
        # Compare hidden_states_out
        for k in y4:
            if k not in y48:
                continue
            a = np.asarray(y4[k]).astype(np.float32).flatten()
            b = np.asarray(y48[k]).astype(np.float32).flatten()
            if a.shape != b.shape:
                continue
            mad = float(np.max(np.abs(a - b))) if a.size else 0.0
            denom = max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
            cos = float(np.dot(a, b) / denom)
            rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9))
            if i == 0:
                print(f"  [sample {i}] {k:24s}: "
                      f"max_abs_diff={mad:.4f}  cos={cos:.6f}  rel={rel:.4f}")
            if k == "hidden_states_out":
                stats["max_abs"].append(mad)
                stats["cos"].append(cos)
                stats["rel"].append(rel)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w4", required=True,
                    help="Path to W4 baseline chunk_1.mlpackage")
    ap.add_argument("--w4a8", required=True,
                    help="Path to W4A8 chunk_1.mlpackage")
    ap.add_argument("--samples", type=int, default=32,
                    help="Number of input samples (default: 32). For real "
                         "data, capped by .npz contents.")
    ap.add_argument("--real-data", default=None,
                    help="Path to .npz from gen_calib_data_real.py. When "
                         "set, forwards real-prompt activations through "
                         "both models instead of synthetic random.")
    args = ap.parse_args()

    if not Path(args.w4).exists():
        sys.exit(f"W4 path not found: {args.w4}")
    if not Path(args.w4a8).exists():
        sys.exit(f"W4A8 path not found: {args.w4a8}")

    t0 = time.time()
    stats = _run_pair(args.w4, args.w4a8, args.samples,
                      real_data_path=args.real_data)
    print(f"\ndone in {time.time()-t0:.1f}s\n")

    if stats["max_abs"]:
        ma = np.array(stats["max_abs"])
        cs = np.array(stats["cos"])
        rl = np.array(stats["rel"])
        print(f"hidden_states_out across {len(ma)} samples:")
        print(f"  max_abs_diff:  mean={ma.mean():.4f}  "
              f"max={ma.max():.4f}  min={ma.min():.4f}")
        print(f"  cosine sim:    mean={cs.mean():.6f}  "
              f"min={cs.min():.6f}")
        print(f"  rel L2 error:  mean={rl.mean():.4f}  "
              f"max={rl.max():.4f}")
        # Stage 1 gate per docs/ROADMAP_2026_04_26.md §2.4: cos sim ≥ 0.95
        # is the GO threshold; 0.99+ ideal.
        if cs.mean() >= 0.99:
            verdict = "PASS (cos≥0.99)"
        elif cs.mean() >= 0.95:
            verdict = "PASS-MARGINAL (0.95≤cos<0.99) — re-run iPhone smoke before ship"
        elif cs.mean() >= 0.90:
            verdict = "WARN (0.90≤cos<0.95) — try more calib samples"
        else:
            verdict = "FAIL (cos<0.90) — structural issue, HOLD"
        print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
