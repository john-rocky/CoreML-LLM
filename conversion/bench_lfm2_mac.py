"""Mac decode benchmark for the converted LFM2.5 mlpackage.

Loads ``output/lfm2.5-350m/model.mlpackage``, primes the stateful KV + conv
caches with a short prompt, then measures sustained 1-token/step decode
throughput on:
  - ALL  (lets CoreML schedule across CPU/GPU/ANE)
  - CPU+ANE  (forces ANE residency, falls back to CPU only for non-ANE ops)
  - CPU only (sanity baseline)

Reports average ms/token and tok/s.  Argmax is computed inside the graph,
so we only round-trip 2 scalars per step.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

import coremltools as ct
from coremltools.models.compute_device import (
    MLCPUComputeDevice,
    MLNeuralEngineComputeDevice,
)


def _load(pkg: str, compute_units: ct.ComputeUnit) -> ct.models.MLModel:
    print(f"Loading {pkg} with compute_units={compute_units}...")
    t0 = time.time()
    m = ct.models.MLModel(pkg, compute_units=compute_units)
    print(f"  load: {time.time() - t0:.2f}s")
    return m


def _make_inputs(pos: int, ctx: int, token_id: int, conv_state: np.ndarray | None = None) -> dict:
    """Build a 1-step decode dict for the given position."""
    causal = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float16)  # -inf in fp16
    causal[..., : pos + 1] = 0.0
    update = np.zeros((1, 1, ctx, 1), dtype=np.float16)
    update[0, 0, pos, 0] = 1.0
    feed = {
        "input_ids": np.asarray([[token_id]], dtype=np.int32),
        "position_ids": np.asarray([pos], dtype=np.int32),
        "causal_mask": causal,
        "update_mask": update,
    }
    if conv_state is not None:
        feed["conv_state_in"] = conv_state
    return feed


def bench(pkg: str, compute_units: ct.ComputeUnit, label: str,
          prompt_ids: list[int], n_decode: int, ctx: int,
          conv_state_shape: tuple | None = None) -> dict:
    print(f"\n=== {label} ===")
    m = _load(pkg, compute_units)

    # Make a brand-new state and zero it (the converter starts states at 0,
    # but if we re-use the model we need an explicit reset for repeatability).
    state = m.make_state()

    # LFM2 keeps the conv state outside MLState — track it in NumPy.
    conv_state = (
        np.zeros(conv_state_shape, dtype=np.float16)
        if conv_state_shape is not None else None
    )

    # --- prefill the prompt (sequential 1-token decode) ---
    print(f"  prefill {len(prompt_ids)} tokens...")
    t0 = time.time()
    last_token_id = None
    for i, tid in enumerate(prompt_ids):
        feed = _make_inputs(i, ctx, tid, conv_state)
        out = m.predict(feed, state=state)
        last_token_id = int(out["token_id"][0])
        if conv_state is not None:
            conv_state = out["conv_state_out"]
    prefill_s = time.time() - t0
    prefill_tps = len(prompt_ids) / prefill_s
    print(f"  prefill: {prefill_s*1000:.1f} ms total → {prefill_tps:.1f} tok/s "
          f"({prefill_s*1000/len(prompt_ids):.2f} ms/tok)")

    # --- warmup decode for 4 steps to settle ANE plan ---
    pos = len(prompt_ids)
    cur = last_token_id
    for _ in range(4):
        feed = _make_inputs(pos, ctx, cur, conv_state)
        out = m.predict(feed, state=state)
        cur = int(out["token_id"][0])
        if conv_state is not None:
            conv_state = out["conv_state_out"]
        pos += 1

    # --- timed decode ---
    print(f"  measured decode for {n_decode} steps...")
    t0 = time.time()
    times = []
    for _ in range(n_decode):
        feed = _make_inputs(pos, ctx, cur, conv_state)
        ts = time.time()
        out = m.predict(feed, state=state)
        times.append(time.time() - ts)
        cur = int(out["token_id"][0])
        if conv_state is not None:
            conv_state = out["conv_state_out"]
        pos += 1
    total_s = time.time() - t0
    avg_ms = sum(times) / len(times) * 1000
    p50 = sorted(times)[len(times) // 2] * 1000
    p95 = sorted(times)[int(len(times) * 0.95)] * 1000
    decode_tps = n_decode / total_s
    print(f"  decode: {avg_ms:.2f} ms/tok avg (p50={p50:.2f}, p95={p95:.2f}) "
          f"→ {decode_tps:.1f} tok/s (over {n_decode} steps)")
    return {
        "label": label,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "avg_ms": avg_ms,
        "p50_ms": p50,
        "p95_ms": p95,
    }


def main() -> None:
    # Default to the no-quant build (quant build also exists at lfm2.5-350m/).
    pkg = os.environ.get("LFM2_BENCH_PKG", "./output/lfm2.5-350m-fp16/model.mlpackage")
    if not os.path.exists(pkg):
        print(f"Missing {pkg} — run convert.py first")
        sys.exit(1)

    ctx = 2048
    # Prime with a short non-trivial prompt.  We don't need the tokenizer
    # for benchmarking — just use the BOS sequence + a few common ids.
    # 1=<|startoftext|>, then arbitrary mid-vocab tokens.
    prompt_ids = [1, 1098, 4605, 10800, 36387, 56586, 730, 5488, 4639, 1893]
    n_decode = 64

    # Quick sanity: list available compute devices.
    print("Compute devices on this Mac:")
    print("  CPU:", any(isinstance(d, MLCPUComputeDevice) for d in
                        ct.models.compute_device.MLComputeDevice.get_all_compute_devices()))
    print("  ANE:", any(isinstance(d, MLNeuralEngineComputeDevice) for d in
                        ct.models.compute_device.MLComputeDevice.get_all_compute_devices()))

    # Conv state shape is read from the model spec — no need to hardcode.
    spec = ct.utils.load_spec(pkg)
    conv_in_shape = None
    for inp in spec.description.input:
        if inp.name == "conv_state_in":
            sd = inp.type.multiArrayType
            conv_in_shape = tuple(sd.shape)
            break

    results = []
    for cu, label in [
        (ct.ComputeUnit.CPU_AND_NE,  "CPU+ANE"),
        (ct.ComputeUnit.ALL,         "ALL  (CPU+GPU+ANE)"),
        (ct.ComputeUnit.CPU_AND_GPU, "CPU+GPU"),
        (ct.ComputeUnit.CPU_ONLY,    "CPU only"),
    ]:
        try:
            results.append(bench(pkg, cu, label, prompt_ids, n_decode, ctx, conv_in_shape))
        except Exception as e:
            msg = str(e).split("\n")[0][:120]
            print(f"\n=== {label} === FAILED: {msg}")

    print("\n=== Summary ===")
    print(f"{'config':<28s}  {'prefill tok/s':>14s}  {'decode tok/s':>14s}  {'avg ms/tok':>11s}")
    for r in results:
        print(
            f"{r['label']:<28s}  {r['prefill_tps']:>14.1f}  "
            f"{r['decode_tps']:>14.1f}  {r['avg_ms']:>11.2f}"
        )


if __name__ == "__main__":
    main()
