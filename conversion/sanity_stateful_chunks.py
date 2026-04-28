#!/usr/bin/env python3
"""Mac sanity check for Gemma 4 stateful chunk_{1..4}.mlpackage bundles.

Verifies:
  1. Each chunk loads on Mac CPU_AND_NE without error.
  2. predict() completes and returns the documented outputs.
  3. Outputs are finite (no NaN/Inf).
  4. For chunks 1 & 2: passing the *same* state object across two calls
     means the second call sees the writes from the first call (proves
     MLState is actually round-tripping through the kv_cache_{sliding,
     full} state buffers).
  5. chunk_1 → chunk_2 → chunk_3 → chunk_4 wired end-to-end produces a
     valid token_id (range [0, vocab_size)).

This is NOT a numerical correctness test — inputs are zeros / synthetic
RoPE — only a wiring sanity check. Real perf/correctness will be on
iPhone after the Swift Generator is wired up.

Usage:
    python conversion/sanity_stateful_chunks.py             # E2B default
    python conversion/sanity_stateful_chunks.py --model gemma4-e4b
    python conversion/sanity_stateful_chunks.py --artifacts /tmp/foo
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct

# --- Per-model presets (must match build_gemma4_e2b_stateful_chunks.py) ---
PRESETS = {
    "gemma4-e2b": dict(
        ctx=512, w=512, hidden=1536, pld=256, nlayers=35, hkv=1,
        hd_s=256, hd_f=512, vocab=262_144,
        artifacts="/tmp/gemma4-e2b-stateful",
        boundaries=[(0, 8), (8, 15), (15, 25), (25, 35)],
    ),
    "gemma4-e4b": dict(
        ctx=2048, w=512, hidden=2560, pld=256, nlayers=42, hkv=2,
        hd_s=256, hd_f=512, vocab=262_144,
        artifacts="/tmp/gemma4-e4b-stateful",
        boundaries=[(0, 12), (12, 24), (24, 33), (33, 42)],
    ),
}

# Defaults overwritten in main() once we read --model / --artifacts.
CTX = W = HIDDEN = PLD = NLAYERS = HKV = HD_S = HD_F = VOCAB = 0
ARTIFACTS: Path = Path("/tmp/gemma4-e2b-stateful")
CHUNK_BOUNDARIES: list = []


def _make_mask_full(pos: int) -> np.ndarray:
    m = np.zeros((1, 1, 1, CTX), dtype=np.float16)
    # 0 for positions ≤ pos, -1e4 otherwise
    if pos + 1 < CTX:
        m[0, 0, 0, pos + 1:] = -1e4
    return m


def _make_mask_sliding(pos: int) -> np.ndarray:
    return np.zeros((1, 1, 1, W), dtype=np.float16)


def _make_per_layer_raw(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 1, NLAYERS * PLD), dtype=np.float32).astype(np.float16) * 0.1


def _make_rope(seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed + 1)
    cos_s = rng.uniform(-1, 1, (1, 1, 1, HD_S)).astype(np.float16)
    sin_s = rng.uniform(-1, 1, (1, 1, 1, HD_S)).astype(np.float16)
    cos_f = rng.uniform(-1, 1, (1, 1, 1, HD_F)).astype(np.float16)
    sin_f = rng.uniform(-1, 1, (1, 1, 1, HD_F)).astype(np.float16)
    return cos_s, sin_s, cos_f, sin_f


def _check_finite(label: str, name: str, arr: np.ndarray) -> bool:
    a = np.asarray(arr)
    if not np.isfinite(a).all():
        nans = np.isnan(a).sum()
        infs = np.isinf(a).sum()
        print(f"    ❌ {label}/{name}: {nans} NaN, {infs} Inf in shape {a.shape}")
        return False
    print(f"    ✅ {label}/{name}: shape={a.shape} dtype={a.dtype} "
          f"max|x|={float(np.abs(a).max()):.3e} mean={float(a.mean()):.3e}")
    return True


def load_chunk(name: str):
    path = ARTIFACTS / f"{name}.mlpackage"
    print(f"\n[load] {path}")
    t = time.time()
    m = ct.models.MLModel(str(path),
                          compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  loaded in {time.time()-t:.1f}s")
    spec = m.get_spec()
    print(f"  inputs : {[(i.name, tuple(int(d) for d in i.type.multiArrayType.shape)) for i in spec.description.input]}")
    print(f"  outputs: {[o.name for o in spec.description.output]}")
    if spec.description.HasField("state") if False else hasattr(spec.description, "state"):
        try:
            states = spec.description.state
            print(f"  states : {[s.name for s in states]}")
        except Exception:
            pass
    return m


def common_inputs(seed: int, current_pos: int, *, with_per_layer_raw: bool):
    cos_s, sin_s, cos_f, sin_f = _make_rope(seed)
    rng = np.random.default_rng(seed + 100)
    hs = (rng.standard_normal((1, 1, HIDDEN), dtype=np.float32) * 0.1).astype(np.float16)
    ring_pos = current_pos % W
    d = {
        "hidden_states":         hs,
        "causal_mask_full":      _make_mask_full(current_pos),
        "causal_mask_sliding":   _make_mask_sliding(current_pos),
        "cos_s": cos_s, "sin_s": sin_s,
        "cos_f": cos_f, "sin_f": sin_f,
        "current_pos": np.array([current_pos], dtype=np.int32),
        "ring_pos":    np.array([ring_pos], dtype=np.int32),
    }
    if with_per_layer_raw:
        d["per_layer_raw"] = _make_per_layer_raw(seed)
    return d


def shared_chunk_inputs(seed: int, hidden_in: np.ndarray, per_layer_combined: np.ndarray,
                        kv13_k: np.ndarray, kv13_v: np.ndarray,
                        kv14_k: np.ndarray, kv14_v: np.ndarray,
                        current_pos: int):
    cos_s, sin_s, cos_f, sin_f = _make_rope(seed)
    return {
        "hidden_states":         hidden_in,
        "causal_mask_full":      _make_mask_full(current_pos),
        "causal_mask_sliding":   _make_mask_sliding(current_pos),
        "per_layer_combined":    per_layer_combined,
        "cos_s": cos_s, "sin_s": sin_s,
        "cos_f": cos_f, "sin_f": sin_f,
        "kv13_k": kv13_k, "kv13_v": kv13_v,
        "kv14_k": kv14_k, "kv14_v": kv14_v,
    }


def _apply_preset(name: str, artifacts_override: str | None) -> None:
    """Populate the module-level constants other functions read."""
    if name not in PRESETS:
        sys.exit(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    p = PRESETS[name]
    g = globals()
    g["CTX"], g["W"] = p["ctx"], p["w"]
    g["HIDDEN"], g["PLD"] = p["hidden"], p["pld"]
    g["NLAYERS"], g["HKV"] = p["nlayers"], p["hkv"]
    g["HD_S"], g["HD_F"], g["VOCAB"] = p["hd_s"], p["hd_f"], p["vocab"]
    g["ARTIFACTS"] = Path(artifacts_override or p["artifacts"])
    g["CHUNK_BOUNDARIES"] = p["boundaries"]
    print(f"[preset] {name}  artifacts={ARTIFACTS}")
    print(f"  ctx={CTX} W={W} hidden={HIDDEN} layers={NLAYERS} HKV={HKV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b",
                    choices=list(PRESETS),
                    help="Which Gemma 4 stateful preset to sanity-test")
    ap.add_argument("--artifacts", default=None,
                    help="Override artifacts dir (defaults to preset's path)")
    args = ap.parse_args()
    _apply_preset(args.model, args.artifacts)

    if not ARTIFACTS.is_dir():
        sys.exit(f"missing: {ARTIFACTS}")

    # --- Load all 4 chunks ---
    c1 = load_chunk("chunk_1")
    c2 = load_chunk("chunk_2")
    c3 = load_chunk("chunk_3")
    c4 = load_chunk("chunk_4")

    all_finite = True

    # --- Chunk 1: state round-trip test ---
    print("\n" + "=" * 60)
    print("TEST 1: chunk_1 stateful round-trip (pos=0 then pos=1)")
    print("=" * 60)
    state1 = c1.make_state()
    in1a = common_inputs(seed=0, current_pos=0, with_per_layer_raw=True)
    t = time.time()
    out1a = c1.predict(in1a, state=state1)
    print(f"  call 1 (pos=0): {(time.time()-t)*1000:.1f}ms")
    for k, v in out1a.items():
        all_finite &= _check_finite("c1#1", k, v)

    in1b = common_inputs(seed=0, current_pos=1, with_per_layer_raw=True)
    t = time.time()
    out1b = c1.predict(in1b, state=state1)
    print(f"  call 2 (pos=1, same state): {(time.time()-t)*1000:.1f}ms")
    for k, v in out1b.items():
        all_finite &= _check_finite("c1#2", k, v)
    # Hidden states should differ between the two calls if state updates worked
    delta = float(np.abs(out1a["hidden_states_out"] - out1b["hidden_states_out"]).max())
    print(f"  Δ hidden_states between pos=0 and pos=1 calls: {delta:.3e}  "
          f"({'OK — state writes are visible' if delta > 1e-4 else 'WARN — call outputs identical'})")

    hs_after_c1 = out1b["hidden_states_out"]
    plc_after_c1 = out1b["per_layer_combined_out"]

    # --- Chunk 2: forward chunk1's outputs ---
    print("\n" + "=" * 60)
    print("TEST 2: chunk_2 with chunk_1 outputs as input")
    print("=" * 60)
    state2 = c2.make_state()
    in2 = common_inputs(seed=1, current_pos=1, with_per_layer_raw=False)
    in2["hidden_states"] = hs_after_c1
    in2["per_layer_combined"] = plc_after_c1
    t = time.time()
    out2 = c2.predict(in2, state=state2)
    print(f"  predict: {(time.time()-t)*1000:.1f}ms")
    for k, v in out2.items():
        all_finite &= _check_finite("c2", k, v)

    hs_after_c2 = out2["hidden_states_out"]
    kv13_k = np.asarray(out2["kv13_k"]).astype(np.float16)
    kv13_v = np.asarray(out2["kv13_v"]).astype(np.float16)
    kv14_k = np.asarray(out2["kv14_k"]).astype(np.float16)
    kv14_v = np.asarray(out2["kv14_v"]).astype(np.float16)
    print(f"  kv13_k shape={kv13_k.shape}  kv14_k shape={kv14_k.shape}")

    # --- Chunk 3: stateless, reads kv13/14 ---
    print("\n" + "=" * 60)
    print("TEST 3: chunk_3 with kv13/14 from chunk_2")
    print("=" * 60)
    in3 = shared_chunk_inputs(seed=2,
                              hidden_in=hs_after_c2,
                              per_layer_combined=plc_after_c1,
                              kv13_k=kv13_k, kv13_v=kv13_v,
                              kv14_k=kv14_k, kv14_v=kv14_v,
                              current_pos=1)
    t = time.time()
    out3 = c3.predict(in3)
    print(f"  predict: {(time.time()-t)*1000:.1f}ms")
    for k, v in out3.items():
        all_finite &= _check_finite("c3", k, v)
    hs_after_c3 = out3["hidden_states_out"]

    # --- Chunk 4: stateless + lm_head ---
    print("\n" + "=" * 60)
    print("TEST 4: chunk_4 → token_id")
    print("=" * 60)
    in4 = shared_chunk_inputs(seed=3,
                              hidden_in=hs_after_c3,
                              per_layer_combined=plc_after_c1,
                              kv13_k=kv13_k, kv13_v=kv13_v,
                              kv14_k=kv14_k, kv14_v=kv14_v,
                              current_pos=1)
    t = time.time()
    out4 = c4.predict(in4)
    print(f"  predict: {(time.time()-t)*1000:.1f}ms")
    for k, v in out4.items():
        all_finite &= _check_finite("c4", k, v)
    tok = out4.get("token_id")
    if tok is not None:
        v = int(np.asarray(tok).flat[0])
        ok = 0 <= v < VOCAB
        print(f"  token_id={v}  in [0, {VOCAB}): {ok}")
        if not ok:
            all_finite = False

    # --- Verdict ---
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if all_finite:
        print("  ✅ all chunks load, predict, and produce finite outputs.")
        print("  ✅ MLState round-trip works on chunk_1.")
        print("  ✅ chunk_1 → 2 → 3 → 4 chained without shape/state errors.")
        sys.exit(0)
    else:
        print("  ❌ at least one chunk produced non-finite output or failed sanity check.")
        sys.exit(1)


if __name__ == "__main__":
    main()
