#!/usr/bin/env python3.12
"""End-to-end PLE INT4 quality probe.

Runs the full chunk_1 → chunk_2 → chunk_3 → chunk_4 stateful chain twice
per test token: once with the production INT8 PLE dequant, once with the
candidate INT4-grouped PLE dequant. Measures cosine similarity at:
  - chunk_1's per_layer_combined_out (PLE-direct effect)
  - chunk_4's hidden_states_out (residual stream before lm_head)
  - chunk_4's token_id (does the argmax flip?)

Mac-side only. Loads compiled .mlmodelc files via ct.models.CompiledMLModel.
~30-60s on Mac Studio (depends on per-chunk predict time).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open


# Paths
BUNDLE = Path("/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/build/gemma4_stateful_ab/linear/gemma4_e2b_stateful_chunks")
HF_MODEL = "/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/output/gemma4-e2b/hf_model/model.safetensors"

# Architecture (per build_gemma4_e2b_stateful_chunks.py + LITERT_CONTAINER_ANALYSIS.md)
VOCAB = 262_144
HIDDEN = 1536
NLAYERS = 35
PLD = 256                # per-layer-dim
PLE_TOTAL = NLAYERS * PLD  # 8960
HD_S = 256               # head_dim sliding
HD_F = 512               # head_dim full
CTX = 512                # context the chunks were built for
W = 512                  # sliding window

# Test config
N_RANDOM = 16            # random vocab tokens
N_WORST = 16             # tokens with worst PLE cos under INT4-g32
TOTAL_TESTS = N_RANDOM + N_WORST
GROUP = 32
RNG_SEED = 0xC0FFEE

# Global scales applied by EmbeddingLookup.swift (model_config.json)
EMBED_SCALE = 39.191835884530846       # = sqrt(hidden_size=1536)
PER_LAYER_EMBED_SCALE = 16.0


def load_chunks():
    print("loading chunk_{1..4}.mlmodelc ...")
    chunks = []
    for i in (1, 2, 3, 4):
        t = time.time()
        m = ct.models.CompiledMLModel(
            str(BUNDLE / f"chunk_{i}.mlmodelc"),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        print(f"  chunk_{i}: {time.time()-t:.1f}s")
        chunks.append(m)
    return chunks


def load_int8(name: str, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.fromfile(BUNDLE / f"{name}_q8.bin", dtype=np.int8)
    scales = np.fromfile(BUNDLE / f"{name}_scales.bin", dtype=np.float16)
    n_rows = scales.shape[0]
    return data.reshape(n_rows, n_cols), scales


def load_bf16_ple() -> np.ndarray:
    print("loading BF16 PLE from safetensors ...")
    t = time.time()
    with safe_open(HF_MODEL, framework="pt") as f:
        x = f.get_tensor("model.language_model.embed_tokens_per_layer.weight").to(torch.float32).numpy()
    print(f"  shape={x.shape}, took {time.time()-t:.2f}s")
    return x


def quant_grouped_int4_dequant(x: np.ndarray, group_size: int = GROUP) -> np.ndarray:
    n_rows, n_cols = x.shape
    assert n_cols % group_size == 0
    n_groups = n_cols // group_size
    x_g = x.reshape(n_rows, n_groups, group_size)
    absmax = np.max(np.abs(x_g), axis=2, keepdims=True)
    absmax_safe = np.where(absmax > 0, absmax, 1.0)
    scale = absmax_safe / 7.0
    q = np.round(x_g / scale).clip(-7, 7)
    return (q * scale).reshape(n_rows, n_cols).astype(np.float32)


def cos_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    af = a.astype(np.float64).reshape(a.shape[0], -1)
    bf = b.astype(np.float64).reshape(b.shape[0], -1)
    num = np.einsum("ij,ij->i", af, bf)
    den = np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)


def cos_one(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    da = np.linalg.norm(af)
    db = np.linalg.norm(bf)
    if da == 0 or db == 0:
        return 1.0
    return float(np.dot(af, bf) / (da * db))


def run_chain(c1, c2, c3, c4, hs: np.ndarray, plr: np.ndarray,
              rope: dict, position: int = 0) -> dict:
    cs, ss, cf, sf = rope["cs"], rope["ss"], rope["cf"], rope["sf"]
    mf = np.zeros((1, 1, 1, CTX), np.float16)
    if position + 1 < CTX:
        mf[0, 0, 0, position + 1:] = -1e4
    ms = np.zeros((1, 1, 1, W), np.float16)
    cp = np.array([position], np.int32)
    rp = np.array([position % W], np.int32)

    state1 = c1.make_state()
    state2 = c2.make_state()

    in1 = {
        "hidden_states": hs, "causal_mask_full": mf, "causal_mask_sliding": ms,
        "per_layer_raw": plr,
        "cos_s": cs, "sin_s": ss, "cos_f": cf, "sin_f": sf,
        "current_pos": cp, "ring_pos": rp,
    }
    out1 = c1.predict(in1, state=state1)
    plc = np.asarray(out1["per_layer_combined_out"])

    in2 = {
        "hidden_states": np.asarray(out1["hidden_states_out"]),
        "causal_mask_full": mf, "causal_mask_sliding": ms,
        "per_layer_combined": plc,
        "cos_s": cs, "sin_s": ss, "cos_f": cf, "sin_f": sf,
        "current_pos": cp, "ring_pos": rp,
    }
    out2 = c2.predict(in2, state=state2)
    h2 = np.asarray(out2["hidden_states_out"])
    kv13k = np.asarray(out2["kv13_k"]).astype(np.float16)
    kv13v = np.asarray(out2["kv13_v"]).astype(np.float16)
    kv14k = np.asarray(out2["kv14_k"]).astype(np.float16)
    kv14v = np.asarray(out2["kv14_v"]).astype(np.float16)

    shared = {
        "causal_mask_full": mf, "causal_mask_sliding": ms,
        "per_layer_combined": plc,
        "cos_s": cs, "sin_s": ss, "cos_f": cf, "sin_f": sf,
        "kv13_k": kv13k, "kv13_v": kv13v,
        "kv14_k": kv14k, "kv14_v": kv14v,
    }
    in3 = {**shared, "hidden_states": h2}
    out3 = c3.predict(in3)
    h3 = np.asarray(out3["hidden_states_out"])

    in4 = {**shared, "hidden_states": h3}
    out4 = c4.predict(in4)
    # chunk_4 outputs vary by build; pick whatever's present
    h4_key = next((k for k in ("hidden_states_out", "hidden_states", "final_hidden_states") if k in out4), None)
    h4 = np.asarray(out4[h4_key]) if h4_key else h3  # fall back to h3 if no h4 emitted
    tok_key = next((k for k in ("token_id", "argmax", "next_token_id") if k in out4), None)
    tok = int(np.asarray(out4[tok_key]).flat[0]) if tok_key else -1
    logit_key = next((k for k in ("token_logit", "logit", "logits") if k in out4), None)
    logit = float(np.asarray(out4[logit_key]).flat[0]) if logit_key else 0.0
    return {"plc": plc, "h4": h4, "tok": tok, "logit": logit, "out4_keys": list(out4.keys())}


def main():
    if not BUNDLE.is_dir():
        sys.exit(f"missing bundle: {BUNDLE}")

    rng = np.random.default_rng(RNG_SEED)

    # 1) Load chunks
    c1, c2, c3, c4 = load_chunks()

    # 2) Load embeddings (token + PLE) — both INT8 production
    print("\nloading embeddings ...")
    emb_int8, emb_scales = load_int8("embed_tokens", HIDDEN)
    print(f"  embed_tokens: {emb_int8.shape}, scales {emb_scales.shape}")
    ple_int8, ple_scales = load_int8("embed_tokens_per_layer", PLE_TOTAL)
    print(f"  PLE: {ple_int8.shape}, scales {ple_scales.shape}")

    # 3) Load BF16 PLE + compute INT4-g32 dequantized
    ple_bf16 = load_bf16_ple()
    print(f"computing INT4 group={GROUP} PLE dequant ...")
    t = time.time()
    ple_int4 = quant_grouped_int4_dequant(ple_bf16, GROUP)
    print(f"  done in {time.time()-t:.1f}s")

    # Pick worst-100 rows by cos vs BF16 (these are the test "hard cases")
    cs_g32 = cos_per_row(ple_int4, ple_bf16)
    worst = np.argsort(cs_g32)[:200]  # worst-200 candidates
    print(f"  worst-200 cos sim range: [{cs_g32[worst[0]]:.4f}, {cs_g32[worst[-1]]:.4f}]")

    # 4) Load RoPE tables (.npy)
    print("\nloading RoPE tables ...")
    cos_full_t = np.load(BUNDLE / "cos_full.npy")
    sin_full_t = np.load(BUNDLE / "sin_full.npy")
    cos_sliding_t = np.load(BUNDLE / "cos_sliding.npy")
    sin_sliding_t = np.load(BUNDLE / "sin_sliding.npy")
    print(f"  cos_full {cos_full_t.shape} {cos_full_t.dtype}")
    print(f"  cos_sliding {cos_sliding_t.shape} {cos_sliding_t.dtype}")

    def rope_at(pos: int) -> dict:
        # tables are [ctx, head_dim], extract pos row, reshape to [1,1,1,head_dim]
        return {
            "cs": cos_sliding_t[pos].reshape(1, 1, 1, HD_S).astype(np.float16),
            "ss": sin_sliding_t[pos].reshape(1, 1, 1, HD_S).astype(np.float16),
            "cf": cos_full_t[pos].reshape(1, 1, 1, HD_F).astype(np.float16),
            "sf": sin_full_t[pos].reshape(1, 1, 1, HD_F).astype(np.float16),
        }

    rope_pos0 = rope_at(0)

    # 5) Pick test tokens
    rand_tokens = rng.choice(VOCAB, size=N_RANDOM, replace=False).tolist()
    worst_tokens = worst[:N_WORST].tolist()
    test_tokens = list(map(int, rand_tokens + worst_tokens))
    labels = ["random"] * N_RANDOM + ["worst"] * N_WORST
    print(f"\nrunning {len(test_tokens)} test tokens "
          f"({N_RANDOM} random + {N_WORST} worst-cos)")

    # 6) Helpers for input lookup (matches EmbeddingLookup.swift exactly)
    def lookup_embed(tok: int) -> np.ndarray:
        row = (emb_int8[tok].astype(np.float32)
               * (float(emb_scales[tok]) / 127.0) * EMBED_SCALE)
        return row.astype(np.float16).reshape(1, 1, HIDDEN)

    def lookup_ple_prod(tok: int) -> np.ndarray:
        row = (ple_int8[tok].astype(np.float32)
               * (float(ple_scales[tok]) / 127.0) * PER_LAYER_EMBED_SCALE)
        return row.astype(np.float16).reshape(1, 1, PLE_TOTAL)

    def lookup_ple_int4(tok: int) -> np.ndarray:
        return (ple_int4[tok] * PER_LAYER_EMBED_SCALE).astype(np.float16).reshape(1, 1, PLE_TOTAL)

    # 7) Run chain twice per token, collect stats
    # Probe chunk_4 outputs once before the main loop
    print("\nprobing chunk_4 output keys ...")
    test_tok0 = test_tokens[0]
    _hs0 = lookup_embed(test_tok0)
    _plr0 = lookup_ple_prod(test_tok0)
    _r0 = run_chain(c1, c2, c3, c4, _hs0, _plr0, rope_pos0, position=0)
    print(f"  chunk_4 outputs: {_r0['out4_keys']}")

    print()
    print(f"{'tok':>7s} {'kind':>7s} {'cos(plc)':>9s} {'cos(h4)':>9s} "
          f"{'tok_int8':>9s} {'tok_int4':>9s} {'agree':>5s}")
    cs_plc_all = []
    cs_h4_all = []
    agree_all = []
    t_loop = time.time()
    for tok, label in zip(test_tokens, labels):
        hs = lookup_embed(tok)
        plr_prod = lookup_ple_prod(tok)
        plr_int4 = lookup_ple_int4(tok)
        r_int8 = run_chain(c1, c2, c3, c4, hs, plr_prod, rope_pos0, position=0)
        r_int4 = run_chain(c1, c2, c3, c4, hs, plr_int4, rope_pos0, position=0)

        cs_plc = cos_one(r_int8["plc"], r_int4["plc"])
        cs_h4 = cos_one(r_int8["h4"], r_int4["h4"])
        agree = r_int8["tok"] == r_int4["tok"]
        cs_plc_all.append(cs_plc)
        cs_h4_all.append(cs_h4)
        agree_all.append(int(agree))

        print(f"{tok:7d} {label:>7s} {cs_plc:9.6f} {cs_h4:9.6f} "
              f"{r_int8['tok']:9d} {r_int4['tok']:9d} {'YES' if agree else 'no':>5s}")

    print(f"\nelapsed: {time.time()-t_loop:.1f}s ({(time.time()-t_loop)/len(test_tokens):.1f}s per token pair)")

    # 8) Aggregates
    cs_plc = np.array(cs_plc_all)
    cs_h4 = np.array(cs_h4_all)
    agree = np.array(agree_all)

    print()
    print("=== aggregate (32 tokens, 16 random + 16 worst) ===")
    print(f"  per_layer_combined_out cos (mean / min / max): "
          f"{cs_plc.mean():.6f} / {cs_plc.min():.6f} / {cs_plc.max():.6f}")
    print(f"  hidden_states_out (chunk_4) cos (mean / min / max): "
          f"{cs_h4.mean():.6f} / {cs_h4.min():.6f} / {cs_h4.max():.6f}")
    print(f"  token_id agreement: {agree.sum()}/{len(agree)} "
          f"({100.0 * agree.mean():.1f}%)")

    # Sub-stats by group
    print()
    for grp_name, sl in (("random", slice(0, N_RANDOM)),
                         ("worst", slice(N_RANDOM, TOTAL_TESTS))):
        a = agree[sl]
        p = cs_plc[sl]
        h = cs_h4[sl]
        print(f"  [{grp_name:>6s}] cos(plc) mean={p.mean():.6f} min={p.min():.6f} | "
              f"cos(h4) mean={h.mean():.6f} min={h.min():.6f} | "
              f"tok agree={a.sum()}/{len(a)}")


if __name__ == "__main__":
    main()
