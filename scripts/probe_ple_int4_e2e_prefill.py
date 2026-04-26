#!/usr/bin/env python3.12
"""End-to-end PLE INT4 quality probe — REAL PREFILL version.

Companion to `probe_ple_int4_e2e.py`. The fresh-state pos=0 test there
showed 44% token disagreement on INT4 g=32 vs INT8 PLE, but pos=0 fresh-
state is the worst-case for measuring quality (no context to attenuate
input perturbations). This script runs multi-step prefill on real prompt
sequences and measures cos sim and token agreement at each position.

For each test prompt:
  1. Run T=1 prefill of N tokens with INT8 PLE.
  2. Run same prefill with INT4 g=32 PLE (token embedding identical).
  3. At each position, record:
     - cos(plc): per_layer_combined_out from chunk_1
     - cos(h4):  hidden_states proxy from chunk_4 (last hidden before lm_head)
     - argmax agreement: did chunk_4 pick the same next token?

Token sequences used: short common prefixes spelled by direct vocab lookup.
We do NOT need a tokenizer — using literal token IDs (chosen to be common
small IDs that the model is trained on heavily).

Mac-side only. ~30s on Mac Studio after first run (cached).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open


BUNDLE = Path("/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/build/gemma4_stateful_ab/linear/gemma4_e2b_stateful_chunks")
HF_MODEL = "/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/output/gemma4-e2b/hf_model/model.safetensors"

VOCAB = 262_144
HIDDEN = 1536
NLAYERS = 35
PLD = 256
PLE_TOTAL = NLAYERS * PLD
HD_S = 256
HD_F = 512
CTX = 512
W = 512

EMBED_SCALE = 39.191835884530846
PER_LAYER_EMBED_SCALE = 16.0
GROUP = 32

# Test prompts as raw token IDs. Pick common-vocab IDs (1-1000 range —
# heavily trained, no special tokens). 4 prompts × 16 tokens prefill.
PROMPT_LEN = 16
PROMPTS_RAW = [
    list(range(2,   2 + PROMPT_LEN)),       # BOS + sequential common tokens
    list(range(40,  40 + PROMPT_LEN)),
    list(range(100, 100 + PROMPT_LEN)),
    list(range(200, 200 + PROMPT_LEN)),
]


def load_chunks():
    print("loading chunks ...")
    return [
        ct.models.CompiledMLModel(str(BUNDLE / f"chunk_{i}.mlmodelc"),
                                  compute_units=ct.ComputeUnit.CPU_AND_NE)
        for i in (1, 2, 3, 4)
    ]


def load_int8(name: str, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.fromfile(BUNDLE / f"{name}_q8.bin", dtype=np.int8)
    scales = np.fromfile(BUNDLE / f"{name}_scales.bin", dtype=np.float16)
    n_rows = scales.shape[0]
    return data.reshape(n_rows, n_cols), scales


def load_bf16_ple() -> np.ndarray:
    with safe_open(HF_MODEL, framework="pt") as f:
        x = f.get_tensor("model.language_model.embed_tokens_per_layer.weight").to(torch.float32).numpy()
    return x


def quant_grouped_int4_dequant(x: np.ndarray, group_size: int) -> np.ndarray:
    n_rows, n_cols = x.shape
    n_groups = n_cols // group_size
    x_g = x.reshape(n_rows, n_groups, group_size)
    absmax = np.max(np.abs(x_g), axis=2, keepdims=True)
    absmax_safe = np.where(absmax > 0, absmax, 1.0)
    scale = absmax_safe / 7.0
    q = np.round(x_g / scale).clip(-7, 7)
    return (q * scale).reshape(n_rows, n_cols).astype(np.float32)


def cos_one(a, b) -> float:
    af = np.asarray(a, np.float64).flatten()
    bf = np.asarray(b, np.float64).flatten()
    da = np.linalg.norm(af); db = np.linalg.norm(bf)
    if da == 0 or db == 0: return 1.0
    return float(np.dot(af, bf) / (da * db))


def make_inputs(hs, plr, position, rope, *, with_per_layer_raw):
    cs, ss, cf, sf = rope[position]
    mf = np.zeros((1, 1, 1, CTX), np.float16)
    if position + 1 < CTX:
        mf[0, 0, 0, position + 1:] = -1e4
    ms = np.zeros((1, 1, 1, W), np.float16)
    cp = np.array([position], np.int32)
    rp = np.array([position % W], np.int32)
    inp = {
        "hidden_states": hs,
        "causal_mask_full": mf, "causal_mask_sliding": ms,
        "cos_s": cs, "sin_s": ss, "cos_f": cf, "sin_f": sf,
        "current_pos": cp, "ring_pos": rp,
    }
    if with_per_layer_raw:
        inp["per_layer_raw"] = plr
    return inp


def step(c1, c2, c3, c4, hs, plr, position, rope, state1, state2):
    in1 = make_inputs(hs, plr, position, rope, with_per_layer_raw=True)
    out1 = c1.predict(in1, state=state1)
    plc = np.asarray(out1["per_layer_combined_out"])
    in2 = make_inputs(np.asarray(out1["hidden_states_out"]), None, position, rope,
                      with_per_layer_raw=False)
    in2["per_layer_combined"] = plc
    out2 = c2.predict(in2, state=state2)
    h2 = np.asarray(out2["hidden_states_out"])
    kv13_k = np.asarray(out2["kv13_k"]).astype(np.float16)
    kv13_v = np.asarray(out2["kv13_v"]).astype(np.float16)
    kv14_k = np.asarray(out2["kv14_k"]).astype(np.float16)
    kv14_v = np.asarray(out2["kv14_v"]).astype(np.float16)
    in3 = make_inputs(h2, None, position, rope, with_per_layer_raw=False)
    in3.update({"per_layer_combined": plc,
                "kv13_k": kv13_k, "kv13_v": kv13_v,
                "kv14_k": kv14_k, "kv14_v": kv14_v})
    out3 = c3.predict(in3)
    h3 = np.asarray(out3["hidden_states_out"])
    in4 = make_inputs(h3, None, position, rope, with_per_layer_raw=False)
    in4.update({"per_layer_combined": plc,
                "kv13_k": kv13_k, "kv13_v": kv13_v,
                "kv14_k": kv14_k, "kv14_v": kv14_v})
    out4 = c4.predict(in4)
    h4_key = next((k for k in ("hidden_states_out", "hidden_normed", "hidden_states") if k in out4), None)
    h4 = np.asarray(out4[h4_key])
    tok_key = next((k for k in ("token_id", "argmax", "next_token_id") if k in out4), None)
    tok = int(np.asarray(out4[tok_key]).flat[0])
    return plc, h4, tok


def run_prefill(c1, c2, c3, c4, prompt, ple_lookup, embed_lookup, rope):
    state1 = c1.make_state()
    state2 = c2.make_state()
    plcs, h4s, toks = [], [], []
    for pos, tok_id in enumerate(prompt):
        hs = embed_lookup(tok_id)
        plr = ple_lookup(tok_id)
        plc, h4, tok = step(c1, c2, c3, c4, hs, plr, pos, rope, state1, state2)
        plcs.append(plc); h4s.append(h4); toks.append(tok)
    return plcs, h4s, toks


def main():
    if not BUNDLE.is_dir():
        sys.exit(f"missing bundle: {BUNDLE}")

    c1, c2, c3, c4 = load_chunks()

    print("\nloading embeddings + RoPE ...")
    emb_int8, emb_scales = load_int8("embed_tokens", HIDDEN)
    ple_int8, ple_scales = load_int8("embed_tokens_per_layer", PLE_TOTAL)
    ple_bf16 = load_bf16_ple()
    ple_int4 = quant_grouped_int4_dequant(ple_bf16, GROUP)
    print(f"  computed INT4 g={GROUP}")
    cos_full_t = np.load(BUNDLE / "cos_full.npy")
    sin_full_t = np.load(BUNDLE / "sin_full.npy")
    cos_sliding_t = np.load(BUNDLE / "cos_sliding.npy")
    sin_sliding_t = np.load(BUNDLE / "sin_sliding.npy")

    rope = []
    for pos in range(PROMPT_LEN):
        cs = cos_sliding_t[pos].reshape(1, 1, 1, HD_S).astype(np.float16)
        ss = sin_sliding_t[pos].reshape(1, 1, 1, HD_S).astype(np.float16)
        cf = cos_full_t[pos].reshape(1, 1, 1, HD_F).astype(np.float16)
        sf = sin_full_t[pos].reshape(1, 1, 1, HD_F).astype(np.float16)
        rope.append((cs, ss, cf, sf))

    def lookup_embed(tok):
        return ((emb_int8[tok].astype(np.float32) * (float(emb_scales[tok]) / 127.0) * EMBED_SCALE)
                .astype(np.float16).reshape(1, 1, HIDDEN))

    def lookup_ple_int8(tok):
        return ((ple_int8[tok].astype(np.float32) * (float(ple_scales[tok]) / 127.0) * PER_LAYER_EMBED_SCALE)
                .astype(np.float16).reshape(1, 1, PLE_TOTAL))

    def lookup_ple_int4(tok):
        return ((ple_int4[tok] * PER_LAYER_EMBED_SCALE)
                .astype(np.float16).reshape(1, 1, PLE_TOTAL))

    # Optionally test multiple group sizes in one run
    group_sizes_env = (sys.argv[1] if len(sys.argv) > 1 else "32").split(",")
    group_sizes = [int(g) for g in group_sizes_env]

    print()
    all_results_by_group = {}
    ple_int4_by_group = {}
    for g in group_sizes:
        if g == GROUP and 'ple_int4' in dir() and ple_int4 is not None:
            ple_int4_by_group[g] = ple_int4
        else:
            print(f"  computing INT4 g={g} ...")
            ple_int4_by_group[g] = quant_grouped_int4_dequant(ple_bf16, g)

    for g in group_sizes:
        ple_g = ple_int4_by_group[g]
        def _lookup_int4_g(tok, _ple=ple_g):
            return ((_ple[tok] * PER_LAYER_EMBED_SCALE)
                    .astype(np.float16).reshape(1, 1, PLE_TOTAL))
        all_results = []
        print(f"\n========== group_size = {g} ==========")
        for pi, prompt in enumerate(PROMPTS_RAW):
            t = time.time()
            plcs_int8, h4s_int8, toks_int8 = run_prefill(c1, c2, c3, c4, prompt, lookup_ple_int8, lookup_embed, rope)
            plcs_int4, h4s_int4, toks_int4 = run_prefill(c1, c2, c3, c4, prompt, _lookup_int4_g, lookup_embed, rope)
            dt = time.time() - t
            agrees = []
            for j in range(PROMPT_LEN):
                cp = cos_one(plcs_int8[j], plcs_int4[j])
                ch = cos_one(h4s_int8[j], h4s_int4[j])
                ag = toks_int8[j] == toks_int4[j]
                agrees.append(int(ag))
                all_results.append((pi, j, cp, ch, ag))
            print(f"  prompt {pi}: agree {sum(agrees)}/{len(agrees)}  ({dt:.1f}s)")

        # Overall for this group size
        cps_all = np.array([r[2] for r in all_results])
        chs_all = np.array([r[3] for r in all_results])
        ags_all = np.array([r[4] for r in all_results])
        print(f"  OVERALL g={g}: cos(plc) {cps_all.mean():.6f} | "
              f"cos(h4) {chs_all.mean():.6f} (min {chs_all.min():.6f}) | "
              f"tok agree {ags_all.sum()}/{len(ags_all)} ({100*ags_all.mean():.1f}%)")
        all_results_by_group[g] = (cps_all, chs_all, ags_all)

    # Comparison across group sizes
    print()
    print("=== summary (lower group size = finer scaling = more storage) ===")
    print(f"  {'group':>5s} | {'mean cos(plc)':>13s} | {'mean cos(h4)':>13s} | {'min cos(h4)':>11s} | {'tok agree':>10s}")
    for g in group_sizes:
        cps, chs, ags = all_results_by_group[g]
        print(f"  {g:>5d} | {cps.mean():13.6f} | {chs.mean():13.6f} | {chs.min():11.6f} | "
              f"{ags.sum():>4d}/{len(ags):>4d} ({100*ags.mean():.1f}%)")


if __name__ == "__main__":
    main()
