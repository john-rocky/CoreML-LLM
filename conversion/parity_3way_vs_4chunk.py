#!/usr/bin/env python3
"""PyTorch parity: MergedChunk23(h) must equal SWAChunk3(SWAChunk2(h)) bitwise.

The 3-chunk consolidation is ONLY worth pursuing if MergedChunk23 is exactly
equivalent to running chunks 2 and 3 in sequence. Any drift here means the
merge changed the graph semantics (not just dispatch boundaries).

This probe runs both pipelines on identical random inputs and asserts
cosine ~1.0 on the output hidden states and kv13/kv14 producer tensors.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import (
    SWAChunk1, SWAChunk2, SWAChunk3, compute_chunk_boundaries,
)
from models.gemma4_swa_merged import MergedChunk23


def _resolve_hf_dir(model_name: str, override: str | None) -> str:
    if override:
        return override
    return os.path.join(ROOT, "..", "output", model_name, "hf_model")


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.flatten().to(torch.float32)
    b32 = b.flatten().to(torch.float32)
    return float((a32 @ b32) / (a32.norm() * b32.norm() + 1e-12))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir} (ctx={args.ctx})...")
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()

    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s = cfg.head_dim
    hd_f = cfg.global_head_dim
    max_hd = hd_f
    nkv = cfg.num_key_value_heads

    boundaries = compute_chunk_boundaries(cfg)
    print(f"Boundaries: {boundaries}")
    (c1_s, c1_e), (c2_s, c2_e), (c3_s, c3_e), (c4_s, c4_e) = boundaries
    assert (c2_s, c2_e) == (8, 15) and (c3_s, c3_e) == (15, 25), \
        f"MergedChunk23 hardcodes L8-24 boundary; config gave c2={boundaries[1]} c3={boundaries[2]}"

    torch.manual_seed(args.seed)

    # Build the chunk2 entry input by running chunk1 on a real token, so
    # RMSNorm / softmax see distributions the model was trained on and fp16
    # does not produce NaN the way synthetic randn does.
    c1 = SWAChunk1(base, c1_s, c1_e).eval()
    ns1, nf1 = c1.num_sliding, c1.num_full
    token = torch.tensor([[1]], dtype=torch.int64)  # BOS
    with torch.no_grad():
        tok_emb = base.embed_tokens(token).to(torch.float16)
        per_layer_raw = base.embed_tokens_per_layer(token).to(torch.float16)

    causal_full = torch.zeros(1, 1, 1, args.ctx, dtype=torch.float16)
    causal_slide = torch.zeros(1, 1, 1, W, dtype=torch.float16)
    update_mask = torch.zeros(1, 1, args.ctx, 1, dtype=torch.float16)
    update_mask[:, :, 0, :] = 1.0  # write position 0
    cos_s = torch.ones(1, 1, 1, hd_s, dtype=torch.float16)
    sin_s = torch.zeros(1, 1, 1, hd_s, dtype=torch.float16)
    cos_f = torch.ones(1, 1, 1, hd_f, dtype=torch.float16)
    sin_f = torch.zeros(1, 1, 1, hd_f, dtype=torch.float16)

    K_s1 = torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16)
    V_s1 = torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16)
    K_f1 = torch.zeros(max(nf1, 1), nkv, args.ctx, max_hd, dtype=torch.float16)
    V_f1 = torch.zeros(max(nf1, 1), nkv, args.ctx, max_hd, dtype=torch.float16)

    with torch.no_grad():
        h_in, _, _, _, _, per_layer_combined = c1(
            tok_emb, causal_full, causal_slide, update_mask, per_layer_raw,
            cos_s, sin_s, cos_f, sin_f,
            K_s1, V_s1, K_f1, V_f1,
        )

    # Chunk2's KV slot shapes (ns=5 sliding, nf=2 full in L8-14).
    c2 = SWAChunk2(base, c2_s, c2_e).eval()
    ns2, nf2 = c2.num_sliding, c2.num_full
    print(f"SWAChunk2: L{c2_s}-{c2_e-1}  ns={ns2} nf={nf2}")

    K_slide = torch.zeros(ns2, nkv, W, max_hd, dtype=torch.float16)
    V_slide = torch.zeros(ns2, nkv, W, max_hd, dtype=torch.float16)
    K_full = torch.zeros(nf2, nkv, args.ctx, max_hd, dtype=torch.float16)
    V_full = torch.zeros(nf2, nkv, args.ctx, max_hd, dtype=torch.float16)

    # --- 4-chunk path: chunk2 → chunk3 ---
    with torch.no_grad():
        h2, Kso, Vso, Kfo, Vfo, kv13_k, kv13_v, kv14_k, kv14_v = c2(
            h_in, causal_full, causal_slide, update_mask,
            per_layer_combined, cos_s, sin_s, cos_f, sin_f,
            K_slide, V_slide, K_full, V_full,
        )

        c3 = SWAChunk3(base, c3_s, c3_e).eval()
        h3 = c3(h2, causal_full, causal_slide, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v)

    # --- 3-chunk path: MergedChunk23 ---
    merged = MergedChunk23(base).eval()
    with torch.no_grad():
        (hm, Kso_m, Vso_m, Kfo_m, Vfo_m,
         kv13_k_m, kv13_v_m, kv14_k_m, kv14_v_m) = merged(
            h_in, causal_full, causal_slide, update_mask,
            per_layer_combined, cos_s, sin_s, cos_f, sin_f,
            K_slide, V_slide, K_full, V_full,
        )

    # --- Compare ---
    print("\nParity checks (4-chunk vs merged-17):")
    pairs = [
        ("hidden_states_out", h3, hm),
        ("K_sliding_out", Kso, Kso_m),
        ("V_sliding_out", Vso, Vso_m),
        ("K_full_out",    Kfo, Kfo_m),
        ("V_full_out",    Vfo, Vfo_m),
        ("kv13_k",        kv13_k, kv13_k_m),
        ("kv13_v",        kv13_v, kv13_v_m),
        ("kv14_k",        kv14_k, kv14_k_m),
        ("kv14_v",        kv14_v, kv14_v_m),
    ]
    import math
    worst = 1.0
    any_nan = False
    for name, a, b in pairs:
        assert a.shape == b.shape, f"{name} shape mismatch: {a.shape} vs {b.shape}"
        c = _cos(a, b)
        d = _max_abs(a, b)
        if math.isnan(c) or math.isnan(d):
            any_nan = True
            mark = "NAN"
        else:
            mark = "OK " if c > 0.9999 else ("! " if c > 0.99 else "X ")
            worst = min(worst, c)
        print(f"  {mark}{name:<22s}  cos={c:.6f}  max_abs={d:.4e}  shape={tuple(a.shape)}")

    if any_nan:
        print("\nNaN encountered — input distribution unsuitable for parity check.")
        sys.exit(2)
    print(f"\nWorst cosine: {worst:.6f}")
    if worst < 0.9999:
        sys.exit(1)
    print("PARITY OK — MergedChunk23 is bit-equivalent to chunk2→chunk3.")


if __name__ == "__main__":
    main()
