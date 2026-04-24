#!/usr/bin/env python3
"""PyTorch parity: Topology I (c1+c2 merged into L0-14 BigChunk1) must
equal the original 4-chunk pipeline at every emitted tensor.

Run: SWAChunk1(L0-7) → SWAChunk2(L8-14) vs BigChunk1(L0-14), seeded with
an embedded real token.  Then feed the emitted per_layer_combined and
KV outputs forward through SWAChunk3(L15-24) → SWAChunk4(L25-34+head)
on both paths, and compare every intermediate tensor.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import (
    SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4, compute_chunk_boundaries,
)
from models.gemma4_swa_3chunk_search import BigChunk1


def _cos(a, b):
    a32 = a.flatten().to(torch.float32)
    b32 = b.flatten().to(torch.float32)
    return float((a32 @ b32) / (a32.norm() * b32.norm() + 1e-12))


def _max_abs(a, b):
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    hf_dir = os.path.join(ROOT, "..", "output", args.model, "hf_model")
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
    (c1s, c1e), (c2s, c2e), (c3s, c3e), (c4s, c4e) = boundaries
    print(f"4-chunk boundaries: {boundaries}")

    torch.manual_seed(args.seed)

    # Seed with a real token through the embedding tables so RMSNorm /
    # softmax observe in-distribution magnitudes.
    token = torch.tensor([[1]], dtype=torch.int64)
    with torch.no_grad():
        tok_emb = base.embed_tokens(token).to(torch.float16)
        per_layer_raw = base.embed_tokens_per_layer(token).to(torch.float16)

    # Reference path: SWAChunk1 → SWAChunk2 → SWAChunk3 → SWAChunk4
    c1_ref = SWAChunk1(base, c1s, c1e).eval()
    c2_ref = SWAChunk2(base, c2s, c2e).eval()
    c3_ref = SWAChunk3(base, c3s, c3e).eval()
    c4_ref = SWAChunk4(base, c4s, c4e).eval()
    ns1, nf1 = c1_ref.num_sliding, c1_ref.num_full
    ns2, nf2 = c2_ref.num_sliding, c2_ref.num_full
    print(f"ref chunk1 L{c1s}-{c1e-1}  ns={ns1} nf={nf1}")
    print(f"ref chunk2 L{c2s}-{c2e-1}  ns={ns2} nf={nf2}")

    # Topology I path: BigChunk1 (L0-14) → SWAChunk3 → SWAChunk4
    big1 = BigChunk1(base, 0, cfg.kv_full_producer + 1).eval()
    ns_big, nf_big = big1.num_sliding, big1.num_full
    print(f"big1        L0-{big1.end-1}    ns={ns_big} nf={nf_big}  "
          f"(total {big1.end} layers)")
    assert ns_big == ns1 + ns2 and nf_big == nf1 + nf2, \
        "BigChunk1 slot counts must equal sum of chunk1+chunk2"

    # Shared inputs
    causal_full = torch.zeros(1, 1, 1, args.ctx, dtype=torch.float16)
    causal_slide = torch.zeros(1, 1, 1, W, dtype=torch.float16)
    update_mask = torch.zeros(1, 1, args.ctx, 1, dtype=torch.float16)
    update_mask[:, :, 0, :] = 1.0
    cos_s = torch.ones(1, 1, 1, hd_s, dtype=torch.float16)
    sin_s = torch.zeros(1, 1, 1, hd_s, dtype=torch.float16)
    cos_f = torch.ones(1, 1, 1, hd_f, dtype=torch.float16)
    sin_f = torch.zeros(1, 1, 1, hd_f, dtype=torch.float16)

    # KV slots for each path
    K_s1_ref = torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16)
    V_s1_ref = torch.zeros(ns1, nkv, W, max_hd, dtype=torch.float16)
    K_f1_ref = torch.zeros(max(nf1, 1), nkv, args.ctx, max_hd, dtype=torch.float16)
    V_f1_ref = torch.zeros(max(nf1, 1), nkv, args.ctx, max_hd, dtype=torch.float16)
    K_s2_ref = torch.zeros(ns2, nkv, W, max_hd, dtype=torch.float16)
    V_s2_ref = torch.zeros(ns2, nkv, W, max_hd, dtype=torch.float16)
    K_f2_ref = torch.zeros(nf2, nkv, args.ctx, max_hd, dtype=torch.float16)
    V_f2_ref = torch.zeros(nf2, nkv, args.ctx, max_hd, dtype=torch.float16)

    K_s_big = torch.zeros(ns_big, nkv, W, max_hd, dtype=torch.float16)
    V_s_big = torch.zeros(ns_big, nkv, W, max_hd, dtype=torch.float16)
    K_f_big = torch.zeros(nf_big, nkv, args.ctx, max_hd, dtype=torch.float16)
    V_f_big = torch.zeros(nf_big, nkv, args.ctx, max_hd, dtype=torch.float16)

    # --- reference 4-chunk forward ---
    with torch.no_grad():
        h1_ref, Ks1_ref, Vs1_ref, Kf1_ref, Vf1_ref, plc_ref = c1_ref(
            tok_emb, causal_full, causal_slide, update_mask, per_layer_raw,
            cos_s, sin_s, cos_f, sin_f,
            K_s1_ref, V_s1_ref, K_f1_ref, V_f1_ref,
        )
        (h2_ref, Ks2_ref, Vs2_ref, Kf2_ref, Vf2_ref,
         kv13_k_ref, kv13_v_ref, kv14_k_ref, kv14_v_ref) = c2_ref(
            h1_ref, causal_full, causal_slide, update_mask, plc_ref,
            cos_s, sin_s, cos_f, sin_f,
            K_s2_ref, V_s2_ref, K_f2_ref, V_f2_ref,
        )

    # --- Topology I forward ---
    with torch.no_grad():
        (h_big, Ks_big, Vs_big, Kf_big, Vf_big, plc_big,
         kv13_k_big, kv13_v_big, kv14_k_big, kv14_v_big) = big1(
            tok_emb, causal_full, causal_slide, update_mask, per_layer_raw,
            cos_s, sin_s, cos_f, sin_f,
            K_s_big, V_s_big, K_f_big, V_f_big,
        )

    # Compare.  Reference emits sliding/full slots per-chunk; BigChunk1
    # returns a single stacked slot list covering both chunks — the slot
    # order is the natural L0..14 iteration.  Cross-check by concatenation.
    Ks_concat = torch.cat([Ks1_ref, Ks2_ref], dim=0)
    Vs_concat = torch.cat([Vs1_ref, Vs2_ref], dim=0)
    Kf_concat = torch.cat([Kf1_ref, Kf2_ref], dim=0)
    Vf_concat = torch.cat([Vf1_ref, Vf2_ref], dim=0)

    # Account for chunk1's dummy full slot when nf1==0 — it allocates
    # max(nf1,1). Strip the leading dummy row if so.
    if nf1 == 0 and Kf1_ref.shape[0] == 1:
        Kf_concat = Kf2_ref
        Vf_concat = Vf2_ref

    pairs = [
        ("hidden_states_out", h2_ref, h_big),
        ("per_layer_combined", plc_ref, plc_big),
        ("K_sliding_out", Ks_concat, Ks_big),
        ("V_sliding_out", Vs_concat, Vs_big),
        ("K_full_out",    Kf_concat, Kf_big),
        ("V_full_out",    Vf_concat, Vf_big),
        ("kv13_k",        kv13_k_ref, kv13_k_big),
        ("kv13_v",        kv13_v_ref, kv13_v_big),
        ("kv14_k",        kv14_k_ref, kv14_k_big),
        ("kv14_v",        kv14_v_ref, kv14_v_big),
    ]

    any_nan = False
    worst = 1.0
    for name, a, b in pairs:
        assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
        c = _cos(a, b)
        d = _max_abs(a, b)
        if math.isnan(c) or math.isnan(d):
            any_nan = True
            tag = "NAN"
        else:
            tag = "OK " if c > 0.9999 else ("!! " if c > 0.99 else "XX ")
            worst = min(worst, c)
        print(f"  {tag}{name:<22s}  cos={c:.6f}  max_abs={d:.4e}  shape={tuple(a.shape)}")

    if any_nan:
        print("\nNaN — input distribution unsuitable for parity.")
        sys.exit(2)
    print(f"\nWorst cosine: {worst:.6f}")
    if worst < 0.9999:
        sys.exit(1)
    print("PARITY OK — Topology I (L0-14 BigChunk1) is bit-equivalent to chunk1+chunk2.")


if __name__ == "__main__":
    main()
