"""11c verify-vs-serial parity test.

Compares the new SWAVerifyChunkN (K=3 batched) against 3 sequential T=1
SWAChunkN steps starting from the same KV state. They should produce
identical per-position hidden states and argmax token IDs.

If the new verify protocol is implemented correctly, cosine ≥ 0.999 at
each position and identical argmax. Divergence indicates a bug in the
modified verify path (KV bookkeeping, mask, RoPE, kv13/kv14 sharing).

Usage:
    PYENV_VERSION=lama-cml python conversion/test_verify_vs_serial.py \
        --hf-dir /Users/majimadaisuke/Downloads/CoreML-LLM/conversion/output/gemma4-e2b-final/hf_model
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ane_ops import MODEL_DTYPE
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import (
    SWAChunk1, SWAChunk2, SWAChunk3, SWAChunk4,
    SWAVerifyChunk1, SWAVerifyChunk2, SWAVerifyChunk3, SWAVerifyChunk4,
)


def make_serial_masks(pos: int, ctx: int, W: int):
    """Masks for a T=1 decode step at absolute position `pos`."""
    # Full attn: zeros at [0..pos], -inf at [pos+1..ctx-1]
    m_full = torch.full((1, 1, 1, ctx), float("-inf"), dtype=MODEL_DTYPE)
    m_full[..., : pos + 1] = 0.0
    # Sliding: cache stores last W positions [max(0,pos-W+1) .. pos] mapped to slots [W-(pos+1)..W-1]
    valid = min(pos + 1, W)
    m_sliding = torch.full((1, 1, 1, W), float("-inf"), dtype=MODEL_DTYPE)
    m_sliding[..., W - valid :] = 0.0
    # update_mask: (1,1,ctx,1) one-hot at pos
    update_mask = torch.zeros(1, 1, ctx, 1, dtype=MODEL_DTYPE)
    update_mask[0, 0, pos, 0] = 1.0
    return m_full, m_sliding, update_mask


def make_verify_masks(start_pos: int, K: int, ctx: int, W: int):
    """Masks for a verify call at positions [start_pos, start_pos+K-1]."""
    # Full attn: row t allows attend to [0..start_pos+t]
    m_full = torch.full((1, 1, K, ctx), float("-inf"), dtype=MODEL_DTYPE)
    for t in range(K):
        m_full[0, 0, t, : start_pos + t + 1] = 0.0
    # Sliding: extended W-window holds [start_pos+K-W .. start_pos+K-1] in slots [0..W-1].
    # Row t (representing pos start_pos+t) attends slots [0..W-K+t]; rest = -inf.
    # Also clamp left to skip slots representing positions < 0.
    m_sliding = torch.full((1, 1, K, W), float("-inf"), dtype=MODEL_DTYPE)
    for t in range(K):
        right = W - K + t
        # leftmost valid slot represents pos = start_pos + K - W. For early positions
        # this could go negative — those slots should stay masked.
        leftmost_pos = start_pos + K - W
        first_valid_slot = max(0, -leftmost_pos)
        m_sliding[0, 0, t, first_valid_slot : right + 1] = 0.0
    # update_indicator: (1,1,ctx,K), col t one-hot at start_pos+t
    update_indicator = torch.zeros(1, 1, ctx, K, dtype=MODEL_DTYPE)
    for t in range(K):
        update_indicator[0, 0, start_pos + t, t] = 1.0
    return m_full, m_sliding, update_indicator


def slice_rope(model, pos: int, K: int):
    """Returns (cos_s, sin_s, cos_f, sin_f) for K consecutive positions starting at pos."""
    cos_s = model.cos_sliding[pos : pos + K].view(1, 1, K, -1).to(MODEL_DTYPE)
    sin_s = model.sin_sliding[pos : pos + K].view(1, 1, K, -1).to(MODEL_DTYPE)
    cos_f = model.cos_full[pos : pos + K].view(1, 1, K, -1).to(MODEL_DTYPE)
    sin_f = model.sin_full[pos : pos + K].view(1, 1, K, -1).to(MODEL_DTYPE)
    return cos_s, sin_s, cos_f, sin_f


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--start-pos", type=int, default=0,
                    help="Absolute position where verify begins (and serial step 0 happens)")
    args = ap.parse_args()

    K = args.K
    CTX = args.ctx
    W = 512

    print(f"Loading Gemma 4 from {args.hf_dir}...")
    t0 = time.time()
    model = Gemma4Model.from_pretrained(args.hf_dir, context_length=CTX).eval()
    print(f"  loaded in {time.time()-t0:.1f}s; dtype={MODEL_DTYPE}")

    cfg = model.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    max_hd = cfg.global_head_dim
    assert nlayers == 35

    # Pick K input token IDs deterministically.
    torch.manual_seed(0)
    token_ids = torch.tensor([100, 200, 300][:K], dtype=torch.long)
    print(f"Test token IDs at positions [{args.start_pos}..{args.start_pos+K-1}]: {token_ids.tolist()}")

    # Embed the K tokens (hidden + per-layer raw)
    with torch.no_grad():
        embeds = model.embed_tokens(token_ids).to(MODEL_DTYPE)  # (K, hidden)
        per_layer_embeds = (model.embed_tokens_per_layer(token_ids).to(MODEL_DTYPE)
                            * model.per_layer_embed_scale)  # (K, nlayers*pld)
    embeds_K = embeds.unsqueeze(0)  # (1, K, hidden)
    per_layer_K = per_layer_embeds.unsqueeze(0)  # (1, K, nlayers*pld)

    # ---- Build chunked decode + verify modules (share weights with `model`) ----
    sw1 = SWAChunk1(model).eval()
    sw2 = SWAChunk2(model).eval()
    sw3 = SWAChunk3(model).eval()
    sw4 = SWAChunk4(model).eval()
    vc1 = SWAVerifyChunk1(model, seq_len=K).eval()
    vc2 = SWAVerifyChunk2(model, seq_len=K).eval()
    vc3 = SWAVerifyChunk3(model, seq_len=K).eval()
    vc4 = SWAVerifyChunk4(model, seq_len=K).eval()

    # ---- Path A: 3× serial T=1 decode ----
    print("\n=== Path A: serial T=1 decode ===")
    kSliding1 = torch.zeros(7, 1, W, max_hd, dtype=MODEL_DTYPE)
    vSliding1 = torch.zeros(7, 1, W, max_hd, dtype=MODEL_DTYPE)
    kFull1    = torch.zeros(1, 1, CTX, max_hd, dtype=MODEL_DTYPE)
    vFull1    = torch.zeros(1, 1, CTX, max_hd, dtype=MODEL_DTYPE)
    kSliding2 = torch.zeros(5, 1, W, max_hd, dtype=MODEL_DTYPE)
    vSliding2 = torch.zeros(5, 1, W, max_hd, dtype=MODEL_DTYPE)
    kFull2    = torch.zeros(2, 1, CTX, max_hd, dtype=MODEL_DTYPE)
    vFull2    = torch.zeros(2, 1, CTX, max_hd, dtype=MODEL_DTYPE)

    serial_argmax = []
    serial_h_after_c1 = []
    serial_h_after_c2 = []
    serial_h_after_c3 = []
    serial_normed = []

    for t in range(K):
        pos = args.start_pos + t
        m_full, m_sliding, update_mask = make_serial_masks(pos, CTX, W)
        cs, ss, cf, sf = slice_rope(model, pos, 1)
        h = embeds_K[:, t : t + 1, :].clone()  # (1,1,hidden)
        plr = per_layer_K[:, t : t + 1, :].clone()

        with torch.no_grad():
            h, kSliding1, vSliding1, kFull1, vFull1, plc = sw1(
                h, m_full, m_sliding, update_mask, plr,
                cs, ss, cf, sf,
                kSliding1, vSliding1, kFull1, vFull1,
            )
            serial_h_after_c1.append(h.detach().clone())
            h, kSliding2, vSliding2, kFull2, vFull2, k13, v13, k14, v14 = sw2(
                h, m_full, m_sliding, update_mask, plc,
                cs, ss, cf, sf,
                kSliding2, vSliding2, kFull2, vFull2,
            )
            serial_h_after_c2.append(h.detach().clone())
            h = sw3(h, m_full, m_sliding, update_mask, plc,
                    cs, ss, cf, sf, k13, v13, k14, v14)
            serial_h_after_c3.append(h.detach().clone())
            tok_id, tok_logit, normed = sw4(
                h, m_full, m_sliding, update_mask, plc,
                cs, ss, cf, sf, k13, v13, k14, v14,
            )
        serial_argmax.append(int(tok_id.item()))
        serial_normed.append(normed.detach().clone())
        print(f"  t={t} pos={pos} -> argmax={int(tok_id.item())}")

    # ---- Path B: verify K=3 ----
    print("\n=== Path B: verify K=3 (single batched call) ===")
    # Reset KV state to zeros (verify chunks read pre-verify cache)
    kSliding1_v = torch.zeros(7, 1, W, max_hd, dtype=MODEL_DTYPE)
    vSliding1_v = torch.zeros(7, 1, W, max_hd, dtype=MODEL_DTYPE)
    kFull1_v    = torch.zeros(1, 1, CTX, max_hd, dtype=MODEL_DTYPE)
    vFull1_v    = torch.zeros(1, 1, CTX, max_hd, dtype=MODEL_DTYPE)
    kSliding2_v = torch.zeros(5, 1, W, max_hd, dtype=MODEL_DTYPE)
    vSliding2_v = torch.zeros(5, 1, W, max_hd, dtype=MODEL_DTYPE)
    kFull2_v    = torch.zeros(2, 1, CTX, max_hd, dtype=MODEL_DTYPE)
    vFull2_v    = torch.zeros(2, 1, CTX, max_hd, dtype=MODEL_DTYPE)

    m_full_v, m_sliding_v, update_indicator = make_verify_masks(args.start_pos, K, CTX, W)
    cs_v, ss_v, cf_v, sf_v = slice_rope(model, args.start_pos, K)

    with torch.no_grad():
        h_v, plc_v, nKs1, nVs1, nKf1, nVf1 = vc1(
            embeds_K, m_full_v, m_sliding_v, update_indicator, per_layer_K,
            cs_v, ss_v, cf_v, sf_v,
            kSliding1_v, vSliding1_v, kFull1_v, vFull1_v,
        )
        h_after_c1_v = h_v.detach().clone()
        h_v, nKs2, nVs2, nKf2, nVf2, k13v, v13v, k14v, v14v = vc2(
            h_v, m_full_v, m_sliding_v, update_indicator, plc_v,
            cs_v, ss_v, cf_v, sf_v,
            kSliding2_v, vSliding2_v, kFull2_v, vFull2_v,
        )
        h_after_c2_v = h_v.detach().clone()
        h_v = vc3(h_v, m_full_v, m_sliding_v, plc_v,
                  cs_v, ss_v, cf_v, sf_v, k13v, v13v, k14v, v14v)
        h_after_c3_v = h_v.detach().clone()
        token_ids_v, h_normed_v = vc4(
            h_v, m_full_v, m_sliding_v, plc_v,
            cs_v, ss_v, cf_v, sf_v, k13v, v13v, k14v, v14v,
        )

    verify_argmax = token_ids_v.flatten().tolist()
    print(f"  verify token_ids: {verify_argmax}")

    # ---- Compare ----
    print("\n=== Parity ===")
    print(f"serial argmax: {serial_argmax}")
    print(f"verify argmax: {verify_argmax}")
    argmax_match = (serial_argmax == verify_argmax)
    print(f"argmax match: {argmax_match}")

    print("\nPer-position cosine similarity (serial vs verify):")
    for t in range(K):
        c1 = cos_sim(serial_h_after_c1[t][0, 0], h_after_c1_v[0, t])
        c2 = cos_sim(serial_h_after_c2[t][0, 0], h_after_c2_v[0, t])
        c3 = cos_sim(serial_h_after_c3[t][0, 0], h_after_c3_v[0, t])
        c4 = cos_sim(serial_normed[t][0, 0], h_normed_v[0, t])
        print(f"  t={t}: c1={c1:.6f}  c2={c2:.6f}  c3={c3:.6f}  c4(normed)={c4:.6f}")

    print("\n=== New per-T K/V slice sanity ===")
    print(f"new_K_sliding_c1: {tuple(nKs1.shape)} (expect (7,1,{K},256))")
    print(f"new_K_full_c1:    {tuple(nKf1.shape)} (expect (1,1,{K},512))")
    print(f"new_K_sliding_c2: {tuple(nKs2.shape)} (expect (5,1,{K},256))")
    print(f"new_K_full_c2:    {tuple(nKf2.shape)} (expect (2,1,{K},512))")

    print("\n=== Verdict ===")
    ok = argmax_match
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
