#!/usr/bin/env python3
"""Smoke test: verify ANE micro-optimizations are numerically equivalent.

Tests three changes:
1. exp2 softmax: ane_softmax (exp2) vs reference (exp)
2. MLP tile: (B,C,8,8) reshape vs (B,C,1,1) original
3. GQA broadcast: 5D broadcast matmul vs repeat_interleave

Run:
    python smoke_w2_quality.py                     # unit tests only
    python smoke_w2_quality.py --full-model        # also run full model comparison
"""
from __future__ import annotations
import argparse, sys, os
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ane_ops import MODEL_DTYPE, _LOG2_E, ane_softmax


def ref_softmax(x, dim=-1):
    """Reference softmax using torch.exp (pre-optimization)."""
    x = x.to(MODEL_DTYPE)
    x_max = x.max(dim=dim, keepdim=True).values.to(MODEL_DTYPE)
    x_shifted = (x - x_max).to(MODEL_DTYPE)
    exp_x = torch.exp(x_shifted).to(MODEL_DTYPE)
    exp_sum = exp_x.sum(dim=dim, keepdim=True).to(MODEL_DTYPE)
    return (exp_x / exp_sum).to(MODEL_DTYPE)


def test_exp2_softmax():
    """Verify exp2-based softmax matches exp-based softmax."""
    print("=== Test 1: exp2 softmax ===")
    torch.manual_seed(42)

    for shape_name, shape in [
        ("sliding (1,8,1,512)", (1, 8, 1, 512)),
        ("full (1,8,1,8192)", (1, 8, 1, 8192)),
        ("5D broadcast (1,1,8,1,512)", (1, 1, 8, 1, 512)),
        ("5D broadcast (1,1,8,1,8192)", (1, 1, 8, 1, 8192)),
    ]:
        x = torch.randn(shape, dtype=MODEL_DTYPE)
        # Add large negative mask to simulate causal masking
        mask = torch.zeros(shape, dtype=MODEL_DTYPE)
        mask[..., shape[-1]//2:] = -65504.0
        x = x + mask

        new = ane_softmax(x, dim=-1)
        ref = ref_softmax(x, dim=-1)

        cos = F.cosine_similarity(new.flatten().float(), ref.flatten().float(), dim=0)
        max_diff = (new.float() - ref.float()).abs().max().item()
        print(f"  {shape_name}: cos={cos:.8f}  max_diff={max_diff:.2e}  "
              f"{'PASS' if cos > 0.99999 else 'FAIL'}")
    print()


def test_mlp_tile():
    """Verify (B,C,8,8) tiled MLP produces identical output to (B,C,1,1)."""
    print("=== Test 2: MLP tile reshape ===")
    torch.manual_seed(42)

    C_in, C_mid, C_out = 1536, 6144, 1536
    gate_proj = torch.nn.Conv2d(C_in, C_mid, 1, bias=False, dtype=MODEL_DTYPE)
    up_proj = torch.nn.Conv2d(C_in, C_mid, 1, bias=False, dtype=MODEL_DTYPE)
    down_proj = torch.nn.Conv2d(C_mid, C_out, 1, bias=False, dtype=MODEL_DTYPE)

    x = torch.randn(1, C_in, 1, 1, dtype=MODEL_DTYPE)

    # Reference: (1, C, 1, 1) path
    with torch.no_grad():
        g_ref = gate_proj(x)
        u_ref = up_proj(x)
        g_ref = F.gelu(g_ref, approximate="tanh")
        ref_out = down_proj(g_ref * u_ref)  # (1, C_out, 1, 1)

    # New: (1, C, 8, 8) tiled path
    with torch.no_grad():
        x_tiled = x.expand(1, -1, 8, 8)
        g_new = gate_proj(x_tiled)
        u_new = up_proj(x_tiled)
        g_new = F.gelu(g_new, approximate="tanh")
        new_out = down_proj(g_new * u_new)[:, :, :1, :1]  # slice back

    cos = F.cosine_similarity(new_out.flatten().float(), ref_out.flatten().float(), dim=0)
    max_diff = (new_out.float() - ref_out.float()).abs().max().item()
    print(f"  MLP (1536→6144→1536): cos={cos:.8f}  max_diff={max_diff:.2e}  "
          f"{'PASS' if cos > 0.99999 else 'FAIL'}")

    # Also test double-wide MLP (KV-shared layers)
    C_mid2 = 12288
    gate_proj2 = torch.nn.Conv2d(C_in, C_mid2, 1, bias=False, dtype=MODEL_DTYPE)
    up_proj2 = torch.nn.Conv2d(C_in, C_mid2, 1, bias=False, dtype=MODEL_DTYPE)
    down_proj2 = torch.nn.Conv2d(C_mid2, C_out, 1, bias=False, dtype=MODEL_DTYPE)

    with torch.no_grad():
        ref_out2 = down_proj2(F.gelu(gate_proj2(x), approximate="tanh") * up_proj2(x))
        x_t2 = x.expand(1, -1, 8, 8)
        new_out2 = down_proj2(F.gelu(gate_proj2(x_t2), approximate="tanh") * up_proj2(x_t2))[:, :, :1, :1]

    cos2 = F.cosine_similarity(new_out2.flatten().float(), ref_out2.flatten().float(), dim=0)
    max_diff2 = (new_out2.float() - ref_out2.float()).abs().max().item()
    print(f"  MLP (1536→12288→1536 double-wide): cos={cos2:.8f}  max_diff={max_diff2:.2e}  "
          f"{'PASS' if cos2 > 0.99999 else 'FAIL'}")
    print()


def test_gqa_broadcast():
    """Verify broadcast matmul matches repeat_interleave for GQA."""
    print("=== Test 3: GQA broadcast matmul ===")
    torch.manual_seed(42)

    num_kv = 1
    n_rep = 8
    num_heads = num_kv * n_rep

    for desc, seq_len, hd in [
        ("sliding (W=512, hd=256)", 512, 256),
        ("full (ctx=8192, hd=512)", 8192, 512),
    ]:
        q = torch.randn(1, num_heads, 1, hd, dtype=MODEL_DTYPE)
        K = torch.randn(1, num_kv, seq_len, hd, dtype=MODEL_DTYPE) * 0.1
        V = torch.randn(1, num_kv, seq_len, hd, dtype=MODEL_DTYPE) * 0.1

        mask = torch.zeros(1, 1, 1, seq_len, dtype=MODEL_DTYPE)
        mask[..., seq_len//2:] = -65504.0

        # Reference: repeat_interleave + 4D matmul
        K_exp = K.repeat_interleave(n_rep, dim=1)
        V_exp = V.repeat_interleave(n_rep, dim=1)
        aw_ref = torch.matmul(q, K_exp.transpose(-1, -2))
        aw_ref = aw_ref + mask
        aw_ref = ref_softmax(aw_ref, dim=-1)
        ref_out = torch.matmul(aw_ref, V_exp)  # (1, 8, 1, hd)

        # New: 5D broadcast matmul
        q_g = q.view(1, num_kv, n_rep, 1, hd)
        K_b = K.unsqueeze(2)   # (1, 1, 1, S, hd)
        V_b = V.unsqueeze(2)   # (1, 1, 1, S, hd)
        aw_new = torch.matmul(q_g, K_b.transpose(-1, -2))  # (1,1,8,1,S)
        aw_new = aw_new + mask  # 4D mask broadcasts to 5D
        aw_new = ane_softmax(aw_new, dim=-1)
        new_out = torch.matmul(aw_new, V_b)  # (1,1,8,1,hd)
        new_out = new_out.view(1, num_heads, 1, hd)

        cos = F.cosine_similarity(new_out.flatten().float(), ref_out.flatten().float(), dim=0)
        max_diff = (new_out.float() - ref_out.float()).abs().max().item()
        print(f"  {desc}: cos={cos:.8f}  max_diff={max_diff:.2e}  "
              f"{'PASS' if cos > 0.9999 else 'FAIL'}")
    print()


def test_flash_broadcast():
    """Verify flash decoding with broadcast GQA matches reference."""
    print("=== Test 4: Flash decoding + broadcast GQA ===")
    torch.manual_seed(42)

    sys.path.insert(0, os.path.join(ROOT, "models"))
    from models.gemma4_swa_flash import flash_decode_attention, _flash_one_chunk

    num_kv = 1
    n_rep = 8
    num_heads = num_kv * n_rep
    hd = 512
    ctx = 8192
    num_chunks = 8

    q = torch.randn(1, num_heads, 1, hd, dtype=MODEL_DTYPE)
    K = torch.randn(1, num_kv, ctx, hd, dtype=MODEL_DTYPE) * 0.1
    V = torch.randn(1, num_kv, ctx, hd, dtype=MODEL_DTYPE) * 0.1
    mask = torch.zeros(1, 1, 1, ctx, dtype=MODEL_DTYPE)
    mask[..., ctx//2:] = -65504.0

    # Reference: repeat_interleave + standard attention
    K_exp = K.repeat_interleave(n_rep, dim=1)
    V_exp = V.repeat_interleave(n_rep, dim=1)
    aw_ref = torch.matmul(q, K_exp.transpose(-1, -2)) + mask
    aw_ref = ref_softmax(aw_ref, dim=-1)
    ref_out = torch.matmul(aw_ref, V_exp)

    # New: flash decode with broadcast
    q_g = q.view(1, num_kv, n_rep, 1, hd)
    K_b = K.unsqueeze(2)
    V_b = V.unsqueeze(2)
    new_out = flash_decode_attention(q_g, K_b, V_b, mask, num_chunks, num_kv, n_rep, hd)

    cos = F.cosine_similarity(new_out.flatten().float(), ref_out.flatten().float(), dim=0)
    max_diff = (new_out.float() - ref_out.float()).abs().max().item()
    print(f"  Flash 8-chunk (ctx=8192, hd=512): cos={cos:.8f}  max_diff={max_diff:.2e}  "
          f"{'PASS' if cos > 0.999 else 'FAIL'}")
    print()


def test_full_model():
    """End-to-end: load model, run forward, check output sanity."""
    print("=== Test 5: Full model forward pass ===")
    from models.gemma4 import Gemma4Model
    from models.gemma4_swa_flash import FlashChunk1, FlashChunk2, FlashChunk3, FlashChunk4

    HF_DIR = f"{ROOT}/output/gemma4-e2b-final/hf_model"
    CTX = 8192
    W = 512

    if not os.path.isdir(HF_DIR):
        print(f"  SKIP: model weights not found at {HF_DIR}")
        return

    print("  Loading model...")
    base = Gemma4Model.from_pretrained(HF_DIR, context_length=CTX)
    base.eval()

    c1 = FlashChunk1(base).eval()
    c2 = FlashChunk2(base).eval()
    c3 = FlashChunk3(base).eval()
    c4 = FlashChunk4(base).eval()

    config = base.config
    hidden = config.hidden_size
    pld = config.hidden_size_per_layer_input
    nlayers = config.num_hidden_layers
    max_hd = 512

    with torch.no_grad():
        hs = torch.randn(1, 1, hidden, dtype=MODEL_DTYPE) * 0.01
        cm_full = torch.zeros(1, 1, 1, CTX, dtype=MODEL_DTYPE)
        cm_full[..., 1:] = -65504.0  # position 0 visible only
        cm_slide = torch.zeros(1, 1, 1, W, dtype=MODEL_DTYPE)
        cm_slide[..., 1:] = -65504.0
        um = torch.zeros(1, 1, CTX, 1, dtype=MODEL_DTYPE)
        um[:, :, 0, :] = 1.0
        plr = torch.randn(1, 1, nlayers * pld, dtype=MODEL_DTYPE) * 0.01
        cos_s = torch.randn(1, 1, 1, 256, dtype=MODEL_DTYPE)
        sin_s = torch.randn(1, 1, 1, 256, dtype=MODEL_DTYPE)
        cos_f = torch.randn(1, 1, 1, 512, dtype=MODEL_DTYPE)
        sin_f = torch.randn(1, 1, 1, 512, dtype=MODEL_DTYPE)
        Ks = torch.zeros(7, 1, W, max_hd, dtype=MODEL_DTYPE)
        Vs = torch.zeros(7, 1, W, max_hd, dtype=MODEL_DTYPE)
        Kf = torch.zeros(1, 1, CTX, max_hd, dtype=MODEL_DTYPE)
        Vf = torch.zeros(1, 1, CTX, max_hd, dtype=MODEL_DTYPE)

        hs, Ks, Vs, Kf, Vf, plc = c1(hs, cm_full, cm_slide, um, plr,
                                       cos_s, sin_s, cos_f, sin_f, Ks, Vs, Kf, Vf)
        print(f"  Chunk1: hs norm={hs.norm():.4f}, no NaN={not hs.isnan().any()}")

        Ks2 = torch.zeros(5, 1, W, max_hd, dtype=MODEL_DTYPE)
        Vs2 = torch.zeros(5, 1, W, max_hd, dtype=MODEL_DTYPE)
        Kf2 = torch.zeros(2, 1, CTX, max_hd, dtype=MODEL_DTYPE)
        Vf2 = torch.zeros(2, 1, CTX, max_hd, dtype=MODEL_DTYPE)
        hs, Ks2, Vs2, Kf2, Vf2, kv13k, kv13v, kv14k, kv14v = c2(
            hs, cm_full, cm_slide, um, plc, cos_s, sin_s, cos_f, sin_f, Ks2, Vs2, Kf2, Vf2)
        print(f"  Chunk2: hs norm={hs.norm():.4f}, no NaN={not hs.isnan().any()}")

        hs = c3(hs, cm_full, cm_slide, um, plc, cos_s, sin_s, cos_f, sin_f,
                kv13k, kv13v, kv14k, kv14v)
        print(f"  Chunk3: hs norm={hs.norm():.4f}, no NaN={not hs.isnan().any()}")

        tid, tlogit, normed = c4(hs, cm_full, cm_slide, um, plc, cos_s, sin_s, cos_f, sin_f,
                                  kv13k, kv13v, kv14k, kv14v)
        print(f"  Chunk4: token_id={tid.item()}, logit={tlogit.item():.4f}, no NaN={not normed.isnan().any()}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Smoke test for ANE micro-optimizations")
    parser.add_argument("--full-model", action="store_true", help="Run full model forward pass")
    args = parser.parse_args()

    test_exp2_softmax()
    test_mlp_tile()
    test_gqa_broadcast()
    test_flash_broadcast()

    if args.full_model:
        test_full_model()

    print("All unit tests complete.")


if __name__ == "__main__":
    main()
