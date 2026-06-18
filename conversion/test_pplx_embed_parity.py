#!/usr/bin/env python3
"""Parity check: ANE Qwen3 encoder (PplxEmbedModel) vs the fp32 golden reference.

Validates the architecture port *before* CoreML conversion — fast CPU loop, small
fixed seq-len. Compares both the fp16 pooled embedding and the int8 output against
conversion/pplx_embed_reference.py on a small multilingual sample.

Usage:
    python conversion/test_pplx_embed_parity.py                 # plain, L=64, K=16
    python conversion/test_pplx_embed_parity.py --max-seq-len 128 --rescale-k 16

Pass criteria (cosine vs fp32, zero-norm rows excluded):
    pooled fp16 ≥ 0.999   (encoder port fidelity)
    int8        ≥ 0.997   (foundation gate)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pplx_embed_reference as R  # noqa: E402
from models.qwen3_encoder import (  # noqa: E402
    Qwen3EncoderConfig,
    PplxEmbedModel,
    load_encoder_weights,
    apply_fp16_residual_rescale,
)

SENTENCES = [
    "hello world",
    "Bonjour le monde.",
    "東京は日本の首都です。",
    "Embeddings are dense vectors.",
    "La inteligencia artificial avanza rápido.",
    "Das Wetter ist heute schön.",
    "机器学习改变世界。",
    "Quantum computing uses qubits.",
    "Привет, как дела?",
    "المعرفة قوة.",
    "The mitochondria is the powerhouse of the cell.",
    "Tokyo Shanghai Paris Berlin Cairo.",
]


def _snapshot_dir(hf_repo: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(hf_repo, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.py"])


def main() -> int:
    ap = argparse.ArgumentParser(description="pplx-embed ANE-encoder parity test")
    ap.add_argument("--hf-repo", default="perplexity-ai/pplx-embed-v1-0.6b")
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--rescale-k", type=float, default=8.0,
                    help="fp16 residual rescale factor (0 disables; K=8 is the default sweet spot)")
    ap.add_argument("--pooled-gate", type=float, default=0.999)
    ap.add_argument("--int8-gate", type=float, default=0.997)
    args = ap.parse_args()

    snap = _snapshot_dir(args.hf_repo)
    L = args.max_seq_len
    K = args.rescale_k or None

    cfg = Qwen3EncoderConfig.from_json(os.path.join(snap, "config.json"), max_seq_len=L)
    print(f"[cfg] hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} hd={cfg.head_dim} "
          f"theta={cfg.rope_theta} L={L} K={K}")

    m_pool = PplxEmbedModel(cfg, "pooled_fp16").eval()
    load_encoder_weights(m_pool.encoder, snap)
    m_int8 = PplxEmbedModel(cfg, "int8").eval()
    load_encoder_weights(m_int8.encoder, snap)
    if K:
        apply_fp16_residual_rescale(m_pool.encoder, K)
        apply_fp16_residual_rescale(m_int8.encoder, K)

    ref = R.Reference(args.hf_repo)
    tok = ref.tokenizer

    cos_pool, cos_int8 = [], []
    for t in SENTENCES:
        enc = tok([t], return_tensors="pt", truncation=True, max_length=L)
        ids = enc["input_ids"]
        n = ids.shape[1]
        pid = torch.zeros((1, L), dtype=torch.int32); pid[0, :n] = ids[0].to(torch.int32)
        pam = torch.zeros((1, L), dtype=torch.float16); pam[0, :n] = 1.0
        with torch.no_grad():
            o_pool = m_pool(pid, pam).numpy().astype(np.float32)
            o_int8 = m_int8(pid, pam).numpy().astype(np.float32)
        ref_pool = R.masked_mean(*ref.hidden_states([t])).numpy().astype(np.float32)
        ref_int8 = ref.embed([t]).astype(np.float32)
        cp = R.cosine_similarity(o_pool, ref_pool)[0]
        ci = R.cosine_similarity(o_int8, ref_int8)[0]
        cos_pool.append(cp); cos_int8.append(ci)
        print(f"[txt] n={n:3d} pooled={cp:.6f} int8={ci:.6f}  {t[:28]}")

    cp = np.array(cos_pool); ci = np.array(cos_int8)
    pooled_min, int8_min = float(np.nanmin(cp)), float(np.nanmin(ci))
    n_nan = int(np.isnan(cp).sum() + np.isnan(ci).sum())
    print(f"\n[POOLED] mean={np.nanmean(cp):.6f} min={pooled_min:.6f}  (gate ≥ {args.pooled_gate})")
    print(f"[INT8]   mean={np.nanmean(ci):.6f} min={int8_min:.6f}  (gate ≥ {args.int8_gate})")
    ok = (n_nan == 0) and (pooled_min >= args.pooled_gate) and (int8_min >= args.int8_gate)
    print(f"\n{'PASS' if ok else 'FAIL'}  (nan={n_nan})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
