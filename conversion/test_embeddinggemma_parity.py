#!/usr/bin/env python3
"""Parity check: HF EmbeddingGemma vs our ANE-style EmbeddingGemmaModel.

Prerequisite:
    python conversion/build_embeddinggemma_bundle.py --max-seq-len 512

Then:
    python conversion/test_embeddinggemma_parity.py

Loads the HuggingFace reference via SentenceTransformer (if available) and
our ANE-style PyTorch EmbeddingGemmaModel from the same snapshot, then
compares cosine similarity on a small multilingual sentence set.

Pass criteria:
    mean cosine(hf_emb, ours_emb) at d=768  ≥  0.995
    mean cosine at d=128 after Matryoshka truncate+renorm  ≥  0.98
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Apple Neural Engine inference for on-device language models.",
    "A journey of a thousand miles begins with a single step.",
    "Embedding models represent text as dense vectors.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Machine learning research has accelerated in recent years.",
    "Coffee is one of the most consumed beverages worldwide.",
    "The sun rises in the east and sets in the west.",
    # non-English
    "東京は日本の首都である。",
    "La vida es bella cuando uno tiene amigos.",
    "Die Katze sitzt auf der Matte.",
    "Le chat est assis sur le tapis.",
    "猫が丸くなって寝ている。",
    "人工智能正在改变我们的生活方式。",
    "O rato roeu a roupa do rei de Roma.",
    "Спокойной ночи, Москва.",
]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return float(F.cosine_similarity(a, b, dim=0))


def _matryoshka_truncate(vec: torch.Tensor, dim: int) -> torch.Tensor:
    v = vec.flatten()[:dim]
    return v / (v.norm(p=2) + 1e-12)


def _load_hf_reference(hf_dir: str, sentences: list[str]) -> torch.Tensor:
    """Encode sentences with the HF sentence-transformers reference model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "sentence-transformers not installed. Run: pip install sentence-transformers\n"
            "(Needed only for parity testing — not a runtime dependency.)"
        )
    print("Loading HF reference via SentenceTransformer...")
    st = SentenceTransformer(hf_dir, trust_remote_code=True)
    emb = st.encode(sentences, normalize_embeddings=True, convert_to_tensor=True)
    return emb.detach().cpu().to(torch.float32)


def _encode_ours(hf_dir: str, sentences: list[str], max_seq_len: int) -> torch.Tensor:
    """Encode sentences with our ANE-style PyTorch EmbeddingGemmaModel (CPU)."""
    from transformers import AutoTokenizer
    from models.embeddinggemma import EmbeddingGemmaModel
    from models.gemma3_encoder import EncoderConfig
    from build_embeddinggemma_bundle import (
        _load_state_dicts, _copy_into_model, _detect_dense_dims,
        _load_transformer_config,
    )

    tok = AutoTokenizer.from_pretrained(hf_dir)
    text_cfg = _load_transformer_config(hf_dir)
    text_cfg["max_seq_len"] = max_seq_len
    enc_cfg = EncoderConfig(**text_cfg)

    weights = _load_state_dicts(hf_dir)
    detected = _detect_dense_dims(weights)
    if detected is None:
        raise SystemExit("Dense layer weights missing — snapshot must include 2_Dense/ and 3_Dense/.")
    dense_inter, embed_dim = detected

    model = EmbeddingGemmaModel(enc_cfg, embed_dim=embed_dim, dense_intermediate_dim=dense_inter)
    model.eval()
    _copy_into_model(model, weights)

    out = []
    for s in sentences:
        enc = tok(s, padding="max_length", truncation=True,
                  max_length=max_seq_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(torch.int32)
        attn_mask = enc["attention_mask"].to(torch.float16)
        with torch.no_grad():
            emb = model(input_ids, attn_mask)
        out.append(emb.squeeze(0).to(torch.float32))
    return torch.stack(out, dim=0)


def main():
    parser = argparse.ArgumentParser(description="EmbeddingGemma parity test")
    parser.add_argument("--hf-dir", type=str,
                        default=os.path.join(ROOT, "..", "output",
                                             "embeddinggemma-300m", "bundle", "hf_model"),
                        help="Path to the HF snapshot (with 2_Dense/, 3_Dense/ subdirs)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Fixed-length padding for encoding (default: 512)")
    args = parser.parse_args()

    if not os.path.isdir(args.hf_dir):
        raise SystemExit(
            f"HF dir not found: {args.hf_dir}\n"
            "Run: python conversion/build_embeddinggemma_bundle.py --max-seq-len 512 first."
        )

    hf_emb = _load_hf_reference(args.hf_dir, SENTENCES)
    our_emb = _encode_ours(args.hf_dir, SENTENCES, args.max_seq_len)

    cos768 = [_cosine(h, o) for h, o in zip(hf_emb, our_emb)]
    print(f"\ncosine at d=768: mean={np.mean(cos768):.4f}  min={np.min(cos768):.4f}")

    # Matryoshka truncation test.
    cos128 = []
    for h, o in zip(hf_emb, our_emb):
        h128 = _matryoshka_truncate(h, 128)
        o128 = _matryoshka_truncate(o, 128)
        cos128.append(_cosine(h128, o128))
    print(f"cosine at d=128: mean={np.mean(cos128):.4f}  min={np.min(cos128):.4f}")

    pass768 = float(np.mean(cos768)) >= 0.995
    pass128 = float(np.mean(cos128)) >= 0.980
    print(f"\nPARITY d=768: {'PASS' if pass768 else 'FAIL'}  (need ≥0.995)")
    print(f"PARITY d=128: {'PASS' if pass128 else 'FAIL'}  (need ≥0.980)")
    sys.exit(0 if (pass768 and pass128) else 1)


if __name__ == "__main__":
    main()
