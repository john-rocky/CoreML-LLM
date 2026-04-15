#!/usr/bin/env python3
"""Build a bidirectional vocabulary bridge between Qwen 2.5 and Gemma 4.

Route B / Task 3: cross-vocabulary speculative decoding needs to translate
tokens at inference time.

    Qwen -> Gemma : the drafter proposes Qwen ids; we hand Gemma ids to target.
    Gemma -> Qwen : accepted Gemma tokens advance the drafter's own state.

The map is built by matching decoded surface forms (plain strings). Where
a decoded form appears in both vocabularies we map the ids both ways.
Unmapped ids are -1.

Uses the `tokenizers` library directly (the `transformers.AutoTokenizer`
path is broken for Gemma 4 on current releases). You can pass either a
directory containing `tokenizer.json` or the tokenizer.json file itself.

Output format (binary, little-endian):

    MAGIC  (8 bytes)  = b'QGVMAP01'
    qwen_vocab_size   (uint32)
    gemma_vocab_size  (uint32)
    qwen_to_gemma     (qwen_vocab_size * int32)   # -1 on miss
    gemma_to_qwen     (gemma_vocab_size * int32)  # -1 on miss
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore

MAGIC = b"QGVMAP01"


def load_tokenizer(path: Path) -> Tokenizer:
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found at {path}")
    return Tokenizer.from_file(str(path))


def decode_one(tok: Tokenizer, tid: int) -> str:
    """Decode a single id to its surface form without adding special-token
    markers. Returns empty string on failure."""
    try:
        return tok.decode([tid], skip_special_tokens=False)
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-tokenizer", required=True, type=Path,
                    help="Qwen tokenizer dir or tokenizer.json path")
    ap.add_argument("--gemma-tokenizer", required=True, type=Path,
                    help="Gemma tokenizer dir or tokenizer.json path")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    print(f"[vocab-map] loading Qwen  tokenizer: {args.qwen_tokenizer}")
    qwen = load_tokenizer(args.qwen_tokenizer)
    print(f"[vocab-map] loading Gemma tokenizer: {args.gemma_tokenizer}")
    gemma = load_tokenizer(args.gemma_tokenizer)

    qvs = qwen.get_vocab_size()
    gvs = gemma.get_vocab_size()
    print(f"[vocab-map] qwen vocab_size={qvs}  gemma vocab_size={gvs}")

    # Build surface -> first-id maps (tie-break on lowest id).
    def build_surface_index(tok: Tokenizer, vocab_size: int) -> dict[str, int]:
        surface: dict[str, int] = {}
        for tid in range(vocab_size):
            decoded = decode_one(tok, tid)
            if decoded == "":
                continue
            if decoded not in surface:
                surface[decoded] = tid
        return surface

    q_surface = build_surface_index(qwen, qvs)
    g_surface = build_surface_index(gemma, gvs)
    print(f"[vocab-map] qwen unique surfaces={len(q_surface)}  gemma={len(g_surface)}")

    qwen_to_gemma = [-1] * qvs
    gemma_to_qwen = [-1] * gvs

    # Pass 1: every Qwen id with a Gemma surface match. Multiple Qwen ids
    # sharing a surface all translate to the same Gemma id.
    for tid in range(qvs):
        decoded = decode_one(qwen, tid)
        if decoded == "":
            continue
        gid = g_surface.get(decoded)
        if gid is not None and 0 <= gid < gvs:
            qwen_to_gemma[tid] = gid

    # Pass 2: symmetric — every Gemma id with a Qwen surface match.
    for tid in range(gvs):
        decoded = decode_one(gemma, tid)
        if decoded == "":
            continue
        qid = q_surface.get(decoded)
        if qid is not None and 0 <= qid < qvs:
            gemma_to_qwen[tid] = qid

    q_cov = sum(1 for v in qwen_to_gemma if v >= 0) / qvs
    g_cov = sum(1 for v in gemma_to_qwen if v >= 0) / gvs
    hits = sum(1 for v in qwen_to_gemma if v >= 0)
    print(f"[vocab-map] coverage: qwen->gemma {q_cov*100:.1f}%  gemma->qwen {g_cov*100:.1f}%  hits={hits}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", qvs, gvs))
        f.write(struct.pack(f"<{qvs}i", *qwen_to_gemma))
        f.write(struct.pack(f"<{gvs}i", *gemma_to_qwen))

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"[vocab-map] wrote {args.output} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
