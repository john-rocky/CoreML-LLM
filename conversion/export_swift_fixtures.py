#!/usr/bin/env python3
"""Export fidelity fixtures for the Swift pplx-embed bench.

Native int8 model output is not readable from the Python CoreML bridge on
macOS26, so fidelity/latency for the int8 deliverable is measured in Swift.
This script produces the ground-truth side: pre-tokenized, bucket-padded inputs
plus the fp32-reference int8 embedding for each text.

Output JSON:
    { "L": <bucket>, "hf_repo": ..., "items": [
        {"text": str, "input_ids": [L ints], "n": int, "ref_int8": [1024 ints]} ] }

Usage:
    python conversion/export_swift_fixtures.py --max-seq-len 4096 --out /tmp/pplx_fix.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pplx_embed_reference as R  # noqa: E402

SENTENCES = [
    "hello world",
    "Quantum computing uses qubits.",
    "東京は日本の首都です。",
    "Bonjour le monde.",
    "机器学习改变世界。",
    "المعرفة قوة.",
    "Привет, как дела?",
    "Machine learning has transformed how we process information. " * 8,
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="perplexity-ai/pplx-embed-v1-0.6b")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--out", default="/tmp/pplx_fixtures.json")
    args = ap.parse_args()

    L = args.max_seq_len
    ref = R.Reference(args.hf_repo)
    tok = ref.tokenizer

    items = []
    for t in SENTENCES:
        enc = tok([t], return_tensors="pt", truncation=True, max_length=L)
        ids = enc["input_ids"][0].tolist()
        n = len(ids)
        padded = ids + [0] * (L - n)
        # Reference int8 over exactly these n tokens (matched truncation).
        import torch
        rh = ref.model(input_ids=enc["input_ids"][:, :n],
                       attention_mask=torch.ones((1, n), dtype=torch.long)).last_hidden_state.float()
        ref_i8 = R.int8_tanh_quant(R.masked_mean(rh, torch.ones((1, n)))).reshape(-1).astype(int).tolist()
        items.append({"text": t, "input_ids": padded, "n": n, "ref_int8": ref_i8})
        print(f"  fixture n={n:4d}  {t[:32]}")

    out = {"L": L, "hf_repo": args.hf_repo, "items": items}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(items)} fixtures (L={L}) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
