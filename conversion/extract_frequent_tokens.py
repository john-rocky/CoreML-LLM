#!/usr/bin/env python3
"""Extract top-N frequent token IDs for Gemma 4 by tokenizing a representative
English corpus. Output: frequent_tokens.bin (Int32 LE, top-N IDs in descending
frequency order).

Used by Swift `verifyCandidatesSubset` as the candidate-set padding so the
sparse LM-head matmul covers target's true argmax most of the time. Without a
corpus-derived set, the synthetic 0..1023 fallback misses the 236xxx range of
common Gemma 4 BPE pieces and the lossless-fallback rate climbs to ~100%.

Usage:
    python conversion/extract_frequent_tokens.py --top 1024
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# Compact English chat / code / explanation corpus. ~30 KB of text covering
# common chat replies, lists, code snippets, narrative. Not exhaustive — just
# enough to surface the top ~1024 most frequent Gemma 4 BPE pieces.
CORPUS_SOURCES = [
    "Hello! How are you today? I'm doing well, thanks for asking. I help with " * 10,
    "What is your favourite hobby and why? My favourite hobby is reading because " * 10,
    "I love spending time outdoors hiking and exploring nature with my friends. " * 8,
    "Let me explain how this works in simple terms. First, we need to understand " * 8,
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen " * 8,
    "Could you tell me more about that? Sure, I'd be happy to explain further. " * 8,
    "I think this is a great idea because it solves the problem efficiently. " * 8,
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n" * 5,
    "for i in range(10):\n    print(f'Hello, world! {i}')\n" * 5,
    "import numpy as np\nimport torch\nfrom transformers import AutoTokenizer\n" * 5,
    "The cat sat on the mat. The dog ran across the yard. The bird flew through the sky. " * 8,
    "First, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth. " * 8,
    "January, February, March, April, May, June, July, August, September, October. " * 8,
    "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday. " * 8,
    "Yes, no, maybe, perhaps, definitely, absolutely, certainly, possibly. " * 8,
    "He said that she had been working on it for hours and was almost done. " * 8,
    "The model uses self-attention layers to process input tokens and generate predictions. " * 6,
    "When you run the program, it loads the data, processes each item, and outputs the result. " * 6,
    "Could you please help me understand this concept better? I would really appreciate it. " * 6,
    "On the one hand, this approach is simple. On the other hand, it might not scale. " * 6,
    "1. Open the file. 2. Read the contents. 3. Process each line. 4. Write the output. " * 6,
    "If you have any questions, please don't hesitate to ask. I'm here to help. " * 6,
    "The weather today is sunny with a high of 75 degrees Fahrenheit and low humidity. " * 6,
    "We need to make a decision soon, but first let's discuss the pros and cons. " * 6,
    "I'm sorry to hear that. Is there anything I can do to help you feel better? " * 6,
    "Once upon a time in a faraway kingdom, there lived a wise old wizard who knew the secret. " * 6,
    "The reason why this happens is because the system needs to maintain consistency across all nodes. " * 5,
    "In conclusion, the experiment demonstrates that the hypothesis was correct under the given conditions. " * 5,
    "Step one: gather all the ingredients. Step two: mix them together. Step three: bake at 350 degrees. " * 5,
    "What time is it? It's three o'clock. What day is it? It's Wednesday the fifteenth. " * 5,
    "The book is on the table. The pencil is in the drawer. The chair is by the window. " * 5,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--top", type=int, default=1024,
                    help="Number of top frequent token IDs to emit")
    ap.add_argument("--output", default=None,
                    help="Output .bin path (default: ../output/<model>/frequent_tokens.bin)")
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--extra-corpus", default=None,
                    help="Optional path to extra text to append to the corpus")
    ap.add_argument("--use-gutenberg", action="store_true",
                    help="Append nltk gutenberg sample texts for broader coverage")
    ap.add_argument("--corpus-dir", default=None,
                    help="Path to a directory of .txt files to use as corpus")
    args = ap.parse_args()

    from config import MODEL_REGISTRY
    if args.hf_dir is None:
        args.hf_dir = os.path.join(ROOT, "..", "output", args.model, "hf_model")
    args.output = args.output or os.path.join(
        ROOT, "..", "output", args.model, "frequent_tokens.bin")

    print(f"loading tokenizer from {args.hf_dir} ...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    vocab_size = tok.vocab_size if hasattr(tok, 'vocab_size') else len(tok)
    print(f"  vocab_size={vocab_size}")

    corpus = "\n".join(CORPUS_SOURCES)
    if args.extra_corpus and os.path.isfile(args.extra_corpus):
        with open(args.extra_corpus, "r", encoding="utf-8", errors="ignore") as f:
            corpus += "\n" + f.read()
    if args.corpus_dir and os.path.isdir(args.corpus_dir):
        for fn in sorted(os.listdir(args.corpus_dir)):
            if fn.endswith(".txt"):
                with open(os.path.join(args.corpus_dir, fn), "r",
                          encoding="utf-8", errors="ignore") as f:
                    corpus += "\n" + f.read()
    if args.use_gutenberg:
        try:
            import nltk
            try:
                nltk.data.find('corpora/gutenberg')
            except LookupError:
                nltk.download('gutenberg', quiet=True)
            from nltk.corpus import gutenberg
            for fid in gutenberg.fileids():
                corpus += "\n" + gutenberg.raw(fid)
            print(f"  appended gutenberg corpus ({len(gutenberg.fileids())} files)")
        except Exception as e:
            print(f"  warning: gutenberg load failed: {e}")
    print(f"  corpus size: {len(corpus)} chars")

    # Tokenize. Use plain encode (no special tokens) so we count raw pieces.
    ids = tok.encode(corpus, add_special_tokens=False)
    print(f"  tokenized to {len(ids)} pieces")

    cnt = Counter(ids)
    # Always include Gemma special tokens at the top so generation can
    # terminate (caller already injects them, but keeping them here means
    # the .bin is self-contained).
    SPECIAL_FIRST = [0, 1, 2, 106]  # pad / eos / bos / end_of_turn
    top_ids: list[int] = []
    seen: set[int] = set()
    for s in SPECIAL_FIRST:
        top_ids.append(s); seen.add(s)
    for tok_id, _freq in cnt.most_common(args.top * 2):
        if tok_id in seen: continue
        if tok_id < 0 or tok_id >= vocab_size: continue
        top_ids.append(tok_id); seen.add(tok_id)
        if len(top_ids) >= args.top: break

    print(f"  selected {len(top_ids)} unique IDs (top by frequency)")

    arr = np.array(top_ids, dtype=np.int32)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    arr.tofile(args.output)
    print(f"  wrote {args.output} ({arr.nbytes} bytes)")

    # Preview top 16
    samples = []
    for tok_id in top_ids[:16]:
        try:
            piece = tok.decode([tok_id])
            samples.append(f"{tok_id}={piece!r}")
        except Exception:
            samples.append(f"{tok_id}=?")
    print("  top16: " + ", ".join(samples))


if __name__ == "__main__":
    main()
