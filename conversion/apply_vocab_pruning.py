#!/usr/bin/env python3
"""Apply vocabulary pruning to a Gemma 4 checkpoint (Approach D).

Companion to the existing dry-run analyzer `prune_vocab.py` — this script
actually produces a pruned HF-compatible model. Cuts ~1.7 GB of download
by slicing embed_tokens, embed_per_layer (PLE — the largest tensor), and
lm_head rows to a ~50k retained set.

Scaffold: v1 selects tokens purely by corpus frequency + must-keep
specials. Quality typically within 0.5% of baseline at keep ≥ 40k, but
aggressive pruning (<30k) should be gated by a LongBench regression run
(`conversion/eval_longbench.py`).

Algorithm
---------
1. Tokenize a representative corpus (`eagle_corpus.jsonl`), count id
   frequencies.
2. Union with must-keep special tokens (BOS, EOS, PAD, turn markers,
   image/audio placeholders, low reserved range).
3. Select top-K from the union, sort ascending to produce a stable
   old_id → new_id remap.
4. Slice weight rows of embed_tokens / embed_per_layer / lm_head.
5. Update config.vocab_size, save HF model, emit `vocab_remap.json`
   so the runtime can round-trip token ids between the original
   tokenizer and the pruned model.

Usage
-----
    python conversion/apply_vocab_pruning.py \\
        --model-id google/gemma-4-E2B-it \\
        --corpus /path/to/eagle_corpus.jsonl \\
        --keep 50000 \\
        --output ./gemma-4-E2B-it-pruned

Follow-up (recommended): short QLoRA re-stabilize over the same corpus
(re-use `conversion/finetune_mqa_recovery.ipynb` structure, target the
pruned embed/lm_head). Pair with `conversion/eval_longbench.py` to
verify no regression on long-context tasks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ESSENTIAL_SPECIAL_TOKENS = [
    "<bos>", "<eos>", "<pad>", "<unk>",
    "<|turn>", "<turn|>",
    "<start_of_image>", "<end_of_image>",
    "<start_of_audio>", "<end_of_audio>",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    ap.add_argument("--corpus", type=str, required=True,
                    help="JSONL corpus (download_eagle_corpus.py output or similar)")
    ap.add_argument("--keep", type=int, default=50000,
                    help="Target vocab size after pruning (including specials)")
    ap.add_argument("--output", type=str, required=True,
                    help="Output HF model directory")
    ap.add_argument("--num-samples", type=int, default=30000)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    try:
        from transformers import Gemma4ForConditionalGeneration as TCls
    except Exception:
        from transformers import AutoModelForCausalLM as TCls

    print(f"loading tokenizer {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    orig_vocab_size = tok.vocab_size
    print(f"  original vocab: {orig_vocab_size}")

    # Count id frequencies
    print(f"\ncounting token frequencies in {args.corpus}...")
    counter: Counter[int] = Counter()
    n = 0
    with open(args.corpus) as f:
        for line in f:
            if n >= args.num_samples: break
            text = json.loads(line).get("text", "")
            if not text: continue
            ids = tok.encode(text, add_special_tokens=False)
            counter.update(ids)
            n += 1
    print(f"  counted {n} sequences, {sum(counter.values()):,} total tokens, "
          f"{len(counter)} unique ids seen")

    # Must-keep specials
    must_keep: set[int] = set()
    special_map = getattr(tok, "special_tokens_map", {}) or {}
    for _, tokstr in special_map.items():
        if isinstance(tokstr, list):
            for t in tokstr:
                tid = tok.convert_tokens_to_ids(t)
                if isinstance(tid, int) and tid >= 0: must_keep.add(tid)
        elif tokstr:
            tid = tok.convert_tokens_to_ids(tokstr)
            if isinstance(tid, int) and tid >= 0: must_keep.add(tid)
    for name in ESSENTIAL_SPECIAL_TOKENS:
        tid = tok.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0: must_keep.add(tid)
    must_keep.update(range(0, 8))
    print(f"  must-keep specials: {len(must_keep)}")

    # Retention set
    ranked = sorted(counter.items(), key=lambda kv: -kv[1])
    retained: list[int] = sorted(must_keep)
    seen = set(retained)
    for tid, _ in ranked:
        if len(retained) >= args.keep: break
        if tid in seen: continue
        retained.append(tid); seen.add(tid)
    for tid in range(orig_vocab_size):
        if len(retained) >= args.keep: break
        if tid in seen: continue
        retained.append(tid); seen.add(tid)
    retained_sorted = sorted(retained[: args.keep])
    new_size = len(retained_sorted)
    print(f"  retaining {new_size} ({100*new_size/orig_vocab_size:.1f}% of original)")

    # Load model
    print(f"\nloading model...")
    model = TCls.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="cpu")

    with torch.no_grad():
        tm = model.model
        if hasattr(tm, "language_model"): tm = tm.language_model
        inner = tm.model if hasattr(tm, "model") else tm
        idx_t = torch.tensor(retained_sorted, dtype=torch.long)

        et = inner.embed_tokens
        new_w = et.weight.data.index_select(0, idx_t).contiguous()
        print(f"  embed_tokens: {tuple(et.weight.shape)} -> {tuple(new_w.shape)}")
        et.weight = torch.nn.Parameter(new_w, requires_grad=False)
        et.num_embeddings = new_size

        ple = getattr(inner, "embed_tokens_per_layer", None)
        if ple is not None:
            new_ple = ple.weight.data.index_select(0, idx_t).contiguous()
            print(f"  embed_tokens_per_layer: {tuple(ple.weight.shape)} -> {tuple(new_ple.shape)}")
            ple.weight = torch.nn.Parameter(new_ple, requires_grad=False)
            ple.num_embeddings = new_size
        else:
            print("  embed_tokens_per_layer not found — skipping")

        lh = model.lm_head
        new_lh = lh.weight.data.index_select(0, idx_t).contiguous()
        print(f"  lm_head: {tuple(lh.weight.shape)} -> {tuple(new_lh.shape)}")
        lh.weight = torch.nn.Parameter(new_lh, requires_grad=False)
        lh.out_features = new_size

    # Config
    cfg = model.config
    if hasattr(cfg, "text_config"): cfg.text_config.vocab_size = new_size
    else: cfg.vocab_size = new_size
    model.config = cfg

    # Save
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    print(f"\nsaving to {out}...")
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)   # original tokenizer; runtime remaps via vocab_remap.json

    # Remap table for runtime round-trip
    remap = {
        "original_vocab_size": orig_vocab_size,
        "pruned_vocab_size":   new_size,
        "new_id_to_old_id":    retained_sorted,
        "model_id":            args.model_id,
        "notes": [
            "Tokenize with original tokenizer -> map each old_id to new_id via the",
            "inverse of this table -> feed into pruned model.",
            "Model argmax output is in new_id space -> map back to old_id via",
            "new_id_to_old_id[new_id] for decoding with original tokenizer.",
            "Apps can alternatively rewrite tokenizer.json to match the pruned",
            "space and skip runtime remap entirely (not done here to preserve",
            "interop with downstream tools).",
        ],
    }
    (out / "vocab_remap.json").write_text(json.dumps(remap, indent=2))
    print(f"wrote vocab_remap.json")

    # Size summary
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    h = tcfg.hidden_size
    ple_h = getattr(tcfg, "hidden_size_per_layer_input", 256)
    n_layers = getattr(tcfg, "num_hidden_layers", 35)

    def gb(shape): return (lambda s: s[0] * s[1] * (s[2] if len(s) > 2 else 1))(shape) * 2 / 1e9
    old_et = gb((orig_vocab_size, h))
    new_et = gb((new_size, h))
    old_ple = gb((orig_vocab_size, n_layers, ple_h))
    new_ple = gb((new_size, n_layers, ple_h))
    print(f"\n── Size estimate (fp16) ──")
    print(f"  embed_tokens:      {old_et:.2f} GB -> {new_et:.2f} GB")
    print(f"  embed_per_layer:   {old_ple:.2f} GB -> {new_ple:.2f} GB")
    print(f"  lm_head:           {old_et:.2f} GB -> {new_et:.2f} GB")
    print(f"  delta (fp16):      -{(2*(old_et-new_et) + (old_ple-new_ple)):.2f} GB")
    print(f"  after INT8:        approximately half the delta actually shipped")
    print(f"\nNext: QLoRA re-stabilize on the same corpus, then re-run CoreML conversion.")


if __name__ == "__main__":
    main()
