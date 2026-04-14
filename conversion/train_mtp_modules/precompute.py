#!/usr/bin/env python3
"""Path C training Phase 1: precompute L34 hidden states for a corpus.

For each sequence in the dataset:
  1. Run frozen HF Gemma 4 trunk forward.
  2. Extract L34 raw hidden state (shape (T, H)) at every position.
  3. Save (token_ids, L34_hidden) to disk in a memory-mapped format.

The training loop later consumes these cached pairs without touching the
trunk. This makes training ~100× faster than trunk-in-the-loop training,
and lets us keep the trunk on a single GPU while multi-GPU-training the
tiny MTP modules on a separate GPU.

Output format (per shard):
  {shard_idx}.tokens.bin   — int32 (N, T)  packed
  {shard_idx}.hidden.bin   — fp16 (N, T, H) packed
  {shard_idx}.meta.json    — shapes + stats

Target: 5M tokens on A100 in under 3 hours (Gate 1.5).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def iter_hf_dataset(name_or_path: str, streaming: bool = True, max_samples: int = None):
    """Yield text samples from an HF dataset. Supports streaming for large sets."""
    from datasets import load_dataset
    if name_or_path.startswith("shard:"):
        # Shard spec: "shard:dataset_name:split:start:end"
        parts = name_or_path.split(":")
        ds = load_dataset(parts[1], split=parts[2], streaming=streaming)
        start, end = int(parts[3]), int(parts[4])
        count = 0
        for i, ex in enumerate(ds):
            if i < start: continue
            if i >= end: break
            yield ex
            count += 1
            if max_samples and count >= max_samples: break
        return

    # Common chat/code datasets. Open (no gating) unless noted.
    dataset_presets = {
        # Guaranteed-open (good for first runs)
        "wikitext":    ("Salesforce/wikitext", "train", ["text"]),  # tiny, always works
        "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "train", ["text"]),  # high quality EN
        "oasst1":      ("OpenAssistant/oasst1", "train", ["text"]),  # EN chat
        "stack-small": ("bigcode/the-stack-smol", "train", ["content"]),  # code
        "codealpaca":  ("sahil2801/CodeAlpaca-20k", "train", ["instruction", "input", "output"]),
        # Gated/limited — fall back if these fail
        "lmsys-chat":  ("lmsys/lmsys-chat-1m", "train", ["conversation"]),
        "c4-en":       ("allenai/c4", "train", ["text"]),  # needs subset 'en'
        "c4-ja":       ("allenai/c4", "train", ["text"]),  # needs subset 'ja'
        "sharegpt-ja": ("philschmid/sharegpt-raw", "train", ["conversations"]),
    }
    if name_or_path in dataset_presets:
        repo, split, text_keys = dataset_presets[name_or_path]
        kwargs = {"streaming": streaming}
        if name_or_path == "c4-en":
            kwargs["name"] = "en"
        if name_or_path == "c4-ja":
            kwargs["name"] = "ja"
        if name_or_path == "wikitext":
            kwargs["name"] = "wikitext-103-raw-v1"
        if name_or_path == "oasst1":
            # oasst1 field is "text" but only if role=='assistant'; just take all text
            pass
        ds = load_dataset(repo, split=split, **kwargs)
        count = 0
        for ex in ds:
            # Collect text from configured fields
            text_parts = []
            for key in text_keys:
                val = ex.get(key)
                if isinstance(val, str):
                    text_parts.append(val)
                elif isinstance(val, list):
                    # conversation format
                    for turn in val:
                        if isinstance(turn, dict):
                            text_parts.append(turn.get("content") or turn.get("value") or "")
                        elif isinstance(turn, str):
                            text_parts.append(turn)
            text = "\n".join(t for t in text_parts if t)
            if text:
                yield {"text": text}
                count += 1
                if max_samples and count >= max_samples: break
        return

    # Fallback: raw path or HF repo
    ds = load_dataset(name_or_path, split="train", streaming=streaming)
    for ex in ds:
        yield ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str,
                    default=os.path.expanduser(
                        "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/"
                        "4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
                    help="HF Gemma 4 checkpoint directory")
    ap.add_argument("--dataset", type=str, nargs="+",
                    default=["lmsys-chat", "stack-small"],
                    help="Dataset preset name(s) or HF repo path(s). Multiple = concat.")
    ap.add_argument("--samples-per-dataset", type=int, default=500,
                    help="Max samples per dataset (for 5M token target across 2 datasets).")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--output-dir", type=str, default="./output/mtp_train_cache")
    ap.add_argument("--shard-size", type=int, default=256,
                    help="Sequences per shard on disk.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}  Dtype: {args.dtype}")
    print(f"Loading HF Gemma 4 from {args.hf_dir}...")
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    hf = Gemma4ForConditionalGeneration.from_pretrained(
        args.hf_dir, torch_dtype=dtype, device_map=args.device,
    ).eval()
    lm = hf.model.language_model
    H = lm.config.hidden_size

    total_tokens = 0
    shard_idx = 0
    buf_tokens = []
    buf_hiddens = []

    def flush():
        nonlocal shard_idx, buf_tokens, buf_hiddens
        if not buf_tokens: return
        tokens = np.stack(buf_tokens).astype(np.int32)
        hiddens = np.stack(buf_hiddens).astype(np.float16)  # always store fp16 to save disk
        np.save(out_dir / f"shard_{shard_idx:04d}.tokens.npy", tokens)
        np.save(out_dir / f"shard_{shard_idx:04d}.hidden.npy", hiddens)
        meta = {
            "shard_idx": shard_idx,
            "num_sequences": tokens.shape[0],
            "seq_len": tokens.shape[1],
            "hidden_size": hiddens.shape[-1],
        }
        with open(out_dir / f"shard_{shard_idx:04d}.meta.json", "w") as f:
            json.dump(meta, f)
        print(f"  Flushed shard {shard_idx}: {tokens.shape} tokens, "
              f"{hiddens.shape} hiddens ({hiddens.nbytes / 1e6:.1f} MB)")
        shard_idx += 1
        buf_tokens = []
        buf_hiddens = []

    print(f"\nDatasets: {args.dataset}")
    # Accumulate tokens across ALL examples, THEN chunk into seq_len windows.
    # Per-example chunking fails for short entries like wikitext (lines, not docs).
    bos = tok.bos_token_id
    all_tokens = [bos] if bos is not None else []
    total_samples = 0
    for ds_name in args.dataset:
        print(f"  Streaming {ds_name} (max {args.samples_per_dataset} samples)...")
        ds_tokens_before = len(all_tokens)
        ds_samples = 0
        for ex in iter_hf_dataset(ds_name, streaming=True, max_samples=args.samples_per_dataset):
            text = ex.get("text") or ex.get("content", "")
            if not text:
                continue
            ids = tok(text, add_special_tokens=False).input_ids
            all_tokens.extend(ids)
            ds_samples += 1
        ds_tokens = len(all_tokens) - ds_tokens_before
        total_samples += ds_samples
        print(f"    {ds_name}: {ds_samples} samples, {ds_tokens:,} tokens")

    # Chunk into seq_len windows
    all_chunks = []
    for i in range(0, len(all_tokens) - args.seq_len + 1, args.seq_len):
        all_chunks.append(all_tokens[i:i + args.seq_len])

    print(f"\nTotal: {total_samples} samples → {len(all_tokens):,} tokens → "
          f"{len(all_chunks)} chunks of {args.seq_len}")

    with torch.inference_mode():
        for i, chunk in enumerate(all_chunks):
            ids = torch.tensor([chunk], dtype=torch.long, device=args.device)
            out = lm(input_ids=ids, output_hidden_states=True, use_cache=False)
            # L34 raw hidden = hidden_states[-2]  (last is post-norm; -2 is raw L34 output)
            l34 = out.hidden_states[-2][0].to(torch.float16).cpu().numpy()  # (T, H)
            buf_tokens.append(np.array(chunk, dtype=np.int32))
            buf_hiddens.append(l34)
            total_tokens += len(chunk)

            if len(buf_tokens) >= args.shard_size:
                flush()

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(all_chunks)} chunks, "
                      f"{total_tokens:,} tokens")

    flush()

    print(f"\nDONE. Total tokens: {total_tokens:,} across {shard_idx} shards.")
    print(f"Storage: {sum(f.stat().st_size for f in out_dir.rglob('*')) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
