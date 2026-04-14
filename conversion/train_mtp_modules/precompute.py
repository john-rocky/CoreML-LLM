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


def make_collator(tokenizer, seq_len: int):
    def collate(batch):
        texts = [b.get("text") or b.get("content", "") for b in batch]
        # Tokenize without padding, we'll pack
        all_ids = []
        for t in texts:
            ids = tokenizer(t, add_special_tokens=False).input_ids
            all_ids.extend(ids)
        # Prepend BOS
        bos = tokenizer.bos_token_id
        if bos is not None and (len(all_ids) == 0 or all_ids[0] != bos):
            all_ids = [bos] + all_ids
        # Chunk into seq_len windows
        chunks = []
        for i in range(0, len(all_ids) - seq_len + 1, seq_len):
            chunk = all_ids[i:i + seq_len]
            chunks.append(chunk)
        return chunks
    return collate


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

    # Common chat/code datasets
    dataset_presets = {
        "lmsys-chat": ("lmsys/lmsys-chat-1m", "train", ["conversation"]),
        "oasst-ja": ("kunishou/oasst1-89k-ja", "train", ["text", "conversation"]),
        "sharegpt-ja": ("philschmid/sharegpt-raw", "train", ["conversations"]),
        "c4-en": ("allenai/c4", "train", ["text"]),
        "c4-ja": ("allenai/c4", "train", ["text"]),
        "codealpaca": ("sahil2801/CodeAlpaca-20k", "train", ["instruction", "input", "output"]),
        "stack-small": ("bigcode/the-stack-smol", "train", ["content"]),
    }
    if name_or_path in dataset_presets:
        repo, split, text_keys = dataset_presets[name_or_path]
        kwargs = {"streaming": streaming}
        if name_or_path == "c4-en":
            kwargs["name"] = "en"
        if name_or_path == "c4-ja":
            kwargs["name"] = "ja"
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

    # Tokenize + shard
    collator = make_collator(tok, args.seq_len)

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
    all_chunks = []
    for ds_name in args.dataset:
        print(f"  Streaming {ds_name} (max {args.samples_per_dataset} samples)...")
        ds_chunks = 0
        for ex in iter_hf_dataset(ds_name, streaming=True, max_samples=args.samples_per_dataset):
            chunks = collator([ex])
            all_chunks.extend(chunks)
            ds_chunks += len(chunks)
        print(f"    {ds_name}: {ds_chunks} chunks ({ds_chunks * args.seq_len:,} tokens)")

    print(f"\nTotal chunks to process: {len(all_chunks)} ({len(all_chunks) * args.seq_len:,} tokens)")

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
