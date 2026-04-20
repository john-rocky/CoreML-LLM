#!/usr/bin/env python3
"""Collect EAGLE-3 training data from the DEPLOYED W4A8 mlmodelc chunks.

Replaces the bf16 PyTorch collector for the iPhone-retrain workflow: the
draft needs to learn the W4A8-quantized target's argmax, not HF fp16 argmax.
A draft trained on bf16 hiddens mismatches the on-device decoder because
the 4-bit palettized weights + ANE fp16 numerics shift the target's argmax.

This script runs the same `chunk{1..4}.mlmodelc` that run on iPhone via
coremltools with `cpuAndNeuralEngine`, walks each corpus sequence with
T=1 decode, and captures `hidden_at_L{8,17,34}` + the target's argmax at
every position. Output is a memmap + manifest compatible with the TTT
trainer (`train_eagle3_ttt.py`).

Usage:
    python conversion/collect_eagle_hidden_states_w4a8.py \
        --chunks /Users/you/Downloads/gemma4-e2b-eagle3-sideload \
        --corpus /path/to/eagle_corpus.jsonl \
        --output /path/to/training_data_w4a8 \
        --max-tokens 500000 \
        --seq-len 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct

# Chunk model keys (same for both mlpackage and mlmodelc).
CHUNKS = ("chunk1", "chunk2", "chunk3", "chunk4")

# Static topology (Gemma 4 E2B; matches the decoder chunks in this repo).
HIDDEN = 1536
PLD = 256                 # per-layer-dim
NUM_LAYERS = 35
VOCAB = 262144
EMBED_SCALE = HIDDEN ** 0.5  # = 39.191835...

# KV cache slot counts for E2B.
C1_SLIDING = 7
C1_FULL = 1
C2_SLIDING = 5
C2_FULL = 2
MAX_HD = 512


def _fp16_to_fp32_scalar(h: int) -> float:
    # Convert a single fp16 half word (as uint16) to python float.
    return float(np.frombuffer(np.array([h], dtype=np.uint16).tobytes(), dtype=np.float16)[0])


class QuantEmbed:
    """Per-token int8 × fp16-scale lookup, matching EmbeddingLookup.swift.

    Dequant formula: fp16_out[i] = int8[tok, i] * (scale[tok] / 127.0) * embed_scale.
    """
    def __init__(self, data_path: Path, scales_path: Path,
                 vocab: int, dim: int, embed_scale: float):
        self.data = np.memmap(data_path, dtype=np.int8,
                              shape=(vocab, dim), mode="r")
        self.scales_fp16 = np.memmap(scales_path, dtype=np.float16,
                                     shape=(vocab,), mode="r")
        self.embed_scale = np.float32(embed_scale)
        self.vocab = vocab
        self.dim = dim

    def lookup(self, tok: int) -> np.ndarray:
        row = self.data[tok].astype(np.float32)
        s = np.float32(self.scales_fp16[tok]) / np.float32(127.0) * self.embed_scale
        return (row * s).astype(np.float16)

    def lookup_raw(self, tok: int) -> np.ndarray:
        # Raw (no embed_scale), for per-layer-raw input.
        row = self.data[tok].astype(np.float32)
        s = np.float32(self.scales_fp16[tok]) / np.float32(127.0)
        return (row * s).astype(np.float16)


class PerLayerRawEmbed:
    """Per-layer-input raw embedding (int8 × fp16-scale) — matches EmbeddingLookup
    with a different input file that was dumped at conversion time."""
    def __init__(self, data_path: Path, scales_path: Path,
                 vocab: int, per_layer_dim: int, num_layers: int):
        dim = per_layer_dim * num_layers
        self.data = np.memmap(data_path, dtype=np.int8,
                              shape=(vocab, dim), mode="r")
        self.scales_fp16 = np.memmap(scales_path, dtype=np.float16,
                                     shape=(vocab,), mode="r")
        self.vocab = vocab
        self.dim = dim

    def lookup(self, tok: int) -> np.ndarray:
        row = self.data[tok].astype(np.float32)
        s = np.float32(self.scales_fp16[tok]) / np.float32(127.0)
        return (row * s).astype(np.float16)


def load_rope_table(path: Path) -> np.ndarray:
    """Load .npy RoPE table. Shape (num_positions, dim)."""
    return np.load(path).astype(np.float16)


def rope_row(table: np.ndarray, position: int, dim: int) -> np.ndarray:
    """Return shape (1, 1, 1, dim) fp16 for the requested position."""
    assert table.shape[1] == dim, f"rope dim mismatch {table.shape[1]} != {dim}"
    row = table[position]  # (dim,)
    return row.reshape(1, 1, 1, dim).astype(np.float16)


def make_causal_mask(position: int, length: int) -> np.ndarray:
    """(1, 1, 1, length) fp16. Slots [0..position] are 0, rest -inf."""
    m = np.full((1, 1, 1, length), np.float16(-65504.0), dtype=np.float16)
    valid = min(position + 1, length)
    m[0, 0, 0, :valid] = np.float16(0.0)
    return m


def make_sliding_mask(position: int, W: int) -> np.ndarray:
    """(1, 1, 1, W) fp16. Slot valid for any `i >= W - min(position+1, W)`.

    Matches Swift's `makeSlidingCausalMask` — the chunk shifts the cache
    left by 1 on every step and writes the current token's K at slot W-1,
    so after the shift-and-write, slots [W - (position+1) .. W-1] hold
    real K/V (tail-aligned). Using `position` (not position+1) here misses
    slot W-2 on every step, zeroing attention except for the current token.
    """
    m = np.full((1, 1, 1, W), np.float16(-65504.0), dtype=np.float16)
    valid = min(position + 1, W)
    start = W - valid
    m[0, 0, 0, start:W] = np.float16(0.0)
    return m


def make_update_mask(position: int, ctx: int) -> np.ndarray:
    """(1, 1, ctx, 1) fp16 — one-hot at `position`, 0 elsewhere. Writes
    new K/V to full-attention cache at that absolute position.
    Named `update_mask` in deployed chunks (vs `update_indicator` in
    verify chunks)."""
    m = np.zeros((1, 1, ctx, 1), dtype=np.float16)
    if 0 <= position < ctx:
        m[0, 0, position, 0] = np.float16(1.0)
    return m


class ChunkRunner:
    """Runs one T=1 decode step through the 4 deployed chunks and returns
    (hidden_at_L8, hidden_at_L17, hidden_at_L34, token_id). Manages the KV
    cache in numpy buffers between calls."""

    def __init__(self, model_dir: Path, ctx: int = 2048, W: int = 512):
        self.ctx = ctx
        self.W = W
        cfg = ct.ComputeUnit.CPU_AND_NE
        print("[Load] Loading chunks with compute_units=CPU_AND_NE ...", flush=True)

        def find_chunk(name: str) -> str:
            # coremltools (Python) only loads .mlpackage; .mlmodelc is device-
            # compiled and lacks Manifest.json. Accept either, prefer mlpackage.
            pkg = model_dir / f"{name}.mlpackage"
            if pkg.exists():
                return str(pkg)
            mlmodelc = model_dir / f"{name}.mlmodelc"
            if mlmodelc.exists():
                raise RuntimeError(
                    f"{name}.mlmodelc found but Python coremltools needs "
                    f"{name}.mlpackage. Point --chunks at the mlpackage "
                    f"build output (e.g. output/eagle3-chunks/).")
            raise FileNotFoundError(f"No {name}.mlpackage or .mlmodelc in {model_dir}")

        t0 = time.time()
        self.c1 = ct.models.MLModel(find_chunk("chunk1"), compute_units=cfg)
        self.c2 = ct.models.MLModel(find_chunk("chunk2"), compute_units=cfg)
        self.c3 = ct.models.MLModel(find_chunk("chunk3"), compute_units=cfg)
        self.c4 = ct.models.MLModel(find_chunk("chunk4"), compute_units=cfg)
        print(f"[Load] chunks loaded in {time.time()-t0:.1f}s", flush=True)

        # KV buffers (numpy fp16, zero-init).
        self.kS1 = np.zeros((C1_SLIDING, 1, W, MAX_HD), dtype=np.float16)
        self.vS1 = np.zeros((C1_SLIDING, 1, W, MAX_HD), dtype=np.float16)
        self.kF1 = np.zeros((C1_FULL, 1, ctx, MAX_HD), dtype=np.float16)
        self.vF1 = np.zeros((C1_FULL, 1, ctx, MAX_HD), dtype=np.float16)
        self.kS2 = np.zeros((C2_SLIDING, 1, W, MAX_HD), dtype=np.float16)
        self.vS2 = np.zeros((C2_SLIDING, 1, W, MAX_HD), dtype=np.float16)
        self.kF2 = np.zeros((C2_FULL, 1, ctx, MAX_HD), dtype=np.float16)
        self.vF2 = np.zeros((C2_FULL, 1, ctx, MAX_HD), dtype=np.float16)

    def reset(self) -> None:
        for b in (self.kS1, self.vS1, self.kF1, self.vF1,
                  self.kS2, self.vS2, self.kF2, self.vF2):
            b.fill(0)

    def step(self, *, hidden_states: np.ndarray, per_layer_raw: np.ndarray,
             position: int,
             cos_s: np.ndarray, sin_s: np.ndarray,
             cos_f: np.ndarray, sin_f: np.ndarray) -> tuple:
        """Run one T=1 decode step. KV buffers are updated in place.
        Returns (h_L8, h_L17, h_L34, token_id) as numpy arrays of shape (hidden,)
        fp16 except token_id which is scalar int32."""
        ctx = self.ctx
        W = self.W
        mask_full = make_causal_mask(position, ctx)
        mask_sliding = make_sliding_mask(position, W)
        update_mask = make_update_mask(position, ctx)

        shared_mask_rope = {
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "update_mask": update_mask,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
        }

        # chunk1
        out1 = self.c1.predict({
            "hidden_states": hidden_states,
            **shared_mask_rope,
            "per_layer_raw": per_layer_raw,
            "K_sliding_in": self.kS1, "V_sliding_in": self.vS1,
            "K_full_in": self.kF1, "V_full_in": self.vF1,
        })
        self.kS1[...] = out1["K_sliding_out"]
        self.vS1[...] = out1["V_sliding_out"]
        self.kF1[...] = out1["K_full_out"]
        self.vF1[...] = out1["V_full_out"]
        h1 = out1["hidden_states_out"]
        plc = out1["per_layer_combined_out"]

        # chunk2 — emits hidden_at_L8 + kv13/kv14 for chunks 3/4.
        out2 = self.c2.predict({
            "hidden_states": h1,
            **shared_mask_rope,
            "per_layer_combined": plc,
            "K_sliding_in": self.kS2, "V_sliding_in": self.vS2,
            "K_full_in": self.kF2, "V_full_in": self.vF2,
        })
        self.kS2[...] = out2["K_sliding_out"]
        self.vS2[...] = out2["V_sliding_out"]
        self.kF2[...] = out2["K_full_out"]
        self.vF2[...] = out2["V_full_out"]
        h2 = out2["hidden_states_out"]
        h_L8 = out2["hidden_at_L8"]
        kv13k = out2["kv13_k"]
        kv13v = out2["kv13_v"]
        kv14k = out2["kv14_k"]
        kv14v = out2["kv14_v"]

        shared_shared = {
            **shared_mask_rope,
            "per_layer_combined": plc,
            "kv13_k": kv13k, "kv13_v": kv13v,
            "kv14_k": kv14k, "kv14_v": kv14v,
        }

        # chunk3 — emits hidden_at_L17.
        out3 = self.c3.predict({
            "hidden_states": h2,
            **shared_shared,
        })
        h3 = out3["hidden_states_out"]
        h_L17 = out3["hidden_at_L17"]

        # chunk4 — emits (token_id, token_logit, hidden_at_L34).
        out4 = self.c4.predict({
            "hidden_states": h3,
            **shared_shared,
        })
        token_id = int(np.array(out4["token_id"]).reshape(-1)[0])
        h_L34 = out4["hidden_at_L34"]

        # Squeeze to (hidden,)
        return (np.asarray(h_L8).reshape(-1).astype(np.float16),
                np.asarray(h_L17).reshape(-1).astype(np.float16),
                np.asarray(h_L34).reshape(-1).astype(np.float16),
                token_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True,
                    help="Dir with chunk{1..4}.mlpackage (e.g. output/eagle3-chunks/). "
                         "coremltools loads only .mlpackage — not the device "
                         "compiled .mlmodelc.")
    ap.add_argument("--assets", type=Path, default=None,
                    help="Dir with embed_tokens_*.bin, cos/sin_*.npy, hf_model/. "
                         "Defaults to the sideload bundle path if omitted; "
                         "usually distinct from --chunks.")
    ap.add_argument("--corpus", type=Path, required=True,
                    help="JSONL with {text: ...} per line (from download_eagle_corpus.py)")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output prefix (no extension); writes <output>.bin + "
                         "<output>.manifest.json")
    ap.add_argument("--max-tokens", type=int, default=500_000,
                    help="Stop after collecting this many (position, hiddens, argmax) tuples.")
    ap.add_argument("--seq-len", type=int, default=512,
                    help="Max tokens per corpus sequence (truncate longer).")
    ap.add_argument("--min-seq", type=int, default=32,
                    help="Skip sequences shorter than this many tokens.")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--W", type=int, default=512)
    args = ap.parse_args()

    assets_dir = args.assets if args.assets is not None else args.chunks
    # Load tokenizer (HF tokenizer.json in hf_model/).
    from transformers import AutoTokenizer  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(str(assets_dir / "hf_model"))

    # Embedding tables.
    print("[Load] Mapping embedding tables ...", flush=True)
    embed = QuantEmbed(
        assets_dir / "embed_tokens_q8.bin",
        assets_dir / "embed_tokens_scales.bin",
        vocab=VOCAB, dim=HIDDEN, embed_scale=EMBED_SCALE)
    ple = PerLayerRawEmbed(
        assets_dir / "embed_tokens_per_layer_q8.bin",
        assets_dir / "embed_tokens_per_layer_scales.bin",
        vocab=VOCAB, per_layer_dim=PLD, num_layers=NUM_LAYERS)

    cos_s_table = load_rope_table(assets_dir / "cos_sliding.npy")
    sin_s_table = load_rope_table(assets_dir / "sin_sliding.npy")
    cos_f_table = load_rope_table(assets_dir / "cos_full.npy")
    sin_f_table = load_rope_table(assets_dir / "sin_full.npy")

    runner = ChunkRunner(args.chunks, ctx=args.ctx, W=args.W)

    # Output memmap with a structured row dtype — one record per collected
    # position. Trainer side can mmap the same dtype and slice freely.
    out_bin = Path(str(args.output) + ".bin")
    out_manifest = Path(str(args.output) + ".manifest.json")
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    row_dtype = np.dtype([
        ("h_low",      np.float16, (HIDDEN,)),
        ("h_mid",      np.float16, (HIDDEN,)),
        ("h_high",     np.float16, (HIDDEN,)),
        ("tok_input",  np.int32),
        ("tok_argmax", np.int32),
    ])
    row_bytes = row_dtype.itemsize
    buf = np.memmap(out_bin, dtype=row_dtype,
                    shape=(args.max_tokens,), mode="w+")

    # Sequence boundaries.
    seq_starts: list[int] = []

    # Corpus.
    print(f"[Data] Reading corpus from {args.corpus}", flush=True)
    texts: list[str] = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            try:
                texts.append(json.loads(line)["text"])
            except Exception:
                continue
    print(f"[Data] {len(texts)} texts in corpus.", flush=True)

    collected = 0
    skipped = 0
    t_start = time.time()

    for text_idx, text in enumerate(texts):
        if collected >= args.max_tokens:
            break
        ids = tokenizer.encode(text, truncation=True, max_length=args.seq_len)
        if len(ids) < args.min_seq:
            skipped += 1
            continue
        # Truncate if this sequence would overrun max_tokens.
        remain = args.max_tokens - collected
        if len(ids) > remain:
            ids = ids[:remain]

        runner.reset()
        seq_starts.append(collected)

        for pos, tok in enumerate(ids):
            # Fetch embed (scaled) + per-layer-raw (unscaled).
            hid = embed.lookup(tok).reshape(1, 1, HIDDEN).astype(np.float16)
            plr = ple.lookup(tok).reshape(1, 1, -1).astype(np.float16)
            cos_s = rope_row(cos_s_table, pos, dim=256)
            sin_s = rope_row(sin_s_table, pos, dim=256)
            cos_f = rope_row(cos_f_table, pos, dim=512)
            sin_f = rope_row(sin_f_table, pos, dim=512)

            h_L8, h_L17, h_L34, argmax_tok = runner.step(
                hidden_states=hid, per_layer_raw=plr, position=pos,
                cos_s=cos_s, sin_s=sin_s, cos_f=cos_f, sin_f=sin_f,
            )

            buf[collected]["h_low"] = h_L8
            buf[collected]["h_mid"] = h_L17
            buf[collected]["h_high"] = h_L34
            buf[collected]["tok_input"] = np.int32(tok)
            buf[collected]["tok_argmax"] = np.int32(argmax_tok)
            collected += 1

            if collected % 500 == 0:
                dt = time.time() - t_start
                rate = collected / dt
                eta_s = (args.max_tokens - collected) / max(rate, 1e-6)
                print(f"  collected={collected}/{args.max_tokens} "
                      f"({rate:.1f} tok/s, ETA {eta_s/60:.0f} min, "
                      f"seqs={text_idx+1}, skipped={skipped})",
                      flush=True)

            if collected >= args.max_tokens:
                break

    dt_total = time.time() - t_start
    print(f"[Done] collected={collected} in {dt_total:.0f}s "
          f"({collected/max(dt_total,1):.1f} tok/s), skipped={skipped}",
          flush=True)

    # Flush and truncate to actual collected count.
    buf.flush()
    del buf
    actual_bytes = collected * row_bytes
    with open(out_bin, "ab") as f:
        pass  # memmap already flushed; truncate via os.truncate below
    os.truncate(out_bin, actual_bytes)

    manifest = {
        "format_version": 1,
        "hidden": HIDDEN,
        "row_bytes": row_bytes,
        "row_layout": [
            {"name": "h_low", "dtype": "float16", "dim": HIDDEN,
             "offset": 0, "nbytes": HIDDEN * 2},
            {"name": "h_mid", "dtype": "float16", "dim": HIDDEN,
             "offset": HIDDEN * 2, "nbytes": HIDDEN * 2},
            {"name": "h_high", "dtype": "float16", "dim": HIDDEN,
             "offset": 2 * HIDDEN * 2, "nbytes": HIDDEN * 2},
            {"name": "tok_input", "dtype": "int32", "dim": 1,
             "offset": 3 * HIDDEN * 2, "nbytes": 4},
            {"name": "tok_argmax", "dtype": "int32", "dim": 1,
             "offset": 3 * HIDDEN * 2 + 4, "nbytes": 4},
        ],
        "num_rows": collected,
        "seq_starts": seq_starts,
        "embed_scale": float(EMBED_SCALE),
        "fusion_layers": [8, 17, 34],
        "source": "W4A8 deployed chunks via coremltools (cpuAndNeuralEngine)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Done] wrote {out_bin} ({actual_bytes/1e6:.1f} MB) + {out_manifest}")


if __name__ == "__main__":
    main()
