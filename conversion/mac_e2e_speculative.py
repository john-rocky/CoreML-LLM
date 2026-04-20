#!/usr/bin/env python3
"""End-to-end speculative decoding on Mac, using the EXACT deployed models:
  - chunk{1..4}.mlpackage        (W4A8 decode)
  - verify_chunk{1..4}.mlpackage (W4A8 verify)
  - eagle3_draft.mlpackage
  - eagle3_fusion.mlpackage

Reproduces the iPhone speculative loop on Mac ANE (or CPU) so we can
measure draft accept rate against the actual deployed target, without
needing an iPhone in the loop. If Mac accept ≈ 0% → deployed models are
broken. If Mac accept is 15-30% → iPhone-specific firmware / binary
issue.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))
from collect_eagle_hidden_states_w4a8 import (  # noqa: E402
    ChunkRunner, QuantEmbed, PerLayerRawEmbed,
    load_rope_table, rope_row,
    make_causal_mask, make_sliding_mask,
    HIDDEN, PLD, NUM_LAYERS, VOCAB, EMBED_SCALE,
)


def make_verify_causal_full(position: int, T: int, ctx: int) -> np.ndarray:
    """Mirror Swift's makeVerifyCausalMaskFull. (1,1,T,ctx+T) fp16."""
    last = ctx + T
    m = np.full((1, 1, T, last), np.float16(-65504.0), dtype=np.float16)
    for t in range(T):
        # Cache portion 0..position-1 allowed.
        end_cache = min(position, ctx)
        m[0, 0, t, :end_cache] = 0
        # New positions 0..t allowed within the trailing T slots.
        for j in range(t + 1):
            m[0, 0, t, ctx + j] = 0
    return m


def make_verify_causal_sliding(position: int, T: int, W: int) -> np.ndarray:
    """Mirror Swift's makeVerifyCausalMaskSliding. (1,1,T,W+T) fp16."""
    last = W + T
    m = np.full((1, 1, T, last), np.float16(-65504.0), dtype=np.float16)
    valid = min(position, W)
    start = W - valid
    for t in range(T):
        m[0, 0, t, start:W] = 0
        for j in range(t + 1):
            m[0, 0, t, W + j] = 0
    return m


def build_verify_rope_row(table: np.ndarray, position: int, T: int, dim: int) -> np.ndarray:
    """(1, 1, T, dim) fp16 — rows table[position..position+T-1]."""
    assert table.shape[1] == dim
    out = np.zeros((1, 1, T, dim), dtype=np.float16)
    for t in range(T):
        p = position + t
        if p < table.shape[0]:
            out[0, 0, t] = table[p]
    return out


class VerifyRunner:
    """Runs verify_chunk{1..4} with T=K batched inputs. Read-only KV cache
    (reads runner's kSliding/kFull buffers, does NOT update them).

    Returns per-position target argmax (length K)."""

    def __init__(self, model_dir: Path):
        cfg = ct.ComputeUnit.CPU_AND_NE
        self.v1 = ct.models.MLModel(str(model_dir / "verify_chunk1.mlpackage"), compute_units=cfg)
        self.v2 = ct.models.MLModel(str(model_dir / "verify_chunk2.mlpackage"), compute_units=cfg)
        self.v3 = ct.models.MLModel(str(model_dir / "verify_chunk3.mlpackage"), compute_units=cfg)
        self.v4 = ct.models.MLModel(str(model_dir / "verify_chunk4.mlpackage"), compute_units=cfg)

    def run(self, *, decode_runner: ChunkRunner, candidates: list,
            position: int, embed: QuantEmbed, ple: PerLayerRawEmbed,
            cos_s_tbl: np.ndarray, sin_s_tbl: np.ndarray,
            cos_f_tbl: np.ndarray, sin_f_tbl: np.ndarray) -> np.ndarray:
        K = len(candidates)
        ctx = decode_runner.ctx
        W = decode_runner.W

        # Build batched hidden + per_layer_raw: stack (1, K, hidden)
        hidden_states = np.zeros((1, K, HIDDEN), dtype=np.float16)
        per_layer_raw = np.zeros((1, K, NUM_LAYERS * PLD), dtype=np.float16)
        for k, tok in enumerate(candidates):
            hidden_states[0, k] = embed.lookup(int(tok))
            per_layer_raw[0, k] = ple.lookup(int(tok))

        mask_full = make_verify_causal_full(position, K, ctx)
        mask_sliding = make_verify_causal_sliding(position, K, W)
        cos_s = build_verify_rope_row(cos_s_tbl, position, K, 256)
        sin_s = build_verify_rope_row(sin_s_tbl, position, K, 256)
        cos_f = build_verify_rope_row(cos_f_tbl, position, K, 512)
        sin_f = build_verify_rope_row(sin_f_tbl, position, K, 512)

        shared = {
            "causal_mask_full": mask_full,
            "causal_mask_sliding": mask_sliding,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_f": cos_f, "sin_f": sin_f,
        }

        out1 = self.v1.predict({
            "hidden_states": hidden_states,
            **shared,
            "per_layer_raw": per_layer_raw,
            "K_sliding_in": decode_runner.kS1,
            "V_sliding_in": decode_runner.vS1,
            "K_full_in": decode_runner.kF1,
            "V_full_in": decode_runner.vF1,
        })
        h1 = out1["hidden_states_out"]
        plc = out1["per_layer_combined_out"]

        out2 = self.v2.predict({
            "hidden_states": h1,
            **shared,
            "per_layer_combined": plc,
            "K_sliding_in": decode_runner.kS2,
            "V_sliding_in": decode_runner.vS2,
            "K_full_in": decode_runner.kF2,
            "V_full_in": decode_runner.vF2,
        })
        h2 = out2["hidden_states_out"]
        kv13k = out2["kv13_k_out"]
        kv13v = out2["kv13_v_out"]
        kv14k = out2["kv14_k_out"]
        kv14v = out2["kv14_v_out"]

        shared_kv = {
            **shared,
            "per_layer_combined": plc,
            "kv13_k": kv13k, "kv13_v": kv13v,
            "kv14_k": kv14k, "kv14_v": kv14v,
        }

        out3 = self.v3.predict({"hidden_states": h2, **shared_kv})
        h3 = out3["hidden_states_out"]

        out4 = self.v4.predict({"hidden_states": h3, **shared_kv})
        tok_ids = np.asarray(out4["token_ids"]).reshape(-1).astype(np.int64)
        return tok_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True,
                    help="Dir with {decode,verify}_chunk{1..4}.mlpackage + "
                         "eagle3_{draft,fusion}.mlpackage")
    ap.add_argument("--assets", type=Path, required=True,
                    help="Dir with embed_tokens_*.bin, cos/sin_*.npy, hf_model/")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--max-new", type=int, default=32,
                    help="How many tokens to generate via spec burst")
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()

    # Use the low-level tokenizers library directly — transformers' fast
    # tokenizer path for Gemma currently crashes in recent builds (dict/list
    # confusion in _set_model_specific_special_tokens). The raw tokenizer.json
    # works fine and produces the same IDs.
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.assets / "hf_model" / "tokenizer.json"))
    if args.corpus is not None:
        with open(args.corpus) as f:
            prompt_text = json.loads(f.readline())["text"]
    else:
        prompt_text = args.prompt or "Tell me about the history of Japan briefly."
    prompt_ids = tokenizer.encode(prompt_text).ids
    # Keep prompt modest so we stay in context.
    prompt_ids = prompt_ids[:96]
    print(f"[Prompt] {len(prompt_ids)} tokens")

    # Embed + RoPE tables + decode runner.
    embed = QuantEmbed(
        args.assets / "embed_tokens_q8.bin",
        args.assets / "embed_tokens_scales.bin",
        vocab=VOCAB, dim=HIDDEN, embed_scale=EMBED_SCALE)
    ple = PerLayerRawEmbed(
        args.assets / "embed_tokens_per_layer_q8.bin",
        args.assets / "embed_tokens_per_layer_scales.bin",
        vocab=VOCAB, per_layer_dim=PLD, num_layers=NUM_LAYERS)
    cos_s_tbl = load_rope_table(args.assets / "cos_sliding.npy")
    sin_s_tbl = load_rope_table(args.assets / "sin_sliding.npy")
    cos_f_tbl = load_rope_table(args.assets / "cos_full.npy")
    sin_f_tbl = load_rope_table(args.assets / "sin_full.npy")

    decoder = ChunkRunner(args.chunks, ctx=2048, W=512)
    verifier = VerifyRunner(args.chunks)

    # draft + fusion.
    cfg = ct.ComputeUnit.CPU_AND_NE
    print("[Load] draft / fusion ...")
    fusion = ct.models.MLModel(str(args.chunks / "eagle3_fusion.mlpackage"), compute_units=cfg)
    draft = ct.models.MLModel(str(args.chunks / "eagle3_draft.mlpackage"), compute_units=cfg)

    # Teacher-forced prefill via T=1 decode. Captures tTokNext along the way.
    print(f"[Prefill] running T=1 decode over {len(prompt_ids)} prompt tokens ...")
    t0 = time.time()
    last_argmax = None
    for pos, tok in enumerate(prompt_ids):
        hid = embed.lookup(int(tok)).reshape(1, 1, HIDDEN).astype(np.float16)
        plr = ple.lookup(int(tok)).reshape(1, 1, -1).astype(np.float16)
        cos_s = rope_row(cos_s_tbl, pos, dim=256)
        sin_s = rope_row(sin_s_tbl, pos, dim=256)
        cos_f = rope_row(cos_f_tbl, pos, dim=512)
        sin_f = rope_row(sin_f_tbl, pos, dim=512)
        h_L8, h_L17, h_L34, argmax = decoder.step(
            hidden_states=hid, per_layer_raw=plr, position=pos,
            cos_s=cos_s, sin_s=sin_s, cos_f=cos_f, sin_f=sin_f)
        last_argmax = argmax
    print(f"[Prefill] done in {time.time()-t0:.1f}s  last_argmax={last_argmax}")

    # ── Speculative loop ──
    # currentPosition is the position where the NEXT token will land.
    current_position = len(prompt_ids)
    tTokNext = int(last_argmax)

    # Snapshot the last-seen h_L8/L17/L34 from the most recent decode step,
    # same as ChunkedEngine does (engine.lastHiddenAtL8/L17/L34).
    last_h_low, last_h_mid, last_h_high = h_L8, h_L17, h_L34

    generated_via_spec = 0
    total_matched = 0
    total_proposals = 0
    burst = 0
    emitted = []
    t_burst = time.time()

    while generated_via_spec < args.max_new:
        burst += 1
        # 1. Fuse last hiddens.
        fus_out = fusion.predict({
            "h_low":  last_h_low.reshape(1, 1, HIDDEN),
            "h_mid":  last_h_mid.reshape(1, 1, HIDDEN),
            "h_high": last_h_high.reshape(1, 1, HIDDEN),
        })
        h_fused = fus_out["h_fused"]

        # 2. K draft steps autoregressively.
        h_prev = h_fused
        e_next = embed.lookup(int(tTokNext)).reshape(1, 1, HIDDEN).astype(np.float16)
        proposals = []
        for _ in range(args.K):
            d_out = draft.predict({"h_prev": h_prev, "e_next": e_next})
            pred = int(np.asarray(d_out["token"]).reshape(-1)[0])
            proposals.append(pred)
            h_prev = d_out["h_out"]
            e_next = embed.lookup(pred).reshape(1, 1, HIDDEN).astype(np.float16)

        # 3. Verify on target (read-only KV).
        verify_tokens = [tTokNext] + proposals[:-1]
        target_argmax = verifier.run(
            decode_runner=decoder, candidates=verify_tokens,
            position=current_position, embed=embed, ple=ple,
            cos_s_tbl=cos_s_tbl, sin_s_tbl=sin_s_tbl,
            cos_f_tbl=cos_f_tbl, sin_f_tbl=sin_f_tbl)

        # 4. Accept prefix.
        matched = 0
        accepted = [tTokNext]
        for k in range(args.K):
            if int(proposals[k]) == int(target_argmax[k]):
                accepted.append(int(proposals[k]))
                matched += 1
            else:
                accepted.append(int(target_argmax[k]))
                break

        total_matched += matched
        total_proposals += args.K

        # 5. Commit accepted via plain T=1 decode for each accepted token.
        # This mirrors the T=1 replay path (ignore Blocker 2 for this test).
        for i, tok in enumerate(accepted):
            hid = embed.lookup(int(tok)).reshape(1, 1, HIDDEN).astype(np.float16)
            plr = ple.lookup(int(tok)).reshape(1, 1, -1).astype(np.float16)
            pos = current_position + i
            cos_s = rope_row(cos_s_tbl, pos, dim=256)
            sin_s = rope_row(sin_s_tbl, pos, dim=256)
            cos_f = rope_row(cos_f_tbl, pos, dim=512)
            sin_f = rope_row(sin_f_tbl, pos, dim=512)
            h_L8, h_L17, h_L34, argmax = decoder.step(
                hidden_states=hid, per_layer_raw=plr, position=pos,
                cos_s=cos_s, sin_s=sin_s, cos_f=cos_f, sin_f=sin_f)
        current_position += len(accepted)
        tTokNext = int(argmax)
        last_h_low, last_h_mid, last_h_high = h_L8, h_L17, h_L34

        emitted.extend(accepted)
        generated_via_spec += len(accepted)
        accept_rate = total_matched / max(1, total_proposals)
        print(f"[Burst {burst}] matched={matched}/{args.K}  emitted={len(accepted)}  "
              f"cum_accept={accept_rate*100:.1f}%  proposals={proposals}  "
              f"target_argmax={list(target_argmax)}")

    dt = time.time() - t_burst
    text = tokenizer.decode(emitted)
    print(f"\n[Done] {burst} bursts, {generated_via_spec} emitted in {dt:.1f}s  "
          f"accept_rate={total_matched/max(1,total_proposals)*100:.1f}%")
    print(f"[Text] {text!r}")


if __name__ == "__main__":
    main()
