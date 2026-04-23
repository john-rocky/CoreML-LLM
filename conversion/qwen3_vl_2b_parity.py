"""Mac parity / speed bench for the ANE-optimized Qwen3-VL 4B chunks.

Same 3-prompt EN factual / JP coherence / reasoning chain as
`qwen3_vl_4b_parity.py`, but the head chunk now returns `next_token`
directly (int32) instead of fp32 logits, so the Swift-equivalent loop
no longer runs argmax in numpy.

Use this to A/B compare against `qwen3_vl_4b_parity.py` (v1, no ANE
recipe). Both share the embed_weight.bin sidecar; the only difference
is the chunk graph internals.

Usage:
  python qwen3_vl_4b_parity_ane.py \\
      --chunks-dir /tmp/qwen3_vl_4b_ane/qwen3_vl_4b_decode_chunks
"""
from pathlib import Path
import argparse
import time

import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer

from build_qwen3_vl_2b_text_decode_chunks import (
    load_text_config, MODEL_ID,
    EMBED_BIN_NAME, HEAD_CHUNK_NAME, _body_boundaries, NUM_BODY_CHUNKS,
)


PROMPTS = [
    "What is the capital of France?",
    "こんにちは、元気ですか?",
    "If a train leaves at 9:00 going 60 km/h and another leaves at 9:30 going 90 km/h from 150 km away, when do they meet?",
]


def rope_cos_sin_for_position(cfg, position: int):
    head_dim = cfg.head_dim
    theta = cfg.rope_scaling["rope_theta"]
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    angles = position * freqs
    full = np.concatenate([angles, angles])
    cos = np.cos(full).astype(np.float16).reshape(1, 1, head_dim)
    sin = np.sin(full).astype(np.float16).reshape(1, 1, head_dim)
    return cos, sin


def zero_kv_states(cfg, max_seq, start, end):
    shape = (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
    d = {}
    for i in range(start, end):
        d[f"k_{i}"] = np.zeros(shape, dtype=np.float16)
        d[f"v_{i}"] = np.zeros(shape, dtype=np.float16)
    return d


def run_chunked_decode(embed_weight, body_mlms, head_mlm, boundaries, cfg,
                        max_seq, input_ids_list, max_new,
                        eos_tokens=None):
    if eos_tokens is None:
        eos_tokens = {cfg.eos_token_id}
    chunk_states = [zero_kv_states(cfg, max_seq, s, e) for (s, e) in boundaries]

    def _step(tok_id: int, pos: int) -> int:
        hidden = embed_weight[tok_id:tok_id + 1, :].reshape(
            1, 1, -1).astype(np.float16)
        cos_np, sin_np = rope_cos_sin_for_position(cfg, pos)
        pos_np = np.array([float(pos)], dtype=np.float32)

        for ci, (mlm, d) in enumerate(zip(body_mlms, chunk_states)):
            inp = {
                "hidden_in": hidden,
                "position": pos_np, "cos": cos_np, "sin": sin_np,
                **d,
            }
            out = mlm.predict(inp)
            s, e = boundaries[ci]
            for i in range(s, e):
                d[f"k_{i}"] = out[f"new_k_{i}"]
                d[f"v_{i}"] = out[f"new_v_{i}"]
            hidden = out["hidden"].astype(np.float16)
        head_out = head_mlm.predict({"hidden_in": hidden})
        # head returns `next_token` int32 (1, 1) — read directly,
        # no numpy argmax needed.
        return int(head_out["next_token"].flatten()[0])

    last_token = None
    for t, tok_id in enumerate(input_ids_list):
        last_token = _step(tok_id, t)

    generated = []
    next_token = last_token
    S_prompt = len(input_ids_list)
    for step in range(max_new):
        pos = S_prompt + step
        if pos >= max_seq:
            break
        generated.append(next_token)
        if next_token in eos_tokens:
            break
        next_token = _step(next_token, pos)
    return generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--num-chunks", type=int, default=NUM_BODY_CHUNKS)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=80)
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    embed_bin_path = chunks_dir / EMBED_BIN_NAME
    body_paths = [chunks_dir / f"chunk_{i}.mlpackage"
                  for i in range(args.num_chunks)]
    head_path = chunks_dir / f"{HEAD_CHUNK_NAME}.mlpackage"
    for p in [embed_bin_path, head_path, *body_paths]:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    print("loading config + tokenizer...")
    cfg = load_text_config()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    boundaries = _body_boundaries(cfg.num_hidden_layers, args.num_chunks)
    print(f"  body boundaries: {boundaries}")

    print(f"loading embed ({embed_bin_path.stat().st_size/1e6:.0f} MB) + "
          f"{args.num_chunks} body chunks + head (Mac CPU+ANE)...")
    t0 = time.time()
    embed_weight = np.frombuffer(embed_bin_path.read_bytes(),
                                  dtype=np.float16)
    embed_weight = embed_weight.reshape(cfg.vocab_size, cfg.hidden_size)
    body_mlms = [ct.models.MLModel(str(p),
                                    compute_units=ct.ComputeUnit.CPU_AND_NE)
                 for p in body_paths]
    head_mlm = ct.models.MLModel(str(head_path),
                                  compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  loaded in {time.time()-t0:.1f}s")

    for pi, prompt in enumerate(PROMPTS):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc.input_ids[0].tolist()
        if len(ids) > args.max_seq - args.max_new - 1:
            ids = ids[:args.max_seq - args.max_new - 1]
        print(f"\n=== prompt[{pi}]: {prompt!r} (len={len(ids)}) ===")
        t0 = time.time()
        generated = run_chunked_decode(
            embed_weight, body_mlms, head_mlm, boundaries, cfg,
            args.max_seq, ids, args.max_new)
        dt = time.time() - t0
        gen_text = tok.decode(generated, skip_special_tokens=False)
        full_text = tok.decode(ids + generated, skip_special_tokens=False)
        print(f"  generated {len(generated)} tokens in {dt:.1f}s "
              f"({len(generated)/dt:.1f} tok/s)")
        print(f"  full: {full_text!r}")
        print(f"  gen:  {gen_text!r}")


if __name__ == "__main__":
    main()
