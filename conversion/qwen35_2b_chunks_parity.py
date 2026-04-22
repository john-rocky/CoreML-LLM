"""Mac parity gate for Qwen3.5-2B 5-chunk INT8 decode
(1 embed chunk + N body/tail chunks).

Runs the 3-prompt generated-text test from the handoff doc §3.5:
  - factual: "What is the capital of France?" — must answer Paris and
    not drift into demonstrable falsehoods in the first 60 tokens.
  - Japanese greeting: "こんにちは" — must stay in Japanese, no
    language codeswitch.
  - Japanese recipe: "美味しい餃子のレシピを教えて" — same.

For each prompt: generate ≤ 60 tokens via chained N+1-chunk greedy
decode on Mac ANE; print the produced text. A human verifies the
acceptance criteria.

Usage:
  python qwen35_2b_chunks_parity.py \\
      --chunks-dir /tmp/qwen35_2b_chunks/qwen3_5_2b_decode_chunks
"""
from pathlib import Path
import argparse
import time

import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer

from build_qwen35_2b_decode_chunks import (
    load_2b_text_config, MODEL_ID,
    EMBED_BIN_NAME, BODY_CHUNK_NAMES, _chunk_boundaries,
)
from test_qwen3_5_full_decode_trace import make_zero_states, MAX_SEQ
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding


PROMPTS = [
    "What is the capital of France?",
    "こんにちは、元気ですか?",
    "美味しい餃子のレシピを教えて。",
]


def run_chunked_decode(
    embed_weight, body_mlms, boundaries, cfg, max_seq, input_ids_list, max_new,
    eos_tokens=(248044, 248045, 248046),
):
    """Greedy chained decode: Swift-equivalent mmap embed lookup →
    N body/tail chunks. Returns generated_ids.

    `embed_weight` is a (vocab, hidden) numpy fp16 array (the raw
    embed_weight.bin read back), mirroring what Swift mmaps on device.
    """
    assert len(body_mlms) == len(boundaries)
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    states = make_zero_states(cfg, max_seq)

    # Per-body-chunk state dicts
    chunk_dicts = []
    for (s, e) in boundaries:
        d = {}
        for i in range(s, e):
            d[f"state_{i}_a"] = states[2 * i].numpy().astype(np.float16)
            d[f"state_{i}_b"] = states[2 * i + 1].numpy().astype(np.float16)
        chunk_dicts.append(d)

    def _step(tok_id: int, pos: int):
        # Swift-equivalent embed lookup: single row copy from mmap'd file.
        hidden = embed_weight[tok_id:tok_id + 1, :].reshape(1, 1, -1).astype(np.float16)

        # Rotary for this position
        pos_ids = torch.tensor([[pos]], dtype=torch.long)
        dummy = torch.zeros(1, 1, cfg.hidden_size)
        with torch.no_grad():
            c_t, s_t = rot(dummy, pos_ids)
        cos_np = c_t.numpy().astype(np.float16)
        sin_np = s_t.numpy().astype(np.float16)
        pos_np = np.array([float(pos)], dtype=np.float32)

        last_out = None
        for ci, (mlm, d) in enumerate(zip(body_mlms, chunk_dicts)):
            inp = {
                "hidden_in": hidden,
                "position": pos_np, "cos": cos_np, "sin": sin_np,
                **d,
            }
            out = mlm.predict(inp)
            last_out = out
            s, e = boundaries[ci]
            for i in range(s, e):
                d[f"state_{i}_a"] = out[f"new_state_{i}_a"]
                d[f"state_{i}_b"] = out[f"new_state_{i}_b"]
            # Non-tail chunks emit "hidden" for the next chunk; tail emits logits
            hidden = (out["hidden"] if "hidden" in out else None)
            if hidden is not None:
                hidden = hidden.astype(np.float16)
        return last_out["logits"][0, 0]

    # Recurrent prefill
    last_logits = None
    for t, tok_id in enumerate(input_ids_list):
        last_logits = _step(tok_id, t)

    # Greedy decode
    generated = []
    next_token = int(np.argmax(last_logits))
    S_prompt = len(input_ids_list)
    for step in range(max_new):
        pos = S_prompt + step
        if pos >= max_seq:
            break
        generated.append(next_token)
        if next_token in eos_tokens:
            break
        logits = _step(next_token, pos)
        next_token = int(np.argmax(logits))
    return generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--num-chunks", type=int, default=4,
                    help="number of body/tail chunks (excluding embed)")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--max-new", type=int, default=60)
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    embed_bin_path = chunks_dir / EMBED_BIN_NAME
    body_paths = [chunks_dir / f"{name}.mlpackage"
                  for name in BODY_CHUNK_NAMES[:args.num_chunks]]
    for p in [embed_bin_path, *body_paths]:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    print("loading config + tokenizer...")
    cfg = load_2b_text_config()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    boundaries = _chunk_boundaries(cfg.num_hidden_layers, args.num_chunks)
    print(f"  body boundaries: {boundaries}")

    print(f"loading embed_weight.bin ({embed_bin_path.stat().st_size / 1e6:.0f} MB) "
          f"+ {args.num_chunks} chunks (Mac CPU+ANE)...")
    t0 = time.time()
    # Read raw fp16 embed into (vocab, hidden) numpy array — Swift does the
    # same via mmap + pointer arithmetic.
    embed_weight = np.frombuffer(embed_bin_path.read_bytes(), dtype=np.float16)
    embed_weight = embed_weight.reshape(cfg.vocab_size, cfg.hidden_size)
    body_mlms = [ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE)
                 for p in body_paths]
    print(f"  loaded in {time.time()-t0:.1f}s")

    for pi, prompt in enumerate(PROMPTS):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc.input_ids[0].tolist()
        if len(ids) > args.max_seq - args.max_new - 1:
            ids = ids[:args.max_seq - args.max_new - 1]
        print(f"\n=== prompt[{pi}]: {prompt!r} (len={len(ids)}) ===")
        t0 = time.time()
        generated = run_chunked_decode(
            embed_weight, body_mlms, boundaries, cfg, args.max_seq, ids, args.max_new)
        dt = time.time() - t0
        gen_text = tok.decode(generated, skip_special_tokens=False)
        full_text = tok.decode(ids + generated, skip_special_tokens=False)
        print(f"  generated {len(generated)} tokens in {dt:.1f}s "
              f"({len(generated)/dt:.1f} tok/s)")
        print(f"  full: {full_text!r}")
        print(f"  gen:  {gen_text!r}")


if __name__ == "__main__":
    main()
