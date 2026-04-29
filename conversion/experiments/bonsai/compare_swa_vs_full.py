"""Side-by-side divergence test: SWA W=1024 vs full-attention ctx=4096.

Runs greedy decode on both bundles with the same prompt and prints:
- first divergence position (where top-1 tokens differ)
- per-step top-1 agreement rate
- final decoded outputs for eyeball comparison

Expected:
- positions 0..W-1: 100% agreement (no wraparound yet)
- position >= W: SWA forgets tokens < (pos - W + 1); divergence accumulates
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer


def load_chunks(bundle: Path, cu: ct.ComputeUnit):
    cfg = json.load(open(bundle / "bonsai_1_7b_decode_chunks/model_config.json"))
    chunk_a = ct.models.MLModel(str(bundle / "bonsai_1_7b_decode_chunks/chunk_a.mlpackage"),
                                compute_units=cu)
    chunk_b = ct.models.MLModel(str(bundle / "bonsai_1_7b_decode_chunks/chunk_b.mlpackage"),
                                compute_units=cu)
    return chunk_a, chunk_b, cfg


def build_feeds(cfg: dict, tok_id: int, pos: int) -> dict:
    swa = cfg.get("sliding_window")
    ctx = cfg["context_length"]
    pos_arr = np.array([pos], dtype=np.int32)
    if swa is None:
        L = ctx
        write_slot = pos
        valid_range = range(pos + 1)
    else:
        L = swa
        write_slot = pos % swa
        valid_count = min(pos + 1, swa)
        valid_range = [((pos - i) % swa) for i in range(valid_count)]
    causal = np.full((1, 1, 1, L), -1e4, dtype=np.float16)
    for s in valid_range:
        causal[0, 0, 0, s] = 0.0
    update = np.zeros((1, 1, L, 1), dtype=np.float16)
    update[0, 0, write_slot, 0] = 1.0
    return {
        "input_ids": np.array([[tok_id]], dtype=np.int32),
        "position_ids": pos_arr,
        "causal_mask": causal,
        "update_mask": update,
    }


def step(chunk_a, chunk_b, cfg, state_a, state_b, tok_id: int, pos: int) -> int:
    feed = build_feeds(cfg, tok_id, pos)
    out_a = chunk_a.predict(feed, state=state_a)
    hidden = out_a["hidden"].astype(np.float16)
    feed_b = {**feed, "hidden_in": hidden}
    del feed_b["input_ids"]
    out_b = chunk_b.predict(feed_b, state=state_b)
    return int(out_b["token_id"].item())


def run_bundle(bundle: Path, tok, prompt: str, max_new: int, cu):
    chunk_a, chunk_b, cfg = load_chunks(bundle, cu)
    sa = chunk_a.make_state()
    sb = chunk_b.make_state()
    ids = tok(prompt, return_tensors="np").input_ids[0].tolist()

    nxt = 0
    for i, tid in enumerate(ids):
        nxt = step(chunk_a, chunk_b, cfg, sa, sb, tid, i)
    # After prefill, next model position = len(ids); feed the next token predicted from
    # the last prompt step (nxt) as input for that position, and so on greedily.
    gen: list[int] = []
    cur = len(ids) - 1
    for _ in range(max_new):
        cur += 1
        tok_in = gen[-1] if gen else nxt
        nxt = step(chunk_a, chunk_b, cfg, sa, sb, tok_in, cur)
        gen.append(nxt)
    return ids, gen, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-bundle", required=True, help="non-SWA full-attention bundle dir")
    ap.add_argument("--swa-bundle", required=True, help="SWA bundle dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="Once upon a time there was a small village in the mountains where")
    ap.add_argument("--max-new-tokens", type=int, default=1100)
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    args = ap.parse_args()

    cu = getattr(ct.ComputeUnit, args.compute_units)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"prompt: {args.prompt!r}")

    print("\n=== running FULL (non-SWA) ===")
    t0 = time.time()
    full_prompt, full_gen, full_cfg = run_bundle(Path(args.full_bundle), tok,
                                                 args.prompt, args.max_new_tokens, cu)
    print(f"  done in {time.time()-t0:.1f}s, ctx={full_cfg['context_length']}, "
          f"swa={full_cfg.get('sliding_window')}")

    print("\n=== running SWA ===")
    t0 = time.time()
    swa_prompt, swa_gen, swa_cfg = run_bundle(Path(args.swa_bundle), tok,
                                              args.prompt, args.max_new_tokens, cu)
    print(f"  done in {time.time()-t0:.1f}s, ctx={swa_cfg['context_length']}, "
          f"swa={swa_cfg.get('sliding_window')}")

    W = swa_cfg.get("sliding_window", 0)

    # Find first divergence
    first_div = None
    agree = 0
    for i, (a, b) in enumerate(zip(full_gen, swa_gen)):
        if a == b:
            agree += 1
        elif first_div is None:
            first_div = i

    print(f"\n=== comparison ===")
    print(f"  W={W}, prompt_tokens={len(full_prompt)}, gen_tokens={len(full_gen)}")
    print(f"  first divergence at gen index: "
          f"{first_div if first_div is not None else 'NONE (bit-identical)'}")
    if first_div is not None:
        gen_pos_at_div = len(full_prompt) + first_div
        print(f"  → that's model position {gen_pos_at_div} (W={W}; "
              f"tokens seen in non-SWA but not in SWA at this step: "
              f"{max(0, gen_pos_at_div - W + 1)})")
    print(f"  total agreement: {agree}/{len(full_gen)} "
          f"({100*agree/len(full_gen):.1f}%)")

    # Agreement buckets
    buckets = [(0, W // 2), (W // 2, W), (W, W + W // 2), (W + W // 2, 2 * W)]
    print(f"\n  top-1 agreement by position bucket:")
    for lo, hi in buckets:
        # bucket is model position; gen index = pos - len(prompt)
        lo_i = max(0, lo - len(full_prompt))
        hi_i = min(len(full_gen), hi - len(full_prompt))
        if lo_i >= hi_i:
            continue
        b_agree = sum(1 for a, b in zip(full_gen[lo_i:hi_i], swa_gen[lo_i:hi_i]) if a == b)
        b_total = hi_i - lo_i
        print(f"    pos [{lo:>5}, {hi:>5}): {b_agree}/{b_total} "
              f"({100*b_agree/b_total:.1f}%)")

    print(f"\n=== FULL decoded (first 300 chars) ===")
    print(tok.decode(full_gen)[:300])
    print(f"\n=== SWA decoded (first 300 chars) ===")
    print(tok.decode(swa_gen)[:300])


if __name__ == "__main__":
    main()
