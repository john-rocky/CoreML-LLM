"""Chained decode test for the 2-chunk Bonsai build.

Measures tok/s end-to-end on Mac ANE (and by proxy the iPhone ceiling).
Runs per-token: chunk_a(input_id, pos, ...) → hidden → chunk_b(hidden, pos, ...) → token.
Prefills the prompt then greedily decodes N tokens.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True,
                    help="Dir containing bonsai_1_7b_decode_chunks/{chunk_a,chunk_b}.mlpackage")
    ap.add_argument("--tokenizer", required=True, help="HF model dir for tokenizer")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--compute-units", default="CPU_AND_NE",
                    choices=["CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU", "ALL"])
    args = ap.parse_args()

    bundle = Path(args.bundle).expanduser() / "bonsai_1_7b_decode_chunks"
    cfg = json.load(open(bundle / "model_config.json"))
    ctx = cfg["context_length"]
    swa = cfg.get("sliding_window")  # None = full attention; int = SWA window size
    mask_len = swa if swa is not None else ctx
    print(f"  ctx={ctx}, sliding_window={swa}, attn_len={mask_len}")

    cu = getattr(ct.ComputeUnit, args.compute_units)
    print(f"Loading chunks from {bundle}, compute_units={args.compute_units}")
    t0 = time.time()
    chunk_a = ct.models.MLModel(str(bundle / "chunk_a.mlpackage"), compute_units=cu)
    t_a = time.time() - t0
    t0 = time.time()
    chunk_b = ct.models.MLModel(str(bundle / "chunk_b.mlpackage"), compute_units=cu)
    t_b = time.time() - t0
    print(f"  chunk_a loaded in {t_a:.1f}s, chunk_b in {t_b:.1f}s")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt_ids = tok(args.prompt, return_tensors="np").input_ids[0].tolist()
    print(f"prompt: {args.prompt!r}")
    print(f"  tokens ({len(prompt_ids)}): {prompt_ids}")

    state_a = chunk_a.make_state()
    state_b = chunk_b.make_state()

    def step(tok_id: int, pos: int) -> tuple[int, float]:
        pos_arr = np.array([pos], dtype=np.int32)

        if swa is None:
            # Full-attention: state buffer = ctx. Write at absolute `pos`.
            # Valid positions = [0, pos].
            L = ctx
            write_slot = pos
            valid_range = range(pos + 1)
        else:
            # SWA: state buffer = W. Write at `pos % W` (circular slot). After
            # the first W steps, every slot holds a valid (position-encoded) K/V
            # so attention attends to the whole buffer — ordering doesn't
            # matter because softmax is permutation-invariant and RoPE is
            # already baked into cached K at write time.
            L = swa
            write_slot = pos % swa
            # Early prefill: only slots that have been written are valid.
            # After pos >= W-1, all W slots are valid.
            valid_count = min(pos + 1, swa)
            # Valid slots are the `valid_count` most-recently-written positions,
            # i.e. all slots whose last write was at position > pos - valid_count.
            # For simplicity, during warm-up we mark all W slots valid (including
            # the freshly-overwritten zeros) once we have written >= 1 token;
            # this is equivalent to StreamingLLM without sinks and matches the
            # speed characteristics. For strict correctness during warm-up, use
            # only `valid_count` slots:
            valid_range = [((pos - i) % swa) for i in range(valid_count)]

        causal = np.full((1, 1, 1, L), -1e4, dtype=np.float16)
        for s in valid_range:
            causal[0, 0, 0, s] = 0.0
        update = np.zeros((1, 1, L, 1), dtype=np.float16)
        update[0, 0, write_slot, 0] = 1.0

        feed_a = {
            "input_ids": np.array([[tok_id]], dtype=np.int32),
            "position_ids": pos_arr,
            "causal_mask": causal,
            "update_mask": update,
        }
        out_a = chunk_a.predict(feed_a, state=state_a)
        hidden = out_a["hidden"]

        feed_b = {
            "hidden_in": hidden.astype(np.float16),
            "position_ids": pos_arr,
            "causal_mask": causal,
            "update_mask": update,
        }
        out_b = chunk_b.predict(feed_b, state=state_b)
        return int(out_b["token_id"].item()), float(out_b["token_logit"].item())

    # Prefill: step through prompt, recording per-step time
    prefill_times: list[float] = []
    for pos, tid in enumerate(prompt_ids):
        t1 = time.time()
        next_id, next_logit = step(tid, pos)
        prefill_times.append(time.time() - t1)

    print(f"  prefill {len(prompt_ids)} steps, avg {np.mean(prefill_times)*1000:.1f} ms")
    print(f"  first gen token: {next_id} ({tok.decode([next_id])!r}), "
          f"logit={next_logit:.3f}")

    generated = [next_id]
    decode_times: list[float] = []
    cur = len(prompt_ids) - 1
    for _ in range(args.max_new_tokens - 1):
        cur += 1
        t1 = time.time()
        next_id, next_logit = step(generated[-1], cur)
        decode_times.append(time.time() - t1)
        generated.append(next_id)

    print(f"\ngenerated {len(generated)} tokens: {generated[:20]}...")
    print(f"decoded: {tok.decode(generated)!r}")
    print(f"\noverall:")
    print(f"  prefill: {np.mean(prefill_times)*1000:.1f} ms/tok  "
          f"({1/np.mean(prefill_times):.1f} tok/s)")
    if decode_times:
        print(f"  decode:  {np.mean(decode_times)*1000:.1f} ms/tok  "
              f"({1/np.mean(decode_times):.1f} tok/s)")
        p50 = float(np.median(decode_times))
        p95 = float(np.percentile(decode_times, 95))
        print(f"  decode p50/p95: {p50*1000:.1f} / {p95*1000:.1f} ms/tok")


if __name__ == "__main__":
    main()
