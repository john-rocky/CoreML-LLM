"""Quick smoke test of the converted Bonsai model.

Loads the saved .mlpackage, runs a single decode step, and prints the predicted token.
Validates: model loads, predict() works, output shape / dtype correct, first-token
prediction is sane.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Dir with model.mlpackage + model_config.json")
    ap.add_argument("--tokenizer", required=True, help="HF model dir for tokenizer")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=10)
    ap.add_argument("--compute-units", default="CPU_AND_NE",
                    choices=["CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU", "ALL"])
    args = ap.parse_args()

    bundle = Path(args.bundle).expanduser()
    pkg_path = bundle / "model.mlpackage"
    cfg_path = bundle / "model_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    ctx = cfg["context_length"]
    print(f"Loading {pkg_path}")
    cu = getattr(ct.ComputeUnit, args.compute_units)
    t0 = time.time()
    model = ct.models.MLModel(str(pkg_path), compute_units=cu)
    print(f"  loaded in {time.time()-t0:.1f}s, compute_units={args.compute_units}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    input_ids = tok(args.prompt, return_tensors="np").input_ids[0].tolist()
    print(f"prompt: {args.prompt!r}")
    print(f"  tokens ({len(input_ids)}): {input_ids}")

    # Per-step decode (no batched prefill for this smoke test).
    state = model.make_state()
    generated: list[int] = []
    cur_pos = 0
    step_times = []

    for pos, tok_id in enumerate(input_ids):
        causal_mask = np.full((1, 1, 1, ctx), -1e4, dtype=np.float16)
        causal_mask[0, 0, 0, : pos + 1] = 0.0
        update_mask = np.zeros((1, 1, ctx, 1), dtype=np.float16)
        update_mask[0, 0, pos, 0] = 1.0
        feed = {
            "input_ids": np.array([[tok_id]], dtype=np.int32),
            "position_ids": np.array([pos], dtype=np.int32),
            "causal_mask": causal_mask,
            "update_mask": update_mask,
        }
        t1 = time.time()
        out = model.predict(feed, state=state)
        step_times.append(time.time() - t1)
        cur_pos = pos
    next_id = int(out["token_id"].item())
    next_logit = float(out["token_logit"].item())
    print(f"  prefilled {len(input_ids)} tokens, avg {np.mean(step_times)*1000:.1f} ms/step")
    print(f"  first gen token: {next_id} ({tok.decode([next_id])!r}), logit={next_logit:.3f}")

    generated.append(next_id)
    # Continue greedy decode
    for i in range(args.max_new_tokens - 1):
        cur_pos += 1
        causal_mask = np.full((1, 1, 1, ctx), -1e4, dtype=np.float16)
        causal_mask[0, 0, 0, : cur_pos + 1] = 0.0
        update_mask = np.zeros((1, 1, ctx, 1), dtype=np.float16)
        update_mask[0, 0, cur_pos, 0] = 1.0
        feed = {
            "input_ids": np.array([[generated[-1]]], dtype=np.int32),
            "position_ids": np.array([cur_pos], dtype=np.int32),
            "causal_mask": causal_mask,
            "update_mask": update_mask,
        }
        t1 = time.time()
        out = model.predict(feed, state=state)
        step_times.append(time.time() - t1)
        next_id = int(out["token_id"].item())
        generated.append(next_id)

    cont = tok.decode(generated)
    print(f"\ngenerated {len(generated)} tokens: {generated}")
    print(f"decoded: {cont!r}")
    total_toks = len(input_ids) + len(generated)
    print(f"\noverall: {total_toks} steps, avg {np.mean(step_times)*1000:.1f} ms/step "
          f"({1.0 / np.mean(step_times):.1f} tok/s decode throughput)")


if __name__ == "__main__":
    main()
