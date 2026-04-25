"""Long-range dependency test: does SWA forget setup info >W tokens back?

Prompt structure:
  <SETUP>  : a short memorable fact
  <FILLER> : N tokens of neutral text padding the context to push SETUP out of window
  <TRIGGER>: re-mentions SETUP, asking the model to complete it

If Full retains SETUP (within its ctx-sized attention), its next token after TRIGGER
should match the setup. SWA with W=1024 may have dropped SETUP if (prompt_len +
generated_so_far) - setup_pos > W, and would then continue with something else.

We measure:
  - top-1 agreement between FULL and SWA for the first N generated tokens
  - first divergence
  - whether Full's continuation matches the setup substring (qualitative)
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


def build_feeds(cfg, tok_id: int, pos: int) -> dict:
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


def run_bundle(bundle: Path, tok, prompt_ids: list[int], max_new: int, cu):
    chunk_a, chunk_b, cfg = load_chunks(bundle, cu)
    sa = chunk_a.make_state()
    sb = chunk_b.make_state()

    nxt = 0
    for i, tid in enumerate(prompt_ids):
        nxt = step(chunk_a, chunk_b, cfg, sa, sb, tid, i)
    gen: list[int] = []
    cur = len(prompt_ids) - 1
    for _ in range(max_new):
        cur += 1
        tok_in = gen[-1] if gen else nxt
        nxt = step(chunk_a, chunk_b, cfg, sa, sb, tok_in, cur)
        gen.append(nxt)
    return gen, cfg


# Filler text that doesn't reference the setup. Pure lorem-ipsum-like continuation
# style. Generated from a typical base-model-friendly topic.
FILLER_PARA = (
    "The river flows through the valley, carving paths into the rocks over time. "
    "Birds sing in the trees, their melodies carried by the wind. Farmers tend to "
    "their fields, growing crops that feed the village. In the evening, lanterns "
    "are lit, casting warm light on the cobblestone streets. Children play by the "
    "fountain, laughing and running. The baker opens early every morning, the smell "
    "of fresh bread drifting down the lane. Merchants arrive from distant lands, "
    "bringing goods and stories from faraway places. The old clock tower rings at "
    "noon, marking the time of day. Travelers stop at the inn to rest. "
)


def build_test_prompt(tok, setup: str, trigger: str, min_filler_tokens: int = 1100):
    """Build: <setup> <filler repeated enough> <trigger>, return input_ids."""
    filler = ""
    while len(tok(filler, return_tensors="np").input_ids[0]) < min_filler_tokens:
        filler += FILLER_PARA
    prompt = setup + " " + filler + " " + trigger
    return tok(prompt, return_tensors="np").input_ids[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-bundle", required=True)
    ap.add_argument("--swa-bundle", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    ap.add_argument("--filler-tokens", type=int, default=1100,
                    help="Push setup past W=1024 boundary")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    args = ap.parse_args()

    cu = getattr(ct.ComputeUnit, args.compute_units)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Stable, memorable setup that's easy to verify in continuation.
    setup = "My favorite color is chartreuse, which is a vibrant yellow-green."
    trigger = "To remind you, my favorite color is"

    ids = build_test_prompt(tok, setup, trigger, args.filler_tokens)
    print(f"prompt shape: {len(ids)} tokens "
          f"(~{args.filler_tokens} filler + setup + trigger)")
    setup_ids = tok(setup, return_tensors="np").input_ids[0].tolist()
    trigger_ids = tok(" " + trigger, return_tensors="np").input_ids[0].tolist()
    trigger_starts_at = len(ids) - len(trigger_ids)
    dist_setup_to_end = len(ids) - len(setup_ids)
    print(f"  setup ends around token {len(setup_ids)}, trigger begins around "
          f"token {trigger_starts_at}")
    print(f"  distance from setup to end of prompt ≈ {dist_setup_to_end} tokens "
          f"(> W=1024 means SWA should have forgotten setup)")

    print("\n=== FULL (ctx=4096 full-attn) ===")
    t0 = time.time()
    full_gen, full_cfg = run_bundle(Path(args.full_bundle), tok, ids,
                                    args.max_new_tokens, cu)
    print(f"  done in {time.time()-t0:.1f}s")
    full_text = tok.decode(full_gen)
    print(f"  continuation: {full_text!r}")

    print("\n=== SWA (W=1024) ===")
    t0 = time.time()
    swa_gen, swa_cfg = run_bundle(Path(args.swa_bundle), tok, ids,
                                  args.max_new_tokens, cu)
    print(f"  done in {time.time()-t0:.1f}s")
    swa_text = tok.decode(swa_gen)
    print(f"  continuation: {swa_text!r}")

    # Scoring: does continuation contain "chartreuse"?
    full_recall = "chartreuse" in full_text.lower()
    swa_recall = "chartreuse" in swa_text.lower()

    agree = sum(1 for a, b in zip(full_gen, swa_gen) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(full_gen, swa_gen)) if a != b),
                     None)

    print(f"\n=== comparison ===")
    print(f"  top-1 agreement: {agree}/{len(full_gen)} "
          f"({100*agree/len(full_gen):.1f}%)")
    print(f"  first divergence at gen index: "
          f"{first_div if first_div is not None else 'NONE'}")
    print(f"  Full  recalls 'chartreuse': {full_recall}")
    print(f"  SWA   recalls 'chartreuse': {swa_recall}")
    if full_recall and not swa_recall:
        print(f"  → long-range recall regression confirmed for this prompt")
    elif full_recall and swa_recall:
        print(f"  → both recall: either setup is in window (filler too short) "
              f"or model is robust via other cues")
    elif not full_recall:
        print(f"  → full model didn't recall either; prompt may be too hard "
              f"even for full attention")


if __name__ == "__main__":
    main()
