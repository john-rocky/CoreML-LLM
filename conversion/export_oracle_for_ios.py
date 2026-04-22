"""Export Phase 1 oracle as a compact JSON bundle for iOS drift testing.

Includes per-prompt input_ids, top-1 next-token id, and the last-position logits
(not every position — 10 prompts × 20 positions × 248K vocab is prohibitively
large). On iOS the harness runs prefill, slices logits at S_real-1, and
compares to this reference via cosine similarity.

Output: oracle_ios.json
"""
from pathlib import Path
import argparse
import base64
import json
import torch

ORACLE = Path(__file__).parent / "qwen3_5_reference_logits.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "qwen3_5_oracle_ios.json"))
    args = ap.parse_args()

    oracle = torch.load(str(ORACLE), map_location="cpu", weights_only=False)
    records = []
    for r in oracle["records"]:
        input_ids = r["input_ids"].squeeze(0).tolist()
        S = len(input_ids)
        # Last-position logits as fp16 little-endian bytes, then base64 so it
        # round-trips cleanly through JSON.
        last_logits = r["logits_prefill"][S - 1].to(torch.float16).numpy().tobytes()
        records.append({
            "prompt": r["prompt"],
            "input_ids": input_ids,
            "S_real": S,
            "top1_id": int(r["top10_last_ids"][0].item()),
            "top1_text": r["next_token_text"],
            "last_logits_fp16_b64": base64.b64encode(last_logits).decode("ascii"),
        })

    vocab_size = len(records[0]["last_logits_fp16_b64"]) * 3 // 4 // 2  # fp16 bytes
    out = {
        "model_id": oracle["model_id"],
        "vocab_size": oracle["config"]["vocab_size"],
        "seq_len_bundle": 64,
        "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    import os
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({size_mb:.2f} MB, {len(records)} prompts)")
    print(f"vocab_size={out['vocab_size']}  decoded fp16 per prompt={vocab_size}")


if __name__ == "__main__":
    main()
