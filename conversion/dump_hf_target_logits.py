"""Run HF gemma-4-E2B-it once on the household prompt and dump the
top-N logits at the first generation position. Used to compare against
our CoreML chunk4 verify_qK output to see whether the target distribution
is genuinely concentrated or our INT4 stack is amplifying the gap."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "List 20 common household objects.\n\n"
    "For each item, provide its name in bold, followed by a "
    "two-sentence description. The first sentence should describe "
    "what the object is used for, and the second sentence should "
    "state which room in a house it is typically found in."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--int4", action="store_true",
                    help="Use bnb NF4 4-bit instead of bf16.")
    ap.add_argument("--steps", type=int, default=4,
                    help="How many decode steps to dump logits for.")
    args = ap.parse_args()

    if args.int4:
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=cfg,
            device_map=args.device, dtype=torch.bfloat16).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16).eval().to(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)

    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(args.device)

    cur = ids["input_ids"]
    print(f"prompt_len={cur.shape[1]}  precision={'NF4-int4' if args.int4 else 'bf16'}")
    print()
    for step in range(args.steps):
        with torch.no_grad():
            out = model(input_ids=cur, use_cache=False)
        logits = out.logits[0, -1, :].float().cpu()  # (vocab,)
        top10 = torch.topk(logits, k=10)
        sm = torch.softmax(logits, dim=-1)
        top10_p = sm[top10.indices]
        print(f"=== step {step} ===")
        print(f"  max_logit={top10.values[0].item():.3f}  "
              f"min_top10={top10.values[-1].item():.3f}  "
              f"gap_top0_top1={(top10.values[0]-top10.values[1]).item():.3f}")
        print(f"  p_top1={top10_p[0].item():.6f}  "
              f"sum_top10={top10_p.sum().item():.6f}")
        for i in range(10):
            print(f"    top{i} id={top10.indices[i].item()} "
                  f"logit={top10.values[i].item():.3f} p={top10_p[i].item():.6f}")
        # Append target argmax for next step (greedy emit).
        nxt = top10.indices[0].view(1, 1).to(args.device)
        cur = torch.cat([cur, nxt], dim=1)


if __name__ == "__main__":
    main()
