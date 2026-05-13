#!/usr/bin/env python3
"""HF transformers MTP bench — gold-standard reference for what vendor's
universal drafter actually delivers on a non-CoreML stack.

Runs Gemma 4 E2B (bf16) + assistant (bf16) on MPS via transformers's
SinglePositionMultiTokenCandidateGenerator (auto-selected when
`assistant_model` is the assistant repo). Measures tok/s with and
without MTP on the same prompts our CoreML bench uses, so we can
compare apples-to-apples and decide whether our gap is INT4-specific
or implementation-specific.
"""
from __future__ import annotations
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "fib_recursion":
        "find the nth Fibonacci number using recursion.",
    "repeat":
        'Say "yes" 30 times.',
    "stack":
        "Explain the difference between a stack and a queue in 4 bullet "
        "points, with a Python code example showing each.",
    "qsort":
        "def quicksort(arr):",
    "household":
        "List 20 common household objects.\n\n"
        "For each item, provide its name in bold, followed by a "
        "two-sentence description. The first sentence should describe "
        "what the object is used for, and the second sentence should "
        "state which room in a house it is typically found in.",
    # --- Probe prompt diversity ---
    "json":
        "Return a JSON array of 15 cities, each object containing "
        "fields: name, country, population, latitude, longitude, "
        "and a one-line trivia. Format strictly as valid JSON.",
    "translate":
        "Translate the following 10 sentences from English to French.\n"
        "1. Hello, how are you today?\n2. The cat is on the table.\n"
        "3. I would like a cup of coffee.\n4. Where is the train station?\n"
        "5. We are going to the beach tomorrow.\n6. She reads books every "
        "evening.\n7. They live in a small village.\n8. The weather is "
        "very nice.\n9. Could you help me, please?\n10. I love this song.",
    "qa_long":
        "Explain how a transformer language model generates text, "
        "from input tokenisation to output sampling. Cover embeddings, "
        "self-attention, KV cache, sampling temperature, and how "
        "speculative decoding accelerates inference. Use full "
        "sentences and aim for ~400 words.",
    "code_class":
        "Write a Python class `LRUCache` with get(key) and put(key, "
        "value) methods, both O(1). Include type hints, a docstring, "
        "and 3 unit tests using pytest.",
    "count_to_50":
        "Count from 1 to 50. Output exactly: 1, 2, 3, ..., 50.",
    "markdown_table":
        "Make a markdown table comparing 8 popular programming "
        "languages along these columns: Name, First released, "
        "Typing, Memory model, Typical use case.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="google/gemma-4-E2B-it")
    ap.add_argument("--assistant", default="google/gemma-4-E2B-it-assistant")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompts", nargs="*", default=list(PROMPTS.keys()))
    ap.add_argument("--sample", action="store_true",
                    help="Use rejection sampling (do_sample=True, temp=0.7)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--int4", action="store_true",
                    help="Load both target and assistant via bitsandbytes "
                         "NF4 4-bit quantization (compute in bf16). "
                         "Apples-to-apples vs our CoreML INT4 stack.")
    args = ap.parse_args()

    if args.int4:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        print(f"Loading target {args.target} (NF4 4-bit) on {args.device} ...")
        target = AutoModelForCausalLM.from_pretrained(
            args.target, quantization_config=qcfg,
            device_map=args.device, dtype=torch.bfloat16).eval()
        print(f"Loading assistant {args.assistant} (NF4 4-bit) on {args.device} ...")
        assistant = AutoModelForCausalLM.from_pretrained(
            args.assistant, quantization_config=qcfg,
            device_map=args.device, dtype=torch.bfloat16).eval()
    else:
        print(f"Loading target {args.target} (bf16) on {args.device} ...")
        target = AutoModelForCausalLM.from_pretrained(
            args.target, dtype=torch.bfloat16).eval().to(args.device)
        print(f"Loading assistant {args.assistant} (bf16) on {args.device} ...")
        assistant = AutoModelForCausalLM.from_pretrained(
            args.assistant, dtype=torch.bfloat16).eval().to(args.device)
    tok = AutoTokenizer.from_pretrained(args.target)

    print(f"\nMax new tokens: {args.max_new_tokens}")
    print(f"\n{'Prompt':<14} {'mode':<10} {'tok/s':>8} {'gen_tokens':>11}")
    print("-" * 50)

    for pname in args.prompts:
        prompt = PROMPTS[pname]
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(args.device)

        sample_kw = ({"do_sample": True, "temperature": args.temperature}
                     if args.sample else {"do_sample": False})
        for mode, kwargs in [
            ("no_MTP", {}),
            ("MTP",    {"assistant_model": assistant}),
        ]:
            # warm-up (cuts compile/cache cost)
            _ = target.generate(**ids, max_new_tokens=8, **sample_kw, **kwargs)
            torch.mps.synchronize() if args.device == "mps" else None

            t0 = time.perf_counter()
            out = target.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                **sample_kw, **kwargs)
            torch.mps.synchronize() if args.device == "mps" else None
            dt = time.perf_counter() - t0
            n_new = out.shape[1] - ids["input_ids"].shape[1]
            print(f"{pname:<14} {mode:<10} {n_new/dt:>8.2f} {n_new:>11}")
        print()


if __name__ == "__main__":
    main()
