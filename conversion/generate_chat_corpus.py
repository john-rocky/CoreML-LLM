#!/usr/bin/env python3
"""Generate a chat-style corpus by running Gemma 4 E2B on diverse prompts.

The L12 subset LM-head Phase 1 failed because the corpus-derived frequent-
token list (built from Gutenberg literature) missed common LLM-chat tokens
like ' emotions', ' preferences', ' humans', ' language', ' model'. This
script generates ~50K-200K tokens of Gemma 4 chat responses to representative
chat prompts, producing an IN-DISTRIBUTION corpus for `extract_frequent_tokens.py`
to count against.

Usage:
    python conversion/generate_chat_corpus.py --output output/gemma4-e2b/chat_corpus.txt
"""
from __future__ import annotations
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import MODEL_REGISTRY


PROMPTS = [
    # AI / LLM self-description (the biggest miss source — these tokens are
    # what Gemma actually emits in chat).
    "What is your favourite hobby and why?",
    "Tell me about yourself.",
    "Do you have feelings or emotions?",
    "What can you help me with?",
    "Are you a real person?",
    "What are your preferences?",
    "How were you trained?",
    "What model are you?",
    # General factual / informational
    "Explain photosynthesis in simple terms.",
    "How does the internet work?",
    "What is the capital of France?",
    "Describe the water cycle.",
    "Why is the sky blue?",
    "What are the planets in our solar system?",
    # Reasoning / how-to
    "How do I solve a quadratic equation?",
    "Walk me through making a peanut butter sandwich.",
    "What are the steps to debug a Python program?",
    "How can I improve my time management?",
    # Creative / narrative
    "Write a short story about a robot learning to paint.",
    "Compose a poem about autumn leaves.",
    "Tell me a joke about programmers.",
    "Imagine a world where animals can talk; describe a typical day.",
    # Conversation / opinion
    "What do you think about working from home?",
    "Why is reading important?",
    "What makes a good leader?",
    "Should I learn to play an instrument? Which one?",
    # Code / technical
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain what an SQL JOIN does.",
    "How does Git rebase differ from Git merge?",
    "What is the difference between a list and a tuple in Python?",
    # Lists / structured
    "Give me five tips for studying effectively.",
    "List the steps for a basic morning routine.",
    "Name some popular programming languages and what they are used for.",
    # Empathy / emotional
    "I feel anxious about an upcoming exam. Any advice?",
    "What is the best way to deal with stress?",
    "How do I make friends as an adult?",
    # Refusal / instruction-following style
    "Summarize the plot of Hamlet in one paragraph.",
    "Translate 'Good morning, how are you?' into Spanish.",
    "Rewrite this sentence to sound more formal: 'I'm gonna grab a bite.'",
    # Long-form
    "Explain the concept of machine learning to a 10-year-old.",
    "Why is climate change a concern?",
    "Describe the evolution of the smartphone over the past decade.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--output", default=None)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=200,
                    help="Per-prompt max generation length")
    ap.add_argument("--device", default="mps",
                    help="torch device (mps / cpu / cuda)")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, generate only the first N prompts (smoke test)")
    args = ap.parse_args()

    if args.hf_dir is None:
        args.hf_dir = os.path.join(ROOT, "..", "output", args.model, "hf_model")
    args.output = args.output or os.path.join(
        ROOT, "..", "output", args.model, "chat_corpus.txt")

    print(f"loading Gemma 4 from {args.hf_dir} (device={args.device}) ...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, torch_dtype=torch.float16, device_map=args.device)
    model.eval()

    prompts = PROMPTS if args.limit == 0 else PROMPTS[:args.limit]
    all_text: list[str] = []
    t0 = time.time()
    # Gemma 4 chat template: <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
    # The tokenizer ships without a chat_template attribute, so hardcode.
    GEMMA_CHAT = ("<start_of_turn>user\n{p}<end_of_turn>\n"
                  "<start_of_turn>model\n")
    for i, p in enumerate(prompts):
        text = GEMMA_CHAT.format(p=p)
        inputs = tok(text, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.eos_token_id)
        prompt_len = inputs.input_ids.shape[1]
        gen_ids = out[0][prompt_len:].cpu().tolist()
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        all_text.append(gen_text)
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(prompts)}] {len(gen_ids)} tok ({elapsed:.0f}s) — {p[:50]}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for t in all_text:
            f.write(t + "\n\n")
    print(f"wrote {args.output} ({sum(len(t) for t in all_text)} chars across {len(all_text)} responses)")


if __name__ == "__main__":
    main()
