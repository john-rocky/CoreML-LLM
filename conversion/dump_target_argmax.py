#!/usr/bin/env python3
"""Dump HF target's argmax sequence for the chat-templated capitals prompt.

Compare to Swift target's output token-by-token.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

target = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it", dtype=torch.float16, low_cpu_mem_usage=True).eval()
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
prompt = ("The capital of France is Paris. The capital of Germany is Berlin. "
          "The capital of Italy is")
msgs = [{"role": "user", "content": prompt}]
chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
chat_ids = tok(chat_text, return_tensors="pt", add_special_tokens=False).input_ids
print(f"prompt_len = {chat_ids.shape[1]}")
with torch.no_grad():
    out = target.generate(input_ids=chat_ids, max_new_tokens=64,
                           do_sample=False, pad_token_id=tok.eos_token_id)
new = out[0, chat_ids.shape[1]:].tolist()
print(f"HF target argmax sequence (next {len(new)} tokens):")
print(new)
print(f"decoded: {repr(tok.decode(new))}")
