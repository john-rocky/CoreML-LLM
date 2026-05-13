#!/usr/bin/env python3
"""Capture HF target's L13 (sliding) and L14 (full) K/V on a chat-templated
capitals prompt. Save to .pt for diff vs Swift's kv13/kv14.

These are EXACTLY the K/V the MTP drafter consumes (HF's
shared_kv_states["sliding_attention"] / ["full_attention"] in the
candidate generator).
"""
from __future__ import annotations
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    target = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=torch.float16,
        low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

    prompt = ("The capital of France is Paris. The capital of Germany is Berlin. "
              "The capital of Italy is")
    msgs = [{"role": "user", "content": prompt}]
    chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    chat_ids = tok(chat_text, return_tensors="pt", add_special_tokens=False).input_ids
    print(f"chat_prompt_len = {chat_ids.shape[1]}")
    print(f"prompt token_ids = {chat_ids[0].tolist()}")

    # Run target.forward with output_hidden_states + return_shared_kv_states.
    with torch.no_grad():
        out = target(
            input_ids=chat_ids,
            output_hidden_states=True,
            use_cache=False,
            return_shared_kv_states=True)

    # logits argmax for next position.
    logits = out.logits[0, -1, :]
    top1 = int(logits.argmax().item())
    print(f"target argmax for position {chat_ids.shape[1]} = {top1}  ('{tok.decode([top1])}')")

    # shared_kv_states is a dict with sliding_attention / full_attention each (k, v) tuple.
    skv = out.shared_kv_states
    swa_k, swa_v = skv["sliding_attention"]
    full_k, full_v = skv["full_attention"]
    print(f"swa K: {tuple(swa_k.shape)} dtype={swa_k.dtype}  "
          f"min={float(swa_k.min()):.3f} max={float(swa_k.max()):.3f} std={float(swa_k.std()):.4f}")
    print(f"swa V: {tuple(swa_v.shape)}")
    print(f"full K: {tuple(full_k.shape)} dtype={full_k.dtype}  "
          f"min={float(full_k.min()):.3f} max={float(full_k.max()):.3f} std={float(full_k.std()):.4f}")
    print(f"full V: {tuple(full_v.shape)}")

    # Last hidden (post-final-norm).
    last_hidden = out.hidden_states[-1]  # (1, seq, hidden)
    print(f"last_hidden: {tuple(last_hidden.shape)}  "
          f"min={float(last_hidden.min()):.3f} max={float(last_hidden.max()):.3f}")

    os.makedirs("output/mtp_probe", exist_ok=True)
    cap = {
        "prompt_token_ids": chat_ids[0].cpu(),
        "next_argmax": top1,
        "swa_k": swa_k.cpu(),  # (1, 1, seq, swa_head_dim=256)
        "swa_v": swa_v.cpu(),
        "full_k": full_k.cpu(), # (1, 1, seq, full_head_dim=512)
        "full_v": full_v.cpu(),
        "last_hidden": last_hidden.cpu(),
        "logits_last": logits.cpu(),
    }
    out_path = "output/mtp_probe/hf_kv_chat_capitals.pt"
    torch.save(cap, out_path)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
