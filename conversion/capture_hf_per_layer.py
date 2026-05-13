#!/usr/bin/env python3
"""Capture HF target's hidden_states at every layer boundary on the chat-templated
capitals prompt. Used for chunk-by-chunk parity audit vs Swift's chunks.

Saves per-layer post-norm hidden states + K/V at every layer (not just the
shared L13/L14 ones).
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
    base_ids = tok(chat_text, return_tensors="pt", add_special_tokens=False).input_ids
    # Mirror Swift's bootstrap state: input = [prompt, target_argmax(prompt)]
    # so HF computes hidden states at position N = first generated token position.
    with torch.no_grad():
        first = target.generate(input_ids=base_ids, max_new_tokens=1,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
    chat_ids = first  # length N+1
    print(f"chat_prompt_len = {base_ids.shape[1]}, with bootstrap = {chat_ids.shape[1]}")
    print(f"bootstrap token = {int(chat_ids[0, -1].item())}  ('{tok.decode([int(chat_ids[0, -1].item())])}')")

    with torch.no_grad():
        out = target(input_ids=chat_ids, output_hidden_states=True,
                     use_cache=False, return_shared_kv_states=True)

    # hidden_states[0] = embedding output, hidden_states[i] = output of layer i-1
    # final hidden_states is post-final-norm (for Gemma4 it's last_hidden_state).
    hs = out.hidden_states  # tuple length = num_hidden_layers + 1
    print(f"# hidden_states = {len(hs)} (input embed + {len(hs)-1} layer outputs)")
    for i, h in enumerate(hs):
        print(f"  layer {i:>2}: shape={tuple(h.shape)} min={float(h.min()):>8.3f} max={float(h.max()):>8.3f} std={float(h.std()):.4f}")

    # Save everything.
    os.makedirs("output/mtp_probe", exist_ok=True)
    save = {
        "prompt_token_ids": chat_ids[0].cpu(),
        "hidden_states": [h.cpu() for h in hs],
        "logits_last": out.logits[0, -1, :].cpu(),
        "swa_k": out.shared_kv_states["sliding_attention"][0].cpu() if out.shared_kv_states else None,
        "swa_v": out.shared_kv_states["sliding_attention"][1].cpu() if out.shared_kv_states else None,
        "full_k": out.shared_kv_states["full_attention"][0].cpu() if out.shared_kv_states else None,
        "full_v": out.shared_kv_states["full_attention"][1].cpu() if out.shared_kv_states else None,
    }
    out_path = "output/mtp_probe/hf_per_layer_capitals.pt"
    torch.save(save, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
