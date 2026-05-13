"""Compare our CoreML chunk2 KV outputs to HF target's shared_kv_states.
Cosine similarity per element shows how much INT4 quantization noise
diverges our drafter inputs from the HF reference. Low cosine (< 0.95)
explains low MTP drafter accept rate."""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "count_to_50": "Count from 1 to 50. Output exactly: 1, 2, 3, ..., 50.",
    "translate": (
        "Translate the following 10 sentences from English to French.\n"
        "1. Hello, how are you today?\n2. The cat is on the table.\n"
        "3. I would like a cup of coffee."
    ),
}


def cosine(a, b):
    a, b = a.flatten().astype(np.float32), b.flatten().astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/tmp/coreml_kv_dump")
    ap.add_argument("--prompt", default="count_to_50",
                    choices=list(PROMPTS.keys()))
    ap.add_argument("--target", default="google/gemma-4-E2B-it")
    args = ap.parse_args()

    # Load CoreML dumps.
    info = open(f"{args.dump_dir}/info.txt").read()
    print(f"CoreML dump info: {info.strip()}")
    pos_str = info.split("pos=")[1].split()[0]
    pos = int(pos_str)
    next_id = int(info.split("nextID=")[1].split()[0])
    print(f"  pos={pos} (next position to fill = {pos+1}); nextID={next_id}")

    coreml_kv13_k = np.frombuffer(
        open(f"{args.dump_dir}/kv13_k.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 512, 256)
    coreml_kv13_v = np.frombuffer(
        open(f"{args.dump_dir}/kv13_v.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 512, 256)
    coreml_kv14_k = np.frombuffer(
        open(f"{args.dump_dir}/kv14_k.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 2048, 512)
    coreml_kv14_v = np.frombuffer(
        open(f"{args.dump_dir}/kv14_v.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 2048, 512)
    coreml_h4 = np.frombuffer(
        open(f"{args.dump_dir}/h4_postnorm.fp16", "rb").read(),
        dtype=np.float16).reshape(1, 1, 1536)

    print(f"  coreml kv13_k: {coreml_kv13_k.shape}")
    print(f"  coreml kv14_k: {coreml_kv14_k.shape}")
    print(f"  coreml h4_postnorm: {coreml_h4.shape}")

    # Run HF target with same prompt to get reference KV.
    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16).eval().to("mps")

    msgs = [{"role": "user", "content": PROMPTS[args.prompt]}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("mps")
    print(f"  HF prompt_len={ids['input_ids'].shape[1]}")

    # Take first (pos+1) tokens
    ids_truncated = {"input_ids": ids["input_ids"][:, :pos+1]}
    print(f"  Using first {pos+1} tokens for HF reference")
    with torch.no_grad():
        out = target.model.language_model(
            **ids_truncated, return_shared_kv_states=True,
            output_hidden_states=False)

    hf_swa_k = out.shared_kv_states["sliding_attention"][0]  # (1, nkv, pos+1, 256)
    hf_swa_v = out.shared_kv_states["sliding_attention"][1]
    hf_full_k = out.shared_kv_states["full_attention"][0]
    hf_full_v = out.shared_kv_states["full_attention"][1]
    hf_last_hidden = out.last_hidden_state[:, -1:]  # (1, 1, 1536)

    print(f"  HF swa_k: {hf_swa_k.shape}, hf_full_k: {hf_full_k.shape}")

    # Compare positions [0, min(prompt_len, our_pos)) — only positions HF
    # has filled. Our dump may have extra position(s) from bootstrap step.
    L = min(pos + 1, hf_swa_k.shape[2])
    # IMPORTANT: our chunk2 stores sliding K right-aligned. With pos+1
    # positions written, slot W-(pos+1)+i holds position i's K. HF stores
    # left-aligned (slot 0 = oldest). Compare correctly:
    W_swa = coreml_kv13_k.shape[2]  # 512
    # `pos` = current_position (next to fill). Positions [0..pos-1] written.
    n_written = pos  # off-by-one fix
    start_slot = W_swa - n_written
    coreml_kv13_k_used = coreml_kv13_k[0, 0, start_slot:start_slot + L]
    coreml_kv13_v_used = coreml_kv13_v[0, 0, start_slot:start_slot + L]
    # kv14_k is full attention — left-aligned same as HF
    coreml_kv14_k_used = coreml_kv14_k[0, 0, :L]
    coreml_kv14_v_used = coreml_kv14_v[0, 0, :L]
    hf_swa_k_used = hf_swa_k[0, 0, :L].detach().float().cpu().numpy()  # (L, 256)
    hf_swa_v_used = hf_swa_v[0, 0, :L].detach().float().cpu().numpy()
    hf_full_k_used = hf_full_k[0, 0, :L].detach().float().cpu().numpy()
    hf_full_v_used = hf_full_v[0, 0, :L].detach().float().cpu().numpy()

    print(f"\n=== KV / hidden cosine similarity (CoreML vs HF, first {L} positions; W_swa={W_swa}) ===")
    print(f"  kv13_k (sliding K): cosine = {cosine(coreml_kv13_k_used, hf_swa_k_used):.6f}")
    print(f"  kv13_v (sliding V): cosine = {cosine(coreml_kv13_v_used, hf_swa_v_used):.6f}")
    print(f"  kv14_k (full K):    cosine = {cosine(coreml_kv14_k_used, hf_full_k_used):.6f}")
    print(f"  kv14_v (full V):    cosine = {cosine(coreml_kv14_v_used, hf_full_v_used):.6f}")
    print(f"  h4_postnorm:        cosine = {cosine(coreml_h4, hf_last_hidden.detach().float().cpu().numpy()):.6f}")

    # Per-position breakdown for K
    print(f"\n=== per-position cosine for kv13_k (sliding K, our slot W-L+i vs HF slot i) ===")
    for i in [0, 5, 10, 20, L-5, L-1]:
        if 0 <= i < L:
            c = cosine(coreml_kv13_k_used[i], hf_swa_k_used[i])
            print(f"  pos {i}: {c:.6f}")
    print(f"\n=== per-position cosine for kv14_k (full K) ===")
    for i in [0, 5, 10, 20, L-5, L-1]:
        if 0 <= i < L:
            c = cosine(coreml_kv14_k_used[i], hf_full_k_used[i])
            print(f"  pos {i}: {c:.6f}")


if __name__ == "__main__":
    main()
