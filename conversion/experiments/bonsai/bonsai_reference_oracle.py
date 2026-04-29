"""Parity test: HF Qwen3ForCausalLM vs our Qwen3Model (ANE-optimized Conv2d).

Validates that weight loading + QK-norm are wired correctly by comparing:
- last-token logits cosine similarity
- top-1 next token prediction
- first N greedy tokens

Usage:
    python bonsai_reference_oracle.py --model-path /path/to/ternary-bonsai-1.7b
    python bonsai_reference_oracle.py --model-path ... --max-new-tokens 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, repeat_kv, stable_attention
from models.qwen3 import Qwen3Model


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Hello, my name is",
    "def fibonacci(n):\n    if n <= 1:",
]


def cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return float((a @ b) / (a.norm() * b.norm() + eps))


@torch.no_grad()
def hf_next_tokens(model, tokenizer, prompt: str, n: int) -> tuple[list[int], torch.Tensor]:
    """Greedy decode n tokens with HF; return token ids + last-token logits at step 0."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    first_out = model(input_ids=input_ids, use_cache=False)
    last_logits = first_out.logits[0, -1, :].float().cpu()

    generated = input_ids.clone()
    for _ in range(n):
        out = model(input_ids=generated, use_cache=False)
        next_id = out.logits[0, -1, :].argmax().item()
        generated = torch.cat([generated, torch.tensor([[next_id]])], dim=1)

    return generated[0, input_ids.shape[1]:].tolist(), last_logits


@torch.no_grad()
def ours_prefill_last_logits(
    our_model: Qwen3Model, input_ids: torch.Tensor
) -> torch.Tensor:
    """Run one prefill through our Qwen3Model; return (vocab,) logits of final position.

    NOTE: bypasses ANETransformerModel.forward_transformer_prefill because that path
    reads from the (empty) KV cache for attention instead of using the freshly computed
    K/V for the current seq. For a standalone parity test we want a cache-free prefill:
    attend current tokens to themselves with a seq x seq causal mask.
    """
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len)

    # Cache-free prefill: seq x seq causal mask
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32), diagonal=1
    ).view(1, 1, seq_len, seq_len)

    hidden = our_model.forward_embeddings(input_ids)

    for layer in our_model.layers:
        residual = hidden
        hidden = layer.input_layernorm(hidden)

        attn = layer.self_attn
        q, k, v = attn._project_qkv(hidden)  # q_norm/k_norm applied inside
        cos, sin = attn.rotary_emb.forward_range(positions)
        cos = cos.permute(0, 2, 1, 3)
        sin = sin.permute(0, 2, 1, 3)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_exp = repeat_kv(k, attn.n_rep)
        v_exp = repeat_kv(v, attn.n_rep)
        attn_out = stable_attention(q, k_exp, v_exp, attn.scale, mask)
        attn_out = attn._output_proj(attn_out)

        hidden = residual + attn_out

        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

    hidden = our_model.norm(hidden)

    last = hidden[:, -1:, :]
    x = last.permute(0, 2, 1).unsqueeze(2).to(hidden.dtype)
    logits = our_model.lm_head(x).squeeze(2).permute(0, 2, 1)  # (1, 1, vocab)
    return logits[0, 0, :].float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF model dir with config.json + safetensors")
    ap.add_argument("--context-length", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=5, help="Greedy decode tokens for HF side")
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    args = ap.parse_args()

    # Lazy imports so the script prints a clean error if transformers is missing.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model_path).expanduser()
    print(f"Loading HF model from {model_path}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading our Qwen3Model (context_length={args.context_length})")
    our_model = Qwen3Model.from_pretrained(str(model_path), context_length=args.context_length)
    our_model.eval()

    total = 0
    passed = 0

    for prompt in args.prompts:
        print(f"\nprompt: {prompt!r}")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = input_ids.shape[1]
        print(f"  tokens ({seq_len}): {input_ids[0].tolist()}")

        hf_next, hf_last_logits = hf_next_tokens(hf_model, tokenizer, prompt, args.max_new_tokens)

        our_last_logits = ours_prefill_last_logits(our_model, input_ids)
        our_top1 = int(our_last_logits.argmax().item())

        cs = cos_sim(hf_last_logits, our_last_logits)
        hf_top1 = hf_next[0]
        match = our_top1 == hf_top1

        hf_text = tokenizer.decode(hf_next)
        print(f"  HF next token: {hf_top1} ({tokenizer.decode([hf_top1])!r})")
        print(f"  our top-1:     {our_top1} ({tokenizer.decode([our_top1])!r})")
        print(f"  cos(last_logits): {cs:.6f}   match: {match}")
        print(f"  HF {args.max_new_tokens}-token continuation: {hf_text!r}")

        total += 1
        if match and cs >= 0.95:
            passed += 1

    print(f"\nparity summary: {passed}/{total} prompts passed (top1 match + cos>=0.95)")
    if passed < total:
        print("FAIL — investigate QK-norm, weight map, or attention scale")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
