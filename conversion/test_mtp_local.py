#!/usr/bin/env python3
"""Local Mac test: does our PyTorch MTP drafter agree with HF Gemma4's predictions?

Runs HF model on a real prompt, extracts:
  - L34 hidden state (carry for drafter step 0)
  - Token embedding (for drafter step 0)
  - Target's natural next token (ground truth)
  - KV caches at L13 (SWA) and L14 (full)

Feeds to our drafter with both SCALED and UNSCALED embeddings, prints
top-5 drafts vs target's natural next token. The scale that matches
more often is the correct convention.
"""
from __future__ import annotations
import os
import sys

import torch
import torch.nn.functional as F

HF_DIR = os.environ.get(
    "HF_DIR",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/"
        "4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
    ),
)
PT_DIR = "/Users/majimadaisuke/Downloads/CoreML-LLM"
sys.path.insert(0, os.path.join(PT_DIR, "conversion"))

from mtp_drafter_model import MtpDrafterModel, MtpDrafterConfig


def main():
    device = "cpu"
    dtype = torch.float32

    prompts = [
        "The capital of France is",
        "光合成とは",
        "def fibonacci(n):",
    ]

    # Load HF model
    print(f"Loading HF Gemma4 from {HF_DIR}...")
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(HF_DIR)
    hf = Gemma4ForConditionalGeneration.from_pretrained(
        HF_DIR, torch_dtype=dtype, device_map=device,
    ).eval()
    lm = hf.model.language_model
    H = lm.config.hidden_size  # 1536
    embed_scale = H ** 0.5     # ≈ 39.19

    # Load our drafter
    print(f"Loading PyTorch drafter from output/mtp_probe/mtp_drafter.pt...")
    cfg = MtpDrafterConfig()
    drafter = MtpDrafterModel(cfg).to(dtype).eval()
    sd = torch.load(
        os.path.join(PT_DIR, "output/mtp_probe/mtp_drafter.pt"), map_location=device
    )
    drafter.load_state_dict(sd, strict=False)
    print(f"  Drafter loaded ({len(sd)} tensors)")

    # Embedding table for manual lookup (Gemma4TextModel has embed_tokens at root)
    embed_table = lm.embed_tokens.weight.to(dtype)  # (vocab, H)
    ctx = 512  # drafter's W for SWA

    for prompt in prompts:
        print(f"\n{'='*70}\nPROMPT: {prompt!r}")
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        N = ids.shape[1]
        print(f"  tokens: {ids[0].tolist()}")

        with torch.no_grad():
            out = lm(
                input_ids=ids,
                output_hidden_states=True,
                use_cache=True,
            )

        # L34 hidden state AT LAST POSITION (what target would use to predict next)
        # hidden_states is [embed_out, L0_out, L1_out, ..., L33_out, L34_out_post_norm]
        # Actually HF's hidden_states has 36 entries: embed + 34 layers + post_norm
        # The "L34 output before post-norm" should be at index 35 if post-norm is -1
        # From HF docs: index 0 = embeddings, 1..N = layer outputs, -1 = after final norm
        # For Gemma4 with 35 layers: index 35 is L34 output (hidden AFTER L34 block, BEFORE norm)
        # Actually some HF models include post-norm as last. Check:
        print(f"  num hidden_states: {len(out.hidden_states)}")  # should be 35 or 36
        # Last hidden state before norm = out.hidden_states[34] for 0-indexed L0..L33 +
        # the initial embed. Let's use the second-to-last as L34 output.
        # For Gemma4 (35 layers L0..L34), hidden_states has 36 entries:
        # [embed, L0_out, L1_out, ..., L34_out], index 35 = L34 output
        # The `last_hidden_state` is post-norm version.
        l34_hidden = out.hidden_states[-1][:, -1:, :].to(dtype)  # (1, 1, H)
        # But this might already be post-norm. Let's check by comparing to last_hidden_state:
        post_norm = out.last_hidden_state[:, -1:, :].to(dtype)
        diff_to_post = (l34_hidden - post_norm).abs().max().item()
        print(f"  hidden_states[-1] vs last_hidden_state max diff: {diff_to_post:.6f}")
        # If same → hidden_states[-1] IS post-norm. Use [-2] for raw L34.
        if diff_to_post < 1e-3:
            l34_hidden = out.hidden_states[-2][:, -1:, :].to(dtype)
            print(f"  Using hidden_states[-2] (pre-norm L34) as carry")
        else:
            print(f"  Using hidden_states[-1] (raw L34) as carry")
        print(f"  L34 norm: {l34_hidden.norm().item():.3f}")

        # Natural next token: argmax of target's next-token prediction.
        # Gemma4 ties lm_head with embedding weight.
        last_logits = F.linear(out.last_hidden_state[:, -1:, :], embed_table)  # (1, 1, V)
        # Apply softcap like Gemma
        if hasattr(lm.config, "final_logit_softcapping") and lm.config.final_logit_softcapping:
            cap = lm.config.final_logit_softcapping
            last_logits = torch.tanh(last_logits / cap) * cap
        natural_next = last_logits.argmax(-1).item()
        print(f"  Target natural next token: {natural_next} ({tok.decode([natural_next])!r})")

        # Last token ID (what nextID would be in our flow)
        # In our flow, after decode, nextID = natural_next. But for drafter step 0,
        # we feed embed(nextID) = embed(natural_next).
        next_id = natural_next

        # Extract KV at L13 (SWA) and L14 (full) from past_key_values
        # past_key_values is a tuple of (key, value) per layer
        # Shape: each (1, num_kv_heads, seq, head_dim)
        pkv = out.past_key_values
        # pkv could be Cache object or tuple
        if hasattr(pkv, "layers"):
            k13, v13 = pkv.layers[13].keys, pkv.layers[13].values
            k14, v14 = pkv.layers[14].keys, pkv.layers[14].values
        elif hasattr(pkv, "key_cache"):
            k13 = pkv.key_cache[13]; v13 = pkv.value_cache[13]
            k14 = pkv.key_cache[14]; v14 = pkv.value_cache[14]
        else:
            k13, v13 = pkv[13]
            k14, v14 = pkv[14]
        print(f"  kv13 K: {tuple(k13.shape)}  V: {tuple(v13.shape)}")
        print(f"  kv14 K: {tuple(k14.shape)}  V: {tuple(v14.shape)}")

        # Shapes from HF: (1, num_kv_heads=1, seq, head_dim) for single KV head
        # Our drafter expects:
        #   kv13_k: (1, 1, ctx, 256)
        #   kv13_v: (1, 1, 256, ctx)  ← transposed!
        #   kv14_k: (1, 1, ctx, 512)
        #   kv14_v: (1, 1, 512, ctx)  ← transposed!
        # Pad to ctx length
        def to_drafter_k(k, target_dim, ctx):
            # k: (1, 1, seq, hd)
            seq = k.shape[2]
            hd = k.shape[3]
            assert hd == target_dim, f"hd={hd} != target {target_dim}"
            out = torch.zeros(1, 1, ctx, hd, dtype=k.dtype)
            # RIGHT-ALIGN: put valid tokens at the end (matching Swift mask convention)
            out[:, :, -seq:, :] = k
            return out

        def to_drafter_v(v, target_dim, ctx):
            seq = v.shape[2]
            hd = v.shape[3]
            assert hd == target_dim
            # drafter V: (1, 1, hd, ctx) — transposed
            out = torch.zeros(1, 1, hd, ctx, dtype=v.dtype)
            v_t = v.transpose(-1, -2)  # (1, 1, hd, seq)
            out[:, :, :, -seq:] = v_t
            return out

        kv13_k = to_drafter_k(k13, 256, ctx)
        kv13_v = to_drafter_v(v13, 256, ctx)
        kv14_k = to_drafter_k(k14, 512, ctx * 4)  # full context
        kv14_v = to_drafter_v(v14, 512, ctx * 4)

        # Masks: allow last `seq` positions (right-aligned)
        seq = k13.shape[2]
        mask_swa = torch.full((1, 1, 1, ctx), -1e9, dtype=dtype)
        mask_swa[0, 0, 0, -seq:] = 0
        mask_full = torch.full((1, 1, 1, ctx * 4), -1e9, dtype=dtype)
        mask_full[0, 0, 0, -seq:] = 0

        # Drafter position (where next token goes): seq
        pos = torch.tensor([seq], dtype=torch.int32)

        # Test: embed variations
        embed_raw = embed_table[next_id].unsqueeze(0).unsqueeze(0)  # (1, 1, H), unscaled
        embed_scaled = embed_raw * embed_scale  # (1, 1, H), scaled like main model

        # Also try post-norm hidden as carry
        post_norm_hidden = out.last_hidden_state[:, -1:, :].to(dtype)

        variants = [
            ("UNSCALED+L34raw", embed_raw, l34_hidden),
            ("SCALED+L34raw", embed_scaled, l34_hidden),
            ("UNSCALED+postNorm", embed_raw, post_norm_hidden),
            ("SCALED+postNorm", embed_scaled, post_norm_hidden),
        ]
        for label, embed, carry in variants:
            act = torch.cat([embed, carry], dim=-1)  # (1, 1, 3072)
            with torch.no_grad():
                logits, _ = drafter(
                    act, pos,
                    kv13_k, kv13_v, kv14_k, kv14_v,
                    mask_swa, mask_full,
                )
            top5 = torch.topk(logits[0, 0], 5).indices.tolist()
            is_match = (top5[0] == natural_next)
            natural_in_top5 = natural_next in top5
            print(f"  [{label:22s}] e={embed.norm().item():5.1f} c={carry.norm().item():5.1f} "
                  f"top1={top5[0]} ({tok.decode([top5[0]])!r}) "
                  f"m1={'Y' if is_match else 'n'} "
                  f"t5={'Y' if natural_in_top5 else 'n'}")


if __name__ == "__main__":
    main()
