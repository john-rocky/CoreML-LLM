"""Apples-to-apples comparison: feed identical inputs to HF Python drafter
and our CoreML drafter, dump top-1 predictions per call. If they diverge,
the bug is in our CoreML conversion or input preparation; if they match,
the bug is somewhere else (e.g. mask/RoPE/KV layout).
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "translate": (
        "Translate the following 10 sentences from English to French.\n"
        "1. Hello, how are you today?\n2. The cat is on the table.\n"
        "3. I would like a cup of coffee.\n4. Where is the train station?\n"
        "5. We are going to the beach tomorrow.\n6. She reads books every "
        "evening.\n7. They live in a small village.\n8. The weather is "
        "very nice.\n9. Could you help me, please?\n10. I love this song."
    ),
    "count_to_50": "Count from 1 to 50. Output exactly: 1, 2, 3, ..., 50.",
    "code_class": (
        "Write a Python class `LRUCache` with get(key) and put(key, "
        "value) methods, both O(1). Include type hints, a docstring, "
        "and 3 unit tests using pytest."
    ),
}
PROMPT = PROMPTS["translate"]  # default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="google/gemma-4-E2B-it")
    ap.add_argument("--assistant", default="google/gemma-4-E2B-it-assistant")
    ap.add_argument("--coreml-drafter",
                    default="/tmp/mtp_drafter_lse2/mtp_drafter.mlpackage",
                    help="Our CoreML drafter mlpackage")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-cycles", type=int, default=5)
    ap.add_argument("--prompt", default="translate", choices=list(PROMPTS.keys()))
    args = ap.parse_args()
    global PROMPT
    PROMPT = PROMPTS[args.prompt]

    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16).eval().to(args.device)
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=torch.bfloat16).eval().to(args.device)
    tok = AutoTokenizer.from_pretrained(args.target)
    target_embed = target.model.language_model.embed_tokens

    import coremltools as ct
    coreml_drafter = ct.models.MLModel(args.coreml_drafter,
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)

    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(args.device)
    cur = ids["input_ids"]
    print(f"prompt_len={cur.shape[1]}")
    print(f"prompt last 5 tokens: {cur[0, -5:].tolist()}")

    # Run target on prompt to get initial state.
    with torch.no_grad():
        out = target.model.language_model(
            input_ids=cur, return_shared_kv_states=True,
            output_hidden_states=False)
    last_hidden = out.last_hidden_state  # (1, prompt_len, 1536)
    shared_kv = out.shared_kv_states

    print(f"last_hidden shape: {tuple(last_hidden.shape)} dtype={last_hidden.dtype}")
    print(f"shared_kv keys: {list(shared_kv.keys())}")
    for k, v in shared_kv.items():
        print(f"  {k}: K shape={tuple(v[0].shape)}, V shape={tuple(v[1].shape)}")

    for cycle in range(args.max_cycles):
        last_token_id = cur[:, -1:]
        ph = last_hidden[:, -1:]  # (1, 1, 1536)
        position_ids = torch.tensor([[cur.shape[1] - 1]],
                                     dtype=torch.long, device=args.device)
        # Slice shared KV to current length on the seq_len dim (dim=2)
        kv_slice = {
            k: (v[0][:, :, :cur.shape[1], :], v[1][:, :, :cur.shape[1], :])
            for k, v in shared_kv.items()
        }

        # === HF drafter ===
        emb = target_embed(last_token_id)  # (1, 1, 1536)
        inputs_embeds = torch.cat([emb, ph], dim=-1)  # (1, 1, 3072)
        with torch.no_grad():
            d_out = assistant(inputs_embeds=inputs_embeds,
                               position_ids=position_ids,
                               shared_kv_states=kv_slice,
                               use_cache=False)
        hf_top1 = int(d_out.logits[0, -1, :].argmax().item())
        hf_top5 = torch.topk(d_out.logits[0, -1, :], k=5)
        hf_logit_top1 = float(hf_top5.values[0].item())

        # === CoreML drafter ===
        # Prepare inputs for CoreML drafter:
        #   embed_token: (1, 1, 1536) — apply embed_scale = sqrt(hidden) = sqrt(1536)
        #   proj_act: (1, 1, 1536) — last_hidden_state
        #   kv13_k: (1, num_kv, W, 256) — slice/pad sliding KV to W
        #   kv13_v: (1, num_kv, 256, W) — TRANSPOSED
        #   kv14_k: (1, num_kv, ctx, 512) — slice/pad full KV to ctx
        #   kv14_v: (1, num_kv, 512, ctx) — TRANSPOSED
        #   cos/sin: (1, 128) and (1, 256) — first half of duplicated halves
        W, ctx = 512, 2048
        # `target_embed` already applies embed_scale internally per Gemma3/4.
        # So `emb` is the scaled embedding (matches our engine.embedToken).
        embed_token_np = emb.detach().detach().float().cpu().numpy().astype(np.float16)

        proj_act_np = ph.detach().detach().float().cpu().numpy().astype(np.float16)

        # Sliding K/V: shape from HF is (1, num_kv_swa, prompt_len, hd_swa=256).
        # We need (1, num_kv_swa, W, 256). Take last W entries (or pad if < W).
        swa_k_full = kv_slice["sliding_attention"][0]  # (1, nkv, prompt_len, 256)
        swa_v_full = kv_slice["sliding_attention"][1]  # (1, nkv, prompt_len, 256)
        if swa_k_full.shape[2] > W:
            swa_k = swa_k_full[:, :, -W:, :]
            swa_v = swa_v_full[:, :, -W:, :]
        else:
            # Right-pad to W with zeros (left side is real cache, padded right
            # = unused future slots).
            pad = W - swa_k_full.shape[2]
            swa_k = torch.cat([swa_k_full, torch.zeros(1, swa_k_full.shape[1], pad,
                              swa_k_full.shape[3], device=args.device,
                              dtype=swa_k_full.dtype)], dim=2)
            swa_v = torch.cat([swa_v_full, torch.zeros(1, swa_v_full.shape[1], pad,
                              swa_v_full.shape[3], device=args.device,
                              dtype=swa_v_full.dtype)], dim=2)
        kv13_k_np = swa_k.detach().float().cpu().numpy().astype(np.float16)
        # V transposed: (1, nkv, hd, W)
        kv13_v_np = swa_v.transpose(2, 3).detach().float().cpu().numpy().astype(np.float16)

        # Full K/V
        full_k_full = kv_slice["full_attention"][0]
        full_v_full = kv_slice["full_attention"][1]
        if full_k_full.shape[2] > ctx:
            full_k = full_k_full[:, :, -ctx:, :]
            full_v = full_v_full[:, :, -ctx:, :]
        else:
            pad = ctx - full_k_full.shape[2]
            full_k = torch.cat([full_k_full, torch.zeros(1, full_k_full.shape[1], pad,
                                full_k_full.shape[3], device=args.device,
                                dtype=full_k_full.dtype)], dim=2)
            full_v = torch.cat([full_v_full, torch.zeros(1, full_v_full.shape[1], pad,
                                full_v_full.shape[3], device=args.device,
                                dtype=full_v_full.dtype)], dim=2)
        kv14_k_np = full_k.detach().float().cpu().numpy().astype(np.float16)
        kv14_v_np = full_v.transpose(2, 3).detach().float().cpu().numpy().astype(np.float16)

        # RoPE: precomputed cos/sin tables, position = pos - 1.
        # Drafter expects (1, hd/2) — first half of duplicated halves.
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
            Gemma4AssistantPreTrainedModel as _,
        )
        # Use assistant's own rotary to compute correct cos/sin
        pos = cur.shape[1] - 1
        # Try to access target's rotary embeddings
        rot_full = target.model.language_model.rotary_emb
        # Rotary returns (cos, sin) with shape (1, seq_len, head_dim_full)
        # For our drafter we need shape (1, head_dim/2)
        cos_full_full, sin_full_full = rot_full(
            torch.zeros(1, 1, 1536, device=args.device, dtype=torch.bfloat16),
            torch.tensor([[pos]], device=args.device),
            "full_attention",
        )
        cos_swa_full, sin_swa_full = rot_full(
            torch.zeros(1, 1, 1536, device=args.device, dtype=torch.bfloat16),
            torch.tensor([[pos]], device=args.device),
            "sliding_attention",
        )
        # cos_full_full shape: (1, 1, 512). Take first half = (1, 256).
        # cos_swa_full shape: (1, 1, 256). Take first half = (1, 128).
        cos_full_np = cos_full_full[0, :, :cos_full_full.shape[-1]//2].detach().float().cpu().numpy().astype(np.float16)
        sin_full_np = sin_full_full[0, :, :sin_full_full.shape[-1]//2].detach().float().cpu().numpy().astype(np.float16)
        cos_swa_np = cos_swa_full[0, :, :cos_swa_full.shape[-1]//2].detach().float().cpu().numpy().astype(np.float16)
        sin_swa_np = sin_swa_full[0, :, :sin_swa_full.shape[-1]//2].detach().float().cpu().numpy().astype(np.float16)

        # Masks: causal up to position
        mask_full_np = np.zeros((1, 1, 1, ctx), dtype=np.float16)
        mask_full_np[0, 0, 0, pos+1:] = -65500.0  # mask future
        mask_swa_np = np.zeros((1, 1, 1, W), dtype=np.float16)
        # For SWA, valid is last min(pos+1, W) entries. With our cache layout
        # (oldest first, newest at index pos), unmask [0, pos] up to W.
        if pos + 1 < W:
            mask_swa_np[0, 0, 0, pos+1:] = -65500.0

        out_dict = coreml_drafter.predict({
            "embed_token": embed_token_np,
            "proj_act": proj_act_np,
            "kv13_k": kv13_k_np,
            "kv13_v": kv13_v_np,
            "kv14_k": kv14_k_np,
            "kv14_v": kv14_v_np,
            "cos_swa": cos_swa_np,
            "sin_swa": sin_swa_np,
            "cos_full": cos_full_np,
            "sin_full": sin_full_np,
            "mask_swa": mask_swa_np,
            "mask_full": mask_full_np,
        })
        coreml_top1 = int(out_dict["top_k_indices"][0])
        coreml_top1_logit = float(out_dict["top_k_values"][0])

        match = "✓" if coreml_top1 == hf_top1 else "✗"
        print(f"cycle {cycle}: pos={pos} {match}  "
              f"HF top1={hf_top1} (L={hf_logit_top1:.3f}) "
              f"CoreML top1={coreml_top1} (L={coreml_top1_logit:.3f})")
        if coreml_top1 != hf_top1:
            print(f"  HF top5: {hf_top5.indices.tolist()}")
            print(f"  CoreML top5: {[int(x) for x in out_dict['top_k_indices'][:5]]}")

        # Commit target's argmax for next cycle
        with torch.no_grad():
            t_out = target.model.language_model(
                input_ids=cur, return_shared_kv_states=True,
                output_hidden_states=False)
        target_top1 = int(t_out.last_hidden_state[0, -1, :].argmax().item()
                          if False else  # skip; use logits via lm_head
                          target.lm_head(t_out.last_hidden_state)[0, -1, :].argmax().item())
        next_id = torch.tensor([[target_top1]], device=args.device)
        cur = torch.cat([cur, next_id], dim=1)
        last_hidden = t_out.last_hidden_state
        shared_kv = t_out.shared_kv_states


if __name__ == "__main__":
    main()
