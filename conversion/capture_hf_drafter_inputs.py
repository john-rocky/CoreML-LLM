#!/usr/bin/env python3
"""Capture real HF drafter inputs/outputs from a single round.

Runs `target.generate()` with assistant on the capitals prompt for one
round, intercepts the very first drafter call, and dumps:
  - embed inputs (last_token_embedding, last_hidden_state)
  - shared_kv_states (sliding K/V, full K/V)
  - position_ids
  - drafter's output tokens + last_hidden_state

The capture lets us re-run the same inputs through our PyTorch port +
CoreML build to see EXACTLY where divergence starts in our pipeline.

Output: output/mtp_probe/hf_capture.pt  (a single dict)
"""
from __future__ import annotations
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import candidate_generator as _cg

CAPTURE: dict = {}
_orig_get = _cg.SinglePositionMultiTokenCandidateGenerator.get_candidates


def _patched_get(self, input_ids, model_kwargs, model_outputs,
                 is_first_iteration, n_last_matches, **kwargs):
    """Capture the first non-first-iteration call's inputs to drafter."""
    if is_first_iteration or CAPTURE:
        return _orig_get(self, input_ids, model_kwargs, model_outputs,
                         is_first_iteration, n_last_matches, **kwargs)

    last_hidden_state = model_outputs.hidden_states[-1]
    shared_kv_states = model_outputs.shared_kv_states
    current_length = input_ids.shape[1]
    shared_kv_states = {
        k: (v[0][:, :, :current_length, :], v[1][:, :, :current_length, :])
        for k, v in shared_kv_states.items()
    }
    last_hidden_state_slice = last_hidden_state[:, n_last_matches: n_last_matches + 1]
    last_token_id = input_ids[:, -1:]

    # Run drafter once manually so we can capture both inputs and outputs.
    last_token_embedding = self.target_model_input_embeddings(last_token_id)
    inputs_embeds = torch.cat([last_token_embedding, last_hidden_state_slice], dim=-1)
    position_ids = torch.tensor([[input_ids.shape[1] - 1]], dtype=torch.long,
                                device=self.assistant_model.device)
    with torch.no_grad():
        outputs = self.assistant_model(
            inputs_embeds=inputs_embeds,
            attention_mask=model_kwargs.get("attention_mask"),
            position_ids=position_ids,
            shared_kv_states=shared_kv_states,
            use_cache=False,
        )
    next_token_id = outputs.logits.argmax(dim=-1)
    next_last_hidden = outputs.last_hidden_state

    CAPTURE["input_ids"] = input_ids.detach().cpu()
    CAPTURE["last_token_id"] = last_token_id.detach().cpu()
    CAPTURE["last_token_embedding"] = last_token_embedding.detach().cpu()
    CAPTURE["last_hidden_state"] = last_hidden_state_slice.detach().cpu()
    CAPTURE["inputs_embeds"] = inputs_embeds.detach().cpu()
    CAPTURE["position_ids"] = position_ids.detach().cpu()
    CAPTURE["sliding_k"] = shared_kv_states["sliding_attention"][0].detach().cpu()
    CAPTURE["sliding_v"] = shared_kv_states["sliding_attention"][1].detach().cpu()
    CAPTURE["full_k"] = shared_kv_states["full_attention"][0].detach().cpu()
    CAPTURE["full_v"] = shared_kv_states["full_attention"][1].detach().cpu()
    CAPTURE["drafter_logits"] = outputs.logits.detach().cpu()
    CAPTURE["drafter_token"] = next_token_id.detach().cpu()
    CAPTURE["drafter_last_hidden"] = next_last_hidden.detach().cpu()
    print(f"[capture] input_len={input_ids.shape[1]}  position_ids={int(position_ids[0,0].item())}")
    print(f"[capture] last_token_id={int(last_token_id[0,0].item())}  drafter top-1={int(next_token_id[0,0].item())}")
    print(f"[capture] sliding_k {tuple(shared_kv_states['sliding_attention'][0].shape)}")
    print(f"[capture] full_k    {tuple(shared_kv_states['full_attention'][0].shape)}")
    print(f"[capture] last_hidden_state {tuple(last_hidden_state_slice.shape)}")
    return _orig_get(self, input_ids, model_kwargs, model_outputs,
                     is_first_iteration, n_last_matches, **kwargs)


_cg.SinglePositionMultiTokenCandidateGenerator.get_candidates = _patched_get


def main():
    target = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=torch.float32,
        low_cpu_mem_usage=True).eval()
    drafter = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it-assistant", dtype=torch.float32,
        low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    prompt = ("The capital of France is Paris. The capital of Germany is Berlin. "
              "The capital of Italy is")
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        target.generate(**inputs, max_new_tokens=8, do_sample=False,
                         assistant_model=drafter,
                         pad_token_id=tok.eos_token_id)
    if not CAPTURE:
        print("ERROR: no drafter call captured", file=sys.stderr)
        sys.exit(1)
    os.makedirs("output/mtp_probe", exist_ok=True)
    out = "output/mtp_probe/hf_capture.pt"
    torch.save(CAPTURE, out)
    print(f"[capture] saved {out}")
    # Print high-level stats for sanity.
    sk = CAPTURE["sliding_k"]; fk = CAPTURE["full_k"]
    print(f"[capture] sliding_k stats min={sk.min():.3f} max={sk.max():.3f} mean={sk.mean():.3f} std={sk.std():.3f}")
    print(f"[capture] full_k    stats min={fk.min():.3f} max={fk.max():.3f} mean={fk.mean():.3f} std={fk.std():.3f}")
    lh = CAPTURE["last_hidden_state"]
    print(f"[capture] last_hidden_state stats min={lh.min():.3f} max={lh.max():.3f} mean={lh.mean():.3f}")


if __name__ == "__main__":
    main()
