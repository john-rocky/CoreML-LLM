"""Quick parity check: our Lfm2MonolithicWrapper vs HF Lfm2ForCausalLM.

Exercises a 1-token decode for several positions in a row and checks that the
top-1 token agrees with HF for each step.  This catches architecture bugs
(e.g. wrong norm placement, missing QK-norm, wrong conv state shift) before
we pay the CoreML conversion cost.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.lfm2 import Lfm2Model
from models.lfm2_wrapper import Lfm2MonolithicWrapper


def main() -> None:
    model_path = "./output/lfm2.5-350m/hf_model"
    ctx = 256

    # 1. Load HF reference in fp32 on CPU.
    # transformers 4.55 doesn't know "TokenizersBackend" (v5 class) so we
    # bypass AutoTokenizer.  It also chokes on the v5-style ``dtype: bfloat16``
    # field (logger.info → __repr__ → JSON dump fails on torch.dtype), so we
    # construct the config from a sanitised dict and load the model class
    # directly without going through AutoConfig/AutoModel.
    import json as _json
    from transformers import PreTrainedTokenizerFast
    from transformers.models.lfm2 import Lfm2Config, Lfm2ForCausalLM
    import safetensors.torch

    with open(os.path.join(model_path, "config.json")) as f:
        cfg_dict = _json.load(f)
    cfg_dict.pop("dtype", None)  # the v5-only torch.dtype echo
    hf_cfg = Lfm2Config(**{k: v for k, v in cfg_dict.items() if k != "architectures"})
    hf = Lfm2ForCausalLM(hf_cfg).to(torch.float32)
    sd = safetensors.torch.load_file(os.path.join(model_path, "model.safetensors"))
    sd = {k: v.to(torch.float32) for k, v in sd.items()}
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    print(f"HF load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print("  missing[:3]:", missing[:3])
    if unexpected[:3]:
        print("  unexpected[:3]:", unexpected[:3])
    hf.eval()

    tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(model_path, "tokenizer.json"),
        bos_token="<|startoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
    )

    prompt = "The quick brown fox"
    ids = tok(prompt, return_tensors="pt").input_ids
    print("prompt ids:", ids.tolist(), "decoded:", tok.decode(ids[0]))

    # 2. Build our ANE model (fp16 weights).
    ours = Lfm2Model.from_pretrained(model_path, context_length=ctx)
    ours.eval()
    wrap = Lfm2MonolithicWrapper(ours)
    wrap.eval()

    # 3. Walk the prompt one token at a time through OUR model.
    #
    # IMPORTANT: HF's ``Lfm2ShortConv.slow_forward`` (CPU decode path) has a
    # bug where ``state.roll(-1)`` followed by an indexed write loses the
    # second-most-recent input.  We confirmed this against HF's own prefill
    # path with a hand-rolled probe: prefill is correct, slow_forward is not.
    # Our implementation (``cat([state[..., 1:], Bx], dim=-1)``) matches the
    # prefill semantics, which is what the trained model expects.
    #
    # So we compare against HF's PREFILL logits (single forward over the
    # whole prompt), step by step.
    n_attn = len(wrap.attn_layer_indices)
    n_conv = len(wrap.conv_layer_indices)
    print(f"n_attn={n_attn}, n_conv={n_conv}, hidden={ours.config.hidden_size}, "
          f"head_dim={ours.config.head_dim}, num_kv_heads={ours.config.num_key_value_heads}")

    with torch.no_grad():
        hf_full = hf(input_ids=ids, use_cache=False)
        hf_logits_all = hf_full.logits[0].float()  # (T, vocab)

    # zero states + initial conv_state (now passed as input/output, not MLState)
    wrap.kv_cache_0.zero_()
    n_conv = len(wrap.conv_layer_indices)
    conv_state = torch.zeros(
        n_conv, ours.config.hidden_size, wrap.conv_l_padded, dtype=torch.float16,
    )

    for pos in range(ids.shape[1]):
        tok_id = ids[:, pos:pos+1]
        hf_logits = hf_logits_all[pos]
        hf_top = int(hf_logits.argmax().item())

        # --- our step
        causal_mask = torch.full((1, 1, 1, ctx), float("-inf"), dtype=torch.float16)
        causal_mask[..., :pos + 1] = 0.0
        update_mask = torch.zeros((1, 1, ctx, 1), dtype=torch.float16)
        update_mask[0, 0, pos, 0] = 1.0
        position_ids = torch.tensor([pos], dtype=torch.int32)

        with torch.no_grad():
            our_id, our_logit, conv_state = wrap(
                tok_id.to(torch.int32),
                position_ids,
                causal_mask,
                update_mask,
                conv_state,
            )
        our_top = int(our_id.item())

        match = "OK" if our_top == hf_top else "MISMATCH"
        print(
            f"pos={pos:3d}  prompt={tok.decode([tok_id.item()])!r:>12s}  "
            f"hf_top={hf_top}({tok.decode([hf_top])!r})  "
            f"ours={our_top}({tok.decode([our_top])!r})  {match}"
        )
        if our_top != hf_top:
            # Print top-5 for both to diagnose how close we are
            hf_top5 = torch.topk(hf_logits, 5)
            print("  HF   top5:", list(zip(hf_top5.indices.tolist(), hf_top5.values.tolist())))


if __name__ == "__main__":
    main()
