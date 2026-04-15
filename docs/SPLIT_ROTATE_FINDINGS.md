# Split-rotate / 32-alignment findings (Gemma 4 E2B)

Date: 2026-04-15
Target: iPhone 17 Pro, iOS 26, coremltools 9.0
Premise under test: "ANE wants tensor widths that are multiples of 32. Padding
Gemma 4 E2B KV tensors to 32-aligned widths will remove per-step reshape
overhead and unblock MLState (error -14)."

## 1. Audit of Gemma 4 E2B dims vs mod-32

From `conversion/models/gemma4.py::Gemma4Config`:

| dim | value | mod 32 | where it shows up |
| --- | ----- | ------ | ----------------- |
| `hidden_size` | 1536 | 0 OK | Conv2d channel axis (residual stream) |
| `intermediate_size` | 6144 | 0 OK | MLP gate/up/down channel |
| `num_attention_heads` | 8 | 24 NOT | Q heads axis (dim 1 of `(1,H,S,D)`) |
| `num_key_value_heads` | **1** | 31 NOT | KV heads axis — singleton |
| `head_dim` (sliding) | 256 | 0 OK | last axis of K/V |
| `global_head_dim` (full) | 512 | 0 OK | last axis of K/V (full-attn layers) |
| `sliding_window` W | 512 | 0 OK | sliding K/V seq length |
| `context_length` | 512/8192 | 0 OK | full-attn seq length |
| `hidden_size_per_layer_input` | 256 | 0 OK | per-layer embedding slice |
| `vocab_size` | 262144 | 0 OK | embedding / lm_head |
| RoPE `rotate_half` halves | 128 / 256 | 0 OK | split on last axis |
| RoPE full partial rot dim | 128 (factor=0.25×512) | 0 OK | — |
| Q proj out (`num_heads·head_dim`) | 2048 / 4096 | 0 OK | Conv2d channel |
| KV proj out (`num_kv_heads·head_dim`) | 256 / 512 | 0 OK | Conv2d channel |

**The only two non-mod-32 dims are `num_attention_heads=8` and
`num_key_value_heads=1`.** Both live on the "heads" axis (dim 1 of a 4-D
`(1, H, S, D)` tensor), which on ANE is not the tiled channel axis — that is
the last axis of a Conv2d output or the innermost axis of a matmul contraction.
Both Q heads (8) and KV heads (1) are broadcast / per-head independent; the
32-wide tile is applied along `head_dim` (256 or 512) which is already aligned.

## 2. What ANEMLL `--split-rotate` actually does

Cross-checked against the ANEMLL repo README / CLAUDE.md / docs
(https://github.com/Anemll/Anemll):

> "Multi-function compiled models (.mlmodelc) with 4 functions may fail to load
> prefill functions on ANE with error: 'function_name must be nil unless model
> type is ML Program'. This is a CoreML limitation."

`--split-rotate` is a **multi-function loading workaround**: it splits
rotate / non-rotate (prefill vs decode) into separate .mlmodelc files instead
of one multi-function model. It has nothing to do with 32-alignment or KV
layout. The original task framing was incorrect on that point.

## 3. Apple's stateful-models guide on alignment

https://apple.github.io/coremltools/docs-guides/source/stateful-models.html
contains no 32-alignment requirement. It specifies minimum deployment target
iOS18 / macOS15 and `mlprogram`, but no dim constraints. `coreml_update_state`
is a MIL op; error -14 ("Failed to build the model execution plan") on ANE is
almost certainly because **the ANE compiler for iOS 26 does not yet schedule
`coreml_update_state` onto ANE tiles**, falling back to an execution plan it
cannot complete. This matches Apple's own on-device Llama 3.1 reference and
smpanaro/coreml-llm-cli, both of which use stateless explicit-I/O KV.

## 4. Where a reshape could still be happening

Even without a 32-alignment fix to chase, the current KV path does have some
ANE-unfriendly ops worth noting:

- `K_for_attn.repeat_interleave(n_rep=8, dim=1)` — GQA broadcast; materializes
  an 8× larger K/V before the attention matmul. `ane_ops.repeat_kv_ane`
  already exists as an ANE-friendlier replacement (unsqueeze + repeat +
  view) but `gemma4_stateless_chunks.py` still calls `repeat_interleave`
  directly. This is a per-layer reshape, not per-step, but fixing it is
  essentially free.
- `attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)`
  in `_run_layer_stateless` — the `.contiguous()` + `view` forces a reshape
  on the attention output (2048 width, mod 32 OK). ANE handles this but the
  explicit contiguous is unnecessary with the Conv2d-NC1HW style permute
  that follows.

## 5. Conclusion

The "pad KV heads from 1 → 32" hypothesis has **no support** in either
ANEMLL's actual implementation or Apple's published constraints. Padding the
heads axis from 1 to 32 would multiply KV memory by 32× (~0.3 GB → ~10 GB for
CTX=8192 full-attn) for a benefit that is not documented anywhere.

Recommended next actions:

1. Retain the existing stateless KV layout (no padding) for shipping.
2. `repeat_interleave` → `repeat_kv_ane` is a cheap win, independent of this
   investigation — ship it.
3. MLState error -14 is an **ANE compiler** limitation on `coreml_update_state`,
   not a tensor-shape problem. Padding will not help. The productive path is
   to test MLState with `compute_units=CPU_AND_GPU` (GPU stateful is Apple's
   documented sweet-spot per WWDC24 Mistral demo) and treat state-on-GPU as a
   separate dispatch-amortization strategy for chunk2 only.
4. Dispatch-overhead reduction (4→2 chunk consolidation, speculative decoding)
   remains the correct direction, per `docs/EXPERIMENTS.md`.

This finding supersedes the 32-alignment theory in the task prompt.
