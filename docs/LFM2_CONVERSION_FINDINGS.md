# LFM2 / LFM2.5 conversion — Mac ANE findings

**Date:** 2026-04-28
**Status:** ✅ shipping on iPhone 17 Pro at **52 tok/s** decode, ANE-resident,
correct generation.  fp16 build, conv state passed as input/output tensor.
**Model:** `LiquidAI/LFM2.5-350M` (16 layers, 6 attention + 10 short-conv,
`hidden=1024`, `num_kv_heads=8`, `head_dim=64`, `vocab=65536`,
`conv_L_cache=3`, `tie_embedding=True`).

## Pipeline

| File | Role |
|---|---|
| `conversion/models/lfm2.py` | Weight loader (HF → ANE Conv2d layout) |
| `conversion/models/lfm2_wrapper.py` | Monolithic 1-token decode wrapper |
| `conversion/test_lfm2_parity.py` | HF prefill ≡ our wrapper, top-1 token agreement |
| `conversion/bench_lfm2_mac.py` | Mac decode benchmark across compute units |

## Architecture summary

LFM2 layers alternate between two operator types (per `layer_types` in
`config.json`):

* `full_attention` — GQA, no attention bias, RoPE θ=1e6, Q/K-RMSNorm
  applied **before** RoPE, `out_proj` (not `o_proj`).
* `conv` — `Lfm2ShortConv` block:
  ```
  in_proj (h → 3h)  →  split B,C,x along channel  →  Bx = B*x  →
  depthwise causal Conv1d(kernel=L_cache=3, groups=h)  →  C * conv  →
  out_proj (h → h)
  ```
  The conv layer keeps a `(hidden, L_cache)` rolling window of past `Bx`
  values as its only "cache".

Norms: `operator_norm` (pre-attn / pre-conv), `ffn_norm` (pre-MLP),
final `embedding_norm`. MLP is SwiGLU (`w1`=gate, `w3`=up, `w2`=down).
**Intermediate-size adjustment** (`block_auto_adjust_ff_dim=True`) shrinks
the trained `intermediate_size=6656` to **4608** — easy to miss; the HF
MLP applies the same transform internally.

## HF reference quirk: `Lfm2ShortConv.slow_forward` is buggy

The CPU decode path in HF transformers v4.55:

```python
conv_state = state.roll(shifts=-1, dims=-1)
conv_state[:, :, cache_position] = Bx
```

silently drops one tap of history at every step. We confirmed this with a
hand-rolled probe: HF's PREFILL (real `Conv1d`) and HF's `slow_forward`
DECODE diverge from `pos=1` onwards. Compare against HF's prefill, not
its decode, when validating an autoregressive port. Our wrapper uses
`cat([state[..., 1:], Bx], dim=-1)` which matches prefill exactly.

## ANE blocker: dual-state CoreML programs

Initial design used **two MLState buffers** — `kv_cache_0` for attention
and `conv_cache_0` for the short-conv rolling window. On macOS 26.0.1 /
M-series this hits

```
ANEProgramProcessRequestDirect() Failed with status=0x1d
: statusType=0x9: Program Inference error
```

at predict() time, even though compile succeeds. We bisected the error:

| variant | ANE result |
|---|---|
| 1 state (`kv_cache_0`), conv layers skipped | OK |
| 1 state, conv layers run on **zero state every step** | OK |
| 2 states, conv state read-only (no write) | FAIL |
| 2 states, separate `conv_cache_{0..9}` (one per layer) | FAIL |
| 2 states, `slice_update` on dim 0 (KV pattern) | FAIL |
| 2 states, mask-based update via shift matmul | FAIL |
| 2 states, padded innermost dim (3 → 16) | FAIL |
| 2 states, rank-3 instead of rank-4 | FAIL |
| 2 states, `iOS18` deployment target | FAIL |
| **conv state as input/output tensor (no MLState)** | **OK** |

The dual-state hazard is also flagged in
`conversion/models/gemma4_swa_stateful_chunks.py:961` for the gemma4
sliding+full KV pair on iPhone — same workaround there (collapse into one
unified buffer).

### Working layout

* `kv_cache_0` (rank-4 MLState) — 6 attention layers × 2 (K + V) × 8 KV
  heads × 2048 ctx × 64 head_dim ≈ 24 MB.
* `conv_state_in` / `conv_state_out` — rank-3 fp16 tensor input/output
  pair, shape `(n_conv=10, hidden=1024, L_padded=16)` ≈ 320 KB. The
  runtime feeds the previous step's `conv_state_out` as the next step's
  `conv_state_in`.

The conv kernel is widened from `(hidden, 1, 1, L_cache=3)` to `(hidden,
1, 1, L_padded=16)` with zero-padding past the live taps — the depthwise
conv reads the full padded window with no slice op, semantically
identical to the original L_cache=3 conv. The shift step is a fixed
`(L_padded, L_padded)` matmul + one-hot Bx insert at slot `L_cache - 1`,
both built once at trace time.

`L_padded=16` is overridable via `LFM2_CONV_L_PAD`. Probe envs:
`LFM2_PROBE_SKIP_CONV=1` to drop conv layers entirely and
`LFM2_PROBE_NO_CONV_STATE=1` to feed zeros instead of the rolling state
(diagnostic only — wrong past pos=0).

## Decode benchmarks (LFM2.5-350M, fp16, ctx=2048, L_pad=3)

Mac (M-series, 64 decode steps via `bench_lfm2_mac.py`):

```
config                         decode tok/s
CPU+ANE                                42.3
CPU only                              ~57
```

iPhone 17 Pro CoreMLLLMChat single-turn:

```
decode tok/s
       52.0
```

ANE wins on iPhone once we run with `L_pad=3` (i.e. equal to
`conv_L_cache=3`).  An earlier `L_pad=16` build added 13 zero-weight
taps per conv layer — the math was identical (zeros multiply away) but
those extra taps fed enough fp16 reduction noise into the depthwise
conv that autoregressive output collapsed to "kingkingking…" within a
few tokens.  Dropping the padding fixed both correctness and ANE speed
in one go; the fp32 workaround we briefly shipped is no longer needed.

INT4 palettization runs on CPU but currently fails on ANE (the
palettize pass on the depthwise conv kernel confuses the iOS-26 ANE
planner).  Use fp16 for ANE, INT4 for CPU.

### ANE residency

`python conversion/audit_ane_residency.py output/lfm2.5-350m/bundle/model.mlmodelc`
on the `L_pad=3` fp16 build reports **97.8 % ANE** (970/992 non-const
ops).  The 22 CPU-resident ops are:

* **10 × `ios18.conv`** — the depthwise short-conv kernel
  (`kernel=(1, 3)`, `groups=hidden=1024`).  ANE rejects depthwise convs
  where `groups == in_channels` at this width, so each conv layer
  bounces to CPU.  Boundary cost is ~5 ms × 10 = 50 ms/token.
* 12 × periphery ops (argmax, RoPE gather, conv-state stack/squeeze,
  in-graph cast/select).

The depthwise CPU bounces are why CPU-only is ~30 % faster than CPU+ANE
on a Mac (M-series CPU is fast enough that the round-trip dominates).
On iPhone 17 Pro the ratio inverts: ANE matmul throughput beats the
A19 CPU by enough that 52 tok/s end-to-end on CPU+ANE clears whatever
CPU-only alone would deliver — confirmed in-app.

## Swift runtime integration

* `ModelConfig` reads `num_hidden_layers`, `layer_types`,
  `lfm2_conv_l_pad` from `model_config.json`; counts conv layers from
  `layer_types`.
* `CoreMLLLM.load` allocates a `(n_conv, hidden, L_pad)` fp16 buffer
  the first time it sees a `conv_state_in` input, zeroes it, and feeds
  it on every prediction.  The model's `conv_state_out` is byte-copied
  back into the buffer before returning.  `reset()` zeroes it.
* Compute units honour the caller (`.cpuAndNeuralEngine` from the chat
  app).  Set `LLM_LFM2_USE_CPU=1` to force `.cpuOnly` for benchmarking
  fallback paths.
* Per-architecture EOS set in the decode loop now includes
  `config.eosTokenId` and (for LFM2) token 2 (`<|endoftext|>`), so the
  ChatML `<|im_end|>` (token 7) actually terminates generation instead
  of leaking into the rendered text.

## iPhone sideload

`conversion/build_lfm2_bundle.py` produces a sideload-ready directory:

```
output/lfm2.5-350m/bundle/
  ├── model.mlmodelc          # compiled via xcrun coremlcompiler
  ├── model_config.json       # patched with lfm2_conv_l_pad + tokenizer_repo
  └── hf_model/
      ├── tokenizer.json
      ├── tokenizer_config.json   # TokenizersBackend → PreTrainedTokenizerFast
      └── ...
```

The tokenizer_config sanitisation is required: HF's upstream config
uses ``"tokenizer_class": "TokenizersBackend"`` (a transformers v5 class)
which neither swift-transformers nor transformers ≤ 4.x accepts.  We
rewrite the field to ``PreTrainedTokenizerFast``; the underlying
``tokenizer.json`` is untouched.

## Reproduction

```bash
# 1. Convert (one-time, ~5 minutes)
./.venv-ss/bin/python conversion/convert.py \
    --model lfm2.5-350m --context-length 2048 --quantize none \
    --output ./output/lfm2.5-350m-fp16

# 2. Verify HF parity (top-1 token match for 5 prompt steps)
./.venv-ss/bin/python conversion/test_lfm2_parity.py

# 3. Mac decode bench (CPU / GPU / ANE comparison)
./.venv-ss/bin/python conversion/bench_lfm2_mac.py

# 4. Build the iPhone sideload bundle
./.venv-ss/bin/python conversion/build_lfm2_bundle.py \
    --model-path ./output/lfm2.5-350m/hf_model \
    --source-mlpackage ./output/lfm2.5-350m-fp16/model.mlpackage

# 5. Sideload to a connected iPhone (CoreMLLLMChat must be installed)
DEVICE=$(xcrun devicectl list devices | awk '/connected/{print $3}' | head -1)
xcrun devicectl device copy to --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier com.example.CoreMLLLMChat \
    --source ./output/lfm2.5-350m/bundle \
    --destination Documents/Models/lfm2.5-350m \
    --remove-existing-content true
```
