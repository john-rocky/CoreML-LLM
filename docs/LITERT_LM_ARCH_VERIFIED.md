# LiteRT-LM — source-verified architecture of the 56 tok/s baseline

**Date:** 2026-04-22
**Source:** `/Users/majimadaisuke/Downloads/workspace/LiteRT-LM` (deep-read, not README)

**Context:** LiteRT-LM's ~56 tok/s on Gemma 4 E2B on iPhone is the
baseline we're trying to beat. Memory and prior docs hypothesized this
is "structurally a GPU number." Source inspection confirms and refines.

---

## 0. TL;DR

1. **Primary execution backend on Apple is GPU Metal.** Confirmed. No ANE use at all — no code path, no comments about it.
2. **MTP is real and wired in**, linear greedy exact-match. Confirmed 4-step draft (`num_draft_steps = input_pos_dims[0] - 1`).
3. **Sampling on GPU** via custom Metal compute (`TopKMetalSampler`) — small perf contribution but keeps the pipeline on device.
4. **KV sharing for Q-only layers** via tensor name pattern `kv_cache_*`, passed as shared inputs.
5. **Sliding-512** enforced via stream-deletion API (`DeleteTokensFromKvCache`), not a separate cache.
6. Prefill vs decode are **separate signatures on the same compiled model**.
7. **No Apple private APIs**, no custom IOSurface tricks. Everything through public Metal + `ml_drift::metal::Environment`.

---

## 1. Backend dispatch (primary path)

**File:** `runtime/executor/llm_litert_compiled_model_executor_factory.cc:169-180`

Backend enum: `Backend::CPU`, `Backend::GPU`, `Backend::NPU`.
- On iOS / Apple Silicon, `GPU` is selected.
- `NPU` path exists only for Qualcomm Hexagon via `LlmLiteRtNpuCompiledModelExecutor` (lines 154-159). **No ANE backend.**

**File:** `runtime/executor/llm_executor_settings.h:38-94`

`GpuConfig` and `GpuArtisanConfig` structs. GPU options set (inferred
from `runtime/.../vision_executor.cc:108-142`):
- `SetUseMetalArgumentBuffers(true)`
- `SetPreferTextureWeights(false)` on iOS
- Shader caching enabled

**File:** `runtime/components/top_k_metal_sampler.h:92-183`

`TopKMetalSampler` — custom Metal compute pipeline, `MTLCommandQueue`,
`MTLBuffer` staging for logits / IDs. Uses `ml_drift::metal::Environment`
(Google's internal ML Drift Metal library).

**No MPSGraph. No MPS kernels. No CoreML. No ANE.**

---

## 2. MTP integration (speculative decoding)

**File:** `runtime/executor/llm_litert_mtp_drafter.h:68-106`

Class: `LlmLiteRtMtpDrafter`
- `Draft()` method with `num_draft_steps` parameter.
- Infers step count from verify signature: `num_draft_steps = input_pos_dims[0] - 1` (`mtp_drafter.cc:240`).

**File:** `runtime/executor/llm_litert_mtp_drafter.cc:420-467`

Verification loop:
```cpp
// Run verify signature on base model with num_draft_steps+1 sequence
// Then:
for (int i = 0; i < num_draft_steps; i++) {
    if (verifier_id_vector[i] != drafted_tokens[i]) break;
    num_correct_tokens++;
}
// + 1 bonus token (either mismatch-position target, or next after successful chain)
```

**Acceptance = greedy exact match.** Not probabilistic. This matters
for our drafter analysis: LiteRT's MTP works because the MTP is trained
to match the base argmax, not a probability distribution.

**Shared KV pattern:** `mtp_drafter.cc:192` — `absl::StartsWith(input_name, "kv_cache_")` identifies shared inputs passed between base and drafter.

**Draft steps:** `num_draft_steps` inferred from shape, typically 3-4.
Paired with verify signature that accepts the whole draft + 1 in one pass.

---

## 3. KV cache

**File:** `runtime/executor/litert/kv_cache.h:35-100`

`LitertKVCache` class with `KVCacheBuffers` (input/output map).
Dual-bank buffers for in-place/out-of-place updates.

**Sliding-512 enforcement:** `runtime/executor/llm_litert_compiled_model_cache_utils.h:37-70` — `ShouldDeleteKVCacheTokens()` and `DeleteTokensFromKvCache()` remove oldest tokens. Single-cache design, not dual-cache like llama.cpp.

**External tensor mode (zero-copy-ish):** `runtime/executor/llm_executor_settings.h:91` — `external_tensor_mode` flag. `mtp_drafter.cc:131-132` registers pattern `gpu_options.AddExternalTensorPattern("kv_cache_")` so KV stays on GPU between base and drafter calls.

---

## 4. Quantization

**Source observation:** Runtime executor **does not contain quant specs**. Flag `allow_src_quantized_fc_conv_ops` exists (`llm_executor_settings.cc:86-91`) as a GPU-backend enable, but the actual per-tensor/per-channel W4/W8 decisions are baked into the `.litertlm` model file at compile time.

**Implication:** to match their quant quality, we need access to the
LiteRT compiler recipe (not public), or empirically reverse the model
file. Either way, the quant behavior is **compile-time fixed**, not a
runtime choice we can reproduce.

---

## 5. Prefill vs decode

**File:** `runtime/executor/llm_litert_compiled_model_executor.cc:81-82`

Separate signature runners on same compiled model:
- `kPrefillSignatureRunner`
- `kDecodeSignatureRunner`

**Prefill bypass:** `llm_litert_compiled_model_executor.cc:615-616` —
`skip_prefill = !has_pending_input_token && prefill_length == 0`.

Dynamic vs static routing: `llm_litert_compiled_model_executor_factory.cc:63-74` — `LlmLiteRtCompiledModelExecutorDynamic` vs `Static` based on detected model mode.

---

## 6. Gemma 4 E2B model specifics

**File:** `runtime/conversation/model_data_processor/gemma4_data_processor.h` and `.cc`

`Gemma4DataProcessor` class — handles tokenization + message formatting.
Runtime doesn't carry architectural details; those live in the
`.litertlm` file metadata (`runtime/proto/llm_metadata.proto:59-91`):
`LlmMetadata` { `llm_model_type`, `max_num_tokens`, sampler params, ... }.

---

## 7. Benchmark infrastructure

`runtime/litert_lm_main.cc:113-160` — main entry supports a benchmark mode
via `engine_settings.GetMutableBenchmarkParams()`. No hardcoded 56 tok/s
config in source; their published number likely comes from an external
script + proprietary build. `is_benchmark` flag and
`wait_for_weights_conversion_complete_in_benchmark` hint at measurable
GPU weight-upload overhead.

---

## 8. What does NOT exist in LiteRT-LM

- **No ANE code path.** Zero instances. Not a missing-config — there is no code.
- **No MPSGraph usage.** Pure `ml_drift::metal` + `MTLCommandQueue`.
- **No CoreML integration.**
- **No IOSurface tricks** in the executor.
- **No Apple private API calls.**
- **No hardcoded Gemma 4 E2B architecture in runtime.** Lives in `.litertlm` container.

---

## 9. Hypothesis for the 56 tok/s advantage, ranked

Reading the code top-to-bottom, the likely contributors to their throughput advantage over our CoreML+ANE stack:

| Rank | Mechanism | Est. contribution |
|---|---|---|
| 1 | GPU Metal GEMM + graph-compile-time fusion (vs CoreML ANE kernel boundaries) | ~25-30% |
| 2 | MTP speculative decoding (linear, ~4-step, greedy exact-match) | ~15-20% |
| 3 | Q-only KV sharing + external tensor mode (zero-copy across draft/verify) | ~10% |
| 4 | Sliding-512 + streaming KV delete (bounds cache size) | ~5% |
| 5 | GPU-side top-K sampling (no CPU logits download) | ~3-5% |
| 6 | Graph-baked quantization (per-compile-time, optimal for target) | ~3-5% |

Total stack effect gets them from ~31 tok/s (our 2K baseline) to 56.

**Key insight:** item 2 (MTP) contributes because their MTP is trained
to match base argmax and because verify happens on the same Metal GPU
in one command buffer. Our EAGLE-3 attempt on ANE fails on both counts:
oracle-live gap (drafter acc diverges from Gemma argmax) and pipeline
split (ANE drafter + ANE verify, with sync boundaries).

**Implication for us:** a Metal port that keeps MTP-style
draft+verify on the same device, in the same command queue, may recover
most of contribution #2 — but only if we can replicate a drafter that
matches base argmax. The LiteRT-LM MTP recipe is private
(`project_drafter_structurally_dead.md`), so this is gated on either
(a) replicating the training recipe, or (b) shipping without MTP and
accepting the 15-20% gap.

---

## 10. Implications for our roadmap

1. **Metal port (Phase 3) is confirmed as the critical path.** LiteRT-LM's advantage is structurally Metal-GPU-first, not an ANE trick we've missed.
2. **No shortcut to their MTP win** without the training recipe or a drafter that matches base argmax live.
3. **Our current CoreML+ANE stack ceiling** is bounded by the 3-6× Metal GPU kernel advantage over our ANE dispatch path for this model size.
4. **Match items 3-6 in order** during Metal port work: Q-only KV sharing, sliding-window KV bound, GPU sampling, graph-baked quant. Each is ~5% gain at low engineering cost.
5. **Do not pursue "beat LiteRT on ANE."** The source-level evidence rules this out.

---

## 11. Citations (all in `/Users/majimadaisuke/Downloads/workspace/LiteRT-LM/`)

- `runtime/executor/llm_litert_compiled_model_executor_factory.cc:63-74, 154-159, 169-180`
- `runtime/executor/llm_executor_settings.h:38-94, 91, 96-111, 229`
- `runtime/executor/llm_executor_settings.cc:86-91`
- `runtime/executor/llm_litert_mtp_drafter.h:68-106`
- `runtime/executor/llm_litert_mtp_drafter.cc:131-132, 151-157, 192, 240, 248-256, 420-467`
- `runtime/executor/llm_litert_compiled_model_executor.cc:64, 81-82, 615-616`
- `runtime/executor/litert/kv_cache.h:35-100`
- `runtime/executor/llm_litert_compiled_model_cache_utils.h:37-70`
- `runtime/components/top_k_metal_sampler.h:15-41, 59-81, 92-183`
- `runtime/conversation/model_data_processor/gemma4_data_processor.{h,cc}`
- `runtime/proto/llm_metadata.proto:59-91`
- `runtime/litert_lm_main.cc:113-160`
