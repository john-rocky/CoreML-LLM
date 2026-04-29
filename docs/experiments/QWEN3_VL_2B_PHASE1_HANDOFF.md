# Qwen3-VL 2B — Phase 1 (MLState + multifunction + slice_update) handoff

Target: lift iPhone 17 Pro decode from ~10 tok/s (v1.4.0 current ship) to
**40-55 tok/s** by porting the ANEMLL recipe end-to-end **and getting
the decode/prefill chunks actually on ANE instead of GPU**.

## Current baseline is NOT pure ANE

Running v1.4.0 on iPhone 17 Pro with all 10 chunks (4 decode body +
chunk_head + chunk_0_vision + 4 prefill body + prefill_chunk_0_vision)
shows **phys_footprint ≈ 1.7 GB during inference**. Pure-ANE 2B
(Qwen3.5 2B v1.1.0 precedent) ships at **200–400 MB**. The 1.3 GB
excess is Metal heap — at least the prefill chunks, and possibly some
decode chunks, are silently falling back to GPU even though
`MLComputePlan` reports 93% ANE operator preference. `MLComputePlan`
measures *compile-time operator compatibility*, not *runtime dispatch
device* — A18 ANE's tile-budget scheduler can reject a compiled model
at predict time with no visible error and no log, and the runtime
quietly routes through Metal.

This means:
- 10 tok/s decode is ANE+GPU **mixed**, not a pure ANE floor.
- The 2B fp16-equivalent weight pressure from the matmul-scatter
  KV-write pattern (`(1, 8, max_seq, D)` materialized per layer × 7
  layers) blows ANE's working-set budget on A18 even at T=8. T=32
  failed to compile at all (`MILCompilerForANE Error=(11)`); T=8
  compiles but is spilling live to GPU.
- The Phase 1 rewrite isn't just "match ANEMLL for 30% more tok/s."
  It is the **only route onto pure ANE** for this model class. ANEMLL
  ships Qwen3-1.7B at 47-62 tok/s at ≈250 MB phys_footprint — same
  weight count, same ANE envelope, different dispatch pattern.

Expected Phase 1 landing: **40-55 tok/s decode at 300-500 MB
phys_footprint**, which is the "memory drop" smoke test that confirms
the chunks actually landed on ANE (not GPU).

## Why this work is next

ANEMLL ships Qwen3-1.7B ctx=2048 on the same A18 ANE at **47-62 tok/s**
with the same transformer architecture we run. Our current stack has
three identified gaps, each confirmed by reading their source:

1. **KV cache as `ct.StateType` (MLState) instead of explicit I/O.** We
   copy new_k_N / new_v_N MLMultiArrays in and out every decode step;
   they declare one `state<tensor<fp16,[56,8,2048,128]>>` for all 28
   layers × 2 and use `read_state` / `slice_update` / `write_state`
   inside the graph. No CPU↔ANE KV marshaling per step.
2. **Multi-function mlpackage** — one `.mlpackage` carries `infer`
   (T=1) + `prefill` (T=N) + optional `infer_rotate` / `prefill_rotate`
   sharing the same MLState. We ship them as separate mlpackages with
   separate caches.
3. **`slice_update` instead of matmul-scatter for batched KV write.**
   Our Gemma3-style matmul-scatter materializes a `(1, 8, max_seq, D)`
   intermediate per layer which blew the iPhone ANE tile budget at
   T=32 and forced us down to T=8 (still a 3× improvement, but the
   ceiling). `slice_update` writes only T rows; compile cost and
   runtime memory both drop.

Secondary wins from the same deep-dive (stack on top of the above):

- **In-graph RoPE gather**: bake cos/sin into the mlprogram as fp16
  constants, `gather(cos_table, position_ids)` inside the graph.
  Kills the Swift per-step cos/sin builder. +5–10%.
- **IOSurface-backed preallocated I/O + ping-pong chunk outputs**:
  zero-copy hand-off between chunk N and chunk N+1. +10–15%.
- **4 → 2 chunks** (14 layers each): fewer dispatches. ANEMLL's
  shipping config. Only viable once MLState lands (otherwise
  per-chunk state handles multiply). +10–15%.
- **16-way split LM head**: parallelize the vocab=151936 matmul.
  +5%.

Expected stacked: **10 → 40-50 tok/s**. Last bit to match LiteRT-LM
56.5 tok/s requires speculative decoding on top (Phase 2).

## MLState prognosis for Qwen3-VL 2B specifically

Gemma 4's MLState attempt hit Core ML error -14 because its KV
topology is non-uniform:

- sliding vs full layers have different `head_dim` (256 vs 512)
- `num_kv_shared_layers=20` means 20 layers are Q-only and skip K/V
  entirely
- per-layer PLE augmentation changes state shape

Qwen3-VL 2B has none of that: 28 identical GQA layers, all
`(1, 8, 2048, 128)` fp16, no sliding, no sharing, no PLE. It should
look exactly like ANEMLL's Qwen3-1.7B to the state scheduler.

**Safety check before committing**: convert a 2-layer stub with
MLState, predict on Mac ANE + iPhone ANE. If both compile without
error -14 or "ANEF Error=(11)", proceed with the full rewrite. Takes
~5 minutes.

## Concrete implementation tasks

### Converter (Python — coremltools 8.3, lama-cml env)

1. New file `conversion/build_qwen3_vl_2b_stateful_chunks.py` (fork of
   the current decode converter).
2. Declare state once per chunk boundary:
   ```python
   layers_in_chunk = 14   # 28 / 2
   state = ct.StateType(
       wrapped_type=ct.TensorType(
           shape=(2 * layers_in_chunk,   # K+V interleaved
                  cfg.num_key_value_heads,
                  cfg.max_seq,
                  cfg.head_dim),
           dtype=np.float16),
       name="kv_cache_0")
   ```
3. Rewrite attention to read/write state via `slice_update`:
   ```python
   # In each layer, index into the unified state by layer offset
   layer_k_slice = state[2*li:2*li+1, :, position:position+T, :]
   layer_v_slice = state[2*li+1:2*li+2, :, position:position+T, :]
   ```
   Drop every `k_N` / `v_N` / `new_k_N` / `new_v_N` input/output.
4. Multi-function save — `infer` (T=1) + `prefill` (T=32 or T=64, try
   both on A18). `ct.utils.MultiFunctionDescriptor()` +
   `ct.utils.save_multifunction(...)`. Both functions share the same
   state.
5. Bake RoPE tables as fp16 const, replace the `cos` / `sin` inputs
   with a `position_ids: int32(T,)` input. Inside the graph:
   `gather(cos_table_const, position_ids)`. Keep a separate cos/sin
   pair for vision mRoPE (image tokens) — load a second const table
   and branch via `select` on a `use_vision_rope: bool` input, OR
   feed two position_ids inputs and a mix mask. Simplest: ship the
   stateful build for text-only first, revisit vision mRoPE in a
   follow-up.
6. LM head split 16-way — output `(argmax_idx_i, argmax_val_i)` pair
   per split, Swift does the 16-way max. Converter detail mirrors
   ANEMLL `qwen_converter.py`.
7. Chunk count 2 instead of 4 (14 layers each). Verify compile + ANE
   placement on Mac first, then iPhone.

### Swift (Qwen3VL2BGenerator.swift)

1. Load chunks with `MLModelConfiguration.functionName = "infer"` vs
   `"prefill"`; both point at the same mlpackage URL. iOS 18+ API.
2. `state = model.makeState()` once per generate. Pass via
   `model.prediction(from:, using: state, options:)`. Remove all the
   `lastBodyOuts` / `initialKVFVs` plumbing — state is invisible to
   Swift once created.
3. CVPixelBuffer-backed MLMultiArray for hidden input +
   MLPredictionOptions with output backings pre-wired (see ANEMLL
   `InferenceManager.swift:2312-2321` for the ping-pong pattern).
4. Pre-allocate an `MLDictionaryFeatureProvider` once per chunk and
   reuse it every step.
5. Drop Swift-side RoPE (the tables are baked in).
6. Swift 16-way argmax over the split-head output.

### Vision path adjustments

- `chunk_0_vision`: add DeepStack + `visual_active` on top of the
  stateful recipe. State shape unchanged (14 layers of KV per chunk
  0). DeepStack adds are activations only, don't touch state.
- Vision encoder (`vision.mlmodelc`) unchanged.
- mRoPE for image tokens: either (a) ship stateful text-only first and
  keep recurrent path for vision, or (b) add a second baked RoPE
  table indexed by `rope_axis_ids` so image tokens can pull T/H/W
  per-dim frequencies. (a) is the faster ship; (b) lets vision enjoy
  the same speedup.

## Validation plan

1. **Stub smoke test**: 2 layers, MLState, convert → Mac ANE
   predict → iPhone ANE predict. If -14 or Error=(11), abort and
   diagnose. **This is gate zero.**
2. **Parity**: stateful 28-layer chunk vs current decode chunk on
   a 20-token prompt. Cosine of final hidden ≥ 0.999. Top-1 token
   match ≥ 19/20.
3. **Speed on Mac Studio ANE**: predict latency for 1 decode step.
   Should drop vs current ~30 ms.
4. **Speed + memory on iPhone 17 Pro**:
   - end-to-end tok/s: success = ≥ 25 tok/s, target = ≥ 40.
   - `phys_footprint` during inference: **must drop to < 500 MB**.
     This is the hard ANE-residency signal. If it stays > 1 GB,
     chunks are still on GPU regardless of what `MLComputePlan`
     reports — treat the run as a regression even if tok/s looks
     fine. Qwen3.5 2B v1.1.0 ships at ≈200 MB on identical weight
     count; anything above 500 MB means KV or weights are living
     in Metal heap.
   - Optional corroborator: Instruments → Core ML template, verify
     dispatch device per step reads "Neural Engine" not "GPU".
5. **Vision smoke test**: describe an image end-to-end. Output must
   stay coherent (no "digital corruption" regression).

## Risks

- MLState may still hit a Qwen3-VL-specific ANE quirk that ANEMLL's
  Qwen3-1.7B doesn't trigger. If smoke test fails, fall back to
  `.cpuAndGPU` on the prefill chunk only and keep decode on ANE.
- Multi-function with state sharing is iOS 18+. iOS 17 users don't
  get the speedup; they fall back to the v1.4.0 recurrent+batched
  path. OK as long as we preserve that path.
- 2-chunk consolidation may re-introduce the ANE compile budget
  pressure that pushed us to 4 chunks originally. Mitigation: if
  2-chunk fails, try 3 chunks (Qwen3.5 2B landed at 4; ANEMLL at 2).

## Files likely to change

- `conversion/build_qwen3_vl_2b_stateful_chunks.py` (NEW, ~500 lines)
- `conversion/ane_ops.py` (add `slice_update` helper if it's not
  already lowered cleanly by coremltools)
- `conversion/combine_stateful_multifunction.py` (NEW, ~50 lines —
  mirror `/tmp/anemll_clone/anemll/utils/combine_models.py`)
- `Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen3VL2BGenerator.swift`
  (rewrite `load()`, `stepPredict`, `prefillBatchStep` to use
  `MLState` and drop the KV name-plumbing)
- `Sources/CoreMLLLM/ModelDownloader.swift` (swap file list from
  decode+prefill mlpackages to stateful+multifunction mlpackages)

## Next-session prompt (copy-paste to kick off Phase 1 fresh)

> Implement Qwen3-VL 2B Phase 1 on iPhone ANE. Current v1.4.0 on
> device measures ~10 tok/s decode and **1.7 GB phys_footprint**,
> which is the fingerprint of chunks spilling to GPU Metal heap
> despite `MLComputePlan` reporting 93% ANE operator preference.
> Matmul-scatter KV writes from batched prefill broke A18 ANE's
> tile budget; the runtime silently routed some of the compute
> through Metal. Phase 1's job is to get everything back onto pure
> ANE by porting the ANEMLL recipe end-to-end: `ct.StateType` +
> `MLState` with `slice_update` KV writes in-graph, multi-function
> mlpackage with `infer` (T=1) and `prefill` (T=32 try, fall back
> to T=16/8 if A18 compile fails) sharing the same state, cos/sin
> baked as fp16 const + `gather(table, position_ids)` in-graph,
> 4→2 chunks (14 layers each — ANEMLL shipping config), 16-way
> split LM head, Swift IOSurface-backed MLMultiArray I/O +
> ping-pong output backings between chunks. Branch from `main`
> (v1.4.0 ship with T=8 batched prefill).
>
> Read `docs/QWEN3_VL_2B_PHASE1_HANDOFF.md` first for the full
> plan. Reference implementation lives at
> `/tmp/anemll_clone/` (may need re-clone:
> `git clone https://github.com/Anemll/Anemll /tmp/anemll_clone`);
> the key files are `anemll/ane_converter/qwen_converter.py`,
> `anemll/utils/combine_models.py`, and
> `anemll-swift-cli/Sources/AnemllCore/InferenceManager.swift`.
>
> **Gate zero (~5 min, do this first)**: convert a 2-layer MLState
> stub with the same KV state shape as the full model, compile,
> predict on Mac ANE then iPhone ANE via devicectl. If either hits
> error -14 or `MILCompilerForANE Error=(11)`, STOP and report
> before touching the full converter.
>
> **Success criteria** for the iPhone 17 Pro measurement:
> - decode ≥ 25 tok/s (stretch ≥ 40).
> - `phys_footprint` drops to < 500 MB during inference. **This
>   is the non-negotiable ANE-residency check** — tok/s alone can
>   mask a GPU-resident regression. Qwen3.5 2B v1.1.0 ships at
>   ≈200 MB on the same weight count.
> - Vision smoke test: describe an image end-to-end, no "digital
>   corruption" regression.
>
> Preserve the vision path. DeepStack chunk_0_vision and interleaved
> mRoPE for image tokens can stay on the v1.4.0 recurrent+batched
> path for the first pass; stateful text-only is the ship. Commit
> incrementally (one commit per: stub smoke test, converter, Swift
> rewrite, multi-function combine, chunk consolidation, split LM
> head, IOSurface backings). If a specific optimization doesn't
> compile or regresses memory above 500 MB, revert that commit and
> continue with the rest — each technique is independently shippable.

## References

- ANEMLL (Anemll/Anemll): repo cloned to `/tmp/anemll_clone` during
  the research session.
- ANEMLL Qwen3-1.7B-ctx2048 model card:
  <https://huggingface.co/anemll/anemll-Qwen-Qwen3-1.7B-ctx2048_0.3.5>
- coremltools Stateful Models guide:
  <https://apple.github.io/coremltools/docs-guides/source/stateful-models.html>
- coremltools Multi-function Models guide:
  <https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html>
- Apple On-Device Llama 3.1 recipe (18× speedup from stateful):
  <https://machinelearning.apple.com/research/core-ml-on-device-llama>
