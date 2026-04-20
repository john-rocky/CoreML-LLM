# LiteRT-LM runtime techniques worth adopting for speed and thermal efficiency

**Date:** 2026-04-19
**Scope:** runtime / scheduler features only. Model graph & MTP-head learnings live in
`LITERT_RUNTIME_ANALYSIS.md` and the `MTP_*.md` series.
**Premise:** MTP / separate-architecture speculation is **not working** in this
repository. See `DRAFTER_DEAD_FOR_E2B.md` and `MTP_INTEGRATION_RESULTS.md`. Items
that depend on a working drafter are deferred and listed in §Deferred.

**Implementation status (2026-04-19):** items S1 (router), S2, S4, T1 (LRU
variant), T2, T3, T4, T5 are landed in this PR. S1 (converter),
constant-sharing audit script, and the new ane-residency-gate CLI ship
alongside; conversion artifacts have NOT been re-exported (do that on a
machine with the HF checkpoint and coremltools). S3 (dual-bank KV) and the
in-graph embedding dequant variant of T1 are deferred — both require a
non-trivial converter pass.

---

## What this doc is

A side-by-side comparison of LiteRT-LM's production runtime against `ChunkedEngine`,
filtered to items that move **decode tok/s** or **joule-per-token / sustained tok/s
under thermal load**. Sampling, constrained decoding, tool calling, multi-language
bindings, etc. are out of scope here (see the Engine-wide gap report in chat
history if needed).

Baseline this is measured against: 14.5 tok/s @ ctx 8192 on iPhone 17 Pro
(`BASELINE_SPEED_AUDIT.md`). Goal still 50 tok/s @ 8K (`PRIORITY_ROADMAP.md`).

---

## Speed wins — adoptable without a working drafter

### S1. Multiple precompiled prefill batch sizes (best-match selection)

- **LiteRT-LM:** `AdvancedSettings.prefill_batch_sizes` is an ordered set of
  magic numbers (e.g. {32, 128, 512}). At `Prefill()` the runtime picks the
  smallest size that fits the actual prompt length and dispatches that variant.
- **CoreML-LLM today:** `prefillN = 512` hardcoded in `prefill_chunk*.mlpackage`.
  A 50-token user turn still pays 512-wide compute and bandwidth.
- **Why it's a win on Apple Silicon:** ANE dispatch cost scales with the static
  shape, not the live token count. Padding 50 → 512 wastes ~10× of the prefill
  energy and time.
- **Implementation sketch:**
  - Convert `prefill_chunk{1..4}_b{32,128,512}.mlpackage` (3 size variants).
  - In `ChunkedEngine.runPrefill`, route to the smallest variant ≥ token count;
    loop the variant for the remainder if needed.
  - Adds ~3× model bytes for prefill chunks only; decode chunks unchanged.
- **Expected gain:** TTFT improves several-fold for short turns (chat, agent
  tool replies). Sustained throughput unchanged. Energy per turn drops in lockstep.
- **Effort:** medium (converter + Swift router).
- **Risk:** low. ANE residency identical across variants if shapes are powers of 2.

### S2. Async input/state preparation overlapping the previous step

- **LiteRT-LM:** `AdvancedSettings.sampler_handles_input` lets the sampler
  build the next step's `token / position / mask` tensors while inference is
  still running. This is their "asynchronous inference overlap" knob.
- **CoreML-LLM today:** `ChunkedEngine.predictStep` is fully serial:
  `chunk4` returns → CPU dequantises embedding → builds `update_mask` →
  rewrites `position` → `chunk1` starts. The CPU work between chunks is dead
  time on the ANE.
- **Why it's a win:** the CPU prep window per step is small (~0.5–1.0 ms on
  iPhone 17 Pro: vDSP INT8→FP16 + mask write + position bump). Hiding it behind
  `chunk4` recovers it directly.
- **Implementation sketch:**
  - At greedy temp=0, the next token is determined by `chunk4`'s argmax. We can
    speculatively start preparing inputs for token T+1 the instant `chunk4`
    submits, using the predicted-but-unread output. If wrong (won't happen at
    temp=0 today, but may once sampling lands), discard and rebuild — cheaper
    than the saved time on average.
  - Concretely: split `predictStep` into `submit(t)` and `complete(t)`; run
    `prepareInputs(t+1)` between them on a separate `DispatchQueue` pinned to E-core.
- **Expected gain:** 3–7 % on per-token decode. Modest but cumulative across
  every step and unlocks more once sampling adds work to the CPU prep stage.
- **Effort:** medium. Requires care around the scratch buffer pool
  (`ChunkedEngine.swift:66-74`) — currently single-buffered.
- **Risk:** low/medium. Needs ping-pong scratch buffers.

### S3. Dual-bank KV cache with bank swap

- **LiteRT-LM:** `LitertKVCache` (`runtime/executor/litert/kv_cache.cc`) maintains
  two banks per layer and swaps which one the next decode call writes into.
  No `update_mask` written by the host per step.
- **CoreML-LLM today:** single-bank KV with a per-step `update_mask` (one-hot at
  `current_position`) the host writes into the CoreML graph. Each step pays an
  ANE↔host write of that mask plus the read-modify-write of the live KV slice.
- **Why it's a win:** the mask write is a sequential CPU op on the host that
  also forces a memory barrier with the IOSurface-backed KV buffer. Removing it
  shaves a few hundred µs per step and reduces ANE→CPU traffic (= heat).
- **Implementation sketch:**
  - Author chunk variants whose KV input/output ports are separate buffers
    (`kv_in`, `kv_out`) instead of in-place via mask.
  - Swift side keeps two `MLMultiArray` banks per layer and alternates which is
    `kv_in` vs `kv_out`. Sliding-window layers still need shift logic but the
    write path becomes O(1) append rather than mask-gated update.
- **Expected gain:** 2–5 % decode tok/s. Equivalent reduction in CPU wakes during
  sustained gen → noticeable thermal benefit.
- **Effort:** medium-high. Touches every chunk's KV plumbing.
- **Risk:** medium. Have to re-verify ANE residency stays at 99.78 %.

### S4. Constant-tensor / RoPE / embedding-table sharing across chunks

- **LiteRT-LM:** `AdvancedSettings.share_constant_tensors = true` (default) makes
  the runtime deduplicate identical constants across compiled subgraphs.
- **CoreML-LLM today:** four decode chunks each carry their own RoPE tables and
  any per-chunk constants. Embedding table is external (good) but RoPE
  cos/sin (`cos_sliding.npy`, `sin_sliding.npy`, `cos_full.npy`, `sin_full.npy`)
  may be embedded redundantly inside chunks that need them.
- **Why it's a win:** less working-set pressure on the 32 MB ANE SRAM
  (`PRIORITY_ROADMAP.md` 0g notes the 30 % cliff at 32 MB) and fewer pages to
  keep resident. Sustained throughput improves indirectly via fewer SRAM refills.
- **Implementation sketch:** audit each chunk's constants with
  `coremltools.models.MLModel(chunk).get_spec()`; for any constant ≥1 MB that
  appears in ≥2 chunks, externalise it as an input fed from a shared
  `MLMultiArray`.
- **Expected gain:** ≤2 % decode + measurable reduction in physfootprint.
- **Effort:** low–medium (audit + selective re-export).
- **Risk:** low.

---

## Thermal / sustained-throughput wins

### T1. Pull embedding INT8→FP16 dequantisation into the CoreML graph (or LRU it)

- **LiteRT-LM:** embedding lookup is part of the compiled graph; dequantisation
  runs on the same compute unit as the rest of the layer.
- **CoreML-LLM today:** `EmbeddingLookup.lookup()` does
  `vDSP.convertElements(INT8 → FP32) → multiply by per-token scale →
  vImageConvertPlanarFtoPlanar16F`. **Every decode step, on the P-core.** This is
  measurable heat over a long session and a direct cause of E↔P core migration.
- **Two options:**
  1. **In-graph dequant.** Make the embedder a tiny CoreML model that takes a
     `token_id (1,1)` input, runs gather over a palettised embedding table, and
     emits FP16. Schedule on ANE alongside chunk1.
  2. **LRU cache of recent dequants.** Chats reuse tokens heavily (vocabulary
     of common subword strings). A 256-entry LRU keyed by token_id bypasses
     vDSP for ~70 % of steps in practice.
- **Why it's a thermal win:** removes per-step CPU work from the hot path. The
  ANE is far more J/op than the P-core for this size of work; moving the op to
  ANE both speeds it and cools it.
- **Expected gain:** sustained tok/s under thermal load improves visibly
  (estimate +5–10 % at minute-long generations); peak tok/s ~+1 %.
- **Effort:** option 2 is half a day in Swift; option 1 is a converter change.
- **Risk:** option 1 needs to verify the gather op stays on ANE for a 262 144 ×
  1536 × INT8 table (likely fine — Gemma 4's embedder section is already INT8
  per the LiteRT model graph; see `LITERT_RUNTIME_ANALYSIS.md` §A1).

### T2. ANE residency as a CI gate, not a debug flag

- **LiteRT-LM:** per-component latency breakdown (`BenchmarkInfo`) lands in
  every benchmark run.
- **CoreML-LLM today:** `ComputePlanAudit.swift` is gated by env var
  `COMPUTE_PLAN_AUDIT` and run by hand. Phase-0a pinned residency at 99.78 %
  but there is no regression catch.
- **Why it matters for thermals:** the moment a future change kicks one op off
  ANE onto GPU or CPU, that op runs hotter and slower **and** drags adjacent ops
  with it. We won't notice until the device throttles.
- **Implementation sketch:** add a converter post-step that runs
  `MLComputePlan` against each chunk variant on a Mac in CI and fails the build
  if the per-chunk ANE op fraction drops below 99.5 %. Output the diff against
  the previous baseline.
- **Expected gain:** none directly; protects every other thermal win above.
- **Effort:** low. Reuse `ComputePlanAudit.swift`.

### T3. Bound CPU parallelism during decode

- **LiteRT-LM:** explicit `CpuConfig.number_of_threads` (default 4) and a
  work-stealing pool with deterministic scheduling.
- **CoreML-LLM today:** `withThrowingTaskGroup` is used freely (notably during
  chunk load — fine there). During decode, any incidental concurrency wakes
  P-cores. We don't currently pin or limit.
- **Why it matters:** waking a P-core for a 200 µs job is the worst possible
  energy ratio. Decode prep should stay E-core where possible.
- **Implementation sketch:**
  - `private let decodeQueue = DispatchQueue(label: "decode", qos: .utility)`
    (utility = E-core preferred). Run all per-step Swift prep on it.
  - For prefill chunks where parallelism actually helps, keep the current task
    group.
- **Expected gain:** small on tok/s, real on joules and sustained throughput.
- **Effort:** low. ~40 LoC change in `ChunkedEngine.predictStep`.

### T4. Chunked prefill with mid-stream cancellation

- **LiteRT-LM:** `CpuConfig.prefill_chunk_size` splits a long prompt into N
  sub-batches; each sub-batch checks an `atomic<bool>* cancel` between calls.
- **CoreML-LLM today:** prefill is one shot up to 512 tokens. If the user
  cancels at token 100 of an 8K prompt, we still finish and burn the energy.
  Also, peak power during the 512-wide call drives a sharper thermal spike than
  4 × 128-wide calls would.
- **Why it matters for thermals:** sustained product usage involves cancelled
  generations. Energy on cancelled work is pure overhead. The smaller-batch
  version also flattens the power curve.
- **Implementation sketch:**
  - Combine with S1 above: if we already have a 128-wide variant, decompose
    long prompts into 128-wide passes and check `Task.isCancelled` between them.
- **Expected gain:** product-level — energy on cancelled generations falls to
  near zero; thermal envelope flatter on long prompts.
- **Effort:** low once S1 lands.

### T5. Pre-gated speculation (when speculation is alive at all)

- **LiteRT-LM:** speculation is opt-in; when on, draft step count is fixed and
  acceptance is greedy-argmax. Simple, predictable energy profile.
- **CoreML-LLM today:** `DrafterUnion` runs three sources with static thresholds
  (cross-vocab 0.20, PLD 0.05). When acceptance is poor — which per
  `DRAFTER_DEAD_FOR_E2B.md` is the **typical** state on E2B — every wasted
  verify is ANE energy thrown away.
- **What to adopt now:**
  - Tighten the rolling-EMA gate so the union default is **off** unless live
    acceptance > some honest threshold (e.g. 35 %), with hysteresis so it
    doesn't oscillate.
  - Surface the gate state in the benchmark output so we can see in
    `accept-rate-bench` runs how often speculation actually fires.
- **Why this is on the thermals list:** in the current "drafter is dead"
  steady state, removing the speculative path entirely from the hot loop
  measurably saves energy. The win here is mostly from **not running** the
  drafter, not from running it better.
- **Expected gain:** measurable joule reduction in chat workloads where union
  currently regresses (per `DRAFTER_DEAD_FOR_E2B.md`: 15-21 tok/s vs 32 baseline).
- **Effort:** low. Tweak thresholds in `DrafterUnion.swift`.
- **Risk:** none — strictly conservative.

---

## Deferred (depend on a working drafter / MTP)

The following LiteRT-LM techniques would help, but only after a drafter that
clears the capacity floor exists. Listed here so they aren't forgotten when
revisiting the speculative path:

- **Configurable / adaptive draft-step count K.** LiteRT-LM exposes K as a
  setting; we hardcode K=3. Adaptive K (raise when acceptance is high, lower
  when low) is the obvious extension. Useless until acceptance > break-even.
  See `DRAFTER_DEAD_FOR_E2B.md` for why this is structurally blocked on E2B.
- **MTP-with-per-layer-embedding feed.** LiteRT's MTP drafter consumes
  per-layer hidden states via `EmbeddingLookupManager`. We have the wiring
  (`MtpDraftSource.swift`) but the underlying drafter weights don't predict
  our HF target distribution (`MTP_INTEGRATION_RESULTS.md` §5).
- **Sampler-side speculative input prep.** `sampler_handles_input` overlapping
  *with the verify pass* (separate from S2 which overlaps with chunk4) only
  pays back if speculation actually accepts.

These belong on the roadmap **after** an MTP retrain or comparable success;
they are not standalone speed wins.

---

## Explicitly not adopted

- **LiteRT-LM's GPU/Metal/WebGPU sampler backends** (`runtime/components/sampler_factory.cc`):
  not relevant to ANE-first decode. If we add temperature/top-p we should
  implement on CPU directly given vocab fits in a single Accelerate vDSP pass.
- **`hint_kernel_batch_size` (periodic GPU command flush)**: GPU-only.
- **`gpu_madvise_original_shared_tensors`**: handled implicitly by CoreML's
  weight loader and `mmap` on our embedder file.
- **Work-stealing thread pool**: GCD already covers our use case; reimplementing
  in Swift would not pay back. The discipline we want is T3 (bounding parallelism),
  not a custom pool.

---

## Priority order (ROI = expected J/token & tok/s gain ÷ effort)

| # | Item | Speed | Thermal | Effort | Status |
|---|---|---|---|---|---|
| 1 | **S1** Variable prefill batch size | ◎ TTFT | ○ | medium | router DONE; converter recipe shipped (re-export pending) |
| 2 | **T1** Embedding dequant in graph or LRU | ○ | ◎ | low–medium | LRU DONE; in-graph variant deferred |
| 3 | **T5** Pre-gate speculation off by default | ○ | ◎ | low | DONE |
| 4 | **T2** ANE residency CI gate | — | ○ (regression guard) | low | DONE (`swift run ane-residency-gate`) |
| 5 | **S2** Async input prep overlapping chunks 1-4 | ○ | ○ | medium | DONE |
| 6 | **T3** Bound CPU parallelism in decode | △ | ○ | low | DONE |
| 7 | **T4** Chunked prefill + cancellation | △ | ○ | low (after S1) | DONE (per-chunk checkpoints; loop variant pending S1 re-export) |
| 8 | **S3** Dual-bank KV cache | ○ | ○ | medium-high | DEFERRED (converter required) |
| 9 | **S4** Constant-tensor sharing audit | △ | ○ (SRAM) | low | audit script DONE; results-driven externalisation deferred |

Recommended order to land:
1. **T5** — purely Swift, immediately reduces wasted energy in the current
   "speculation underperforms" steady state.
2. **T2** — protects everything that follows from regressions.
3. **T1 (LRU variant first)** — half-day Swift change with measurable thermal payoff.
4. **S1** — biggest TTFT win; unlocks T4.
5. **S2** — moderate engineering, modest but compounding gain.
6. Re-measure baseline; reassess S3/S4/T1-graph based on what's left.

---

## Cross-references

- Engine-wide gap inventory (sampling, constraints, tool calling, etc.) — chat
  history 2026-04-19 ("LiteRT-LM vs CoreML-LLM 完全ギャップ分析").
- LiteRT-LM model graph & MTP design — `LITERT_RUNTIME_ANALYSIS.md`,
  `LITERT_CONTAINER_ANALYSIS.md`.
- Why separate-architecture drafters are blocked on E2B — `DRAFTER_DEAD_FOR_E2B.md`.
- Current speed baseline & open candidates — `BASELINE_SPEED_AUDIT.md`,
  `PRIORITY_ROADMAP.md`.
- Thermals / power measurements — `POWER_BENCH.md`.
