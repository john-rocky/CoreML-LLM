# Handoff — Stage 8: Stateful multimodal via prefill→decode state bridge

**Created:** 2026-04-28
**Branch suggestion:** `stage8-mlstate-multimodal`
**Predecessor**: Stage 6 (`stage6-multimodal-stateful` branch — the
splice/mask Swift work, salvageable) and Stage 7 (PR #145, merged
`v1.7.0`, ships 3-chunk recurrent decode + 4-chunk prefill default).
**Probe basis**: `docs/MLSTATE_MULTIMODAL_PROBE.md` (Stage 8 PRE-WORK
that proved the architecture is feasible — read this first).
**Goal**: stateful Gemma 4 E2B (MLState + slice_update KV + cross-turn
KV reuse) gains image / audio / video so `Gemma4StatefulEngine` can
replace the legacy multimodal bundle as the unified default.

---

## Why this is needed

Stage 3 shipped the stateful Linear 3-chunk path as text-only because
multifunction `prefill_bN` with dual MLState was rejected by iPhone
ANE 18 (`ANECCompile FAILED 11`). With T=8 batched prefill the model
can't see the full 256-token image-pad span at once, breaking the
bidirectional within-vision-group attention that Gemma 4 was trained
with.

Stage 6 confirmed this empirically (image prompts parroted back).
Stage 7 retreated to legacy 4-chunk recurrent prefill (T=1024,
correct vision attention) + 3-chunk recurrent decode as a pragmatic
ship.

Stage 8 PRE-WORK (probes 1 and 2 on `mlstate-multimodal-probe` branch,
2026-04-28, iPhone 17 Pro) showed:
1. **Single-function** stateful prefill at T=288 compiles on iPhone
   ANE 18 (7.3 s, no `ANECCompile FAILED`). The wall is specific to
   *multifunction* T>1 stateful, not to single-function.
2. **MLState buffer bridging** — `state.withMultiArray(for:) { ... }`
   gives a CPU-readable view of the underlying KV buffer; a memcpy
   between two states' buffers (via nested `withMultiArray` closures)
   produces a verbatim copy. Tested on iPhone 17 Pro, MATCH.

Together these unblock a clean architecture: a separate prefill
mlpackage (T=288 single-function) processes the full image span in
one forward, then a CPU-side memcpy bridges the KV state into the
decode mlpackage (T=1, existing Stage 3 chain). Multimodal works AND
stateful's wins (cross-turn KV reuse, multi-turn TTFT -95%) apply to
image chat.

---

## Architecture

```
                  ┌────────────────────────────────────────┐
                  │ prefill mlpackage (single function)    │
                  │   T = 288 (or as low as fits image span+text margin) │
                  │   MLState { kv_cache_sliding,          │
                  │             kv_cache_full }            │
                  │   inputs:  hidden_states (1, T, H)     │
                  │            causal_mask_full (1,1,T,ctx)│
                  │            causal_mask_sliding (1,1,T,W)│
                  │            per_layer_raw / cos / sin / …│
                  │            current_pos = 0, ring_pos=0 │
                  │   inside: vision-aware mask unmasks    │
                  │           the 256-tok image-pad run    │
                  │           (HF Gemma4 token_type_ids    │
                  │           bidirectional behavior)      │
                  │   slice_update writes K/V at [0, T-1]  │
                  └────────────────────┬───────────────────┘
                                       │
                          withMultiArray(for: stateName)
                          memcpy src.dataPointer → dst.dataPointer
                          (nested closure, fp16 buffer copy)
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │ decode mlpackage (existing Stage 3)    │
                  │   T = 1, MLState { sliding, full }     │
                  │   continues from current_pos = T       │
                  │   per-token chunked decode chain       │
                  │   (chunk_1 → chunk_2 → chunk_3)        │
                  └────────────────────────────────────────┘
```

State buffer dimensions (Gemma 4 E2B, ctx=2048, W=512, H=1536, HKV=4):
- `kv_cache_sliding`: `(2*ns, HKV, W, max_hd)` per chunk where
  `ns = max(num_sliding_layers_in_chunk, 1)`. For chunk_1 (L0-7),
  num_sliding=7, num_full=1, so ns=7 → shape `(14, 4, 512, 512)`.
- `kv_cache_full`: `(2*nf, HKV, ctx, max_hd)` per chunk. For chunk_1,
  nf=1 → shape `(2, 4, 2048, 512)`.

Both prefill and decode declare identical StateType (same chunk class
in `gemma4_swa_stateful_chunks.py`), so memcpy is shape-safe.

---

## Implementation plan

### Step 1 — Build prefill chunks at T=288 single-function

**Reference**: probe builder
`conversion/probe_stateful_singlefunc_prefill.py` already produces
`chunk_1_prefill_T288.mlpackage` (148.6 MB INT4 g32). Extend for
chunks 2 and 3 (3-chunk merged variant) AND chunks 4 (4-chunk variant
if we want feature parity).

**Files to touch**:
- New: `conversion/build_gemma4_stateful_singlefunc_prefill.py`
  (mirror `build_gemma4_e2b_stateful_chunks.py` but skip the
  `MultiFunctionDescriptor` merge step — emit each chunk's prefill
  as a standalone mlpackage).
- Re-use existing classes:
  - `SWAStatefulChunk1Prefill` — already exists in
    `conversion/models/gemma4_swa_stateful_chunks.py`
  - `SWAStatefulChunk2Prefill` — exists
  - `SWAStatefulChunk3Prefill` — exists (3-chunk: stateless KV-shared)
  - `SWAStatefulMergedChunk23Prefill` — exists for the 3-chunk merged
    middle variant. Use this for chunks_3way prefill.
  - `SWAStatefulChunk4Prefill` — exists (4-chunk: lm_head + argmax)
- CLI: same flags as the existing builder but `--separate-prefill`
  (default true) skips the multifunction merge.

**Output naming convention** (suggested):
- 4-chunk variant:
  - `chunk_1_prefill_T288.mlpackage`
  - `chunk_2_prefill_T288.mlpackage`
  - `chunk_3_prefill_T288.mlpackage`
  - `chunk_4_prefill_T288.mlpackage`
- 3-chunk merged variant (preferred — matches Stage 3 ship):
  - `chunk_1_prefill_T288.mlpackage`
  - `chunk_2_3way_prefill_T288.mlpackage` (uses
    `SWAStatefulMergedChunk23Prefill`)
  - `chunk_3_prefill_T288.mlpackage`

**Verification**: each mlpackage produces ANE placement >90% per the
coremltools planner output line `ANE placement: NNNN/MMMM (XX.X%)`.
Mac compile via `xcrun coremlcompiler compile` succeeds. iPhone-side
ANE re-compile: trigger via `MLModel(contentsOf:configuration:)` with
`.cpuAndNeuralEngine`; expect compile time on the order of 5-15 s for
each chunk on A19 Pro. Probe 1 already validated chunk_1 — chunks 2/3
have larger graphs (more layers per chunk) and need an explicit
re-probe.

### Step 2 — Refactor `Gemma4StatefulEngine.swift` for prefill+decode dual-load

The Stage 3 engine loads the chunked decode mlpackages and uses
multifunction prefill_bN when present. Stage 8 changes:

1. Add storage for prefill models (separate from decode models):
   ```swift
   private var chunk1Prefill288: MLModel?
   private var chunk2Prefill288: MLModel?  // or chunk2_3way_prefill
   private var chunk3Prefill288: MLModel?
   private var prefillT: Int { 288 }       // hard-coded for Stage 8
   ```
2. `load(modelDirectory:)`: detect `chunk_*_prefill_T*.mlmodelc` files,
   load with `cfg.computeUnits = .cpuAndNeuralEngine`. Don't fail
   loading if absent — fall back to existing T=1 prefill loop.
3. New private method `prefillStateful288(inputIds:imageFeatures:audioFeatures:...)`:
   - Build `hidden_states` shape `(1, T=288, H)` with image/audio
     feature splice at IMAGE/AUDIO/VIDEO_TOKEN_ID positions, embed
     lookup at text positions. Same logic as Stage 6's `step()` /
     `prefillStep()` splice but extended to T=288.
   - Build `causal_mask_full` and `causal_mask_sliding` with the
     vision-aware bidirectional unmask within the image-pad group
     (see Stage 6's `fillBatchMasksVisionAware` — copy verbatim,
     adjust shape to `(1,1,288,ctx)` and `(1,1,288,W)`).
   - Run prefill chunks in sequence with `MLState` instances created
     from the prefill models.
   - After prefill: bridge state buffers to the decode-side MLStates
     (see Step 3 below).
   - Return the position pointer (=T, e.g. 288 for the next decode
     step).
4. Modify the existing `generate(...)` flow:
   - When prefill T=288 chunks are loaded AND prompt has multimodal
     OR length>=288, route through `prefillStateful288`.
   - Else fall back to existing T=1 prefill loop or T=8 multifunction
     (text-only).

**Stage 6 work to salvage** (from `stage6-multimodal-stateful`
branch):
- `loadMultimodalEncoders` — vision/video/audio mlmodelc loading
- `processImage` / `processVideoFrame` / `processAudio`
- `multimodalSpliceT1` — for the per-token decode-loop fallback
- `computeVisionGroupIds` — for the bidirectional mask
- `fillBatchMasksVisionAware` — generalize from T=8 to T=288
- `prlZerosT1Buffer` — also useful for T=N (extend to a `prlZerosT`
  buffer)
- The new `generate()` overload that accepts image/audio params

Most of these can be cherry-picked into Stage 8 unmodified or with
minor shape generalization.

### Step 3 — Implement state-buffer bridging

**Reference**: probe 2 demonstrated the working pattern.

After prefill completes, the prefill MLStates hold valid K/V for
positions [0, T-1]. Bridge to the decode model's MLStates:

```swift
// For each prefill chunk's state, copy both kv_cache_sliding and
// kv_cache_full into the matching decode chunk's state.
func bridge(from src: MLState, to dst: MLState) {
    for stateName in ["kv_cache_sliding", "kv_cache_full"] {
        src.withMultiArray(for: stateName) { srcArr in
            dst.withMultiArray(for: stateName) { dstArr in
                let bytes = srcArr.count * MemoryLayout<UInt16>.stride
                memcpy(dstArr.dataPointer, srcArr.dataPointer, bytes)
            }
        }
    }
}
```

Notes:
- The closure-nested form is critical — the docs warn that the buffer
  pointer is only valid within the closure scope.
- chunk_3 in 3-chunk-merged is stateless (KV-shared, reads kv13/kv14
  from chunk_2's outputs); no state to bridge.
- chunk_1 + chunk_2_3way both have own-KV layers → bridge their states.

After bridging, set `current_pos` on the engine to T and continue the
decode loop from there. The existing `step(token:position:states:opts:)`
handles per-token decode unchanged once `state1`/`state2` reflect the
bridged contents.

### Step 4 — HF upload + ModelDownloader

Add prefill mlmodelc files to
`mlboydaisuke/gemma-4-E2B-stateful-coreml`:
- `chunk_1_prefill_T288.mlmodelc/`
- `chunk_2_3way_prefill_T288.mlmodelc/` (or `chunk_2_prefill_T288/`)
- `chunk_3_prefill_T288.mlmodelc/`
- Plus the encoder mlmodelc files Stage 6 was meant to upload
  (vision / vision_video / audio + sidecars, ~990 MB).

`buildGemma4StatefulLinearFileList` extension: add the prefill T=288
chunks to the file list. ALSO extend with the multimodal encoder
files Stage 6 prepared. Default download multimodal-on, with the
existing multimodal toggle from Stage 7.

Bundle size budget (estimate):
- decode chunks 1/2_3way/3 (Stage 3 ship): ~1.15 GB
- prefill chunks 1/2_3way/3 at T=288: ~1.50 GB (roughly chunk-size
  weights with INT4 g32; same as decode side)
- embed_tokens_per_layer (PLE): 2.35 GB
- embed_tokens: 0.40 GB
- encoders (vision + vision_video + audio + sidecars): 0.99 GB
- misc sidecars + tokenizer: 0.06 GB
- **Total: ~6.45 GB download, ~5.7 GB on disk after hardlink dedup**

Hardlink candidates (Stage 7 patterns):
- `chunk_1.mlmodelc/weights/weight.bin` ↔
  `chunk_1_prefill_T288.mlmodelc/weights/weight.bin` — likely
  bit-identical (same Linear projection weights, T-axis only differs
  in graph structure not weights). Confirm via md5 once both are
  built.
- `chunk_3.mlmodelc/weights/weight.bin` ↔
  `chunk_3_prefill_T288.mlmodelc/weights/weight.bin` — same.

If md5-identical, `linkItem` saves ~680 MB on disk; otherwise just
download both.

### Step 5 — iPhone verification matrix

**Critical gates** — failing any blocks ship:

| Test | Expected | Gate |
|---|---|---|
| Mac CLI smoke (text prompt) | decode 34+ tok/s, no parrot | sanity |
| iPhone 17 Pro: load all chunks | no `ANECCompile FAILED` | gate |
| iPhone 17 Pro: image prompt | "describe this image" → real description | quality |
| iPhone 17 Pro: multi-turn image follow-up | TTFT << 1 s on 2nd turn | wins gate |
| iPhone 17 Pro: text-only chat | no regression vs Stage 7 (33+ tok/s) | regression |
| iPhone 15 / 16 (8 GB RAM): load chunks | compile passes | cross-device |
| iPhone 15 (6 GB RAM): load chunks | may fail; if so, mark Pro-only | scope |

If non-Pro 8 GB devices reject T=288, two fallback levers:
1. Reduce T (T=224 covers 256-tok image with no leading-text margin;
   prepare to truncate text before image when prompt has both).
2. Mark stateful multimodal as Pro-only (`gemma4e2b3way` stays the
   non-Pro default).

### Step 6 — Picker default + back-compat

If everything passes:
- Promote `gemma4e2bStatefulLinear` (multimodal-enabled) to picker
  default.
- Demote `gemma4e2b3way` (Stage 7 recurrent ship) to second.
- Demote `gemma4e2b` (legacy 4-chunk multimodal recurrent) to third.
- Stateful research entry stays in `LLM_SHOW_EXPERIMENTAL=1`.

If only Pro passes:
- Stateful multimodal becomes `Gemma 4 E2B (Pro, stateful multimodal)`
  picker entry, not default.
- `gemma4e2b3way` (Stage 7) stays as the cross-device multimodal
  default.

---

## Risks + mitigations

1. **Cross-device compile budget on non-Pro RAM (8 GB).** Probe 1
   used 17 Pro (12 GB). Compile peak for T=288 chunk_2 (17 layers
   merged) is the unknown. Mitigation: probe per-chunk on iPhone 16
   non-Pro and iPhone 15 Pro before promoting to default. Fallback
   T=224 if 8 GB fails.

2. **Prefill→decode position alignment.** The prefill writes K/V at
   slots [0, T-1]; decode resumes at `current_pos = T`. If the decode
   chunks compute mask/RoPE using the wrong starting position, output
   is garbage. Mitigation: dedicated parity test — run a fixed text
   prompt through (a) legacy T=1024 prefill + decode and (b) stateful
   T=288 prefill + bridge + decode; assert top-1 token agreement on
   first 32 decode steps.

3. **Numerical parity vs legacy multimodal.** Vision attention is
   bidirectional within image group in both legacy 4-chunk prefill
   and our T=288 stateful prefill (we mask the same way). The graph
   structure differs (recurrent vs MLState slice_update); fp16 / INT4
   palettization rounding can drift. Mitigation: top-1 token
   agreement check on a fixed image prompt comparing legacy vs
   stateful output for the first 32 decode tokens.

4. **State buffer copy overhead.** Two memcpy's per chunk (sliding
   + full) = 4 memcpy's for 3-chunk decode (chunks 1 and 2_3way
   only — chunk_3 is stateless). Sliding buffer is ~14 MB, full is
   ~8 MB per chunk → ~22 MB × 2 chunks = ~44 MB total memcpy. On
   iPhone 17 Pro CPU this is sub-millisecond. Negligible vs the
   prefill compute itself (~270 ms per the multifunction T=8 bench).

5. **Stage 6 Swift code drift.** The
   `stage6-multimodal-stateful` branch was last touched 2026-04-27.
   Main has moved (Stage 7 PR #145 merged). When cherry-picking the
   multimodal helpers, expect minor merge conflicts in
   `Gemma4StatefulEngine.swift` and `LLMRunner.swift`. Mitigation:
   rebase Stage 6 work onto current main first, resolve conflicts,
   then start Stage 8 from there.

---

## Pre-merge checklist

- [ ] Step 1: prefill mlpackages built, ANE placement >=90%, Mac
      compile clean
- [ ] Step 2: `Gemma4StatefulEngine` loads prefill+decode dual,
      Stage 6 helpers cherry-picked + shape-generalized
- [ ] Step 3: state bridging implemented, parity test passes (top-1
      vs legacy on text prompt)
- [ ] Step 4: prefill mlmodelc + encoder files uploaded to HF;
      `ModelDownloader` extended; multimodal toggle still works
- [ ] Step 5: iPhone 17 Pro full bench (text + image + audio +
      video, multi-turn TTFT). Cross-device probe on 8 GB device.
- [ ] Step 6: picker default decision (universal vs Pro-only)
- [ ] PR opened with the same iPhone-baseline-check gate Stage 7
      followed (per `~/.claude/CLAUDE.md` memory:
      `feedback_code_pr_iphone_check`)

---

## Branches and artifacts

- `mlstate-multimodal-probe` (current PR / staging branch — has
  probe code + this doc + `MLSTATE_MULTIMODAL_PROBE.md`). Decide
  whether to merge before Stage 8 starts.
- `stage6-multimodal-stateful` (preserved, NOT merged to main —
  contains Swift multimodal helpers that Stage 8 will salvage).
- `main` post-Stage 7: PR #145 merged at 2026-04-27, tag `v1.7.0`.

## How to resume

1. Read `docs/MLSTATE_MULTIMODAL_PROBE.md` (the verdict + raw probe
   evidence).
2. Read this doc end to end.
3. Branch from current `main`: `git checkout -b stage8-mlstate-multimodal`.
4. Cherry-pick Stage 6 multimodal helpers from
   `stage6-multimodal-stateful` (use `git show <commit>` to inspect,
   `git cherry-pick` or manual port — they're contained in
   `Gemma4StatefulEngine.swift` and `LLMRunner.swift`).
5. Implement Steps 1-6 above in order.
6. Iterate iPhone gate, rebase as needed.
