# Session 2026-04-24 followup 2 — E4B prefill port shipped, next candidates

Continuation of `SESSION_2026_04_24.md` and `SESSION_2026_04_24_FOLLOWUP.md`.
Option **[F]** from FOLLOWUP (Gemma 4 E4B batched prefill port) landed as
commit `c6775a6`.

## Shipped 2026-04-24 (late)

| Commit | Change |
|---|---|
| `c6775a6` | `feat(prefill): config-driven prefill chunks for Gemma 4 E2B + E4B` |

### What changed

Python side:
- `conversion/models/gemma4_prefill_chunks.py` — `PrefillChunk{1..4}` take
  `(start, end)`; replaced hardcoded `layer_idx == 13/14` with
  `config.kv_{sliding,full}_producer`; added `chunk_kv_layout` /
  `chunk_output_names` helpers so build scripts enumerate per-chunk K/V
  outputs from config.
- `conversion/build_prefill_multifunction.py` — spec inputs/outputs
  derived from `compute_chunk_boundaries(config)`; kv13/kv14 input
  shapes use `config.num_key_value_heads` (fixes E4B nkv=2 trace error).
- `conversion/build_gemma4_bundle.py` — wires prefill subprocess
  (`--skip-prefill`, `--prefill-sizes`); writes `layer_types` and
  `num_kv_shared_layers` into `model_config.json`.

Swift side:
- `Sources/CoreMLLLM/ModelConfig.swift` — `layerTypes`,
  `numKvSharedLayers`, `isFullAttention`, `isKvShared`,
  `kvSlidingProducer`, `kvFullProducer`, `chunkBoundaries`. Fallback
  path (old bundles without `layer_types`) uses E2B's "every 5th is
  full, `num_kv_shared=20`" — matches HF truth exactly.
- `Sources/CoreMLLLM/ChunkedEngine.swift:buildKVSlotMap` — single
  config-driven replacement for the four hand-written
  `kvMapChunk{1,2}{Sliding,Full}` tables. Python+Swift mirror test
  confirms byte-exact parity with the old E2B hardcoded mapping.
- `writeSliding/FullFromPrefill` — use `nkv` from the cache's declared
  shape instead of assuming `nkv=1`. E2B behaviour unchanged; fixes
  latent shape bug exposed by E4B's `nkv=2` path.
- `prefillN` — switched from `let` to `private(set) var`; `attachPrefill`
  re-reads the real batch width from `hidden_states`. Resolves silent
  prefill fallback in the default `LLM_DEFER_PREFILL=1` path where `p1`
  was nil at init and the old fallback (512) did not match v1.2.0+
  ship models (1024). Fallback default bumped to 1024.

### Verification

- Mac regression: rebuilt E2B prefill chunks with refactored code; input/
  output spec and palettize output **file size byte-identical** to the
  v1.2.0 ship across all 4 chunks.
- iPhone 17 Pro, E4B bundle (7.7 GB) sideloaded and running:
  - KV shape `c1/c2 sliding=10x2 full=2x2` — correct for E4B `nkv=2`.
  - Decode steady **~14 tok/s @ 2K** (~45% of E2B's 31 tok/s;
    consistent with E4B having ~2× E2B's parameters).
  - Prefill 120 tokens in **1.43 s (84 tok/s effective, ~6× per-token
    decode)**. Prefill win is smaller than the doc-stated 10× because
    chunks 3/4 are all KV-shared — they don't benefit much from the
    N=1024 batch since their per-token decode is already cheap.
  - First post-prefill decode step was 993 ms (c4=681 ms) — cold ANE
    transition, absorbed inside `warmPrefillOnly` so not user-visible
    on the first real prompt.
- HF re-upload deferred — old E2B ship bundle (no `layer_types` in
  `model_config.json`) works via the Swift fallback; E4B remains USB-
  sideload only for now.

### Bug fixed as a side effect

v1.2.0 E2B shipped with a silent prefill failure in the default
`LLM_DEFER_PREFILL=1` path: p1 was nil at init, `prefillN` locked at
fallback 512, and the 1024-shaped ship model threw a mask-shape
mismatch on every real prefill — the caller silently fell back to
per-token decode. The E2B "prefill lengths 442/620/715 verified" line
in `SESSION_2026_04_24.md` was exercised either with
`LLM_DEFER_PREFILL=0` or by accident; the default code path silently
never hit the fast prefill until this commit.

## Open items carried from FOLLOWUP

### F1 UI freeze after turn 3 EOS (still open)
SwiftUI / `ChatView` state-machine issue on multi-turn wrap-up; Swift
inference stack completes cleanly. Entry point `ChatView.swift` stream
termination + `AsyncThrowingStream` continuation finish handling.
Unchanged, user-facing polish, orthogonal to decode speed.

### Bg-load long-prompt fallback (known tradeoff, not a bug)
During the ~160 s window while prefill chunks are loading in the
background, long prompts (>64 tokens) degrade to per-token decode
because `hasPrefill` returns false. Mitigation available (auto-`await
prefillLoadTask.value` in `generate()` when `tokens.count >= threshold`)
but not shipped — 10-line Swift change. Short prompts are unaffected.

## Execution order, next sessions

User confirmed priority order on 2026-04-24:

1. **Next session — UX micro-fixes (0.5–1 day).** Ship small quality
   improvements visible to the user.
   - Auto-`await prefillLoadTask.value` in `generate()` when
     `tokens.count >= 64` (empirical break-even given E4B's 12 ms/tok
     prefill vs 70 ms/tok decode plus ~1.2 s prefill fixed cost).
   - F1 UI freeze — diagnose ChatView stream termination path; the
     Swift inference stack is already clean per the 2026-04-24 logs.
   - Housekeeping: `docs/KNOWLEDGE_BASE.md`-level refresh if anything
     about the E4B port belongs in the mobile LLM knowledge base.

2. **Session +2 — [B] Decode chunk pipelining (~1 week).** Main track.
   - Current: chunks 1→2→3→4 serial, ~10 ms CPU prep/copy between each
     (E4B's prep/copy is longer, 20–30 ms).
   - Target: async-dispatch chunk N+1 input preparation during chunk N
     compute so ANE stays saturated.
   - Expected: **+10–20 % decode** (E2B 31 → 35–37, E4B 14 → 16–17
     tok/s @ 2K). Benefits every Gemma 4 variant and any future model
     with the same 4-chunk shape.
   - Sources: `docs/BASELINE_SPEED_AUDIT.md:59-64`,
     `docs/ANE_ONLY_LEVERS.md` item B.

3. **Session +3 onward — deferred.**
   - **[D] R7-1 COMPACT** (3–5 days, Mac-heavy, +2–3 tok/s @ 2K,
     model-specific post-training). Lower-priority than pipelining
     because it's per-model work whereas pipelining is model-agnostic
     runtime infra — "once done, every model benefits" beats "+2 tok/s
     on one model". See `docs/ROUND7_FINDINGS.md` §R7-1 for the full
     calibration-based vocab + FFN pruning recipe.
   - **[I] Metal Phase 3 port** — the real ANE-ceiling breaker
     (LiteRT hits 56 tok/s via Metal GPU). Weeks of work; reference
     `docs/METAL_PORT_REFERENCE.md` + llama.cpp's
     `src/models/gemma4-iswa.cpp`.

## Anti-list

Do not re-evaluate unless premise changes:

- Carried from `SESSION_2026_04_24_FOLLOWUP.md` (multi-length prefill
  variants, default-on PrefixCache, shelved
  `feat/litert-perf-adoptions`, K=V alias on E2B, PREFILL_N > W without
  SWA write fix, all drafter-based speculative paths, chunk
  consolidation 4→2, MLState).
- **HF re-upload of E2B** — old bundle + new Swift fallback works
  exactly. Defer until a breaking change forces a repush.
- **R7-1 before chunk pipelining** — user explicitly picked pipelining
  first on 2026-04-24; R7-1 is architectural per-model work and should
  follow the model-agnostic runtime win.

## Ceiling reminder

~40 tok/s @ 2K remains the ANE-only ceiling. Pipelining pushes toward
that; R7-1 helps further but both are asymptotic. Beyond the ceiling,
only Metal Phase 3 closes the gap to LiteRT's 56 tok/s.
