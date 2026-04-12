# Branch state & TODO snapshot

**Updated:** 2026-04-12.

Summary of what exists on each remote branch and what the next actionable work
is. Read alongside `POST_BENCH_PRIORITIES.md` (priority axis) and
`EAGLE3_INTEGRATION_STATE.md` (EAGLE-3 specifics).

---

## Branches

All three `feature/*` branches diverge from `main` by 53–55 commits and carry
a shared base of infrastructure (docs, MQA, EAGLE-3 Colab scaffolding, W8A8
prototypes, A–F scaffolds). Each then adds its own specialization.

### `main`
Minimal shipping baseline. Recent: lazy-load prefill chunks, weight sharing
between decode/prefill (-1.1 GB download), UIFileSharingEnabled.

**No specialized decode / speculative / audio work.** Stable.

### `feature/audio-support`  (tip: `079795f`)
Shared base + **audio multimodal pipeline** (Mel spectrogram → Conformer
CoreML model → Swift float32 projection → 50 audio tokens injected into
embeds). CLAUDE.md describes it as shipping-ready.

- Unique: `AudioProcessor.swift`, `audio.mlmodelc` build path,
  multimodal token insertion
- Includes: A–F scaffolds (duplicated from w8a8-8k), but **not EAGLE-3 Phase
  2A/2B/3 work**
- Status: presumed ready; not yet merged to main

### `feature/eagle3-speculative`  (tip: `c9b7bda`, **current branch**)
Shared base + **full EAGLE-3 Phase 2A/2B/3 pipeline** (verify chunks, Swift
SpeculativeTarget conformance, on-device iPhone 17 Pro bench). Does NOT
include audio.

- Unique: `conversion/models/gemma4_verify_chunks.py`,
  `conversion/build_eagle3_verify.py`,
  verify-chunk `.mlpackage` artifacts (under `output/eagle3-chunks/`, not
  tracked), Swift `loadVerifyChunks` + mask builders + SpeculativeTarget
  conformance in `ChunkedEngine.swift`, auto-load wiring in `CoreMLLLM.swift`,
  `docs/POST_BENCH_PRIORITIES.md`
- On-device status: pipeline works end-to-end; speculative **does not
  currently beat baseline** because of two independent blockers documented in
  `EAGLE3_INTEGRATION_STATE.md` §Phase 3 (distribution mismatch between
  training-time HF target and deployed custom target; commit-decode overhead)
- Includes: A–F scaffolds (duplicated)

### `feature/w8a8-8k`  (tip: `4a95d57`)
Shared base + **W8A8 (INT8 activations + INT8 weights) quantization pipeline
and uploaded artifacts**. Does NOT include audio or EAGLE-3 Phase 2A/2B/3.

- Unique: `build_w8a8_all.py`, `build_w8a8_proper.py`, `upload_w8a8.py`,
  W4A8 variant, ModelDownloader "Gemma 4 E2B W8A8 (8K, Experimental)" entry
- Bench: **Mac Studio M4 Max shows 0% speedup** (lacks INT8-INT8 ANE fast
  path). iPhone 17 Pro (A19 Pro) is expected to have the fast path per Apple
  docs → **not yet measured on-device**
- Includes: A–F scaffolds (duplicated)

### Shared A–F scaffolds across all three feature branches
All three branches independently cherry-picked the same A–F scaffolds. The
code exists but **none has been run end-to-end on device**:

| # | File | Approach |
|---|---|---|
| A | `conversion/build_prefill_gpu.py`, `Sources/CoreMLLLM/ComputePreferenceLoader.swift` | GPU prefill on A19 Pro tensor cores |
| B | `Sources/CoreMLLLM/MirrorSpeculativeLoop.swift`, `conversion/build_eagle3_gpu.py` | Mirror speculative decoding (draft→GPU) |
| C | `conversion/models/gemma4_swa_cascading.py`, `cascading_runtime.py`, `eval_cascading_quality.py` | Cascading KV cache for 8K quality |
| D | `conversion/apply_vocab_pruning.py` | Vocabulary pruning 262k → ~50k |
| E | `Sources/CoreMLLLM/PrefixKVCache.swift` | Persistent prefix KV cache |
| F | `conversion/optimize_mlpackage_graph.py` | MIL graph optimization pass |

---

## Merge / consolidation advice

The A–F scaffolds are duplicated across 3 branches — future conflict risk is
high. Recommended order:

1. **Merge `feature/audio-support` into `main`** first (least coupled to the
   experimental decoding tracks; audio is a product feature).
2. Rebase `feature/w8a8-8k` and `feature/eagle3-speculative` onto the new
   main so they share the audio base and pick up A–F scaffolds from one
   upstream place instead of three.
3. After W8A8 iPhone bench (task #24): merge `feature/w8a8-8k` to main if it
   wins, shelve if it doesn't.
4. `feature/eagle3-speculative` stays as an experimental branch until the
   Tier 2 work in `POST_BENCH_PRIORITIES.md` lands.

---

## TODO order under "performance-first" priority

Tasks are also tracked in the TaskList. Numbers here match task IDs.

### Immediate (this week)
- **#24 W8A8 iPhone 17 Pro bench** — highest expected win on decode (1.3–1.6×),
  models already uploaded, just needs device measurement
- **#25 F. MIL graph optim pass** — low-risk try-first, revert if it regresses
- **#26 audio-support → main merge review** — consolidate the base before
  more feature branches fan out

### Medium (weeks)
- **#27 P2 bandwidth-reduction triad** (pre-alloc masks, KV-share Q-batching,
  INT8 KV cache) — ~1.15× at 2K, larger at 8K, training-free
- **#28 EAGLE-3 Tier 2** — ONLY if decode gap to target remains after 24+25+27;
  indivisible set (6a retrain + 6b K/V direct-write + 6c Mirror v1)
- **#29 Pick 8K decode path** — TriForce vs DuoAttention, decide after baseline
  decode speed lands

### Long-term
- **#30 Cascading KV** — ship only if 8K ctx becomes default
- **#31 A. GPU prefill** — TTFT 13s → ~5s; UX win, not decode
- **#32 E. Prefix KV cache** — composes with A for cached prefixes
- **#33 D. Vocab pruning (LAST)** — requires product decision on multilingual
  coverage (keep ≥ 100k for 100+ language SKU); quality claims are
  English-heavy-benchmark scoped

### Ceiling model (stacked, from `POST_BENCH_PRIORITIES.md`)
```
baseline @ 2K (post-MQA, post-weight-share)      : 28 tok/s
+ F MIL optim                       × 1.03       →  29
+ W8A8                              × 1.4        →  41
+ P2 (INT8 KV + Q-batch + pre-alloc)× 1.15       →  47
+ EAGLE-3 full (6a+6b+6c)           × 1.55       →  73
+ Mirror v2 cross-burst             × 1.15       →  84
```

At 8K with TriForce/DuoAttention stacked: **~50–80 tok/s** per
`SPEED_8K.md` §2.

---

## Links
- [POST_BENCH_PRIORITIES.md](POST_BENCH_PRIORITIES.md) — performance-first priority axis
- [EAGLE3_INTEGRATION_STATE.md](EAGLE3_INTEGRATION_STATE.md) — EAGLE-3 Phase 3 bench report
- [SPEED_8K.md](SPEED_8K.md) — 8K roadmap (W8A8, DuoAttention, TriForce details)
- [UNEXPLORED_APPROACHES.md](UNEXPLORED_APPROACHES.md) — pre-bench A–F deep dive
- [ALTERNATIVE_APPROACHES.md](ALTERNATIVE_APPROACHES.md) — outside-Gemma-4 options
