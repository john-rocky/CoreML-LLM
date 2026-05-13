# L12 Subset LM Head — Phase 1 Mac Findings (2026-05-13)

Phase 1 Swift integration is complete and lossless on Mac, but does **not**
deliver a speedup on Mac under the current candidate-set strategy. iPhone push
is held per the "Mac confirms gain first" constraint. This doc summarises the
implementation, the empirical numbers, and the remaining gap.

## Implementation shipped on this session

Files touched on `feat/mtp-iphone-perf`:

- `Sources/CoreMLLLM/ChunkedEngine.swift`
  - `chunk4Subset: MLModel?` + `lmHeadFp16Data: Data?` properties
  - `verifyCandidatesSubset(tokens:, candidateIds:, startPosition:)`
    method returning `SubsetVerifyResult { argmax, argmaxLocalIdx,
    maxLogits, subsetLogits, candidateIds }`
  - `sparseMatmulFp32(...)` static helper — gather + vImage fp16→fp32 +
    `cblas_sgemm` (K=3, M, H=1536) → fp32 logits (K, M)
  - Load path: when `MTP_SUBSET_LM_HEAD=1`, looks for `chunk4_subset.mlmodelc`
    in the bundle directory and `lm_head_fp16.bin` (768 MB). Loaded fully into
    RAM (no mmap — see "Critical fix" below).

- `Sources/CoreMLLLM/MtpSpeculativeEngine.swift`
  - `subsetEnabled: Bool` flag computed at init from env + engine capability
  - `subsetCapM` and `subsetConfidenceFloor` knobs
  - `frequentTokensBase: [Int32]` loaded from `MTP_SUBSET_FREQ_BIN` path
  - `buildSubsetCandidates(drafterTopK:, verifyTokens:, cap:)` helper —
    Gemma special stops + drafter top-K + verify input + recent emit (30) +
    frequent base, deduped, capped at M
  - Verify branch in `speculateStep`: when `useSubsetThisCycle`, calls
    `verifyCandidatesSubset`; on `minLogit < floor` re-runs `verifyCandidates`
    full chunk4 for losslessness
  - FLy / strict accept logic adapted: when `subsetResult != nil`, ranks
    drafter token within subset via `isInTopKSubset(...)` helper instead of
    full-vocab top-K
  - `MTP_SUBSET_VALIDATE=1` diagnostic mode runs both subset and full each
    cycle, prints argmax mismatches, uses full argmax for actual decision

- `conversion/extract_frequent_tokens.py` (new) — tokenizes an English
  corpus (hardcoded chat snippets + optional NLTK Gutenberg + optional
  directory of .txt files), counts token frequencies, writes top-N IDs to
  `frequent_tokens.bin` (Int32 LE).

## Critical fix found mid-session

Initial implementation used `.alwaysMapped` for the 768 MB LM head buffer.
Per-cycle gather of 1024 rows × 1536 fp16 = 3 MB of random reads through the
mmap demand-paged on Mac/iPhone with **8-92 ms variability per call**. After
switching to `Data(contentsOf:)` (no mmap, resident in RAM), matmul cost
dropped to a stable **0.5 ms at M=1024**. iPhone 17 Pro has 12 GB so 768 MB
resident is acceptable; INT4 palettization to ~200 MB is a future optimization
if memory pressure becomes an issue.

## Empirical numbers (Mac, hobby prompt, 60 tokens)

| Configuration | tok/s | MTP accept | Notes |
|---|---|---|---|
| Baseline (no subset, MTP on) | **34.6** | 0.13 | strict-only verify cycle |
| Subset M=1024, floor=25, freq=4096 | 22.2 | 0.13 | **bit-identical output** to baseline |
| Subset M=8192, floor=25, freq=8192 | 20.5 | 0.13 | bit-identical output |

Verify-only timing (steady state) breakdown:

```
                                  chunks  chunk3  chunk4  matmul   total
Baseline (full chunk4 + LM head):  c12  +  c3   +  c4   +  -      = ~31 ms
Subset M=1024 (no fallback):        c12  +  c3   +  c4s  + 0.5 ms = ~28 ms
Subset M=1024 (with fallback):     above + full verify again    = ~59 ms
Subset M=8192 (no fallback):        c12  +  c3   +  c4s  + 8.1 ms = ~36 ms
```

Mac chunk4 LM head saving is only ~3 ms (matches Mac empirical doc
`SUBSET_LM_HEAD_PROGRESS_2026_05_13.md`). With every-cycle fallback the path
costs 28 ms (subset) + 31 ms (fallback full chunk4) = 59 ms per cycle.

## Why the speedup fails on Mac

Two factors stack against us:

1. **Mac chunk4 savings are small** (~3 ms / 10 % of verify). iPhone is
   supposed to be ~7-10 ms but cannot be measured without push.

2. **Candidate-set coverage is ~50 %** — `MTP_SUBSET_VALIDATE=1` diagnostic
   on a 60-token hobby generation:

```
Total cycles: 43
Cycles with at least one slot miss: 21 (49 %)
Per-position miss rate: 29/63 = 46 %
```

Even with 8192 corpus-derived frequent tokens + drafter top-K + recent emit
history + Gemma special stops, target's argmax falls outside the candidate
set ~50 % of the time. With a confidence floor strict enough to catch every
miss (floor=25), fallback rate hits 50 %.

Break-even fallback rate for net speedup:

| Platform | chunk4 savings | full chunk4 cost | Break-even p (subset must hit ≥ 1-p of the time) |
|---|---|---|---|
| Mac | 3 ms | 31 ms | 9 % |
| iPhone | 7-10 ms | 30 ms (projected) | 19-25 % |

Observed fallback rate ≈ 50 % is more than 2× over even the iPhone budget.

## Candidate-set sources observed in the dump

The MISS argmax tokens span the vocab (2455, 5192, 19979, 22195, 66577, 8126,
14464, 61874, 6841, 1144, 1657, 21964, 37051, 3577 …). Many are common
sentence pieces that simply weren't frequent enough in our 11.8 MB corpus to
crack the top 4096-8192. Doubling the corpus only marginally raises coverage;
the long tail of "rare but model-confident" tokens is fundamental.

## Path forward (recommendations)

Three options, ordered by effort × likely payoff:

1. **Lossy-mode trial (low effort)** — disable the floor (`MTP_SUBSET_FLOOR=0`),
   trust subset argmax even on low-confidence cycles. Equivalent quality
   trade to FLy top-K=16 which already ships at iOS default. Could land
   1.3-1.4× iPhone if accept rate doesn't collapse. **Requires user OK on
   the quality risk** — output will diverge token-by-token from the strict
   path.

2. **Semantic candidate selection (multi-day)** — replace the static
   frequent-token base with a per-cycle approximate nearest-neighbor lookup
   against the LM head, using `normed_hidden` as query. Precompute an HNSW
   index over the 262 K LM head rows; at runtime fetch top-1024 by inner-product
   similarity. Coverage should leap to >95 % at small (~1-2 ms) per-cycle cost.

3. **Path B drafter retraining** — already on the roadmap as the only
   structurally lossless 1.5× path. Self-distillation MTP (arxiv 2602.06019)
   should run in 1 GPU-week.

## Holds / not-yet-done

- **iPhone push (Task #9)**: held per `iPhone test は Mac で gain 確証後のみ`
  constraint. Mac did not confirm gain.
- **Path A (training-free) cannot deliver lossless 1.5× iPhone** with the
  candidate-set approach unless option 2 lands.
- Phase 1 implementation is clean and ready to be reused once a better
  candidate strategy is in place.

## Env vars added this session

| Var | Default | Notes |
|---|---|---|
| `MTP_SUBSET_LM_HEAD` | `0` | Enable subset path |
| `MTP_SUBSET_M` | `1024` | Candidate-set cap |
| `MTP_SUBSET_FLOOR` | `12.0` | Min subset logit for non-fallback. Use 25 for lossless on Mac |
| `MTP_SUBSET_FREQ_BIN` | unset | Path to `frequent_tokens.bin` (Int32 LE, top-N) |
| `MTP_SUBSET_DUMP` | `0` | Per-cycle subset argmax + logits dump |
| `MTP_SUBSET_TIMING` | `0` | Per-cycle subset path timing breakdown |
| `MTP_SUBSET_VALIDATE` | `0` | Run both subset + full each cycle, log mismatches, use full |
