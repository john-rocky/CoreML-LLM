# Subset LM Head — iPhone 1.5× lever (in progress)

Date: 2026-05-13. Continues after 16-lever inventory closed with 1.16× ceiling.

## Concept

Target's chunk4 LM head is a 600M-param matmul (V=262144 × hidden=1536),
costing ~10-12 ms per cycle on iPhone ANE. For greedy MTP we only need the
ARGMAX. If we provide a candidate set of ~1024 tokens (drafter top-K ∪ PLD
n-grams ∪ frequent-token set), we can:

1. Skip the full LM head in chunk4 → output post-norm hidden state instead
2. In Swift, gather LM-head rows for candidates (~1024 × 1536 × INT4 = 600 KB)
3. Sparse matmul on CPU via Accelerate (~1 ms for K=3 batch)
4. Argmax over 1024 candidates → map back to actual token ID
5. Fallback: if confidence < threshold, run full chunk4 (lossless)

## Math (iPhone projection)

| | Current | + Subset LM head | + L5 async (parallel) |
|---|---|---|---|
| drafter cost | 11 ms | 11 ms | hidden ≤ verify |
| verify cost | 35 ms | ~28 ms (chunk4 -7ms) | ~28 ms |
| sparse matmul (Swift) | — | ~1 ms | ~1 ms |
| **cycle** | **47 ms** | **40-42 ms** | **~30 ms** |
| emit_avg (free-form, FLy=16) | 1.78 | 1.78 | 1.78 |
| **tok/s** | **37.9** | **44** | **59** |
| **speedup vs plain 32** | **1.18×** | **1.38×** | **1.85×** ✓ |

## Empirical Mac validation (2026-05-13)

- `chunk4.mlpackage` (full LM head, verify_qK K=3): **11.4 ms** median
- `chunk4_subset.mlpackage` (no LM head, normed_hidden output): **8.4 ms** median
- **Savings: 3 ms on Mac** (Mac has high mem bandwidth; iPhone delta expected ~5-7 ms)

## Progress (commits in this session)

1. **`SWAVerifyChunk4Subset`** class in `conversion/models/gemma4_swa_chunks.py`
   - Drops `lm_head`; outputs `(normed_hidden, hidden_states)` after final norm
   - Same layer pipeline as `SWAVerifyChunk4` for attention compute

2. **`conversion/build_chunk4_subset.py`** standalone build script
   - Required: `--ctx 2048` (default 512 collapses full-attention mask)
   - INT4 palettize default
   - Output: `chunk4_subset.mlpackage` ~311 MB (vs 320 MB w/ LM head)

3. **Built artifact**: `output/gemma4-e2b/chunks_subset/chunk4_subset.mlmodelc` (311 MB)
   - Mask shapes verified: `causal_mask_full (1,1,3,2048)`, `causal_mask_sliding (1,1,3,512)` ✓
   - Outputs `normed_hidden (1,3,1536)` + `hidden_states_out (1,3,1536)` ✓

## Remaining work (multi-day)

### Swift integration (8-12 hours)

1. **Load full LM head weights as separate buffer** (one-time at app start)
   - Extract from existing `chunk4.mlmodelc` or via Python export
   - INT4 palettized ~150 MB, or fp16 ~600 MB
   - Load via `MLMultiArray` or `Data` blob, gather via Accelerate

2. **Modify `ChunkedEngine.swift`** to optionally use `chunk4_subset`
   - Add `chunk4Subset: MLModel?` property
   - Add `verifyCandidatesSubset(tokens:, candidateIds:)` method
     - Run chunks 1-3 normally
     - Run chunk4_subset → get normed_hidden
     - Sparse matmul in Swift (vDSP_mmul or BNNS)
     - Argmax over candidates → token IDs

3. **`MtpSpeculativeEngine`** — build candidate set per cycle
   - Drafter top-K from `drafterTopKByStep` (already collected for FLy/sampling)
   - PLD history (`PromptLookupDraft.propose` with smaller maxDraftLen)
   - Recent emit history (last 30 unique tokens)
   - Top-N frequent English tokens (~900 tokens hardcoded)
   - Deduplicate, cap at 1024

4. **Fallback for low-confidence**
   - After subset argmax, compute confidence from logits
   - If `max_logit < threshold (e.g., 15.0)`: rerun verify with full chunk4
   - Threshold tuned empirically

### iPhone deploy + bench (1-2 hours)

- Push chunk4_subset.mlmodelc + LM head buffer to device
- Bench with hobby / Kalman / transformer prompts
- Compare emit_avg vs strict-only and FLy top-K=16
- Tune candidate set composition

## Risks

- **Sparse coverage gap**: if candidate set misses target's true argmax, emit wrong token.
  - Mitigation: fallback to full chunk4 on low confidence
  - Or: union with top-500 frequent tokens to guarantee high coverage
- **Swift sparse matmul performance**: vDSP/BNNS should give 1-2 ms; if slower, net loss.
  - Mitigation: profile, optimize via Accelerate
- **iPhone ANE actual savings**: Mac measured 3 ms; iPhone could be 5-7 ms or less.
  - Mitigation: A/B test empirically before claiming gain

## Files

- `conversion/models/gemma4_swa_chunks.py` — `SWAVerifyChunk4Subset` class
- `conversion/build_chunk4_subset.py` — build script
- `conversion/extract_lm_head.py` — extract fp16 LM head weights
- `conversion/extract_frequent_tokens.py` — corpus-derived top-N frequent tokens
- `output/gemma4-e2b/chunks_subset/chunk4_subset.mlmodelc` — built artifact (311 MB)
- `output/gemma4-e2b/lm_head_fp16.bin` — extracted LM head (768 MB)
- `output/gemma4-e2b/frequent_tokens.bin` — top-N freq IDs (Int32 LE)
- `docs/IPHONE_SPEEDUP_LEVER_INVENTORY_2026_05_13.md` — 16 levers empirical inventory
- `docs/SUBSET_LM_HEAD_PROGRESS_2026_05_13.md` — this file
- `docs/SUBSET_LM_HEAD_PHASE1_MAC_FINDINGS.md` — Phase 1 Mac empirical writeup

## Phase 1 final state (2026-05-13)

Swift integration is **complete and lossless on Mac** (verified bit-identical
output to baseline with `MTP_SUBSET_FLOOR=25`). Mac shows the path is **net
negative for tok/s** because:

- chunk4 LM head saving on Mac is only ~3 ms (chunk4=11 ms, chunk4_subset=8 ms)
- Candidate-set coverage with corpus-derived frequent tokens (4-8 K) + drafter
  top-K + recent history is ~50 % per slot. Floor catches misses via fallback
  → lossless but slow.

| Config | Mac tok/s | Status |
|---|---|---|
| Baseline (no subset) | 33.5 | reference |
| Subset M=1024 floor=25 | 22.2 | bit-identical (lossless), 35 % slower |
| Subset M=8192 floor=25 | 20.5 | lossless, slower (matmul grows linearly) |
| Subset M=1024 floor=0 (lossy) | 29.1 | drifts into garbage tokens |
| Subset M=8192 floor=-1000 (lossy) | ? | better but still drifts |

iPhone math projection from Mac data: same coverage issue applies → projected
break-even or marginal positive. The training-free 1.5× lossless target is
not deliverable with this candidate-set strategy. Full Phase 1 writeup +
options:`docs/SUBSET_LM_HEAD_PHASE1_MAC_FINDINGS.md`.

## How to test on iPhone (next-session recipe)

```bash
# 1. Push artifacts to device sandbox
xcrun devicectl device copy to --device <DEVICE-ID> \
  --domain-type appDataContainer --domain-identifier com.example.CoreMLLLMChat \
  --source output/gemma4-e2b/chunks_subset/chunk4_subset.mlmodelc \
  --destination Documents/Models/gemma4-e2b/chunk4_subset.mlmodelc
xcrun devicectl device copy to --device <DEVICE-ID> \
  --domain-type appDataContainer --domain-identifier com.example.CoreMLLLMChat \
  --source output/gemma4-e2b/lm_head_fp16.bin \
  --destination Documents/Models/gemma4-e2b/lm_head_fp16.bin
xcrun devicectl device copy to --device <DEVICE-ID> \
  --domain-type appDataContainer --domain-identifier com.example.CoreMLLLMChat \
  --source output/gemma4-e2b/frequent_tokens.bin \
  --destination Documents/Models/gemma4-e2b/frequent_tokens.bin

# 2. Build + install the app from Xcode (or via xcodebuild) AFTER adding
#    these env vars to the scheme (Edit Scheme → Run → Arguments → Env Vars):
#
#      MTP_SUBSET_LM_HEAD=1
#      MTP_SUBSET_M=1024
#      MTP_SUBSET_FLOOR=25
#      MTP_SUBSET_FREQ_BIN=<absolute path to frequent_tokens.bin in app sandbox>
#
#    Note: env vars in scheme apply when launching from Xcode, not from
#    devicectl device process launch. For a pure-devicectl test, hardcode
#    the flag in CoreMLLLM.load() or LLMRunner.swift.

# 3. Open the app, type a free-form prompt ("What is your favourite hobby
#    and why?"), record tok/s shown in the UI.

# 4. Compare against the current 1.16× ship (FLy top-K=16 + never-bail).
#    Target: ≥1.3× for L12 to be worth shipping.
```
