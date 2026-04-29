# Action plan: ANE acceleration hints for Gemma 4 — verification & status

**Date:** 2026-04-25
**Sources verified:** ml-ane-transformers, Anemll, maderix-ANE, executorch
**Companion docs:**
- `SESSION_2026_04_25_RESIDUAL.md` — fp16 residual probe results
- `SESSION_2026_04_25_MASK_VALUE.md` — mask -Inf vs -1e4 probe results
- `ANEMLL_SOURCE_NOTES.md` — earlier (2026-04-22) ANEMLL deep-read

## Summary table

| # | Hint | Source verified | Our impl status | Outcome |
|---|------|-----------------|-----------------|---------|
| A | mask = -1e4 (Apple recommendation) | `transformer.py:130` | runtime uses 0xFC00 (-Inf) | **CLOSED — no action**. Bitwise identical on Mac CoreML, no mask composition in our graph |
| B | SDPA op's attn_mask broken on ANE | `maderix README:189` | gemma4_swa_chunks.py:140,632 already uses `matmul→add(mask)→softmax→matmul` | **CLOSED — already correct** |
| C | fp16 residual scaling (α=0.5) | `gemma3_converter.py:2238-2322` | not present | **CLOSED — not needed**. E2B max=135, E4B max=161 (both <<30k threshold). anemll's α=0.5 hypothesis empirically refuted |
| D | per-head split einsum attention | `multihead_attention.py:80-116` | batched matmul | **DEFERRED**. Current 32 tok/s solid; per-head rewrite is high-cost, uncertain win |
| E | 16-way LM head split | `qwen_model.py:1006-1124` | single Conv2d(2304→262144) | **PREPARED — ready for iPhone A/B**. SWAChunk4_LMSplit class added; build flag `--lm-splits {1,2,4,8,16}`. Smoke test confirms math equivalence |
| F | ISWA local/global split state | `gemma3_converter.py:288-395` | TensorType I/O passthrough (not StateType) | **N/A** — different runtime architecture |
| G | executorch known-bug ops | `coreml_partitioner.py:78-117` | grep clean | **CLOSED — none used** |

## What's done (Mac-side, this session)

- `conversion/probe_residual_overflow.py` — residual stream max-abs probe (E2B + E4B)
- `conversion/probe_mask_value.py` — Mac CoreML mask-value A/B
- `conversion/smoke_lmsplit.py` — math-equivalence sanity for SWAChunk4_LMSplit
- `conversion/models/gemma4_swa_chunks.py` — added `SWAChunk4_LMSplit(n_splits)` class
- `conversion/build_gemma4_3way.py` — added `--lm-splits {1,2,4,8,16}` flag

All Mac-side probes ran on `python3.12` (coremltools 9.0 has no Python 3.14 binaries).

## What's left (gated on next iPhone session)

### Build (Mac, before iPhone trip)

```
python3.12 conversion/build_gemma4_3way.py \
    --model gemma4-e2b --only chunk3 --lm-splits 16
python3.12 conversion/build_gemma4_3way.py \
    --model gemma4-e2b --only chunk3 --lm-splits 8
```

Each writes `output/gemma4-e2b/chunks_3way_lmsplit{N}/chunk3_3way.mlpackage`.
Reuse the production `chunk1_3way` and `chunk2_3way` artifacts (lm_head
isn't there).

### Measure (iPhone, one session)

1. Profile chunk3 latency breakdown on baseline (lm_split=1) using
   Instruments → CoreML signposts. Record lm_head share of chunk3
   time. **If <15 %**, drop the split work — there's no measurable
   prize.
2. Otherwise, run 8-way and 16-way variants under the same harness.
   Compare:
   - chunk3 latency p50/p99
   - end-to-end decode tok/s
   - first-token argmax stability vs baseline (should match)
3. Log results to a follow-up SESSION doc.

### Decision rule

- If 16-way > baseline by ≥ 5 %, ship it (default behind a flag first).
- If 8-way ≥ 16-way and ≥ 3 % over baseline, ship 8-way (smaller
  artifact, simpler graph).
- If neither moves: `git rm` the variant code; document as
  REJECTED_APPROACH per repo convention.

## Items dropped from the plan

| Item | Reason |
|------|--------|
| FP16 residual scaling (α=0.5) | Empirical evidence (residual max ≤ 161 across E2B/E4B at production T) shows scaling is unnecessary |
| Swap mask fill -Inf → -1e4 | Bitwise-identical CoreML output; no composition in our graph |
| ISWA `ct.StateType` split | Runtime uses `ct.TensorType` I/O passthrough — different design entirely |
| Per-head einsum attention | High implementation cost, unclear ANE benefit on top of current 32 tok/s |
| SDPA attn_mask audit | Already correct — production uses additive mask |

## Open questions

1. Long-context (T≈2048) residual — extrapolation says safe, only worth
   re-measuring if decode quality regresses.
2. iPhone ANE softmax behavior with `-Inf` — Mac says identical to
   `-1e4`, but Mac CPU_AND_NE may not have dispatched to ANE for the
   tiny softmax probe. Production runtime tok/s evidence is stronger.
3. Whether 8-way or 16-way wins on iPhone ANE for vocab=262k
   lm_head — see iPhone-session checklist above.
