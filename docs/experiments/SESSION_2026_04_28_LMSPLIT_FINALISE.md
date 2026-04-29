# Stage 4 lm-split finalise — REJECTED (2026-04-28)

**Branch:** `stage4-lmsplit-finalise`
**Carries:** stash@{0} from 2026-04-25 (deploy + sanity scripts).
**Verdict:** **REJECTED — 16-way LM head split makes Gemma 4 E2B (vocab=262 144) decode slower, not faster.** ANEMLL's `qwen_model.py:1006-1124` claim does not transfer to vocab=262k.

## Hypothesis under test

ANEMLL splits the final `Conv2d(hidden, vocab, 1)` projection into 16 parallel heads (each `Conv2d(hidden, vocab/16, 1)`) for vocab=128k Qwen3, claiming the unsplit kernel is too large to tile efficiently in ANE SRAM. Our Gemma 4 E2B vocab is 262 144 (2× larger), so the priors said the technique should help **more**, not less.

Theoretical chunk3 weight-byte share of `lm_head`: ~54 % (302 / 562 MB). If ANEMLL's claim held, expected E2E decode tok/s lift was +3 to +5 %.

## Method (this session)

| step | result |
|------|--------|
| Mac build (chunk3 lm_split=16, ctx=2048, INT4/g32) | `output/gemma4-e2b/chunks_3way_lmsplit16/chunk3_3way.mlpackage` 502.8 MB |
| Smoke test (`smoke_lmsplit.py`) | bitwise-identical token_id vs SWAChunk4 ✓ |
| Sanity (`sanity_chunk3_lmsplit.py`) | Mac CPU_AND_NE pass ✓ |
| iPhone 17 Pro deploy | `Documents/Models/gemma4-e2b-lmsplit16/` (4.8 GB) |
| iPhone 17 Pro measurement | scheme `LLM_3CHUNK=1 + LLM_PROFILE_EVERY_STEP=1`, prompt `Write a short paragraph about sushi.`, ~150 tok |

Comparison: same iPhone session, picker-switched between production `gemma4-e2b` (lm_split=1, single Conv2d) and `gemma4-e2b-lmsplit16`. Both bundles share chunk1 + chunk2_3way at ctx=2048; only chunk3_3way differs (lm_head structure).

## Steady-state Profile (cycle 30+)

```
baseline (lm_split=1):
  emb=0.4ms mask=0.2ms | c1=5.6 c2=12.0 c3=0.0 c4=10.8 (sum=28.4ms) | predict=28.5ms total=28.9ms (34.6 tok/s)

lmsplit16:
  emb=0.4ms mask=0.2ms | c1=5.9 c2=12.0 c3=0.0 c4=11.8 (sum=29.7ms) | predict=29.8ms total=30.3ms (33.0 tok/s)
```

`c2` is the merged L8-24 chunk (17 layers, identical graph in both bundles) → confirms apples-to-apples comparison. `c4` is `chunk3_3way` (L25-34 + LM head, the only graph difference between variants).

## Numbers

| chunk | baseline | lmsplit16 | Δ |
|------|---|---|---|
| c1 (L0-7) | 5.6 ms | 5.9 ms | +0.3 ms (≈noise) |
| c2 (L8-24, 17 layers) | 12.0 ms | 12.0 ms | 0 ms (graph parity check) |
| **c4 (L25-34 + LM head)** | **10.8 ms** | **11.8 ms** | **+1.0 ms (+9.3 %)** |
| sum | 28.4 ms | 29.7 ms | +1.3 ms (+4.6 %) |
| **E2E tok/s** | **34.6** | **33.0** | **−4.6 %** |

Output text identity: **identical** for the sushi prompt (no token_id divergence; argmax + softcap + concat chain bitwise correct).

## Interpretation

The 16-way split is a real ANE pessimisation here, not a wash:

- Single `Conv2d(2304 → 262144)` already tiles efficiently inside the ANE compiler at vocab=262k. ANE handles internal tiling for large kernels; manual splitting just adds dispatch overhead.
- 16 separate ops + concat add fan-in/fan-out cost that exceeds whatever SRAM tile fit improvement the split offers.
- The chunk3 latency lift (+1.0 ms / +9.3 %) is consistent with concat overhead at this op count; on small vocab models it might be amortised, but on vocab=262k each split head is still 2304 → 16384 — that's a non-trivial individual op, and 16 of them paid back-to-back is more expensive than one bigger op.

This is the opposite signal to ANEMLL's. Their Qwen3 vocab=128 256 likely sits below the threshold where the unsplit kernel beats the split version on Apple's ANE compiler. We're 2× larger and the compiler clearly handles it.

## Decision per `docs/SESSION_2026_04_25_LMSPLIT_DEPLOY.md` table

| outcome | matched row |
|---------|-------------|
| `±2 % within → drop variant code, REJECTED_APPROACHES entry` | result is **−4.6 %** (worse than ±2 % band) → REJECTED is even more decisive than "null" |

8-way variant: not measured. Per the deploy doc this would be a "find the knee" data point. The 16-way result is decisively negative; 8-way might or might not also be negative, but the original hypothesis under test (ANEMLL's specific 16-way claim) is refuted regardless. Stopping here.

## Cleanup actions taken (this commit)

- `Sources/CoreMLLLM/ModelDownloader.swift` — remove 3 lm-split `ModelInfo` entries + their inserts in `defaults`.
- `conversion/build_gemma4_3way.py` — remove `--lm-splits {1,2,4,8,16}` flag.
- `conversion/models/gemma4_swa_chunks.py` — remove `SWAChunk4_LMSplit` class.
- `conversion/{smoke,sanity_chunk3}_lmsplit.py` — delete.
- `scripts/assemble_lmsplit_bundles.sh` — delete.
- `Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj/.../CoreMLLLMChat.xcscheme` — drop `LLM_3CHUNK=1` env (it's not Stage-4-specific, but we re-add it manually if revisiting).

Kept (closed but durable investigations from the same 2026-04-25 cycle):
- `docs/SESSION_2026_04_25_MASK_VALUE.md` + `conversion/probe_mask_value.py`
- `docs/SESSION_2026_04_25_RESIDUAL.md` + `conversion/probe_residual_overflow.py`
- `docs/SESSION_2026_04_25_ACTION_PLAN.md` (cross-references closed probes; lm-split row will be marked closed-REJECTED in this commit's roadmap §6 update)

## Roadmap impact

Stage 4 status row in `docs/ROADMAP_2026_04_26.md` §6 → "REJECTED 2026-04-28". v1.7.0 critical path unaffected (Stage 4 was always declared independent / low coupling).

## Memory entry

`project_lmsplit_rejected.md` — "ANEMLL 16-way LM head split rejected on Gemma 4 E2B vocab=262k (2026-04-28). −4.6 % decode tok/s, +9.3 % chunk3 latency. Single `Conv2d(2304 → 262144)` is already ANE-optimal at this scale. Don't revisit lm_head splits without a different vocab size or radically different ANE compiler behaviour."
