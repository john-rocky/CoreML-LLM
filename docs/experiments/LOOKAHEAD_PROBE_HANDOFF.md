# LookAhead K=8 probe — next-session handoff

**Date prepared:** 2026-04-22
**Prepared by:** drafter-refutation session (user direction)
**Estimated elapsed:** 1-2 days (probe) → 1.5-2 weeks (full impl if probe passes)

## 0. TL;DR

All drafter-based speculative routes on Gemma 4 E2B are empirically dead (see `project_drafter_structurally_dead.md` memory). The only remaining drafter-free speculation route is **LookAhead / Jacobi decoding**. Success hinges on a single unknown: does our `verify_qK` scale flat or linearly with K on ANE? Measure this first with a cheap probe before committing to full implementation.

## 1. Context (read first)

- `project_drafter_structurally_dead.md` (memory) — why drafters are dead on E2B
- `docs/SPECULATIVE_DECODING_SURVEY.md` §1.10 — LookAhead / Jacobi mechanism
- `docs/ROUND7_FINDINGS.md:39-45` — measured iPhone timings (decode_q1 = 32.3ms, verify_qK=3 = 31.5ms, i.e. near-flat)
- `docs/OUR_STACK_ANATOMY.md` — current decode path and chunk layout
- `docs/KNOWLEDGE_BASE.md` — master index to everything else

## 2. The probe (go/no-go gate, before any real work)

### 2.1 Build K=8 verify chunks

```bash
cd conversion
PYENV_VERSION=lama-cml python build_verify_chunks.py --K 8 \
    --output output/verify-k8 \
    --context-length 2048 --sliding-window 512 \
    --palettize-int4
```

**Artifacts:** `chunk1-4.mlpackage` with `verify_qK` signature for K=8.

**Compile to mlmodelc:**
```bash
for c in chunk1 chunk2 chunk3 chunk4; do
  xcrun coremlcompiler compile output/verify-k8/${c}.mlpackage output/verify-k8/
done
```

Expected time: 30-60 minutes for the build, 5 minutes compile.

### 2.2 iPhone deployment (bundle for probe)

Create a probe bundle that mirrors an existing one but with K=8 verify chunks:
- Copy `device_deploy_mtp_v3/` or equivalent bundle to `device_deploy_lookahead_probe/`
- Replace `chunk1-4.mlmodelc` with K=8 variants
- Remove any MTP / EAGLE-3 drafter artifacts (we don't want them to interfere)

Sideload via `xcrun devicectl device copy to` to `Documents/Models/gemma4-e2b-lookahead/`.

### 2.3 Measure K=8 verify cost on device

Run CoreMLLLMChat or a dedicated probe harness (may need a minimal new Swift CLI `verify-k8-probe` that:
1. Loads the bundle
2. Prefills 17 tokens  
3. Calls `verifyCandidates(tokens: [...], startPosition: 17)` with 8 tokens
4. Prints the wall-clock time

Or simpler: run `coreml-llm-smoke` with a K=8-compatible MtpSpeculativeEngine modification that just calls verify once per cycle and logs the time. Can also gate via `LLM_VERIFY_K=8` env var if you extend `ModelConfig`.

Record **iPhone 17 Pro verify_qK=8 time** across 20+ cycles.

### 2.4 Decision

| Measured verify_qK=8 | Verdict | Action |
|---|---|---|
| **< 50 ms** | Go ahead with full LookAhead | Continue to §3 |
| **50-70 ms** | Marginal — only worth it if Metal Phase 3 gets held up | Document, defer |
| **> 70 ms** | K-linear scaling confirmed, LookAhead dead on ANE | Document, close route |

**Cheap probe total effort: 1-2 days (1 iPhone trip)**

## 3. Full LookAhead implementation (only if probe passes)

### 3.1 What to build

1. **Extended `verifyCandidates` API** to accept K up to 15 (or 8-12 as sweet spot). See `SpeculativeTarget` protocol.
2. **Jacobi loop in Swift** — new `LookaheadEngine.swift` ~300 LOC, paralleling `MtpSpeculativeEngine.swift` structure:
   - Maintain `W=7-15` guess tokens
   - Each cycle: batch forward via verify_qK, collect fixed points, update guesses
   - Reference: `llama.cpp/examples/lookahead/lookahead.cpp` (mechanism), `llama.cpp/common/speculative.cpp` (accept logic)
3. **N-gram cache** — ring buffer `[vocab][G][N-1]` per llama.cpp's pattern. G=15, N=5 typical. ~150 LOC. Can be Swift `Dictionary<UInt32, [[Int32]]>` with LRU eviction.
4. **Sampling integration** — since our stack is argmax-only, Jacobi's fixed-point detection is exact-match comparison (no temperature math).
5. **Wire into `CoreMLLLM.swift` decode loop** — new `lookaheadEnabled` flag + speculation path selection alongside MTP/EAGLE/DrafterUnion.

### 3.2 Parameters to start with

Per llama.cpp defaults + our constraints:
- W (lookahead window) = 7
- N (n-gram size) = 5
- G (max verify n-grams) = 15
- K (verify batch) = W + N - 1 = 11 — if probe passed at K=8, try K=11; if probe showed linear scaling, stay at K=8

### 3.3 Validation gates

| Gate | Where | Pass threshold |
|---|---|---|
| 1. Mac parity | `swift test` | Jacobi loop produces identical tokens vs serial decode |
| 2. Mac tok/s | `coreml-llm-smoke` | ≥ 32 tok/s (no regression) |
| 3. iPhone tok/s | device | ≥ +15% over 30 baseline = ≥ 35 tok/s |

### 3.4 Known risks

- **N-gram cache cold start** — first 20-30 tokens have no cache hits, acceptance = 0. Paper solution: cache persists across prompts.
- **Fixed-point detection on argmax-only stack** — Jacobi convergence is typically probabilistic; exact-argmax match is stricter. May converge slower.
- **ANE compile budget** — each distinct K value is a separate compiled graph. We're already using K=1 (decode) + K=3 (verify). Adding K=8 pushes us toward the ~100/process limit. Check after deploy.

## 4. Parallel work this unblocks

Metal Phase 3 (+14-19 tok/s ceiling) can proceed in parallel — different codebase regions. Expected to land before or around the same time as LookAhead full impl.

Potential stacking:
- Metal port (no drafter): +45-50 tok/s
- Metal port + LookAhead: +45-50 × 1.15 ≈ 52-58 tok/s = **LiteRT parity or above**
- Both required to beat 56 tok/s

## 5. Kill criteria

Abandon the route if any of:
- Probe §2.4 gives > 70 ms verify_qK=8
- Full impl produces ≥ 30 tok/s but regression in output quality (argmax divergence)
- ANE compile budget exhausted by adding K=8 bucket (would need to drop K=3)

Document the kill reason in a new `docs/LOOKAHEAD_RESULTS.md`; append to `project_drafter_structurally_dead.md` under "Still dead".

## 6. Code pointers

- `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` — closest analog; copy structure for LookaheadEngine
- `Sources/CoreMLLLM/ChunkedEngine.swift:1370-1533` — `verifyCandidates` / `verifyCandidatesTopN` entry points
- `Sources/CoreMLLLM/SpecProfile.swift` — logging pattern for per-cycle metrics
- `conversion/build_verify_chunks.py` — verify_qK builder
- `conversion/models/gemma4_swa_chunks.py` — chunk structure, `SWAVerifyChunk1-4` classes
- `llama.cpp/examples/lookahead/lookahead.cpp` — reference implementation (~300 LOC)
- `llama.cpp/common/speculative.cpp` — tree/accept infrastructure

## 7. Before opening PR

- [ ] iPhone probe numbers recorded in this doc (append §2.4 result)
- [ ] `swift test --filter ChunkedEngineKVParityTests` passes (with `KV_PARITY_MODEL_DIR`)
- [ ] No regression in `coreml-llm-smoke` baseline (default mode)
- [ ] iPhone default-mode smoke confirms baseline tok/s unchanged
- [ ] Update `project_drafter_structurally_dead.md` memory with LookAhead verdict

## 8. If you need more context

Ask the user to point you at the docs knowledge base (`docs/KNOWLEDGE_BASE.md`) and the drafter-dead memory. The current session already validated: drafter training paths are closed, Metal Phase 3 is parallel path, LookAhead probe is the cheapest remaining experiment.
