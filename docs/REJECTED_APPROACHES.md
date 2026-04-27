# Rejected approaches — index

One-stop reference for approaches we have already ruled out, so future
sessions (and agents) stop re-exploring them. This file is an **index**
only — point at the authoritative doc, commit, or memory entry. Do not
duplicate numbers here.

**Last refreshed:** 2026-04-24. When adding a rejection, land it in the
source doc first, then link from here.

Categories:

1. [Speculative / drafter](#speculative--drafter)
2. [Quantization](#quantization)
3. [Runtime / kernel](#runtime--kernel)
4. [Topology / chunking](#topology--chunking)
5. [Prefill / KV layout](#prefill--kv-layout)
6. [Diagnostic hypotheses refuted](#diagnostic-hypotheses-refuted)
7. [Round 7 candidates rejected](#round-7-candidates-rejected)
8. [Round 8 candidates rejected](#round-8-candidates-rejected)

Revisit rules: if you want to retry something below, you must (a) cite a
new upstream result or API change, (b) link the reproduction harness, and
(c) update this file with the new evidence.

---

## Speculative / drafter

| Approach | Date | Result | Source |
|---|---|---|---|
| MTP Path A (TFLite drafter extraction) | 2026-04-14 | Extraction failed; needs 3-5 day A100 training to continue | `MTP_INVESTIGATION_PATH_A_AUTOPSY.md`, memory `project_mtp_pivot.md` |
| MTP Path C (live integration) | 2026-04 | Accept rate gap; dead end on Gemma 4 E2B | `MTP_PATH_C_FINDINGS.md`, `MTP_INTEGRATION_RESULTS.md` |
| EAGLE-3 HASS (live) | 2026-04 | Phase 3 bench 11–17 tok/s with fallback to T=1; live speedup ≈0.96–1.11× (net even after oracle-live accept-rate gap). Demoted below ship bar. | `EAGLE3_INTEGRATION_STATE.md`, `ROUND7_FINDINGS.md` §Math verification |
| LayerSkip at L14 | 2026-04-08, 2026-04-16 | 0/60 match twice; L14 hidden useless without refinement | `HANDOFF.md:56` (`LAYERSKIP_PROBE=1`) |
| Path A post-11c drafter | 2026-04 | 0% Mac CPU across 29 rounds | memory `project_drafter_structurally_dead.md` |
| **HF Gemma 4 E2B base K-future signal** | **2026-04-27** | **H1 probe: linear K=2 = 11.7 %, K=3 = 6.9 %; MLP probe (non-linear capacity, ±2 pt). All drafter retrain variants (Section 9 init, larger drafter, ANE-collected labels) information-theoretically bounded. Base lacks MTP-aware imprint.** | `docs/SESSION_2026_04_27_H1_PROBE.md` |
| verify_qK fp16 write-before-accept (item 11c) | 2026-04 | Structural blocker: verify writes drafter proposals into KV at P+1..P+K-1 *before* acceptance is decided, corrupting all spec-decode paths. Confirmed by MTP Path C, EAGLE-3 Blocker 1, PHASE_B. | `MTP_PATH_C_FINDINGS.md:149`, `SESSION_STATE.md:179` |
| PLD-only standalone mode | — | Standalone PLD is not a speedup route; PLD *inside* the union orchestrator is still default-ON (the `pld-only` flag in `union-bitexact` exists as an isolation harness, not a product mode). | `Sources/union-bitexact/Verifier.swift`, `HANDOFF.md:34` |

Umbrella verdict: **all accessible drafter routes are closed on Gemma 4
E2B.** Only remaining drafter-adjacent path is ROUND7 post-training
(not started). PR #113 records the runtime fix so a future drafter can
plug in.

## Quantization

| Approach | Date | Result | Source |
|---|---|---|---|
| W8A8 activation quant | — | `ANECCompile()` rejects quantize/dequantize MIL ops | `HANDOFF.md:54` |
| W2 / W3 post-training palettization | — | Complete gibberish; requires QAT (multi-day GPU) | `HANDOFF.md:55` |
| W2A16 (as "untried") | — | Rejected despite `FUNDAMENTAL_UNTRIED.md` optimism | `HANDOFF.md:182` |

Surviving quantization lever: **W2-QAT**, still the only large headroom
item per `CPU_BOTTLENECK_INVESTIGATION.md`.

## Runtime / kernel

| Approach | Date | Result | Source |
|---|---|---|---|
| MLState stateful KV | iOS 26 confirmed | ANE compiler rejects `coreml_update_state`; GPU-only | `HANDOFF.md:53`, `HANDOFF.md:180` (`CONVERSION.md` still claims "Shipping" — ignore) |
| `LLM_DOUBLE_BUFFER_KV` | 2026-04-18 | IOSurface-backed MLMultiArray locks when used as input; cannot reuse | `HANDOFF.md:118` |
| ANE parallelism (PR #75) | — | 0 % gain — ANE driver serialises all submissions | `HANDOFF.md:50` |
| Native softmax swap (Gemma 4) | 2026-04-24 | Mac zero-delta; not worth iPhone trip | commit `4aebd33` |
| PR #17 micro-opts (MLP tile, GQA broadcast, exp2) | — | 5.5× slower — ops fell off ANE fast path | `HANDOFF.md:52` |
| `INTEGRATED_ROADMAP` conversion opts (softmax swap, FusedQKV, NCHW…) | — | No effect; coremltools 9 already applies internally | `HANDOFF.md:51` |
| `feat/litert-perf-adoptions` branch (S1/S2/T1-T5 wholesale) | — | Shelved; cherry-pick only with per-item verification | memory `project_litert_perf_branch_shelved.md` |

## Topology / chunking

| Approach | Date | Result | Source |
|---|---|---|---|
| D1b chunk pipelining (ANE+GPU split) | — | −24 % all categories; c3→c4 strict data dep | `HANDOFF.md:49` |
| Chunk consolidation 4→2 | — | +1 tok/s; dispatch-overhead theory refuted | `CHUNK_CONSOLIDATION_BENCH.md`, memory `project_chunk_consolidation_dead.md` |
| Topology I (c1 absorbs c2) | 2026-04 | iPhone A19 rejected | commits `d3703ee`, `72785c4` |
| Global-layer K=V alias (Gemma 4 E2B) | 2026-04-24 | K≠V at weight level — architecturally impossible for E2B | memory `project_gemma4_k_eq_v_false.md` |

## Prefill / KV layout

| Approach | Date | Result | Source |
|---|---|---|---|
| Multi-prefill-length variants as iPhone lever | 2026-04-23 | iPhone ANE is realLen-aware; padding is free. Prefix cache is the real TTFT lever. | memory `project_iphone_ane_sparsity.md` |
| SWA prefill write (pre-fix) | 2026-04-24 | Silent prefill bug on E2B; fixed in `a878c44` / `14a9965`. See `project_swa_prefill_write_bug.md`. | memory + commits |

## Diagnostic hypotheses refuted

| Hypothesis | Date | Outcome | Source |
|---|---|---|---|
| CPU orchestration saturates a P-core (thermal) | 2026-04-18 | Refuted on E4B device: `cpu_active=3.0ms (4% CPU)`, ANE is the heat source | `CPU_BOTTLENECK_INVESTIGATION.md`, `HANDOFF.md:105` |
| Dispatch density is the latency lever | 2026-04-18 | Refuted; ANE is busy 97% of step time | same |

## Round 7 candidates rejected

`ROUND7_FINDINGS.md` has the full receipts. In short: COMPACT, LaRoSA,
SCAP, R-Sparse survived a 2026-04-21 source-read; **xKV and the other
candidates were rejected** (fabricated benchmark numbers / wrong hardware
targets / nonexistent "CoreML-friendly" claims). See memory
`feedback_agent_hallucination_verify.md` for the verification process.

## Round 8 candidates rejected

`ROUND8_FINDINGS.md` (2026-04-26) has the full receipts.

| Approach | Why dead | Source |
|---|---|---|
| TurboQuant 3-bit KV (Walsh-Hadamard) | ANE forces FP16 decomp regardless of stored precision; Hadamard has no native ANE op | `SPEED_8K.md:34`, `UNEXPLORED_SOURCES.md:183`, `ROUND8_FINDINGS.md` |
| Spark Transformer (NeurIPS 2025, 2506.06644) | Production version of Gemma 3n's training-time activation sparsity, but Gemma 4 E2B does NOT inherit `activation_sparsity_pattern`. Static-mask offline variant collapses to ROUND7 R7-2/R7-4 territory | `ROUND7_FINDINGS.md:102-130`, `ROUND8_FINDINGS.md` |
| AFM-style cross-block KV sharing (5:3) | Gemma 4 E2B already shares 20/35 layers — more aggressive than AFM's 37.5 % | `CPU_BOTTLENECK_INVESTIGATION.md:150`, `ROUND8_FINDINGS.md` |
| Cross-Layer Attention (CLA), Tensor Product Attention (TPA) | Both require fine-tuning ~1B+ tokens; bundle with W2-QAT campaign rather than separate workstreams | `ROUND8_FINDINGS.md` |
| `reshapeFrequency = .infrequent` standalone hint | Tried, deleted. `LLM_PREFIX_CACHE=1` combo reproducibly triggers `MILCompilerForANE: failed to compile ANEF` on iPhone 17 Pro / iOS 26 | `Sources/CoreMLLLM/ChunkedEngine.swift:300-305` |
| **Joint compression: INT8 LUT entries** (Apple `linear_quantize_weights(joint_compression=True)`) | **Mac probe 2026-04-26**: cml9 accepts the API, but 73 INT8-dequant ops fall off ANE (92.9 → 86.7 %), Mac latency +3.4 %, cos sim 0.83 vs gate ≥0.95, bundle size unchanged (W4 indices dominate). Failure matches predicted top failure mode in ROUND8_FINDINGS §1. The `--joint-int8-lut` converter flag is kept in tree (default OFF) for future cml/iOS retry. | `docs/SESSION_2026_04_26_ROUND8_INT8_LUT_PROBE.md` |
| **Joint sparse + palettized** (N:M 2:4 prune + W4 LUT) | **Mac probe 2026-04-27**: cml9 `prune_weights(OpMagnitudePrunerConfig(n_m_ratio=(2,4)))` + `palettize_weights` ran clean. Bundle 155.8 → 349.6 MB (+124 % worse, sparse encoding non-efficient on cml9), step latency 5.12 → 5.69 ms (+11 %), ANE placement unchanged (92.9 % vs baseline same), cos sim hidden_states_out = 0.449 (quality destroyed by calibration-free magnitude prune). All four axes regressed. The `--prune-n-m` converter flag is kept in tree (default OFF) for cml10 retry. | `docs/SESSION_2026_04_27_ROUND8_JOINT_SPARSE_PROBE.md` |

---

## Still-alive shortlist (for contrast)

Kept here so readers don't leave with the impression *everything* is
dead. Authoritative plan lives in `HANDOFF.md`.

- GPU prefill via MLX-Swift (item 27) — current focus
- W2-QAT — only large remaining decode lever
- Chunk pipelining Phase 1 — shipped default-on 2026-04-24
  (commit `bf494b1`, memory `project_chunk_pipeline_phase1.md`)
- Round 7 ANE-only survivors (COMPACT / LaRoSA / SCAP / R-Sparse)
- MLState gate-zero micro re-verification, native SDPA re-test — cheap
  residual probes only
