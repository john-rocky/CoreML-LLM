# Speculative decoding — comprehensive survey

**Date:** 2026-04-22
**Scope:** All speculative / multi-token / lookup decoding variants we found, with source citation, mechanism, and our applicability.

## 0. TL;DR — what's usable for Gemma 4 E2B (source-verified)

Every method fights the same battle: **driver acceptance rate must hold up live, not just on training corpus**. Our measured oracle-live acc gap on Gemma 4 E2B is 3-9×. This invalidates any drafter trained on corpus-argmax disagreement.

| Status | Method | Reasoning |
|---|---|---|
| **DEAD for us** | EAGLE-3, Medusa, Hydra, separate-arch drafters | Oracle-live gap. Measured 0.96-1.11× live speedup. |
| **DEAD for us** | LiteRT-LM MTP | Recipe private; cannot reproduce. |
| **Maybe viable** | L-MTP (train base to predict K tokens) | By construction aligns with base argmax. Requires 1-2 A100 days of training. |
| **Narrow use** | PLD / suffix trie / n-gram cache | Drafter cost ≈ 0; only helps on prompt-repetition / quote-heavy workloads. Already in our DrafterUnion. |
| **Unproven on ANE ≤3B** | All published 2024-2026 work | No published ANE benchmarks for this model class. Qualcomm NPU exists (sd.npu) but is different architecture. |

## 1. Complete method list

### 1.1 Classical speculative (distribution-matching)

**Paper:** Chen et al. 2023 (DeepMind); Leviathan et al. 2023 (Google).

**Mechanism:** Draft model samples N tokens; target model evaluates all N+1 in parallel; accept via probability-ratio rejection.

**In llama.cpp:** `examples/speculative/speculative.cpp`. Dual-model, tree-capable via `p_draft_split` threshold.

**Our applicability:** Greedy-only decode makes rejection trivial (exact match), no probability math needed.

### 1.2 EAGLE / EAGLE-2 / EAGLE-3

**Papers:** Li et al. 2024/2025. [arxiv 2503.01840](https://arxiv.org/abs/2503.01840) (EAGLE-3).

**Mechanism:** Autoregressive drafter trained on hidden-states of target; draws draft branches informed by target's internal representations.

**In llama.cpp:** Stub only (`common/speculative.cpp:553` `common_speculative_state_eagle3`, PR-18039 pending).

**Our status:** Dead. Our own implementation shipped opt-in with `LLM_EAGLE3_ENABLE=1`, 22% accept, 234 MB ANE cost. Per `docs/ROUND7_FINDINGS.md`: `live_speedup = 0.96-1.11×`.

**Root cause:** Oracle-live acc gap. Drafter learns target's argmax on corpus, but live decoding drifts due to quantization + previous-token dependencies.

### 1.3 Medusa

**Paper:** Cai et al. 2024. [arxiv 2401.10891](https://arxiv.org/abs/2401.10891).

**Mechanism:** Multiple MLP heads predict different future tokens from base model's hidden state. Tree of candidates.

**In our stack:** `conversion/build_speculative.py` builds Medusa (3 heads, shared lm_head). Not in active runtime path per `docs/OUR_STACK_ANATOMY.md`.

**Status:** Same oracle-live gap concern as EAGLE-3. Not currently wired in decode loop.

### 1.4 Sequoia

**Paper:** Chen et al. 2024. [arxiv 2402.12374](https://arxiv.org/abs/2402.12374).

**Mechanism:** Hardware-aware tree topology for speculative drafts. Y-tree (2 children + 1 grandchild) beats linear chain at p=0.6.

**Reference in our docs:** `docs/ANE_OPTIMIZATION_SURVEY.md:17-21` (Top 5 findings) — T=4 Y-tree gives 3.20 vs 2.96 tokens/cycle.

**Status:** Tree topology optimization applies to any acceptance-rate; relevant if we revive Medusa. Cheaper (2 drafter calls vs 3).

### 1.5 SpecInfer

**Paper:** Miao et al. 2023. [arxiv 2305.09781](https://arxiv.org/abs/2305.09781).

**Mechanism:** Retrieval-based drafter + tree verification.

**Status:** Tree verification not ANE-friendly (dynamic dispatch). GPU-targeted.

### 1.6 Kangaroo

**Paper:** Liu et al. 2024. [arxiv 2404.18911](https://arxiv.org/abs/2404.18911).

**Mechanism:** Self-speculative with adapter layer. Base model drafts its own tokens via early-exit.

**Status:** Adapter training required. Similar oracle-live gap risk.

### 1.7 Hydra

**Paper:** Ankner et al. 2024. [arxiv 2402.05109](https://arxiv.org/abs/2402.05109).

**Mechanism:** Medusa variant with better heads (minimal training needed).

**Status:** Improvement on Medusa, same basic trade-offs.

### 1.8 Multi-Token Prediction (MTP)

**Paper:** Gloeckle et al. (Meta) 2024. L-MTP: Liu et al. 2025. [arxiv 2505.17505](https://arxiv.org/abs/2505.17505).

**Mechanism:** Train base model to predict K tokens at every position (not just next). Extra MLP heads for positions +1, +2, +3. **Multi-token prediction is built-in, not a separate drafter.**

**In LiteRT-LM:** Linear 4-step greedy exact-match via `llm_litert_mtp_drafter.cc:420-467`. Recipe private.

**L-MTP (NeurIPS 2025):** Curriculum learning (NTP → k=2 → k=4 → k=6). Code available on GitHub.

**Our status:** DEFERRED per `project_drafter_structurally_dead.md`. Gated on item 11c. Training cost: 1-2 A100 days.

**Why this might actually work:**
- MTP heads are trained to match base argmax BY CONSTRUCTION (they ARE the base model's forward pass).
- Oracle-live gap collapses because there's no drafter-vs-target distribution mismatch.
- LiteRT's 56 tok/s uses this; we just can't get the recipe.

**Action:** If we ever revive drafter effort, L-MTP is the ONLY credible path.

### 1.9 Prompt Lookup Decoding (PLD)

**Paper:** Saxena 2024. [arxiv 2401.15987](https://arxiv.org/abs/2401.15987).

**Mechanism:** N-gram match against prompt. If recent N tokens match a substring of the prompt, draft the following K tokens from prompt.

**In llama.cpp:** `examples/lookup/lookup.cpp`. Three cache tiers: context / dynamic / static.

**In our stack:** `PromptLookupLoop.swift` (9.2 KB). PLD n=2/3 is part of DrafterUnion.

**Status:** Low drafter cost (CPU substring match). Helps on prompt-heavy workloads (summarization, quote-answering). Hit: PL3 at 0.667 acc on one prompt; PL2 at 0.55 on summaries (`docs/ROUND7_FINDINGS.md:92-95`).

### 1.10 LookAhead / Jacobi decoding

**Paper:** Xia et al. (Stanford LMSYS) 2023.

**Mechanism:** W-wide lookahead window + N-gram verification. Ring buffer `[vocab][G][N-1]`.

**In llama.cpp:** `examples/lookahead/lookahead.cpp`. W=15, N=5, G=15.

**Status:** Complex. Deterministic but requires large working set. Not obviously better than PLD for our use case.

### 1.11 Suffix tree / SAM (Suffix Automaton)

**In our stack:** Part of DrafterUnion (via `DrafterUnion.swift:46-48`). Longest-repeated-suffix match against recent output.

**Status:** Active. Narrow applicability (repeated phrases). Tie-break AFTER pldN3 in DrafterUnion priority order.

### 1.12 Cross-vocab drafter (Qwen as drafter for Gemma)

**In our stack:** Qwen 0.5B drafts, token IDs remapped via `build_qwen_gemma_vocab_map.py`. **Disabled by default** — 1.8 tok/s on iPhone 17 Pro, 10× slower than projection drafter (`CoreMLLLM.swift:87-90`).

**Status:** Mac-side only. Cross-vocab mapping adds overhead; Qwen not argmax-aligned with Gemma.

### 1.13 Retrieval-based (Mirror SD, sd.npu)

**Paper:** Mirror SD (Apple, arxiv 2510.13161) — GPU+Qualcomm Hexagon for 14-66B. sd.npu (arxiv 2510.15312) — Qualcomm Hexagon only.

**Status:** Wrong hardware (Qualcomm NPU), wrong model size (14-66B). Not applicable.

### 1.14 DrafterUnion (our own composite)

**Source:** `Sources/CoreMLLLM/DrafterUnion.swift`.

**Mechanism:** Run multiple drafters in parallel, pick longest proposal. Priority tie-break: `crossVocab > pldN3 > suffix > pldN2`. Single verify call per burst.

**Status:** **Currently our active speculative path** (when aux assets loaded).

## 2. Our active drafter stack (what's really running)

Per source read:

```
if LLM_EAGLE3_ENABLE=1 and all EAGLE-3 assets loaded:
    use EAGLE-3 (22% accept, 234 MB)
elif DrafterUnion assets loaded:
    use DrafterUnion:
        proposals = [
            crossVocab (Qwen → Gemma remap) if enabled (default OFF),
            pldN3 (prompt lookup, 3-gram),
            pldN2 (prompt lookup, 2-gram),
            suffix (suffix-trie recent-output match)
        ]
        pick longest, tie-break crossVocab > pldN3 > suffix > pldN2
else:
    serial T=0 decode (no drafting)
```

**Best live path for most workloads:** DrafterUnion (pldN3 + suffix handle repetition; crossVocab off).

## 3. Break-even math (copied from ROUND7_FINDINGS)

From `docs/ROUND7_FINDINGS.md:44-76`:

```
Base decode @ 2K:     32.3 ms
Drafter forward (ANE): 5 ms (EAGLE-3)
Verify K=3 per call:  31.5 ms

cycle_2K = 5 + 31.5 = 36.5 ms
break_even = 36.5 / 32.3 = 1.13

paper_speedup (τ=2.13) = 2.13 / 1.13 = 1.89× → 31 → 58 tok/s  [on paper]
live_speedup (live acc 0.08-0.25) = 1.08-1.25 / 1.13 = 0.96-1.11× [net even]
```

**Key insight:** Even MTP (if we could train it) would need to hit > 1.13 break-even. LiteRT's 4-step MTP implies τ > 1.4 live, which is a high bar; they achieve it because MTP heads are trained with the base forward pass, not against a separate drafter.

## 4. Action items

### 4.1 No-op

- EAGLE-3: leave opt-in; do not prioritize revival until L-MTP is attempted.
- Medusa: build remains in conversion pipeline but no runtime wire-up.
- Sequoia Y-tree: keep in mind IF Medusa ever re-activated.

### 4.2 Maintain

- DrafterUnion: active path, works for repetition-heavy prompts.
- PLD n=2/3 + suffix: cheap, active.

### 4.3 Future (gated on 11c per memory)

- L-MTP training on Gemma 4 E2B: 1-2 A100 days. Only credible path to +15-20% tok/s via speculative.

### 4.4 Deprioritize permanently

- Cross-vocab Qwen drafter as primary (too slow on iPhone; Mac-only research use).
- EAGLE-3 re-training (oracle-live gap won't close with more data).
- Sequoia / SpecInfer / Mirror SD / sd.npu (wrong hardware).

## 5. Why we keep the dead weight in the codebase

EAGLE-3, Medusa build scripts, build_speculative.py, build_flash.py, build_wfa.py, cascading_runtime.py — all remain in conversion/. Reasons:
1. **Re-activation without re-writing** — if L-MTP succeeds, Medusa heads + EAGLE-3 architecture are both reusable.
2. **Comparative benchmarking** — accept-rate-bench uses these to measure live vs oracle gap.
3. **Document-by-code** — the build scripts ARE the architectural notes on each method.

No reason to delete; do not delete.
