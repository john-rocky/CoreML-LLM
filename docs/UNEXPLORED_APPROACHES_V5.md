# Unexplored Approaches — Round 5 (MTP & new speculation families)

Nine techniques surfaced via external review (2026-04-13) that are **not
documented anywhere** in `SPEED_8K.md`, `ALTERNATIVE_APPROACHES.md`,
`UNEXPLORED_APPROACHES.md` (V1–V3), `FUNDAMENTAL_UNTRIED.md`, or
`EAGLE3_DEPLOY.md`.

Filtered against all existing docs. Only genuinely new-to-this-repo items
below.

**Key finding:** the most promising direction — **Multi-Token Prediction
(MTP)** — is absent from our entire roadmap. It may be how Google's
AIEdgeGallery achieves its speed on Gemma 4, and it sidesteps the need for
a separately trained draft model entirely.

---

## Priority tier

| Priority | Technique | Why |
|----------|-----------|-----|
| **S-tier** | §1 Native MTP / post-hoc MTP head | No draft model, ANE-friendly, likely what Google ships |
| **A-tier** | §2 Speculative Streaming | Medusa re-examination with ~10000× fewer params |
| **A-tier** | §6 Cross-vocabulary SD | Escape hatch from EAGLE training |
| **B-tier** | §3 Draft & Verify, §4 Kangaroo, §5 SWIFT | Self-speculative variants |
| **B-tier** | §7 DISCO | Free +10% on any speculative method |
| **C-tier** | §8 SAM-Decoding, §9 Lookahead Decoding | Retrieval complements to SuffixDecoding |

---

## 1. Native MTP / Post-hoc MTP Head — **highest priority, not yet explored**

### What it is
Multi-Token Prediction (MTP) trains or attaches additional prediction heads
to the target model so it produces **multiple future tokens per forward
pass** instead of one. Unlike speculative decoding, there is no separate
draft model — the target model itself predicts ahead.

Three flavors:

1. **Native MTP**: the model was pre-trained with MTP objectives (DeepSeek-V3,
   Meta MTP paper). The heads already exist in the weights.
2. **Post-hoc MTP head**: attach a small gated LoRA or linear head to an
   existing AR model and fine-tune it to predict N+1, N+2, ... tokens.
   Apple's 2025 paper uses this approach.
3. **LiteRT-embedded MTP**: Google's LiteRT export of Gemma 4 may include
   MTP heads used for on-device speculative/parallel decoding. The HF
   discussion confirms "the exported model contains MTP prediction heads
   for on-device use."

### Why this is the top candidate for our pipeline

- **No separate draft model.** EAGLE-3 requires training and shipping a
  ~50 MB draft. MTP adds ~5–15 MB of heads to the existing model. On ANE
  where memory pressure is the constraint, this is a categorical advantage.
- **Single forward pass.** No draft→verify serial round-trip. The MTP heads
  run as extra outputs in the same ANE dispatch as the main forward pass.
  This directly attacks the dispatch-count bottleneck identified in
  `FUNDAMENTAL_UNTRIED.md` §0.
- **Reported numbers.** Apple 2025: gated LoRA MTP on their 3B model yields
  **~2.5× on general chat, ~5× on code/math**. vLLM treats MTP as "native
  speculative decoding" and reports comparable gains to EAGLE-class methods.
- **Gemma 4 may already have MTP support.** Google's official HF discussion
  states that the LiteRT exported model includes MTP prediction heads.
  AIEdgeGallery's speed on Gemma 4 may come from this, not from a separate
  draft model. If the MTP head weights can be extracted from the LiteRT
  graph, we skip the training step entirely.

### ANE compatibility
- MTP heads are small linear layers (H → V or H → H → V). Fully ANE-
  compilable as extra outputs on existing chunks.
- Static shapes: each head predicts one position. K heads = K extra scalar
  outputs. No dynamic shapes.
- Multi-function mlpackage: one function for AR-only decode (current), one
  for MTP-enabled decode (adds the heads). Shared weights via
  `MultiFunctionDescriptor`.

### Implementation paths (ordered by cost)

**Path A — Extract from LiteRT (lowest cost, highest uncertainty)**
1. Download Gemma 4 E2B LiteRT export from Google.
2. Inspect the TFLite graph for MTP head subgraphs.
3. If found: extract weights, convert to CoreML, attach to our chunks.
4. Cost: ~2 days if heads exist. 0 training.

**Path B — Post-hoc MTP head training (moderate cost, proven)**
1. Freeze the Gemma 4 backbone.
2. Attach K=2–3 prediction heads (gated linear, ~5M params each).
3. Train on the existing EAGLE-3 corpus (30k examples, A100, ~4–8 h).
4. Convert heads to CoreML, add as extra chunk4 outputs.
5. Cost: ~3–4 days including conversion.

**Path C — Apple-style gated LoRA MTP (higher cost, highest quality)**
1. Per Apple 2025 paper: insert gated LoRA adapters into the backbone +
   MTP heads. Fine-tune jointly.
2. Preserves original AR behavior (gate starts at 0).
3. Cost: ~5–7 days, A100 time required.

### Composability
- ✅ With SuffixDecoding: MTP as learned draft, suffix tree as retrieval
  draft. Union candidates → shared Q=K verifier.
- ✅ With MLState: MTP heads are stateless; KV management is orthogonal.
- ✅ With vocab pruning, prefix KV cache — fully orthogonal.
- ⚠️ Partially overlaps with EAGLE-3: both are learned speculation. Choose
  one as primary, or use MTP for short-range (K=2) + EAGLE for longer (K=5).

### Relationship to existing roadmap
EAGLE-3 (`EAGLE3_DEPLOY.md`) is an external draft model. MTP is an internal
multi-head. They are alternatives for the same goal (multi-token speculation),
but MTP is architecturally simpler and uses less memory. If MTP achieves
comparable acceptance rates, it supersedes EAGLE-3 for our ANE-constrained
deployment.

### Sources
- [Apple Foundation Models 2025 — MTP section (arXiv 2507.13575)](https://arxiv.org/abs/2507.13575)
- [DeepSeek-V3 Technical Report — native MTP (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437)
- [Meta — Better & Faster LLMs via MTP (arXiv 2404.19737)](https://arxiv.org/abs/2404.19737)
- [vLLM — MTP as speculative decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html)
- [Google AI Edge Gallery — Gemma 4 on-device (github.com/google-ai-edge)](https://github.com/google-ai-edge/gallery)

---

## 2. Speculative Streaming — Medusa re-examined with minimal parameters

### What it is
Speculative Streaming predicts future n-grams without an auxiliary model,
using **~10000× fewer additional parameters** than Medusa. Instead of
training K independent heads over the full vocabulary, it trains a small
stream head that predicts likely n-gram continuations from the hidden state.

### Why reconsider when Medusa was rejected
Medusa-1 was rejected (`SPEED_8K.md` §0) at 1.3% acceptance on Gemma 4.
Speculative Streaming differs in three ways:
1. **N-gram prediction, not independent-position**: captures token
   dependencies (bigram/trigram) that Medusa-1 ignores.
2. **Minimal parameters**: stream head is ~1K–10K params vs Medusa's
   ~100M params per head. ANE memory impact is negligible.
3. **Resource-constrained devices**: the paper explicitly targets devices
   where a full Medusa head set is infeasible.

### Numbers
- 1.8–3.1× speedup reported across model sizes.
- The low parameter count means training is fast (~1h on a single GPU).

### ANE compatibility
- Stream head is a tiny linear layer. Static shapes. Pure ANE.
- Verification uses the same Q=K verifier as other speculative methods.

### Cost
- ~2–3 days: train stream head on EAGLE-3 corpus, convert, integrate.
- Much cheaper than EAGLE-3 training.

### Composability
- ✅ With SuffixDecoding, EAGLE-3, MTP — candidate sources are additive.
- ✅ With Q=K verifier (G1) — shares the same verification infrastructure.

### Honest gap
- No published results on Gemma-class models. Medusa's 1.3% failure may
  share root causes (Gemma 4's architecture specifics). Needs measurement.

### Sources
- [Speculative Streaming (arXiv 2402.11131)](https://arxiv.org/abs/2402.11131)

---

## 3. Draft & Verify — training-free self-speculative decoding

### What it is
Uses the same model's intermediate layers as a draft. No additional training,
no additional memory. The model's own early-layer representations are used to
predict future tokens, then the full model verifies.

### Difference from LayerSkip (already in FUNDAMENTAL_UNTRIED §4)
- LayerSkip requires pre-training with early-exit loss. Draft & Verify is
  **fully training-free** — it adds a skip connection from an intermediate
  layer to the LM head at inference time.
- LayerSkip needs a learned exit head. Draft & Verify re-uses the existing
  LM head directly.

### Numbers
- Up to 1.99× without any additional training or memory.
- Original distribution preserved (lossless).

### ANE compatibility
- ✅ Static shapes. Uses existing chunk boundaries.
- For Gemma 4: exit at L14 (end of chunk2), project to LM head (in chunk4).
  Needs a lightweight bridge — or build a "draft-only" mlpackage that runs
  chunks 1-2 + a copy of the LM head projection.

### Cost
- ~2 days. No training. Needs a small LM-head-only mlpackage for draft
  scoring from L14 hidden states.

### Sources
- [Draft & Verify (arXiv 2309.08168)](https://arxiv.org/abs/2309.08168)

---

## 4. Kangaroo — lightweight adapter self-speculative

### What it is
Adds a small adapter network (a single-layer MLP) at an intermediate layer
to produce draft tokens. Much lighter than a full draft model, much more
accurate than raw early exit.

### Numbers
- Up to 1.68× speedup.
- Adapter is ~5M params — negligible memory footprint.

### ANE compatibility
- ✅ Small adapter is a single linear layer. Static, ANE-native.
- Fits naturally at the L14 boundary (chunk2 exit) in our pipeline.

### Cost
- ~2 days: train adapter on EAGLE-3 corpus (fast, <1h GPU), convert, wire.

### Composability
- ✅ With SuffixDecoding, MTP — orthogonal draft sources.
- Strictly between Draft & Verify (training-free, lower acc) and
  EAGLE-3 (full draft, higher acc) in cost/quality.

### Sources
- [Kangaroo (arXiv 2404.18911)](https://arxiv.org/abs/2404.18911)

---

## 5. SWIFT — training-free self-speculative with distribution preservation

### What it is
Self-speculative decoding that skips FFN layers in the draft pass while
preserving the original output distribution. No training required.

### Key distinction
- Guarantees **exact same output distribution** as the target model (lossless
  in the strict sense).
- 1.3–1.6× speedup by skipping 30–50% of FFN computations in draft mode.
- Zero additional parameters or memory.

### ANE compatibility
- ✅ Two mlpackage functions: one full (verify), one FFN-skipped (draft).
  Shared weights via multi-function. Static shapes.

### Cost
- ~2 days. Build FFN-skipped variant of each chunk. No training.

### Honest gap
- 1.3–1.6× is modest. May not justify the engineering over methods with
  higher ceilings (MTP, SuffixDecoding).

### Sources
- [SWIFT (arXiv 2410.06916)](https://arxiv.org/abs/2410.06916)

---

## 6. Cross-Vocabulary Speculative Decoding — use any small model as drafter

### What it is
Enables lossless speculative decoding when the draft model and target model
use **different tokenizers and vocabularies**. Previously, drafter and target
needed the same tokenizer. This constraint is removed via token-level
probability alignment at the vocabulary boundary.

### Why this matters for us
- EAGLE-3 requires training a custom draft model specifically for Gemma 4.
  Cross-vocab SD lets us use **any existing small model** as a drafter:
  Gemma 3 270M, Qwen 2.5 0.5B, SmolLM2 135M, etc.
- These models already exist, are already optimized, and some are already
  converted to CoreML in this repo (Qwen 2.5 0.5B).
- **Zero training cost.** Download a small model, run cross-vocab SD.

### Numbers
- Reported as lossless (exact target distribution preserved).
- Speedup depends on draft model acceptance — typically 1.5–2.5× with
  a well-matched small drafter.
- Integrated into HuggingFace Transformers (production-grade).

### ANE compatibility
- ✅ Two models co-resident on ANE. Our pipeline already supports multiple
  models (vision + audio + LLM chunks).
- Static shapes for both draft and target.
- Cross-vocab alignment logic runs on CPU (token remapping).

### Implementation sketch
1. Load Qwen 2.5 0.5B (already in `ModelDownloader.swift`) as drafter.
2. Draft K=3–5 tokens with Qwen.
3. Remap draft tokens from Qwen vocab → Gemma vocab via the alignment
   algorithm.
4. Verify with Gemma 4 chunks using Q=K verifier.
5. Accept/reject per standard speculative decoding protocol.

### Cost
- ~3–4 days. The alignment algorithm needs careful implementation, but
  no model training is required.
- Drafter model is already available (Qwen 2.5 0.5B = 309 MB, downloaded
  via existing infrastructure).

### Composability
- ✅ With SuffixDecoding: cross-vocab SD as learned draft, suffix tree as
  retrieval draft. Union candidates.
- ✅ With MTP: use cross-vocab for long-range speculation, MTP for short.
- ⚠️ Partially conflicts with EAGLE-3: both are "use a separate model as
  draft." Choose one. Cross-vocab SD has zero training cost; EAGLE-3 has
  higher acceptance (purpose-trained).

### Sources
- [Cross-Vocabulary Lossless Speculative Decoding (arXiv 2502.11926)](https://arxiv.org/abs/2502.11926)
- [HuggingFace Transformers — assisted generation](https://huggingface.co/blog/assisted-generation)

---

## 7. DISCO — Dynamic Speculation Control

### What it is
Instead of using a fixed speculation lookahead (K=3 always), DISCO
dynamically adjusts K per step based on the draft model's confidence and
the observed acceptance rate. When the draft is confident, speculate deeper
(K=5–8). When uncertain, fall back to K=1 or K=2.

### Why it matters
- **Free +10% on top of any speculative method.** Exact same output text,
  just smarter scheduling of when to speculate aggressively vs conservatively.
- Pure algorithmic improvement — no model changes, no training, no
  reconversion.

### ANE compatibility
- ✅ Does not touch the forward pass. CPU-side scheduling logic only.
- Works with EnumeratedShapes: pre-compile verify functions for K ∈ {1,2,3,5}
  and select per step.

### Cost
- ~1 day. Pure Swift logic in the speculative loop.
- Requires a speculative method to already be in place (EAGLE-3, MTP,
  SuffixDecoding, or cross-vocab SD).

### Composability
- ✅ Layered on top of any speculative method. Strictly additive.

### Sources
- [DISCO (arXiv 2405.13019)](https://arxiv.org/abs/2405.13019)

---

## 8. SAM-Decoding — Suffix Automaton for fast retrieval speculation

### What it is
Builds a suffix automaton (more space-efficient than a suffix tree) over
the generated history and prompt. Matches the current context against
the automaton to retrieve likely continuations.

### Relationship to SuffixDecoding (FUNDAMENTAL_UNTRIED §1)
- SuffixDecoding uses a suffix tree. SAM-Decoding uses a suffix automaton.
- Suffix automaton is **O(n) space** vs suffix tree's O(n²) worst case.
- SAM-Decoding reports faster lookup times on long histories.
- The two are complementary data structures for the same retrieval approach.

### Numbers
- Reported as improving over standard suffix-tree retrieval on long outputs.
- Exact speedup varies by workload (same as SuffixDecoding).

### ANE compatibility
- ✅ Pure CPU data structure. ANE pipeline unchanged.

### Cost
- ~2 days. Alternative data structure for the SuffixDecoding implementation
  planned in `FUNDAMENTAL_UNTRIED.md` §1. Choose one at implementation time.

### Sources
- [SAM-Decoding (arXiv 2411.10666)](https://arxiv.org/abs/2411.10666)

---

## 9. Lookahead Decoding — trie-based parallel generation

### What it is
Training-free, lossless method. Maintains a trie of n-grams generated
during the current session. At each step, looks up the current context in
the trie and generates candidates in parallel via Jacobi iteration.

### Difference from SSSD (UNEXPLORED_APPROACHES_V3 §A2)
- SSSD builds a trie cache on-the-fly from generated tokens (similar data
  structure). Lookahead Decoding additionally uses **Jacobi iteration** to
  generate candidates in parallel, not just lookup.
- Lookahead reports 2.66–6.26× (higher ceiling than SSSD's 1.5–2.9×).
- The Jacobi iteration component requires parallel forward passes, which
  maps to Q=K verify on ANE.

### ANE compatibility
- ⚠️ Jacobi iteration needs multiple parallel forward passes per step.
  Implementable via Q=K multi-token dispatch (G1 from V2 doc), but the
  iteration count must be fixed (static shapes).
- Trie lookup is CPU-side, ANE-transparent.

### Cost
- ~3 days. Needs Q=K verifier + trie data structure + Jacobi loop.

### Honest gap
- The 6.26× ceiling is for highly repetitive workloads (code generation).
  General chat may see 1.5–2×.

### Sources
- [Lookahead Decoding (arXiv 2402.02057)](https://arxiv.org/abs/2402.02057)
- [github.com/hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)

---

## Recommended sequencing

### Immediate investigation (before committing to more EAGLE-3 training)
1. **§1 Path A: probe Gemma 4 LiteRT for MTP heads** (~2 days). If found,
   this reshapes the entire speculation strategy.
2. **§7 DISCO** (~1 day). Free gain on any speculative method. Implement
   as soon as any speculation ships.

### Next sprint (if MTP heads not found in LiteRT)
3. **§1 Path B: train post-hoc MTP heads** (~4 days). More promising than
   continuing EAGLE-3 given ANE memory constraints.
4. **§6 Cross-vocab SD with Qwen 0.5B** (~4 days). Zero-training alternative
   to EAGLE-3. Uses existing model in the downloader.

### After primary speculation method ships
5. **§2 Speculative Streaming** (~3 days). Re-examine whether Medusa-class
   methods work with minimal parameters. Low cost experiment.
6. **§8 SAM-Decoding or §9 Lookahead** (~2–3 days). Retrieval complement to
   learned speculation. Stack with SuffixDecoding from `FUNDAMENTAL_UNTRIED.md`.

### Background / lower priority
7. **§3 Draft & Verify, §4 Kangaroo, §5 SWIFT** — self-speculative variants
   are valuable but overlap with LayerSkip (already designed in
   `FUNDAMENTAL_UNTRIED.md` §4). Pursue only if LayerSkip fails.

---

## Impact on existing roadmap

| Item | Current status | Change |
|------|---------------|--------|
| EAGLE-3 (in training) | High priority | **Re-evaluate**: if MTP heads are viable, EAGLE-3 becomes the fallback, not the primary speculation method |
| LayerSkip (`FUNDAMENTAL_UNTRIED` §4) | Planned | **Complementary**: Draft & Verify (§3) is the training-free version of the same idea |
| SuffixDecoding (`FUNDAMENTAL_UNTRIED` §1) | Planned | **Unchanged**: retrieval-based, composes with all learned methods |
| Q=K verifier (V2 §G1) | Prerequisite | **Unchanged**: still needed by every speculative method |
| Medusa-1 | Rejected (1.3% acc) | **Partially revisit**: Speculative Streaming (§2) may not share the same failure mode |

---

## References

- [Apple Foundation Models 2025 — MTP (arXiv 2507.13575)](https://arxiv.org/abs/2507.13575)
- [DeepSeek-V3 — native MTP (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437)
- [Meta — MTP (arXiv 2404.19737)](https://arxiv.org/abs/2404.19737)
- [vLLM — speculative decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode.html)
- [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)
- [Speculative Streaming (arXiv 2402.11131)](https://arxiv.org/abs/2402.11131)
- [Draft & Verify (arXiv 2309.08168)](https://arxiv.org/abs/2309.08168)
- [Kangaroo (arXiv 2404.18911)](https://arxiv.org/abs/2404.18911)
- [SWIFT (arXiv 2410.06916)](https://arxiv.org/abs/2410.06916)
- [Cross-Vocabulary Lossless SD (arXiv 2502.11926)](https://arxiv.org/abs/2502.11926)
- [DISCO (arXiv 2405.13019)](https://arxiv.org/abs/2405.13019)
- [SAM-Decoding (arXiv 2411.10666)](https://arxiv.org/abs/2411.10666)
- [Lookahead Decoding (arXiv 2402.02057)](https://arxiv.org/abs/2402.02057)
