# Fundamental Untried Approaches — the bets that could actually move the needle

Round 3 of the unexplored-approaches investigation (2026-04-13). Where round 2
(`UNEXPLORED_APPROACHES_V2.md`) was honest that G1–G5 were runtime tweaks, this
round is the result of pushing back against that — *if everything documented
so far is incremental, what is the actually fundamental lever we have not
tried?*

The four candidates below clear two bars round 2 did not:
- **Lossless** (no fine-tune, no quality regression risk beyond fp16 rounding)
- **Decode-throughput-changing**, not just runtime mechanics — credible
  individual ≥ 1.5× wall-clock claims with citations

Plus one **reframing** (§0) that re-prices everything else in the existing
roadmap. None of these are implemented or measured on device. All numbers are
expectations from cited literature + the codebase's known dispatch shape.

---

## 0. The reframing: ANE peak utilization is ~0.07%, the bottleneck is dispatch count

The reverse-engineering work that landed in late 2025 / early 2026 (Orion,
arXiv 2603.06728; "Inside the M4 ANE Part 2", maderix.substack 2025-12)
quantified the ANE microarchitecture for the first time. The relevant
numbers for this codebase:

- iPhone 17 Pro ANE peak: **19 TFLOPS FP16**.
- A 2.7B model at 50 tok/s sustains ~13.5 GFLOPS = **0.07% of peak**.
- **INT8 and FP16 throughput are identical** on ANE — the hardware
  dequantizes INT8 to FP16 internally before compute.
- **`matmul` is ~3× slower than `Conv1x1`** on ANE; matmul is emulated,
  conv is the native primitive.
- **Per-dispatch IOSurface round-trip**: ~2.3 ms per KV-bearing chunk on
  M4 (likely similar on iPhone 17 Pro).
- Single-op utilization is ~30% of peak; only deep graphs (32+ layers in
  one dispatch) hit ~94% of peak.

What this reorders in our existing roadmap:

| Item in current docs | Why we pursued it | What the data says |
|---|---|---|
| **W8A8** (`SPEED_8K.md` §1 A2, Tier D) | Apple's "INT8-INT8 fast path" claim | INT8 has the **same throughput as FP16 on ANE**. The "1.3-1.6×" Apple ResNet50 number is bandwidth, not compute. Even if W8A8 compiled, the wall-clock win would be ≤ the bandwidth fraction (small for 8K decode where compute dispatch dominates). Continuing to chase it has been wrong target. |
| **INT8 KV cache** (`SPEED_8K.md` §1 A2) | Halve KV bandwidth | Already concluded "no ANE speedup" — same root cause: ANE re-dequantizes. |
| **MIL graph optim** (`UNEXPLORED §F`) | Reduce op count | Op count matters less than dispatch count; per-op compute is already free. |
| **GPU prefill** (`UNEXPLORED §A`) | Compute-bound win for prefill | Still valid — prefill genuinely is compute-bound. Decode is dispatch-bound. |

The single most important finding: **decode wall-clock is dominated by the
number of ANE re-entries, not by compute or bandwidth in the classical
sense**. The four bets below all attack this.

---

## 1. SuffixDecoding — CPU-only draft, zero ANE competition

### What it is
NeurIPS 2025 Spotlight (arXiv 2411.04975). Build a suffix tree over (a)
the current request and (b) every prior request's output the app has ever
generated. At each decode step, look up the last *k* output tokens in the
tree, retrieve the most-frequent continuation, ship it as a draft to the
target's Q=K verifier.

### Numbers
- **1.9–5.3× wall-clock** on chat/agentic workloads
- **Beats EAGLE-2/3 by 2.8×** in production benchmarks
- **86.4% prefix cache hit** on voice-clone workloads
- **75–95% hit rates** on multi-turn chat with stable system prompts
- Production-deployed in **vLLM ArcticInference** (Snowflake)

### Why this is fundamental, not a tweak
Every other speculative method here either (a) needs training (EAGLE-3,
Medusa, ReDrafter, LayerSkip-trained) or (b) uses ANE time for the draft
(EAGLE-3 again, Mirror SD's draft side). SuffixDecoding's draft cost is
**~20 µs per token on CPU**. The target ANE has 100% of its time for
verification. **No other proposal in our docs decouples draft cost from
ANE utilization the way this does.**

It is also genuinely cumulative across sessions — unlike Prompt Lookup
Decoding (which only matches against the current prompt), SuffixDecoding
remembers everything the model has ever output for this user. Hit rates
*climb over time* as the tree fills up with the user's actual usage
patterns.

### ANE compatibility
Trivial. The draft is CPU-side; the target is the existing chunk pipeline
plus a Q=K verifier (which `Sources/CoreMLLLM/SpeculativeLoop.swift`
already scaffolds, and which G1 in `UNEXPLORED_APPROACHES_V2.md` plans).
Same Q=K mlpackage works for SuffixDecoding, EAGLE-3, PLD, and Lookahead
— pick the draft source, the verifier is shared.

### Composability
- ✅ With EAGLE-3: use SuffixDecoding when tree has high-confidence match,
  fall back to EAGLE-3 draft otherwise. **Strictly additive.**
- ✅ With W2A16 weights, MLState KV, in-model top-K — orthogonal.
- ⚠️ Privacy / storage: the user-output tree must persist across sessions.
  Default disk budget ~50–500 MB depending on retention. App can scope it
  per-conversation if privacy demands it (loses cross-session benefit).

### Cost
- ~2-3 days Swift: suffix-tree data structure (or simpler suffix-array
  with periodic rebuild), tokenizer-stream ingestion, candidate selection,
  reuse the existing/planned `verifyCandidates(K:)` path.
- 0 training, 0 model surgery, 0 reconversion of the existing chunks.
- The only mlpackage change is the Q=K verifier from G1, needed by every
  speculative method anyway.

### Honest gap
- No published numbers on Gemma-class (≤4B) models specifically. The
  reported 1.9–5.3× is on 7B+ workloads. Smaller models may see lower
  absolute speedup because the per-step compute is already small relative
  to the dispatch overhead — but for the same reason, fewer ANE entries
  per accepted token may help proportionally more.
- Workload-conditional. Open-ended creative generation (no repetition)
  collapses to ~1.0×. The target user workloads (chat with stable system
  prompts, RAG/Q&A, code editing) all have high repetition.

### Sources
- [SuffixDecoding (arXiv 2411.04975)](https://arxiv.org/abs/2411.04975)
- [CMU CSD blog](https://www.cs.cmu.edu/~csd-phd-blog/2025/suffix-decoding/)
- [Snowflake ArcticInference / vLLM integration](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)

---

## 2. MLState (iOS 18+) for stateful KV — eliminate the per-dispatch round-trip

### Observation that re-prices this
The current `gemma4_stateless_chunks.py` was chosen over an `MLState`
variant because *at the time of that decision* "MLState introduces int64
state indices that break ANE placement on the model sizes we ship"
(`docs/EXPERIMENTS.md`). That decision was made before iOS 18 stateful
models matured and before iOS 26 shipped. **It has not been re-evaluated
on the current OS.**

Per Orion's measurements, the dominant cost in our 4-chunk pipeline is
not compute, not DRAM bandwidth, but the **per-prediction IOSurface
round-trip** that ships KV state in and out of ANE memory. Stateful
models keep KV resident inside ANE-managed memory; the model takes only
the new token's hidden state as input and writes incrementally into its
own state.

### Why this could be fundamental
At ctx=8192, every decode step currently passes:
- 7 sliding KV blocks × (W=512) × 256 ≈ 7 MB
- 7 full KV blocks × 8192 × 512 ≈ 56 MB
- ≈ 60+ MB of KV crossing ANE/host per chunk on full-attn-bearing chunks

Even if the IOSurface backing makes that "zero-copy" in the bus sense,
the **dispatch boundaries** still serialize and synchronize — every
chunk pays the round-trip. With MLState the KV input never appears in
the predictionInput dict; the ANE re-uses its own resident state, and
the only host-side work is feeding the next token's embedding.

### Numbers (bounded)
- If MLState compiles cleanly and the compiler fuses the slot update
  into the attention op: **1.3–2.0× decode wall-clock**
- If the compiler emits the same logical read-modify-write pattern under
  the hood: **1.0×** (no improvement)
- The risk is asymmetric: cost of trying ≈ 3-4 days conversion rewrite,
  no quality risk, and even a negative result narrows the search.

### Why nobody has shipped this for an 8K LLM yet
- ANEMLL predates `MLState` and explicitly notes it avoids it for OS
  compatibility (their min-OS is older).
- `smpanaro/coreml-llm-cli` similarly predates the API.
- Apple's reference `ml-ane-transformers` is from 2022, before stateful
  models.
- Apple's own Foundation Models 3.18B uses stateful KV inside the OS
  daemon — not exposed to third-party developers, but the same primitive
  is now public via `MLState`.

### ANE compatibility
- iOS 18+ shipping API, fully documented.
- Static state shapes work; the existing `(slots, 1, ctx, max_hd)`
  layout maps directly to `MLStateType`.
- The mask-based KV update (`K*(1-umask) + new_k*umask`) needs to lower
  to a `slice_update` semantics. Whether the iOS 26 ANE compiler honors
  this is the open question that one day of prototyping answers.

### Cost
- ~3-4 days: rewrite `gemma4_stateless_chunks.py` to declare KV tensors
  as `coremltools.StateType`, drop the `K_in/K_out` I/O pairs, replace
  the mask-based update with `slice_update_along_axis(K_state, new_k,
  position)`. Reconvert all four chunks.
- Swift side: drop the persistent `MLMultiArray` KV buffers, use
  `MLState`-bearing predictions instead. Remove the per-step `copyBack`
  in `ChunkedEngine.predictStep`.
- A/B parity test on a held-out prompt set vs the stateless variant.

### Composability
- ✅ With SuffixDecoding (different layer of the stack)
- ✅ With W2A16 (state stays fp16; weights are orthogonal)
- ✅ With G1 multi-function Q=K (state shape is invariant in Q dim)
- ⚠️ Mutually exclusive with the *existing* explicit-KV-I/O pipeline —
  this is a wholesale rewrite of the KV layer, not a flag.

### Sources
- [Apple — MLState documentation](https://developer.apple.com/documentation/coreml/mlstate)
- [coremltools — Stateful Models guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [Orion (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728) — dispatch-overhead measurements
- [Inside the M4 ANE Part 2 — maderix.substack](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Enclave AI — KV Cache explained on iPhone (2026-03-22)](https://enclaveai.app/blog/2026/03/22/kv-cache-explained-local-llm-memory-iphone-mac/)

---

## 3. W2A16 palettization — the quantization Apple actually ships, not W8A8

### Observation that re-prices this
The roadmap has chased **W8A8** as the path to ANE INT8 compute — and
the iPhone ANE compiler keeps rejecting it (`SPEED_8K.md` Tier D, dated
2026-04-13: "REJECTED — `ANECCompile() FAILED`"). Meanwhile Apple's own
2025 tech report on Apple Intelligence Foundation Models confirms their
shipping recipe for the on-device 3.18B is:

1. **Block KV-share** (which Gemma 4 already does)
2. **2-bit QAT weights** (W2A16) — the dominant quantization
3. **ReDrafter** speculative decoding

**They do not use W8A8.** Per Orion's INT8/FP16-throughput-equivalence
finding, this makes sense: there is no INT8 compute speedup on ANE; the
only quantization win is **bandwidth**, which scales with the bit-width
of the weights, not the activations. W2 cuts weight bandwidth 8× vs FP16,
W8 cuts it 2×. The 4× ratio between them is the real shipping gap.

### What W2A16 actually does on ANE
- coremltools `linear_quantize_weights(mode="palettization", n_bits=2,
  granularity="per_grouped_channel")` produces a lookup-table-encoded
  weight tensor with 4 codewords per group.
- ANE supports lookup-table palettization natively (this is how Apple's
  2-bit QAT model runs). Unlike `linear_quantize_activations` which
  emits `quantize`/`dequantize` MIL ops that the iPhone ANE compiler
  rejects, palettization is a *weight encoding*, not an activation
  pathway. **Different code path, no compile failures observed in the
  wild.**
- At W2: weight payload ≈ 0.34 GB (vs 1.35 GB at INT4 palettized vs
  2.7 GB at FP16). At 8K decode, the dominant memory motion is the KV
  cache, not weights — but for compute-bound prefill, this is a clear
  win, and for decode the smaller working set keeps more of the model
  in SRAM (Orion: 32 MB SRAM, working sets ≤ 24 MB stay at peak).

### Numbers
- Apple's own 3.18B at W2-QAT runs at parity quality with FP16 (per the
  Foundation Models tech report). That's QAT — quality-aware training.
- *Post-training* W2 palettization (no fine-tune) typically has 1–3 PPL
  bump on standard benchmarks. Acceptable for the 3rd party path *if*
  validated on LongBench.
- **Wall-clock**: 1.4–2.0× decode at 8K is the conservative range, from
  the bandwidth ratio. May be more if working set crosses the 24 MB
  SRAM threshold.

### Honest caveat
- The 1.4–2.0× is bandwidth-bounded. If MLState (§2) lands first and
  eliminates per-dispatch KV bandwidth, the W2 win shrinks to "weights
  only" which is a smaller fraction of total bandwidth. **Sequence
  matters: do W2 first, then re-measure before committing to MLState.**
- Quality validation (LongBench v2 + held-out chat set) is mandatory
  before shipping. If post-training W2 regresses too far, fall back to
  W4 palettized (already shipping in `exporter.py`).

### Cost
- ~1-2 days: existing `_quantize_model` in `conversion/exporter.py`
  already supports palettization at INT4. Lower the `n_bits` parameter
  from 4 to 2, set `granularity="per_grouped_channel"`, calibrate group
  size on Mac Studio first.
- A/B quality run on held-out prompts.
- 0 model surgery; same Gemma 4 architecture.

### Composability
- ✅ With SuffixDecoding, MLState, EAGLE-3, MQA — orthogonal.
- ✅ With G1 multi-function — palettized weights deduplicate identically
  across functions.

### Sources
- [Apple Foundation Models 2025 Tech Report (arXiv 2507.13575)](https://arxiv.org/abs/2507.13575)
- [Apple ML — Foundation Models 2025 Updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)
- [coremltools — Palettization Performance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)
- Orion arXiv 2603.06728 — INT8/FP16 throughput equivalence on ANE

---

## 4. LayerSkip exploiting Gemma 4's KV-share boundary at L15

### The structural insight no paper has connected
Gemma 4 E2B's 35 layers split as:
- L0–L14: own K/V (computed locally)
- **L15–L34: KV-shared** — all 20 layers read kv13 (sliding) and kv14
  (full) from L13/L14

This means the post-L14 stack is **architecturally already a "prediction
head" over a frozen KV view**. There are no K/V projections in those 20
layers; they only do Q-side computation against the cached K/V.

LayerSkip-style self-speculative decoding (Meta, ACL 2024, integrated
into HuggingFace transformers Nov 2024) exits at some layer *k*, treats
the partial-stack output as a draft, and verifies with the full stack.
The catch on most models: an "early exit" disturbs the KV cache state
because layers k+1 through N would normally have written their own K/V.

**On Gemma 4, exiting at L15 disturbs nothing.** L15+ never wrote K/V
to begin with. The early-exit boundary is *naturally* at L14 → L15.

### What this enables
- **Draft path**: run chunk1 + chunk2 only (L0–14), use the post-L14
  hidden state plus a small projection head (~5M params) as a token
  prediction.
- **Verify path**: run chunk3 + chunk4 (L15–34) on the K accepted draft
  tokens, in a single Q=K dispatch.
- Draft compute = 14/35 = 40% of full forward. Even at 50% acceptance,
  amortized cost per accepted token = (0.4 + 0.6/2) ≈ 0.7× full forward
  → 1.4× speedup.

### Why this is not just "EAGLE-3 lite"
- **No separate draft model to ship.** Same weights, same mlpackage
  pipeline, just a different entry point that skips chunks 3-4.
- **No training required for the architecture** — Gemma 4's KV-share is
  already the early-exit boundary. Only the small "draft head" needs
  training (one linear layer on top of L14 hidden state, supervised by
  the model's own outputs on the existing EAGLE-3 corpus).
- Acceptance unknown without measurement, but the structural alignment
  with KV-share is a stronger prior than a generic LayerSkip on a
  homogeneous transformer.

### Numbers (speculative)
- Best case: 50%+ acceptance → 1.4–1.6× wall-clock
- Median case: 30–40% acceptance → 1.15–1.3×
- Worst case: <20% acceptance → ≤ 1× (overhead dominates) — same risk
  profile as Medusa

### Cost
- Draft head training: hours on a single A100 using the existing
  EAGLE-3 corpus collection pipeline (`collect_eagle_hidden_states.py`
  but exit at L14 instead of the EAGLE-3 hidden state of choice).
- Conversion: package chunks 1-2 with a small added classifier head as
  the "draft" mlpackage. Chunks 3-4 unchanged (used as verifier).
- Swift: a TaskGroup that runs draft → conditional verify.
- ~3-4 days end-to-end.

### Composability
- ✅ With SuffixDecoding: use suffix tree first; fall back to LayerSkip
  draft on tree miss.
- ✅ With EAGLE-3 (when it lands): use whichever draft has higher
  confidence per step.
- ✅ With MLState: state lives in the verifier path only; draft is
  stateless.
- ⚠️ Conflicts with G1 multi-function only in the sense that both want
  the verifier-side Q=K function — but they're the same need.

### Honest gap
- Acceptance rate on Gemma 4 specifically is unmeasured. The "structural
  alignment with KV-share" argument is a *prior*, not a proof.
- Worst case is worse than SuffixDecoding (which never regresses below
  ~1.0× on bad workloads). LayerSkip-style draft can actively hurt if
  acceptance is too low.

### Sources
- [LayerSkip (arXiv 2404.16710)](https://arxiv.org/abs/2404.16710)
- [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)
- [HuggingFace transformers — assisted decoding integration](https://huggingface.co/blog/assisted-generation)
- [Maarten Grootendorst — Visual guide to Gemma 4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4) (KV-share structure)

---

## Composite plan — the only stack that has a credible >2× lossless story

| Step | Action | Standalone gain | Cumulative @ 8K | Days |
|---|---|---|---|---|
| 0 | Bench current baseline (already done: 14.5 tok/s) | — | 14.5 | 0 |
| 1 | **W2A16 palettization** — Apple's actual recipe | ×1.4–2.0 | 20–29 | 2 |
| 2 | **MLState stateful KV** — re-evaluate on iOS 26 | ×1.3–2.0 | 26–58 | 4 |
| 3 | **Q=K verifier** (G1 from V2 doc, prerequisite) | ×1.0 (enabling) | 26–58 | 2 |
| 4 | **SuffixDecoding** on chat/agentic workloads | ×1.8–3.0 | 47–174 | 3 |
| 5 | **LayerSkip on Gemma-4 KV-share boundary** (optional) | ×1.15–1.6 | 54–278 | 4 |

Realistic wedge (multiplicative compounding × 0.65 ANE-overhead correction):
- Conservative (W2 hits low, MLState 0×, Suffix 1.8×, no LayerSkip):
  14.5 × 1.4 × 1.0 × 1.8 × 0.65 ≈ **24 tok/s @ 8K**
- Median (W2 1.7×, MLState 1.4×, Suffix 2.3×, LayerSkip 1.3×):
  14.5 × 1.7 × 1.4 × 2.3 × 1.3 × 0.65 ≈ **67 tok/s @ 8K** ← *crosses 50*
- Optimistic (everything lands near upper bound): 100+ tok/s @ 8K

**Critical sequencing**: W2A16 first (cheapest, validates the bandwidth
hypothesis); MLState second (confirms or refutes the dispatch-overhead
hypothesis); SuffixDecoding third (the speculative method that doesn't
require EAGLE-3 to land first); LayerSkip last (highest risk, lowest
training cost speculative).

---

## What this reframes in the existing roadmap

After this list lands (or partially lands), the items in `SPEED_8K.md`
and `ALTERNATIVE_APPROACHES.md` rank differently:

| Item | Old priority | New priority | Reason |
|---|---|---|---|
| W8A8 calibration | High (Tier D) | **Skip** | INT8 compute on ANE = FP16; only bandwidth wins, and W2A16 does that better |
| INT8 KV cache | Skip (already concluded) | Skip | Same root cause |
| EAGLE-3 (in training) | High | **Still high** | Composes with SuffixDecoding as fallback. Lossless. |
| MQA conversion | Selected | **Defer** | Helps full-attn bandwidth — but MLState may obviate this |
| GPU prefill | High (UNEXPLORED §A) | **Still high** | Different phase. Doesn't conflict. |
| Mirror SD | Medium-High | **Defer** | Strictly worse than SuffixDecoding for chat-style usage; revisit if pure compute throughput becomes the bottleneck |
| Vocab pruning | High | Still valid | Orthogonal UX win |

---

## What this list is *not*

- Not a guarantee. The MLState bet specifically is a measure-or-die
  proposition: one day of prototyping decides whether it's 0× or 2×.
- Not exhaustive. The reframing in §0 (matmul vs Conv1x1, dispatch count
  as the bottleneck) opens further investigations not enumerated here.
- Not implementations. This is a research note. Each item still requires
  a converter pass + Swift wiring + on-device A/B before any number above
  becomes a measurement.

---

## References

- [Orion: Characterizing and Programming Apple's Neural Engine (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728)
- [Inside the M4 ANE Part 2 — maderix.substack](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Apple Foundation Models 2025 Tech Report (arXiv 2507.13575)](https://arxiv.org/abs/2507.13575)
- [Apple ML — Foundation Models 2025 Updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)
- [Apple — MLState documentation](https://developer.apple.com/documentation/coreml/mlstate)
- [coremltools — Stateful Models guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [coremltools — Palettization Performance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)
- [SuffixDecoding (arXiv 2411.04975)](https://arxiv.org/abs/2411.04975)
- [Snowflake ArcticInference — vLLM SuffixDecoding integration](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)
- [LayerSkip (arXiv 2404.16710)](https://arxiv.org/abs/2404.16710)
- [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)
- [Maarten Grootendorst — Visual guide to Gemma 4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)
- [Enclave AI — KV Cache explained on iPhone (2026-03-22)](https://enclaveai.app/blog/2026/03/22/kv-cache-explained-local-llm-memory-iphone-mac/)
