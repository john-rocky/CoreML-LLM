# Unexplored Approaches — Deep Dive

Six directions complementary (not overlapping) with the current EAGLE-3 /
W8A8 / MQA / DuoAttention stack, chosen because they hit axes the current
stack does not touch.

**First batch — decode / long-context** (scaffolds pushed, ready to run once
EAGLE-3 + W8A8 iPhone bench lands):

- **A. Prefill on A19 Pro GPU tensor cores** — TTFT, compute-bound path
- **B. Mirror Speculative Decoding (Apple 2026)** — NPU+GPU parallel verify
- **C. Cascading KV Cache** — training-free long-context quality

**Second batch — size / UX / compiler** (scaffolds pushed, independent of
the first batch):

- **D. Vocabulary pruning** — -1.7 GB download (262k → ~50k tokens)
- **E. Persistent prefix KV caching** — 4–35× TTFT on cache hit
- **F. MIL graph optimization pass** — 20–40% fewer ops, faster ANE compile

This doc is a design study. Scaffold files are committed (see
`conversion/build_prefill_gpu.py`, `build_eagle3_gpu.py`, `apply_vocab_pruning.py`,
`optimize_mlpackage_graph.py`, `Sources/CoreMLLLM/MirrorSpeculativeLoop.swift`,
`ComputePreferenceLoader.swift`, `PrefixKVCache.swift`,
`conversion/models/gemma4_swa_cascading.py`, `cascading_runtime.py`,
`eval_cascading_quality.py`, `benchmark_prefill.py`), but none have been run
end-to-end on device yet.

---

## A. Prefill on A19 Pro GPU tensor cores

### Problem with current stack
Every speedup in EAGLE-3 / W8A8 / MQA / DuoAttention targets **decode**
throughput. **TTFT (time to first token) is untouched.** On a 2048-token chat
prefill, current path takes ~13 s on iPhone 17 Pro (154 tok/s prefill). EAGLE-3
does nothing for that first wait.

### What A19 Pro added
- GPU cores now ship with **Neural Accelerators** — tensor cores à la NVIDIA.
- Peak: **7.5 TFLOPS FP16 / 13.5 TOPS INT8** per 5-core GPU.
- Optimal tile size 32×32. Matmul transposition hardware-supported, zero
  overhead.
- Access only via **Metal Performance Primitives (MPP)** + Metal Tensor APIs,
  Xcode 26.1+. Direct MSL access isn't available.
- **MPSGraph is the Core ML GPU backend**, so routing a Core ML graph through
  it can unlock tensor cores with zero kernel code.

Apple's own `ml-argmax` benchmark shows **2.5–3.1× speedup on iPhone 17 Pro
GPU vs iPhone 16 Pro** for compute-bound workloads (speech-to-text ConvNet
encoder). ANE only improves 1–1.15× in the same test — because ANE didn't
change between A18 Pro and A19 Pro, GPU did.

### Why this is a prefill win specifically
- Decode @ ctx=2048 reads 48 MB KV per step → memory-bandwidth bound → ANE
  wins because its cache hierarchy is tighter.
- **Prefill @ seq=512 is compute-bound** on Q@K^T and FFN matmuls:
  `512 × 1536 × 1536 ≈ 1.2 GFLOPs per layer`, ×35 layers = ~42 GFLOPs per batch.
  This is exactly the kind of large dense matmul tensor cores are built for.
- GPU tensor cores hit 7.5 TFLOPS; at 50% utilization = 3.75 TFLOPS, prefill
  batch runs in ~11 ms vs ~66 ms on ANE. **~6× speedup on that path**.
- Decoding stays on ANE (bandwidth-bound, ANE wins).

### Implementation sketch
1. **Compile a separate prefill-only mlpackage** with `compute_units=.cpuAndGPU`.
   Keep decode chunks on `.cpuAndNeuralEngine`.
2. Optional: use coremltools `scaled_dot_product_attention_sliced_q` graph
   pass (+34% / −45% memory on long sequence Q slicing, tested on Depth-Anything).
3. Swift side: load two `MLModel` instances per chunk — prefill GPU variant,
   decode ANE variant. Fast-switch on batch size (N>1 → prefill, N=1 → decode).
4. Weights are shared verbatim between prefill and decode — no double-download.

### Expected gain
Conservative: **prefill 154 → 400+ tok/s** on iPhone 17 Pro, TTFT on 2K prompt
**13 s → ~5 s**. Combined with EAGLE-3 + W8A8 decode @ 70–95 tok/s, user-
perceived latency for 30-token reply drops from ~13.3 s to ~5.4 s — a 59% UX
win that no decode optimization can touch.

### Cost
- No training, no calibration.
- Engineering: ~1–2 days to build + test dual-mlpackage pipeline.
- Risk: Xcode 26.1+ dependency; MPP API is new, potential compile-time
  regressions with coremltools 8.x.

### ANE purity
Technically **violates** "ANE-only decode" — but does so for **prefill only**.
Decode path remains pure ANE. This matches Apple's own guidance: "prefer GPU
tensor cores for compute-bound, ANE for bandwidth-bound".

### Composability
- ✅ Fully composes with EAGLE-3 (different phase)
- ✅ Fully composes with W8A8 (both paths benefit from INT8 on A19 Pro)
- ✅ Fully composes with MQA / DuoAttention (KV layout unchanged)

### Verdict
**Highest-confidence unexplored win.** Pure upside, no quality risk, modest
engineering cost. Ship as "fast TTFT mode" flag.

---

## B. Mirror Speculative Decoding (Apple, 2026)

### What it is
Apple Machine Learning Research paper "Mirror Speculative Decoding: Breaking
the Serial Barrier" (2026). **Bidirectional speculation** across heterogeneous
accelerators: draft proposes tokens for target to verify, while target
simultaneously suggests correction paths for the draft — both running in
parallel on NPU + GPU.

### Reported numbers
- **2.8–5.8× wall-clock speedup** on 14–66B server models.
- **+30% over EAGLE-3** baseline on the same hardware.
- Explicitly designed for NPU+GPU parallel execution, with the paper noting
  "iPhone ANE compatibility was a design consideration."

### Why it matters for us
EAGLE-3 is serial: draft runs on ANE, then target runs on ANE to verify. Both
use the same ANE, so they queue.

Mirror splits:
- **Draft on GPU** (tensor cores, good for 1-layer compute-bound work)
- **Target verify on ANE** (bandwidth-bound, ANE wins)
- Both run in parallel → latency is max(draft, verify), not sum.

### ANE feasibility on iPhone 17 Pro
- A19 Pro GPU Neural Accelerators enable cheap draft execution.
- MPP API + MLModel config can route specifically to GPU.
- IOSurface-backed KV can be read by both ANE and GPU without copy (already
  working in our pipeline).
- Risk: the draft hidden state hand-off between draft (GPU) and verify (ANE)
  crosses a memory hierarchy boundary. IOSurface minimizes the cost but it's
  not zero.

### Architecture sketch
```
┌──────────── GPU (tensor cores) ────────────┐
│ eagle3_draft.mlpackage (compute_units=GPU) │
│ ←─ h_prev, e_next                          │
│ ─→ h_out, token, logit                     │
└────────┬───────────────────────────────────┘
         │ IOSurface
         ▼
┌──────────── ANE (matrix engine) ────────────┐
│ verify_chunk{1..4}_K3 (compute_units=ANE)   │
│ ←─ [t_tok_next, proposals[0..K-2]]          │
│ ─→ target_argmax[K]                         │
└─────────────────────────────────────────────┘
         ↕  concurrent via DispatchQueue
```

Draft + verify run on **different accelerators simultaneously**. The critical
path becomes `max(draft_latency, verify_latency)` instead of sum.

### Expected gain
- EAGLE-3 serial: verify ~50 ms + draft ~15 ms = 65 ms per burst.
- Mirror parallel: max(50, 15) = 50 ms per burst. **+30% throughput on the
  speculative path.**
- Stacks with W8A8 (both sides get INT8) for additional 1.3–1.6×.

### Cost
- The *draft* mlpackage has to be re-built with `.cpuAndGPU`. Our existing
  `build_eagle3.py` targets ANE; need a sibling `build_eagle3_gpu.py` or a
  flag.
- Swift dispatch logic: run draft and verify concurrently on two
  `DispatchQueue`s, synchronize results.
- No new model training. The trained draft from EAGLE-3 works as-is.
- ~2–3 days engineering.

### Risk
- Cross-accelerator latency is real on mobile. M5/A19 Pro hardware scheduler
  should handle it, but needs measurement on device.
- Thermal: running GPU + ANE simultaneously is hotter than ANE-only. May
  regress under sustained load (our unique selling point).

### Composability
- ✅ Direct successor / wrapper around EAGLE-3 — same trained draft
- ✅ Composes with W8A8 on both sides
- ⚠️ Partial conflict with "ANE purity" — but only in the speculative path;
  the base decode remains ANE-only
- ❌ Mutually exclusive with "Prefill on GPU" for the same decode burst
  (can't run prefill + draft both on GPU simultaneously — but they're in
  different phases, so just time-share)

### Verdict
**Second-highest-value unexplored win.** If EAGLE-3 lands at 60–95 tok/s, Mirror
brings it to 78–124. Worth implementing after EAGLE-3 iPhone benchmark
baseline is known.

---

## C. Cascading KV Cache

### Problem that WFA exposed
Gemma 4 E2B ships with 7 full-attention layers whose KV grows linearly with
context. At 8K:
- Compute path scales 4× from 2K (we measured this).
- Naive fixed window (WFA) at FW=2048 broke long-context quality (bench
  session confirmed this).
- StreamingLLM + QLoRA recovery is the "proper fix" — but it requires
  fine-tuning.

### What Cascading KV Cache does (arXiv 2406.17808)
Training-free modification: instead of a single sliding window, maintain
**hierarchical windows** where each level retains 1/2 of the previous level's
tokens, biased toward high-attention-score tokens (EMA tracking).

Result: **effective context 3.75× longer** than the fixed window size, with
- +5.6 % on LongBench
- +1.2 % streaming perplexity on PG19
- +0.6 % on MMLU STEM

All without any fine-tune. Works as a drop-in replacement for the KV update
logic.

### ANE compatibility
This is the killer question. Cascading KV requires:
1. Per-token attention score tracking (EMA) — simple accumulator, static shape ✅
2. Hierarchical eviction at specific step counts — decidable at compile time
   from the graph structure (not data-dependent) ✅
3. Multiple small sliding windows instead of one big one — more `slice+concat`
   ops, but each is fixed shape ✅

**Can be compiled to static-shape ANE graph** if we fix the cascade depth and
per-level window sizes at model build time. Dynamic cascading (depth varies
with input) is ANE-hostile, but the fixed-depth variant is not.

### Expected benefit
- With cascading: full-attention layers see "effective 2K" context but the
  retained KV includes tokens from up to ~8K of real context.
- 8K throughput stays at the 2K cost (~31 tok/s), and **long-context quality
  is preserved without fine-tuning**.
- Cascading is **orthogonal to MQA, DuoAttention, EAGLE-3, W8A8** — all
  compose.

### Comparison to StreamingLLM + QLoRA
| Axis                      | Cascading KV (A) | StreamingLLM + QLoRA (B) |
|---------------------------|------------------|--------------------------|
| Training cost             | **0**            | ~4-8 h A100              |
| Quality recovery          | +5.6% LongBench  | Typically +6-8 %         |
| ANE implementation        | Static, feasible | Static, feasible         |
| Gemma-4 architecture touch| None             | None                     |
| Composability             | Full             | Full                     |
| Risk                      | Untested on Gemma4| Well-understood          |

Cascading is **strictly cheaper** — no fine-tune needed. If it performs on
Gemma 4 as the paper reports on Llama-2, it's a free quality-preservation
layer on top of WFA-style compute savings.

### Implementation sketch
1. `conversion/models/gemma4_swa_cascading.py` — new model module modeling
   the full-attention KV update as a cascade of 3 buckets (e.g., [first 4
   tokens sink] + [recent 512 W1] + [sampled 512 W2] + [sampled 1024 W3]).
2. Attention score EMA accumulator registered as a small state buffer.
3. Eviction indices computed from EMA-decay threshold, evaluated statically.
4. `build_speculative.py` variant that enables cascading (flag).

### Cost
- ~2-3 days conversion code.
- Validation: LongBench v2 subset via `eval_longbench.py` (already pushed).
- Zero fine-tune, zero calibration.

### Verdict
**Highest ROI for long-context 8K quality.** Until we have it, we can't ship
"8K mode" without either quality loss or fine-tune. Cascading is the only
training-free option that preserves quality.

---

## D. Vocabulary pruning (262k → ~50k)

### Problem
Gemma 4 E2B's 262k-row SentencePiece vocab covers 100+ languages and code.
For a Japanese+English+some-code app, most of that is dead weight. Three
tensors blow up with vocab size:

  embed_tokens:       V × hidden × fp16              = 0.8 GB
  embed_tokens_per_layer: V × L × per-layer × fp16   = 4.6 GB  ← dominant
  lm_head:            V × hidden × fp16              = 0.8 GB

(INT8 halves each on disk.) Total ship-time cost of the vocab alone on
Gemma 4 E2B: ~2.6 GB of the 2.7 GB package.

### What vocab pruning does
Rank tokens by corpus frequency, union with must-keep specials (BOS/EOS/
PAD/turn markers/image/audio placeholders + low reserved range), retain
top K (default 50k = 19% of original). Slice the three embedding tables
accordingly. Emit `vocab_remap.json` so the runtime can round-trip ids
between the original tokenizer (which remains usable for encoding and
decoding) and the pruned model's input/output space.

### Expected size delta
  Original (fp16) 2.7 GB → Pruned (fp16) 1.0 GB
  After existing INT4 palettization → equivalent footprint to Apple's
  shipped 3B on-device model.

### Quality
Training-free at keep ≥ 50k the quality impact is typically < 1 % on
common benchmarks for models with heavy multilingual+code vocabs. Below
40k, a short QLoRA re-stabilize (hours on A100) is recommended.
`conversion/eval_longbench.py` is the gate.

### ANE compat
✅ Pure tensor row slicing. The embedding lookups become cheaper (lookup
cost is linear in retained V, not original V). No graph changes required
in the attention layers. Model config's `vocab_size` is updated; CoreML
conversion re-runs unchanged.

### Scaffold: `conversion/apply_vocab_pruning.py`
Produces a pruned HF model + `vocab_remap.json`. Companion to the
existing dry-run `prune_vocab.py` analyzer.

### Verdict
**Day-1 UX improvement**. Makes CoreML-LLM competitive with Apple's
Foundation Model on footprint while retaining the "any open model"
moat. Implementation is the simplest of all unexplored directions.

---

## E. Persistent prefix KV cache

### Problem
TTFT for a 2K system prompt is ~13 s on iPhone 17 Pro (ANE prefill). For
chat apps with a **stable system prompt** (assistant persona, memory,
RAG context), that prefill is paid every cold start and every session
switch. EAGLE-3 / W8A8 do nothing for this.

### What persistent prefix KV does
After prefilling a prefix, serialize all KV buffers (sliding + full-
attention + shared) to disk under a hash of the prefix token sequence.
On future cold starts: tokenize the prefix, hash, probe cache. On hit,
deserialize KV into the engine and skip the forward pass entirely.

### Expected speedup
Literature: 4–35× TTFT at 1–4k prefix, scaling to ~136× at 32k. Size is
small — a 2K prefix stores ~48 MB fp16 of full-attn KV plus ~7 MB sliding.
INT4 quantization cuts it to ~12 MB. An app with 64 cached prefixes ≈
1 GB sandbox storage.

### ANE compat
✅ No graph change. Only affects Swift-side state management. The KV
buffers are already IOSurface-backed half-precision arrays; serialization
is a memcpy to disk.

### Scaffold: `Sources/CoreMLLLM/PrefixKVCache.swift`
Declares a `PrefixKVSnapshotable` protocol (which `ChunkedEngine` is
expected to conform to) plus a file-system cache with LRU eviction,
SHA-256-based key, model/version validation, and automatic location in
the app's Caches directory.

### Verdict
**Quick UX win**, deliverable alongside or ahead of the decode-path
work. No training, no calibration. Requires Swift-side
`PrefixKVSnapshotable` implementation in ChunkedEngine (a few hundred
lines of buffer I/O).

---

## F. MIL graph optimization pass

### Problem
The converted mlpackages contain many small ops (reshape, transpose,
add, mul) that could be fused by the CoreML compiler's built-in pass
pipeline but aren't, because `coremltools.convert()` defaults to a
conservative pass set for compatibility. Each extra op costs a kernel
launch on ANE and adds to first-run compile time (1–2 min cold).

### What MIL graph optim does
Reload an already-converted mlpackage, re-extract its MIL program, run
additional passes — `dead_code_elimination`, `const_elimination`,
`fuse_linear_bias`, `fuse_layernorm_or_instancenorm`,
`merge_consecutive_reshapes`, `merge_consecutive_transposes`,
`fuse_matmul_weight_bias`, `fuse_gelu_*` — and re-save. Weights are
byte-identical; the MIL structure is leaner.

### Expected gain
Typical 20–40% op-count reduction on transformer chunks. First-run
compile time shrinks proportionally; decode throughput sees small gains
(1–5%) from fewer kernel launches.

### ANE compat
✅ All passes produce equivalent semantics; the resulting graph remains
within ANE's supported op set. The optional `--verify-equivalence` flag
runs both models on random input and compares outputs byte-for-byte.

### Scaffold: `conversion/optimize_mlpackage_graph.py`
Pass list is configurable; default is a conservative set known safe on
Gemma-shaped graphs. Run once per chunk post-conversion.

### Verdict
**Free win if it works**, bounded risk if it doesn't (revert to un-
optimized mlpackage). One hour of testing per chunk decides it.

---

## Priority ordering (all 6)

| # | Approach | Confidence | Effort | Payoff | Quality risk | Composes with current stack |
|---|---|---|---|---|---|---|
| 1 | **D. Vocab pruning**  | **High**   | 1 day (+QLoRA half-day) | -1.7 GB download | <1 % at keep≥50k | ✅ all |
| 2 | **A. GPU prefill**    | **High**   | 1–2 days | TTFT 60 % ↓ | none      | ✅ all |
| 3 | **E. Prefix KV cache**| **High**   | 1 day Swift | TTFT 4–35× on hit | none | ✅ all |
| 4 | **F. MIL graph optim**| Medium-High| 1 day | 20–40 % ops down, compile faster | none (equiv verified) | ✅ all |
| 5 | **B. Mirror SD**      | Medium-High| 2–3 days | +30 % over EAGLE-3 | thermal watch | ✅ EAGLE-3 successor |
| 6 | **C. Cascading KV**   | Medium     | 2–3 days | 8K quality preserved | paper-vs-our-model gap | ✅ all |

Total engineering for all six: ~10 working days.

### Recommended sequencing (when current stack lands)

1. **Immediate (parallel with EAGLE-3 iPhone bench)** → run **D (vocab
   pruning)**: first-class shipping artifact, matches Apple's 1 GB footprint.
   → try **F (graph optim)**: low-cost try-it, revert if it doesn't pan out.
   → implement **E (prefix KV cache)**: UX win independent of decode path.
2. **After EAGLE-3 ships on iPhone** → **A (GPU prefill)** in parallel with
   **B (Mirror SD)**. A is the safer quick win, B is the larger reward.
3. **After W8A8 ships** → if long-context 8K is still a product goal,
   implement **C (Cascading KV)** as the training-free quality layer
   before falling back to StreamingLLM+QLoRA.

### What this means for the Gemma-4 ceiling

Updated ceiling checklist:

- [ ] **D. Vocabulary pruning** (−1.7 GB, ship-side)
- [ ] **E. Persistent prefix KV caching** (4–35× TTFT on cached prefixes)
- [ ] **F. MIL graph optimization pass** (op count −20-40%, compile faster)
- [ ] **A. A19 Pro GPU prefill** (TTFT 13 s → 5 s)
- [ ] **B. Mirror Speculative Decoding** (+30% over EAGLE-3)
- [ ] **C. Cascading KV Cache** (8K quality without fine-tune)

With all six plus the existing stack (EAGLE-3 + W8A8 + MQA + DuoAttention),
the Gemma-4 ceiling moves to **80–120 tok/s @ 8K, TTFT ~5 s, 1 GB
download**. At that point leaving Gemma-4 scope for a distilled "Turbo
SKU" (#1 / #5a in `ALTERNATIVE_APPROACHES.md`) becomes genuinely the next
bottleneck.

---

## References
- Apple Machine Learning Research — [Mirror Speculative Decoding (2026)](https://machinelearning.apple.com/research/mirror)
- Apple A19 / M5 GPU Neural Accelerators — [Tzakharko benchmark](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)
- Argmax — [iPhone 17 on-device inference benchmarks](https://www.argmaxinc.com/blog/iphone-17-on-device-inference-benchmarks)
- [Cascading KV Cache (arXiv 2406.17808)](https://arxiv.org/html/2406.17808v1)
- coremltools — [sliced_q attention graph pass](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.passes.defs.html)
- Apple — [Metal Performance Primitives (Xcode 26.1+)](https://developer.apple.com/metal/)
