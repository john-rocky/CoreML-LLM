# ANE Optimization Survey — Beyond LiteRT-LM

**Date:** 2026-04-14
**Sources:** Apple AFM (arXiv 2507.13575), ANEMLL (github.com/Anemll/Anemll),
ExecuTorch CoreML backend (github.com/pytorch/executorch), Sequoia (arXiv 2402.12374),
ReDrafter (arXiv 2403.09919), SwiftKV
**Scope:** Non-LiteRT optimization techniques applicable to ANE LLM inference

---

## Executive Summary — Top 5 New Findings

1. **Prefill bypass (TTFT -40%)** — Apple's AFM paper reveals: L15-34 never produce
   KV during prefill. Skip chunk3+4 for all prompt tokens except the last one.
   Zero model changes, zero quality loss. (AFM / SwiftKV)

2. **Y-tree speculative topology (+15% over linear chain)** — Sequoia analysis shows
   the optimal T=4 tree is [root → 2 children → 1 grandchild], giving 3.20 vs 2.96
   tokens/cycle at p=0.6. Requires only 2 drafter calls vs 3 (cheaper AND better).
   (Sequoia)

3. **RMSNorm-as-LayerNorm trick** — ANEMLL's `concat([x, -x])` repurposes ANE's
   hardware LayerNorm kernel for RMSNorm, avoiding manual variance computation.
   (ANEMLL)

4. **Output buffer pooling** — ExecuTorch's NSCache-based MLMultiArray recycling
   avoids per-prediction allocation overhead. Combined with `outputBackings` for
   zero-copy output writes. (ExecuTorch)

5. **Ping-pong + ring buffers for ANE race conditions** — ANEMLL uses 2-deep
   ping-pong for chunked models and 16-deep ring buffers for monolithic models
   to prevent async read/write hazards. (ANEMLL)

---

## 1. Prefill Bypass (Apple AFM / SwiftKV)

**Source:** AFM Tech Report (arXiv 2507.13575), SwiftKV (Snowflake)

Apple states: "because Block 2 does not produce any keys or values, the prefill
stage is able to bypass all of its computation and reduce TTFT by ~37.5%."

**Applied to Gemma 4 E2B:**
- L0-14 (chunk1+2): produce all 15 unique KV caches
- L15-34 (chunk3+4): Q-only layers, read kv13/kv14, never produce KV
- During prefill of N prompt tokens, only chunk1+2 are needed for tokens 1..N-1
- Only the LAST prompt token needs the full pipeline (chunk1-4) to produce output

**Implementation (ChunkedEngine.swift):**
```
for token in prompt_tokens[0..<N-1]:
    run chunk1(token)  // produces kv0-kv7
    run chunk2(token)  // produces kv8-kv14, kv13_shared, kv14_shared
    // SKIP chunk3, chunk4

// Last token: full pipeline
run chunk1(prompt_tokens[N-1])
run chunk2(prompt_tokens[N-1])
run chunk3(prompt_tokens[N-1])  // reads kv13/kv14
run chunk4(prompt_tokens[N-1])  // reads kv13/kv14, outputs logits
```

**Expected impact:** Chunk3+4 account for ~47% of per-token prefill time
(15.2 + 17.3 = 32.5ms out of 67ms). Skipping them for N-1 tokens:
- Current 8K TTFT: ~870ms (8192 * 67ms / chunk_prefill_efficiency)
- Optimized: ~870ms * 0.53 = ~460ms for prompt processing
- **TTFT reduction: ~47%**

**Effort:** 0.5 day Swift. **Quality impact:** None (identical output).

---

## 2. Y-Tree Speculative Topology (Sequoia)

**Source:** Sequoia (arXiv 2402.12374), OPT-Tree (arXiv 2406.17276)

For T=4 verify budget and acceptance rate p=0.6, the optimal tree topology is:

```
Linear chain:           Y-tree (optimal):
[t0]->[d1]->[d2]->[d3]  [t0]--+--[d1a]--[d2a]
                              |
                              +--[d1b]
```

| Metric | Linear (K=3) | Y-tree | Delta |
|---|---|---|---|
| Drafter calls | 3 | 2 | **-33%** |
| E[tokens/cycle] at p=0.6 | 2.96 | 3.20 | **+8.1%** |
| Wall-clock (5ms draft, 67ms verify) | 82ms | 77ms | **-6.1%** |
| Throughput | 36.1 tok/s | 41.6 tok/s | **+15.2%** |

**How it works:**
- Step 0: run drafter once → take top-1 (d1a) AND top-2 (d1b) from `top_k_indices`
- Step 1: run drafter once from d1a → get d2a
- Verify: batch [t0, d1a, d1b, d2a] with tree-aware causal mask
- Accept: walk tree to find longest accepted root-to-leaf path

**Mask structure (4 query positions over KV cache):**
```
t0:  sees [0..p]                          (root)
d1a: sees [0..p] + t0                     (child 1)
d1b: sees [0..p] + t0                     (child 2, NOT d1a)
d2a: sees [0..p] + t0 + d1a              (grandchild, NOT d1b)
```

**RoPE positions:** `[p, p+1, p+1, p+2]` (siblings share same depth).

**KV cache write:** Append-only at physical positions [p+1, p+2, p+3, p+4].
Siblings (d1a, d1b) get different physical positions but same RoPE encoding.
After acceptance, stale entries are masked out (same mechanism as linear chain
per LITERT_RUNTIME_ANALYSIS.md §B1.3).

**Key advantage:** The MTP drafter already outputs top-k(8) indices. Branching
at depth 1 is free — just read index [0] and [1] from the same drafter call.

**Effort:** 1-2 days. Build after linear MTP is working.

---

## 3. ANEMLL Techniques

**Source:** github.com/Anemll/Anemll

### 3a. RMSNorm-as-LayerNorm Trick

ANEMLL repurposes ANE's hardware-optimized LayerNorm for RMSNorm:
```python
doubled = torch.cat([x, -x], dim=-1)  # mean becomes exactly 0
normed = F.layer_norm(doubled, (2 * hidden_size,))
normed = normed[..., :hidden_size]
return normed * self.weight
```

By concatenating `[x, -x]`, the mean is forced to zero, making LayerNorm
equivalent to RMSNorm. This leverages ANE's optimized LayerNorm hardware path
instead of manual `pow → mean → rsqrt → mul`.

**Our current approach:** `ANERMSNorm` uses `cat + LayerNorm + slice` — this
is the same trick! Verify our implementation matches this pattern exactly.

### 3b. Ping-Pong and Ring Buffers

ANE executes asynchronously. Without buffer synchronization, the ANE can still
be reading from a buffer when the next prediction writes to it.

**Chunked models:** 2-deep ping-pong alternating IOSurface buffers between
consecutive FFN chunks.

**Monolithic models:** 16-deep ring buffer:
```swift
let bufferSlot = tokenCounter % 16
outputBackings[bufferSlot] = ...
```

**Our risk:** If `ChunkedEngine` reuses the same `MLMultiArray` for hidden states
between chunk1 and chunk2, the ANE may have a race condition. Adding ping-pong
alternation is cheap insurance.

### 3c. LM Head Vocabulary Splitting

Split the 262144-vocab LM head into 8-16 separate Conv2d heads, each projecting
to a ~16K-32K subset. Combined with in-model argmax, this reduces the output
from 262144 FP16 values (512 KB) to 8-16 (index, value) pairs (~64 bytes).

**Our current approach:** Already using in-model top-k(8) in the MTP drafter.
Apply the same to the main model's LM head in chunk4.

### 3d. Multi-Function Model Weight Deduplication

When combining infer + prefill functions into one `.mlpackage` via
`ct.utils.MultiFunctionDescriptor`, identical quantized weights are
deduplicated. Saves ~50% on-disk for dual-function models.

**Also confirmed by ExecuTorch:** Their `preprocess_multimethod` uses the
same `ct.utils.save_multifunction` with positional weight sharing.

### 3e. Chunk Size Limits

ANEMLL's empirical limit: **950 MB per chunk** (with 10% safety overhead).
Chunks are balanced by layer count, not weight size. For Gemma 4 at INT4,
each layer is ~21 MB, so 950 MB / 21 = ~45 layers per chunk (well above
our 35 total). The 15-layer compile-hang threshold is more restrictive than
the 950 MB size limit.

---

## 4. ExecuTorch CoreML Backend Techniques

**Source:** github.com/pytorch/executorch

### 4a. Output Buffer Pooling (NSCache)

ExecuTorch recycles MLMultiArray backing storage via NSCache:
```objc
NSMutableData *backing = [cache objectForKey:descriptor];
if (!backing) {
    backing = [[NSMutableData alloc] initWithLength:size];
}
// ... use for prediction ...
// On dealloc, return to cache
```

Combined with `predictionOptions.outputBackings` (iOS 16+) for zero-copy
output writes. If the initial attempt with outputBackings fails, falls back
to standard allocation with `ignoreOutputBackings = YES`.

**Actionable:** Implement similar pooling in `ChunkedEngine`. Each decode
step currently allocates/deallocates output MLMultiArrays. Pooling eliminates
this overhead.

### 4b. SDPA Custom Op Preservation

ExecuTorch registers `coreml::sdpa` to prevent SDPA decomposition during export,
ensuring CoreML's iOS 18 fused `iOS18.sdpa` op is used.

**Relevance for us:** Our `SPEED_8K.md` says "SDPA fusion incompatible with
Gemma4 QK-norm scale=1.0." This may have been resolved in iOS 18's fused SDPA
implementation. Worth retesting with `ct.target.iOS18` or `iOS26`.

### 4c. `repeat_interleave` is Slow on CoreML

ExecuTorch explicitly blocklists `repeat_interleave` for CoreML delegation due
to poor performance. This is the op typically used for GQA head expansion.

**Confirms our approach:** We already use broadcast matmul instead of
repeat_interleave for GQA (PRIORITY_ROADMAP.md item #4).

### 4d. Pre-Compiled Model Shipping

ExecuTorch uses async model compilation with a 5-minute timeout. No magic
trick to avoid ANE compiler hangs — just timeout and fallback.

**Recommendation:** Ship pre-compiled `.mlmodelc` bundles to avoid on-device
compilation entirely. Use `MLModel(contentsOf:configuration:)` with the
compiled URL.

### 4e. Default CPU for Small LLMs

ExecuTorch defaults to `CPU_ONLY` for LLMs with CoreML, finding CPU 8x
faster than GPU for small models (stories110M on iPhone 15 Pro).

**Not applicable to us:** Gemma 4 E2B is much larger; ANE is the right target.
But worth benchmarking `CPU_ONLY` as a sanity check.

---

## 5. Apple AFM Additional Techniques

### 5a. ReDrafter Architecture (arXiv 2403.09919)

Apple's speculative decoding uses a **single-layer simple RNN** (not LSTM/GRU):
```
s_t = f(U * s_{t-1} + W * e_t + b)
g_t = [s_t, h]  // h = target LLM's last hidden state
logits = MLP_with_skip(g_t)
```

- Beam width = 1 on mobile (sequential RNN, no beam search)
- K = 5 draft tokens
- 2.3x speedup on Apple Silicon (MLX/Metal), 2.8x on H100

**Comparison to Google's MTP drafter:**

| | Google MTP | Apple ReDrafter |
|---|---|---|
| Architecture | 4-layer transformer | 1-layer RNN + MLP |
| Size | 44 MB | ~5-10 MB (estimated) |
| Draft speed | Slower (full attention) | Faster (RNN, no attention) |
| Quality | Reads target KV directly | Uses hidden state only |
| Training | Unknown (Google internal) | KD from frozen target |

**Actionable:** If Google's MTP drafter shows poor acceptance on our pipeline,
ReDrafter is a lighter backup. The RNN is trivially ANE-compatible (Linear + GELU).

### 5b. QAT Recipe Details

Apple's 2-bit QAT (for reference, not immediate action):
- **Set:** {-1.5, -0.5, 0.5, 1.5} balanced quantile
- **Init:** Newton-Raphson for clipping scalar `c`
- **Optimizer:** AdamW, weight_decay=0 (encourages full range usage)
- **Gradient scaling:** 1/sqrt(neuron_count) per layer
- **EMA** of weights for smoothing
- **Post-QAT:** rank-32 LoRA for quality recovery (~50 MB additional)
- **Training cost:** ~1-5% of original pre-training tokens (40-200B for 4T model)

### 5c. Constrained Decode + Speculative Integration

Apple's OS daemon runs constrained decoding AND speculative decoding together.
Draft tokens that violate format constraints (JSON, tool calls) are rejected
alongside distribution mismatches.

**Actionable:** When implementing MTP verify, add a constraint mask check in the
acceptance loop. CPU-side token validity check, negligible latency.

### 5d. MoE Self-Distillation for QAT Teacher

Apple's 90% cost reduction: upcycle the 3B student into a 64-expert MoE using
only 1T tokens, then distill back with QAT objectives.

**Actionable if pursuing QAT:** No need for a separate larger teacher model.
Bootstrap from Gemma 4 E2B itself.

---

## Consolidated Action Priority

### Immediate (this week, no model changes)

| # | Action | Source | Gain | Effort |
|---|---|---|---|---|
| 1 | **Prefill bypass** (skip chunk3+4 for prompt tokens) | AFM/SwiftKV | TTFT -47% | 0.5 day |
| 2 | **Output buffer pooling** (NSCache pattern) | ExecuTorch | -1-2ms/step | 0.5 day |
| 3 | **Ping-pong buffer audit** | ANEMLL | correctness | 0.5 day |

### Near-term (after linear MTP works)

| # | Action | Source | Gain | Effort |
|---|---|---|---|---|
| 4 | **Y-tree topology** for speculative verify | Sequoia | +15% tok/s | 1-2 days |
| 5 | **Traversal Verification** acceptance logic | arXiv 2505.12398 | +10-20% | 0.5 day |
| 6 | **Multi-function weight dedup** (infer+prefill) | ANEMLL/ExecuTorch | -50% disk | 1 day |

### Research-grade (requires training)

| # | Action | Source | Gain | Effort |
|---|---|---|---|---|
| 7 | **ReDrafter RNN** as backup drafter | AFM/ReDrafter | 2.3x (Apple's number) | 3-4 days |
| 8 | **2-bit QAT** with Apple's recipe | AFM | -50% bandwidth | 1-2 weeks GPU |
| 9 | **LoRA quality recovery** post-QAT | AFM | +1-3 MMLU | 1 day after QAT |

### Confirmed not needed

- RMSNorm-as-LayerNorm: **already implemented** as `ANERMSNorm`
- GQA broadcast (no repeat_interleave): **already implemented**
- In-model top-k: **already implemented** in MTP drafter
- Vocab splitting for LM head: worth testing but lower priority than prefill bypass
