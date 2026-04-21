# ANEMLL — source-verified tricks applicable to our CoreML+ANE stack

**Date:** 2026-04-22
**Source:** `/Users/majimadaisuke/Downloads/workspace/repo-review/Anemll` (deep-read)

**Context:** ANEMLL is a production-grade open-source ANE LLM runtime
supporting Llama / Qwen 2/2.5/3 / Gemma 3 / DeepSeek. Prior
`docs/ANE_OPTIMIZATION_SURVEY.md` cited them for 3 tricks. Source read
finds several more, and also refines earlier claims.

---

## 0. What ANEMLL does NOT have

- No speculative decoding / MTP / EAGLE / n-gram drafter.
- No W4A8 quantization. **LUT-only, 4/6/8-bit configurable** (`anemll/ane_converter/gemma3_converter.py:418-516`).
- No custom MIL-level fusion passes. Relies on coremltools default ANE-target optimization.
- No Gemma 4 E2B support. Gemma 3 only (`anemll/models/gemma3_model.py`, hidden=640).
- No in-process re-compile; precompiled `.mlmodelc` is the unit.

---

## 1. Confirmed tricks (some already in our stack)

### 1.1 RMSNorm-as-LayerNorm via `concat([x, -x])`

**File:** `anemll/models/gemma3_model.py:264, 308`

```python
doubled = torch.cat([x, -x], dim=-1)  # double + mirror → zero mean
normed = F.layer_norm(doubled, normalized_shape=(2*hidden_size,), weight=None, bias=None, eps=eps)
normed = normed[..., :hidden_size]    # drop mirror half
return normed * (1.0 + self.weight)   # Gemma gain
```

Happens in **Python pre-conversion**. CoreML graph gets LayerNorm ops,
which ANE has a first-party optimized kernel for. Status: already noted
in our `ANE_OPTIMIZATION_SURVEY.md`.

### 1.2 Ring / ping-pong KV backing buffers

**File:** `anemll-swift-cli/Sources/AnemllCore/InferenceManager.swift:55-60, 99-100, 1417-1449`

```swift
private var monolithicOutputBackingsRing: [[String: MLMultiArray]] = []
private let monolithicRingBufferDepth = 16
private var hiddenStatesBackings_ffnPingPong: [[String: MLMultiArray]] = []  // depth 2
```

- **Monolithic model:** 16-deep ring buffer.
- **Chunked model:** 2-deep ping-pong.
- Serial `DispatchQueue` (predictionQueue) gates all ANE predictions.
- **Reason:** ANE may read/write async; overlapping buffers without this causes silent corruption.

**Status:** we don't have this pattern confirmed in our stack. Likely a
concrete +stability / potential +throughput win to adopt on multi-chunk
decode.

### 1.3 Output buffer pooling — **modulo-indexed, NOT NSCache**

**File:** `anemll-swift-cli/Sources/AnemllCore/InferenceManager.swift:1738-1759`

```swift
let pool = hiddenStatesBackings_ffnPingPong
let chosenBackings = pool[index % pool.count]
```

Fixed array, pre-allocated, indexed by `index % pool.count`. No
NSCache. Simpler than ExecuTorch's NSCache-based scheme referenced in
our existing survey — this is the minimal working pattern.

**Correction:** `docs/ANE_OPTIMIZATION_SURVEY.md` earlier implied NSCache-based pooling across repos. ANEMLL does NOT use NSCache. Keep this file as the source of truth for ANEMLL's actual pattern.

### 1.4 Balanced-by-param layer chunking

**File:** `anemll/utils/calc_chunk_split.py:245-248`

Chunk boundaries placed at layer edges (no mid-layer splits). Splits
computed to minimize total-parameter-count imbalance across chunks.

**Implication for our 4-chunk Gemma 4 E2B:** We already chunk by layer.
Check if our boundaries are param-balanced or latency-balanced; if
latency-balanced, consider a param-balanced alternative for comparison.

### 1.5 Four-function chunk models for sliding window

**File:** `anemll-swift-cli/Sources/AnemllCore/FFNChunk.swift:6-34`

Each chunk ships four compiled graphs:
- `prefillModel` (batch>1 prefill)
- `inferModel` (token-by-token decode)
- `prefillRotate` (for sliding-window position crossover)
- `inferRotate` (ditto)

For Gemma 3's sliding window, runtime selects rotate variant when
current position exceeds window. Graceful fallback if rotate is absent.

**Implication for Gemma 4 E2B (sliding-512 alternating):** this pattern
is directly applicable. Our current chunks may need a rotate variant
pair to handle position > 512 correctly without re-compile.

### 1.6 IOSurface-backed CVPixelBuffer for MLMultiArray

**File:** `anemll-swift-cli/Sources/AnemllCore/InferenceManager.swift:330-335, 903-915`

MLMultiArray storage is wrapped around `CVPixelBuffer` with
IOSurface backing — not plain-allocation MLMultiArray. Pre-allocated
`MLPredictionOptions` with `outputBackings` dict points at these.

**Why:** better ANE synchronization + alignment guarantees. Avoids
per-prediction allocation overhead.

**Implication:** direct adoption in our Swift runtime is low-risk and
likely worth doing even without other changes.

### 1.7 Per-component LUT-bit overrides

**File:** `anemll/ane_converter/gemma3_converter.py:418-516`

Converter supports `lut_embeddings_bits`, `lut_lmhead_bits`, etc. —
different quant depths per component. Embedding and lm_head often get
higher-bit LUT to avoid quality loss in vocab tail.

**Implication:** we use W4A8 uniformly. Their observation that small
tensors (embed, head) benefit from looser quant is consistent with our
finding that lm_head dominates chunk4 time. Worth a quality/latency
sweep with per-component overrides on our stack.

---

## 2. Attention handling — ANEMLL contradicts maderix

**File:** `anemll/models/gemma3_model.py:416-712`

ANEMLL builds an **explicit `fullCausalMask` MLMultiArray** and passes
it as an attention input. **Attention runs on ANE** with the mask.
Causal masking is NOT decomposed to CPU.

**This contradicts maderix's "hardware ignores attn_mask"** claim
(maderix decomposes instead). Two plausible readings:
- ANEMLL uses CoreML-converted attention; maderix writes raw MIL. The CoreML converter may synthesize a mask-supporting attention kernel composition that maderix's hand-written MIL doesn't.
- The constraint is SDPA-native-mask-arg specific; passing the mask as a separate tensor input avoids it.

**Implication:** "attention on ANE" is NOT physically blocked on our
CoreML-converted stack. We should not cite maderix's claim as a
decoder-side architectural limit.

---

## 3. Prefill vs decode — completely separate graphs

**File:** `anemll-swift-cli/Sources/AnemllCore/FFNChunk.swift:6-34`

Each `FFNChunk` carries `prefillModel` and `inferModel` as two
compiled `.mlmodelc` artifacts. Prefill uses batch ≤64, stateful KV
writes; decode uses batch=1 stateful KV reads.

No async prefill; prefill is synchronous before decode loop starts.

**Implication for us:** we already split prefill/decode at chunk level.
Verify we aren't unnecessarily sharing graph structure that could be
specialized per mode.

---

## 4. Dispatch model — serial, no overlap

**File:** `anemll-swift-cli/Sources/AnemllCore/InferenceManager.swift:2735-2857`

Single serial queue, one ANE prediction at a time. No overlapping
prefill+decode, no multi-model concurrency, no micro-batching.

**Read:** conservative. Our stack could experiment with pipelined
prefill + speculative decode, but ANEMLL declining to suggests they
measured and it didn't help or caused races.

---

## 5. Known coremltools bugs they work around

**File:** `anemll/ane_converter/gemma3_converter.py:1532, 1754`

- LUT quantization **fails with multiple workers** in coremltools — forced to single-worker. TODO in their source. Check if this affects our conversion pipeline.
- Rotary embedding length hack: `anemll/models/llama_model.py:178` — "ensure rotary embeddings long enough for context length" comment.

---

## 6. Top 3 adopt-today items for our stack

1. **IOSurface-backed CVPixelBuffer** for MLMultiArray I/O — low-risk code change, proven on ANEMLL.
2. **Ring buffer (N=16) or ping-pong (N=2)** for output buffers — measurable stability; uncertain throughput impact, worth A/B.
3. **Sliding-window rotate model variants** — if we ship sliding-512 correctness for Gemma 4 E2B's ISWA layers, we need the ANEMLL pattern or a functional equivalent.

## Top 2 "measure-first" items

1. **Per-component LUT-bit overrides** — quality vs tok/s sweep on lm_head and embeddings.
2. **Param-balanced chunk boundaries** — compare against our current layout.

---

## 7. Citations (all in `/Users/majimadaisuke/Downloads/workspace/repo-review/Anemll/`)

- `anemll/utils/calc_chunk_split.py:245-248`
- `anemll/models/gemma3_model.py:118-125, 264, 308, 416-712`
- `anemll/models/llama_model.py:178`
- `anemll/ane_converter/gemma3_converter.py:418-516, 1532, 1754`
- `anemll-swift-cli/Sources/AnemllCore/InferenceManager.swift:55-60, 99-100, 330-335, 903-915, 1417-1449, 1738-1759, 2735-2857`
- `anemll-swift-cli/Sources/AnemllCore/FFNChunk.swift:6-34`
