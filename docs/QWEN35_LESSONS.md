# Qwen3.5 / Hybrid SSM on CoreML — Lessons Learned

Consolidated knowledge base from the 2026-04-20 → 2026-04-22 Qwen3.5-0.8B ship (v1.0.0 → v1.0.3). Organized by topic so future work on Qwen3.5 variants — or any hybrid SSM/attention LLM targeting Apple Neural Engine — can skip the dead ends.

This doc is the **source of truth** for cross-cutting learnings. Session-specific handoff (2B continuation) lives in `QWEN35_2B_CHUNKED_HANDOFF.md`.

---

## 1 · Architecture-level surprises

### 1.1 Qwen3.5 is the first hybrid SSM/attention LLM on CoreML
- 18 `linear_attention` (Gated DeltaNet) + 6 `full_attention` (GQA) layers, interleaved `[L,L,L,F]×6`
- No prior CoreML port of Mamba/DeltaNet/Gated-DeltaNet existed — this repo's `conversion/test_qwen3_5_*_trace.py` is the reference.
- `mamba_ssm_dtype = float32` in config hints that the SSM state accumulator is drift-sensitive. Confirmed empirically: fp16 ANE exhibits argmax-fragility, fp32 CPU is bit-exact.

### 1.2 AutoModelForCausalLM unwraps VL checkpoints
Qwen3.5-2B on HF is tagged as `Qwen3_5ForConditionalGeneration` (vision-language). But `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B")` returns a plain `Qwen3_5ForCausalLM` with the same `.model.layers / .model.embed_tokens / .lm_head` layout as 0.8B — the vision tower is silently dropped. No manual stripping needed.

### 1.3 `bfloat16` on disk must be forced to fp32 for tracing
All Qwen3.5 checkpoints ship weights as bf16. Load with `torch_dtype=torch.float32, low_cpu_mem_usage=True` before tracing — otherwise the SSM's cumulative-sum / Neumann iteration paths accumulate noise that CoreML then bakes into the model.

### 1.4 `tie_word_embeddings=True` for 2B (not 0.8B)
2B's `lm_head.weight` is tied with `model.embed_tokens.weight`. Converter must read `cfg.tie_word_embeddings` and pick the right source tensor.

---

## 2 · Precision reality on ANE

### 2.1 Argmax fragility is a diagnostic, not a model bug
On fp16 ANE over 24 layers, hidden state stays at cos ≥ 0.9998 vs fp32 — essentially bit-identical. But the 248K-vocab logit projection amplifies tiny drift into argmax flips on close-tie tokens. Strict greedy top-1 matches fp32 only ~60%, but oracle top-1 is in the model's top-3 for **100% of tested positions**. See `QWEN35_IPHONE_BENCH.md` for the measurement harness.

**Rule:** top-1 alone is misleading for fp16 ANE LLMs. Always report top-3 + top-5 containment + mean top-5 overlap alongside.

### 2.2 Sampling-mode generation is indistinguishable from fp32
`temperature > 0 + top-K` samples from a distribution that overlaps fp32's by ~88% in top-5 and 100% in top-3. Human-perceived quality difference is zero. Greedy decoding is the only mode where fp16 drift shows.

### 2.3 Root cause of SSM drift: per-layer, not temporal
Per-token state recurrence does NOT compound drift noticeably (measured 30 tokens stable at cos 0.9998). The drift is per-layer fp16 accumulation in the SSM's Gated-DeltaNet recurrent path. The `full_attention` layers are effectively clean (cos 0.999999 per-layer). See `memory/qwen35_ane_decode_precision_ceiling.md` for the layer-type attribution experiments.

### 2.4 ANE fp16 is HARDWARE, not a graph bug
Tried and rejected: Conv2d 1x1 replacement for Linear (`conversion/test_qwen3_5_decode_conv2d.py`), RMSNorm swaps, chunk split with fp32 boundary (`conversion/build_qwen35_decode_chunks.py`). None of these moved the precision needle — ANE's accumulator is fp16-locked and no graph-level rewrite escapes it. Accept top-3 metric and ship.

---

## 3 · iPhone ANE deployment budget

### 3.1 Silent GPU fallback is the #1 deployment killer
`MLComputeUnits.cpuAndNeuralEngine` is a *preference*, not a requirement. When any single mlprogram exceeds the ANE compile budget, Core ML silently routes the whole graph to GPU — which loads weights into Metal heap and blows past iOS's ~5 GB jetsam cap for app memory. Symptoms: jetsam kill during model load, or compute plan audit reports ANE=0%.

### 3.2 Single-mlpackage ANE budget: ~1.4 GB on iPhone 17 Pro
- 0.8B fp16 decode (1.4 GB) — compiles on ANE ✓
- 0.8B stateful prefill (1.4 GB monolithic) — fails ANE compile, falls to GPU (OK, runs, but uses Metal)
- 2B INT8 (1.8 GB) — fails ANE, falls to GPU, **jetsam kill** on device
- 2B INT4 (900 MB) — compiles on ANE ✓

Above ~1.4 GB per mlpackage, you need to **chunk** (Gemma 4 E4B pattern) or face silent GPU fallback.

### 3.3 Palettization saves disk, NOT memory
Apple's k-means palettization shrinks download/storage 2× (INT8) or 4× (INT4) but at load time ANE dequantizes weights back to fp16 in its memory region. Verified on 0.8B: fp16 bundle 1.5 GB → INT8 bundle 754 MB, but runtime app memory is ~1.6 GB in BOTH cases. Bandwidth during inference IS reduced (15-20% speedup on ANE because ANE is bandwidth-bound for small models).

**Rule:** palettize for download size + modest speed improvement. Don't rely on it to fit bigger models into device memory.

### 3.4 Chunking IS the only way to fit >1.4 GB models
Gemma 4 E4B ships at 5.5 GB total as 4 chunks of ~1.4 GB each. Each chunk independently compiles on ANE. Per-chunk plan cache is ~300 MB, so 4 chunks = ~1.2 GB cache + 5.5 GB weights = total stays within jetsam margin.

For Qwen3.5-2B shipping: 2-chunk INT8 (layers 0-11 + 12-23, each ~950 MB) is the path.

### 3.5 INT4 is unsafe for small SSM models
0.8B INT4: top-3 parity drops 100% → 80%. Recipes can paper over this with sampling.
2B INT4: **hard-fails quality** — factual errors mid-stream, JP→DE language codeswitching. NOT shippable. Never use INT4 for Qwen3.5 in any size we've tested; use INT8 or chunked INT8.

### 3.6 `MLComputePlan` op classification (iOS 18+)
`plan.deviceUsage(for: op)?.preferred` returns `MLComputeDevice` which is an **enum**, not a protocol. Use switch-case:

```swift
switch plan.deviceUsage(for: op)?.preferred {
case .cpu:          cpu += 1
case .gpu:          gpu += 1
case .neuralEngine: ane += 1
default:            other += 1
}
```

`is MLNeuralEngineComputeDevice` NEVER matches (learned the hard way — first diag printed ANE=0 for every op).

---

## 4 · Swift marshal path wins (the +40% on 0.8B)

Qwen35Generator's CPU-side marshaling was the bottleneck, not compute. Measured path on iPhone 17 Pro, 0.8B ANE decode:
- Baseline (naive MLModel.prediction per step): 13 tok/s
- After all fixes: 28 tok/s (**+115%**)

### 4.1 Custom `MLFeatureProvider` for zero-copy state handoff
Instead of extracting 48 state MLMultiArrays via `out.featureValue(for: "new_state_X_Y")?.multiArrayValue` + re-wrapping as 48 `MLFeatureValue`s for the next call, `Qwen35DecodeFeatures` (private class) **delegates** `featureValue(for:)` to the previous call's `MLFeatureProvider`. Swift-side sees the same state tensors without any copy or wrap.

```swift
private final class Qwen35DecodeFeatures: NSObject, MLFeatureProvider {
    private let prevOut: MLFeatureProvider?
    private let stateRename: [String: String]  // "state_0_a" -> "new_state_0_a"
    // ...
    func featureValue(for name: String) -> MLFeatureValue? {
        if let prev = prevOut, let out = stateRename[name] {
            return prev.featureValue(for: out)
        }
        // ...
    }
}
```

Savings: 48 ObjC-bridge wraps × ~100 μs each = ~5 ms per step.

### 4.2 Reusable `MLMultiArray` for per-step scalar inputs
`input_token`, `position`, `cos`, `sin` are tiny tensors that are rewritten every step. Allocate once at `Qwen35Generator.init()`, keep a matching pre-wrapped `MLFeatureValue`, rewrite the data pointer each step. Saves ~4 allocs/step.

### 4.3 `fastArgmax` via native Float16 NEON compare
NO Float32 conversion buffer. Single-pass fp16 compare directly off the raw `MLMultiArray.dataPointer` (bound to `Float16`). Apple Silicon NEON instructions do fp16 compare natively. Total ~1 ms for 248K vocab on iPhone.

### 4.4 `fastArgmaxAvoidingRecent` for loop-breaking
If greedy converges on a repeat (INT8 quantization + SSM state recurrence can produce attractor states), single-pass argmax that tracks BOTH the global max AND the best-non-recent token. Return the alternative if the winner is in the recent-64-token set. Same cost as `fastArgmax` plus a Set lookup per iteration. No 1 MB buffer needed.

Default is `rep_penalty = 1.0` (off) — Mac bench confirmed full EOS set is enough to prevent loops without rep_penalty.

### 4.5 Batched status updates
Per-decode-step `status = "Decoding \(step)/\(max)"` causes SwiftUI to invalidate the view per token. Cost: ~10-15 ms/step main-thread work. Update every 8 steps (`if (step & 0x7) == 0`) — shaves 5-10 ms/step.

### 4.6 vDSP path for temperature > 0 sampling
`vDSP_vsmul` for temperature scaling, `Array.sort(by: >)` for top-K threshold detection, then linear scan to collect indices ≥ threshold. Avoids O(V×K) insert-sort.

### 4.7 Wins are Qwen3.5-specific, NOT Gemma
Tried transplanting (4.1) + (4.2) to Gemma's ChunkedEngine (PR #119). Measured impact: < 0.5 tok/s because Gemma's CPU-side marshal is only ~3-4% of per-step time (95% is ANE wait). Closed PR. Lesson: **identify the real bottleneck before porting**. Qwen and Gemma have different ratios.

---

## 5 · Shipping correctness

### 5.1 Qwen EOS token set
Stop on ALL of:
- 248044 `<|endoftext|>`
- 248045 `<|im_start|>` (next turn starts before current `<|im_end|>` fires)
- 248046 `<|im_end|>`
- `tok.eosTokenId` (depends on tokenizer config version)

Missing any → visible-stream leak: the model emits the special token as literal text and continues generating a fake "Human:" follow-up turn.

### 5.2 Chat template is mandatory for instruct models
Qwen3.5-0.8B is instruct-tuned. Raw `tok.encode(prompt)` → the model completes in "base" mode and loops on short inputs (e.g., "こんにちは" → "おはようございます" loop 10×). Wrapping in chat template via `tok.applyChatTemplate(messages:)` fires `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` and the model responds coherently.

### 5.3 Filter system-role messages before chat template
The app's `ChatView` appends UI-status system messages ("Loading…", "Model loaded!"). These MUST NOT reach the chat template — otherwise Qwen treats them as real system prompts and loops on the next user turn. `LLMRunner.generateQwen35` filters `role == .system` before building the template message array.

### 5.4 Multi-byte UTF-8 streaming
Qwen BPE splits emoji (😊) and some CJK glyphs across multiple tokens. Per-token `tok.decode(tokens: [id])` returns broken UTF-8 that renders as U+FFFD mojibake. Use accumulate-decode-emit-diff:

```swift
var accumIds: [Int] = []; var emittedText = ""
onToken: { id in
    accumIds.append(Int(id))
    let current = tok.decode(tokens: accumIds)
    if current.count > emittedText.count, current.hasPrefix(emittedText) {
        continuation.yield(String(current.dropFirst(emittedText.count)))
        emittedText = current
    }
}
```

O(N²) in decoded bytes but chat token rates make it negligible.

### 5.5 Stride-safe logit reading
ANE output `MLMultiArray` sometimes has `strides = [0, 0, 0]` instead of the contiguous `[V, V, 1]`. Trusting reported strides blindly causes all vocab positions to resolve to `dataPointer[0]` → model emits the same token forever (tokenizer's ID 0 = "!" on Qwen → visible output "!!!!!").

Guard: if `strides.last < 1` or `strides[-2] < vocab`, fall back to contiguous `(vStride=1, pStride=vocab)`.

### 5.6 Explicit `memset` on MLMultiArray state init
`MLMultiArray(shape:dataType:)` is NOT guaranteed to zero-init on iOS (contrary to a common misconception). Using uninitialized memory as initial SSM state produces garbage logits. Always `memset(buf.dataPointer, 0, buf.count * elementSize)` explicitly.

---

## 6 · Compute units decision tree

| selected | prefill | decode | use when |
| --- | --- | --- | --- |
| `cpuAndNeuralEngine` (default) | recurrent via decode on ANE | ANE | shipping path, 0 GB Metal heap |
| `cpuAndGPU` | monolith on GPU (267 tok/s) | GPU (27 tok/s) | bit-exact vs fp32 needed (Metal heap ~3 GB) |
| `cpuOnly` | broken on iOS 26.1 prefill | CPU ~20 tok/s | decode only, but iOS CPU runtime bug makes prefill unusable |
| `.all` | driver chooses | driver chooses | Core ML may route partially — unpredictable |

The `.cpuOnly` prefill bug is iOS 26.1-specific — same compiled mlmodelc gives cos=0.30 (garbage) on iPhone CPU but cos=1.0 on Mac CPU. Not fixable graph-side; workaround is GPU or ANE prefill.

---

## 7 · Developer workflow (deploy / iterate)

### 7.1 Documents-first model loader
Xcode app-bundle caching makes same-named mlpackages sticky across reinstalls (confirmed on iOS 26.1). To hot-swap variants without uninstall, push compiled `.mlmodelc` via `xcrun devicectl device copy to` into `Documents/`. The Generator's `resolveModelURL` prefers Documents over bundle.

```bash
xcrun coremlcompiler compile NAME.mlpackage /tmp/out
xcrun devicectl device copy to \
  --device $DEVICE_UUID \
  --domain-type appDataContainer \
  --domain-identifier com.example.CoreMLLLMChat \
  --source /tmp/out/NAME.mlmodelc \
  --destination Documents/NAME.mlmodelc
```

### 7.2 E5 bundle stale cache error is informational
```
ANE model load has failed for on-device compiled macho.
Must re-compile the E5 bundle.
```
Happens on first ANE load after rebuild, after reinstall, or when iOS purges the ANE cache. Core ML automatically recompiles; takes 4+ minutes for a 1.4 GB model. App appears to hang — it's actually compiling. Subsequent loads are cached and fast.

### 7.3 `coremltools` environment
Use `lama-cml` env (coremltools 8.3, Python 3.10). coremltools 9.0's BlobWriter is broken on Mac.

### 7.4 Oracle-based parity testing
`conversion/qwen3_5_reference_oracle.py` generates HF fp32 reference logits for a fixed prompt set. iOS-side `Qwen35DecodeBenchmark` compares CoreML output top-K against the oracle. Export the JSON via `conversion/export_oracle_for_ios.py`.

---

## 8 · Shipping snapshot (as of v1.0.3)

**iPhone 17 Pro, Qwen3.5-0.8B INT8 ANE:**
- Decode: ~28 tok/s
- Prefill (recurrent via decode): proportional
- Metal heap sustained: 0 GB
- Total app memory: ~1.6-2 GB (weights mmap + ANE plan cache + app baseline)
- Bundle: 754 MB download, ~1.9 GB dequantized in ANE memory
- Chat template: auto-applied
- EOS: full stop set 248044/045/046 + tokenizer eos
- First load: ~4 min ANE compile (cached after)

**Beats Google LiteRT-LM 56.5 tok/s baseline on prefill 3.0×.** Decode is 39% of LiteRT, structural (SSM recurrence + ANE fp16 ceiling; LiteRT runs on GPU).

---

## 9 · MLKV (KV-only MLState) — the +54% / +30% unlock (added 2026-04-28)

After v1.0.3 / v1.1.0 shipped on stateless I/O, Mac M4 measured an
extra 1.5×-1.7× by moving KV cache into Core ML's `MLState` and writing
via `ios18.slice_update`. Final Mac M4 numbers:

| Model | stateless ANE | MLKV ANE | uplift |
|-------|---------------|----------|--------|
| 0.8B  | 33.1 tok/s    | **51.0** | **+54%** |
| 2B    | 24.6 tok/s    | **32.2** | **+30%** |

Output ✓ on English fact (Paris/Berlin/Rome) + Japanese recipe.

### 9.1 The "single ct.StateType per chunk" rule

`ct.StateType` works on iOS 18 ANE if **at most a small fixed number
of StateTypes per chunk**. VL Phase 1 ships 1 (unified `kv_cache_0`);
Gemma 4 ships 2 (`kv_cache_sliding` + `kv_cache_full`); both ANE-stable.

The first attempt for Qwen3.5 hybrid
(`build_qwen35_decode_chunks_mlstate.py`) used 3 StateTypes per chunk
(`kv_cache` + `conv_state` + `rec_state`). Mac compiles fine, audit
reports 100% ANE; predict-time runtime fails with
`ANEProgramProcessRequestDirect Error=(11)` and the CPU runtime
miscompiles slice_update on multi-state and emits garbage tokens.
GPU is the only working path on the multi-state model.

**Fix**: keep KV in MLState (the biggest state — 12 MB/step on 0.8B),
move SSM `conv_state` + `rec_state` back to conventional input/output
tensors. Single StateType per chunk → ANE happy.

The pattern lives in `conversion/qwen3_5_decode_layer_mlkv.py` /
`build_qwen35_decode_chunks_mlkv.py` / Swift
`Qwen35MLKVGenerator.swift`. Reusable for any hybrid model where
multiple state types coexist — pick the biggest and put it in MLState,
push the rest through input/output. See `docs/MLSTATE_PATTERN.md` for
the generic recipe.

### 9.2 In-graph TopK head

chunk_d's tail emits `next_token` (1, 1) int32 via `topk[k=1]` instead
of fp32 logits (1, 1, 248320). Per-step ANE→Swift transfer drops from
~1 MB to 4 byte. On Mac the Python predict() path is local memory so
no measurable change; on iPhone the transfer crosses the runtime
boundary, so this is a real win there.

Per `chunk4_argmax_topk_parity` memory, `topk[0]` (not `argmax`) is
the right primitive — argmax has ANE-vs-CPU divergence on palettized
heads.

### 9.3 What didn't help on Mac M4 ANE

- **2-chunk vs 4-chunk MLKV**: same 51 tok/s. Per-chunk dispatch
  overhead is sub-ms; consolidating doesn't move the needle.
- **3-chunk MLKV**: 49 tok/s (-3%). Per-chunk graph grows faster than
  dispatch reduction.
- **Native softmax (`USE_NATIVE_SOFTMAX=1`)**: 50.4 vs 51 — slightly
  slower. Decomposed `ane_softmax` (max/sub/exp/sum/div) wins.
- **Precomputed RoPE table**: same 50.4. The torch eval per step was
  not the bottleneck.

---

## 10 · Open research directions

- **2-chunk INT8 for 2B**: see `QWEN35_2B_CHUNKED_HANDOFF.md`. Enables 2B on iPhone without jetsam.
- **(Resolved 2026-04-28) MLState API**: ship via the MLKV path (§9.1).
- **Gemma spec decoding (EAGLE-3)**: already implemented in stack, disabled by default, would give Gemma 4 E2B +30-50% if acceptance rate ≥ 40%. Pending: PR #111's EAGLE-3 retrain recovery.
- **In-graph argmax output**: tested on 0.8B (`build_qwen35_decode_argmax.py`), measured 2 ms/step SLOWER on Mac. Argmax forces CPU placement, so the 500 KB logit transfer happens anyway. Not viable.
- **Larger Qwen3.5 variants (4B/9B)**: VL checkpoints; text backbone extraction works via AutoModelForCausalLM (§1.2). Same 24-layer pattern. Bigger than 2B will definitely need chunking. Acceptance criteria from §8 apply.

---

## 11 · Memory / commit trail

All session-specific discoveries were saved to auto-memory under `~/.claude/projects/.../memory/qwen35_*.md` during the 2026-04-20 → 2026-04-22 + 2026-04-28 work. This doc consolidates them into a single reference; memory files remain the source for timeline-ordered details.

Key commits on main (chronological):
- #112 ship to main ChatView
- #116 ANE default restore
- #117 README for v1.0.0
- #118 UTF-8 streaming fix
- #120 INT8 palettized default + EOS fix + rep_penalty + marshal
- v1.0.3 release
- v1.x (this branch) MLKV path: `fix/qwen35-mseq-reconcile`

Feature branch (open): `fix/qwen35-mseq-reconcile` — MAX_SEQ=2048 + Gemma 4 ANE recipe + MLKV (KV in MLState, SSM stateless), 0.8B 51 tok/s / 2B 32 tok/s on Mac M4 ANE. See `docs/SESSION_2026_04_28_QWEN35_2K_ANE.md`.
