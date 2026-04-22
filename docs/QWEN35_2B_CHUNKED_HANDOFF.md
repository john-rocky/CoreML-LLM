# Qwen3.5-2B Chunked Decode — Shipping Handoff

**Status (2026-04-22):** 2B monolithic INT8 fails on iPhone — jetsam kill during load because Core ML silently falls back to GPU (ANE single-mlprogram compile budget exceeded), and Metal heap + dequant-to-fp16 together blow past iOS's ~5 GB per-app memory cap. INT4 shrinks the bundle but introduces quality regression severe enough to reject (factual errors, language code-switching).

**This doc is the next session's starting point** to ship 2B on iPhone via 2-chunk INT8 (layers 0-11 + 12-23), following the same pattern Gemma 4 E4B already uses successfully at 5.5 GB total / ~1.4 GB per chunk.

---

## 1 · What already worked (0.8B baseline)

- `conversion/build_qwen35_decode_int4.py` does k-means palettization (nbits=4 or 8) on any existing fp16 decode mlpackage
- `conversion/test_qwen3_5_full_decode_trace.py` is the fp16 converter (reused by the 2B builder)
- `conversion/build_qwen35_2b_decode.py` is the 2B fp16 converter (thin wrapper — same architecture, just wider)
- On iPhone 17 Pro ANE, 0.8B INT8 ships at ~28 tok/s decode with 100% top-3 parity

## 2 · What failed for 2B monolithic (evidence)

| path | on-disk | in-memory | iPhone ANE compile | iPhone result |
| --- | --- | --- | --- | --- |
| fp16 | 3.77 GB | 3.77 GB | ANE=0% (GPU fallback) | not tested (obviously too big) |
| INT8 | 1.88 GB | ~3.8 GB dequantized | ANE=0% on iPhone (**measured via jetsam crash**) | JETSAM KILL |
| INT4 | 899 MB | ~1.9 GB dequantized | ANE=90.4% (Mac) | **quality broken** — wrong facts, JP↔DE codeswitch |

Mac numbers from 2026-04-22 sessions: 2B INT4 on Mac ANE = 40 tok/s but output for "What is the capital of France?" was `"Paris. It is the largest city in the country, the second-largest in Europe, and the most populous in the world. It is the only city in the world with a population of more than 1 million people…"` — all facts after the first sentence are wrong.

**Root cause of monolithic OOM:** Core ML treats `.cpuAndNeuralEngine` as a *preference*, not a requirement. When ANE's single-program budget (~1-1.5 GB in-use on iPhone) is exceeded, it silently routes to GPU. GPU loads the 1.88 GB INT8 weights + dequantization state + Metal intermediates into its unified-memory heap, pushing total app footprint past the ~5 GB jetsam ceiling.

**Proof by analogy:** Gemma 4 E4B ships at 5.5 GB total and loads fine because it's 4 chunks of ~1.4 GB each, each a separate mlpackage / MLModel, each individually within the ANE compile budget.

## 3 · Design: 2-chunk INT8 for Qwen3.5-2B

### 3.1 Split

- **chunk_a:** input_token → embed → layers 0-11 → hidden_out (fp16)
- **chunk_b:** hidden_in (fp16) → layers 12-23 → final_norm → lm_head → logits (fp32)

Each chunk has its own 24 state tensors (12 layers × 2 states each).

### 3.2 fp16 vs fp32 hidden boundary — IMPORTANT

The research-era `conversion/build_qwen35_decode_chunks.py` used **fp32 hidden handoff** between chunks to probe whether fp16 drift was the ANE precision issue (it wasn't — see `memory/qwen35_ane_decode_precision_ceiling.md`). For **shipping**, use **fp16 hidden boundary** — fp32 round-trip is wasteful and doesn't improve parity. The prior script can be reused but this specific change is required:

```python
# In DecodeChunkA's ct.TensorType output:
ct.TensorType(name="hidden", dtype=np.float16)  # was np.float32
# In DecodeChunkB's ct.TensorType input:
ct.TensorType(name="hidden_in", shape=(1, 1, cfg.hidden_size), dtype=np.float16)
```

### 3.3 Each chunk INT8-palettized

After converting both fp16 chunks, run `build_qwen35_decode_int4.py --nbits 8 --fp16-pkg <chunk_path>` twice. Expected per-chunk size:
- chunk_a fp16: ~1.9 GB → INT8: **~950 MB**
- chunk_b fp16: ~1.9 GB → INT8: **~950 MB**
- total on disk: ~1.9 GB (same as monolithic INT8)
- total in-memory: ~3.8 GB dequantized, **BUT** each chunk is ~950 MB compiled artifact → iPhone ANE budget OK per-chunk

Rule of thumb from the iPhone side: what matters isn't total model size, it's per-mlpackage compiled-ANEF size. 0.8B decode at 1.4 GB fp16 compiled fine → per-chunk at ~950 MB INT8 definitely fits.

### 3.4 State tensors — inherit the 0.8B shapes

`Qwen3_5TextConfig` for 2B has identical dims except hidden/intermediate:
- `num_hidden_layers = 24` (same)
- `num_key_value_heads = 2`, `head_dim = 256` → full-attn state: `(1, 2, max_seq, 256)` (same as 0.8B)
- `linear_num_value_heads = 16`, `linear_key_head_dim = 128` → linear state: `(1, 16, 128, 128)` for state_b, `(1, 6144, 4)` for state_a (same as 0.8B)

State tensor shapes are layer-type-determined, not hidden-size-determined. The existing Swift `makeZeroStates` hardcoded constants in `Qwen35Generator.swift` work unchanged for 2B.

### 3.5 Mac parity test required

Before committing to the split, re-run the 3-prompt long-gen test on Mac ANE with the 2-chunk INT8 setup to verify:
- factual accuracy on "What is the capital of France?" (should say Paris + correct attributes — NOT the INT4 breakage)
- Japanese coherence on "こんにちは" + "美味しい餃子のレシピを教えて"
- No mid-stream language codeswitch

Existing test script location: similar to `conversion/qwen35_mac_generator_sim.py` pattern. Write one that chains chunk_a → chunk_b.

## 4 · Swift-side changes

`Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen35Generator.swift` currently loads a single `MLModel decode`. To support chunks:

### 4.1 Load phase — `loadDecodeOnly()`

Add a `decodeIsChunked` flag + `decodeChunkA`, `decodeChunkB: MLModel?` fields. Preference order:

1. `qwen3_5_2b_decode_chunks/{chunk_a,chunk_b}.mlpackage` under the model folder → chunked path
2. `qwen3_5_2b_decode_int8_mseq128.mlpackage` → monolithic path (keep for Mac compatibility)
3. Fall through to 0.8B variants as today

### 4.2 Per-step decode — forward through both chunks

```swift
// Pseudocode — plug into the existing profiling fences
let outA = try await decodeChunkA.prediction(from: inA)   // input_token + cos + sin + pos + state_0..11_ab
let hidden = outA.featureValue(for: "hidden")!.multiArrayValue!
let inB = /* hidden + cos + sin + pos + state_12..23_ab */
let outB = try await decodeChunkB.prediction(from: inB)
let logits = outB.featureValue(for: "logits")!.multiArrayValue!
// state updates — 24 from chunk_a, 24 from chunk_b
```

**Reuse `Qwen35DecodeFeatures`** (custom MLFeatureProvider) for BOTH chunks to preserve the zero-copy state plumbing. Each chunk needs its own provider since their state-name sets differ (0-11 vs 12-23).

### 4.3 Compute plan audit

Audit both chunks on load. The existing `auditComputePlan` takes one URL; make it loop over both chunk URLs and print per-chunk ANE%.

### 4.4 `makeZeroStates` — split

Currently allocates all 48 state tensors in one dict. For chunked path, allocate two dicts (12 layers each) so chunk_a gets only its 24 states and chunk_b gets only its 24.

## 5 · ModelDownloader changes

`Sources/CoreMLLLM/ModelDownloader.swift`:

```swift
public static let qwen35_2b = ModelInfo(
    id: "qwen3.5-2b", name: "Qwen3.5 2B (ANE, chunked)", size: "1.9 GB",
    downloadURL: "https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreML/resolve/main",
    folderName: "qwen3.5-2b")

private func buildQwen35_2B_FileList() {
    // Ship BOTH chunks as the 2B entry. Each mlpackage has the 3 standard files.
    let pkgs = ["qwen3_5_2b_decode_chunks/chunk_a.mlpackage",
                "qwen3_5_2b_decode_chunks/chunk_b.mlpackage"]
    for pkg in pkgs {
        pendingFiles += [
            .init(remotePath: "\(pkg)/Manifest.json",
                  localPath: "\(pkg)/Manifest.json",
                  estimatedSize: 700),
            .init(remotePath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                  localPath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                  estimatedSize: 900_000),
            .init(remotePath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                  localPath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                  estimatedSize: 950_000_000),
        ]
    }
    pendingFiles.sort { $0.estimatedSize > $1.estimatedSize }
    // ...
}
```

`localModelURL` should recognize the chunked layout (presence of `chunk_a.mlpackage/Data/…/weights/weight.bin`).

## 6 · Implementation order

1. **Converter** — port `build_qwen35_decode_chunks.py` to `build_qwen35_2b_decode_chunks.py`, change fp32→fp16 boundary, add INT8 palettize hook after save.
2. **Mac parity** — 3-prompt long-gen bench with chunked INT8. Gate: factual answer on Paris + clean JP + no codeswitch.
3. **HF upload** — create directory layout `qwen3_5_2b_decode_chunks/chunk_a.mlpackage` + `chunk_b.mlpackage` under `mlboydaisuke/qwen3.5-2B-CoreML`. Include fp16 monolithic too as a research artifact.
4. **Swift refactor** — `Qwen35Generator` chunked load + per-step chain + `Qwen35DecodeFeatures` per chunk. **Preserve all marshal-wins from PR #120.**
5. **Device bench** — push to iPhone, verify no jetsam, measure tok/s, measure compute plan per-chunk.
6. **ModelDownloader** — wire menu entry.
7. **README / release** — v1.1.0 candidate.

## 7 · Pitfalls already hit today — avoid re-hitting

- **Compute plan enum classify**: `MLComputeDevice` is an iOS-18 enum. Type-match via `switch plan.deviceUsage(for: op)?.preferred { case .cpu / .gpu / .neuralEngine: }`. Do NOT use `is MLNeuralEngineComputeDevice` — it never matches.
- **Qwen EOS token set**: must include all of 248044 (`<|endoftext|>`), 248045 (`<|im_start|>`), 248046 (`<|im_end|>`) plus `tok.eosTokenId`. Missing any → visible-stream leak and fake-turn fabrication.
- **System-role messages**: ChatView appends UI-status system messages ("Loading…", "Model loaded!"). These MUST be filtered out before passing into Qwen's chat template — otherwise the instruct model treats them as real system prompts and loops.
- **Multi-byte UTF-8 streaming**: BPE splits emoji / CJK across multiple tokens. Per-token `tok.decode` yields broken UTF-8. Use the accumulate-decode-emit-diff pattern in `LLMRunner.generateQwen35`.
- **Palettize script filename quirk**: `build_qwen35_decode_int4.py` hardcodes the output mlpackage name as `qwen3_5_0_8b_*`. Rename the output after running for 2B.
- **`bfloat16` weights on disk**: 2B checkpoint stores weights as bf16. The converter must force fp32 during PyTorch trace; `torch_dtype=torch.float32, low_cpu_mem_usage=True` works.
- **`Qwen3_5ForConditionalGeneration` load**: 2B is a VL checkpoint on HF. But `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B")` returns a text-only `Qwen3_5ForCausalLM` with the same `.model.layers / .model.embed_tokens / .lm_head` layout as 0.8B — no manual stripping needed.
- **Rep-penalty fast path**: `fastArgmaxAvoidingRecent` in `Qwen35Generator.swift` should remain default-off (rep_penalty=1.0). Enabling it globally adds ~0.3 ms/step for no benefit when EOS is properly configured. Mac bench proved plain greedy + full EOS set doesn't loop.

## 8 · Non-goals for this handoff

- fp16 monolithic 2B ship (too big, fails ANE, blows jetsam)
- INT4 ship (quality fail — see §2)
- Re-testing 0.8B (already shipping at v1.0.3)
- Vision tower from the 2B VL checkpoint (out of scope, text only)

## 9 · File references

- Converter base: `conversion/test_qwen3_5_full_decode_trace.py`
- 2B fp16 converter (already written, monolithic): `conversion/build_qwen35_2b_decode.py`
- Palettize script: `conversion/build_qwen35_decode_int4.py`
- Research-era chunked converter: `conversion/build_qwen35_decode_chunks.py` (fp32 boundary — needs update for shipping)
- Swift generator: `Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen35Generator.swift`
- Swift runner integration: `Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift`
- Swift download manifest: `Sources/CoreMLLLM/ModelDownloader.swift`
- HF 0.8B repo (template): `https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreML`
- HF 2B repo (already created today, empty): `https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreML`

## 10 · Acceptance criteria for v1.1.0

1. 2B chunked INT8 loads on iPhone 17 Pro without jetsam
2. Compute plan audit prints ANE ≥ 85% for BOTH chunks
3. Factual parity: "What is the capital of France?" produces a response starting with "Paris" and contains no demonstrable falsehoods in the first 60 tokens
4. Japanese: "こんにちは" and "美味しい餃子のレシピを教えて" produce coherent Japanese with no language codeswitching
5. EOS fires naturally — no `<|endoftext|>` literal in visible stream
6. Decode throughput ≥ 12 tok/s on iPhone 17 Pro (acceptable for chat)
7. Total app memory ≤ 3.5 GB sustained (below jetsam margin)
8. Existing 0.8B path untouched — switching between 0.8B and 2B in picker works without reinstall
