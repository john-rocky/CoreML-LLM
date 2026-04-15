# ANE / CoreML LLM Conversion — Techniques NOT Yet in This Repo (2026-04 scan)

Scope: publications, repos, frameworks, APIs shipped **mid-2024 → 2026-04** that apply to the PyTorch → CoreML → ANE pipeline for Gemma 4 E2B on iPhone 17 Pro, and that are genuinely distinct from the already-documented recipe. Every "novel" item has been cross-checked against `docs/ANE_CONVERSION_RECIPE_2026.md`, `docs/FUNDAMENTAL_UNTRIED.md`, `docs/UNEXPLORED_APPROACHES*.md`, `docs/ANE_OPTIMIZATION_SURVEY.md`.

---

## 1. Executive summary — what's actually new and applicable

Ten months of harvesting has brought real additions. The landscape is **not** fully mined. Ranked by expected ROI on our tok/s goal:

1. **ANEMLL-Dedup / multifunction + constant dedup** (`ct.utils.save_multifunction`) — genuine weight-dedup across prefill / decode / rotate functions within a single `.mlpackage`. Cuts package size ~50% without changing per-op tok/s, which in turn frees ANE mmap budget to keep more chunks resident. Not in our repo; most of our docs still treat prefill and decode as separate packages.
2. **In-model argmax with (index, value) output** — moves a 256K-vocab softmax/argmax off the host boundary. Reduces every decode-step I/O from ~512 KB (fp16 logits) to 8 B. This is documented as "possible" in two places in our repo but not implemented; ANEMLL ship it as `--argmax` flag, and LiteRT-LM does **not** have it.
3. **Multi-function rotation for sliding-window attention** (ANEMLL's 4-function pattern: `infer / infer_rotate / prefill / prefill_rotate`). The rotation variants swap the sliding-window KV layout at the 1K-token boundary without a re-export. Directly applicable to Gemma 4 E2B (its L0-L14 sliding-window layers).
4. **Yetter / SqueezeBits disaggregated prefill-on-ANE + decode-on-MLX** packaged as a single multifunction mlpackage. Different from our current "CoreML everywhere" plan; the novelty is the zero-copy KV handoff format between frameworks.
5. **`AllowLowPrecisionAccumulationOnGPU` optimization hint** (coremltools 9.0b1 / iOS 26, Nov 2025). Applies to the GPU-offloaded lm_head chunk we already have; potential small speedup at measurable accuracy cost.
6. **Delta compilation** (Orion, Feb 2026) — patches weight blobs on disk and reloads without going through ANECCompile. 8.5× faster reload. Only matters to us if we adopt live adapter hot-swap; otherwise this is research-only.
7. **LoRA-adapter-as-IOSurface-input** (Orion) — adapters passed as runtime inputs, not baked weights. Would let us A/B test fine-tuned variants without re-export. **Non-App-Store** (private API).
8. **`ProfileComputePlan` diagnostic API** (iOS 18+, but under-used) — exposes per-op dispatch target + estimated latency. Already accessible via coremltools 9.0's `MLComputePlan`. We currently grep Instruments output; `MLComputePlan.forModelStructure` is more reliable.
9. **Embedding-table 4-bit with joint QAT** (AFM 2025, §3.2 of 2507.13575). Weight-level only, but note: Apple **jointly trains** the embedding quant with the transformer QAT, unlike our post-training palettization. If we ever run QAT, the embedding goes in the same optimizer loop.
10. **`optimize_im2col` PyTorch op lowering** (coremltools 9.0). Doesn't apply to transformer LLMs directly, **flagged only to reject**.

Everything else discovered is either re-packaged from the recipe already in `ANE_CONVERSION_RECIPE_2026.md`, not applicable to iOS App Store distribution, or research-grade.

---

## 2. Per-technique analysis

### 2.1 ANEMLL-Dedup — surgical constant deduplication across mlpackage functions

- **What it is**: `coremltools.utils.save_multifunction` identifies byte-identical constant tensors across function branches of a single `mlprogram` and stores them once. ANEMLL extended this with "surgical" dedup that also catches near-duplicate weights after Conv1x1 absorption (where RMSNorm scale was folded into Conv).
- **Source**: [anemll/anemll (Mar 2026)](https://github.com/Anemll/Anemll), [coremltools multifunction docs](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html).
- **Applicability to Gemma 4 E2B**: direct. If we publish prefill-chunk + decode-chunk + rotate-chunk as three functions inside one `.mlpackage` instead of three standalone files, the attention projection and FFN weights are shared on disk and in RAM.
- **Risk**: low. The catch is that the two functions must share the **same weight tensor pointers** post-conversion, which means we have to convert them from the **same** `nn.Module` instance and feed both to `save_multifunction`. Our current scripts convert prefill and decode from separate module instances (lines in `conversion/convert_gemma4.py`) — the tensor identities won't match even if values do. Needs a code change to thread the same weight objects through both traces.
- **Estimated tok/s gain**: not direct. Indirect gain: ~50% smaller on-disk footprint → more chunks cacheable in the mmap region → fewer cold-path page faults on the 3rd+ decode step. Estimate +1 to +2 tok/s for a 4-chunk model under memory pressure.

### 2.2 In-model argmax / (index, value) output

- **What it is**: terminate the lm_head not with logits but with a `topk(k=1, axis=-1)` producing `(index_i32, value_fp16)`. Host reads 8 bytes per token instead of `vocab_size × 2` bytes.
- **Source**: [anemll/anemll](https://github.com/Anemll/Anemll) ships `--argmax` flag; referenced briefly in our `ANE_OPTIMIZATION_SURVEY.md:163` and `LITERT_RUNTIME_ANALYSIS.md:261` as "not yet done."
- **Applicability**: direct; a one-line addition to the lm_head export (`logits = top1(logits)`).
- **Risk**: breaks temperature sampling and speculative-decoding verification (we need the top-K logits for MTP tolerance checks). Solution: emit **two** lm_head functions in the mlpackage — `lm_head_greedy` (argmax) and `lm_head_topk` (top-K, K≈8) — dedup their Conv2d weight via §2.1, and pick at runtime.
- **Estimated tok/s gain**: 1–3 tok/s on the decode path purely from reduced host I/O. For 256K-vocab Gemma 4 it's the single biggest per-token data transfer today.

### 2.3 Multi-function rotation for sliding-window attention

- **What it is**: ANEMLL's Gemma 3 support uses four functions (`infer`, `infer_rotate`, `prefill`, `prefill_rotate`) where `_rotate` variants recompute the sliding-window KV layout when the decode position crosses the window boundary (typically 1024 or 4096). The non-rotating function is used for steady-state decoding.
- **Source**: [ANEMLL Gemma 3 notes](https://github.com/Anemll/Anemll).
- **Applicability**: Gemma 4 E2B has sliding-window local attention on L0-L14 (window=1024 per our `SLIDING_WINDOW_FINDINGS.md`). We currently handle this with a single model that does a masked recompute every step — this is strictly more work than needed. Switching to the 4-function pattern eliminates the mask recompute on 1023 of every 1024 decode steps.
- **Risk**: medium. We must verify that function switching has no ANE re-warm penalty (CoreML caches specialization; should be free after first call).
- **Estimated tok/s gain**: Gemma 4 E2B has 14 sliding-window layers out of 34. If current attention cost is ~30% of decode and rotation accounts for ~25% of that, we recover ~7.5% on decode path → ~3-4 tok/s at current baseline.

### 2.4 Yetter / SqueezeBits disaggregated CoreML(prefill) + MLX(decode)

- **What it is**: single multifunction package exports `prefill` (stateless, outputs KV as tensors) and tokenizes the handoff: the Swift runtime reads KV tensors from the CoreML output and re-wraps them as MLX arrays. INT4 per-channel for the CoreML side, INT4 per-group-64 for the MLX side, FP16 activations.
- **Source**: [SqueezeBits disaggregated inference blog, 2025](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176), [AtomGradient/hybrid-ane-mlx-bench](https://github.com/AtomGradient/hybrid-ane-mlx-bench).
- **Applicability**: partially conflicts with our current direction. Our decode path already uses ANE for most layers; we're not decode-bound on GPU. But this technique is the first **shipped** approach that successfully shares weights between two frameworks, and is worth considering if we hit the ANE decode ceiling.
- **Risk**: high complexity; two quantization recipes, two runtime stacks. KV tensors have to be exported — undoing most of the value of stateful KV.
- **Estimated tok/s gain**: unknown for Gemma 4 E2B; they report "comparable to MLX decode" for a 0.6B model, with ANE-class TTFT. For us decode already runs on ANE, so probably a wash.

### 2.5 `AllowLowPrecisionAccumulationOnGPU` optimization hint

- **What it is**: added in coremltools 9.0 (Nov 2025) / iOS 26. `MLModelConfiguration.optimizationHints.allowLowPrecisionAccumulationOnGPU = true` lets GPU matmul accumulate in FP16 instead of FP32. ~20-30% GPU matmul speedup on M3+/A17+.
- **Source**: [coremltools 9.0 release notes](https://github.com/apple/coremltools/releases), [Apple docs](https://developer.apple.com/documentation/coreml/mlmodelconfiguration/allowlowprecisionaccumulationongpu).
- **Applicability**: our lm_head runs on `.cpuAndNeuralEngine` today but we've been considering moving it to GPU. If we move it, this hint applies.
- **Risk**: potential LM-head logits divergence. Quick A/B on a prompt set of 1K prompts, target KL < 0.001 vs FP32 accumulation.
- **Estimated tok/s gain**: small (lm_head is a ~5-10% tail of total compute). ~0.5 tok/s.

### 2.6 Delta compilation (Orion)

- **What it is**: at adapter-swap time, the Orion runtime rewrites only the weight blob in the compiled `.aneb` and reloads, skipping ANECCompile (`4.2 s → 494 ms`, 8.5×).
- **Source**: [Orion arXiv 2603.06728](https://arxiv.org/abs/2603.06728), [mechramc/Orion](https://github.com/mechramc/Orion).
- **Applicability**: only if we want **runtime adapter swap**. For a single-model deployment, this is irrelevant. Flag for future multi-adapter iteration.
- **Risk**: uses private `_ANEClient` interface. **App Store rejected.**
- **Estimated tok/s gain**: 0 for steady-state decode.

### 2.7 LoRA-adapter-as-IOSurface-input (Orion)

- **What it is**: compile-time frontend rewrites `Y = conv1x1(x, W_base)` to `Y = conv1x1(x, W_base) + α · (x @ A) @ B`, where `A, B` are runtime IOSurface inputs. Lets you hot-swap LoRA adapters without recompile.
- **Source**: [Orion](https://arxiv.org/abs/2603.06728).
- **Applicability**: only if we need multi-adapter serving. Could also enable per-user personalization on-device.
- **Risk**: private API; same App Store issue as §2.6.
- **Estimated tok/s gain**: the IOSurface-input pattern itself costs 2-3% per layer over baked weights. Net loss unless adapter swap frequency is high.

### 2.8 `MLComputePlan.forModelStructure` for per-op dispatch analysis

- **What it is**: iOS 18 API (under-used). Returns the ANE/GPU/CPU decision for every op in the compiled model, with estimated latency per op. coremltools 9.0 exposes a convenience wrapper.
- **Source**: [coremltools 9.0 notes](https://github.com/apple/coremltools/releases), [MLComputePlan docs](https://developer.apple.com/documentation/coreml).
- **Applicability**: direct and cheap. Replace our manual Instruments-based inspection of "which ops fell back to CPU/GPU" with this programmatic call. Run it in CI after every conversion.
- **Risk**: none.
- **Estimated tok/s gain**: 0 directly; 2× faster debugging cycle, which translates into finding more real wins faster.

### 2.9 Jointly-quantized embedding table (AFM 2025)

- **What it is**: Apple jointly trains embedding quantization (4-bit) with transformer 2-bit QAT, not as a separate post-training step. Embedding dequant runs on CPU/GPU, not ANE.
- **Source**: [Apple Intelligence Foundation Models 2025, §3.2, arXiv 2507.13575](https://arxiv.org/html/2507.13575v3).
- **Applicability**: only if we adopt QAT. Our current pipeline is post-training palettization. The novelty is that jointly-trained embedding quant recovers most of the accuracy loss that hit our earlier W2/W3 palette experiments.
- **Risk**: needs 1-2 weeks GPU + the original Gemma 4 E2B pretraining data (we don't have it; we'd need Hugging Face Gemma 4 E2B-QAT if/when Google publishes one — Gemma 3 QAT exists, Gemma 4 QAT does not as of 2026-04).
- **Estimated tok/s gain**: 0 (quality recovery only, no speed).

### 2.10 Balanced 2-bit palette `{-1.5, -0.5, +0.5, +1.5}` (AFM 2025)

- **What it is**: Apple's choice, which empirically trains more smoothly than `{-2, -1, 0, 1}`. Noted as a standalone curiosity in our `ANE_CONVERSION_RECIPE_2026.md:363` but never tried as W2 post-training either.
- **Applicability**: we rejected W2/W3 post-training (gibberish). The balanced palette **won't fix that** — it only helps during QAT. Flagged to reject as PTQ.

### 2.11 Register-Window (RW) mechanism for vision

- **Source**: [AFM 2025, §4.1.2](https://arxiv.org/html/2507.13575v3).
- Applies to the vision encoder only (ViTDet-L + RW). Not in scope for Gemma 4 E2B text. **Reject.**

### 2.12 Auto chunk calculator + FP16 preflight (ANEMLL tooling)

- **What it is**: two scripts — one computes optimal chunk boundaries by simulating SRAM 32 MB working-set pressure (Orion method formalised), one runs weight-distribution analysis for FP16 overflow risk and recommends a pre-scaling α.
- **Source**: [ANEMLL](https://github.com/Anemll/Anemll).
- **Applicability**: replace our current 4-chunk hard-coded split with a size-aware computation. Risk that it recommends fewer/more chunks than we have now, which would require re-benchmarking.
- **Estimated gain**: if our current split is already near-optimal (we tuned it empirically), gain is small. If we add QKV packing later, re-running this is useful.

---

## 3. Techniques investigated and rejected

| Technique | Why rejected |
|---|---|
| **Delta compilation + direct `_ANEClient`** | Private API, App Store rejected. |
| **LoRA-adapter-as-IOSurface** | Same. |
| **Training on ANE via maderix reverse-engineered APIs** | Same. |
| **Balanced 2-bit palette as PTQ** | Only helps during QAT; our W2 PTQ already produces gibberish. |
| **Register-Window vision** | Scope = text only. |
| **`optimize_im2col` pass** | Convolution im2col doesn't apply to LLM decoders. |
| **Yetter-style disaggregated prefill/decode** | We're not decode-bound on ANE; complexity > gain at current ops. Revisit if ANE decode ceiling is hit. |
| **ChunkAttention / BucketServe dynamic batching** | Server-serving concepts; on-device single-stream decode doesn't benefit. |
| **MLX M5 "Neural Accelerators"** | iPhone 17 Pro is A19, not M5; the NA path is GPU-integrated and exclusive to Mac. MLX-Swift on iPhone still uses stock GPU. |
| **Gemma 3 QAT checkpoints for Gemma 4** | Architecture differs; can't transfer. Gemma 4 QAT not published. |
| **End-to-end on-device QAT (OpenReview 2025)** | Interesting but needs 20+ hours of on-device training per iteration; research, not product. |

---

## 4. Honest verdict

**The landscape is not fully harvested.** Four techniques warrant immediate engineering time, in order:

1. **In-model argmax + dual-function lm_head** (§2.2) — smallest delta, biggest bang, ~1-3 tok/s.
2. **Multifunction mlpackage with constant dedup** (§2.1) — unlocks memory headroom, reshapes how we structure the pipeline.
3. **4-function sliding-window rotation** (§2.3) — directly addresses Gemma 4's L0-L14 local-attention overhead.
4. **`MLComputePlan` in CI** (§2.8) — not a speedup but a force multiplier on every other optimization effort.

Items 5-12 are either **non-App-Store** (Orion/maderix direct-ANE variants) or **quality-focused** (2-bit QAT, jointly-quantized embeddings), neither of which moves tok/s on the deployable path.

**What the field has NOT published in 2025-2026**, despite claims of activity:

- No public paper showing ANE > 50% utilization on transformer decode.
- No reverse-engineering paper on A19 specifically (only A18 / M4 Orion work).
- No CoreML compiler pass that fuses multiple transformer layers into a single ANE dispatch (the "whole-model-in-one-dispatch" noted in `GPU_WHY_FAST.md:315` remains blocked by the ≈20 undocumented graph-depth limits).
- No App-Store-legal equivalent of Orion's direct `_ANEClient` path.
- No Apple-blessed 1-bit or 1.58-bit quantization tooling (GPTQ 1-bit / BitNet territory is still MLX-only, not CoreML).

Gemma 4 E2B specifically has **no** dedicated published recipe yet. We are still the early adopters for this model; ANEMLL ships Gemma 3 only. This is an advantage for us: the novel items above can be combined with our existing recipe before the community catches up.

## Sources

- [Apple Intelligence Foundation Language Models Tech Report 2025 (arXiv 2507.13575v3)](https://arxiv.org/html/2507.13575v3)
- [Apple ML Research — Foundation Models 2025 Updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)
- [Apple ML Research — Foundation Models Tech Report 2025 (landing)](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)
- [Apple ML Research — Exploring LLMs with MLX and M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Orion — arXiv 2603.06728 (Feb 2026)](https://arxiv.org/html/2603.06728v1)
- [mechramc/Orion GitHub](https://github.com/mechramc/Orion)
- [ANEMLL GitHub](https://github.com/Anemll/Anemll)
- [ANEMLL project site](https://www.anemll.com/)
- [smpanaro/coreml-llm-cli](https://github.com/smpanaro/coreml-llm-cli)
- [AtomGradient/hybrid-ane-mlx-bench](https://github.com/AtomGradient/hybrid-ane-mlx-bench)
- [SqueezeBits — Disaggregated Inference on Apple Silicon (NPU prefill + GPU decode)](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- [apple/coremltools releases](https://github.com/apple/coremltools/releases)
- [coremltools Multifunction Models guide](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html)
- [coremltools Quantization Algorithms guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html)
- [Apple Developer — MLModelConfiguration.allowLowPrecisionAccumulationOnGPU](https://developer.apple.com/documentation/coreml/mlmodelconfiguration/allowlowprecisionaccumulationongpu)
- [WWDC25 Session 360 — Discover ML & AI Frameworks](https://developer.apple.com/videos/play/wwdc2025/360/)
- [Argmax — iPhone 17 On-Device Inference Benchmarks](https://www.argmaxinc.com/blog/iphone-17-on-device-inference-benchmarks)
- [tzakharko — A19/M5 GPU Neural Accelerators benchmark](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)
- [Gemma 3 QAT release (Google, Apr 2025)](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)
- [InsiderLLM — Apple Neural Engine for LLM Inference](https://insiderllm.com/guides/apple-neural-engine-llm-inference/)
