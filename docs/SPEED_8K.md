# 8K Context Speed — Exhaustive Research & Roadmap (v3)

**Status:** ctx=2048 ships at 31 tok/s · ctx=8192 currently 15 tok/s · WFA rejected on quality · EAGLE-3 in training.
**Goal:** 8K @ 50+ tok/s *without quality loss*, with a documented path to 80+.
**Updated:** 2026-04-12. Code unchanged by this doc.


## Primary design principle: **ANE execution is the product**

This is a core objective, not just a means. All technique selection is filtered through "runs on ANE without CPU/GPU fallback" first, "quality preserved" second, "speed gain" third. Rationale:

- **Thermal**: ANE sustains 31 tok/s for 10+ min on iPhone 17 Pro. Competitors on GPU (MLX, MLC LLM) lose ~44% after thermal throttle.
- **Coexistence**: GPU stays free — users can run LLM alongside games, video editors, photo processing.
- **Energy**: ~0.07 J/tok (ANE) vs ~0.67 J/tok (GPU) — 10× more efficient.
- **App Store**: ANE is Apple's sanctioned path, zero App Store risk.
- **Moat**: we are the only ANE-native LLM library targeting iPhone with latest models. Losing ANE = commoditized to the MLX ecosystem.

Therefore this document rejects any proposal whose decode path requires CPU/GPU fallback, even if "faster in isolation."

---
---

## 0. Hard constraints (all confirmed)

| Constraint | Implication |
|---|---|
| ANE is FP16 internally | Weight-only INT quant = 0 speed gain (size-only). W8A8 opens INT8-INT8 path → 1.3-1.6× Apple-documented. |
| ANE SRAM 32 MB | KV is in DRAM anyway; SRAM spill risk only for intermediate softmax tensors. |
| Static shapes only | Use EnumeratedShapes (≤ 128 shapes, all ANE-eligible) for K / ctx / window variations. RangeDim falls back to GPU/CPU. |
| Manual-attention ceiling | SDPA fusion incompatible with Gemma4 QK-norm scale=1.0. |
| Current 8K KV: 48 MB | 7 full-attn layers × (1,1,8192,512) × fp16 × 2 ≈ 48 MB. Memory-bandwidth bound. |
| 4 chunked decode mlpackages | KV-share: L19/24/29/34 read kv14 (shared from L14). |

**Rejected outright:** SDPA fusion · naive WFA (quality NG past FW) · weight-only quant for speed · full GPU-only decode · Medusa (1.3% acc) · Self-speculative without pretraining (0% acc) · TurboQuant 3-bit KV (ANE forces FP16 decomp).

---

## 1. 2026 state of the art — compatibility matrix

Sorted by ANE feasibility × quality-preservation × ROI. All numbers are from cited papers on GPU; ANE-equivalent will be lower (typically 0.6-0.8× of GPU speedup), noted where applicable.

### Tier A — Training-free, ANE-implementable, big wins (stack these)

#### A1. DuoAttention (ICLR 2025, MIT HAN Lab) — retrieval vs streaming heads
- **Idea**: ~50% of attention heads in a GQA model are "retrieval heads" (need full KV), the other 50% are "streaming heads" (only need fixed-length sink+window).
- **Cost to deploy**: one-shot *optimization-based* pass on synthetic data (~hours on A100) to classify heads. Inference is fully training-free — just load the head mask.
- **Published numbers (Llama-3 GQA)**: 1.65× memory, **1.50× decode speedup**, quality preserved.
- **ANE mapping**: KV per head is already per-head; streaming heads just compile with a smaller (fixed) KV buffer. Fits our chunked pipeline and EnumeratedShapes cleanly.
- **Expected for Gemma 4 E2B @ 8K**: **15 → ~22 tok/s**, 40% KV memory saved, zero quality loss.
- **Source**: [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention) · [paper](https://arxiv.org/abs/2410.10819)

#### A2. KIVI — 2-bit KV quantization (ICML 2024) — **REJECTED for ANE**
- **Idea**: per-channel for Key, per-token for Value, symmetric INT2. Zero fine-tune.
- **Numbers (GPU)**: 2.6× memory, **2.35-3.47× throughput**, <2% accuracy loss on Llama/Mistral.
- **ANE reality (measured)**: KV-only INT8 gives **~0 speedup on ANE** because the ANE compute pipeline is FP16-internal and CoreML dequantizes INT8 KV to FP16 before the matmul. The theoretical DRAM→SRAM bandwidth halving does NOT materialize as wall-clock speedup on ANE. Confirmed by our bench session.
- **What was hoped to work**: full **W8A8** (weights + activations both INT8) opens the int8-int8 compute path on A17 Pro+ per Apple's ResNet50 docs.
- **What actually works** (2026-04-13 update): **W8A8 is ANE-incompatible on iPhone**. `coremltools.experimental.linear_quantize_activations` inserts quantize / dequantize MIL ops that the iPhone ANE compiler rejects — it's not a file-mixing or calibration issue, the op set itself is unsupported. Mac Studio M4 Max also showed 0% speedup (no INT8-INT8 fast path on M4). W8A8 is off the table for ANE-only decode on this device family. See `docs/EXPERIMENTS.md` for the incident record.
- **Conclusion**: both KIVI and W8A8 are shelved on ANE. Decode speed has to come from structural rewrites (Q-batching, DuoAttention, TriForce) and speculative decoding (EAGLE-3, Mirror), not from quantizing beyond weights.
- **Source**: [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) (GPU-focused)

#### A3. TriForce — self-speculative with sparse KV draft (MLSys 2025)
- **Idea**: draft = **same target weights** with top-k *retrieved* KV (e.g., 2K out of 8K), target = full 8K. Losslessly verifies.
- **Numbers**: 97.6% acceptance with 1K sparse tokens out of 128K. **7.78× end-to-end at 128K**. At 8K you'd use ~512-1024 top-k out of 8K → expected acceptance 95%+ → ~2-2.5× speedup.
- **Training-free.** GQA support emerging (Llama-3.1 confirmed).
- **ANE catch**: top-k selection is dynamic; workaround is **block-level top-k with fixed budget** (select top-16 blocks of size 32 = fixed 512 tokens every time). This is ANE-compilable via EnumeratedShapes or static gather-indices.
- **Hierarchical mode**: EAGLE-3 small draft (60% acceptance) → TriForce sparse target (verify 3 tokens) → full target (verify any rejects). Triple-tier if we get brave.
- **Expected @ 8K**: 15 × ~2.2 = **33 tok/s** solo, or **compounds with EAGLE-3**.
- **Source**: [TriForce project](https://infini-ai-lab.github.io/TriForce/) · [arXiv](https://arxiv.org/abs/2404.11912)

#### A4. Quest — query-aware sparse KV (ICML 2024)
- **Idea**: keep per-block min/max keys, estimate block criticality via Q @ [min,max], attend only to top-K blocks.
- **Numbers**: **7.03× self-attention**, 2.23× end-to-end latency, training-free.
- **ANE catch**: dynamic top-K like TriForce. Same fixed-block-budget workaround.
- **Expected @ 8K**: 15 → ~22-27 tok/s solo.
- **Overlap with TriForce**: TriForce is a superset (same retrieval idea + hierarchical speculation). Pick one — **TriForce preferred** because it composes with EAGLE-3.
- **Source**: [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest)

#### A5. MiniCache — layer-depth KV merging (NeurIPS 2024)
- **Idea**: middle-to-deep layers have high KV similarity across adjacent layers. Merge them (magnitude + direction decomposition). Training-free.
- **Numbers**: 5× compression, **~5× throughput** on ShareGPT for Llama-2.
- **Our situation**: Gemma 4 **already KV-shares L15-34 to L13/L14 structurally**. MiniCache would merge *projections'* outputs across adjacent layers, not replace the existing sharing. Orthogonal.
- **Expected compound gain on top of existing sharing**: likely modest (much of the depth redundancy is already claimed). ~1.1-1.2× plausible.

#### A6. InfLLM — training-free block memory (NeurIPS 2024)
- **Idea**: past KV into blocks, pick block representatives, attend to top-k block reps.
- **Practical on ANE**: yes, with static block count and top-k budget.
- **Overlap**: subsumed by TriForce / Quest functionally. Skip.

### Tier B — Training-free, benefit uncertain / less composable

- **SnapKV** (prompt-eviction, great for long prompts but we're at 8K ctx not 32K — marginal).
- **PyramidKV** (pyramid budget per layer, for >64K prompts — overkill).
- **NACL / AdaKV / ChunkKV / KVMerger / MorphKV / Expected Attention / RocketKV / RetroAttention** — all KV-compression variants; pick best-of-breed after ablation. `NVIDIA/kvpress` has a standardized benchmark of 20+ methods — **consult it before implementing**.
- **Mamba / Jamba / Samba / Zamba hybrid SSM**: architectural rewrite, non-starter for our Gemma 4 retrofit. Flag for v1.0+ if we do our own model.

### Tier C — Quality-preserving but require fine-tune (days on GPU)

- **StreamingLLM + QLoRA recovery**: sink=4 + window=2048 for full-attn layers, fine-tune for long-ctx. Closes the WFA quality gap.
- **MHA → MLA retrofit (MHA2MLA, ACL 2025)**: 4-14% KV size, better perplexity than GQA. v0.6+ candidate.
- **LayerSkip**: needs pretraining.

### Tier D — Apple-specific levers

- ~~**W8A8 with real calibration**~~ — **ANE-incompatible, shelved 2026-04-13**. `linear_quantize_activations` emits quantize/dequantize MIL ops the iPhone ANE compiler refuses. Mac Studio M4 Max also measured 0% gain. Archived script under `conversion/build_w8a8_proper.py`; do not reintroduce without evidence the op set situation has changed in a future coremltools release.
- **EnumeratedShapes {2048, 8192}**: single mlpackage, ANE-safe for all shapes. Ship "fast mode" (2K) + "long mode" (8K) under one binary.
- **Prefix caching / persistent KV**: for repeated system prompts. **4-35× TTFT on cache hits at 1-4K, up to 136× at 32K** (safetensors-based, survives reboot). Not decode throughput but huge UX win.
- **A19 Pro GPU tensor cores** (Metal Performance Primitives, Xcode 26.1+): 7.5 TFLOPS FP16 / 13.5 TOPS INT8. Argmax-bench: 2.5-3.1× vs A18 Pro GPU. **Route prefill (compute-bound) to GPU**; keep decode (bandwidth-bound) on ANE.
- **Apple Recurrent Drafter (ReDrafter)**: Apple's own draft architecture — RNN-based + dynamic tree attention + target-model distillation. 2.3× on Apple Silicon MLX. Basis of iOS 26's 48.77M / 3.18B ratio. **EAGLE-3 is equivalent or better per recent benchmarks** — no need to switch.
- **Apple Mirror Speculative Decoding (2026 research)**: bidirectional speculation, **NPU+GPU parallel execution**, 30% better than EAGLE-3 on server. Implementation on iPhone ANE+GPU is the Apple-blessed parallel path. Worth watching; paper's focus is 14-66B server models but the NPU+GPU split idea is transferable.

---

## 1b. ANE-compatibility verdict per technique

| Technique | ANE decode path? | Notes |
|---|---|---|
| Pre-alloc masks/RoPE | ✅ pure ANE | Swift-only buffer reuse, ANE graph unchanged |
| KV-share Q-batching | ✅ pure ANE | Just widens matmul Q dim |
| INT8 KV cache | ❌ no-op on ANE (confirmed 2x) | (a) CoreML dequantizes INT8 KV to FP16 before ANE compute → no bandwidth halving; (b) any coremltools quantize/dequantize op insertion hits the same ANE compiler rejection as W8A8. Shelved. |
| W8A8 | ❌ ANE compiler rejects | `coremltools.experimental.linear_quantize_activations` inserts quantize/dequantize MIL ops that the iPhone ANE compiler doesn't support. Mac Studio M4 Max also shows 0% speedup. Not a calibration issue — the op set itself is unsupported. See EXPERIMENTS.md 2026-04-13. |
| DuoAttention | ✅ pure ANE | head classification offline; runtime uses 2 static KV banks |
| EAGLE-3 draft + verify | ✅ pure ANE | small decoder layer, EnumeratedShapes for K=1/3 |
| StreamingLLM + QLoRA | ✅ pure ANE (after FT) | standard attention post-fine-tune |
| MHA → MLA retrofit | ✅ pure ANE (after retrain) | latent projection matmul |
| MiniCache | ✅ pure ANE | merged KV = standard tensor |
| Prefix caching | ✅ pure ANE | just I/O, ANE graph unchanged |
| **TriForce / Quest** | ⚠️ **partial** | dynamic top-K → CPU/GPU fallback risk; needs block-static redesign to stay ANE |
| A19 Pro GPU tensor cores (decode) | ❌ violates ANE goal | *But* prefill-only route is acceptable |
| Mamba / SSM hybrid | ❌ unproven on ANE | selective scan not yet ported |
| Orion / private ANE API | ❌ App Store incompatible | research only |

**Stack preference: everything in ✅ rows, TriForce only if it can be made static-shape block-level, nothing from ❌.**

---

## 2. Recommended stack for 8K — honest expected throughput

Stacked sequentially (each factor applies to the current baseline). Updated 2026-04-13 after W8A8 was ruled out on iPhone ANE:

```
             step-speedup    cumulative tok/s @ 8K
baseline                     15
+ pre-alloc masks      ×1.03     15.5   ← measured micro-opt (was hoped ×1.07)
+ KV-share Q-batch     ×1.08     16.7
+ DuoAttention (A1)    ×1.50     25.0   ← quality preserved, needs 4-8h A100 calib
+ EAGLE-3 (retrain)    ×1.45     36.3   ← fully lossless via verify; retrain required
+ Mirror v1 (GPU draft) ×1.15    41.8
```

**Removed from the stack**:
- **INT8 KV / KIVI** — both (a) measured ineffective on ANE (CoreML dequantizes to FP16 before ANE compute) AND (b) any coremltools quant/dequant op insertion hits ANE compiler rejection. Shelved for good on this device family.
- **W8A8** — same ANE compiler rejection of quantize/dequantize MIL ops. Shelved.

**Conservative pessimism adjustment (ANE overhead, non-linear compounding)**: multiply final by 0.70 → **~29 tok/s @ 8K** is the realistic landing zone without W8A8. Adding TriForce on top of EAGLE-3 (see alternative path below) pushes this to 50-70 tok/s.

Alternative path replacing EAGLE-3 with TriForce hierarchical:
```
+ TriForce (sparse draft + full verify)  ×2.3 → ~80 tok/s
+ EAGLE-3 on top of TriForce (3-tier)    ×1.3 → ~105 tok/s   (research territory)
```

For perspective:
- Apple's own 3B on iOS 26 reports 2-4× from draft-model speculative. We're in the same league.
- MLX-based 7B LLMs on M4 Max see ~45 tok/s. Beating 60 tok/s @ 8K for a 2.7B on iPhone ANE is at or above industry SOTA.

---

## 3. Execution order (updated)

### Parallel track P1 — EAGLE-3 (already running)
On Colab, 2 epochs × 30k corpus, acc0 tracking 52% @ 7%. Finish, deploy, measure on iPhone. **Deliverable**: ctx=2048 at 55-70 tok/s, ctx=8192 at ~30 tok/s.

> **Status snapshot 2026-04-12**: the `acc0 52% @ 7%` figure is from the most recent `train_eagle_draft.ipynb` / `train_eagle3_draft.ipynb` run (both untracked — the notebooks are still iterated in Colab). Before quoting this number elsewhere, open the latest notebook output cell and verify it still holds. The data-prep and hidden-state collection pipeline is tracked: `conversion/download_eagle_corpus.py` + `conversion/collect_eagle_hidden_states.py`. CoreML export of the draft model goes through `conversion/build_speculative.py` (modified, not yet merged). See `docs/EXPERIMENTS.md` for the abandoned Medusa comparison that motivated choosing EAGLE-3.

### Parallel track P2 — ANE bandwidth reduction (training-free)
1. **Pre-allocate masks/umask/RoPE** in `ChunkedEngine.swift`. Landed in PR #6; measured ~0% on 2K (mask-fill was already sub-ms). Kept for architectural cleanup and larger-ctx where it may still matter.
2. **KV-share Q-batching** for the 4 full-attention shared layers (L19/24/29/34) in `gemma4_swa_chunks.py`. Fuses 4 Q projections + 4 attention matmuls into batched ops. ~40 LoC conversion-side change. Open.
3. ~~**INT8 KV cache**~~ — **shelved 2026-04-13**, same ANE compiler rejection of quantize/dequantize ops as W8A8. Even the "no quant op" variant measures 0% on ANE (CoreML dequantizes to FP16 before compute).

### Parallel track P3 — DuoAttention head identification
1. Port DuoAttention training script to Gemma 4 E2B (setup: BookSum data, ~4-8h on A100).
2. Produce head classification (~50% retrieval, ~50% streaming).
3. Modify `gemma4_swa_chunks.py` to use streaming-window KV for streaming heads while keeping full KV on retrieval heads. Per-layer per-head budget known at build time → ANE-compilable.

### ~~Parallel track P4 — W8A8 calibration pipeline~~ (shelved 2026-04-13)
**Reason**: `coremltools.experimental.linear_quantize_activations` inserts quantize/dequantize MIL ops the iPhone ANE compiler rejects. Mac Studio M4 Max also measured 0% gain. Not a calibration issue — the op set itself is unsupported. Scripts remain at `conversion/build_w8a8.py` / `build_w8a8_proper.py` for historical reference; do not reinvest without evidence a future coremltools release adds ANE support for these ops.

### Sequential track S1 — after P3 lands: TriForce or advanced sparse
Once DuoAttention is in, consider TriForce as a natural extension (it uses the same head-wise KV idea at the sequence dimension). Hierarchical mode with EAGLE-3 is the upper bound.

### Track T1 — Apple-blessed path (lower priority unless the 3B model becomes the product)
Foundation Models framework (iOS 26) ships Apple's 3.18B + ReDrafter. For Gemma 4 E2B specifically, we can't piggyback — but for *UX wins* we can use persistent KV / prefix caching patterns regardless of model.

---

## 4. Comprehensive reference list

**Sparse attention + KV eviction**
- [DuoAttention (MIT HAN Lab, ICLR 2025)](https://arxiv.org/abs/2410.10819) · [repo](https://github.com/mit-han-lab/duo-attention)
- [Quest (ICML 2024)](https://arxiv.org/abs/2406.10774) · [repo](https://github.com/mit-han-lab/Quest)
- [MInference](https://arxiv.org/abs/2407.02490) — pattern-based sparse at prefill
- [SnapKV](https://arxiv.org/abs/2404.14469)
- [PyramidKV](https://arxiv.org/html/2406.02069v1)
- [NACL (ACL 2024)](https://arxiv.org/abs/2408.03675)
- [AdaKV (NeurIPS 2025)](https://github.com/FFY0/AdaKV)
- [H2O (NeurIPS 2023)](https://arxiv.org/abs/2306.14048) — heavy-hitter eviction
- [StreamingLLM (ICLR 2024)](https://arxiv.org/abs/2309.17453) — attention sinks
- [InfLLM (NeurIPS 2024)](https://arxiv.org/abs/2402.04617)
- [Cascading KV Cache](https://arxiv.org/html/2406.17808v1)
- [SAGE-KV](https://arxiv.org/html/2503.08879v1)
- [RocketKV](https://arxiv.org/html/2502.14051v3)
- [Expected Attention](https://arxiv.org/html/2510.00636v1)
- [Training-Free Native Sparse Attention (NeurIPS 2025)](https://openreview.net/forum?id=sQjYtFSEuZ)
- [NVIDIA/kvpress — unified benchmark](https://github.com/NVIDIA/kvpress)

**KV quantization**
- [KIVI (ICML 2024)](https://arxiv.org/abs/2402.02750) · [repo](https://github.com/jy-yuan/KIVI) — 2-bit K/V training-free
- [KITTY (Nov 2025)](https://www.arxiv.org/pdf/2511.18643) — 2-bit KV improvements
- [Coupled Quantization](https://arxiv.org/abs/2405.03917) — 1-bit KV experiments
- [TurboQuant (ICLR 2026)](https://github.com/AmesianX/TurboQuant) — 3-bit KV, 5× compression

**KV merging / layer-depth**
- [MiniCache (NeurIPS 2024)](https://arxiv.org/abs/2405.14366)
- [KVMerger](https://arxiv.org/abs/2407.08454)
- [ChunkKV](https://openreview.net/forum?id=20JDhbJqn3)
- [KVTC — transform coding](https://openreview.net/forum?id=aNVKROYpLB)
- [MorphKV (2025)](https://arxiv.org/html/2603.20397)

**Speculative decoding**
- [EAGLE-3 (NeurIPS 2025)](https://arxiv.org/abs/2503.01840) · [repo](https://github.com/SafeAILab/EAGLE) — our current training
- [TriForce (MLSys 2025)](https://arxiv.org/abs/2404.11912) · [project](https://infini-ai-lab.github.io/TriForce/)
- [QuantSpec (ICML 2025, Apple)](https://arxiv.org/abs/2502.10424) — 4-bit KV + self-spec, 90% accept
- [SpecPV (Dec 2025)](https://arxiv.org/html/2512.02337v1)
- [LongSpec (Feb 2025)](https://arxiv.org/html/2502.17421v2)
- [Apple ReDrafter (MLX)](https://machinelearning.apple.com/research/recurrent-drafter) — 2.3× on Apple Silicon
- [Apple Mirror SD (2026)](https://machinelearning.apple.com/research/mirror) — NPU+GPU parallel, +30% over EAGLE-3 on server
- [SpecEE (ISCA 2025)](https://arxiv.org/abs/2504.08850) — speculative early exit
- [MagicDec (2024)](https://arxiv.org/html/2408.11049v1)

**Hybrid architectures (for future v1.0+)**
- [Jamba (AI21)](https://arxiv.org/abs/2403.19887) — Transformer+Mamba+MoE
- [Samba (Microsoft)](https://arxiv.org/html/2406.07522v1) — Mamba+SWA, 256K extrapolation from 4K training
- [Zamba](https://www.zyphra.com/research) — 1 attention layer + Mamba
- [DeepSeek MLA (V2/V3)](https://arxiv.org/abs/2405.04434) · [MHA2MLA (ACL 2025)](https://arxiv.org/abs/2502.14837) — retrofit path

**Apple platform**
- [On-Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- [Apple Foundation Models Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)
- [Apple Intelligence Updates 2025](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)
- [coremltools — Quantization Perf](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)
- [coremltools — Flexible Input Shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)
- [A19/M5 Neural Accelerators benchmark (Tzakharko)](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)
- [Argmax — iPhone 17 benchmarks](https://www.argmaxinc.com/blog/iphone-17-on-device-inference-benchmarks)
- [Stephen Panaro — KV Cache for ANE](https://stephenpanaro.com/blog/kv-cache-for-neural-engine)
- [ANEMLL](https://github.com/Anemll/Anemll)

**Agent memory / prompt caching**
- [Agent memory below the prompt (Q4 persistent KV)](https://arxiv.org/html/2603.04428v1) — 4-35× TTFT at 1-4K, 136× at 32K
- [HotPrefix (ACM SIGMOD 2025)](https://dl.acm.org/doi/10.1145/3749168)
