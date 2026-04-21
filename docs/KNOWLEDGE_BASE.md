# CoreML-LLM Knowledge Base — master index

**Date:** 2026-04-22
**Pass:** source-level deep-read of 13 external repos + our own Swift+Python stack

## What this is

Not the world's fastest LLM stack (yet) — potentially the world's best-documented ANE+CoreML+Metal LLM knowledge base. This index points at dense docs, each source-verified with file:line citations.

## Docs in this knowledge base

| Doc | Scope | Source-verified |
|---|---|---|
| `OUR_STACK_ANATOMY.md` | What our Swift runtime and Python conversion actually do | Our own Sources/ + conversion/ |
| `GEMMA4_ARCHITECTURE_VERIFIED.md` | Gemma 4 E2B + E4B spec from our config, HF transformers, llama.cpp, Google refs | 4 independent sources |
| `METAL_ANE_KERNELS_CATALOG.md` | Reusable Metal + ANE kernel patterns | llama.cpp + whisper.cpp + MLX + ml-ane-transformers |
| `COREMLTOOLS_AND_IOS18.md` | Converter passes, palettize/quantize, iOS 18 SDPA + stateful ops | coremltools source + WWDC24 |
| `MOBILE_RUNTIME_COMPARISON.md` | llama.cpp / MLX / MLC-LLM / ExecuTorch / ANEMLL / LiteRT-LM matrix | All 6 repos read |
| `SPECULATIVE_DECODING_SURVEY.md` | 10+ SD variants, paper + code + our applicability | llama.cpp examples + arxiv refs |
| `QUANTIZATION_SURVEY.md` | SpinQuant, QuaRot, AWQ, SmoothQuant, COMPACT, LaRoSA, SCAP, R-Sparse with ANE fit | Research repos + papers |
| `ANE_PRIVATE_API_CONSTRAINTS.md` | Private API constraints, maderix vs ANEMLL cross-check | maderix/ANE + Anemll source |
| `METAL_PORT_REFERENCE.md` | llama.cpp Gemma 4 + MLX STEEL blueprint | source-verified |
| `LITERT_LM_ARCH_VERIFIED.md` | LiteRT-LM 56 tok/s architecture decomposition | source-verified |
| `ANEMLL_SOURCE_NOTES.md` | ANEMLL actionable tricks | source-verified |
| `ROUND7_FINDINGS.md` | Post-training ANE-only candidate audit | prior pass; updated 2026-04-22 |

## Top 10 surprises from this deep-read

1. **EAGLE-3 is opt-in dead weight in our own code.** `LLM_EAGLE3_ENABLE=1` gate. 22% accept rate. 234 MB ANE. Default is DrafterUnion (CV Qwen + PLD n=2/3 + suffix trie), not EAGLE-3. `CoreMLLLM.swift:203-293`.

2. **Cross-vocab Qwen drafter disabled by default.** Too slow on iPhone (1.8 tok/s). Opt-in only. `CoreMLLLM.swift:87-90`.

3. **All sampling is argmax.** No top-k/top-p on-device. `verifyCandidatesTopN` exists but only for acceptance tolerance tuning. `ChunkedEngine.swift:1370-1533`.

4. **SDPA fusion was attempted and reverted.** "produces slightly different results from manual attention, causing wrong token predictions." Manual `ane_softmax` used instead. `models/gemma4_swa_chunks.py:136-138`.

5. **PLE scale bug was the biggest accept-rate killer.** Missing ×16 per_layer_embed_scale on collected hidden states caused 0% accept on iPhone EAGLE-3 for a long time. `collect_eagle_hidden_states_w4a8.py:88-94`.

6. **coremltools auto-applies RMSNorm overflow prevention.** `max_val = reduce_max(abs(x))` scaling built-in since 2024+. We don't need to do it manually. `coremltools/converters/mil/frontend/torch/ops.py:3107-3171`.

7. **iOS 18 has native SDPA MIL op AND stateful read_state/write_state.** We're likely not using either at full capacity. `coremltools/mil/ops/defs/iOS18/transformers.py:18-167`.

8. **Multifunction models are NOT supported in coremltools PTQ.** `@_multifunction_unsupported` decorator. Relevant because our `build_verify_chunks.py` builds multifunction (decode_q1 + verify_qK). PTQ workaround needed.

9. **llama.cpp has a production Gemma 4 E2B path.** `src/models/gemma4-iswa.cpp` with ISWA dual-KV, `has_kv()` branching, fused GeGLU via `ggml_geglu_split`, FlashAttention kernel for exact head-dim 256. Three `TODO @ngxson` comments are known uncertainty points we need to cover in a port.

10. **ExecuTorch's `export_static_llm_coreml.py` is a direct template.** Per-layer graph break + per-layer partition to CoreML + NamedDataStore weight sharing across multifunction methods. Matches our 4-chunk design.

## Top 5 CONFIRMED dead ends (do not reopen)

1. **Drafter on Gemma 4 E2B** — oracle-live acc gap 3-9×. Neither maderix nor ANEMLL addresses it. LiteRT's MTP recipe is private. Confirmed dead.

2. **ANE-only path to 56 tok/s** — with all ROUND7 candidates + ANEMLL infra + maderix Private API tricks, ceiling is ~40 tok/s @ 2K. LiteRT 56 is Metal + MTP.

3. **MLX lazy eval keeping intermediates in GPU L2** — Zenn article's causal explanation. MLX source has no L2-pinning logic; fusion is compile-time op-inlining. Refuted.

4. **"MLXANE" module in mlx-swift** — referenced in Zenn article. Does not exist. Refuted by mlx-swift source.

5. **Q-only KV-sharing retrofit** — L15-34 KV-share design is baked in Gemma 4. Cannot be improved by post-training. ANEMLL does not have it; only llama.cpp's `gemma4-iswa.cpp` and our own path support it.

## Top 5 ACTIONABLE items (high ROI, source-verified)

1. **Adopt ExecuTorch's `export_static_llm_coreml.py` pattern** for future model conversions — per-layer graph break is cleaner than our hand-rolled approach. Knowledge reuse for Qwen2.5, future models. See `RUNTIME_COMPARISON.md` §2.

2. **iOS 18 per-block INT4 palettization** for lm_head and embeddings — saves ~500 MB on E4B bundles. Negligible accuracy loss. Requires iOS 18+ deployment target. See `COREMLTOOLS_AND_IOS18.md` §3.

3. **iOS 18 stateful `read_state` / `write_state` ops** for KV cache — removes explicit copyBack memcpy, reduces dispatch overhead. Verify whether our current `ChunkedEngine.copyBack` path can be replaced. See `COREMLTOOLS_AND_IOS18.md` §4.

4. **Metal Phase 3 port using `llama.cpp/src/models/gemma4-iswa.cpp`** as direct blueprint + `kernel_flash_attn_ext_f32_dk256_dv256` Metal shader. 3-6 week effort; ceiling ~50 tok/s without MTP. See `METAL_PORT_REFERENCE.md`.

5. **ANEMLL IOSurface-backed CVPixelBuffer pattern audit** — we already do something similar (`ChunkedEngine.swift:546-568`); compare against ANEMLL's explicit `ring-16 / ping-pong-2` and adopt any diff. See `ANEMLL_SOURCE_NOTES.md` §1.2.

## Top 5 open questions (worth a next-session investigation)

1. **iOS 18 SDPA op vs our manual `ane_softmax`.** Our conversion comment says SDPA causes wrong predictions. Is that still true on iOS 18 native SDPA, or was it a pre-iOS-18 limitation? Quick A/B test on Mac Studio.

2. **Multifunction + PTQ path.** `@_multifunction_unsupported` is a blocker for quantizing our verify_qK paths. Does coremltools optimize.torch pre-export quantization bypass this? Check.

3. **ExecuTorch's CoreML delegate weight sharing** — `NamedDataStore` across methods may already do what our 4-chunk split approximates, but with cleaner compile-time handling. Worth comparing per-layer graph-break export to our current pipeline.

4. **L-MTP training feasibility on Gemma 4 E2B.** [L-MTP GitHub](https://github.com/Xiaohao-Liu/L-MTP) ships training code. 1-2 A100 days per paper. If live acceptance matches training acceptance (which is L-MTP's design goal), this is the only credible path past drafter death.

5. **A19 Pro Tensor Cores** (2025) — new GPU-side tensor acceleration separate from ANE. If Metal port happens, these could outrun ANE decode. No public benchmark yet.

## Cross-doc source citations index

Primary external sources (all in `/Users/majimadaisuke/Downloads/workspace/repo-review/` unless noted):

- `maderix-ANE/` — Private API reverse-engineering, training on ANE
- `Anemll/` — Production ANE LLM runtime (Llama/Qwen/Gemma3)
- `llama.cpp/` — Metal + Gemma variants + speculative decoding catalog
- `whisper.cpp/` — Encoder CoreML offload pattern + Metal shader library
- `gemma.cpp/` + `gemma_pytorch/` — Google reference (Gemma 1-3 only, no 4)
- `ml-ane-transformers/` — Apple ANE optimization principles
- `coremltools/` — Converter internals, MIL passes, iOS 18 ops
- `executorch/` — PyTorch Edge + CoreML backend blueprint
- `mlc-llm/` — TVM Relax + Metal + speculative
- `mlx-examples/` + `mlx-swift/` + `mlx-swift-examples/` — MLX LLM path
- `mlx/` (in our project) — MLX C++ core
- `SCAP/` + `R-Sparse/` — ROUND7 research candidates
- `LiteRT-LM` (in workspace) — The 56 tok/s baseline

HF transformers Gemma 4 at `/Users/majimadaisuke/Downloads/workspace/repo-review/transformers/src/transformers/models/gemma4/` (sparse-checked).
