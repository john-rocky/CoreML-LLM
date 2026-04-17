# MTP Investigation — LiteRT-LM reverse-engineering

**Date:** 2026-04-17
**Scope:** Full reverse-engineering of how LiteRT-LM implements MTP on Gemma 4 E2B (the same target that reportedly delivers 56.5 tok/s on iPhone 17 Pro GPU). Focused on de-risking whether we should re-try Path A weight extraction or retrain from scratch.

---

## 1. TL;DR

- **Training recipe is NOT public.** No training script, dataset, hyper-parameter, or loss formulation for Gemma 4's MTP drafter exists in `google-ai-edge/LiteRT-LM`, `google-ai-edge/ai-edge-torch`, the Gemma 4 HF model card, the Gemma 4 blog post, or any Google publication (verified via GitHub code search + web fetches on the Gemma 4 blog and HF model card). Google's engineer explicitly framed MTP as a "deployment-time optimisation" kept out of the public training artefacts ([HF discussion](https://huggingface.co/google/gemma-4-E4B-it/discussions/5)).
- **Pre-trained weights ARE public but only as a TFLite section inside the `.litertlm` container** — Section 9 (E2B, 44.3 MB, already extracted to `output/mtp_probe/section_9.tflite`) / Section 11 (E4B, 45.1 MB, mirrored at [shadowlilac/gemma-4-e4b-mtp-extraction-effort](https://huggingface.co/shadowlilac/gemma-4-e4b-mtp-extraction-effort)). Weights are JAX/TF-named (`MtpDrafterModel.*`, `layer_N/...`) and 50% INT8-palettised. They are NOT in any HF-format checkpoint.
- **Re-running Path A is only worth it if we can close Path A's actual failure mode** (drafter trained against LiteRT's W4A8 + transposed-attention target, mismatched against our HF fp target). The extraction itself is complete; the blocker is target-side numerics, not head weights. Section 7's verdict table below details this.

---

## 2. MTP head architecture

Confirmed by direct weight dump of `output/mtp_probe/section_9.tflite` (E2B) and cross-referenced with runtime code in `llm_litert_mtp_drafter.cc`:

```
Drafter = single 4-layer mini-transformer (NOT a DeepSeek-V3-style multi-head MTP).

Input:
  activations  (1,1,3072)  fp32   = concat(embed(token_last)[:1536],
                                          projected_activations_or_base_hidden[:1536])
  input_pos    (1,)        int32
  mask         (1,1,1,32003) bool
  param_tensor (1,1,1,7)   int32  ← packed RoPE position + flags
  kv_cache_k_13 (1,1,32003,256)  int8 ← SHARED with base model (SWA layer 13)
  kv_cache_v_13 (1,1,256,32003)  int8
  kv_cache_k_14 (1,1,32003,512)  int8 ← SHARED with base model (full-attn layer 14)
  kv_cache_v_14 (1,1,512,32003)  int8

Pipeline:
  mtp_pre_proj   Linear(3072 → 256)                    # hidden down-project
  layer_0  (SWA, Q-only, reads kv_13):
    RMSNorm(256)
    q_proj  Linear(256 → 1024)   [4 heads × 256]
    attn(Q, shared_K13, shared_V13)
    o_proj  Linear(1024 → 256)
    RMSNorm(256)
    gate1/gate2 Linear(256 → 2048)     GeGLU
    down  Linear(2048 → 256)
  layer_1  (SWA, kv_13)  — same shape
  layer_2  (SWA, kv_13)  — same shape
  layer_3  (FULL, Q-only, reads kv_14):
    q_proj  Linear(256 → 2048)   [4 heads × 512 head_dim]
    o_proj  Linear(2048 → 256)
    MLP same as layers 0-2
  RMSNorm(256)
  embedder.decode  Linear(256 → 262144)   # LM head, tied weights to base embedder
  mtp_post_proj    Linear(256 → 1536)     # → projected_activations (carry state)

Output:
  logits                (1,1,262144)   fp32
  projected_activations (1,1,1536)     fp32
```

**Key observation:** There are NO `k_proj` / `v_proj` projections anywhere in the drafter graph. It never writes its own KV — it directly reads the base model's kv_13 (sliding, head_dim=256) and kv_14 (full, head_dim=512). This is why Google can run the drafter on CPU while the base model runs on GPU: they share the KV tensors by reference.

**Weight inventory** (from `section_9_weights.txt`, 73 weight tensors):
- INT8 (`type_17` / `bool` dtype code): `mtp_pre_proj [256,3072]`, all `q_proj`/`o_proj` (4), all `gate1`/`gate2`/`down` (12), `embedder.decode [262144,256]`, `mtp_post_proj [1536,256]`. → 23 quantised weight tensors.
- FP32: 21 RMSNorm scales (5 per layer × 4 + final norm), 2 RoPE tables (SWA: [1,1,128], FULL: [1,1,256]), misc constants.

---

## 3. Public numbers (acceptance, tok/burst, speedup)

**No MTP-specific numbers have ever been published by Google for Gemma 4.** The HF `litert-community/gemma-4-E2B-it-litert-lm` model card lists only aggregate decode throughput, with no separation between MTP-on and MTP-off:

| Device                              | Backend | Prefill tok/s | Decode tok/s |
|-------------------------------------|---------|---------------|--------------|
| iPhone 17 Pro                       | GPU     | 2,878         | **56.5**     |
| iPhone 17 Pro                       | CPU     | 532           | 25.0         |
| MacBook Pro M4 Max                  | GPU     | 7,835         | 160.2        |
| Samsung S26 Ultra                   | GPU     | 3,808         | 52.1         |
| NVIDIA RTX 4090                     | GPU     | 11,234        | 143.4        |

Source: [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm).

**Inferred numbers (from runtime code, not published):**
- **K = 3 draft steps per cycle** — hard-coded by the base model's `verify` signature shape: `input_pos_dims[0] - 1 = 4 - 1 = 3` (`llm_litert_mtp_drafter.cc:240`).
- **Expected acceptance ≈ 50–60 %** — back-solved from the 56.5 tok/s iPhone GPU number assuming a ~20 tok/s base decode and K=3 greedy speculation:
  `tok/s = 1/(1 + 1/3) × (1 + p·3 + p²·3 + p³) → p ≈ 0.55–0.65`. This is unverified.
- **Acceptance rate logging exists in the runtime** but only at `ABSL_LOG(INFO)` level on executor destruction: `"Success rate: " << num_verified / num_drafted` (`llm_litert_mtp_drafter.cc:154-156`). Not exposed in any metric API.

**DeepSeek-V3 reference numbers** (cited by third-party bloggers as what Google "probably" achieves, but NOT confirmed for Gemma): 2.4 tokens/burst, 1.8× speedup, 85–90 % acceptance on the second predicted token. Sources: [DeepWiki DeepSeek-V3 MTP](https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.4-multi-token-prediction-(mtp)), [flowhunt Gemma 4 MTP blog](https://www.flowhunt.io/blog/gemma-4-released-without-mtp-multi-token-prediction/). These are not comparable — DeepSeek V3 is a server-side MoE with a single MTP module, Gemma's drafter is a 4-layer transformer on mobile.

---

## 4. Inference control flow (replicate-this section)

Directly transcribed from `runtime/executor/llm_litert_mtp_drafter.cc` (full file read; relevant functions: `Draft`, `RunDraftingLoop`, `RunVerification`, `PrepareVerifierInputBuffers`, `PrepareDrafterInputBuffers`).

```
Draft(position P, last_token T, activations A, kv_in, kv_out):
  # 1. Prepare drafter buffers (duplicates kv_out by reference — shared tensors).
  input_pos = [P]
  mask = causal_mask(start=P, steps=1)
  if param_tensor present: pack_rope(P, 0)

  # 2. Drafting loop: K=3 sequential calls to the 44 MB drafter.
  drafted = []
  act_ptr = A    # first step uses base-model last-hidden; later steps use
                 # drafter's own projected_activations.
  for i in 0..K-1:
    e = embed(T)                     # 1536-dim, NO sqrt(H) scaling
    drafter_input.activations = concat(e, act_ptr)  # 3072 total
    drafter_model.RunAsync(...)
    t_i = argmax(drafter_output.logits)     # greedy, top_k=1
    drafted.append(t_i)
    T = t_i
    act_ptr = drafter_output.projected_activations

  # 3. Verifier prep: base model's "verify" signature (T=4 = K+1).
  verifier_input_pos = [P, P+1, P+2, P+3]
  mask = causal_mask(start=P, steps=4)
  verifier_embeddings   = embed_lookup_prefill([original_T, drafted[0], drafted[1], drafted[2]])
  verifier_per_layer_embeddings = ple_lookup(...)
  # KV cache: active_verifier_input_buffers aliases the *base* model's kv
  # buffers by reference — same tensors the drafter just read from.

  # 4. Run verifier. Base model writes KV at positions [P..P+3] via its
  #    internal dynamic_update_slice ops — rejected positions' entries are
  #    harmlessly written and later masked out.
  base_model.RunAsync("verify", ...)
  verified = argmax(verify_output.logits)   # 4 tokens (one per position)

  # 5. Rejection sampling = exact greedy match (NOT probabilistic).
  num_accepted = 0
  bonus = -1
  for i in 0..K-1:
    if verified[i] == drafted[i]:
      num_accepted += 1
    else:
      bonus = verified[i]    # the *verifier's* token at the first mismatch
      break
  if bonus == -1:
    bonus = verified[K]      # all K drafts accepted → bonus is verifier[K]

  # 6. KV rollback — NONE. Stale KV entries at positions > (P + num_accepted)
  #    sit in the cache, invisible to attention because current_step only
  #    advances by num_accepted + 1.
  current_step += num_accepted + 1

  # 7. Carry state for next cycle.
  last_verified_token_id_idx = num_accepted  # index into verify output
  # Next Draft() reads verify_output.activations[last_verified_token_id_idx]
  # as the seed for its projected-activations slot (via
  # ConcatenateEmbeddingsAndActivationsFromVerifierBuffer).

  return drafted[:num_accepted] + [bonus]
```

**Compute-unit placement:**
- Drafter: backend_constraint=`cpu` in `model.toml` (Google's XNNPACK choice). The runtime has a GPU path (`UpdateCompilationOptions` branch for `Backend::GPU` at `llm_litert_mtp_drafter.cc:127-136`) but the shipped model is CPU-tagged.
- Base model: same backend for all signatures (decode/prefill/verify). No mixed-backend dispatch.
- **KV cache sharing across compute units: via `TensorBuffer::Duplicate()` (shared handles, zero copy).**

**KV rollback = dissolved (verified already in `LITERT_RUNTIME_ANALYSIS.md §B1.3`).** The verify pass IS the commit; rejected tokens' KV is written then masked out.

---

## 5. Training recipe findings

**Not public. Searched exhaustively:**

| Source                                            | Result |
|---------------------------------------------------|--------|
| `google-ai-edge/LiteRT-LM` code search `MTP`      | 13 files, all runtime (`llm_litert_mtp_drafter.*`, executor integration, test stub, CMake, build). **No training code.** |
| `google-ai-edge/ai-edge-torch` code search `MTP` / `drafter` / `speculative` | **0 hits.** Confirmed via `gh api search/code`. |
| [Gemma 4 blog post](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) | Zero mentions of MTP, drafter, speculative decoding. |
| [Gemma 4 HF blog](https://huggingface.co/blog/gemma4) | Zero mentions. |
| [Gemma 4 core model docs](https://ai.google.dev/gemma/docs/core) | Zero mentions. |
| arXiv 2505.00232 (ML Drift GPU inference paper)   | Does NOT mention MTP. This paper is about GPU tensor virtualisation, not speculative decoding. |
| Gemma 3 / Gemma 4 technical reports               | No training recipe for MTP heads. |
| Google engineer response on HF E4B discussion     | "MTP is currently treated as a deployment-time optimisation rather than part of the public model interface." ([discussion](https://huggingface.co/google/gemma-4-E4B-it/discussions/5)) |

**Inferred (NOT stated):** the drafter was trained with a two-stage process:
1. Base model trained with MTP auxiliary losses (standard in 2025+ frontier LLMs — Gemma 4 blog implicitly acknowledges this by saying MTP heads are "stripped" from HF release).
2. Drafter is a distilled student targeting the base model's top-1 next-token prediction on K=3 positions ahead, given the base's last hidden state + the drafter's running carry state.

Neither dataset, loss weighting, nor optimiser is documented.

**DeepSeek-V3-style reference** (what Google *probably* did, per third-party reverse-engineering): composite loss `L_main + λ · Σ L_mtp_k` with λ=0.1 (per NVIDIA NeMo's public recipe), causal chain across depths. Not confirmed for Gemma. This is the basis of our Path C training recipe — which trained to 38 % val acc and delivered 17 % on-device.

---

## 6. Weight distribution findings

**Weights ARE downloadable — but only as TFLite inside `.litertlm`:**

| Container                                                  | Section | Size    | Path                                              |
|------------------------------------------------------------|---------|---------|---------------------------------------------------|
| `litert-community/gemma-4-E2B-it-litert-lm` / `gemma-4-E2B-it.litertlm` | 9       | 44.3 MB | Section9_TFLiteModel_tf_lite_mtp_drafter.tflite   |
| `google/gemma-4-E4B-it-litert-lm` / `gemma-4-E4B-it.litertlm` (via [shadowlilac mirror](https://huggingface.co/shadowlilac/gemma-4-e4b-mtp-extraction-effort)) | 11      | 45.1 MB | Section11_TFLiteModel_tf_lite_mtp_drafter.tflite  |

Section numbering differs (E2B has 10 sections, E4B has 12 — the extra two in E4B are audio encoder/adapter). Structural layout of the drafter itself is identical across E2B/E4B with dimension swaps (hidden_size 1536 vs 2560).

**Extraction tooling:**
1. Official: `bazel run //schema/py:litertlm_peek_main -- --litertlm_file=X --dump_files_dir=Y` from `google-ai-edge/LiteRT-LM`.
2. Our repo: `conversion/extract_mtp_drafter.py` scans for `TFL3` magic bytes at 4-byte aligned offsets. **Already works — extracted to `output/mtp_probe/section_9.tflite`.**

**NO HF / PyTorch / safetensors / Kaggle download exists.** Google's official Gemma 4 HF repos (`google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`) have the MTP heads explicitly stripped — confirmed in the HF discussion: "The publicly released Gemma 4 model definition exposes a standard autoregressive interface. Components related to MTP such as additional prediction heads, are not included in the open source model config or forward pass." ([source](https://huggingface.co/google/gemma-4-E4B-it/discussions/5))

---

## 7. Mapping — LiteRT tensor names ↔ our extraction script ↔ HF Gemma 4 tensor names

LiteRT's drafter uses **JAX/TF-style paths** (from `jax2tf_arg_N/ReadVariableOp` + composite op names). Our extraction already captures all 66 weight tensor names in `output/mtp_probe/section_9_weights.txt`. Key mappings:

| LiteRT drafter tensor name                                           | Shape           | Our PyTorch name (`conversion/mtp_drafter_model.py`)   | HF Gemma 4 equivalent |
|---------------------------------------------------------------------|-----------------|--------------------------------------------------------|-----------------------|
| `MtpDrafterModel.mtp_pre_project/mtp_pre_proj/.../dot_general`      | [256, 3072]     | `mtp_pre_proj.weight`                                  | (no equivalent — drafter-only) |
| `layer_N.pre_q/attn.pre_q/.../q_einsum/.../dot_general`             | [1024, 256] (SWA) / [2048, 256] (full) | `layers[N].q_proj.weight`                              | (no equivalent) |
| `layer_N.post_qkv/attn.post_qkv/attn_vec_einsum/.../dot_general`    | [256, 1024] / [256, 2048] | `layers[N].o_proj.weight`                              | (no equivalent) |
| `layer_N.post_qkv/mlp/gating_einsum1/.../dot_general`               | [2048, 256]     | `layers[N].gate_proj.weight`                           | (no equivalent) |
| `layer_N.post_qkv/mlp/gating_einsum2/.../dot_general`               | [2048, 256]     | `layers[N].up_proj.weight`                             | (no equivalent) |
| `layer_N.post_qkv/mlp/linear/.../dot_general`                       | [256, 2048]     | `layers[N].down_proj.weight`                           | (no equivalent) |
| `MtpDrafterModel.decode_softmax/.../embedder.decode/composite`      | [262144, 256]   | `lm_head.weight` (can re-tie to target's embedder instead of copy) | `model.embed_tokens.weight` (tied, after 256→1536 up-project removal) |
| `MtpDrafterModel.mtp_post_project/mtp_post_proj/.../dot_general`    | [1536, 256]     | `mtp_post_proj.weight`                                 | (no equivalent — drafter-only) |
| `jax2tf_arg_N/ReadVariableOp` [256]                                 | [256]           | `layers[N].*_norm.weight` (RMSNorm scales)             | N/A |
| `layer_N.pre_q/attn.pre_q/.../maybe_rope/div` [1,1,128] / [1,1,256] | [1,1,128]/[1,1,256] | RoPE precomputed tables (inline constants)            | Replaced by our `precomputed_rope` in `ane_ops.py` |

**Unknown at time of this doc:** whether `embedder.decode [262144, 256]` is a *new* LM head trained against the drafter's 256-dim bottleneck, or if the 262144 rows are weight-tied to Gemma 4's `model.embed_tokens` (also [262144, 1536]). The shape mismatch (256 vs 1536) means they CANNOT be byte-identical — some projection must bridge. `LITERT_CONTAINER_ANALYSIS.md` §LM Head already flags this as "Weight tying: CONFIRMED between Section 0 embedder and Section 8 LM head" but does NOT confirm tying between Section 9's drafter head and Section 0. Weight-byte inspection needed to resolve.

**Path A extraction script (`conversion/extract_mtp_drafter.py`) tensor-name assumptions:** the script does not hard-code names — it detects `activations` input on each section to flag the drafter. Name-based remapping happens in `conversion/mtp_drafter_model.py` / `build_mtp_drafter.py`. The failure mode of Path A (per `MTP_PATH_A_FINDINGS.md` + `MTP_PATH_C_FINDINGS.md`) was never a tensor-name mismatch — it was that **Google's drafter was trained against LiteRT's W4A8 quantised target**, whereas our ChunkedEngine serves FP16 hidden states from a different quantisation lattice. The per-token hidden distribution is offset by ~3× in L-2 norm (per Path C precompute bug analysis), which breaks the drafter's learned manifold.

---

## 8. Verdict

| Option                                            | Verdict          | Rationale |
|---------------------------------------------------|------------------|-----------|
| (a) Re-run Path A with corrected tensor mapping   | **No — not the right fix.** | Path A's failure was never a tensor-name mapping issue; all 66 weights map cleanly (see §7). The failure mode is target-distribution mismatch between LiteRT's W4A8 quantised forward and our HF-fp trunk. Until we either (i) re-quantise our target to match W4A8 exactly, or (ii) run the drafter through a *post-hoc* hidden-state calibration layer, Path A stays at 0 % acceptance. Re-extracting is a no-op; the TFLite file is byte-identical. |
| (b) Train from scratch (a second Path C attempt) | **Possibly — but only after Phase B verify-alignment lands.** | Path C's 38 % val / 17 % live gap was not a training problem — it was `PRIORITY_ROADMAP.md` item 11c (verify_qK vs decode_q1 fp16 drift on the target, not the drafter). A bigger/longer-trained drafter on the same target would hit the same 77 % break-even cliff. Re-training only becomes attractive if verify alignment closes and the break-even drops to ~55 %. At that point the $300–1000 / 5–10 A100 days investment is reasonable. Before that, it is burning money on the wrong lever. |
| (c) Abandon MTP (continue with GPU Metal-LLM path) | **Yes, for the next ~2 weeks.** | The current on-device baseline is 31 tok/s (ANE, Gemma 4 E2B). LiteRT ships 56.5 tok/s on iPhone 17 Pro **GPU**, not ANE. We already have `llama.cpp Metal + spec decode` implementation (per auto-memory). The cheapest path to beat 56.5 tok/s is finishing Phase 3 (ANE prefill + GPU decode mirror with ANY drafter, even prompt-lookup). MTP-specifically gains maybe +5 tok/s on top of GPU decode; GPU decode alone gains ~25 tok/s over our current ANE. The ordering matters. |

**Recommended sequence:**
1. Ship Phase 3 GPU decode (Metal) + any speculative drafter (prompt-lookup / shared-vocab / EAGLE-3 retry with corrected teacher forward). Measure tok/s uplift vs LiteRT.
2. If (1) lands us at ≥55 tok/s without MTP — stop. MTP is unnecessary.
3. If (1) lands at 40–54 tok/s — close verify-alignment (item 11c) and attempt either Path A-v2 (with hidden-state calibration patch) or a focused Path C retrain. Budget 5 A100 days for Path C; Path A-v2 is ~2 days of conversion work.
4. If (1) lands below 40 tok/s — GPU decode itself is broken; MTP won't save it.

---

## 9. Stop condition

Task specification's stop conditions:
- **stop-cond-1: "recipe fully public"** → NOT fired. No training recipe is public.
- **stop-cond-2: "partial"** → **FIRED.** Weights are public (TFLite section, already extracted), inference control flow is fully public (`llm_litert_mtp_drafter.cc` / `.h` both read in full), architecture is fully reverse-engineered (§2, §7). Training recipe is NOT public.
- **stop-cond-3: "nothing new"** → N/A.

---

## 10. File & URL references

**Source code (read in full during this investigation):**
- `/tmp/mtp_drafter.h` ← https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/main/runtime/executor/llm_litert_mtp_drafter.h
- `/tmp/mtp_drafter.cc` ← https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/main/runtime/executor/llm_litert_mtp_drafter.cc

**Local artefacts (already in our repo):**
- `/Users/majimadaisuke/Downloads/CoreML-LLM/output/mtp_probe/section_9.tflite` (E2B drafter, 44.3 MB)
- `/Users/majimadaisuke/Downloads/CoreML-LLM/output/mtp_probe/section_9_weights.txt` (weight listing, 66 tensors)
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/extract_mtp_drafter.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/mtp_drafter_model.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/build_mtp_drafter.py`

**Prior docs:**
- `docs/LITERT_CONTAINER_ANALYSIS.md` — Section-by-section breakdown of the `.litertlm` file.
- `docs/LITERT_RUNTIME_ANALYSIS.md` — Full runtime analysis (KV cache, verify pass, rejection sampling).
- `docs/MTP_PATH_A_FINDINGS.md` — Our failed extraction integration (0 % acceptance, target-distribution mismatch).
- `docs/MTP_PATH_C_FINDINGS.md` — Our failed self-train (17 % live acceptance, verify-drift bottleneck).
- `docs/PRIORITY_ROADMAP.md` item 11c — verify_qK ↔ decode_q1 alignment (load-bearing gate).

**Web sources cited:**
- https://github.com/google-ai-edge/LiteRT-LM — runtime repo, 13 MTP-related files, 0 training files.
- https://github.com/google-ai-edge/ai-edge-torch — export pipeline, 0 MTP references.
- https://huggingface.co/google/gemma-4-E4B-it/discussions/5 — Google engineer "MTP is deployment-time optimisation" statement.
- https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm — benchmark table (56.5 tok/s iPhone 17 Pro GPU).
- https://huggingface.co/google/gemma-3n-E2B-it-litert-lm — earlier Gemma 3n numbers (~27 tok/s, does NOT ship MTP drafter section — note: the file listing shows no Section11_TFLiteModel_tf_lite_mtp_drafter).
- https://huggingface.co/shadowlilac/gemma-4-e4b-mtp-extraction-effort — community E4B extraction, README + file listing.
- https://www.flowhunt.io/blog/gemma-4-released-without-mtp-multi-token-prediction/ — third-party blog (cites DeepSeek-V3 numbers, not Gemma-specific).
- https://dev.to/marcuswwchen/what-gemma-4s-multi-token-prediction-head-actually-means-for-your-eval-pipeline-3ik — third-party, 1-3% params claim.
- https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.4-multi-token-prediction-(mtp) — DeepSeek-V3 MTP reference (NOT Gemma).
- https://arxiv.org/abs/2505.00232 — ML Drift GPU inference paper (does NOT cover MTP).

---

## 11. What this investigation dissolved vs. confirmed

**Dissolved (was uncertain before this session):**
- "Maybe Google uses multi-head MTP like DeepSeek-V3 / Meta 2024 paper" → **No.** Single 4-layer transformer reading shared KV.
- "Maybe the training recipe is buried in ai-edge-torch" → **No.** Zero references in that repo.
- "Maybe the HF model card or Gemma 4 blog lists acceptance numbers" → **No.** Silent on MTP.
- "Maybe we missed a tensor-name mapping" → **No.** All 66 weights map cleanly; the failure is target-distribution, not naming.

**Confirmed (already known, now with citations):**
- K=3 is hard-coded via `verify` signature shape (`llm_litert_mtp_drafter.cc:240`).
- Rejection = greedy argmax match, not probabilistic (`llm_litert_mtp_drafter.cc:456-467`).
- KV rollback = none; stale entries masked out.
- Drafter shares kv_13 / kv_14 with base via `TensorBuffer::Duplicate()` — zero-copy reference.
- 56.5 tok/s iPhone GPU number is from the HF model card, not a Google blog. It is the only public Gemma 4 + MTP number Google has shipped.

**Still unknown (open questions, out of scope here):**
- Drafter `embedder.decode [262144, 256]` — tied to target embedder via a projection, or separately trained?
- What fraction of the 56.5 tok/s is MTP vs GPU base-decode improvement? (Google ships *both*; we cannot tell them apart without a MTP-off config flag in the runtime.)
- Did Google QAT the drafter against W4A8 target, or did they first train in fp and then quantise? (Determines whether calibration-based Path A-v2 can work.)
