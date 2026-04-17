# MTP Investigation — HF Checkpoint Inspection

**Date:** 2026-04-17
**Scope:** Read-only inspection of `conversion/output/gemma4-e2b-final/hf_model/model.safetensors`
plus upstream HF + `transformers` source evidence.
**Goal:** Decide whether the HF checkpoint already ships MTP / drafter weights
that Path A could have / should have reused.

---

## 1. TL;DR

- **The HF checkpoint contains ZERO MTP / drafter / speculative / extra-head
  weights.** Every one of 2011 tensors belongs to either the text transformer
  (35 layers), the vision tower, the audio tower, or the two cross-modal
  embedding projections. `lm_head` itself is absent — it is tied to
  `embed_tokens` (`tie_word_embeddings: true`).
- **HuggingFace `transformers` has no MTP surface** in either `gemma3n/` or
  `gemma4/` model folders. `Gemma3nForCausalLM` / `Gemma4ForCausalLM` hold only
  `self.model` and `self.lm_head`; `forward()` returns a single
  `(B, T, vocab)` logits tensor per step. No files with `mtp`, `drafter`,
  `multi_token`, or `speculative` in the name.
- **Google's own statement (quoted via FlowHunt article attributing @srikanta-221,
  a Google engineer) confirms the exclusion is deliberate**: *"The public model
  exposes only a standard autoregressive interface 'for broad compatibility.'
  MTP heads are excluded from the model config, forward pass, and checkpoint."*
  MTP weights live only inside the LiteRT-LM `.litertlm` export that Path A
  extracted (`Section11_TFLiteModel_tf_lite_mtp_drafter.tflite`).

**Implication for Path A:** `weights never existed in HF base`. Path A glued
Google's LiteRT-trained drafter (W4A8 quantized, trained against LiteRT's
quantized target) onto our HF-fp16 target. There was never an HF-side MTP head
that could have been used as a drop-in or reference. Path A's 0 % acc is the
direct consequence.

---

## 2. Full HF safetensors tensor list (categorized)

`model.safetensors`, single-file, 10.2 GB, `metadata = {"format":"pt"}`,
2011 tensors total. All BF16.

Source archive: `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/output/gemma4-e2b-final/hf_model/model.safetensors`.

### 2.1 Counts by subsystem

| Subsystem | Tensor count | Notes |
|---|---|---|
| `model.audio_tower.*` | 751 | 12 encoder layers × 62 + conv subsample + output_proj |
| `model.vision_tower.*` | 658 | 16 encoder layers + patch embedder + position table |
| `model.language_model.layers.*` | 595 | 35 layers × 17 weights per layer |
| `model.language_model.<top>` | 5 | see §2.2 |
| `model.embed_audio.embedding_projection.weight` | 1 | `[1536, 1536]` |
| `model.embed_vision.embedding_projection.weight` | 1 | `[1536, 768]` |
| **Anything outside those five prefixes** | **0** | checked explicitly |

Every tensor key starts with one of `model.audio_tower`, `model.vision_tower`,
`model.language_model`, `model.embed_audio`, `model.embed_vision`. **There is
no top-level `lm_head.*`, no `mtp_*`, no `drafter_*`, no `extra_*`, no
`head_*` anywhere.** (Confirmed with an `any(k.startswith(p) for p in [...])`
negation sweep — count = 0.)

### 2.2 `language_model` top-level (non-layer)

```
model.language_model.embed_tokens.weight                  [262144, 1536]   BF16  ← LM input embedding (also output, tied)
model.language_model.embed_tokens_per_layer.weight        [262144, 8960]   BF16  ← PLE (per-layer embeddings), 35*256 = 8960
model.language_model.norm.weight                          [1536]           BF16  ← final RMSNorm (pre-lm_head)
model.language_model.per_layer_model_projection.weight    [8960, 1536]     BF16  ← main→PLE mix
model.language_model.per_layer_projection_norm.weight     [256]            BF16  ← PLE norm scale
```

`tie_word_embeddings: true` in `config.json` → the LM head reuses
`embed_tokens.weight` as its projection matrix. No separate weight is stored.

### 2.3 Per-layer pattern (35 layers, identical for each)

```
input_layernorm.weight              [1536]
layer_scalar                        [1]
mlp.down_proj.weight                [1536, 6144]
mlp.gate_proj.weight                [6144, 1536]
mlp.up_proj.weight                  [6144, 1536]
per_layer_input_gate.weight         [256, 1536]
per_layer_projection.weight         [1536, 256]
post_attention_layernorm.weight     [1536]
post_feedforward_layernorm.weight   [1536]
post_per_layer_input_norm.weight    [1536]
pre_feedforward_layernorm.weight    [1536]
self_attn.k_norm.weight             [256]
self_attn.k_proj.weight             [256, 1536]
self_attn.o_proj.weight             [1536, 2048]
self_attn.q_norm.weight             [256]
self_attn.q_proj.weight             [2048, 1536]
self_attn.v_proj.weight             [256, 1536]
```

17 weights × 35 layers = 595. All sliding / full-attn layers share this
signature (the distinction lives only in `config.json.text_config.layer_types`
and `num_kv_shared_layers`).

### 2.4 Pattern sweep for MTP-like names

Direct substring search over all 2011 keys, case-insensitive:

| Pattern | Matches |
|---|---|
| `mtp` | 0 |
| `draft` | 0 |
| `extra_head` | 0 |
| `embedding_extra` | 0 |
| `speculative` | 0 |
| `medusa` | 0 |
| `eagle` | 0 |
| `multi_token` | 0 |
| `n_future` | 0 |
| `next_token` | 0 |
| `future` | 0 |
| `aux_head` | 0 |
| `lm_head` | 0 (tied, see §2.2) |
| `head.` / `.head` / `classifier` / `output_proj` | only `audio_tower.output_proj.{bias,weight}` — an audio-encoder output projection, unrelated to LM heads |

---

## 3. MTP-candidate tensor analysis

**None found.** Every tensor in the archive has a direct counterpart in the
HuggingFace `Gemma4ForConditionalGeneration` / `Gemma3nForCausalLM` module
tree (self-attention, MLP, PLE projections, vision/audio towers). There is no
weight surface for:

- a second LM head (Medusa-style),
- K≥2 future-token heads (Apple MTP paper / DeepSeek V3 MTP),
- a separate drafter sub-graph (Google LiteRT's 44 MB MTP drafter),
- an extra hidden-state projection that could feed an external drafter
  (nothing resembling `mtp_pre_proj`, `mtp_post_proj`, `projected_activations`,
  or `early_exit_head`).

`config.json` is equally clean: no `num_mtp_heads`, no `mtp_*`, no
`draft_*`, no `speculative_*` key. The only unusual keys are multimodal
(`audio_token_id`, `boa_token_id`, `vision_soft_tokens_per_image`, etc.) and
PLE (`vocab_size_per_layer_input`, `hidden_size_per_layer_input`,
`num_kv_shared_layers`). All account for tensors already enumerated above.

---

## 4. Upstream HF + transformers findings

### 4.1 HF model card — `google/gemma-4-E2B` / `google/gemma-4-E2B-it`

- https://huggingface.co/google/gemma-4-E2B
- https://huggingface.co/google/gemma-4-E2B-it
- https://huggingface.co/blog/gemma4

Cards and blog describe: hybrid SWA / full attention, Unified K&V in global
layers, p-RoPE, PLE, MoE variants, vision encoder, audio encoder.
**Zero mentions of MTP, multi-token prediction, drafter, speculative decoding,
extra heads, or auxiliary output projections** across all three sources.
Both cards only recommend deployment via `AutoModelForCausalLM` — they do not
discuss LiteRT-LM.

### 4.2 HF model card — `google/gemma-3n-E2B-it` (family parent)

- https://huggingface.co/google/gemma-3n-E2B-it

Same finding. Card emphasizes MatFormer nesting, AltUp, LAuReL, PLE, MobileNet
v5 vision, USM audio. No MTP / drafter surface mentioned.

### 4.3 `huggingface/transformers` source

- Folder `src/transformers/models/gemma4/`:
  `__init__.py`, `configuration_gemma4.py`, `convert_gemma4_weights.py`,
  `feature_extraction_gemma4.py`, `image_processing_gemma4.py`,
  `image_processing_pil_gemma4.py`, `modeling_gemma4.py`, `modular_gemma4.py`,
  `processing_gemma4.py`, `video_processing_gemma4.py`.
  → **No file named `*mtp*` / `*drafter*` / `*multi_token*` / `*speculative*`.**
- Folder `src/transformers/models/gemma3n/`: same story
  (`modeling_gemma3n.py`, `modular_gemma3n.py`, etc., no MTP file).
- Class `Gemma3nForCausalLM` (`modeling_gemma3n.py`):
  - `__init__` registers `self.model` (`Gemma3nTextModel`) and
    `self.lm_head` (`nn.Linear(hidden, vocab)`). No MTP / drafter / future
    head module.
  - `forward()` returns `CausalLMOutputWithPast` with a single
    `logits` tensor of shape `(B, T, vocab_size)`. Slicing is used for
    "only compute necessary logits" efficiency, **not** for emitting multiple
    future-token predictions per step.
- No `extra_output_projection`, `multi_token`, `future_token`, or multiple
  `lm_head` attributes anywhere in either model file.

### 4.4 Direct Google statement (quoted in third-party article)

- https://www.flowhunt.io/blog/gemma-4-released-without-mtp-multi-token-prediction/

The article attributes the decision to a Google engineer (handle
`@srikanta-221`) and quotes:

> "The public model exposes only a standard autoregressive interface 'for
> broad compatibility.' MTP heads are excluded from the model config, forward
> pass, and checkpoint."

Framing: MTP is treated as a LiteRT-only deployment-time optimization, not
a base-model feature. Direct match for the empirical picture in §2 and §4.3.

---

## 5. Cross-check vs our Path A extraction

Path A (`MTP_PATH_A_FINDINGS.md`) extracted the drafter from
`gemma-4-E2B-it.litertlm` Section 11. Names observed there vs the HF
safetensors:

| Tensor name in LiteRT drafter | Shape (E2B) | Present in HF safetensors? |
|---|---|---|
| `mtp_pre_proj` / `MtpDrafterModel.mtp_pre_project/mtp_pre_proj` | `(3072 → 256)` | **No** |
| `mtp_post_proj` | `(256 → 1536)` | **No** |
| `embedder.decode` (drafter LM head, 256→262144) | `(256, 262144)` | **No** (base HF has no 256-dim head — vocab projection is tied to the 1536-dim `embed_tokens`) |
| `layer_{0..3}.q_einsum` / `.o_proj` / `.gate1` / `.gate2` / `.down` | drafter-internal, hidden=256, FFN=2048 | **No** (HF layers are hidden=1536, FFN=6144) |
| RMSNorm scales inside the drafter | 256-dim | **No** |
| Signature input tensor `activations` `(1,1,3072)` | — | — |
| Signature input tensors `kv_cache_{k,v}_13`, `kv_cache_{k,v}_14` | shared with target | HF stores no KV caches (those are inference-time buffers), but the shapes match the target-layer-13/-14 roles. Cache contract is compatible; drafter weights are not. |

**Intersection = ∅.** Not a single drafter weight has a counterpart in
`model.safetensors`. The drafter is an entirely separate 4-layer model with
its own hidden width (256) and its own vocabulary projection.

This matches the inverse of what we'd want: the drafter reads the target
model's runtime KV (via layer-13/-14 contract) but carries its own weights.
Those weights ship only in the LiteRT container.

---

## 6. Implication for Path A failure diagnosis

**Verdict: "weights never existed in HF base."**

This is the cleanest of the three stop conditions. Specifically:

1. The HF safetensors are a faithful dump of `Gemma4ForConditionalGeneration`
   with multimodal encoders. It is complete for its declared architecture.
2. The MTP drafter is a *separate* 44 MB model that Google trains against the
   LiteRT-quantized (W4A8) target and ships only in
   `Section11_TFLiteModel_tf_lite_mtp_drafter.tflite` of the `.litertlm`
   container. Weights-wise it is disjoint from the HF checkpoint.
3. Path A's failure ("0 % acceptance") is the expected consequence of
   grafting a drafter trained against LiteRT's quantized forward onto our
   HF-fp16 forward, with no retraining. MTP_PATH_A_FINDINGS §6 already noted
   this at the hidden-state-distribution level; this investigation confirms
   the weight-level story: there is no secondary source of drafter weights in
   HF for Path A to have used as an anchor.
4. The HF release does **not** hide MTP behind a config flag that we missed.
   There is no `enable_mtp`, no `num_mtp_heads`, no conditional
   `lm_head_aux`. The weights are genuinely absent.

### 6.1 What this unblocks

- Path A as a weight-reuse strategy is **dead by construction** for HF-based
  targets. The only way to make Path A work is to retrain the drafter
  against our HF-fp16 forward (which collapses into Path C territory —
  already shelved for different reasons, see `MTP_PATH_C_FINDINGS.md`).
- Any future "just extract MTP from Google" plan should be killed at the
  meeting, not the retrospective.
- If a HF-side MTP release ever lands (watch `config.json` for
  `num_mtp_heads` or a new `lm_head_mtp_*` tensor prefix on future snapshots
  of `google/gemma-4-E2B*`), the math changes. Current snapshot
  `transformers_version: 5.5.0.dev0` has nothing.

### 6.2 What this does NOT change

- Path C's blocker (verify-chunk fp16 drift, item 11c) is orthogonal. Even
  if HF had shipped MTP heads, the verify-side numerical drift would still
  cap the achievable acceptance-per-cycle math below the break-even threshold
  on our current chunked target (see `PHASE_B_LIVE_ACCEPT_RATE_GAP.md`).
- The non-MTP hypotheses in `SURVIVING_HYPOTHESES.md` (GPU verify S0,
  chunk-merge S2, prefill bypass B1, runtime hints A2) are unaffected.

---

## 7. Sources

- HF model card — Gemma 4 E2B: https://huggingface.co/google/gemma-4-E2B
- HF model card — Gemma 4 E2B-it: https://huggingface.co/google/gemma-4-E2B-it
- HF model card — Gemma 3n E2B-it: https://huggingface.co/google/gemma-3n-E2B-it
- HF Gemma 4 launch blog: https://huggingface.co/blog/gemma4
- transformers Gemma3n docs: https://huggingface.co/docs/transformers/en/model_doc/gemma3n
- transformers source — gemma4 dir: https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4
- transformers source — gemma3n dir: https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma3n
- `modeling_gemma3n.py`: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma3n/modeling_gemma3n.py
- FlowHunt article with Google engineer quote: https://www.flowhunt.io/blog/gemma-4-released-without-mtp-multi-token-prediction/
- Our Path A findings (LiteRT drafter): `docs/MTP_PATH_A_FINDINGS.md`
- Our Path C findings (self-trained): `docs/MTP_PATH_C_FINDINGS.md`
- Surviving hypotheses context: `docs/SURVIVING_HYPOTHESES.md`
