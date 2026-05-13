# Gemma 4 Official MTP Drafter — Knowledge Base (2026-05-06)

Compiled 2026-05-06 from web sources released after Google's 2026-05-05
official MTP drafter announcement. Targets three downstream applications:

- `CoreML-LLM` (this repo) — iPhone/Mac ANE/GPU CoreML inference.
- `../llama.cpp` — GGUF / GGML CPU+GPU inference.
- `mlx-swift-lm` — MLX Swift on Apple Silicon.

Everything below is sourced; URLs at the bottom of each section. Numbers
that are not directly quoted from a primary source (Google blog, HF
config.json, vendor PR) are flagged as "community-reported".

---

## 0. TL;DR

- **Release:** 2026-05-05, Apache 2.0. Four sibling drafters:
  `google/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant`.
- **Architecture:** All four drafters are 4-layer Gemma 4 text decoders
  (`model_type=gemma4_assistant`, sub-config `gemma4_text`). Layer
  pattern `[sliding, sliding, sliding, full]`. **All 4 layers run with
  `is_kv_shared_layer=True`**: drafter has no KV cache of its own and
  reads K/V from the target's last sliding + last full layer.
- **First-class runtimes (vendor-supported):** transformers ≥ 5.8 (HF),
  vLLM (PR #41745), SGLang, MLX (mlx-vlm 0.4.5+), Ollama (v0.23.1),
  LiteRT-LM. Google ran benchmarks against all of these.
- **Not yet supported:** **llama.cpp** (issue #22337 + PR #22673 add
  generic MTP for Qwen 3.6, but Gemma 4 needs further changes).
  **mlx-swift-lm** (no Gemma 4 MTP wiring; mlx-vlm is the working
  Apple-Silicon path).
- **Headline speedups (vendor-reported, not all reproduced):**
  - vLLM H100 31B `γ=8`: **3.19×** (PR #41745 author bench).
  - vLLM H100 26B-A4B `γ=4`: **1.78×**; E4B `γ=4`: **1.78×**;
    E2B `γ=2`: **1.30×**.
  - DGX Spark 26B `γ=4`: 67-69 % accept, **2.34× sequential / 1.91×
    concurrent**, 175 vs 104 tok/s peak.
  - mlx-vlm M-series 26B-A4B `B=4`: **3.94×** (greedy, byte-identical).
  - mlx-vlm M-series 31B `B=4`: **2.29×**.
  - Ollama macOS 31B coding: "**>2×**" (vendor blurb).
  - HN community (vLLM, RTX 5090, 4-bit AWQ): 200+ tok/s on 26B.
  - HF transformers 31B Mac CPU fp32 capitals raw: **2.24×** /
    90 % accept (verified locally in this repo's
    `conversion/bench_hf_chat_template.py`).

---

## 1. Drafter architecture (verified from `config.json` of each repo)

All four configs share the same outer schema — only sizes vary.

### 1.1 Outer config (`Gemma4AssistantConfig`)

```jsonc
{
  "architectures": ["Gemma4AssistantForCausalLM"],
  "model_type": "gemma4_assistant",
  "backbone_hidden_size": <T>,        // = target.hidden_size
  "num_centroids": 2048,
  "centroid_intermediate_top_k": 32,
  "use_ordered_embeddings": <bool>,   // E2B/E4B true, 26B/31B false
  "tie_word_embeddings": true,
  "transformers_version": "5.7.0.dev0",
  "text_config": { ... gemma4_text ... }
}
```

Multimodal special tokens (`audio_token_id=258881`, `image_token_id=258880`,
boi/eoi/boa/eoa) are duplicated from the base for vocab alignment but
the drafter never embeds image/audio data — it consumes token IDs only.

### 1.2 Inner `text_config` (`gemma4_text` / `Gemma4TextConfig`)

| Field | E2B | E4B | 26B-A4B | 31B |
|---|---|---|---|---|
| `num_hidden_layers` | 4 | 4 | 4 | 4 |
| `hidden_size` (drafter d) | 256 | 256 | 1024 | 1024 |
| `intermediate_size` | 2048 | 2048 | 8192 | 8192 |
| `num_attention_heads` | 4 | 4 | 16 | 32 |
| `num_key_value_heads` | 1 | 2 | 8 | 16 |
| `num_global_key_value_heads` | null | null | 2 | 4 |
| `head_dim` | 256 | 256 | 256 | 256 |
| `global_head_dim` | 512 | 512 | 512 | 512 |
| `attention_k_eq_v` | **false** | **false** | **true** | **true** |
| `sliding_window` | 512 | 512 | 1024 | 1024 |
| `max_position_embeddings` | 131072 | 131072 | 262144 | 262144 |
| `layer_types` | `[swa,swa,swa,full]` | same | same | same |
| `num_kv_shared_layers` | 4 | 4 | 4 | 4 |
| `final_logit_softcapping` | null | null | null | null |
| `rms_norm_eps` | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| `tie_word_embeddings` | true | true | true | true |
| `vocab_size` | 262144 | 262144 | 262144 | 262144 |
| `backbone_hidden_size` (target) | 1536 | 2560 | 2816 | 5376 |
| `use_ordered_embeddings` | true | true | false | false |

RoPE per layer:

```jsonc
"rope_parameters": {
  "full_attention":    { "partial_rotary_factor": 0.25,
                         "rope_theta": 1e6,  "rope_type": "proportional" },
  "sliding_attention": { "rope_theta": 1e4,  "rope_type": "default" }
}
```

Activation: `gelu_pytorch_tanh` (same as Gemma 4 base).

### 1.3 What the architecture means in practice

1. **Token-only API.** Inputs are `input_ids`, `position_ids`,
   `target_hidden`, `past_key_values`. Output is logits over 262 144
   tokens, plus updated `target_hidden` carry. No raw image/audio
   embeddings are consumed.
2. **No own KV cache.** Every drafter layer sets
   `is_kv_shared_layer=True`. K and V are pulled from the **target's
   last full-attention layer** (one global KV) and the **target's last
   sliding layer** (one local KV). The drafter only maintains the
   per-burst Q for its 4 layers.
3. **Per-step recurrent state = `last_hidden_state`.** Between
   draft steps, the drafter receives the previous step's
   final-norm hidden state from the target (or its own L4 hidden on
   subsequent draft steps within a burst).
4. **Input concat then down-project.** Step input is
   `[target_embed(last_token), last_hidden] ∈ R^{2T}`, projected by
   `pre_projection` Linear to `R^{d}` (d = drafter hidden).
5. **Constant `position_ids`.** Per HF docs and
   `SinglePositionMultiTokenCandidateGenerator`, the drafter is
   "locked into a constant position_ids value" set once at the start of
   each draft burst. *(Empirical caveat from this repo's Mac bench —
   per-step `pos+k` was slightly better on CoreML, but HF reference
   uses constant. Possibly an ANE numerics quirk.)*
6. **`layer_scalar` and `masked_embedding` (E2B/E4B only).**
   - `layer_scalar`: per-layer learned scalar applied at the very end
     of each `Gemma4TextDecoderLayer`
     (`hidden_states *= self.layer_scalar`). E2B observed values
     L0=0.042, L1=0.246, L2=0.426, L3=0.120 (this repo, source-read).
   - `masked_embedding` (a.k.a. `Gemma4AssistantMaskedEmbedder`): a
     `(num_centroids=2048, vocab_per_centroid=128)` clustering layer
     that scores 2048 centroids, picks `top_k=32` (~4096 of 262 144
     tokens), and scatters those logits back into the full vocab tensor
     with `min(selected) - 1` masking. Cuts LM-head FLOPs by ~45×
     vs full vocab. **`use_ordered_embeddings=true` is the flag that
     enables this path; 26B/31B run the dense tied lm_head instead.**
7. **`attention_k_eq_v` flag.** 26B/31B drafters fold K and V into the
   same projection (saves params). E2B/E4B keep K and V separate.
8. **Bigger drafter for bigger target.** Drafter hidden goes from 256
   (E2B/E4B) to 1024 (26B/31B); drafter intermediate from 2048 to 8192.
   Target hidden ratios: drafter / target = 256/1536 (16.7 %) for E2B,
   256/2560 (10.0 %) for E4B, 1024/2816 (36.4 %) for 26B-A4B,
   1024/5376 (19.0 %) for 31B.

### 1.4 Drafter parameter counts (vendor model cards)

| Drafter repo | params | bf16 | int4 (palettized) |
|---|---|---|---|
| `gemma-4-E2B-it-assistant` | 78 M | 156 MB | ~40 MB |
| `gemma-4-E4B-it-assistant` | 78.8 M | 158 MB | ~40 MB |
| `gemma-4-26B-A4B-it-assistant` | ~0.4 B | ~840 MB | ~210 MB |
| `gemma-4-31B-it-assistant` | ~0.5 B | ~940 MB | ~235 MB |

Sources: model cards on HF, mlx-community quantized counterparts.

---

## 2. Hugging Face transformers reference path

This is the only fully-documented integration path; treat it as the
ground-truth oracle.

### 2.1 Required versions

- `transformers >= 5.7.0` registers `gemma4_assistant`. The released
  5.7.0 ships with a **regression** on assistant loading on some
  paths; 5.8.0+ recommended. NVIDIA forum verified PR-bundled **dev
  build of 5.8.0** is the working combo.
- `torch >= 2.5`, `safetensors`, `huggingface-hub`.

### 2.2 Loading + generation (canonical)

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

TARGET = "google/gemma-4-E2B-it"
ASSIST = TARGET + "-assistant"

processor = AutoProcessor.from_pretrained(TARGET)
target = AutoModelForCausalLM.from_pretrained(
    TARGET, torch_dtype=torch.bfloat16, device_map="auto"
)
assistant = AutoModelForCausalLM.from_pretrained(
    ASSIST, torch_dtype=torch.bfloat16, device_map="auto"
)

# Heuristic schedule: +2 on full accept, -1 on any reject.
assistant.generation_config.num_assistant_tokens = 4
assistant.generation_config.num_assistant_tokens_schedule = "heuristic"

inputs = processor(text=prompt, return_tensors="pt").to(target.device)
out = target.generate(
    **inputs,
    assistant_model=assistant,
    max_new_tokens=256,
    do_sample=False,
)
```

### 2.3 Internals — read these source files for canonical behaviour

- `src/transformers/generation/candidate_generator.py`,
  `SinglePositionMultiTokenCandidateGenerator.get_candidates` (~L1357
  in 5.7.x). Key facts:
  - `position_ids` is set **once** at the start of each draft burst
    (constant across all K draft steps).
  - Carry seed = `model_outputs.hidden_states[-1]`, i.e. the target's
    **post-final-norm hidden** for the most recent committed token.
  - On reject, seed re-forms from the latest accepted target hidden.
- `src/transformers/models/gemma4/modeling_gemma4.py` —
  `Gemma4ScaledWordEmbedding` applies `embed.weight * sqrt(hidden_size)`
  inside the embedding lookup. **Drafter expects the scaled embed,
  not the raw row.** (This repo's CoreML port hit this as bug #2.)
- `src/transformers/models/gemma4_assistant/modeling_gemma4_assistant.py`
  — `Gemma4AssistantMaskedEmbedder` (centroid lm-head),
  `Gemma4AssistantModel` (KV-shared decoder),
  `Gemma4AssistantForCausalLM` (top-level wrapper).

### 2.4 Tuning

- `num_assistant_tokens`: 2–4 for E2B/E4B targets, 4–8 for 26B-A4B/31B.
- `num_assistant_tokens_schedule = "heuristic"` is recommended by HF;
  dynamic adjustment improved bench numbers in HN-reported runs.
- `do_sample=False` gives byte-identical output. With sampling, the
  rejection-sampling correction (Leviathan-Chen) preserves the same
  distribution, but accept rate drops ~30-50 % at T=0.7+.

---

## 3. Runtime support matrix (2026-05-06)

| Runtime | Status | Tooling | Speedup (vendor) | Notes |
|---|---|---|---|---|
| HF transformers | ✅ shipped | `target.generate(assistant_model=...)` | 1.5-2× CPU/GPU | requires ≥5.8 dev |
| vLLM | 🟡 PR open #41745 | `--speculative-config '{"model": ..., "method":"gemma4_mtp", "num_speculative_tokens": N}'` | E2B 1.30×, E4B 1.78×, 31B 3.19× (H100, γ=2/4/8) | needs PR cherry-pick + transformers 5.8 dev; auto-MoE-backend (Marlin) |
| SGLang | 🟡 announced, no public Gemma 4 example | EAGLE-style flags `--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens N` | n/a | docs not Gemma-4-specific yet |
| MLX (mlx-vlm) | ✅ shipped 0.4.5 | `--draft-model … --draft-kind mtp --draft-block-size {3,6}` | 26B-A4B 3.94× B=4, 31B 2.29× B=4, byte-identical at T=0 | uses `gg-hf-am/...` mlx-community quantized variants too |
| Ollama | ✅ shipped v0.23.1 (macOS only, MLX runner) | `ollama run gemma4:31b-coding-mtp-bf16` | "over 2×" 31B coding | mac-only via MLX; supports `ollama create` with safetensors drafter and Modelfile `DRAFT` |
| LiteRT-LM | ✅ shipped v0.10.1 | bundled in Google AI Edge Gallery (Android/iOS) | (vendor only) | Google's reference on-device runtime |
| llama.cpp | ❌ not yet | PR #22673 adds generic MTP for Qwen 3.6; Gemma 4 path needs more work. Issue #22337 currently fails to load `gemma-4-E2B-it` even as a plain draft (out-of-bounds in `load_tensors`) due to per-layer-embeddings tensor shape mismatch with 31B target | Qwen 3.6 27B: 1.85× (PR author bench) | community workaround uses 277 MB EAGLE3 head, 1.72× claim |
| mistral.rs | ✅ shipped (general Gemma 4) | day-0 install script | n/a | MTP-specific support not advertised |
| LM Studio | UI-stub only | UI exposes spec-decoding option | non-functional in current build (HN comment) | wait for next LM Studio release |
| MTPLX (Apple Silicon, custom) | ✅ shipped | `mlx-mtplx-0.31.2-qmm` fork | Qwen 3.6 27B 2.24× M5 Max | **does not support Gemma 4** as of 2026-05-06; uses model's own MTP heads, not external assistant; useful as KV-rollback / verify-loop reference |
| SwiftLM | ⚠ no Gemma 4 MTP yet | `--draft-model --num-draft-tokens` for general spec-decoding | n/a | Mamba/Qwen 3.5 focused; iOS app (SwiftBuddy) does not expose drafting |

---

## 4. Cross-runtime alignment gotchas (verified)

The same alignment bugs appear across implementations. If your numbers
are off, check these in order:

1. **`embed_tokens.weight * sqrt(hidden_size)` scaling** — the drafter
   expects the *scaled* token embedding (`Gemma4ScaledWordEmbedding`),
   not the raw lookup row. CoreML port: `lookupRawEmbed` vs
   `embedToken` distinction. mlx-vlm and vLLM PR fold this inside the
   `pre_projection` linear during loading.
2. **Carry seed for the first burst** — must be the target's
   **post-final-norm hidden state** for the most recent committed
   token (`model_outputs.hidden_states[-1]`). Zero-init halves accept
   rate. CoreML port pulls this from `chunk4.hidden_states_out`.
3. **Constant vs per-step `position_ids`** — HF docs and reference
   code use constant, locked to the bonus-token's absolute position.
   This repo's empirical Mac bench preferred per-step `pos+k`; vLLM PR
   uses constant; mlx-vlm uses constant. Constant is the
   training-aligned default.
4. **Partial rotary on full-attention layer** — Gemma 4 base spec is
   `partial_rotary_factor=0.25` `rope_type=proportional` on full
   layers (only the first 25 % of head dims rotated;
   `head_dim=256 → 64 rotated + 192 nope`). Older builds that emit
   full-rotary `cos_full.npy` will produce K caches that the drafter
   was *not* trained against. Symptom: K-cache cosine drops to
   ~0.59-0.78 at chat-template boundary tokens; argmax still passes
   but accept rate halves.
5. **K = V alias on global layers (26B/31B only)** —
   `attention_k_eq_v=true` on these two drafters. The corresponding
   target's global layers must expose K=V to keep the KV-share
   pointer correct. **E2B/E4B run with K≠V**, so do not aliasing-
   "optimise" their target's global K cache.
6. **Final logit softcapping disabled** — `final_logit_softcapping:
   null`. Prior LiteRT artefacts had `tanh(x/30)*30` on output;
   removing it is required for HF parity.
7. **Sliding-attention mask flip** — HF
   `gemma4_assistant.create_attention_masks` does
   `sliding_attention_mask.flip(dims=(1,))`. CoreML and Metal
   implementations that hand-build a right-aligned causal SWA mask
   without flipping will be subtly wrong on the SWA layers.
8. **Centroid lm-head (E2B/E4B only).** Bypassing the centroid path
   for full-vocab lm-head works at T=0 (top-1 usually agrees with the
   centroid path), but at higher temps you'll see distribution drift
   on the rare path where the drafter's true top token isn't in the
   top-32 selected centroids.

---

## 5. Per-target integration recipes

### 5.1 CoreML-LLM (this repo)

State as of 2026-05-06: PyTorch port + ANE-friendly mlpackage are
**bit-faithful to HF reference** (`logits cosine 1.000005`). End-to-end
Mac CoreML net-positive at `K_USE=2` on chat-templated structured
prompts (+5.8 % capitals, +2.2 % code), neutral-to-slightly-negative on
free-form. iPhone push deferred — see
`docs/SESSION_2026_05_06_MTP_MAC.md` for the residual K-cache cosine
0.59-0.78 chat-boundary issue.

What's already done:
- `conversion/mtp_drafter_model.py` — full HF-faithful PyTorch port
- `conversion/build_mtp_drafter.py` — Conv2d-rewrite + INT4 palettize
- `conversion/test_mtp_parity.py --full-lm-head` — proven equivalent
- `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` — drop-in drafter
  consumer, per-step `pos+k` retained, env knobs documented in handoff
- `conversion/replay_capture_through_{port,coreml}.py` — bisection
  harness for any future regression

What needs doing (in order):
1. **Fix K-cache numerical drift** in chunked target build to match
   HF reference (RoPE order / RMSNorm convention / fp16 accumulation
   audit). Accept rate is currently capped at HF's chat-templated
   ceiling because of this.
2. **Build a 3-chunk multifunction stateful bundle with `verify_qK`
   K=3** to unlock `Gemma4StatefulEngine` baseline (34.6 tok/s).
3. **Centroid lm-head port** if T>0 sampling shows distribution
   drift on E2B/E4B.
4. **iPhone empirical bench** once 1+2 land — gating ≥0.50 rolling
   accept on 20-prompt SpecBench corpus.

### 5.2 llama.cpp (`../llama.cpp`)

**Currently blocked.** Two-step fix path likely:

1. **Land PR #22673** ("MTP Support" by `am17an`). It already adds
   generic MTP framework — `override_arch` switch, partial-tensor
   loading via `qwen35.nextn_predict_layers`, hidden-state hook after
   each ubatch, recurrent-state rollback for up to `draft_max` tokens
   (`llama_memory_recurrent` / `n_rs_seq`).
2. **Add `gemma4_assistant` arch on top.** Gemma 4-specific deltas:
   - 4 separate decoder layers (vs Qwen's "head-on-base"
     `nextn_predict_layers` count — Gemma 4's drafter is a separate
     bundle).
   - `is_kv_shared_layer=True` for all 4 layers — wire the drafter's
     K/V tensors to the target's last sliding (L13 in E2B target
     numbering) and last full (L14) layer.
   - `gemma4_assistant.pre_projection` (`R^{2T} → R^{d}`) and
     `post_projection` (`R^{d} → R^{T}`) on the carry path.
   - `Gemma4AssistantMaskedEmbedder` for E2B/E4B (centroid 2048 ×
     128, top-32 selection, scatter back to vocab=262144).
   - Partial-rotary factor 0.25 on full layer (Gemma 4 base spec —
     llama.cpp may already have this for base).
3. **Fix issue #22337** as a prerequisite: the per-layer-embeddings
   tensor shape mismatch (E2B/E4B has `n_embd_per_layer > 0`, 31B has
   0) currently OOBs on `load_tensors` even for plain draft loading.

Workaround until 1+2 ship: community 277 MB EAGLE3 draft head (1.72×
claim, lossy) — see `groundy.com` LiteRT comparison.

### 5.3 mlx-swift-lm

No native Gemma 4 MTP support yet. Two practical paths:

1. **Adopt mlx-vlm conventions** (Python). Production-ready today via:
   ```
   pip install -U mlx-vlm
   python -m mlx_vlm generate \
       --model mlx-community/gemma-4-31B-it-bf16 \
       --draft-model mlx-community/gemma-4-31B-it-assistant-bf16 \
       --draft-kind mtp --draft-block-size 6 \
       --prompt "..." --max-tokens 256 --temperature 0
   ```
   Quantized BF16 packs are 156 MB / 158 MB / 839 MB / 939 MB.
   Public `load_drafter(repo, kind="mtp")` API.

2. **Port the mlx-vlm `gemma4_assistant` drafter to mlx-swift-lm.**
   Reference module:
   `mlx_vlm/speculative/drafters/gemma4_assistant/` in
   `Blaizzy/mlx-vlm` (note: 0.4.5+, also see issue #981). Key Swift
   pieces to author:
   - `Gemma4AssistantConfig` decoder (mirror the four-variant
     config table above).
   - `Gemma4AssistantMaskedEmbedder` (centroid 2048 × 128, top-32,
     scatter-min mask).
   - `Gemma4AssistantDecoderLayer` (4 layers, KV-shared, layer_scalar).
   - Speculative loop: constant `position_ids`, carry =
     post-final-norm target hidden, leviathan rejection at T>0.
   - Verify against `mlx-vlm`'s byte-identical T=0 output as the
     parity oracle.

3. **MTPLX cross-pollination.** `youssofal/MTPLX` is the
   highest-perf MLX MTP code today (2.24× on Qwen 3.6 27B M5 Max
   verified) and includes a custom Metal kernel `linear-gdn-from-conv-tape`
   for innovation-tape replay on rollback. **Does not yet support
   Gemma 4** but the verify-loop / KV-rollback design is reusable.

---

## 6. Open community problems (worth tracking)

1. **Chat-template halves accept on raw HF runtime.** Multiple
   reports + this repo verified 90 %→33 % drop when chat template is
   applied. Root cause unknown — likely the heavy `<start_of_turn>`
   token sequence is lower entropy than the drafter expects. No fix
   in any runtime as of 2026-05-06; we work around it by gating
   speculation behind rolling accept ≥ 0.50.
2. **Pre-fill speedup on llama.cpp PR #22673 = 0.51×** (regression).
   Author noted, fix in progress.
3. **vLLM PR #41745 `intermediate_size` config nesting** —
   `intermediate_size` lives under `text_config`, not the outer
   `Gemma4AssistantConfig`. Earlier PR commits OOB'd; latest commits
   fixed.
4. **Quant-config inheritance.** Drafter ships unquantized BF16. If
   the target uses NVFP4, do NOT inherit quant_config into the
   drafter — pass `quant_config=None` for assistant load. (vLLM PR
   author hit this.)
5. **mlx-vlm requires `mlx-vlm` 0.4.5+, not bare `mlx-lm`.** Plain
   `mlx-lm` only does generic draft-model spec-decoding (separate
   from MTP). The drafter packages on HF
   (`mlx-community/...-assistant-bf16`) explicitly target mlx-vlm.

---

## 7. Verified benchmark numbers (sources cited)

### 7.1 Vendor / first-party

| Setup | Drafter | Hardware | Number | Source |
|---|---|---|---|---|
| Gemma 4 31B coding | E2B-equivalent? | NVIDIA RTX PRO 6000 | "**~3×**" upper bound | Google blog |
| Gemma 4 26B-A4B | 26B-A4B-it-assistant | NVIDIA RTX PRO 6000 | "**half the wait time**" (≈2×) | Google blog |
| vLLM E2B γ=2 | E2B-it-assistant | H100 | **1.30×** | PR #41745 |
| vLLM E4B γ=4 | E4B-it-assistant | H100 | **1.78×** | PR #41745 |
| vLLM 31B γ=8 | 31B-it-assistant | H100 | **3.19×** | PR #41745 |
| vLLM 26B-A4B γ=4 TP=1 | 26B-A4B-it-assistant | DGX Spark | **2.34× seq / 1.91× concur**; mean accept length 3.68/4 (67-69 %); peak 175 tok/s vs 104 tok/s baseline | NVIDIA forum (hospedales) |
| mlx-vlm 26B-A4B B=4 | 26B-A4B-it-assistant-bf16 | M-series | **3.94×** byte-identical T=0 | mlx-community card |
| mlx-vlm 31B B=4 | 31B-it-assistant-bf16 | M-series | **2.29×** | mlx-community card |
| Ollama 31B coding | 31b-coding-mtp-bf16 | macOS | "**over 2×**" | Ollama v0.23.1 notes |
| Transformers 31B (H100, 128 tok) | 31B-it-assistant | H100 | **22.05 vs 11.43 tok/s** (≈1.93×) | NVIDIA forum |

### 7.2 Community / HN (single-user, mark as anecdotal)

| Setup | Number | Source |
|---|---|---|
| vLLM RTX 5090 26B AWQ-4 | 200+ tok/s | HN #48024540 |
| Ollama M1 Max | 54 tok/s | HN #48024540 |
| Ollama Galaxy Z Fold 7 | 21 → 37.8 tok/s (1.8×) | HN #48024540 |
| Qwen 3.6 27B Q8 (community-trained MTP, not Gemma 4) | 20 → 46 tok/s (2.3×) | HN #48024540 |

### 7.3 This repo (verified locally on M3 Ultra)

| Setup | Number | File |
|---|---|---|
| HF target Mac CPU fp32, capitals raw | **2.24× / 90 % accept** | `conversion/bench_hf_assistant.py` |
| HF target Mac CPU fp32, capitals chat-templated | 1.09× / 33 % accept | same |
| HF target Mac CPU fp32, code (chat) | 1.19× / 47 % accept | `conversion/bench_hf_chat_template_v2.py` |
| Swift CoreML K_USE=2, capitals | +5.8 % / 0.09 accept | `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` |
| Swift CoreML K_USE=2, code | +2.2 % / 0.06 accept | same |
| Swift CoreML K_USE=2, essay | -10 % / 0.02 accept (fallback) | same |

---

## 8. Quick links (verified 2026-05-06)

### 8.1 Primary

- Google blog: <https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/>
- HF MTP transformers docs: <https://ai.google.dev/gemma/docs/mtp/mtp>
- HF Gemma 4 transformers reference: <https://huggingface.co/blog/gemma4>

### 8.2 Drafter HF repos

- E2B: <https://huggingface.co/google/gemma-4-E2B-it-assistant>
- E4B: <https://huggingface.co/google/gemma-4-E4B-it-assistant>
- 26B-A4B: <https://huggingface.co/google/gemma-4-26B-A4B-it-assistant>
- 31B: <https://huggingface.co/google/gemma-4-31B-it-assistant>
- mlx-community 26B-A4B: <https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-assistant-bf16>
- mlx-community 31B: <https://huggingface.co/mlx-community/gemma-4-31B-it-assistant-bf16>

### 8.3 Vendor PRs / issues

- vLLM PR #41745 (Gemma 4 MTP): <https://github.com/vllm-project/vllm/pull/41745>
- vLLM Gemma 4 recipe: <https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md>
- llama.cpp issue #22337 (E2B draft load fail): <https://github.com/ggml-org/llama.cpp/issues/22337>
- llama.cpp PR #22673 (generic MTP for Qwen 3.6): <https://github.com/ggml-org/llama.cpp/pull/22673>
- llama.cpp discussion #21975 (split-mode + spec-decoding): <https://github.com/ggml-org/llama.cpp/discussions/21975>
- llama.cpp speculative.md: <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md>
- mlx-vlm gemma4_assistant drafter README: <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/gemma4_assistant/README.md>
- mlx-vlm issue #981 (server-side spec-decoding): <https://github.com/Blaizzy/mlx-vlm/issues/981>
- Ollama v0.23.1 release: <https://github.com/ollama/ollama/releases/tag/v0.23.1>
- Ollama Gemma 4 31B coding MTP tag: <https://ollama.com/library/gemma4:31b-coding-mtp-bf16>
- MTPLX (Apple Silicon native MTP, Qwen 3.6 only today): <https://github.com/youssofal/MTPLX>
- SwiftLM: <https://github.com/SharpAI/SwiftLM>

### 8.4 Community analysis

- HN thread: <https://news.ycombinator.com/item?id=48024540>
- claypier write-up: <https://claypier.com/en/gemma-4-mtp-drafter-launch/>
- conzit summary: <https://conzit.com/post/gemma-4-speeds-up-ai-with-multi-token-prediction-drafters>
- Maarten Grootendorst visual guide: <https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4>
- Groundy LiteRT-LM vs llama.cpp gap: <https://groundy.com/articles/litert-lm-v0101-ships-gemma-4-mtp-heads-that-llamacpp-cant-access/>
- NVIDIA DGX Spark recipe (working vLLM bench): <https://forums.developer.nvidia.com/t/gemma-4-mtp/369123>
- NVIDIA DGX Spark thread (draft model availability): <https://forums.developer.nvidia.com/t/gemma4-draft-models-are-now-available/369114>
- aiproductivity.ai (community-found-it-first narrative): <https://aiproductivity.ai/news/gemma-4-multi-token-prediction-mtp-undocumented/>
- Benjamin Marie pre-release wishlist (X): <https://x.com/bnjmn_marie/status/2041414272031261154>

### 8.5 This repo's prior work

- Cross-device handoff: `docs/MTP_GEMMA4_OFFICIAL_HANDOFF.md`
- Mac bench session log: `docs/SESSION_2026_05_06_MTP_MAC.md`
- Speculative survey: `docs/SPECULATIVE_DECODING_SURVEY.md`
- Old (now superseded) retreat verdict: `docs/MTP_INVESTIGATION_SUMMARY.md`

---

## 9. Action items prioritised for our 3 stacks

| # | Stack | Priority | Action | Blocker |
|---|---|---|---|---|
| 1 | CoreML-LLM | P0 | Audit chunked target K-cache vs HF reference (RoPE order, RMSNorm, fp16 accumulation) | none — has bisection harness |
| 2 | CoreML-LLM | P1 | Re-bench K_USE=3 with fixed K-cache; iPhone push if rolling accept ≥ 0.50 | depends on #1 |
| 3 | CoreML-LLM | P2 | Port `Gemma4AssistantMaskedEmbedder` (centroid path) for E2B/E4B sampling correctness | T=0 ships without |
| 4 | llama.cpp | P0 | Land PR #22673 + add `gemma4_assistant` arch on top | merger upstream |
| 5 | llama.cpp | P0 | Fix issue #22337 (per-layer-embed OOB) as prerequisite | upstream load_tensors path |
| 6 | mlx-swift-lm | P1 | Bring mlx-vlm's `gemma4_assistant` drafter into Swift; cross-validate against mlx-vlm Python at T=0 | none |
| 7 | mlx-swift-lm | P2 | Inspect MTPLX's KV-rollback / innovation-tape Metal kernel for sampling path | n/a until P1 lands |
| All | All | P3 | Document the chat-template accept-rate halving as a known issue; gate speculation behind rolling accept | community-wide |

---

*Compiled 2026-05-06. Will rot fast — re-fetch URLs in §8 every 1-2
weeks. Keep this file in `docs/` rather than auto-memory.*
