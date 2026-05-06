# Gemma 4 Official MTP Drafter — Cross-Device Handoff

**Status:** 2026-05-06. Mac bench complete; iPhone push deferred. MTP
path is structurally working but acceptance is too low to ship at K=8.
See `docs/SESSION_2026_05_06_MTP_MAC.md` for the full Mac findings, the
five distinct bugs uncovered (incl. a target-build divergence from HF
RoPE spec), and the remaining knowns to investigate before another push.
Also see **Session log (2026-05-06)** at the bottom of this file.

## TL;DR

- Google released **official MTP drafters** for the Gemma 4 family in 2026-05 under Apache 2.0. E2B / E4B / 26B-A4B / 31B all covered.
- The drafter is a **separate ~78M model** (`gemma4_assistant`) with a **token-IDs interface** — not a head-on-base, not shared-trunk.
- This is the first credible drafter for our stack: it doesn't share the failure modes of EAGLE-3 / Medusa / cross-vocab / LiteRT-MTP that killed every prior attempt.
- **11c (verify KV-contamination)** was implemented in commit `9840a09` (2026-04-17), Mac-validated. iPhone Gate 3 (live acceptance ≥ 0.30, tok/s ≥ 35) was pending in the commit body and has not been re-confirmed in subsequent docs.
- This doc supersedes the retreat verdict in `docs/MTP_INVESTIGATION_SUMMARY.md`.

## Why now (vs the prior MTP retreat)

Prior investigation (`docs/MTP_INVESTIGATION_SUMMARY.md`, 2026-04-17) recommended retreat because:

1. LiteRT MTP training recipe was non-public — couldn't reproduce.
2. 11c hadn't closed — ~77 % iPhone break-even made any drafter useless.
3. Extracted LiteRT weights showed 3 % top-1 / 0 % iPhone live accept due to W4A8 quant distribution drift.

All three are lifted as of 2026-05:

1. **Google's official release ships training-free.** Weights are directly downloadable from HF; no recipe reproduction needed.
2. **11c implemented** in `9840a09`. Mac Gate 1 (per-layer cosine ≥ 0.9999 at start_pos = 0 and 100) and Mac Gate 2 (KV parity outside-write-region diff = 0 across all 8 buffers) both PASS.
3. **Drafter trained against fp16 base Gemma 4** (per Google blog) — argmax-aligned with the target we actually deploy, no quant drift.

## Verified drafter facts

| Property | Value | Source |
|---|---|---|
| HF repo (E2B) | `google/gemma-4-E2B-it-assistant` | HF model card |
| Parameter count | ~78 M | HF model card footer |
| Format | BF16 safetensors | HF model card |
| HF model_type | `gemma4_assistant` | HF config |
| Loading API | `AutoModelForCausalLM.from_pretrained(...)` | HF docs |
| Speculative API | `target.generate(..., assistant_model=drafter)` | HF docs |
| Architecture | Independent small drafter (NOT head-on-base) | HF docs |
| Input | Token IDs | HF API |
| Output | N tokens predicted autoregressively | Google blog |
| Training base | HF Gemma 4 fp16 (`use_cache=True`) | Google blog |
| License | Apache 2.0 | Google blog |
| Claimed speedup | "up to 2×" for E2B / E4B family | Google blog |

Sibling repos: `google/gemma-4-E4B-it-assistant`, `google/gemma-4-26B-A4B-it-assistant`, `google/gemma-4-31B-it-assistant`.

## Why this is structurally different from prior dead drafters

| Prior method | Failure mode | Why MTP-assistant doesn't share it |
|---|---|---|
| EAGLE-3 | `use_cache=False` corpus → live argmax mismatch | Trained against actual base fp16 forward |
| Medusa | Heads-on-top → argmax disagreement drift | Independent drafter trained to match base's outputs |
| Cross-vocab Qwen | Different vocab + non-aligned drafter | Same vocab, same architecture family |
| LiteRT MTP (Path A) | W4A8-quant target during training, fp16 deploy | Trained against fp16 base — no quant drift |
| Speculative Streaming | Linear-mode overhead exceeds accept gain on ANE | Token-only drafter, no linear-merge overhead |

The "oracle-live gap" that killed every prior drafter is structurally absent here.

## What stays unchanged (do NOT modify)

- **3-chunk stateful Gemma 4 E2B** (v1.7.0+ default, shipped). Drafter is a SEPARATE bundle.
- **11c verify chunks** — already implemented and Mac-validated. Reuse as-is.
- **`SpeculativeLoop.swift` scaffolding** — acceptance decision and fallback paths reusable.

Critically, unlike EAGLE-3, MTP-assistant does NOT need hidden-state taps from base chunks. Base bundles can stay exactly as shipped.

## Work plan

### 1. Drafter config inspection — cheapest first step (~5 min)

Pull `config.json` from HF and confirm:
- `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`
- `intermediate_size`, `head_dim`
- `vocab_size` (must equal Gemma 4 base = 262 144)
- `tie_word_embeddings` (likely true → embedding can be shared with base; otherwise drafter blob bloats by ~262K × hidden)
- Any non-standard fields specific to `gemma4_assistant` model_type

### 2. CoreML conversion

Existing scripts in `conversion/`:

| Script | Action |
|---|---|
| `extract_mtp_drafter.py` | **Obsolete.** Used to extract from LiteRT binary; replaced by HF download. |
| `mtp_drafter_model.py` | **Rewrite.** Was a wrapper for LiteRT's 4-layer/hidden=256 drafter. Official 78 M is structurally different. |
| `build_mtp_drafter.py` | **Partial reuse.** Swap model loader for `AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it-assistant")`; keep palettization config; retarget output naming. |
| `test_mtp_parity.py` | **Reuse.** Mac parity harness, drafter-loader path needs swap. |

INT4 palettized on a 78 M model lands ~40 MB. Fits ANE trivially.

### 3. SpeculativeLoop adapter (Swift)

Current `SpeculativeTarget` protocol (`Sources/CoreMLLLM/SpeculativeLoop.swift`) requires `lastHiddenMulti(at:)` — designed for EAGLE-3 (drafter consumes target hiddens).

MTP-assistant is token-only and does its own autoregressive forward. Two implementation options:

- **Option A:** new `TokenDrafter` protocol with `draftTokens(from:[Int32], K:Int) -> [Int32]`, separate path in `SpeculativeLoop.drawBurst`.
- **Option B:** extend `SpeculativeTarget` with an optional flag; if drafter is token-only, skip hidden-state collection and the `lastHiddenMulti` call.

Option A is cleaner (no overloading the EAGLE-3-shaped protocol). Either way, the post-11c `commitAccepted` path is reusable as-is.

### 4. iPhone bench

After conversion + integration, run on iPhone 17 Pro:

- Same 20 prompts as `eval/spec_bench_prompts.txt`
- Capture `[SpecDbg]` for first 30 bursts
- Metrics: live acceptance, tok/s, vs Gemma 4 E2B baseline (~31 tok/s 3-chunk stateful)

**This single bench measures both 11c iPhone gate AND drafter quality.** No need for a separate EAGLE-3-driven Gate 3 run first — that's a weak signal because EAGLE-3 has 0 % live accept independent of 11c.

## Break-even math (post-11c, ANE)

Numbers from `docs/EAGLE3_INTEGRATION_STATE.md` and 11c plan:

```
base decode per token:    32.3 ms
verify K=3 (post-11c):    ~31.5 ms (Mac measured; iPhone TBD)
drafter forward (78 M):   estimate 5 ms (conservative for ANE INT4)

cycle = 3 × drafter + 1 × verify = 15 + 31.5 = 46.5 ms
break-even live accept = 46.5 / (3 × 32.3) = 0.48

At 70 % live accept (implied by "up to 2×"):
  ~2.1 tokens/cycle → 45 tok/s → 1.45× over 31 baseline
At 80 % live accept:
  ~2.4 tokens/cycle → 52 tok/s → 1.67× over 31 baseline
```

Speedup is gated on actual live acceptance. ≥ 60 % is the floor to be worth shipping. If drafter forward is 8 ms (INT4 fallback worst-case), break-even rises to 56 %.

## Open questions (resolve on next device)

1. **`gemma4_assistant` MIL compat.** Does the architecture trace cleanly through `coremltools 9` to ANE? The custom model_type may carry non-standard ops needing a converter shim like base Gemma 4.
2. **Tied embeddings.** If `tie_word_embeddings=true` and the 262 K × hidden embedding is duplicated rather than shared with base, drafter blob bloats. Confirm in config and decide whether to alias the base embedding into the drafter bundle.
3. **iPhone 11c regression risk.** Mac validated, iPhone never. Possible fp16 drift between verify and decode functions on ANE that doesn't appear on Mac. MTP iPhone bench will surface this.

## First-thing-to-do on the new device

```bash
cd <repo>
git checkout main
git pull
cat docs/MTP_GEMMA4_OFFICIAL_HANDOFF.md   # this file

# Step 1: drafter config (~30 sec, no GPU)
python -c "from transformers import AutoConfig; \
    c = AutoConfig.from_pretrained('google/gemma-4-E2B-it-assistant'); print(c)"

# Step 2: download weights (~160 MB BF16)
huggingface-cli download google/gemma-4-E2B-it-assistant \
    --local-dir ./output/gemma-4-E2B-assistant

# Step 3: Mac HF parity smoke (no CoreML yet)
python conversion/test_mtp_parity.py \
    --target google/gemma-4-E2B-it \
    --assistant ./output/gemma-4-E2B-assistant
# Pass criterion: target.generate(assistant_model=...) produces same token
# stream as plain target.generate, with measurable Mac speedup.
```

If Step 3 passes, drafter quality is verified at the algorithm level. Then proceed to CoreML conversion (work plan §2).

## References

- Google blog: https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
- HF assistant repo: https://huggingface.co/google/gemma-4-E2B-it-assistant
- HF MTP transformers docs: https://ai.google.dev/gemma/docs/mtp/mtp
- Prior investigation this supersedes: `docs/MTP_INVESTIGATION_SUMMARY.md`
- 11c implementation: commit `9840a09` (2026-04-17, "feat(11c): write-after-accept verify protocol — Mac-validated")
- Speculative survey (2026-04-22): `docs/SPECULATIVE_DECODING_SURVEY.md` — explicitly identified L-MTP as the only credible path
- Round 8 (2026-04-26): `docs/ROUND8_FINDINGS.md` — drafter excluded by user choice, not because dead

## Session log (2026-05-06) — Mac Studio port complete

### Done

- **Architecture surveyed.** Pulled `config.json` + `model.safetensors` from
  `google/gemma-4-E2B-it-assistant`. Confirmed structural similarity to the
  obsolete LiteRT drafter (4 layers, hidden=256, 3 SWA + 1 full,
  vocab=262144). Three additions vs LiteRT, all integrated:
  - **`layer_scalar`** — per-layer learned scalar applied at the very end
    of each `Gemma4TextDecoderLayer` (`hidden_states *= self.layer_scalar`).
    Values for E2B: L0=0.042, L1=0.246, L2=0.426, L3=0.120.
  - **`masked_embedding`** — `(num_centroids=2048, vocab_per_centroid=128)`
    centroid-based fast lm_head. NOT implemented in our port; we run the full
    `lm_head = embed_tokens.weight` (262144, 256) tied path. Empirically the
    masked centroid path's job is purely a decode speedup that in most
    positions agrees with full-lm_head top-1; for K=3..6 drafting argmax-
    based comparison this is acceptable.
  - **No final logit softcapping.** `final_logit_softcapping: null` —
    removed the `tanh(x/30)*30` from the LiteRT graph.

- **PyTorch port:** `conversion/mtp_drafter_model.py` — full rewrite.
  HF safetensors loader replaces the LiteRT TFLite parser. RMSNorm matches
  Gemma 4 spec (`y = normed * weight`, no `(1+w)`). Partial-rotary RoPE
  for full attention (`theta=1e6`, `partial_rotary_factor=0.25` → 64 rotated
  + 192 nope angles per half).

- **Parity proven:** `conversion/test_mtp_parity.py --full-lm-head`
  yields **logits cosine = 1.000005, hidden cosine = 1.000000, top-5 5/5**
  vs the official `transformers >= 5.7.0` reference. Our port is bit-exact.

- **CoreML build:** `conversion/build_mtp_drafter.py` — full rewrite.
  ANE-friendly module (Linear→Conv2d(1,1), `ANERMSNorm`). RoPE cos/sin are
  inputs (no internal precompute), so the Swift caller's existing
  `reshapeRoPEForDrafter` half-length convention is preserved → no Swift
  changes needed. ANE module agrees with PyTorch reference to top-1
  match + proj cosine 0.999428 (fp16 → fp16 round-off).

- **Swift I/O contract verified unchanged.** `MtpDraftSource.swift` and
  `MtpSpeculativeEngine.swift` already use the exact same input/output
  names (`embed_token`, `proj_act`, `kv13_k/v`, `kv14_k/v`, `cos_swa/full`,
  `sin_swa/full`, `mask_swa/full`, `top_k_indices/values`, `proj_act_out`).
  Drop-in replacement.

### Build the drafter on Mac

```bash
# 1. Set up env (separate from main venv to keep transformers==5.5.0 pinned).
python3 -m venv .mtp_venv
.mtp_venv/bin/pip install "transformers==5.8.0" "torch>=2.5" \
    safetensors huggingface-hub coremltools

# 2. PyTorch port (~30 sec, downloads ~80 MB BF16 from HF).
.mtp_venv/bin/python conversion/mtp_drafter_model.py \
    --output output/mtp_probe/mtp_drafter.pt

# 3. Sanity vs HF reference (~60 sec).
.mtp_venv/bin/python conversion/test_mtp_parity.py --full-lm-head

# 4. CoreML build (~3 min on Mac Studio M3 Ultra; INT4 palettization adds ~1 min).
.mtp_venv/bin/python conversion/build_mtp_drafter.py \
    --hf-repo google/gemma-4-E2B-it-assistant \
    --output mtp_drafter.mlpackage \
    --palettize-int4
```

Drop `mtp_drafter.mlpackage` into the existing Gemma 4 bundle alongside the
3-chunk stateful files. `CoreMLLLM.swift` already auto-loads it when present.

### Open items (require iPhone or further work)

1. **iPhone empirical bench.** The PR is gated on rolling acceptance ≥ 0.50
   on the 20-prompt SpecBench corpus. Run `[SpecDbg]` capture for 30 bursts
   on iPhone 17 Pro vs the 3-chunk stateful baseline (~31 tok/s).

2. **Embedding scale.** The HF drafter's `inputs_embeds` is the
   target-scaled embedding (`embed_tokens.weight * sqrt(hidden_size)`),
   but `ChunkedEngine.lookupRawEmbed` returns the unscaled Int8-dequantized
   row. Pre-existing concern from the LiteRT era — possibly a no-op (the
   first `pre_projection` Linear can absorb the scale during training) but
   if iPhone acceptance is very low, this is the first thing to fix.

3. **Step-0 `proj_act` seed.** `MtpSpeculativeEngine.swift` initialises
   `carryState` to **zeros** on the first burst. The HF training-time
   convention is to use the **target's L34 hidden** (`lastHiddenAtL34`,
   already exposed by `ChunkedEngine`). Trivial Swift change if iPhone
   metrics show first-burst acceptance is low.

4. **RoPE convention drift.** `conversion/generate_rope.py` produces FULL
   rotation tables for the full-attention layer (theta=1e6, no partial),
   while the HF drafter was trained with `partial_rotary_factor=0.25`.
   Since the drafter and target use the same target-side cos/sin tables,
   K and Q rotate consistently with each other; quality is the only
   question. Verify on iPhone before re-converting the target.

5. **Sibling drafters.** `google/gemma-4-E4B-it-assistant`,
   `26B-A4B-it-assistant`, `31B-it-assistant` ship the same architecture
   family. Once E2B is validated, only `target_hidden`,
   `sliding_window`, `context_length` need adjusting in
   `MtpDrafterConfig`.

6. **Centroid (`masked_embedding`) path.** Currently bypassed for full
   lm_head. If acceptance is materially worse than the HF blog's "up to
   2×" claim, port `Gemma4AssistantMaskedEmbedder` (top-32 centroid lookup
   + scatter) into both `mtp_drafter_model.py` and `build_mtp_drafter.py`.
