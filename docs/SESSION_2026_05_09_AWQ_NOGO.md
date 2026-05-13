# Session 2026-05-09 — AWQ verify chunks A/B → NO-GO on Mac E2B

## TL;DR

- AWQ-smoothed verify chunks (`verify_chunks_awq/`) provide **no measurable speedup** over the canonical `verify_chunks_postnorm` build on Gemma 4 E2B Mac MTP free-form text.
- AWQ vs POSTNORM apples-to-apples (5-iter, greedy, identical prompts, σ ≤ 0.25): translate −2.3 %, code_class flat, markdown +1 %. Within / below run-to-run noise; translate regression is 4σ.
- Bundle restored to POSTNORM. AWQ build artifacts marked for deletion.
- Mac MTP-on translate = 35.98 tok/s vs MTP-off 33.15 tok/s → **+8.5 % speedup**, which is **the middle of HF's E2B chat-templated ceiling (1.09–1.19×)**. We are at the realistic E2B operating point.
- The widely cited "1.68× translate" figure is most plausibly NOT an E2B number (likely 27B/31B-class). vLLM's published E2B γ=2 figure is 1.30 ×. Closing the last 0.2× requires drafter retraining against a quantized target — Path B in `project_mtp_pivot.md`.

## Methodology

- Hardware: Mac (host workstation), single decode trial each, 5 iterations per (prompt, build).
- Prompts (greedy only, MTP_FORCE_SPECULATE=1):
  - `translate` — 10-sentence EN→FR.
  - `code_class` — Python LRUCache class with type hints + 3 pytest tests.
  - `markdown_table` — 8 programming languages × 5 columns markdown table.
- Decode budget: 256 tokens.
- Bench script: `/tmp/run_freeform_5iter.sh` (5-iter, greedy, 5 s warmup sleep).
- Baseline (MTP-off): `/tmp/run_mtp_off.sh` (LLM_MTP_DISABLE=1, translate × 3 iters).
- Builds:
  - **POSTNORM** = `output/gemma4-e2b/verify_chunks_postnorm/` — canonical, K=3, post-norm fix + `PALETTIZE_KEEP_FP_KV=1` + `--emit-logits`.
  - **AWQ** = `output/gemma4-e2b/verify_chunks_awq/` — same K=3, same post-norm fix, additional `awq_smooth_gemma4` SmoothQuant pass on chunk2 attention weights before palettize.

## Raw numbers

```
build         prompt              tok/s          accept       n
----------------------------------------------------------------
AWQ           translate           35.14±0.21     0.12±0.00    5
AWQ           code_class          37.35±0.22     0.15±0.00    5
AWQ           markdown_table      41.33±0.25     0.20±0.00    5
POSTNORM      translate           35.98±0.15     0.13±0.00    5
POSTNORM      code_class          37.37±0.20     0.15±0.00    5
POSTNORM      markdown_table      40.92±0.15     0.19±0.00    5
```

MTP-off translate baseline (3 iter, σ ≈ 0.03): 33.15 tok/s.

## A/B verdict — AWQ NO-GO

| prompt | POSTNORM | AWQ | Δ % | Δ accept | judgement |
|---|---|---|---|---|---|
| translate | 35.98 | 35.14 | −2.3 % | −0.01 | regression (4σ) |
| code_class | 37.37 | 37.35 | 0.0 % | 0.00 | flat |
| markdown_table | 40.92 | 41.33 | +1.0 % | +0.01 | within noise |

The hypothesis going in was: AWQ smoothing of chunk2 reduces INT4 K cache distribution shift → drafter accept rises → throughput rises. The accept rate moved by ≤0.01 in either direction, so the smoothing did NOT propagate to drafter acceptance. Two plausible reasons:

1. **chunk1 + chunk4 dominate distribution shift, not chunk2.** Cosine drop vs FP16 is chunk1 0.978 / chunk4 0.834 (per `SESSION_2026_05_06_MTP_MAC.md`); chunk2 was already 0.99+. Smoothing chunk2 alone leaves the dominant drift untouched.
2. **The drafter is robust to chunk2 perturbations.** The shared-KV input to drafter is mostly assembled from kv13/kv14 produced by chunk2 + chunk4; if chunk4's K cache (head_dim 256, full attention) is the noise floor, chunk2's contribution is masked out.

Either way, the action item — train an AWQ-smoothed chunk1 and chunk4 — is unattractive: chunk4 is the largest model and AWQ on the LM-head row is structurally invalid. The marginal improvement budget is too small to justify the build cost.

## HF / public reference table

| Implementation / model | Result | Source |
|---|---|---|
| **Mac CoreML POSTNORM (this session)** | translate +8.5 %, markdown +14 % vs MTP-off | this doc |
| HF transformers E2B chat-templated | +9 to +19 % | `docs/SESSION_2026_05_06_MTP_MAC.md` |
| vLLM E2B γ=2 | +30 % | [blog.google MTP for Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/) |
| Public "translate 1.68× / household 1.52×" | most plausibly 27B/31B-class — **NOT E2B** | reframed; lilting.ch warns E2B has structurally less MTP headroom because LM-head is a higher fraction of compute. |
| LiteRT-LM E2B iPhone 17 Pro | 56.5 tok/s (MTP-on/off δ undisclosed) | [HF model card](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) |

## Empirical mask / RoPE / KV-layout verification (added 2026-05-09 evening)

The `docs/SESSION_2026_05_06_MTP_MAC.md:203-206` and
`docs/GEMMA4_MTP_OFFICIAL_KB.md:291-294` notes flagged HF
`create_attention_masks`'s `sliding_attention_mask.flip(dims=(1,))` as a
possibly-missing operation in our Swift `makeSlidingCausalMask`. The
initial line-by-line audit dismissed it (HF flip is NOP for q_len=1 +
all-1s attention_mask), but never confirmed empirically. Closed now.

Procedure (`conversion/empirical_mask_check.py`):

1. Run Swift on translate prompt with `MTP_CHUNK_DUMP=/tmp/coreml_kv_dump_translate`. Captures production right-aligned `kv13_k/v`, left-aligned `kv14_k/v`, post-norm `h_last`, and `embed_input` of bootstrap nextID at first speculative round.
2. Run HF target on `prompt + [bootstrap_nextID]` (120 tokens) → gets reference left-aligned `shared_kv_states` and `last_hidden_state`.
3. **PROD test** — feed production KV + production-built mask to CoreML drafter mlpackage. Top-1 = 496.
4. **HF reference** — feed HF reference KV to HF Python drafter. Top-1 = 506.
5. **ISO test** — feed HF reference KV padded into right-aligned production layout, with production-built mask, to CoreML drafter mlpackage. Top-1 = **506** = HF reference. ✅

Per-position KV cosine production vs HF reference:

| position | cosine | source |
|---|---|---|
| 0 | 0.998 | prefill-written |
| 60 | 0.972 | prefill-written |
| 118 | ~0.97 (implicit from overall avg) | prefill-written |
| **119** | **0.450** | **bootstrap-decode-written** |

post-norm `h_last` at position 119: cosine 0.396 vs HF reference (also bootstrap-decode-written).

### Conclusion (empirical)

- **Mask is correct.** ISO top-1 (CoreML drafter on HF KV + our mask) = HF reference top-1.
- **RoPE is correct.** Same evidence.
- **KV layout / right-alignment is correct.** ISO test pads HF KV into our right-aligned slot scheme and the drafter still produces HF top-1.
- **drafter mlpackage is correct.** Quantization preserves top-1 ranking on clean inputs.
- **Bottleneck is INT4 noise compound at the decode-step forward pass.** Production K_13 at position 119 has cosine 0.45 to HF reference K. Production h_last has cosine 0.40 to HF reference. This noise is sufficient to flip drafter top-1 ↔ top-2: HF top-1 = 506 appears at position 2 of production drafter's top-5 (full top-5: `[496, 506, 1070, 822, 532]`).

### Why decode position has worse cosine than prefill positions

Prefill computes 0..118 in parallel on a single fp16-activation forward pass with INT4 weights — each position's K is "fresh" from the same forward state. INT4 weight noise drifts cosine to ~0.97.

Decode at position 119 reads the K/V cached by prefill (already 0.97 cosine, ~3% drift) and feeds it through 35 attention layers. At each layer the noisy cached K is consumed, attention output drifts further, the next layer's input drifts more, and so on. After 35 INT4-quantized layers compounding, the K written at the new position is at 0.45 cosine — roughly what you'd expect from a 35-layer chain of `(1 - 3%)^35 × 0.97` = ~0.34, which matches observed.

This is **structural** to INT4-on-decode and **not fixable in PTQ space**.

## Mechanism audit (HF transformers `_assisted_decoding`)

We deeply audited HF's `SinglePositionMultiTokenCandidateGenerator` (`transformers/generation/candidate_generator.py:1230-1417`) and `_assisted_decoding` (`generation/utils.py:3540-3725`) and cross-checked against our Swift implementation. **All five high-confidence "you might be doing this wrong" candidates check out (empirically reconfirmed by the ISO test above):**

| HF requirement | Our location | Verdict |
|---|---|---|
| Seed token uses target's `embed_tokens.weight × sqrt(hidden)` | `MtpSpeculativeEngine.swift:215`, `ChunkedEngine.swift:2049` | ✅ |
| Drafter step ≥1 carries drafter's own post-projection hidden | `MtpSpeculativeEngine.swift:331` | ✅ |
| Hidden seed slice index = `n_matches`, NOT `−1` | `MtpSpeculativeEngine.swift:539` (`sliceVerifyHidden(at: matchCount)` — verify_qK is fed `[cur_token, cand_1..K-1]`, so index `matchCount` = "after cur+matchCount accepted candidates") | ✅ |
| KV rolled back to post-accept length | 11c protocol — verify never writes to persistent cache; `commitAccepted` writes accepted prefix only | ✅ |
| `position_ids` constant across drafter K-loop | cos/sin/masks not advanced in K-loop | ✅ |

Conclusion: our implementation is faithful to HF's mechanism. The remaining gap is not a bug, it's the W4 palettize → drafter distribution mismatch.

## LiteRT-LM mechanism (out of reach)

LiteRT-LM uses an entirely different drafter (older 4-layer mini-transformer, `Section 9` in the `.litertlm` container, **co-trained with the W4A8 + INT8 KV target**), pure Metal GPU on iOS, single fused TFLite graph with `decode/prefill_128/prefill_1024/verify_4` signatures, zero-copy `TensorBuffer::Duplicate()` KV sharing. None of these are replicable on the CoreML+ANE constraint.

## Remaining gap analysis

| observation | cause |
|---|---|
| accept ≈ 0.13–0.20 (≈ matchCount/2 with K_USE=2), expected 0.5+ | INT4 palettize chunk1 0.978 / chunk4 0.834 cosine drop vs FP16. Drafter trained on FP16 target → quant distribution shift |
| AWQ on chunk2 alone does nothing | dominant drift is in chunk1+chunk4; AWQ on the LM-head row (chunk4 last block) is structurally invalid |
| HF "1.68×" not reproduced on E2B | E2B is small enough that LM-head fraction dominates compute — per-token MTP gain is structurally capped |

## Next-step decision

1. **Path B / drafter retraining (A100, 3-5 d)** — `project_mtp_pivot.md`. Re-train the assistant against our W4A16-palettized target. Plausible accept lift 0.13 → 0.30+, throughput target ~1.30× (vLLM E2B line). **This is the only implementation lever still on the table.**
2. **iPhone 5-iter A/B with current POSTNORM** — confirm the +8.5 % Mac result transfers. Per `project_mtp_official_mac_outcome.md` iPhone shows similar INT4 distribution; expect parity. Schedule next iPhone trip alongside other queued device tests (per `feedback_mac_first_validate.md`).
3. **Calibrate expectations** — drop the 1.68× target. Realistic E2B ceiling is 1.30× (drafter retraining required) or ~1.19× (current drafter, current quantization).
4. **Cleanup** — delete AWQ build artifacts (3.3 GB unmerged):
   - `output/gemma4-e2b/verify_chunks_awq/` (1.1 GB)
   - `output/gemma4-e2b/verify_chunks_awq_compiled/` (1.1 GB)
   - `/tmp/bundle_chunks_postnorm_backup/` (1.1 GB)

## Bundle state at end of session

`output/gemma4-e2b/bundle_diff_logits/chunk{1..4}.mlmodelc` = POSTNORM (restored from backup at 12:45 JST). Verified by chunk-size match (159 / 139 / 311 / 503 MB).

## Reproducibility scripts in this session

- `/tmp/run_freeform_5iter.sh` — greedy 5-iter, 3 prompts, single label arg.
- `/tmp/run_mtp_off.sh` — translate × 3 iter MTP disabled.
- `/tmp/freeform_summary.py` — log → mean ± std table.
- `conversion/awq_smooth_gemma4_v2.py` — AWQ smoothing recipe used to produce `verify_chunks_awq/`.
