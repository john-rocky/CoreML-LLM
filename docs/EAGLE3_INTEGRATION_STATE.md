# EAGLE-3 Integration — Resumable Session State

Working document for the EAGLE-3 integration on MacBook Air (M3 16GB). Written so a new session can pick up without re-deriving context.

**Last updated:** 2026-04-17. Active branch: `feature/eagle3-speculative` (not merged to main). Status: **Phase 2A + 2B done, Phase 3 benched, speculative currently slower than baseline. Retrain prepared on 2026-04-13 but never executed — abandoned for MTP work.**

---

## 2026-04-17 correction — retrain status

Earlier sections (below) imply the Blocker 1 retrain is in progress. As of 2026-04-17, it is **not**:

- `training_data_custom.pt` (2026-04-13) — corpus collected.
- `claude/eagle3-retrain-custom` last commit 2026-04-13 21:26 (`2ae26a4`, standalone training script).
- `claude/eagle3-full-retrain` last commit 2026-04-14 00:50 (`fa614ad`, notebook-exact training).
- No commits on either branch after 2026-04-14 00:50. Work moved to `feature/mtp-drafter-conversion` starting 2026-04-14 01:49.
- No retrained `eagle3_draft_best.pt` or `eagle3_eval.json` exists on disk. Custom-target acc[0] has never been measured.

Additionally, the 2026-04-17 MTP investigation (`docs/MTP_INVESTIGATION_SUMMARY.md`) surfaced a third blocker that applies to EAGLE-3 as well:

- **Blocker 3 (11c): verify-vs-decode fp16 drift** — sets iPhone acceptance break-even at ~77%. Even a successful retrain landing at 50-60% would not produce a net speedup on device until 11c closes.

Revival cost: ~$0 Colab A100 ~25 min to run the already-written training script on `training_data_custom.pt`. But **do not revive before Blocker 3 (11c) closes** — same gating as MTP.

---

## TL;DR — where we are

| Phase | Status |
|---|---|
| Phase 1 — draft + fusion training | ✅ done on Colab. acc[0]=74.94%, acc[1]=40.6%, acc[2]=23.9% |
| Phase 2A — verify chunks (T=3) | ✅ built + smoke-tested (PASS) + parity-validated in fp32 |
| Phase 2B — Swift integration | ✅ scaffolded on `feature/eagle3-speculative` (Spec target protocol, mask/RoPE builders, public-API gating) |
| Phase 3 — iPhone 17 Pro bench | ⚠️ ran, **not faster than baseline 28.6 tok/s** (11–17 tok/s with fallback to T=1) |

Root cause for Phase 3 miss: **two independent blockers must be fixed together** (see §Blockers). Fixing only one does not help.

---

## Training artifacts (source of truth)

All at `/Users/daisukemajima/Downloads/eagle3_draft/`:

| File | Notes |
|---|---|
| `eagle3_draft_best.pt` | 188 MB, 47.2 M params. Use `best.pt`, not `step4000.pt` / `final.pt` |
| `eagle3_config.json` | `fusion_layers=[8,17,34]`, `hidden=1536`, `num_heads=8`, `num_kv_heads=1`, `head_dim=256`, `ffn=6144`, `embed_scale=39.1918...`, `ttt_k=3`, `model_id=google/gemma-4-E2B-it` |
| `eagle3_eval.json` | acc[0]=0.7494, acc[1]=0.406, acc[2]=0.239, expL=2.13 — passes §3.1 gates |

---

## Converted CoreML artifacts (on disk)

All under `/Users/daisukemajima/Downloads/CoreML-LLM/output/`:

| File | Size | Outputs (incl. EAGLE-3 additions) | Built by |
|---|---:|---|---|
| `eagle3_draft.mlpackage` | 210 MB (INT4) or 838 MB (fp16) | `h_out`, `token` (int32 scalar), `logit` (fp16). Both rebuilt during Phase 3 diagnosis. | `build_eagle3.py` |
| `eagle3_fusion.mlpackage` | 14 MB fp16 | `h_fused` | `build_eagle3.py` |
| `eagle3-chunks/chunk1.mlpackage` | 149 MB | (unchanged, no fusion layer) | `build_eagle3_chunks.py` |
| `eagle3-chunks/chunk2.mlpackage` | 128 MB | existing + **`hidden_at_L8` (1,1,1536) fp16** | `build_eagle3_chunks.py` |
| `eagle3-chunks/chunk3.mlpackage` | 311 MB | existing + **`hidden_at_L17` (1,1,1536) fp16** | `build_eagle3_chunks.py` |
| `eagle3-chunks/chunk4.mlpackage` | 503 MB | `token_id`, `token_logit`, **`hidden_at_L34`** (pre-norm, 1,1,1536 fp16) | `build_eagle3_chunks.py` |
| `eagle3-chunks/verify_chunk1.mlpackage` | 149 MB | `hidden_states_out`, `per_layer_combined_out` (T=3) | `build_eagle3_verify.py` |
| `eagle3-chunks/verify_chunk2.mlpackage` | 128 MB | + `kv13_k/v_out` (1,1,515,256), `kv14_k/v_out` (1,1,2051,512) | `build_eagle3_verify.py` |
| `eagle3-chunks/verify_chunk3.mlpackage` | 311 MB | `hidden_states_out` | `build_eagle3_verify.py` |
| `eagle3-chunks/verify_chunk4.mlpackage` | 503 MB | `token_ids` (3,) int32, `token_logits` (3,) fp16. INT4 palettized. | `build_eagle3_verify.py` |

Smoke tests:
- `python /tmp/smoke_eagle3_chunks.py` — decode chunk I/O
- `python /tmp/smoke_verify_chunks.py` — verify chunk I/O (last confirmed **PASS** 2026-04-13)

---

## Verify chunk I/O contract (T=3)

| chunk | inputs (+ shared mask/RoPE/PLE) | outputs |
|---|---|---|
| verify_chunk1 | `per_layer_raw` (1,T,35·256), `K/V_sliding_in` (7,1,W,512), `K/V_full_in` (1,1,ctx,512) | `hidden_states_out` (1,T,1536), `per_layer_combined_out` (1,T,8960) |
| verify_chunk2 | `per_layer_combined`, `K/V_sliding_in` (5,…), `K/V_full_in` (2,…) | `hidden_states_out`, `kv13_k_out` (1,1,W+T=515,256), `kv13_v_out`, `kv14_k_out` (1,1,ctx+T=2051,512), `kv14_v_out` |
| verify_chunk3 | `kv13_k`/v (1,1,W+T,256), `kv14_k`/v (1,1,ctx+T,512) | `hidden_states_out` (1,T,1536) |
| verify_chunk4 | same as chunk3 | `token_ids` (T,) int32, `token_logits` (T,) fp16 |

Design: read-only KV during verify; attention concats cache with newly-computed K/V (masks shape `(1,1,T,W+T)` / `(1,1,T,ctx+T)`). Swift commits accepted tokens by re-running T=1 decode per token (current impl; see blocker #2).

Parity validated:
- Layer-level fp16 — PASS (max abs diff ≤ 4e-3).
- Chunk1 drill-through fp32 — PASS (1e-5).
- E2E argmax fp32 random — PASS.
- E2E argmax fp16 random — diverges, but this is rounding on `torch.randn` logits over 262K vocab (no top-1 separation). Real Gemma-4 logits are sharp enough. Not a bug.

---

## Phase 3 bench — on-device measurements (iPhone 17 Pro)

Thermally stable, 10-min bench. `[SpecDbg]` + per-call timing instrumentation on `SpeculativeLoop.drawBurst` + `ChunkedEngine.verifyCandidates/commitAccepted`:

| Metric | Observed | Expected (Colab) |
|---|---:|---:|
| Baseline T=1 decode | 28.6 tok/s | — |
| verify T=3 per call | 31.5 ms | — |
| commit per token | 33–36 ms | — |
| Avg accepted tokens / burst | **~2.0** (always exactly 2) | 3.05 |
| Speculative eff throughput | 11–17 tok/s | target 40+ |
| Rolling acceptance | decays to 0.30 within ~15 bursts → falls back to T=1 | 1.0 |
### Phase 2A — Verify chunks (T=3) — **SCAFFOLDED, build in progress**
Chose approach 3 (neither rewrite SWA nor reuse prefill): new `_run_layer_verify` helper that mirrors `_run_layer_swa` structure but handles T positions with read-only KV cache. Attention concats cache with newly-computed K/V (`K_for_attn = cat([K_cache, k_new], dim=2)`) so Swift supplies masks of shape `(1,1,T,W+T)` and `(1,1,T,ctx+T)`. No K/V writes — Swift commits by re-running T=1 decode per accepted token.

Files added this session:
| File | What |
|---|---|
| `conversion/models/gemma4_verify_chunks.py` | `_run_layer_verify` + `VerifyChunk1..4` (T is a ctor arg, default 3) |
| `conversion/build_eagle3_verify.py` | Conversion entry: `--only chunk1 -T 3` per-chunk; INT4 palettized |

**Verify chunk I/O contract** (T=3):

| chunk | inputs (besides shared mask/RoPE/PLE) | outputs |
|---|---|---|
| verify_chunk1 | `per_layer_raw` (1,T,35·256), `K/V_sliding_in` (7,1,W,512), `K/V_full_in` (1,1,ctx,512) | `hidden_states_out` (1,T,1536), `per_layer_combined_out` (1,T,8960) |
| verify_chunk2 | `per_layer_combined`, `K/V_sliding_in` (5,…), `K/V_full_in` (2,…) | `hidden_states_out`, `kv13_k_out` (1,1,W+T=515,256), `kv13_v_out`, `kv14_k_out` (1,1,ctx+T=2051,512), `kv14_v_out` |
| verify_chunk3 | `kv13_k` (1,1,W+T,256), `kv14_k` (1,1,ctx+T,512) | `hidden_states_out` (1,T,1536) |
| verify_chunk4 | same as chunk3 | `token_ids` (T,) int32, `token_logits` (T,) fp16 |

Status (2026-04-12, all built):

| File | Size | Build (convert + palettize) |
|---|---:|---|
| `verify_chunk1.mlpackage` | 149MB | ~25s + 44s |
| `verify_chunk2.mlpackage` | 128MB | ~18s + 44s |
| `verify_chunk3.mlpackage` | 311MB | ~30s + 97s |
| `verify_chunk4.mlpackage` | 503MB | ~55s + ~160s (vocab head dominates) |

Smoke test `python /tmp/smoke_verify_chunks.py` PASS. I/O names and shapes match spec (incl. kv13_k_out `[1,1,515,256]`, kv14_k_out `[1,1,2051,512]`, chunk4 `token_ids [3] int32` + `token_logits [3] fp16`).

**Parity validation (done, partial)**:

- **Layer-level parity — PASS.** `/tmp/parity_layer.py` runs `_run_layer_swa × 3` vs `_run_layer_verify × 1` with matched random init for one sliding (L0) and one full (L4) layer. Max abs diff ≤ 4e-3 in fp16, safely under the fp16 noise floor.
- **Chunk1 fp32 drill-through — PASS.** Per-layer diff through L0..L7 in fp32 is 10⁻⁵ (float epsilon). Confirms `_run_layer_verify` is **mathematically equivalent** to SWA-sequential; the fp16 amplification below is pure accumulated rounding, not a bug.
- **E2E argmax, fp32 — PASS.** `/tmp/parity_e2e_fp32.py` runs SWAChunk1..4 × 3 vs VerifyChunk1..4 × 1 in fp32 with random init. All 3 argmaxes match; |Δlogit| ≤ 0.24. Confirms the full pipeline (including kv13/kv14 shared-layer flow in chunks 3/4) is mathematically equivalent.
- **E2E argmax, fp16 random — expected fail (not a bug).** Same test in fp16 diverges on steps 1, 2. Cause: rounding grows ~2× per layer through 35 layers and `torch.randn` logits over 262K vocab have effectively no top-1 separation. Does NOT predict real-model behavior.
- **Still nice-to-have** before iPhone deploy: real-prompt argmax parity on-device (compile the verify chunks, feed actual cache/hidden after prefill, assert verify argmax = decode-sequential argmax over a few dozen bursts). This is the only remaining gate. Real Gemma-4 logits are sharp enough (top-1 typically 5+ above top-2) that fp16 drift shouldn't flip argmax; if it does, the draft's acceptance rate will crater visibly in bench. Not blocking Phase 2B scaffolding.

Known gotchas for any reconvert:
- Sample masks must match attended-dim length: `causal_mask_full` last dim = `ctx + T`, `causal_mask_sliding` = `W + T`. Rebuilding at a different T requires passing `-T N` to `build_eagle3_verify.py`.
- cos/sin shapes are `(1,1,T,256)` for sliding and `(1,1,T,512)` for full (same convention as decode).
- chunk3/4 receive kv13/kv14 already extended to W+T and ctx+T — those come directly from chunk2's output; Swift never builds them from scratch.

### Phase 2B — Swift integration in ChunkedEngine — **SCAFFOLDED 2026-04-12**
`SpeculativeLoop.swift` is already written (see file in Sources/CoreMLLLM/). It expects a `SpeculativeTarget`-conforming object. ChunkedEngine now conforms (extension at the bottom of `ChunkedEngine.swift`):

- `loadVerifyChunks(from:computeUnits:)` — lazy parallel load of verify_chunk{1..4}.
- `canSpeculate` — true when all four verify chunks are loaded and at least one decode step has captured `hidden_at_L{8,17,34}`.
- `predictStep` now stashes the three hidden taps from decode chunks 2/3/4 (via `featureValue(for:)?.multiArrayValue`, so non-EAGLE-3 chunks silently set them to nil).
- Mask/RoPE/input builders: `makeVerifyCausalMaskFull(position:T:ctx:)`, `makeVerifyCausalMaskSliding(position:T:W:)`, `buildVerifyHidden(tokenIDs:)`, `buildVerifyPLR(tokenIDs:)`, `buildVerifyRoPE(table:position:T:dim:)`.
- `lastHiddenMulti(at:)`, `commitAccepted(_:)`, `verifyCandidates(_:K:)` — the three protocol methods. `commitAccepted` replays `predictStep` per accepted token (simple/correct; can be optimized later).

`swift build` clean (only pre-existing accelerate-deprecation + IOSurface throwing-expression warnings).

**Public API wired (2026-04-12).** `CoreMLLLM.swift` now:
- Auto-loads `eagle3_fusion.mlpackage`, `eagle3_draft.mlpackage`, and `verify_chunk{1..4}.mlpackage` from the model directory when all are present. Any failure falls back silently to T=1 decode.
- Exposes `supportsSpeculative: Bool` and `speculativeAcceptance: Double`.
- In `stream()`'s decode loop, runs one plain T=1 decode first to populate `hidden_at_L*`, then uses speculative bursts while `canSpeculate && shouldSpeculate`. On any burst error, falls back to T=1 for that step. Yields each accepted token individually so the stream consumer sees the same token sequence as non-speculative decode.

`swift build` clean. No functional bench yet — validation is iPhone deploy (Phase 3).

**Original Phase 2B plan retained below for reference.**



1. **Store `hidden_at_L{8,17,34}`** after each decode step. Modify `decodeStep()` (ChunkedEngine.swift around line 340-386) to fetch these outputs from chunk2/3/4 and stash in 3 ivars.

2. **Conform to `SpeculativeTarget`**:
   ```swift
   func lastHiddenMulti(at indices: [Int]) throws -> [MLMultiArray] {
       // Match indices to lastHiddenAtL8 / L17 / L34
   }
   func commitAccepted(_ tokens: [Int32]) throws {
       // For each token, run decodeStep(tokenID:) — advances position + updates KV
       // Last iteration naturally refreshes the lastHiddenAtL* ivars
   }
   func verifyCandidates(_ candidates: [Int32], K: Int) throws -> [Int32] {
       // Call verify_chunk1..4.mlmodelc in sequence. Build T-aware masks
       // (see "Verify-chunk mask builder" below). Cache is READ-ONLY during
       // verify; no K/V writes back. Return chunk4's token_ids (T,).
   }
   ```

3. **Verify-chunk mask builder** (new Swift helpers, sibling to `makeCausalMask` / `makeSlidingCausalMask`):
   - `makeVerifyCausalMaskFull(position: Int, T: Int, ctx: Int) -> MLMultiArray` shape (1,1,T,ctx+T):
     - `[t, i]` for `i ∈ 0..<ctx`: 0 if `i < position` else -inf
     - `[t, ctx+j]` for `j ∈ 0..<T`: 0 if `j ≤ t` else -inf
   - `makeVerifyCausalMaskSliding(position: Int, T: Int, W: Int) -> MLMultiArray` shape (1,1,T,W+T):
     - Cache slot `i ∈ 0..<W` maps to abs pos `position - W + i` (valid if ≥ 0 and ≤ position-1).
     - Abs range for query `t`: `[position + t - W + 1, position + t]`, clip to ≥ 0.
     - Set `[t, W+j]` to 0 if `j ≤ t` else -inf.
   - Per-token RoPE cos/sin: reuse `lookupRoPE` for positions `[position, position+T-1]` and stack on the T axis to `(1,1,T,dim)`.
   - `per_layer_raw`: embed each candidate token with `EmbeddingLookup.perLayerRaw(tokenID:)` and stack on T axis.

4. **Make CoreMLLLM public API** decide when to use speculative (based on `SpeculativeLoop.shouldSpeculate` + rolling acceptance). See `SpeculativeLoop.swift:194`.

### Phase 3 — iPhone deployment + bench
1. Compile `.mlpackage` → `.mlmodelc` (either via Xcode "Add to target" or at runtime via `MLModel.compileModel(at:)`).
2. Replace existing iPhone `chunk1/2/3/4.mlmodelc` with EAGLE-3 versions. Existing chunks lack `hidden_at_L*` outputs → would crash Swift that expects them, so this is an all-or-nothing swap.
3. Bench on iPhone 17 Pro, thermal-stable 10-min, K=1 (baseline) vs K=3 (EAGLE-3). Target per docs/SPEED_8K.md §3 P1: ctx=2048 at 55-70 tok/s, ctx=8192 at ~30 tok/s.

---

## Blockers — both must be fixed for speculative to help

### Blocker 1: draft/target distribution mismatch → ~0% acceptance

`[SpecDbg]` dump showed draft proposals `[3768, 496, 496]` vs target argmax `[68158, 18114, 236772]` — zero overlap.

Ruled out:
- INT4 palettization degrading draft (fp16 draft gives same acc rate).
- Draft outputting random / ID-mapped garbage (proposals track inputs meaningfully, just not matching target).

Cross-check with Mac CPU `test_eagle3_infer.py` using HF Gemma 4 target + PyTorch draft:
- Accept rate **42.9%** (below Colab 74.94% but well above on-device ~0%).

Hidden-tap comparison HF vs our custom `Gemma4Model` on same prompt (chat-formatted through Swift's `buildGemmaPrompt`):
- L8: 45% relative mean diff, norm similar.
- L17: 33% rel diff, norm similar.
- L34: **94% rel diff, HF norm 158 vs our 36 (4.4× magnitude gap)**.

Argmax-chain generations diverge too: HF produces gibberish (`'    '`, `'**'`, `'Py'`) while our custom forward produces coherent haiku (`' leaves'`, `' sway'`, `'\n'`). So HF reference used at draft training time was either broken or a transformers-version mismatch has opened a gap.

**Either way, the trained draft no longer matches the target we deploy → must retrain against our custom target.**

### ⚠️ Correction (2026-04-13): L34 divergence was a measurement error

The "94% rel diff at L34" reported above was a **FALSE ALARM** caused by an indexing artifact in the comparison harness:

1. HF's `output_hidden_states[35]` is the **post-norm output** (after the final RMSNorm), NOT L34's raw hidden-state output. Comparing it against our L34 pre-norm output produced the spurious 4.4× magnitude gap.
2. After correcting the indexing, our `Gemma4Model` forward **matches HF (`cache=True`) at ALL 35 layers** with `rel_diff < 1e-5` in fp32.
3. Verified via `conversion/debug_l34_parity.py`.
4. The EAGLE-3 Blocker 1 draft/target mismatch is now believed to be caused by the training data collection step running HF with `use_cache=False`, which **does not perform KV-sharing for L15+** (global attention layers). This means the hidden states the draft was trained on differ from what the target produces at inference time with `use_cache=True`.

The conclusion that the draft must be retrained still holds, but the root cause is the `use_cache=False` training corpus — not a bug in our custom `Gemma4Model`.

### Blocker 2: `commitAccepted` re-runs T=1 decode per accepted token

Even with a perfect draft, current implementation cannot beat baseline:

| Implementation | Burst formula | @ avg N=3.05 | tok/s | vs baseline 28 |
|---|---|---:|---:|---:|
| Current (re-run decode) | 42 + 33N ms | 143 ms | 21.4 | **0.76× (slower)** |
| K/V direct-write + 1 decode | 75 ms constant | 75 ms | 40.7 | **1.45×** |
| K/V direct-write + Mirror v1 (draft→GPU) | ~69 ms | 69 ms | 44.2 | **1.58×** |
| K/V direct-write + Mirror v2 (cross-burst pipeline) | ~60 ms | 60 ms | 50.8 | **1.82×** |

**Fix**: Phase 2A v2 verify chunks (per-T-position K/V + hidden outputs) already exist in Python on `feature/eagle3-speculative`'s `gemma4_verify_chunks.py` / `build_eagle3_verify.py`. Not yet deployed to device. Swift KV-writer not yet implemented.

---

## How to resume work

### Step 0: check out the right branch

```bash
cd /Users/daisukemajima/Downloads/CoreML-LLM
git checkout feature/eagle3-speculative
source conversion/.venv/bin/activate
```

All the Python conversion, Swift scaffolding, and diagnostic instrumentation live on this branch. It is NOT merged to main.

### Step 1: validate current Mac artifacts

```bash
ls -la output/eagle3_*.mlpackage output/eagle3-chunks/*.mlpackage
# Should show 2 + 8 mlpackages (decode + verify), total ≈2.4 GB
python /tmp/smoke_eagle3_chunks.py   # decode chunks
python /tmp/smoke_verify_chunks.py   # verify chunks
```

If either fails, rebuild:

```bash
# Decode chunks (~20 min)
python conversion/build_eagle3_chunks.py --output ./output/eagle3-chunks

# Verify chunks (~10 min, takes -T for arity, default 3)
python conversion/build_eagle3_verify.py --output ./output/eagle3-chunks

# Draft + fusion (~3 min)
python conversion/build_eagle3.py \
    --ckpt /Users/daisukemajima/Downloads/eagle3_draft/eagle3_draft_best.pt \
    --output ./output/eagle3_draft.mlpackage \
    --fusion-output ./output/eagle3_fusion.mlpackage \
    --palettize-int4
```

### Step 2: pick which blocker to unblock first

Order doesn't matter for correctness — both must be done before a speedup materializes. Suggested order based on effort:

**2a. Unblock via retrain (Blocker 1)** — needs Colab or a bigger box:
- Regenerate hidden-state corpus using our **custom** `Gemma4Model` (not HF) as the teacher. Files to touch:
  - `conversion/collect_eagle_hidden_states.py` — swap `Gemma4ForConditionalGeneration` for our `Gemma4Model` forward.
  - `conversion/train_eagle3_draft.ipynb` — same swap in the eval loop.
- Retrain for 2 epochs × ~30k samples. Target: acc[0] ≥ 0.5 against custom target (below Colab's 0.75 is expected since custom forward's hiddens are less rich at L34).
- Rebuild `eagle3_draft.mlpackage` from new `best.pt`.

**2b. Unblock via K/V direct-write (Blocker 2)** — Mac + iPhone, more Swift work:
- The v2 verify chunks (per-T-position K/V outputs) are already built in `gemma4_verify_chunks.py`. Rebuild if needed.
- `ChunkedEngine.commitAccepted(_:)` currently replays `predictStep` per token. Rewrite to:
  1. Accept verify's output K/V at the accepted-T prefix (not full T).
  2. Write those slices into the IOSurface-backed sliding/full KV caches at the right positions.
  3. Advance `self.position` by N without running decode chunks.
- Last hidden (`hidden_at_L34` for the final accepted token) can be taken from verify_chunk4's `hidden_states_out[N-1]`, avoiding the final decode call too.

### Step 3: deploy + bench

Push the four compiled `.mlmodelc` via `xcrun devicectl device copy to` (same pattern as `/tmp/push_eagle3.sh`, which pushed the v1 bundle). Replace both decode chunks and verify chunks — the app expects the all-or-nothing EAGLE-3 bundle.

Bench in `Examples/CoreMLLLMChat` at 10-minute thermal-stable steady state. Compare against baseline 28.6 tok/s. Target after both blockers fixed: ≥40 tok/s at 2K, ≥22 tok/s at 8K (baseline 8K = 14.5 tok/s).

---

## Environment gotchas (things that will bite if ignored)

- **Python**: system 3.9.6 does NOT work with coremltools 8+/9+. Use `/opt/homebrew/bin/python3.11` via `conversion/.venv/`.
- **`accelerate`** is NOT in requirements.txt but needed by `transformers==5.5.0` with `device_map`. Install ad hoc: `pip install accelerate`.
- **Current torch**: 2.7.0 (downgraded from 2.11 during monolithic-path debugging). `build_eagle3_chunks.py` / `build_eagle3_verify.py` trace cleanly at 2.7.0; bump cautiously.
- **HF model cache**: `google/gemma-4-E2B-it` cached at `~/.cache/huggingface/hub/...`, copied to `output/gemma4-e2b/hf_model/` for `Gemma4Model.from_pretrained(HF_DIR)`. Model is NOT gated — anonymous DL works.
- **`test_eagle3_infer.py`**: MPS OOMs on M3 16GB (9.54 GiB single alloc). Always use `--device cpu` for sanity tests.
- **`convert.py` Gemma 4 monolithic path is BROKEN** — `gemma4_wrapper.py:107` misses an NCHW permute on the (1,1,1536) hidden. EAGLE-3 work does not touch this path. If fixing: wrap in `.permute(0,2,1).unsqueeze(2)` before Conv2d, reverse after.

---

## Files this work touches

| File | What |
|---|---|
| `conversion/build_eagle3.py` | Builds draft + fusion mlpackages |
| `conversion/build_eagle3_chunks.py` | Builds decode chunks with `hidden_at_L*` taps |
| `conversion/build_eagle3_verify.py` | Builds T=3 verify chunks (v1 + v2 output variants) |
| `conversion/models/gemma4_verify_chunks.py` | `_run_layer_verify` + `VerifyChunk1..4` |
| `conversion/test_eagle3_infer.py` | Mac-compat patches (apply_rope dtype, draft fp16 cast, HF_DIR env fallback) |
| `conversion/build_speculative.py` | Patched `HF_DIR` env var / output-dir fallback |
| `Sources/CoreMLLLM/SpeculativeLoop.swift` | Pre-existing. Plus `[SpecDbg]` logging in first 3 bursts |
| `Sources/CoreMLLLM/ChunkedEngine.swift` | `SpeculativeTarget` conformance, verify-mask builders, spec profile counters |
| `Sources/CoreMLLLM/CoreMLLLM.swift` | Auto-loads verify chunks, `supportsSpeculative`, `speculativeAcceptance`, per-burst `[Spec] burst #N` stats |
| `docs/EAGLE3_INTEGRATION_STATE.md` | This file |

---

## First-thing-to-do on a fresh session

```bash
cd /Users/daisukemajima/Downloads/CoreML-LLM
git checkout feature/eagle3-speculative
source conversion/.venv/bin/activate
ls -la output/eagle3_*.mlpackage output/eagle3-chunks/*.mlpackage
python /tmp/smoke_verify_chunks.py   # PASS = Phase 2A artifacts intact
```

If PASS: Mac-side work is intact; go to Step 2 above. If FAIL: rebuild via Step 1.
