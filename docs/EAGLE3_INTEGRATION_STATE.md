# EAGLE-3 Integration — Resumable Session State

Working document for the ongoing EAGLE-3 integration on this MacBook Air (M3 16GB). Written so a new session can pick up from here without re-deriving context.

**Last updated:** 2026-04-12 (Phase 2A scaffolded). Branch: `feature/audio-support`.

---

## Training artifacts (source of truth)

| File | Path | Notes |
|---|---|---|
| `eagle3_draft_best.pt` | `/Users/daisukemajima/Downloads/eagle3_draft/` | 188MB, 47.2M params |
| `eagle3_config.json` | same dir | `fusion_layers=[8,17,34]`, `hidden=1536`, `num_heads=8`, `num_kv_heads=1`, `head_dim=256`, `ffn=6144`, `embed_scale=39.1918...`, `ttt_k=3`, `model_id=google/gemma-4-E2B-it` |
| `eagle3_eval.json` | same dir | **acc[0]=74.94%, acc[1]=40.6%, acc[2]=23.9%, expL=2.13** — well above §3.1 gates (≥55% / ≥2.0). Projection 30→63.8 tok/s. |
| `eagle3_training.log` | same dir | 2 epochs × ~30k samples on Colab |

Use `best.pt`, not `step4000.pt` or `final.pt` — best has highest acc[0].

---

## Converted CoreML artifacts (ready for iPhone)

All under `/Users/daisukemajima/Downloads/CoreML-LLM/output/`:

| File | Size | Outputs (incl. EAGLE-3 additions) | Built by |
|---|---:|---|---|
| `eagle3_draft.mlpackage` | 210MB | `h_out`, `token` (int32 scalar), `logit` (fp16) | `build_eagle3.py` |
| `eagle3_fusion.mlpackage` | 14MB | `h_fused` | `build_eagle3.py` |
| `eagle3-chunks/chunk1.mlpackage` | 149MB | (unchanged, no fusion layer) | `build_eagle3_chunks.py` |
| `eagle3-chunks/chunk2.mlpackage` | 128MB | existing + **`hidden_at_L8` (1,1,1536) fp16** | `build_eagle3_chunks.py` |
| `eagle3-chunks/chunk3.mlpackage` | 311MB | existing + **`hidden_at_L17` (1,1,1536) fp16** | `build_eagle3_chunks.py` |
| `eagle3-chunks/chunk4.mlpackage` | 503MB | `token_id`, `token_logit`, **`hidden_at_L34`** (pre-norm, 1,1,1536 fp16) | `build_eagle3_chunks.py` |
| `eagle3-chunks/verify_chunk1.mlpackage` | 149MB | `hidden_states_out`, `per_layer_combined_out` (T=3) | `build_eagle3_verify.py` |
| `eagle3-chunks/verify_chunk2.mlpackage` | 128MB | + `kv13_k/v_out` (1,1,515,256), `kv14_k/v_out` (1,1,2051,512) | `build_eagle3_verify.py` |
| `eagle3-chunks/verify_chunk3.mlpackage` | 311MB | `hidden_states_out` | `build_eagle3_verify.py` |
| `eagle3-chunks/verify_chunk4.mlpackage` | 503MB | `token_ids` (3,) int32, `token_logits` (3,) fp16 | `build_eagle3_verify.py` |

All INT4 palettized (group_size=32). Output dtype 65552 = `MLMultiArrayDataType.float16`.

Smoke-tested with `/tmp/smoke_eagle3_chunks.py`: all output names / shapes verified.

---

## Environment gotchas (things that will bite if ignored)

### Python version
- System Python 3.9.6 does **not** work with coremltools 8+/9+. Need 3.10-3.12.
- Installed: `brew install python@3.11` → `/opt/homebrew/bin/python3.11` (Python 3.11.15).

### Venv and pinned deps
- `conversion/.venv/` is the active venv. Activate with `source conversion/.venv/bin/activate` from repo root.
- `requirements.txt` pins `torch==2.11.0` but the monolithic path bug surfaces with `torch==2.7.0` too — the bug is not a torch version issue (see below).
- `accelerate` is NOT in requirements.txt but is required by `transformers==5.5.0` when using `device_map`. Installed ad hoc: `pip install accelerate`.

### Current installed torch: 2.7.0
Downgraded from 2.11 to try to fix convert.py (didn't help — bug is elsewhere). If you bump torch back up, verify `build_eagle3_chunks.py` still traces cleanly.

### HF access
- `google/gemma-4-E2B-it` is **not** gated (at time of writing) — anonymous DL works even without `HF_TOKEN`. The "unauthenticated requests" warning is cosmetic.
- Gemma 4 model already cached at `~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/` (~5.5GB) and copied to `output/gemma4-e2b/hf_model/` for `Gemma4Model.from_pretrained(HF_DIR)` to use.

### convert.py is broken for Gemma 4 monolithic — **do not use**
- Error: `gemma4_wrapper.py:107` passes (1,1,1536) hidden directly to a Conv2d(1536, 8960, 1) without the NCHW permute (`.permute(0,2,1).unsqueeze(2)`). Fails trace with "expected input[1, 1, 1, 1536] to have 1536 channels, but got 1".
- **Not needed for EAGLE-3 work.** Use `build_eagle3_chunks.py` (bypasses the broken wrapper, uses `SWAChunk1-4` directly which have correct NCHW handling).
- If someone wants to fix it: wrap hidden_states in `.permute(0,2,1).unsqueeze(2)` before the Conv2d, then `.squeeze(2).permute(0,2,1)` after.

### test_eagle3_infer.py Mac-compat patches (already applied)
1. `apply_rope` now casts cos/sin to `x.dtype` (was fp32, would break fp16 q/k).
2. Draft cast to fp16 after loading (was fp32, target forward is fp16 → dtype mismatch on fusion).
3. Pass `--device cpu` or `--device mps`. **MPS OOMs on M3 16GB** (9.54 GiB single alloc). Use `cpu` — slow but works for sanity.

### Sanity check result (for reference)
```
[1/2] target-only greedy generation ... 1.11 tok/s
[2/2] EAGLE-3 speculative (K=3) ....... 0.97 tok/s
  draft accept rate: 33.3% of 3 proposals per step
  outputs match: True    ← THIS IS THE GATE, not speedup (CPU is apples-to-oranges vs ANE)
```

---

## Remaining work

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

### Phase 3 — iPhone 17 Pro deploy & bench — **RAN, SPECULATIVE NOT FASTER** (2026-04-12)

All EAGLE-3 artifacts compiled to `.mlmodelc` and pushed to iPhone 17 Pro via `xcrun devicectl device copy to`:

| File | Size | Notes |
|---|---:|---|
| chunk{1..4}.mlmodelc | 149 / 128 / 311 / 503 MB | Overwrote non-EAGLE-3 decode chunks |
| verify_chunk{1..4}.mlmodelc | same | New — Phase 2A v1 outputs only (no K/V direct-write outputs yet) |
| eagle3_fusion.mlmodelc | 13.5 MB | fp16, not palettized |
| eagle3_draft.mlmodelc | 838 MB fp16 (replaced 210 MB INT4) | fp16 tested after INT4 ruled out as acc rate culprit |

App: `Examples/CoreMLLLMChat`, bundle `com.example.CoreMLLLMChat`, domain path `Documents/Models/gemma4-e2b/`. `swift build` on Phase 2B changes clean (no new warnings).

**On-device results** (with `[SpecDbg]` and per-call timing instrumentation added to `SpeculativeLoop.drawBurst` + `ChunkedEngine.verifyCandidates/commitAccepted`):

| Metric | Observed | Expected (Colab eval) | Notes |
|---|---:|---:|---|
| Baseline T=1 decode | 28.6 tok/s | — | steady-state, `[Profile]` lines |
| verify T=3 per call | 31.5 ms | — | close to 1 decode's cost |
| commit per token | 33-36 ms | — | equals T=1 decode (intentional — re-runs decode) |
| Avg accepted tokens / burst | **2.00-2.07** (always exactly 2 per burst) | 3.05 (from acc[0]=0.75, acc[1]=0.41, acc[2]=0.24) | **Draft proposals match target ~0% of the time** |
| Speculative eff throughput | 11-17 tok/s | target 40+ | burst overhead >> draft gain |
| Rolling acceptance | decays to 0.30 fallback within ~15 bursts | 1.0 | fallback triggers → runtime silently drops to T=1 |

**Verdict**: End-to-end speculative pipeline works (fusion + draft + verify + commit + fallback all run), but on-device **draft proposals almost never match target argmax**, so no speedup — and in fact speculative is slower than T=1 until the rolling-acceptance fallback kicks in, at which point tok/s recovers to baseline.

### Phase 3 diagnostic — root cause found

`[SpecDbg]` dump of first 3 bursts showed proposals like `[3768, 496, 496]` vs target argmax `[68158, 18114, 236772]` — zero overlap in draft's predictions.

Ruled out:
1. **INT4 palettization degrading draft**: rebuilt draft at fp16 (no `--palettize-int4`), pushed 838 MB version. Same acc rate. Draft quantization is fine.
2. **Draft outputs ID-mapping or random**: proposals track inputs meaningfully (distinct outputs for distinct tTokNext), just not matching target.

Ran `test_eagle3_infer.py` on Mac CPU with HF Gemma 4 target + PyTorch draft:
- Accept rate: **42.9%** (1.29 avg proposals accepted per K=3 burst)
- This is below Colab's 74.94% but well above on-device ~0%

Diagnostic: compared HF `output.hidden_states[L+1]` at fusion layers vs our custom `Gemma4Model` + `SWAChunk1..4` forward on the same "Write a haiku." prompt:
- L8: rel_mean diff 45%, norm similar
- L17: rel_mean diff 33%, norm similar
- L34: **rel_mean diff 94%, HF norm 158 vs our 36** (4.4× magnitude gap)

And on real argmax-chain generation (same chat-formatted prompt through Swift's `buildGemmaPrompt`):
- HF direct: `140 ('    '), 1018 ('**'), 16251 ('Py')` — gibberish
- On-device custom: `6895 (' leaves'), 47934 (' sway'), 107 ('\n')` — coherent haiku start

→ **Our custom `Gemma4Model` produces DIFFERENT final hiddens than HF's reference Gemma 4 forward.** The draft was trained on HF's hidden distribution (via `collect_eagle_hidden_states.py` + `train_eagle3_draft.ipynb`, both using `Gemma4ForConditionalGeneration`). On-device it sees our custom forward's hiddens, which are out-of-distribution → acc ≈ 0.

The custom forward appears to produce "correct for Gemma-4-it chat" behavior (tokens make sense), while HF with the same version of transformers produces gibberish — so the HF reference used at training time may have been broken, or a transformers-version mismatch has opened a gap since. Either way, **the trained draft no longer matches the target we deploy**.

### Phase 3 — decision matrix for the actual speedup

Separate from the acceptance-rate issue, the current `commitAccepted` implementation re-runs `predictStep` per accepted token. That means even a PERFECT draft cannot beat baseline:

| Implementation | Burst formula | Burst @ avg N=3.05 | tok/s @ N=3.05 | vs baseline 28 |
|---|---|---:|---:|---:|
| Current (re-run decode) | 42 + 33N ms | 143 ms | 21.4 | **0.76x (slower)** |
| K/V direct-write + 1 decode | 75 ms constant | 75 ms | 40.7 | **1.45x** |
| K/V direct-write + Mirror v1 (draft→GPU) | ~69 ms | 69 ms | 44.2 | **1.58x** |
| K/V direct-write + Mirror v2 (cross-burst pipeline) | ~60 ms | 60 ms | 50.8 | **1.82x** |

**Two independent blockers must be fixed together** for a real speedup:
1. **Retrain draft against our custom target** (not HF), so acc reaches Colab's ~0.75.
2. **Rewrite `commitAccepted` to use verify's K/V / hidden outputs directly** — the Phase 2A v2 verify chunks (K/V + hidden per T position outputs) are already built in Python but NOT yet deployed; the Swift writer isn't implemented.

Fixing only one of the two leaves speculative at or below baseline.

### Phase 3 — current working-copy state (for resume)

Code on `feature/eagle3-speculative` (local HEAD ahead of origin by diagnostic patches):
- `Sources/CoreMLLLM/SpeculativeLoop.swift` — added `debugBurstsRemaining` + `[SpecDbg]` logging (first 3 bursts)
- `Sources/CoreMLLLM/CoreMLLLM.swift` — added `[Spec] burst #N` per-burst stats with verify/commit breakdown, first-call gate, error fallback
- `Sources/CoreMLLLM/ChunkedEngine.swift` — added `specVerifyMs/Calls`, `specCommitMs/Tokens` counters + `resetSpecProfile()`
- `conversion/models/gemma4_verify_chunks.py` — Phase 2A v2 outputs added (K_sliding_new/V_sliding_new/K_full_new/V_full_new per chunk1/2, hidden_at_L{8,17,34} per chunk2/3/4). **Not yet deployed** — v1 packages are on device.
- `conversion/build_eagle3_verify.py` — spec updated for v2 outputs. Rebuilt mlpackages exist at `output/eagle3-chunks/verify_chunk{1..4}.mlpackage` but not pushed.
- `/tmp/compare_hidden_taps.py`, `/tmp/eagle3_draft_fp16.mlmodelc` — diagnostic artifacts, not in repo.

These diagnostic additions are valid and worth committing if we continue the work; they don't affect non-speculative paths.

---

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

## Command cheat sheet (Mac)

```bash
# Always start from repo root:
cd /Users/daisukemajima/Downloads/CoreML-LLM
source conversion/.venv/bin/activate

# Sanity check (CPU, slow but validates match)
python conversion/test_eagle3_infer.py \
    --ckpt /Users/daisukemajima/Downloads/eagle3_draft/eagle3_draft_best.pt \
    --prompt "The capital of Japan is" --max-new 16 --K 3 --device cpu

# Rebuild fusion + draft mlpackages (≈3 min)
python conversion/build_eagle3.py \
    --ckpt /Users/daisukemajima/Downloads/eagle3_draft/eagle3_draft_best.pt \
    --output ./output/eagle3_draft.mlpackage \
    --fusion-output ./output/eagle3_fusion.mlpackage \
    --palettize-int4

# Rebuild all 4 decode chunks (≈20 min total)
python conversion/build_eagle3_chunks.py --output ./output/eagle3-chunks
# Or one at a time: --only chunk2

# Smoke-test output contracts
python /tmp/smoke_eagle3_chunks.py
```

---

## Files I own in this work

| File | What |
|---|---|
| `conversion/build_eagle3_chunks.py` | Builds decode chunks with hidden taps |
| `conversion/models/gemma4_verify_chunks.py` | New — verify T=3 helper + chunks, read-only KV |
| `conversion/build_eagle3_verify.py` | New — conversion entry for verify chunks |
| `conversion/test_eagle3_infer.py` | Patched for Mac (apply_rope dtype, draft fp16 cast, HF_DIR fallback) |
| `conversion/build_speculative.py` | Patched `HF_DIR` to env var / `../output/gemma4-e2b/hf_model` fallback |
| `docs/EAGLE3_INTEGRATION_STATE.md` | This file |

`SpeculativeLoop.swift` was already in place; unchanged in this session.

---

## Quick validation to run first in a new session

```bash
cd /Users/daisukemajima/Downloads/CoreML-LLM
source conversion/.venv/bin/activate
ls -la output/eagle3_*.mlpackage output/eagle3-chunks/*.mlpackage
# Should show 2 + 4 mlpackages, total ≈1.4GB
python /tmp/smoke_eagle3_chunks.py  # should print PASS
```

If that passes, the Mac-side conversion work is intact and you can move to Phase 2A (verify chunks) or 2B (Swift).
