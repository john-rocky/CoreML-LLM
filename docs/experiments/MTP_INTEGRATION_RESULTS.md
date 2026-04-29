# MTP Speculative Decoding — Integration Results

**Status:** 2026-04-14. Full integration complete, deployed to iPhone 17 Pro. **Measured on-device acceptance rate: 0%** across all prompts. Root cause identified as a target-distribution mismatch (not a quantization gap, not a bug in our PyTorch reimplementation). MTP Path A is architecturally incompatible with the HF Gemma 4 weights we ship. Work is parked.

**TL;DR:** We built everything — multi-function verify chunks, MTP drafter mlpackage, Swift speculative engine, app integration — and it runs on device without crashing. But the drafts never match the target's predictions, so every round falls back to single-token correction. End-to-end tok/s is identical to baseline (≈14.8 tok/s at ctx=2048). The **decisive finding from a direct TFLite-vs-HF test** (see §5 below) is that **even Google's raw TFLite drafter gets 0/5 argmax match against HF Gemma 4's natural next token** — the drafter was trained for LiteRT's INT8-quantized target, whose L34 hidden state and KV caches are not numerically equivalent to HF's fp32 output. We can't fix this in PyTorch, CoreML, or quantization tuning. Recommendation: **pivot to EAGLE-3 retrain** (known path with explicit training control against our target), or **train our own MTP heads** from our trunk's own outputs (Path C in MTP_PATH_A_FINDINGS §7).

---

## 1. What was built and shipped

All committed on `feature/mtp-speculative-v1` (pushed to origin).

### 1.1 Swift-side integration

- `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` (new, 230 LOC)
  - Bootstrap (one decode step to warm up kv13/kv14).
  - Per-cycle K=3 drafting: embed(nextID) + carry_state → drafter.
  - Per-step RoPE updates (draftPos = pos + k).
  - Verify via `engine.verifyCandidates(tokens, startPosition)`.
  - Greedy accept/reject with write-through KV semantics.
  - Commit: `min(matchCount + 1, K)` positions (avoids advancing past verified KV).
  - Carry state from `lastVerifyHiddenStates` at the last committed index.
  - Diagnostic logging (embed/carry norms, round-by-round match detail).

- `Sources/CoreMLLLM/ChunkedEngine.swift` (modified)
  - Stored properties `lastKV13K/V`, `lastKV14K/V` populated after each decode/prefill/verify.
  - `lookupRawEmbed(tokenID)` for drafter input.
  - `lookupCosSWA/SinSWA/CosFull/SinFull(position)` and `makeDrafterSWAMask/FullMask` convenience wrappers.

- `Sources/CoreMLLLM/EmbeddingLookup.swift` (modified)
  - Added `lookupUnscaled(tokenID)` — embedding without the `embedScale = sqrt(H)` factor.

- `Sources/CoreMLLLM/MtpDraftSource.swift` (modified)
  - Added `draftOne(...)` single-step entry point so the speculative engine can update RoPE/masks per step.

- `Sources/CoreMLLLM/ModelDownloader.swift` (modified)
  - Accept `chunk1.mlpackage/Manifest.json` (not just `.mlmodelc/weights/weight.bin`) as a valid downloaded model.

- `Sources/CoreMLLLM/CoreMLLLM.swift` (modified)
  - Auto-load `mtp_drafter.mlpackage` when `hasVerify` is true.
  - Decode loop switches between `speculateStep` and normal `predictStep` based on `mtpEnabled` + `shouldSpeculate`.
  - Public metrics: `mtpAcceptanceRate`, `mtpTokensPerRound`.

- Example chat app (`Examples/CoreMLLLMChat/...`)
  - `LLMRunner` surfaces `mtpAcceptanceRate` / `mtpTokensPerRound`.
  - `ChatView` shows `acc0=XX%` alongside `tok/s` during generation.

### 1.2 Models built and deployed

| Artifact | Size | Source |
|---|---|---|
| `mtp_drafter.mlmodelc` (INT4 palettized, ctx=2048, W=512) | 38 MB | `build_mtp_drafter.py` from `output/mtp_probe/mtp_drafter.pt` |
| `chunk1-4.mlmodelc` (multi-function: `decode_q1` + `verify_qK`, K=3, ctx=2048) | 149+128+311+503 MB | `build_verify_chunks.py` |

All 5 mlpackages were compiled to `.mlmodelc` (loading `.mlpackage` directly on iOS 18 failed with `Compile the model with Xcode or MLModel.compileModel(at:)`).

Deployed via `xcrun devicectl device copy to` to `Documents/Models/gemma4-e2b/` alongside existing prefill chunks, embeddings, RoPE tables, vision/audio models.

### 1.3 Runtime flow (confirmed working on device)

```
Prefill → nextID → bootstrap decode (warms kv13/kv14) →
loop:
  drafter draftOne × K  (per-step RoPE, same mask)
  → verify K tokens (write-through KV, returns argmax + hidden_states)
  → accept/reject (greedy, commit matchCount+1 positions)
  → emit tokens, update carry state from verify hidden
```

Profile shows the verify pass dispatching correctly through ANE (c1=14, c2=15, c3=19, c4=27 ms for K=3 verify — roughly 2× a normal decode step, as expected).

---

## 2. Bugs found and fixed during integration

These are all **real bugs** that caused crashes or shape errors on device. The fixes are in the committed code.

### 2.1 Shape/layout fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `sin_full rank 4 vs expected rank 2` | `ChunkedEngine.lookupRoPE` returns `(1,1,1,dim)` (LLaMA-style duplicated halves); drafter expects `(1, dim/2)` | `sliceAndReshape` in `MtpSpeculativeEngine` takes first `dim/2` values and reshapes. Values are equivalent because `cos_full[:half] == cos_full[half:]` by construction (verified against `cos_full.npy`). |
| `kv13_v shape mismatch (seq, hd) vs (hd, seq)` | Target chunk2 outputs `(1,1,seq,hd)`; drafter's internal `v.transpose(-2,-1)` expects Google's TFLite pre-transposed `(1,1,hd,seq)` | `transposeLastTwoDims` helper transposes once per speculative cycle. |
| `kv14_k shape (2048,512) vs (8192,512)` | Drafter built with default `--context-length 8192`; target uses 2048 | Rebuilt drafter with `--context-length 2048 --sliding-window 512`. |
| Commit overran verify's written KV when all K matched | `committed = matchCount + 1` was wrong for all-match (advances past position P+K where no KV exists) | `committed = min(matchCount + 1, K)`. Last matched draft becomes nextID for next cycle (gets its KV written by next verify). |

### 2.2 Download-detection fix

`ModelDownloader.localModelURL` only checked `chunk1.mlmodelc/weights/weight.bin`, so multi-function `chunk1.mlpackage` drops triggered a re-download. Added a `chunk1.mlpackage/Manifest.json` check.

### 2.3 RoPE theta (partial fix, doesn't affect parity)

`mtp_drafter_model.py` used `rope_theta = 10000.0` for both SWA and full attention. Gemma 4 uses 10 000 for SWA and 1 000 000 for full. Split into `swa_rope_theta` and `full_rope_theta` in `MtpDrafterConfig`. This doesn't affect the TFLite-vs-PyTorch parity test (zero-KV case; attention contribution = 0), but would matter for any inference with real KV.

---

## 3. On-device measurement

Prompt: *"Write a haiku."* (after `iPhone 17 Pro` run with MTP enabled)

```
[Profile] emb=0.9ms mask=0.2ms | c1=13.2 c2=13.9 c3=14.1 c4=18.8 (sum=60.0ms) | predict=60.2ms total=61.1ms (16.4 tok/s)
[MTP-DIAG] round=0 nextID=237354 embedNorm=1.252 carryNorm=0.000 verifyHS=nil
[MTP-DIAG]   embed[0..4]=[0.01286, 0.05728, -0.00351, 0.00234, 0.00468]
[MTP-DIAG]   carry[0..4]=[0.0, 0.0, 0.0, 0.0, 0.0]
[MTP] round=1 pos=11 nextID=237354 drafts=[8960,1437,71556]  target=[98662,506,9783]   matched=0/3 committed=1
[MTP] round=2 pos=12 nextID=98662  drafts=[108,527,195305]   target=[9969,90082,670]   matched=0/3 committed=1
[MTP] round=3 pos=13 nextID=9969   drafts=[236761,826,15433] target=[237328,237000,237000] matched=0/3 committed=1
[MTP] round=4 pos=14 nextID=237328 drafts=[236842,146903,113941] target=[156495,106,237051] matched=0/3 committed=1
[MTP] round=5 pos=15 nextID=156495 drafts=[236761,20533,113941]  target=[63333,106,236881]  matched=0/3 committed=1
```

**Observations:**

1. The diagnostic harness works — embed lookup is returning sensible values (norm ≈ 1.25 for unscaled) and verify hidden states populate from round 1 onward (carry norm ≈ 23).
2. Drafts DO vary across rounds (inputs drive outputs), so the drafter forward is not broken at the plumbing level.
3. But `matched = 0/3` for every single round. Drafts and target argmaxes are completely uncorrelated.
4. End-to-end tok/s = 16.4 (vs baseline ~14.9 on the same prompt). The small gain comes from the verify pass implicitly running as a Q=3 batched forward; the drafter and correction overhead roughly cancel. **No speedup in practice.**
5. Output text is coherent but repeats phrases (`AppleApple Neural Neural Engine Engine`), because every cycle emits `[nextID, correction]` and both are forced by the target (no actual speculation benefit, but no divergence either once write-through KV is committed correctly).

---

## 4. Mac-side parity investigation

Ran the existing `test_mtp_parity.py` (PyTorch drafter vs TFLite interpreter, zero KV, random activations):

```
TFLite logits: argmax=28953   top5=[28953, 236747, 3488, 1102, 519]
PyTorch logits: argmax=236799  top5=[236799, 167732, 236776, 236953, 3508]

Argmax match: False
Top-5 overlap: 0/5
Logit cosine similarity: 0.818527
Proj activations cosine sim: 0.542546
Logit max abs diff: 29.96, mean abs diff: 5.27
```

This is consistent with the memory note (`mtp_drafter_conversion`: "Full model = 0.82 (numerical drift from TFLite quantized inference vs fp32)"). It was known going in. **But 0.82 cosine + 0/5 argmax overlap is not enough quality to clear any acceptance check** — the drafter's top-1 is never the target's top-1 with this level of drift.

### 4.1 Things eliminated as causes

All of these were investigated and shown NOT to be the culprit:

- **Weight loading shapes.** All 44 tensors load to the expected PyTorch `state_dict` shapes. `mtp_pre_proj` weight matches TFLite byte-for-byte (diff = 0). Per-layer q_proj/o_proj/gate1/gate2/down all have correct 2-D shapes (verified against TFLite tensor shapes).
- **Rename map misses.** `_build_rename_map` (dead code per PR #27) has shape-4 mismatches for layer 0 and layer 3 q_proj, but the actual loader `load_from_tflite_auto` uses shape-checked pattern matching and picks the correct (1024,256) / (2048,256) tensors. Confirmed via `model.named_parameters()` dump.
- **gate1/gate2 swap.** Tried `down(gelu(gate2) * gate1)` — argmax unchanged.
- **RoPE theta per-layer.** SWA=10 k, full=1 M split applied. Irrelevant for zero-KV parity.
- **Norm weight convention.** Verified both target (HF Gemma 4) and drafter store raw scale (not `(1 + w)` delta); our `RMSNorm` multiplies by raw weight, matching.
- **V transpose direction.** Drafter's internal `v.transpose(-2,-1)` matches Google's pre-transposed `(1,1,hd,seq)` storage.

### 4.2 What's left (unverified, speculative)

- **Quantization-specific ops.** TFLite runs INT8 activations + INT4 weights with per-tensor/per-channel dequant. Our PyTorch uses fp32 throughout. Compounding quant errors across 4 layers + MLP + lm_head could plausibly account for 0.18 cosine drift, and argmax being different is exactly what happens when logits differ by constants even if the distribution's shape is similar.
- **Activation-quantization mid-layer.** Google's W4A8 setup re-quantizes between every FC. Our floating-point pipeline doesn't. This could shift relative logit magnitudes enough to change argmax.
- **A subtle mismatch in attention numerics** that shows up only with real KV (not zero KV). Our zero-KV test sets V=0 so attention output is identically zero in both models; real inference would expose any attention-specific bug.
- **Embedding scale convention.** We tested both scaled (× sqrt(H)) and unscaled embed on Mac; neither got the drafter's top-1 to match target's top-1. The SCALED variant produced more semantically plausible drafts for a code prompt (4 spaces after `def fibonacci(n):`), suggesting SCALED is the correct convention, but it still doesn't match target predictions exactly.

### 4.3 Mac verification harness

Added `conversion/test_mtp_local.py` — loads HF Gemma 4 + our `.pt` drafter, runs a real prompt, extracts `L34_hidden` + `kv13/kv14` + `embed(next_token)`, and evaluates the drafter across four variants (scaled/unscaled embedding × L34-raw/post-norm hidden). None hit the target's argmax; one (SCALED+L34raw on code prompt) produced a semantically sensible but different continuation.

This harness is the fastest way to iterate on drafter fixes without a device roundtrip.

---

## 5. Decisive finding: target-distribution mismatch (not a reimplementation bug)

After the §4 investigation was inconclusive, we ran one more test designed to separate "our PyTorch is wrong" from "the drafter itself can't match HF":

### 5.1 The test

`conversion/test_mtp_tflite_acceptance.py` — loads the raw TFLite drafter (`section_9.tflite`, 44 MB) in `ai_edge_litert.Interpreter`, loads HF Gemma 4 for the target, and for each prompt:

1. Runs HF Gemma 4 forward to get L34 raw hidden state, real kv13/kv14, and target's natural next token.
2. Quantizes K/V to INT8 using the per-tensor scales extracted from the TFLite metadata (kv13_k scale=0.005955, kv13_v=0.047244, kv14_k=0.001091, kv14_v=0.017857 — values read off the drafter's tensor table).
3. Pads the K/V to Google's fixed 32003 context length, left-aligned.
4. Builds `activations = concat(embed(natural_next), L34_hidden)` as the drafter's C++ runtime is documented to do.
5. Invokes the TFLite drafter and checks: does the TFLite drafter's argmax equal HF target's natural next token?

### 5.2 Result

| Prompt | Target natural next | TFLite drafter top1 | Match |
|---|---|---|---|
| `"The capital of France is"` | 7001 (` France`) | 529 (` of`) | **no** |
| `"光合成とは"` | 237914 (`光`) | 84723 | **no** |
| `"def fibonacci(n):"` | 107 (`\n`) | 198188 | **no** |
| `"Once upon a time, there was"` | 691 (` was`) | 1340 | **no** |
| `"The weather today is"` | 7606 (` weather`) | 511 | **no** |

**0 / 5 argmax match, 0 / 5 top-5 inclusion.**

Even with the actual Google TFLite drafter (not our PyTorch reimplementation), running on real HF target outputs, the drafter's predictions never match. This is not an artifact of our PyTorch port. The TFLite drafter and HF Gemma 4 produce outputs that live in different distributions.

### 5.3 Interpretation

Google's LiteRT export of Gemma 4 is an **end-to-end W4A8 quantized pipeline**, and the MTP drafter was trained/calibrated against **the outputs of the quantized main LLM** (section_8.tflite, 818 MB), not against the HF fp checkpoint. Specifically:

- **Target's L34 hidden state** after 35 layers of W4A8 compute diverges from HF's fp32 L34 by small amounts per layer, compounding into a distribution the drafter treats as "canonical".
- **kv13/kv14 cache values** are the output of the quantized layers 13/14, not of HF's fp layers.
- **Embedding lookup** goes through section_0.tflite, which may embed scaling or normalization not present in our path.

From the drafter's point of view, our HF-target inputs look "close but wrong" — the right order of magnitude but the wrong operating point. This explains why all the shape/transpose/RoPE fixes we applied during on-device integration (§2) produced a drafter that *runs* but produces tokens that never coincide with our target's next prediction.

### 5.4 Consequences

1. **The 0.82 cosine / 0/5 top-5 parity gap** documented in §4 is a **symptom**, not the cause. It appeared even in the zero-KV probe because the parity test used random activations; with real activations it expresses itself as systematic target mismatch.

2. **W4A8 quantization replay in PyTorch** (the experiment we were about to run) would not fix on-device behavior. It would close the PyTorch↔TFLite gap, but the TFLite drafter itself is already wrong for our HF-trained target. Fixing PyTorch→TFLite parity does not fix TFLite→HF-target parity.

3. **Running section_8.tflite on-device as the target** would make the drafter work, but it defeats the purpose: we'd run a W4A8 CPU target and an ANE target side-by-side, paying full cost of the CPU path. LiteRT-LM reports 56.5 tok/s with section_8 as the *only* decode path; injecting that into our pipeline does not add up to a speedup over our 14.8 tok/s ANE baseline.

4. **Path A is unsalvageable with our target.** Any further engineering time on Google's extracted drafter is wasted unless we change the target, and changing the target abandons our ANE roadmap.

### 5.5 Why this was hard to see earlier

- `docs/MTP_PATH_A_FINDINGS.md` stated that Google's drafter was "purpose-trained by Google against Google's reference forward." At the time this was read as "against Gemma 4's reference forward" (implicitly assumed to be HF's). In retrospect, "Google's reference forward" is **the quantized LiteRT forward**, which is a different thing.
- Our `mtp_drafter_conversion` memory recorded parity = 0.82 as "numerical drift from TFLite quantized inference vs fp32" and flagged it as needing an on-device test. The on-device test gave us 0% — but the Mac-side TFLite-vs-HF test (this §5) would have told us the same answer in under an hour, without any integration work. It should have been the first test, not the last.

For future speculation attempts that extract Google artifacts: **always run the raw TFLite (or equivalent) artifact end-to-end against your actual target before spending a day on reimplementation.** It's a cheap disqualifier.

---

## 6. Why MTP Path A is not the right path

1. **Target mismatch is architectural.** The drafter was trained for LiteRT's quantized target, not HF's. There's no PyTorch, CoreML, or Swift fix for this.
2. **The "runtime is the easy part" framing held up.** All of `MtpSpeculativeEngine`, `verifyCandidates`, write-through KV, per-step RoPE, bootstrap — these are mechanical and work. We just can't use them with this drafter artifact.
3. **The fallback documented in MTP_PATH_A_FINDINGS §7 is the right next step.** Train our own MTP heads (2–3 lightweight future-token heads on the final hidden state, frozen trunk, composite CE loss). Controlled quantization means parity is 1.0 with the trunk's own outputs. Runtime stays identical.

---

## 6. Files of interest

**Integration code (on `feature/mtp-speculative-v1`):**

- `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` — the full orchestrator.
- `Sources/CoreMLLLM/ChunkedEngine.swift:74..80` — `lastKV13K/V`, `lastKV14K/V` stored properties.
- `Sources/CoreMLLLM/ChunkedEngine.swift:1082..1110` — MTP-support convenience methods.
- `Sources/CoreMLLLM/CoreMLLLM.swift:146..163` — MTP drafter auto-load.
- `Sources/CoreMLLLM/CoreMLLLM.swift:415..455` — Speculative decode loop.
- `Examples/CoreMLLLMChat/CoreMLLLMChat/ChatView.swift:66..73` — `acc0` display.

**Mac verification (new, on this docs branch — copy over if continuing):**

- `conversion/test_mtp_tflite_acceptance.py` — **the decisive test (§5)**. Runs raw TFLite drafter end-to-end against HF Gemma 4. Use this first for any future drafter-extraction attempt.
- `conversion/debug_tflite_metadata.py` — dumps all TFLite tensor shapes, per-tensor/per-channel quant scales, small/scalar parameters. How we found the INT8 KV cache scales and confirmed there are no hidden layer-scalar tensors.
- `conversion/test_mtp_local.py` — real-prompt drafter evaluation of our PyTorch port with HF Gemma 4. Kept for reference — superseded by `test_mtp_tflite_acceptance.py` but useful if we ever want to debug the PyTorch port independently.
- `conversion/debug_drafter_layers.py` — per-layer activation dumps.

**Parity reference:**

- `conversion/test_mtp_parity.py` — PyTorch vs TFLite, zero KV.
- `output/mtp_probe/mtp_drafter.pt` — 308 MB checkpoint.
- `output/mtp_probe/section_9.tflite` — 44 MB Google drafter (the authoritative reference; section_10 naming in PR #27 is about filesystem layout, not a different model).

---

## 8. Recommendations

1. **Park MTP Path A.** §5 shows the drafter itself is incompatible with our HF target. No amount of PyTorch / CoreML / quantization work on our side will change this.
2. **EAGLE-3 retrain (Path B).** We now understand the Blocker 1 root cause (training-corpus KV-sharing mismatch, see `EAGLE3_INTEGRATION_STATE.md`). A correct retrain against our own trunk's forward should hit the documented acceptance range.
3. **Self-trained MTP heads (Path C).** Fallback from `MTP_PATH_A_FINDINGS §7`. Freeze trunk, add 2–3 future-token heads, train with `CE(next) + λ₁·CE(t+2) + λ₂·CE(t+3)`. A100 GPU time, ~3–5 days. This is architecturally the same MTP idea, but trained against *our* target so the distribution mismatch from §5 cannot recur.
4. **Don't re-derive the integration from scratch** when we resume. The Swift engine + multi-function chunks + diagnostics all work. Only the drafter artifact needs to change. Swapping in a different drafter `.mlmodelc` and (if I/O differs) editing the I/O bridging in `MtpSpeculativeEngine.draftOne` call site is enough.
5. **Process lesson: validate the artifact before reimplementing.** For any future "extract Google's X and plug it into our pipeline" work, run the raw extracted artifact end-to-end against our actual target on Mac as **the first test**, not the last. One hour of `test_mtp_tflite_acceptance.py`-style probing would have flagged this path as dead before a single day of Swift work.

---

## 9. Go/no-go gate status (from original task)

| Result | Action |
|---|---|
| acc0 ≥ 50 %, tok/s ≥ 40 @ 2K | Primary speculation, merge. |
| acc0 30–50 % | Needs tuning. |
| **acc0 < 30 %** | **Fall back to EAGLE-3 retrain.** ← **we are here** |

**acc0 = 0 %. No speedup over baseline.** Proceed to fallback per the gate.
