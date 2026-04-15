# MTP Path C — Self-trained drafter findings

**Status:** 2026-04-15. Full pipeline (train → convert → deploy) completed. **iPhone 17 Pro result: Path C is net negative vs baseline — ~16 tok/s vs 31 tok/s baseline, with subtle output drift.** Shelved in favor of the DrafterUnion path (Phase B) and item 11c (verify-chunk numerical alignment).

**TL;DR.** We self-trained DeepSeek V3-style K=2 sequential MTP modules against our own frozen Gemma 4 E2B trunk (no LiteRT dependency), fixed three structural bugs via Mac-first verification, confirmed end-to-end numerical equivalence between PyTorch and CoreML, and deployed. On iPhone 17 Pro with coherent English prompts we measured ~16.3 tok/s at acc0=17 % — **about half of the 31 tok/s baseline**. The drafter is correct; the gap is driven by the same target-side fp16 divergence between `verify_qK` and `decode_q1` chunks that was surfaced in PR #54 and filed as PRIORITY_ROADMAP item 11c. With item 11c unresolved, **no self-trained drafter can break even on this target**: per-cycle verify cost is ~2.3× decode on ANE, which requires ~77 % acceptance just to match baseline. Training another drafter doesn't buy that — closing item 11c does. **Recommendation: park Path C, resume Phase B, treat item 11c as the unlock for any future MTP effort (Path A, Path C, or EAGLE-3).**

Branch of record: `feature/mtp-speculative-v1`. Final commits: `f5bbac0`, `881500c`, `ad0dc91`. All Mac-side tooling that caught the bugs is in `conversion/train_mtp_modules/`.

---

## 1. What Path C is (vs Path A)

Path A (parked 2026-04-14, see `MTP_INTEGRATION_RESULTS.md`): extract Google's TFLite MTP drafter from `gemma-4-E2B-it.litertlm`, re-implement in PyTorch, convert to CoreML. Failed at 0 % acc0 because the drafter was trained against LiteRT's W4A8 quantized target — incompatible with our HF fp target at the artifact level, unfixable in post.

Path C: **train our own drafter against our own target's hidden states.** No upstream compatibility question by construction. Two sequential transformer modules (~80 M params each, Gemma SWA layer shape) following DeepSeek V3's K=2 MTP formulation:

- Module 0 input: `(L34[t], embed(tokens[t+1])) → target: tokens[t+2]`
- Module 1 input: `(h_0[t], embed(tokens[t+2])) → target: tokens[t+3]`

Deploy via the existing multi-function `verify_qK` chunks (K=3). The Swift side (`Sources/CoreMLLLM/MtpModuleStackEngine.swift`) runs both modules sequentially, then one K=3 verify, accepts/rejects, commits, and carries the committed-position L34 hidden from verify's `hidden_states_out` into the next cycle. Auto-loads when `mtp_module_0.mlmodelc` + `mtp_module_1.mlmodelc` are present alongside verify-capable chunks.

Training pipeline:

- `conversion/train_mtp_modules/precompute.py` — runs frozen Gemma 4 E2B trunk on a corpus, caches `(token_ids, last-layer-pre-norm)` shards.
- `conversion/train_mtp_modules/train.py` — trains `MtpStack` against cached pairs with DeepSeek V3 indexing.
- `conversion/train_mtp_modules/build_mtp_coreml.py` — loads one trained module, traces to CoreML mlpackage.
- `conversion/train_mtp_modules/verify_coreml_equiv.py` — Python vs CoreML numerical equivalence test.
- `conversion/train_mtp_modules/verify_accept_logic.py` — pure-Python mirror of the Swift accept/emit logic, scenario-based.

Corpus: 16.2 M tokens over 64 shards (50 GB cache), 5 datasets (`fineweb-edu`, `oasst1`, `codealpaca`, `wikitext`, `codesearch`). Training: 3 epochs, K=2 loss weights `[1.0, 0.8]`, AdamW bf16, cosine LR on A100, ~18 min wall-clock after precompute.

Final training stats: `eval L1=4.25 acc=30.6 %`, `eval L2=4.11 acc=33.2 %` at step 2000; EMA at step 5800 was `L1 acc=37.8 %, L2 acc=40.2 %`.

---

## 2. Three structural bugs caught before the iPhone trip

All three were caught on Mac in <30 minutes of verification each. All three would have silently wasted one or more full Colab overnight runs. Commits carry the full context.

### 2.1 DeepSeek V3 indexing was shifted by one (commit `f5bbac0`)

**Symptom before fix:** training converged to val acc 54.9 % (module_0) / 61.9 % (module_1) on the old 7 M-token cache — looked great. iPhone acc0 = 0 %.

**Cause:** `mtp_modules.py` was using `tok_k = token_ids[:, k:k+T]` and training target `tokens[:, k+1:k+1+T_eff]`. For module_0 that's `embed(tokens[t]) → tokens[t+1]` — **exactly the task the trunk's own `lm_head` already solves**. Module_0 learned to mimic `lm_head`, validating with inflated accuracy, and producing drafts that duplicated what the trunk would emit at position `t+1` rather than predicting `t+2`.

**Fix:** shift embed input by `+1` and target by `+2`, matching DeepSeek V3 paper Figure 3:
- `tok_k = token_ids[:, k+1:k+1+T]` (embed = *next* token)
- `target = tokens[:, k+2:k+2+T_eff]` (predict *two* tokens ahead)

Affects `mtp_modules.py`, `train.py`, `data.py` (shard length bookkeeping), `smoke_test.py`. The Swift side was already correct (`hiddenIdx = matchCount` gives the L34 at the last-committed position, which the new training expects).

### 2.2 Precompute extracted the wrong layer (commit `881500c`)

**Symptom:** after the indexing fix retrained, training acc looked normal but the deployed drafter would still have seen a *different* hidden state than training data.

**Cause:** `precompute.py` used `out.hidden_states[-2]` as "L34". Empirical test on HF Gemma 4 E2B:

| Tensor | Norm (3-token input) | Meaning |
|---|---|---|
| `hidden_states[-2]` | 120.2 | **L33 output** (penultimate layer, pre-norm) |
| `hidden_states[-1]` | 475.2 | Post-final-norm (= `last_hidden_state`, fed to `lm_head`) |
| `layer[-1]` pre-norm via forward hook | 49.9 | What `SWAVerifyChunk4` returns as `hidden_states_out` |

Training fed L33 (norm ~120), deployment gives chunk 4's last-layer pre-norm (norm ~50). Different layer, 3× different magnitude. Untrainable through.

**Fix:** register a forward hook on `lm.layers[-1]` during precompute and capture its pre-norm output directly. Existing cache was structurally unusable and had to be regenerated (another ~3 h on A100).

### 2.3 CoreML topk silently truncated indices to 16 bits (commit `ad0dc91`)

**Symptom:** `verify_coreml_equiv.py` on the converted drafter reported `hidden_out` matching within 0.035 (rel 0.9 %), KV caches matching within 0.025, but **top_k_indices differed by exactly `k*65536`** for the top 4 positions. PT idx `236764`, ML idx `40156`, delta `196608 = 3 × 65536`. A 16-bit wrap of a 262 144-vocab index.

**Cause:** `torch.topk(k=8)` on an isolated V=262 144 tensor round-trips fine through coremltools 9.0. The bug only appears after the full `MtpModuleANE` graph — specifically the `softcap → squeeze → topk` tail. Likely a coremltools bug on this specific op composition when applied to a 1D tensor with V > 65 536. Not investigated to root — sidestepped.

**Fix:** replace `torch.topk(k=8)` with `torch.argmax`, emit a `(1,)` int32 tensor under the same output name (`top_k_indices`) for Swift compatibility. Swift side reads `topIds.pointee` (first element), so the shape change is transparent. Equivalence test now passes: PT argmax `236764` == ML argmax `236764` on module 0, `107 == 107` on module 1.

**Why this is load-bearing:** this is a pure runtime bug. Had we not run the equivalence test on Mac, every drafted token on iPhone would have been the correct argmax's lower-16-bit alias — a random token. Every previous Path A acc0=0 % result would have had this as a co-confound if Path A's drafter had ever produced real argmaxes to begin with. Future MTP-style drafters should include a Python-vs-CoreML argmax parity check as a mandatory gate.

### 2.4 Meta-point: the methodology, not just the bugs

The 3 bugs above share a pattern. Each was a **structural mismatch between training and deployment that training loss can't signal**: the loss goes down on the wrong objective, or on the wrong inputs, or the output gets bit-scrambled on the way out. None are detectable from training curves. All three are detectable in minutes from a Mac-side parity test.

The pipeline now includes three such harnesses, listed here so future MTP work doesn't regress them:

- `verify_accept_logic.py` — pure-Python scenario replay of `speculateStep` accept/emit rules. Catches double-emission, KV commit off-by-one, carry-index errors.
- `verify_coreml_equiv.py` — PT MtpModuleANE vs the converted mlpackage on identical random inputs. Asserts `hidden_out` max-abs-diff < 0.2, argmax match. Catches conversion-side numerical drift and op-composition bugs (e.g. §2.3).
- Forward-hook parity check in `precompute.py` — the hidden-state capture is now at the layer chunk 4 actually returns, not a guessed-by-name tuple index.

Add any future equivalent harness under `conversion/train_mtp_modules/` and document it here.

---

## 3. Mac-side end-to-end result

Deploy bundle: copy of `staging-2k-fast-prefill/gemma4-e2b` (2K ctx, multi-function verify_qK K=3 chunks, prefillN=512) with the two compiled `mtp_module_{0,1}.mlmodelc` added. No other changes. `Sources/MtpMacCheck` auto-loads Path C on seeing both modules.

Two prompts run:

**English — "Hello"** (32 tokens max):

```
Output: "Hello! How can I help you today?"  (coherent, no duplicates)
tok/s: 13.80
mtp acceptance rate: 16.7 %
mtp tokens per round: 1.33
```

Baseline on the same deploy dir with both `mtp_module_*.mlmodelc` removed (so the engine falls back to non-speculative):

```
Output: "Hello! How can I help you today?"  (same)
tok/s: 30.60
```

**Japanese — "こんにちは"** (32 tokens):

```
WITHOUT MTP (decode only):   "こんにちは！何かお手伝いできますか？ 😊"    ✓ clean
WITH MTP (speculation on):   "こんにちは！😊\n\nしゅゅゅゅゅゅゅゅ…"       ✗ locks into "ゅ" loop
                             tok/s 23.16, acc 64.3 %, tokens/round 2.29
```

Same model, same prompt. Toggling Path C changes the main model's argmax at the second generated token (`何か` → `😊`), and the distribution degenerates into a repeat-one-token trap within ~5 tokens. Acceptance inflates because the drafter and the drifted main-model path agree on the degenerate token.

This is not a correctness bug in Path C — the main model is running the same weights either way. The only difference between the two traces is whether position P+1 is written by `decode_q1` (one token at a time) or by `verify_qK` (three tokens batched). See §4.

---

## 4. Root cause: verify-vs-decode fp16 divergence is target-side

`verify_qK` and `decode_q1` are two functions of the same multi-function mlpackage. They share weights exactly. At position P they compute the *same* math under a causal mask. But they are two separately-compiled ANE programs, with different tensor shapes, tiling, and SIMD schedules, run on the same fp16 device logic. Numerically they drift.

For most inputs the drift is below the argmax-flipping threshold. For some — in our test, Japanese character continuations after "こんにちは！" — the top-1 margin is narrow enough that the drift flips argmax. Once the KV history contains a token the pure-decode path would never have emitted, subsequent positions compound the divergence.

This is the phenomenon filed in `PRIORITY_ROADMAP.md` as **item 11c — Verify-chunk numerical alignment (K=3 ↔ K=1)**. Path C is an independent confirmation of it, from a different drafter (not Qwen cross-vocab, not Path A), with a clean Mac equivalence test on the drafter itself. **The drift is in the target's chunks, not in any drafter we've tried.**

### 4.1 Speed math: why any Path-C-like drafter is net negative today

The drafter itself is cheap on ANE; the per-cycle cost is dominated by the K=3 verify. Measured on iPhone 17 Pro:

- Baseline decode: 31 tok/s → 32 ms / token.
- Path C verify (one cycle): produced 16.3 tok/s with acc0 = 17 %.

Expected tokens per cycle (K=2 drafts + 1 bonus):

```
E[tokens] = 1 + p0 + p0*p1
         ≈ 1 + 0.17 + 0.17*0.17
         ≈ 1.20
```

Cycle cost relative to decode:

```
C_cycle = E[tokens] × (decode ms/token) / (measured ms/token)
        = 1.20 × 32 / (1000/16.3)
        ≈ 1.20 × 32 / 61.3
        ≈ 0.63  ← speculation path is 0.63× the throughput of decode
```

Inverting, the per-cycle cost is `1.20 / 0.63 ≈ 1.9×` decode. Drafter forward is ~0.1× decode (small model, ANE-friendly shape), so **verify alone is ~1.8–2.3× decode** on this device. Budget:

```
Break-even acceptance (K=2):
  1 + p + p² ≥ verify_multiplier
  ≥ 2.3    →    p ≈ 0.77
```

To beat baseline we need **~77 % acceptance per module**. Our training landed at 38 % val acc; on-device acc0 ran 9–17 % (quality-drift loop eats some of the gap). No amount of more data pushes 38 % to 77 % for an 80 M-param drafter at this capacity tier — that regime is "bigger drafter + more data + more epochs", a multi-week investment with heavy diminishing returns.

Closing item 11c, by contrast, attacks the multiplier directly. If verify chunks can be made to dispatch closer to `K × decode` wall-clock (currently they land at ~2× instead of ~3× — the K=3 batching already saves some, but dispatch overhead dominates), the break-even acceptance drops fast and existing acc rates become useful.

---

## 5. Shelving decision

**Status:** Path C is shelved. `feature/mtp-speculative-v1` stays on origin as a reference branch; the fix commits (`f5bbac0`, `881500c`, `ad0dc91`) and the Mac-side verification scripts are the lasting artifact. No docs-only PRs already open reference it by handle beyond this file and the cross-links below.

Rationale:

- Path C is not blocked by a bug in our drafter, training, or conversion. The drafter is correct. The deployment is correct.
- The blocker is item 11c — target-side numerical drift between `decode_q1` and `verify_qK`. That is load-bearing for **every** speculative drafter on this target (Path A, Path C, Qwen cross-vocab, prompt-lookup, future EAGLE-3). Training a third or fourth drafter doesn't help.
- The main line of work (Phase B DrafterUnion, see `SESSION_STATE.md` and `MAC_FIRST_EXECUTION_PLAN.md`) is independent of Path C's outcome. Nothing that session does depends on whether Path C ships.
- `mtpEnabled` remains `true` as a default — Path A's code path is dormant (no drafter on the iPhone bundle) and Path C's `MtpModuleStackEngine` only auto-activates if both `mtp_module_*.mlmodelc` files are on the device, which the current shipped bundle does not carry. No user-visible surface area changes from this decision.

### 5.1 Re-open criteria

Any of the following unlocks re-opening Path C:

1. **Item 11c closes** with a concrete reduction in verify multiplier (e.g. ~1.3× decode). At 1.3× break-even, our existing 38 % trained acc clears and Path C becomes net-positive without retraining.
2. **Drafter capacity grows to where acc_0 consistently exceeds ~65 %** on this target. Requires bigger modules (multi-layer, multi-hundred-M params) and a training corpus closer to 100 M tokens. Would need separate justification vs DrafterUnion's measured gains.
3. **A different acceleration path becomes attractive for Path C specifically** — e.g. running drafters on GPU while ANE runs verify concurrently (Phase C's Mirror SD, `docs/MAC_FIRST_EXECUTION_PLAN.md`). Path C's drafters are small and fp16, plausibly fast on GPU; if GPU dispatch hides the drafter cost entirely, break-even math shifts.

### 5.2 What to keep building on even with Path C shelved

- The Mac-first verification methodology (§2.4). Three harnesses caught three structural bugs in one day. This pattern generalizes — every future drafter trip should open with its own equivalence test before any iPhone build.
- `precompute.py`'s forward-hook approach to extracting a specific layer's pre-norm output. Applicable to any distillation/speculation training that has to match what a chunked target serves at inference.
- The DeepSeek V3 MTP indexing in the training code. If someone else wants to train a K>2 drafter later, this is the correct baseline to fork.
- `scripts/measure_verify_drift.sh` — a reproducible, iPhone-free handoff artefact for whoever investigates item 11c next (§7).

---

## 6. (reserved)

---

## 7. Handoff for item 11c: Mac-side drift measurement harness

`scripts/measure_verify_drift.sh` runs `MtpMacCheck` twice against the same deploy directory — once with the two `mtp_module_*.mlmodelc` moved aside (so the speculative path is disabled and generation stays on `decode_q1`), and once with them in place (so generation goes through `verify_qK`). The only variable between the two runs is which chunk function produces each position's hidden state. Any token-stream divergence between the two runs is item 11c's drift surfacing on the target's chunks, in isolation from drafter quality.

### Use

```bash
scripts/measure_verify_drift.sh <deploy-dir-with-mtp_modules> "<prompt>"
```

Required inputs in `<deploy-dir>`: verify-capable `chunk1-4.mlmodelc` (i.e. multi-function with both `decode_q1` and `verify_qK`) plus both `mtp_module_0.mlmodelc` and `mtp_module_1.mlmodelc`. `staging-2k-fast-prefill/gemma4-e2b` + Path C's compiled modules meets this exactly.

Outputs under `/tmp/verify_drift/`:

- `base_tokens.txt` — tokens produced by `decode_q1`-only generation.
- `spec_tokens.txt` — tokens produced with speculation on.
- `diff.txt` — unified diff.
- Console summary: first-flip position, flip rate over the common prefix, baseline vs speculation tok/s, mtp acceptance.

### Recorded baselines (2026-04-15, Path C modules trained on 16.2 M tokens)

| Prompt | Base tok/s | Spec tok/s | Acc rate | First flip | Flip rate over common prefix |
|---|---|---|---|---|---|
| `"Hello"` | 30.3 | 13.8 | 16.7 % | — | 0 / 9 = 0 % |
| `"こんにちは"` | 30.4 | 22.8 | 64.3 % | position 3 | 8 / 10 = 80 % |

Interpretation:

- **English:** no drift. `decode_q1` and `verify_qK` produce the *same* argmax at every position the two paths both emit. The ~55 % tok/s loss is entirely the verify-cycle overhead (~2× decode) relative to the number of accepted drafts; no correctness issue. If item 11c only addressed the overhead multiplier (e.g. reducing per-cycle cost), English workloads would become net-positive at current training acceptance rates.
- **Japanese:** drift at position 3. By the time the model has emitted "こんにちは", "！", and one more token, the two paths agree; from position 3 onward the speculation path emits "😊" while decode emits "何か" and subsequent positions cascade. Acceptance inflates to 64.3 % because once the main model enters the degenerate "ゅ" loop, every drafter prediction agrees. This is the case that argues item 11c is a **correctness** concern, not purely perf.

### Success metric for an item 11c patch

A candidate 11c fix (reconverted chunks with fp32 accumulator, altered SIMD tiling, fp32 logit cast before argmax, etc.) is validated by rerunning this script on the patched chunks and checking that the Japanese flip rate drops from ~80 % toward 0 %. English flip rate is already 0 %; the English branch is the no-regression check.

The script is intentionally zero-dependency: just builds `MtpMacCheck`, runs it twice, diffs. No Colab, no iPhone, no Python env. Iteration cost ≈ 2 × runtime of one MtpMacCheck pass (~5 minutes incl. cold-start ANE compile on Mac).

---

## 8. Cross-links

- `docs/PRIORITY_ROADMAP.md` item 11c — the verify-chunk alignment blocker, upgraded from "could lift accept rates" to "load-bearing gate for any MTP drafter on this target" by this session's evidence.
- `docs/MTP_INTEGRATION_RESULTS.md` — Path A (TFLite drafter) findings. Complementary failure mode (target-distribution mismatch vs target-runtime-drift).
- `docs/MTP_PATH_A_FINDINGS.md` §7 — originally sketched Path C as the fallback for Path A. Keep for the I/O contract and the "why Path A failed" reasoning; Path C's own result supersedes §7's speculation about what Path C would look like.
- `docs/SESSION_STATE.md` Phase B Task 4 (output quality investigation) — this doc gives that task an independent corroboration that the issue is 11c and not a drafter-specific bug.
- Branch: `feature/mtp-speculative-v1` on `origin`. Commits `f5bbac0`, `881500c`, `ad0dc91` are the final fixes; the Mac-side tooling lives under `conversion/train_mtp_modules/`.
