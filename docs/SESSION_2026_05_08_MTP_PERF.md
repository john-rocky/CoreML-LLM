# MTP iPhone perf — 2026-05-08

iPhone 17 Pro Gemma 4 E2B MTP path. Result: **yes-yes 32 → 61 tok/s
(+90%)** with three layered fixes. Handoff doc — what shipped, what's
still open, how to keep going.

## TL;DR

| Change | Effect |
|---|---|
| `PALETTIZE_KEEP_FP_KV=1` rebuild of verify chunks | iPhone ANE 18 INT4 K-cache contamination eliminated → drafter accept 0.30 alt → 0.85 sustained |
| `lastKV14V_T` dual-buffer in ChunkedEngine | per-MTP-round 21 ms full transpose → 0 ms (incremental column update + zero-init) |
| 6-core parallel tiled 64×64 transpose for kv13V | 9 ms → 4 ms |
| `commit=` measurement in SpecProfile | exposed the 87 ms gap that turned out to be transpose14 |

`yes 30 times` UI-displayed throughput on iPhone 17 Pro: 61.3 tok/s.
Engine `[MTP cycle]` steady state: 41–45 ms / 3 emit.

## What's still open

- Real-world chat (free-form text, code) on iPhone is back at baseline
  ~32 tok/s. HF transformers Mac MPS bf16 reference is **also ~1.13×**
  on `household` greedy — so this is drafter-model-bounded, not a bug
  on our side.
- HF at **`temperature=1.0` rejection sampling** lifts `household` to
  **1.52×** and `translate` to 1.68×. We attempted this — see
  "Rejection-sampling attempt" below for why it stalled.

## Rejection-sampling attempt (2026-05-08, follow-up)

Tried to bring HF's temp=1.0 sampling lift to the Swift CoreML path.
End result: code is in place but the win does not materialize, and
the verify-chunk graph change required to expose logits regresses
greedy accept on Mac. Branch shipped greedy-only; sampling path is
opt-in via `MTP_TEMPERATURE` env but should be considered experimental.

### What was built

- `conversion/models/gemma4_swa_chunks.py:SWAVerifyChunk4` gained an
  `emit_logits=True` mode that returns the post-softcap logits
  `(1, K, vocab)` fp16 alongside `token_ids` + `hidden_states`.
- `conversion/build_verify_chunks.py` got `--emit-logits` to opt in.
- `Sources/CoreMLLLM/ChunkedEngine.swift` exposes
  `lastVerifyLogits: MLMultiArray?` populated when chunk 4 has the
  output (no-op otherwise).
- `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` got a sampling
  branch in `speculateStep`: drafter samples its top-8 with
  temperature, target verifies via full-vocab softmax, accept with
  `min(1, p_t/p_d)`, residual sample on reject. Greedy path is
  unchanged when `samplingTemperature == 0`.
- `Sources/CoreMLLLM/CoreMLLLM.swift` reads `MTP_TEMPERATURE` env at
  load time and sets the property when > 0.
- Built `output/gemma4-e2b/verify_chunks_fp16kv_logits/` (compiled,
  ready to push) and a Mac test bundle `bundle_logits/` (chunk1-4
  swapped, the rest symlinked).

### What we learned (Mac, prompt = "Repeat yes 30 times…")

| Build | greedy tok/s | greedy accept | sampling tok/s | sampling accept |
|---|---|---|---|---|
| `bundle/` (INT4 K/V) | 66.5 | 0.65 | n/a (no logits) | n/a |
| `bundle_fp16kv/` (fp16 K/V) | 64.0 | 0.62 | n/a (no logits) | n/a |
| `bundle_logits/` (fp16 K/V + emit_logits) | **47.1** | **0.32** | 42.3 | 0.33 |

Two distinct findings:

1. **Adding `logits_fp16` as a verify-chunk output halves Mac drafter
   accept** (0.62 → 0.32) even though all 4 chunks have **byte-identical
   weights** between the fp16kv and fp16kv+logits builds. The graphs
   differ only in whether the lm_head's post-softcap fp16 tensor is a
   named program output or an internal intermediate. CoreML's ANE
   compiler appears to fold/promote-precision differently when the
   tensor is internal vs an output, and that small numeric drift is
   enough to cut drafter agreement in half on Mac. iPhone behavior
   unknown — but plausibly the same regression.
2. **Rejection sampling is neutral on yes-yes and doesn't help on
   free-form** because our centroid MTP drafter only emits top-8 ids
   per step. The HF reference assistant emits full vocab logits, so
   its sampler can sometimes hit a target-favoured token outside the
   drafter argmax. Our top-8 cap forces near-zero `p_d` for any
   token target really wants, which is exactly the regime where
   `min(1, p_t/p_d)` gives ≈ 0 acceptance.

### What it would take to actually win

- **Greedy regression fix**: split chunk 4 verify into two functions
  (`verify_qK` argmax-only, `verify_qK_logits` argmax + logits). Greedy
  callers stay on the existing graph, sampling callers pay for the
  extra output. Requires `build_multifunction` to support 3 functions
  per package and Swift-side dual-loading.
- **Drafter rebuild**: the centroid + top-8 LM head was the win that
  closed the runtime gap with HF (commit `d2559a3`), but it caps the
  drafter's effective vocabulary at ~4096 tokens (32 × 128). For
  rejection sampling to work the drafter needs full-vocab logits or
  at minimum top-K ≥ 256. Either re-export the existing drafter to
  emit a wider top-K, or train/import a non-centroid drafter.
- **Or skip sampling**: empirical HF gain on yes-yes is +1.5%; the
  big wins (`household` 1.13→1.52×, `translate` 0.74→1.68×) require
  the full-vocab drafter. Greedy + dual-buffer + fp16 K/V already
  delivers the iPhone 32 → 61 tok/s win on yes-yes. Free-form chat
  is drafter-model-bounded; closing it needs a better drafter.

### Files left in the tree (not committed)

- `output/gemma4-e2b/verify_chunks_fp16kv_logits/` (built mlpackages
  + mlmodelc — ~1.1 GB on disk, can be deleted)
- `output/gemma4-e2b/bundle_logits/` (Mac test bundle with chunks
  swapped in, rest symlinked — small)
- `output/gemma4-e2b/bundle_fp16kv/` (Mac test bundle for the
  intermediate fp16kv-only A/B)

The Swift code changes are minimal and behind `MTP_TEMPERATURE`
env, so leaving them in place is safe — they don't activate unless
both the env is set and the verify model exposes logits.

### Next-session checkpoint

If you pick this back up: start by deciding between
(a) the dual-function chunk 4 split to recover greedy on logits-emit
builds, or
(b) the wider drafter rebuild. Without one of these, the sampling
branch produces text but doesn't beat greedy.

## Original groundwork (kept for reference)

- Drafter side already exposes top-K (id, logit) pairs via the new
  `MtpDraftSource.draftOneFull(...)` (groundwork for sampling).
- Engine `samplingTemperature` property is now wired end-to-end
  (`MTP_TEMPERATURE` env → `MtpSpeculativeEngine.samplingTemperature`
  → sampling fork in `speculateStep`).
- LiteRT-LM bundle's `mtp_drafter` is structurally different from the
  HF `gemma-4-E2B-it-assistant` we use:
  - `activations` input is **3072-dim** (HF: 1536)
  - KV cache stored as **int8** (HF: fp16)
  - LM head is **full-vocab argmax** (HF: 4096-candidate centroid path)
  - `ai-edge-litert.Interpreter` can load it directly on Mac — see
    `extract_mtp_drafter.py` + `/tmp/litertlm_extract/section_9.tflite`.
  - Porting to CoreML would require a fresh PyTorch→CoreML build with
    that 3072-dim signature. 1-week effort, payoff unproven.

## Files changed (all under `feat/mtp-iphone-perf` branch)

| Path | What |
|---|---|
| `Sources/CoreMLLLM/ChunkedEngine.swift` | `lastKV14V_T` storage + `mirrorKV14VTransposed` hooks in predictStep / runPrefill / both verify paths. `commit=` instrumentation feeds SpecProfile. iOS `bypassVerifyCommit` hard-off (env override removed for iOS to keep stale scheme env from poisoning the path). |
| `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` | Reads `engine.lastKV14V_T` directly when present. `transpose13` parallelised + tiled 64×64 (was naive O(A·B) loop). `samplingTemperature` property (plumbed, unused). `[MTP setup]` and `[MTP cycle]` instrumentation. mask-offset env behaviour reverted to default 1. Dirty-K diagnostic comments updated. |
| `Sources/CoreMLLLM/MtpDraftSource.swift` | `draftOneFull(...)` exposing top-K (ids, fp32 logits) for the upcoming rejection-sampling path. Old `draftOne` re-implemented as a thin wrapper. fp16→fp32 helper. |
| `Sources/CoreMLLLM/CoreMLLLM.swift` | `[MTP cycle] spec=… yield=… total=… emit=… (yield/tok=…)` log around the speculateStep call; isolated the 87 ms transpose-residual to inside speculateStep so it didn't get blamed on UI. |
| `Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift` | iOS now hard-routes to MTP regardless of leftover scheme env (`SPECULATIVE_PROFILE`, `MTP_FORCE`, etc. ignored). macOS keeps `SPECULATIVE_PROFILE=1` as legacy union diagnostic. |
| `conversion/bench_hf_mtp.py` | `--sample`, `--temperature`, more prompts (`household`, `json`, `translate`, `qa_long`, `code_class`, `count_to_50`, `markdown_table`). |

## How to reproduce the iPhone run

1. Rebuild verify chunks with `PALETTIZE_KEEP_FP_KV=1` (k_proj / v_proj
   stay fp16, all other weights INT4):
   ```bash
   PYENV_VERSION=lama-cml PALETTIZE_KEEP_FP_KV=1 \
     python conversion/build_verify_chunks.py \
       --model gemma4-e2b --K 3 --ctx 2048 \
       --output output/gemma4-e2b/verify_chunks_fp16kv
   for c in chunk1 chunk2 chunk3 chunk4; do
     xcrun coremlc compile output/gemma4-e2b/verify_chunks_fp16kv/$c.mlpackage \
       output/gemma4-e2b/verify_chunks_fp16kv
   done
   ```

2. Push the four `.mlmodelc` to iPhone (replaces the existing INT4 chunks
   under `Documents/Models/gemma4-e2b/`):
   ```bash
   bash scripts/push_gemma4_e2b_bundle.sh
   ```

3. Build & run `Examples/CoreMLLLMChat` on iPhone 17 Pro. Scheme env
   should be empty — iOS routing forces MTP regardless. Expected steady
   state on `Say yes 30 times`: `[MTP cycle] spec=43–45ms ... emit=3`,
   UI 60+ tok/s.

## HF transformers reference numbers (Mac MPS bf16)

384 new tokens, E2B + assistant (Google official HF release):

| Prompt | greedy | temp=0.7 sampling | temp=1.0 sampling |
|---|---|---|---|
| count_to_50 | 2.44× | 2.47× | 2.26× |
| repeat (yes×30) | 2.78× | n/a | n/a |
| translate | 0.74× | 1.46× | 1.68× |
| household | 1.13× | 1.08× | **1.52×** |
| code_class | 1.46× | 1.53× | 1.48× |
| json | 1.18× | 1.46× | 1.30× |
| markdown_table | 1.19× | n/a | 1.26× |
| qa_long | 1.07× | 0.80× | 0.86× |
| fib_recursion | 0.95× | n/a | n/a |

Take-away: **drafter-model accept rate is the ceiling** for free-form
text. Our iPhone INT4 stack matches HF MPS bf16 ratios within
quantization noise. The straightforward "implement rejection sampling"
path was tried this session and stalled — see "Rejection-sampling
attempt" above. To exceed the HF ceiling we still need either a wider
drafter (the centroid + top-8 LM head caps draft variety) or a
different drafter architecture (LiteRT extraction now possible via
`ai-edge-litert`, payoff unproven).
