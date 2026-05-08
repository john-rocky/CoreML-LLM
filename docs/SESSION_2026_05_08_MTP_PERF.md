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
  **1.52×** and `translate` to 1.68×. Iphone Swift path is currently
  argmax-only. Implementing rejection sampling is the next big win.
  - Drafter side already exposes top-K (id, logit) pairs via the new
    `MtpDraftSource.draftOneFull(...)` (groundwork for sampling).
  - Engine `samplingTemperature` property is plumbed but unused —
    legacy greedy path remains active when `samplingTemperature == 0`.
  - Verify chunks already emit `logits_fp16` (top-32 via
    `verifyCandidatesWithLogits`) — full-vocab logits available.
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
quantization noise. To exceed the HF ceiling we have to either
implement rejection sampling (planned) or swap drafters (LiteRT
extraction now possible via `ai-edge-litert`, payoff unproven).
