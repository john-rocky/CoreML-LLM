# SmolLM 135M drafter — Phase 2 progress (2026-05-14)

Phase 2 of the sparse-activation roadmap: ANE-fit small drafter to
amortize the bandwidth wall on Gemma 4 E2B verify cycles. SmolLM 135M
is the candidate because (a) its INT4 weight footprint is small enough
to dramatically reduce per-token bytes read vs E2B, (b) it shares
~94% of SmolLM tokens with Gemma 4 by surface form so cross-vocab
speculative decoding is viable, (c) it's Llama-style which now has a
direct dispatch path in our conversion pipeline.

## Build status

| step | result | notes |
|---|---|---|
| HF download (`HuggingFaceTB/SmolLM2-135M-Instruct`) | ✅ 1.8 GB | tokenizer + safetensors |
| Cross-vocab map vs Gemma 4 | ✅ 94.3% | `output/smollm135_gemma_vocab.bin` 1.19 MB |
| Llama wrapper for conversion pipeline | ✅ committed | `conversion/models/llama.py` (`caf8bb4`) |
| `convert.py` Llama dispatch | ✅ committed | architecture="llama" → LlamaModel |
| `exporter.py` iOS-target fallback | ✅ committed | iOS26 → iOS18 → iOS17 graceful |
| Weight load (272 tensors, tied lm_head) | ✅ | from /tmp/smollm2-135m |
| Trace + CoreML convert (iOS18 target) | ✅ | monolithic mlprogram, MLState KV |
| INT4 palettize (group_size=32) | ✅ | 79.1 MB mlpackage |
| `coremlcompiler compile` → mlmodelc | ✅ | clean compile, no warnings |
| Mac inference smoke test | ⚠️ low quality | see below |

## Smoke test outcome

Prompt: `"The history of computing began with"`

Generated (greedy argmax, 32 tokens):

```
a team at MIT in the, which is based on assumptions made across.||
everal libraries across across across across across across across
```

Recognisable English for the first 12-15 tokens, then degrades into
repetition (`across across across`). This is consistent with:

* Small model (135M) absolute quality ceiling
* INT4 palettization at `group_size=32` is somewhat lossy on small
  models with tight weight distributions
* Greedy argmax with no temperature → degenerate loops

For **drafter use**, this is acceptable. The drafter does NOT need to
generate good text on its own; it only needs to **agree with the
target's argmax often enough** that the verify pass commits accepted
proposals faster than running target alone. Even a 20-30% agreement
rate amortises verify cost positively when the cycle math closes.

The smoke test script lives at `conversion/smoke_smollm.py` for future
re-runs.

## Output schema observation

The current `convert.py` wrapper builds a **monolithic decode head**:

```
inputs:  input_ids, position_ids, causal_mask, update_mask
outputs: token_id (Int32, argmax), token_logit (Float16, max logit)
```

This emits ONE token per call. The existing `CrossVocabDraft.swift`
also drives Qwen 0.5B drafter chain-style (one token per call, K-1
sequential calls per cycle), so the interface fits.

What this build does NOT expose:

* Top-K logits (for tree-style verify or FLy K=16 matching)
* Full vocab logits (for rejection sampling)

If we later want tree-verify or FLy on this drafter, the exporter
needs a top-K head. For chain-style cross-vocab speculative decoding
(the current production path on iPhone), the single-token output is
sufficient.

## Memory math (iPhone)

SmolLM 135M INT4 = 79 MB on disk.

* iPhone bandwidth ~60-77 GB/s
* per-token bytes read ≈ 79 MB
* theoretical ceiling ≈ 60 / 0.079 ≈ **760 tok/s** (drafter call alone)
* observed for monolithic small Llamas on ANE: 200-400 tok/s

vs Gemma 4 E2B decode at 30-40 tok/s. The drafter cycle would cost
~3-5 ms per chain step, total drafter chain (K-1=2 steps) ~6-10 ms.
Compare to current MTP drafter chain at 11-12 ms on iPhone.

Net win modest unless the SmolLM drafter shows higher acceptance rate
than the trained MTP drafter for our use case. **Empirical question
that needs iPhone bench.**

## Cross-vocab coverage

Built via `build_qwen_gemma_vocab_map.py`:

```
qwen vocab_size=49152  gemma vocab_size=262144
qwen->gemma 94.3%  gemma->qwen 17.7%  hits=46343
```

94.3% means most SmolLM-proposed tokens have a Gemma 4 equivalent and
can be verified directly. The 5.7% miss rate would force per-cycle
fallback to single-token T=1 decode for that step. Tolerable.

The Gemma → SmolLM coverage of 17.7% matters only when seeding the
drafter from a freshly accepted Gemma token. If we choose to **always
seed the drafter from its own previous output** rather than the
target's argmax, the 17.7% number doesn't affect throughput — only
quality of the drafter's first cycle proposal after each cycle.

## Mac end-to-end smoke (`SPECULATIVE_PROFILE=1 UNION_TRIP=1`)

Drafter loads and runs through `DrafterUnion`:

```
[CrossVocab] Drafter loaded (K=3, coverage q->g=94.3%)
[SpecProfile union #0001 src=cv] draft_total=34.57ms verify=32.14ms accepted=2/2 emitted=3 matches=11
[SpecProfile union #0002 src=cv] draft_total=23.38ms verify=33.74ms accepted=0/1 emitted=1 matches=0
[SpecProfile union #0003 src=cv] draft_total=34.70ms verify=33.68ms accepted=0/2 emitted=1 matches=01
... (similar for 18 cycles)
[SpecProfile union #0012 fallback] target_step=34.01ms
```

Result: **same failure mode as Qwen 0.5B cross-vocab drafter**. Cycle 1
got lucky on a common-token continuation, then every subsequent cycle
mispredicted at slot 0 (`matches=00`). DrafterUnion bailed to T=1
fallback at cycle 12. Net wall-clock is **negative** (verify cycle
~32 ms + cv drafter call ~34 ms = 66 ms for 1 token on most cycles vs
baseline 32 ms).

Two root causes, both expected:

1. **Drafter not on ANE.** Mac `computeUnits=CPU_AND_GPU` ran the
   SmolLM monolith on CPU+GPU; 34 ms/chain is consistent with that.
   On iPhone with `.cpuAndNeuralEngine` (or `.all`) the drafter
   *might* land on ANE (it's 79 MB INT4 + MLState — ANE compatibility
   uncertain because of writeState/readState which our docs say ANE
   rejects). Even at 5 ms/chain, with accept rate ≤ 5% the math
   doesn't close.

2. **Cross-model distribution mismatch.** 94.3% surface-form vocab
   overlap doesn't translate to 94% next-token agreement. SmolLM 135M
   was trained on a different corpus mix and weight scale than Gemma
   4 E2B; their next-token distributions diverge after the first few
   easy tokens. This is the same wall the Qwen 0.5B drafter hit per
   `docs/REJECTED_APPROACHES.md` / memory `project_drafter_structurally_dead`
   ("All accessible drafter routes are closed on Gemma 4 E2B").

Conclusion: **SmolLM 135M as a cross-vocab drafter for Gemma 4 E2B
does not deliver a net speedup**, regardless of which compute unit
runs it. The 94% vocab map is necessary but not sufficient — the
underlying LM has to agree with the target on next-token, which a
different model family doesn't.

## What would unblock this

The path that COULD work:
* **Distill SmolLM-class architecture from Gemma 4 E2B**. Train a
  130-200M model to match Gemma 4's next-token distribution on a
  large corpus. Vocab can stay 49k; cross-vocab map handles the
  translation. With matched distributions, accept rate could climb
  to 50-70%, and a 5 ms ANE drafter + 32 ms verify = ~37 ms/cycle
  with 2-3 tokens/cycle = 60-80 tok/s.
* This is **training, ~1 GPU-week** (same as the original drafter
  Path B retrain). Out of scope for "training-free" tonight.

The path that empirically doesn't:
* Off-the-shelf small LM (SmolLM, Qwen 0.5B, Phi-3-mini, etc.) used
  as cross-vocab drafter against Gemma 4 E2B target.

## Status against original roadmap

| phase | item | status |
|---|---|---|
| α-1 | Gemma 4 E2B activation sparsity calibration | ✅ done — dense, not useful |
| α-2 | Gemma 3n sparsity calibration | ⏳ download in progress (4.3/10 GB) |
| **2** | **SmolLM 135M → CoreML INT4** | **✅ done** |
| 2 | SmolLM 360M → CoreML INT4 (alternative) | ⏳ pending (deferred) |
| 2 | Cross-vocab map SmolLM ↔ Gemma 4 | ✅ done, 94.3% coverage |
| 2 | Wire SmolLM as cross-vocab drafter in CoreMLLLM | ❌ not started |
| 2 | iPhone bench | ❌ blocked on previous + iPhone reconnect |
| 3 | External-routed mini-MoE | ❌ not started |
| 4 | MLState MoE (deferred) | ❌ blocked by ANE compiler |

## Next concrete actions

1. **Wire SmolLM drafter into existing CrossVocab path**:
   - Build the `cross_vocab/` subdirectory in the iPhone bundle with
     SmolLM mlmodelc + the vocab map
   - `setup_cross_vocab_drafter.py` already handles this — just point
     it at SmolLM build dir
   - The existing Swift `CrossVocabDraft` / `CrossVocabSpeculativeEngine`
     classes work as-is for chain-style decoding (no top-K needed)

2. **iPhone bench**:
   - Once iPhone reconnects, push bundle with SmolLM drafter
   - Run AutoBench with explicit `crossVocabEnabled=true` (the engine
     hierarchy in CoreMLLLM.swift line 1240-1255 currently makes
     CrossVocab the LAST fallback, so MTP must be disabled to test it)
   - Expected: tok/s comparison vs MTP path. Win iff SmolLM accept
     rate × cheap drafter call > MTP drafter accept rate × MTP
     drafter call cost.

3. **Top-K exporter** (deferred): if drafter shows promise but FLy
   K=16 acceptance is needed for higher hit rates, modify
   `exporter.py` to emit top-K logits alongside the argmax output.

## Files

* `conversion/models/llama.py` — Llama family architecture wrapper
* `conversion/smoke_smollm.py` — standalone Mac inference test
* `/tmp/smollm2-135m/` — HF weights (not committed)
* `/tmp/smollm135_coreml/model.mlpackage` — CoreML build (not committed)
* `/tmp/smollm135_coreml/model.mlmodelc` — compiled artifact (not committed)
* `output/smollm135_gemma_vocab.bin` — cross-vocab map (not committed)
