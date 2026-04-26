# Session 2026-04-26 — Stage 3: multifunction `prefill_bN`

**Branch:** `stage3-prefill-bn`
**Roadmap entry:** `docs/ROADMAP_2026_04_26.md` §2 Stage 3 (Phase 2b).
**Goal:** drop first-turn TTFT 4–5× by batching `T = 8` prefill tokens
through a single multifunction CoreML forward, with T=1 fallback for
the tail and any sliding-cache wrap.

Outcome: 4-chunk + 3-chunk merged Mac bench, single-buffer + 1-chunk
all-in-one probes, full iPhone test matrix. Final ship variant is
**3-chunk merged dual-state Linear**.

---

## Headline numbers

**Mac (Mac Studio, ctx=2048, --linear-projections, int4):**

| variant | decode tok/s | T=8 prefill speedup | bundle (chunks) |
|---|---|---|---|
| baseline (legacy 4-chunk recurrent) | 25.8 | — | 2.18 GB |
| 4-chunk dual-state stateful | 32.6 | 7.74× | 1.14 GB |
| **3-chunk merged dual-state** ← ship | **34.6** | **7.77×** | **1.15 GB** |
| 3-chunk merged single-buffer | 34.3 | 7.77× | 1.15 GB |
| 1-chunk all-in-one | 30.8 | 7.14× | 1.09 GB |

**iPhone 17 Pro (50-tok prompt, ctx=2048):**

| variant | T>1 multifunction | decode tok/s |
|---|---|---|
| 4-chunk dual T=4/T=8 | NG (ANECCompile FAILED 11) | (T=1 fallback) ~32 |
| 4-chunk Conv2d T=8 | NG (same) | — |
| **3-chunk dual T=8** | **NG** (graceful T=1 fallback) | **33.4** |
| 3-chunk single-buffer T=8 | NG (same) | 33.2 |
| 1-chunk all-in-one T=8 | NG (ANE helper crash) | n/a |

iPhone result: T>1 multifunction is structurally rejected by iPhone
ANE 18 for our Gemma 4 stateful path, regardless of chunk count,
state buffer count, or projection type. Engine falls back to T=1
gracefully — no regression. Mac gets the full 7.77× prefill win.

---

## Ship decision

**3-chunk merged dual-state Linear** is the production default:

- **iPhone**: decode +5% over 4-chunk Phase 2a (33.4 vs 32 tok/s).
  Cross-turn KV reuse (Phase 2a) intact. T=1 prefill fallback.
- **Mac**: decode +6% over 4-chunk (34.6 vs 32.6) + prefill 7.77×
  via multifunction (T=8 batched).
- **Bundle (chunks portion)**: 1.15 GB, same as 4-chunk.
- **Engine auto-detection**: chunk_3 having `token_id` output identifies
  the merged-final layout; chunk_4 absence triggers 3-chunk mode.

Single-buffer / 1-chunk classes are preserved as future probes (in
case iOS 19+ ANE removes the multifunction T>1 limit) but not
shipped.

### Why ISWA + multifunction T>1 fails on iPhone

Gemma 4 has 35 layers split as 4 sliding + 1 full repeating (28
sliding + 7 full). Our stateful build needs two MLState buffers
(kv_cache_sliding W=512, kv_cache_full ctx=2048). Probes confirmed
the iPhone ANE 18 compiler rejects the dual-state slice_update + T>1
multifunction combination at any T (we tested T=4 and T=8) and at any
state buffer count (single unified buffer also fails). 1-chunk
all-in-one (35 layers + lm_head in one mlpackage with one MLState)
makes the iPhone ANE compiler daemon crash entirely. Qwen3-VL ships
T=8 multifunction on iPhone successfully — it has 28 layers and a
single state, so the limit lives somewhere in the (size × dual-state
× multifunction) interaction we can't bypass without a major
architectural rewrite (vocab pruning or QAT for INT4 PLE — see
`docs/PLE_INT4_PROBE.md` for the rejected drop-in path).

---

## Files

### Production (ship in this branch)

- `conversion/models/gemma4_swa_stateful_chunks.py`
    - `_run_layer_swa_stateful_prefill` (T=N variant)
    - `SWAStatefulChunk{1..4}Prefill` — original 4-chunk T=N
    - `SWAStatefulMergedChunk23{,Prefill}` — merged middle (L8-24)
- `conversion/build_gemma4_e2b_stateful_chunks.py`
    - `--prefill-batches` flag, multifunction merge via
      `coremltools.utils.MultiFunctionDescriptor`
- `conversion/build_gemma4_e2b_stateful_3chunks.py` (new)
    - 3-chunk emit (chunk_1 + merged middle + final).
- `Sources/CoreMLLLM/Gemma4StatefulEngine.swift`
    - 3-chunk auto-detection (`chunk_4` absent OR `chunk_3` has
      `token_id` output).
    - `prefillStep()` for batched T=N dispatch.
- `Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift`
    - Bundle detection accepts chunk_1..3 (chunk_4 optional).
- `scripts/assemble_gemma4_stateful_bundle.sh`
    - Auto-detect 3 vs 4 chunk via `CHUNKS` env override.

### Probes (preserved in branch, NOT shipped)

- `_run_layer_swa_stateful_*_single` + `SWAStatefulChunk1Single` /
  `SWAStatefulMergedChunk23Single` / `SWAStatefulModel1Chunk` —
  unified-MLState and 1-chunk all-in-one variants.
- `conversion/build_gemma4_e2b_stateful_1chunk.py` — 1-chunk emit.
- `--single-buffer` flag in 3-chunk converter.

### Sanity / bench tooling

- `conversion/sanity_prefill_bn{,_chain}.py`
- `conversion/bench_prefill_bn.py`
- `conversion/benchmark_stateful_chunks.py`
- `conversion/benchmark_stateful_3chunks.py`
- `conversion/benchmark_stateful_1chunk.py`

---

## Bug fix bundled with this branch

Default `gemma4-e2b` HF download was pulling the legacy 3-chunk
recurrent files (`chunk2_3way` + `chunk3_3way`) regardless of whether
the user opted into them via `LLM_3CHUNK=1`. The MLState 3-chunk we're
shipping replaces that experiment. Removed those entries from
`ModelDownloader.swift` chunkFiles list — **default install -987 MB**
(6 GB → ~5 GB).

---

## Build commands

3-chunk Linear multifunction bundle (production):
```
python3.12 conversion/build_gemma4_e2b_stateful_3chunks.py \
    --output /tmp/g4_3chunk \
    --hf-dir /path/to/gemma4-e2b/hf_model \
    --ctx 2048 --linear-projections --prefill-batches "8"
```

Assemble for sideload / iPhone push:
```
SRC_CHUNKS=/tmp/g4_3chunk \
OUT_PARENT=$PWD/build/gemma4_stateful_3chunk_linear \
bash scripts/assemble_gemma4_stateful_bundle.sh
```

---

## Open follow-ups (separate stages)

1. **Tied weight dedup (-192 MB)** — drop lm_head from chunk_3
   mlpackage, compute logits in Swift via embed_tokens sidecar.
   Trade-off: +7-8 ms per decode step (bandwidth-bound GEMV) =
   ~-20% throughput. Needs design pass and perf measurement.
   Out of scope for Stage 3.
2. **HF upload + URL swap** (Phase B/C of roadmap) — separate stage.
3. **iOS 19 / future ANE update revisit** — re-probe T>1 multifunction
   when Apple ships a new ANE compiler. Single-buffer + 1-chunk code
   already in branch as a starting point.
