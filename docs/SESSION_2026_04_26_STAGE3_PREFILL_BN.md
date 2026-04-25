# Session 2026-04-26 — Stage 3: multifunction `prefill_bN`

**Branch:** `stage3-prefill-bn`
**Roadmap entry:** `docs/ROADMAP_2026_04_26.md` §2 Stage 3 (Phase 2b).
**Goal:** drop first-turn TTFT 4–5× by batching `T = 8` prefill tokens
through a single multifunction CoreML forward, with T=1 fallback for
the tail and any sliding-cache wrap.

This session lands the converter + runtime work. iPhone validation is
deferred to a follow-up session (per Mac-first policy).

---

## Outcome

### Mac-side (Mac Studio)

- **Multifunction PoC (chunk_1)** — `infer` (T=1) and `prefill_b8`
  (T=8) coexist in one mlpackage. Both ~93 % ANE individually. Merged
  package 149 MB (vs 148.6 MB single-function — coremltools dedup).
- **Full 4-chunk multifunction bundle**:
  | chunk | infer ANE | prefill_b8 ANE | merged size |
  |---|---|---|---|
  | chunk_1 | 92.9 % | 92.7 % | 156 MB |
  | chunk_2 | 92.9 % | 92.6 % | 135 MB |
  | chunk_3 | 90.7 % | 90.5 % | 326 MB |
  | chunk_4 | 89.7 % | 89.5 % | 528 MB |

  T=8 ANE residency lost <0.5 % vs T=1 — the ANE schedules batched
  tensors essentially identically.
- **Latency probe** (chunk_1, Mac Studio, 30 reps):
  ```
  T=8 prefill_b8:    4.29 ms/call (= 0.54 ms/tok)
  T=1 infer ×8:     32.39 ms total
  speedup: 7.55×
  ```
  Same for chunk_4: T=8 = 9.36 ms, T=1×8 = 73.23 ms → 7.83×.
- **Chained sanity** — `sanity_prefill_bn_chain.py` runs T=8 through
  all four chunks sharing the same MLState, then runs a T=1 follow-up
  at `current_pos = T`. All outputs finite, kv13/kv14 alias shapes
  match the expected (1, HKV, W or ctx, hd_*).

### Engine

`Gemma4StatefulEngine.swift` now probes for `prefill_b{16, 8}` on every
chunk at `load()`. When all four chunks carry the same N, the prefill
loop dispatches T-token windows via a new `prefillStep`; tail (or
sliding-cache wrap, when `position % W + T > W`) falls back to the T=1
`step`. T=1 decode loop is unchanged.

The dispatch-trace prefix gets a tag:

```
[Gemma4Stateful] prefill 50 tok in 178ms (281.0 tok/s) [batched=6x8 t1=2] | decode …
```

(Numbers above are projection — actual iPhone numbers TBD.)

### Files touched

- `conversion/models/gemma4_swa_stateful_chunks.py` — `_run_layer_swa_stateful_prefill`
  + `SWAStatefulChunk{1,2,3,4}Prefill` (T-aware reshape + slice_update).
- `conversion/build_gemma4_e2b_stateful_chunks.py` — `--prefill-batches`
  flag, per-chunk T=N converters, `merge_multifunction` via
  `MultiFunctionDescriptor`.
- `conversion/sanity_prefill_bn.py` (new) — single-chunk sanity.
- `conversion/sanity_prefill_bn_chain.py` (new) — 4-chunk chained sanity.
- `conversion/bench_prefill_bn.py` (new) — T=N vs T=1×N latency probe.
- `Sources/CoreMLLLM/Gemma4StatefulEngine.swift` — `prefillStep`,
  multifunction probe in `load()`, batched-prefill dispatch in
  `generate()`.
- `scripts/assemble_gemma4_stateful_bundle.sh` — `SRC_CHUNKS` /
  `OUT_PARENT` env overrides for bundling alternative builds.

---

## Build commands

Mac build (E2B, Linear, multifunction T=8):

```
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/g4_prefill/multi \
    --hf-dir /path/to/gemma4-e2b/hf_model \
    --linear-projections \
    --prefill-batches "8"
```

Bundle assemble for iPhone:

```
SRC_CHUNKS=/tmp/g4_prefill/multi \
OUT_PARENT=$PWD/build/gemma4_stateful_prefill_bn \
bash scripts/assemble_gemma4_stateful_bundle.sh
```

Sanity (Mac):

```
python3.12 conversion/sanity_prefill_bn_chain.py \
    --bundle /tmp/g4_prefill/multi --T 8

python3.12 conversion/bench_prefill_bn.py \
    --pkg /tmp/g4_prefill/multi/chunk_1.mlpackage --T 8 --reps 30
```

iPhone push (deferred to a follow-up session):

```
DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
xcrun devicectl device copy to --device $DEVICE \
    --domain-type appDataContainer \
    --domain-identifier com.example.CoreMLLLMChat \
    --source build/gemma4_stateful_prefill_bn \
    --destination Documents/Models/gemma4-e2b-stateful
```

Expected iPhone first-turn TTFT (50-tok prompt): 1.2 s → 0.16–0.20 s.

---

## Open follow-ups

- **iPhone validation** — push the bundle, run a 50-tok prompt under
  Profile mode, confirm `[batched=6x8 t1=2]` output and the TTFT delta
  matches the Mac latency ratio.
- **Wrap handling** — T=1 fallback past `position = W` is correct but
  loses ~7× per token in the wrap window. For a 2048-ctx build with
  W=512 this hits the last 1.5 K tokens of long prompts. A split-write
  prefill (write up to W, then wrap to slot 0 for the remaining T-x)
  is doable but not pursued in this session.
- **Combination with cross-turn KV reuse (Phase 2a)** — the dispatch
  loop already starts at `resumeAt`, so a multi-turn long-prompt user
  gets BOTH wins automatically.
- **T=16 build** — easy follow-up: re-run with `--prefill-batches "8,16"`
  to add a wider batch. Engine load-probe already prefers 16 over 8.
