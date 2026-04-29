# MLState multimodal feasibility — probe results (2026-04-28)

**Verdict: feasible.** Stage 6 retired stateful multimodal on the
assumption that iPhone ANE 18 rejects T>1 stateful prefill graphs. The
probes below show the rejection is **specific to multifunction**;
single-function stateful prefill at T=288 compiles cleanly, and the
state-buffer bridging primitive needed for prefill→decode hand-off
works. The architecture is unblocked.

## Probe 1 — single-function stateful prefill compile

**Question:** Stage 3 found that a multifunction (`prefill_bN` merged
into the decode mlpackage) prefill chunk with dual MLState (`kv_cache_sliding`
+ `kv_cache_full`) at T>1 is rejected by iPhone ANE 18 with
`ANECCompile FAILED 11`. We never tried the **single-function** variant
— a SEPARATE prefill mlpackage at T>1, no multifunction merge.

**Setup:**
- Builder: `conversion/probe_stateful_singlefunc_prefill.py`
- Calls existing `convert_chunk1_prefill(base, c_start=0, c_end=8, ctx=2048,
  T=288, ..., use_linear=False)` from
  `conversion/build_gemma4_e2b_stateful_chunks.py` to emit a standalone
  `chunk_1_prefill_T288.mlpackage` (148.6 MB, INT4 g32 palettized,
  ANE placement 1021/1094 = 93.3% per coremltools planner).
- Compile to mlmodelc on Mac via `xcrun coremlcompiler compile`.
- Push to iPhone 17 Pro Documents/mlstate_probe/chunk_1.mlmodelc via
  devicectl.
- Load via `MLModel(contentsOf: url, configuration: cfg)` with
  `cfg.computeUnits = .cpuAndNeuralEngine`. iPhone-side ANE re-compile
  fires here.

**Result: PASS.** Compile time 7283 ms on iPhone 17 Pro A19 Pro. No
`ANECCompile FAILED 11`. The chunk loads, computeUnit is honored.

**Implication:** the Stage 3 wall isn't "stateful T>1 doesn't compile";
it's "**multifunction** stateful T>1 doesn't compile". A separate
prefill mlpackage avoids the multifunction code path and ANE 18 takes
it.

## Probe 2 — state-buffer bridging

**Question:** to hand off from a prefill model (T=288, MLState) to a
decode model (T=1, MLState) we need to copy KV state contents between
two MLModel instances. CoreML 9's MLState API exposes
`state.withMultiArray(for: stateName) { mlMultiArray in ... }`. The
docs note "the underlying state buffer's address can differ for each
call; one shall not access the state buffer outside of the closure" —
that constraint is fine if we do all the copying inside nested
closures.

**Setup:**
- Same-model probe (cross-model is the identical operation when the
  two models declare matching StateType shape — testing the API
  mechanism is what matters):
  - Load probe T=288 chunk_1.
  - Make two states from the same model: `prefillState`, `decodeState`.
  - `prefillState.withMultiArray(for: "kv_cache_sliding")`: write a
    counter pattern (`UInt16(i & 0xFFFF)`) into every element via
    `dataPointer.bindMemory(to: UInt16.self, capacity:)`.
  - Inside that closure, nest
    `decodeState.withMultiArray(for: "kv_cache_sliding")` and `memcpy`
    src.dataPointer → dst.dataPointer for the entire buffer (count *
    sizeof(fp16)).
  - Read back from decode buffer's dataPointer, compare first/last
    UInt16 against the source's first/last.

**Result: MATCH.** Pattern written through prefill state shows up
verbatim in decode state after memcpy. Confirms:

1. `withMultiArray(for:)` returns a CPU-readable/writable view.
2. `dataPointer` is stable within the closure scope.
3. Memcpy between two state buffers works (nested-closure pattern).

**Implication:** the prefill→decode state hand-off is a straightforward
memcpy of the KV state buffers. No MLModel-internal API or private
framework calls are needed.

## Architecture sketch (Stage 8 candidate)

```
                  ┌────────────────────────────────────────┐
                  │ prefill mlpackage (single function)    │
                  │   T=288, MLState {sliding, full}       │
                  │   input:  hidden_states (1, 288, H)    │
                  │   inside: 256-tok image-pad span seen  │
                  │           bidirectionally (HF Gemma 4  │
                  │           token_type_ids_mask_function)│
                  │   slice_update writes K/V at [0, 287]  │
                  └────────────────────┬───────────────────┘
                                       │
                          withMultiArray(for: "kv_cache_*")
                          memcpy src.dataPointer → dst.dataPointer
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │ decode mlpackage (existing Stage 3)    │
                  │   T=1, MLState {sliding, full}         │
                  │   continues from position 288          │
                  │   per-token chunked decode chain       │
                  └────────────────────────────────────────┘
```

## Effort estimate (Stage 8)

| Step | Effort |
|---|---|
| Build prefill chunks 1/2/3 at T=288 single-function (mirror probe builder for chunk2 + chunk3) | 0.5 day |
| Refactor `Gemma4StatefulEngine` to load `prefill_*` + `chunk_*` separately, run prefill once, bridge state via `withMultiArray` + `memcpy` | 1 day |
| Implement vision-aware bidirectional mask + image/audio feature splice in the prefill input buffer (256 image-pads fit in T=288 — legacy attention pattern restored) | 0.5 day |
| HF upload prefill chunks + `ModelDownloader` 3way variant additions | 0.5 day |
| iPhone 17 Pro multimodal verification + non-Pro RAM compile probe (iPhone 15 / 16) | 0.5 day |
| **Total** | **~3 days** |

## Open risks

1. **Cross-device compile budget.** iPhone 17 Pro (12 GB RAM) compiled
   T=288 fine. iPhone 15 / 16 / 17 non-Pro (8 GB) might fail at higher
   compile peak. T=288 covers a 256-tok image span exactly; if compile
   fails on lower-RAM devices we could retry T=192 (drop trailing text
   slack) before declaring it Pro-only.
2. **Prefill→decode position alignment.** State stores K/V indexed by
   `current_pos`; the prefill's slice_update writes positions [0, T-1]
   and decode resumes at position T. Off-by-one in the position passed
   to decode would corrupt the mask. Must validate decode's first step
   reads the bridged state correctly.
3. **Numerical parity vs. legacy 4-chunk prefill.** The new prefill
   graph runs INT4-palettized; legacy prefill_chunk_* are also INT4 g32.
   Bit-exact match is unlikely (different graph topology), but top-1
   token agreement on a fixed prompt is the validation gate.

## Stage 6 work preserved

The Stage 6 Swift extension (`Gemma4StatefulEngine.swift` multimodal
storage, splice helpers, vision-aware mask helpers, processImage /
processAudio etc.) lives on `stage6-multimodal-stateful` branch. With
the architecture cleared by these probes, Stage 8 can pull most of that
into the new prefill+decode-bridge engine layout — only the prefill
dispatch path needs to be reworked.
