# Implementation Log — Pre-Conversion Optimizations Branch

**Branch:** `feat/pre-conv-optimizations`
**Base:** `device-bench`
**Session dates:** 2026-04-15 / 2026-04-16
**Goal:** Apply all Python-side / MIL-pass optimizations that do not require
retraining or device test iteration, so the conversion produces faster
mlpackages without changing quality.

Target: baseline 15 tok/s @ 8K on iPhone 17 Pro → 22–28 tok/s projected from
combined audit gains, assuming parity holds.

---

## Delivered this session

### Applied (code-complete, parity-test pending on device)

| # | Change | Files | Expected gain |
|---|---|---|---|
| A1 | Bumped `minimum_deployment_target` from iOS18 → iOS26 across 12 build scripts | conversion/build_*.py | Unlocks iOS 26 MIL ops; no direct tok/s but future-proofs pipeline |
| A2 | Expanded `optimize_mlpackage_graph.py` DEFAULT_PASSES from 9 to 14 passes, added ordering fix, size-guard around unsafe passes, opt-in separation | conversion/optimize_mlpackage_graph.py | +5–10 % (audit estimate) |
| A3 | Exposed programmatic `optimize_mlpackage()` API (callable from other build scripts) | conversion/optimize_mlpackage_graph.py | enabling change |
| A4 | Added `--optimize` CLI flag to `build_merged_chunks.py` and `build_verify_chunks.py`; threaded through all per-chunk conversion call sites | conversion/build_merged_chunks.py, conversion/build_verify_chunks.py | activates A2 |
| A5 | `ANERMSNorm(affine=False)` mode: skips the scale multiply so it can be absorbed into adjacent Conv weights | conversion/ane_ops.py | +3–5 % when paired with A7 |
| A6 | `absorb_rmsnorm_scale_into_conv(norm, conv)` utility: idempotent fold of RMSNorm scale into a 1×1 Conv2d weight | conversion/ane_ops.py | (tooling for A7) |
| A7 | `ane_fused_softmax()`: MIL-`softmax`-friendly variant as drop-in next to the decomposed `ane_softmax`. Both live side by side; caller opts in per call site. | conversion/ane_ops.py | fewer ops (1 vs ~8–12 per softmax) |
| A8 | `FusedQKV` and `FusedGateUp` drop-in modules with `.from_split()` weight-concat constructors and a `fuse_layer_projections()` helper | conversion/models/gemma4_fused_modules.py (new file) | +8–12 % (QKV) + +5–8 % (Gate/Up) once wired |

### Documentation

| # | Doc | Purpose |
|---|---|---|
| D1 | docs/CONVERSION_AUDIT_2026_04_15.md | 19-item Python-side audit: 10 done / 3 missing / unused tooling |
| D2 | docs/GEMMA4_ANE_REWRITES.md | 8 concrete rewrites with pasteable PyTorch code |
| D3 | docs/ANE_CONVERSION_RECIPE_2026.md | Apple `ml-ane-transformers` + coremltools 9.0 canonical recipe |
| D4 | docs/MIL_PASSES_ADDITIONAL.md | All 73 MIL passes reviewed; dangerous passes called out |
| D5 | docs/ANE_NEW_TECHNIQUES_2026.md | 2025-2026 new techniques not in prior docs |
| D6 | docs/MLPACKAGE_STRUCTURE_AUDIT.md | Compiled-graph analysis: softmax decomposition and V-layout issues found |
| D7 | docs/GPU_WHY_FAST.md | Structural reasons Metal beats ANE; ANE-only ceiling ~22–28 tok/s |
| D8 | docs/PARITY_TEST_PROTOCOL.md | Acceptance bars + per-optimization verification procedure |

---

## Not applied — deferred with rationale

### D-1. Wiring Fused QKV / Fused Gate/Up into chunk forward code

The fused modules in `gemma4_fused_modules.py` are drop-in ready but every
chunk builder (`SWAChunk1..4`, `MergedChunk12`, `MergedChunk34`, `MergedChunk1`,
prefill chunks) directly references `layer.self_attn["q_proj"]`,
`layer.self_attn["k_proj"]`, `layer.self_attn["v_proj"]`, `layer.mlp["gate_proj"]`,
`layer.mlp["up_proj"]` in their `_run_layer_*` functions. Switching requires:

1. Call `fuse_layer_projections(layer)` in model-setup before conversion.
2. Replace the three `q_proj/k_proj/v_proj` calls with one `qkv_fused(x)` +
   slice in each chunk's per-layer forward (SWA, verify, prefill, merged, and
   merged-verify variants).
3. Replace `gate_proj(x)` + `up_proj(x)` with one `gate_up_fused(x)` + split.
4. Update `Gemma4Model.load_weights` to ignore split weights when the fused
   module is present (or pack them at load time).
5. Run parity on a single layer first, then scale out.

This is mechanical but ~10–15 edits across 5 chunk files and requires per-
layer parity verification. Deferred to a follow-up commit once the simpler
wins (A2/A7) are validated on device.

### D-2. RMSNorm scale absorption full wiring

Helper A6 is ready. Driver code in `Gemma4Model` that calls
`absorb_rmsnorm_scale_into_conv(input_layernorm, qkv_fused.fused)` and
`absorb_rmsnorm_scale_into_conv(pre_feedforward_layernorm, gate_up_fused.fused)`
at the right point in `load_weights()` is intentionally not added because:

- It depends on D-1 (needs fused Conv2d as the absorbing target)
- Absorbing into one of three separate `q/k/v` projections introduces three
  differently-scaled weight tensors for what was previously one scale, which
  changes the INT4 palette sample distribution and may shift perplexity.
  Fused QKV collapses the three into one weight tensor where absorption is
  unambiguous.

Apply D-2 AFTER D-1 and measure palettization quality separately.

### D-3. Drafter V-layout mismatch fix

Per `docs/MLPACKAGE_STRUCTURE_AUDIT.md` the chunks emit `kv14_v` as
`(1, 1, CTX, 512)` but the drafter expects `(1, 1, 512, CTX)`. Runtime Swift
applies an implicit transpose each decode step (~0.5 ms).

Cleanest fix is to add a transpose inside `extract_mtp_drafter.py` to ingest
the producer's layout, rather than emitting a different layout from chunk4
(which would cascade through chunks 3/4 forward code and their verify
variants). Deferred; requires reading the drafter module's math to confirm
the transpose is correct.

### D-4. NCHW end-to-end residual stream

Also per `docs/MLPACKAGE_STRUCTURE_AUDIT.md`: each chunk has ~55 layout ops
vs ~90 compute ops (ratio ~0.6). Chunk1 has 162 transposes + 98 reshapes vs
73 conv + 18 matmul. Keeping hidden states in `(B, C, 1, S)` throughout the
stack instead of round-tripping to `(B, S, C)` between layers would drop
~100 layout ops per chunk. Requires end-to-end rewrite of the layer forward
code (touches every chunk module). High-impact but out of scope for this
branch — tagged as its own follow-up.

### D-5. Stateful KV retry on iOS 26

`conversion/build_stateful.py` already exists (in a worktree per audit
findings) with correct `StateType` usage. Previously failed with ANE error
-14 on iOS 18. Apple ships iOS 26 with documented stateful-model fixes worth
retesting. Deferred — requires a device run to confirm; not a Python-side
change.

### D-6. Multi-function mlpackage via `materialize_symbolic_shape_program`

One mlpackage with prefill-512, decode-1, verify-K functions sharing a
single weight blob. Cuts ship size ~50 % and eliminates model-swap latency.
Already known in MIL_PASSES_ADDITIONAL.md. Requires a new top-level build
script; deferred to its own branch to isolate the ship-path change.

---

## Rollout order (suggested)

Run in this order. Each step has its own parity check (see
`PARITY_TEST_PROTOCOL.md`):

1. **A1 iOS26 bump** — baseline for everything else; zero numerical change expected.
2. **A2+A4 optimize_mlpackage DEFAULT_PASSES via `--optimize` flag** — +5–10 %, op count visible in converter logs.
3. **A7 ane_fused_softmax** in non-attention call sites first (softcapping, LM head), then attention. Parity-check each site.
4. **D-1 wiring of A8 FusedQKV/FusedGateUp** — +13–20 % when it lands.
5. **D-2 RMSNorm absorption** on top of D-1 — +3–5 %.
6. **D-3 drafter V-layout** — +0.5 ms/step on decode, only matters when MTP drafter is active.
7. **D-4, D-5, D-6** — separate branches.

Expected cumulative after steps 1–5: **22–26 tok/s** (within audit's projected window).

---

## Files touched

```
conversion/ane_ops.py                         # A5, A6, A7
conversion/optimize_mlpackage_graph.py        # A2, A3
conversion/build_merged_chunks.py             # A1, A4
conversion/build_verify_chunks.py             # A1, A4
conversion/build_eagle3.py                    # A1
conversion/build_eagle3_chunks.py             # A1
conversion/build_eagle3_gpu.py                # A1
conversion/build_flash.py                     # A1
conversion/build_mtp_drafter.py               # A1
conversion/build_speculative.py               # A1
conversion/build_w8a8.py                      # A1
conversion/build_w8a8_proper.py               # A1
conversion/build_wfa.py                       # A1
conversion/models/gemma4_fused_modules.py     # A8 (new)
docs/IMPLEMENTATION_LOG_2026_04_15.md         # this doc
docs/PARITY_TEST_PROTOCOL.md                  # verification procedure
docs/ANE_CONVERSION_RECIPE_2026.md            # research (prev commit)
docs/ANE_NEW_TECHNIQUES_2026.md               # research (prev commit)
docs/CONVERSION_AUDIT_2026_04_15.md           # research (prev commit)
docs/GEMMA4_ANE_REWRITES.md                   # research (prev commit)
docs/GPU_WHY_FAST.md                          # research (prev commit)
docs/MIL_PASSES_ADDITIONAL.md                 # research (prev commit)
docs/MLPACKAGE_STRUCTURE_AUDIT.md             # research (prev commit)
```

No existing build script semantics change by default. All new behavior is
behind the `--optimize` flag or requires explicit wiring (A8 helpers).

---

## Next session

1. Run rollout step 1 (A1) on device, record baseline and optimized tok/s.
2. Wire A8 FusedQKV/FusedGateUp into a single chunk (start with SWAChunk1),
   run parity; only proceed to other chunks if cosine passes.
3. Measure incremental gains step-by-step; fill in the result table in
   `PARITY_TEST_PROTOCOL.md` section 6.
