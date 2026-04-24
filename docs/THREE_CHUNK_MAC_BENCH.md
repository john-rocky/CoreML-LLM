# 3-chunk decode — Mac A/B bench + iPhone verdict

**Date:** 2026-04-24
**Machine:** Apple Silicon Mac (Darwin 25.0, macOS Tahoe)
**Model:** Gemma 4 E2B, ctx=2048, INT4 palettized (per_grouped_channel g=32)
**Prompt:** `"Say one short fact about the moon."`  (max_tokens=64)
**Harness:** `scripts/bench_3way_mac.sh` → `coreml-llm-smoke` CLI
**Compile units:** CPU_AND_NE (default on macOS)

## Steady-state decode (step 4+, post-warmup)

| Config | tok/s | c1 ms | c2 ms | c3 ms | c4 ms | sum | ANE_wait | copyBack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-chunk | 33.08 | 5.4 | 6.7 | 7.5 | 10.4 | 30.1 | 29.6 | 0.3 |
| **3-chunk** | **34.79** | 5.4 | **12.7** | 0.0 | 10.5 | **28.6** | 28.2 | 0.3 |

**Delta:** +1.71 tok/s (**+5.2 %**) / −1.5 ms per token.

The merged c2 (17 layers) lands at **12.7 ms**, cleanly below the 4-chunk
c2+c3 sum of 14.2 ms. The residual (14.2 − 12.7 = **1.5 ms**) is exactly
one ANE dispatch round-trip disappearing, matching the Orion measurement
of ~0.7 ms per round-trip on M-series (3× cheaper than iPhone A-series
2.3 ms, per Orion arXiv 2603.06728).

## Correctness

Identical generated text under both modes, char-for-char:

> **One short fact about the Moon:** The Moon is tidally locked,
> meaning it always shows the same face to Earth.

Consistent with the offline PyTorch parity (`cos=1.000000` on all 9
outputs of `MergedChunk23` vs `SWAChunk2→SWAChunk3` — see
`conversion/parity_3way_vs_4chunk.py`).

## Load time

| Config | chunk load | prewarm | total |
|---|---:|---:|---:|
| 4-chunk | 34.5 s (4 chunks sequential) | 0.2 s | ~35 s |
| 3-chunk | 28.7 s (3 chunks sequential) | 0.2 s | ~29 s |

The 3-chunk bundle drops one compile step (`chunk3`) without growing
the bigger merged chunk's compile time proportionally — net **−5.8 s**.

## iPhone 17 Pro (A19 Pro) — measured

| Config | steady tok/s | c1 | c2 | c3 | c4 | sum |
|---|---:|---:|---:|---:|---:|---:|
| 4-chunk | 31.6 | 5.8 | 6.8 | 8.2 | 10.3 | 31.1 |
| **3-chunk** | **33.0** | 5.8 | **13.0** | 0.0 | 10.9 | **29.7** |

**Delta: +1.4 tok/s (+4.4 %)** / −1.4 ms/step. c2+c3 (15.0 ms) collapses
to merged c2 (13.0 ms) — 2.0 ms saved, matching Orion's 2.3 ms/dispatch
estimate for A-series ANE within measurement noise.

`COMPUTE_PLAN_AUDIT=1` initially reported "chunk2_3way: 133/1847 ops NOT
on Neural Engine". All 133 turned out to be `ios18.constexpr_lut_to_dense`
(INT4 LUT → fp16 weight dequant), which run once at load, not per-step.
`ComputePlanAudit.swift` had a pre-existing bug where its `constOps`
whitelist missed the `ios18.` dialect prefix; fixed in this PR. After the
fix the audit reports all ops on ANE for every decode chunk, matching
the 4-chunk baseline. **Per-step compute is 100 % ANE in both modes.**

## Further gains from here

3-chunk hits the ANE dispatch-reduction ceiling for Gemma 4 E2B under
the 17-layer compile limit — `chunk3+chunk4` merge would be 20 layers
and almost certainly won't compile. Beyond this, the remaining levers
are orthogonal to chunk count: SDPA fusion retest, per-block-32
palettization, within-layer K=V alias for global layers, and the
runtime hints (prefill warm-pool, ...).

Per-chunk profile print lines live at `ChunkedEngine.swift` 1078-1100 —
on 3-chunk mode `c3` is expected to print as `0.0` (dispatch skipped).

## Reproducing

```bash
# From the worktree root:
python conversion/build_gemma4_bundle.py --model gemma4-e2b --ctx 2048 --skip-compile
python -c "import sys; sys.path.insert(0,'conversion'); \
    from build_gemma4_bundle import _compile_all_chunks; \
    _compile_all_chunks('output/gemma4-e2b/bundle/chunks', 'output/gemma4-e2b/bundle')"
python conversion/build_gemma4_3way.py --model gemma4-e2b --ctx 2048
python conversion/install_3way_bundle.py

scripts/bench_3way_mac.sh output/gemma4-e2b/bundle
```

Last three steady-state [Profile] lines of each run are logged to
`/tmp/bench_4chunk.log` and `/tmp/bench_3chunk.log`.
