# 3-chunk decode — Mac A/B bench

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

## iPhone extrapolation

Mac per-dispatch cost ≈ 1.5 ms in this run vs Orion's iPhone A19 Pro
measurement of ≈ 2.3 ms. One dispatch removed → **≈ +7 % tok/s** on
iPhone (31 → 33 tok/s on the current ANE-only baseline). Realising that
gain on device requires:

1. Re-running `install_3way_bundle.py` on the device-shipped bundle.
2. `xcrun devicectl device copy to … --source output/gemma4-e2b/bundle …`
3. Xcode scheme env `LLM_3CHUNK=1` (+ `COMPUTE_PLAN_AUDIT=1` once to
   confirm the 17-layer chunk stays 100 % ANE-resident).

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
