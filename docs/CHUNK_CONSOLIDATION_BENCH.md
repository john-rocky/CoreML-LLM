# Chunk consolidation bench — device run book

Date opened: 2026-04-15. Target hardware: iPhone 17 Pro, iOS 26.

## Why this exists

`docs/BASELINE_SPEED_AUDIT.md` confirms the 4-chunk decode chain is
bottlenecked by ANE dispatch overhead: ~2.3 ms/round-trip × 4 chunks =
~9 ms/step on top of compute. Halving chunks roughly halves that cost.

Predicted gain (back-of-envelope, 2K context, Gemma 4 E2B INT4):

  4 chunks → 2 chunks: ~9 ms → ~4.6 ms per step ≈ **+14 tok/s**
  4 chunks → 1 chunk : ~9 ms → ~2.3 ms per step ≈ **+21 tok/s (if ANE holds)**

From the 31.4 tok/s @ 2K baseline the theoretical ceilings are ~45 tok/s
(2-chunk) and ~52 tok/s (1-chunk). Both still below the 56.5 tok/s
LiteRT-LM target, but on the right trajectory.

## Pre-bench gate (CANNOT be skipped)

Run these on the Mac before copying anything to device:

```bash
# 1. Build both variants. Requires ./output/gemma4-e2b/hf_model present.
cd conversion
python build_merged_chunks.py --mode both --output /tmp/gemma4-merged

# 2. Parity: 2-chunk and 1-chunk must match the 4-chunk reference.
#    Uses CPU torch, no CoreML conversion noise.
python test_merged_parity.py --mode both
# Expected: "PARITY OK across 16 steps (cos >= 0.9999)."
# If it reports FAIL on any step: DO NOT PROCEED. Inspect the first
# failing step and diff the merged module vs its SWAChunk source.
```

If parity passes:

```bash
# 3. Copy the merged mlpackages alongside the existing 4-chunk files.
cp -R /tmp/gemma4-merged/merged_chunk1.mlpackage  ~/path/to/gemma4-e2b/
cp -R /tmp/gemma4-merged/merged_chunk2.mlpackage  ~/path/to/gemma4-e2b/
# Optional: copy merged_full.mlpackage too to try the 1-chunk path.
# Keep chunk1..chunk4 in place — the runtime uses them for
# prefill/speculative-verify, and as the 4-chunk fallback.
```

## On device — smoke

Copy the model directory to the device (USB sideload or `ModelDownloader`),
then launch the bench target.

The Swift runtime auto-detects the layout at load time and prints which
one it chose. Confirm before trusting numbers:

```
[Load] Decode layout: two-chunk
```

Or `one-chunk` if `merged_full.mlpackage` is present, or `four-chunk`
if neither merged variant is found.

### Smoke prompts

Pick one from each category in `Sources/accept-rate-bench/Prompts.swift`:
- `chat-define-transformer`   (chat)
- `code-complete-sum`          (code)
- `qa-where-is-swift`          (qa)
- `sum-para-ane`               (summary)

Run each with `--max-tokens 100`. The chat prompt is the primary signal
(`docs/PHASE_B_CHAT_CV_RESIDUAL.md`).

### Expected `[Profile-2chunk]` line shape

```
[Profile-2chunk] emb=0.0ms mask=0.0ms | m1=~20 m2=~20 (sum=~40ms) | predict=~40ms (~25 tok/s → target ~45 tok/s)
```

The `[Profile-1chunk]` line collapses to one `full=...` timing.

## Compute-plan audit (REQUIRED before reporting numbers)

The biggest risk (PR #17 pattern) is that the merged chunk silently
spills ops to CPU/GPU because its layer count exceeds the ANE compiler's
per-function stability ceiling (~15 layers historically). When that
happens tok/s regresses *below* the 4-chunk baseline even though wall-
clock looks like fewer dispatches.

Run the built-in audit by setting the env var before launch:

```bash
# Xcode: Product > Scheme > Edit Scheme > Run > Arguments > Environment:
#   Name: COMPUTE_PLAN_AUDIT
#   Value: 1

# Or from Terminal when using the CLI bench target:
COMPUTE_PLAN_AUDIT=1 ./coreml-llm-smoke --model ~/path/to/gemma4-e2b ...
```

The audit walks every MIL op in every decode chunk (auto-detects which
layout is present) and prints any op whose preferred device is not the
ANE. Summary line at the end:

```
[ComputePlan] merged_chunk1: all 1247 ops on Neural Engine
[ComputePlan] merged_chunk2: 42/1683 ops NOT on Neural Engine     ← red flag
[ComputePlan] ── summary: 42 non-ANE op(s) out of 2930 total across all chunks
```

**Pass criterion for merged variant**: `non-ANE op(s)` ≤ 5 % of total
ops. Anything higher: flag as regression, do not ship the merged
variant. Fall back to 4-chunk by simply deleting the `merged_*.mlpackage`
files (the runtime will auto-switch at next load).

## Tok/s bench

Three cold-cache runs per layout (reboot device between runs to clear
ANE compiler caches — first-token latency counts against us, but tok/s
is the headline):

```bash
# Layout auto-selected by file presence. To force 4-chunk for the
# control run, temporarily move merged files aside:
mkdir -p /tmp/merged-stash
mv ~/.../gemma4-e2b/merged_*.mlpackage /tmp/merged-stash/

# Reboot iPhone, then run:
./coreml-llm-smoke --model ~/.../gemma4-e2b \
    --prompt "Explain attention." --max-tokens 100

# Record tok/s from the final [Profile] line. Then restore merged files,
# reboot, and run again. Record again. Repeat with merged_full to test
# 1-chunk.
```

### Target numbers

| Layout   | Predicted tok/s @ 2K | Pass if ≥ | Ship if ≥ |
|----------|----------------------|-----------|-----------|
| 4-chunk  | 31.4 (baseline)      | —         | —         |
| 2-chunk  | ~45                  | 36        | 40        |
| 1-chunk  | ~52 (aspirational)   | 40        | 48        |

"Pass" means the consolidation mechanism didn't hurt. "Ship" means the
gain is large enough to justify the extra model download.

### At 8K context

Dispatch overhead is invariant with ctx, so the absolute ms/step saving
is the same. Percentage win shrinks because compute grows:

| Layout   | Baseline 8K | Predicted 8K |
|----------|-------------|--------------|
| 4-chunk  | 14.5        | 14.5         |
| 2-chunk  | 14.5        | ~16.0        |
| 1-chunk  | 14.5        | ~16.8        |

## Rollback / safety

- The 4-chunk `chunk1..chunk4.mlpackage` files are NOT modified or
  deleted by this work. They remain the default when merged files are
  absent.
- `ChunkedEngine.reset()` now clears merged caches too; no lifecycle
  changes in the public API.
- If the merged variant silently falls off ANE on a future iOS update,
  the mitigation is to delete `merged_*.mlpackage` from the model
  directory — the runtime will auto-detect four-chunk layout on next
  load without any code change.

## Reporting back

Post a comment on the consolidation PR with:
  1. `[Load] Decode layout: X-chunk` line (confirms what was exercised)
  2. ComputePlan summary line (confirms ANE placement)
  3. Three tok/s numbers per layout at 2K (cold-cache, reboot between)
  4. One tok/s number per layout at 8K (optional, signals whether the
     ctx-dependent compute growth is stable)

## Known open items

- Prefill path still uses the 4-chunk layout. Merging prefill chunks
  was explicitly deferred: prefill is a single call per prompt, so the
  dispatch saving is tiny, and the prefill graphs have different
  shapes (N=512 batched) that would need their own build script.
- Speculative verification (`verify_qK` functions) also stays on the
  4-chunk layout for the same reason.
- No 3-chunk variant was attempted. If 2-chunk compiles cleanly but
  1-chunk falls off ANE, a 3-way split (e.g. 0-11 / 12-23 / 24-34) is
  the obvious next experiment.
