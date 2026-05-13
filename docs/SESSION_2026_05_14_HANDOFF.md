# Next-Session Handoff — iPhone iteration playbook (2026-05-14)

Drop-in prompt + cookbook for the next session. Continues the 2026-05-13
session's empirical Mac + iPhone push without re-deriving the constraints.

---

## Quick state

* Branch: `feat/mtp-iphone-perf` (19 commits ahead of `fc31660`).
* Production iPhone defaults (from today's commits, all confirmed
  empirically on iPhone 17 Pro):
  * 4-chunk + MTP + FLy K=16 + centroid drafter + fp16 K/V verify chunks
  * `MTP_K_USE` = K-1 = 2 (cold-start robust; commit `e9af22d`)
  * `MTP_FALLBACK_THRESHOLD` = 0.30 (iOS only; commit `beb5bef`)
  * decode output backings DISABLED (iPhone crash; commit `5b68fb3`)
  * verify output backings DISABLED (same root cause)
  * AutoBench with 24-token "Hello." warmup (commit `bc5b04a`)
* Bundle layout on iPhone: `Documents/Models/gemma4-e2b/`
  (dereferenced copy of `output/gemma4-e2b/bundle_diff_logits/` + the
  149 MB centroid drafter).
* **Overnight 2026-05-14 (commits `93e8c5f`, `7b69320`)**: Round E
  per-prompt K_USE adapter implemented and opt-in via
  `MTP_PER_PROMPT_KUSE=1`. No default change. Telemetry line emits
  one decision per prompt for grep-friendly bench post-mortem.
* **Overnight 2026-05-14 (commit `019854d`)**: Round F SuffixDecoding
  wired into MTP / DrafterUnion / LookaheadEngine. The engine code
  existed as 559 LOC of dead code — `union.suffix` was never set.
  Now opt-in via `LLM_SUFFIX_DRAFT=1` (load trie) +
  `MTP_SUFFIX_DRAFT=1` (consult inside MTP). Lossless by construction.
  See Round F below.

## Validated baselines

| platform | prompt | T=1 | MTP-on (today's stack) | gain |
|---|---|---|---|---|
| Mac M | narrative essay | 32.0 | **43.1** | +35 % |
| Mac M | code BST | 32.0 | **63.9** | +100 % |
| Mac M | list 30 emperors | 32.0 | **49.4** | +54 % |
| iPhone 17 Pro (cool) | narrative essay | ~32 | **31** (auto-bail → T=1) | parity (no regression) |
| iPhone 17 Pro (cool, warm drafter) | code BST | ~32 | **40** (K_USE=2) or **50** (K_USE=1) | **+28 %** / +58 % |
| iPhone (long-form, thermal=serious) | any | — | 20-30 | thermal-bound |

Details + variability sources: `docs/IPHONE_PERF_EMPIRICAL_2026_05_13.md`.

## Tomorrow's prioritised lever queue

iPhone must be **cool** (overnight idle, `state=fair`) before any
bench. The first bench load should complete in ~20 s (vs 100+ s when
thermal). If first chunk takes > 1 s, abort and wait another 10 min.

### Round A — Validate today's defaults on cool iPhone

```bash
DEVICE=$(xcrun devicectl list devices | grep "iPhone 17 Pro" | grep connected \
  | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}' | head -1)
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_MAX_TOKENS": "256", "LLM_AUTOBENCH_PROMPTS": "narrative,code,list"}' \
  com.example.CoreMLLLMChat > /tmp/bench_round_a.log 2>&1
```

Expected (cool):
* narrative ~31, code ~40, list ~37 tok/s.
* Confirm output coherence (not "def def def" garbage).

If expected → proceed to Round B. If garbage or crash → diagnose
before sweeping.

### Round B — K_USE=1 cool + warmup retest

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_MAX_TOKENS": "256", "LLM_AUTOBENCH_PROMPTS": "narrative,code", "MTP_K_USE": "1"}' \
  com.example.CoreMLLLMChat > /tmp/bench_kuse1.log 2>&1
```

Hypothesis: warmup prompt activates the drafter so K_USE=1 cold-bail
doesn't fire. If code lands at **50+ tok/s** reproducibly across 2-3
clean runs, K_USE=1 becomes the right iOS default for code-heavy
workloads. Cool down 10 min between repeats.

### Round C — FLy K sweep (use `scripts/iphone_autobench_sweep.sh`)

```bash
bash scripts/iphone_autobench_sweep.sh fly_topk code
```

Tests `MTP_FLY_TOPK` ∈ {8, 16, 24, 32} on code only (most likely
prompt to expose the FLy ceiling). Cool down 10 min between sweep
values (the script does NOT auto-cool — manually pause if needed).

* K=16 is current default. K=24 / 32 on Mac plateaued or hurt; iPhone
  may differ because the drafter's marginal extra top-K candidates
  are cheaper to verify on iPhone's lighter K=1 cycle.

### Round D — Fallback threshold sweep

```bash
bash scripts/iphone_autobench_sweep.sh bail_threshold narrative
```

Tests `MTP_FALLBACK_THRESHOLD` ∈ {0.20, 0.25, 0.30 (current), 0.35, 0.40}
on narrative (the prompt class that hits the bail). 0.20 might
recover some narrative MTP gain at the cost of some net-negative
cycles; 0.35-0.40 may bail too eagerly. Empirical sweet spot lives
somewhere in this range; today's default was a snap judgment.

### Round E — Per-prompt K_USE adapter (SHIPPED 2026-05-14 night)

Heuristic: after the first non-fallback spec cycle, sense `matchCount`.
* `matchCount == compareLen` (K-1 full match): drafter on streak → stick
  `K_USE = 1` for the rest of the prompt → save the K-1 → 1 drafter
  step (~6 ms iPhone) per cycle while keeping 2 tokens/cycle on
  continued full accept.
* otherwise: stick `K_USE = K-1` (default) so partial-match cycles
  keep rolling acceptance above the 0.30 bail threshold.

Reset on `reset()` so each prompt re-senses. Sticky (one-shot) so a
mid-prompt accept dip doesn't flip back — flipping every cycle was
the HF +2/-1 schedule that already lost (-10 pp vs static).

Opt-in: `MTP_PER_PROMPT_KUSE=1`. Skipped silently when user sets
`MTP_K_USE` explicitly or `MTP_K_ADAPTIVE=1`.

Approx +5-10 % on code, neutral on narrative (narrative auto-bails
to T=1 anyway; adapter won't fire because the first MTP cycle
never gets to full-match before EMA decays).

Bench:
```bash
bash scripts/iphone_autobench_sweep.sh per_prompt_kuse code
```
Compares `MTP_PER_PROMPT_KUSE=0` (current default K-1) vs `=1` (adapter
on). Cool iPhone, 10-min gap between runs.

### Round F — SuffixDecoding inside MTP path (SHIPPED 2026-05-14 night)

Cross-session persistent suffix trie. Implementation existed
(`SuffixSpeculativeEngine`, `SuffixTree`, ~559 LOC) but had never been
wired — `DrafterUnion.suffix` / `LookaheadEngine.suffix` were
unconditionally nil. This commit wires the shared trie into
DrafterUnion + LookaheadEngine + **MtpSpeculativeEngine** (a new
integration), with `<Documents>/suffix_tree.json` persistence
(`saveEvery=256` on iOS — every ~10 s of decode, vs the original 64
that would have stalled every couple seconds).

In the MTP path: after L5 / PLD miss, the trie is queried with
`emittedHistory + nextID` as context. Hit (≥ `kEffective` proposals
returned) skips the drafter chain entirely (~6-12 ms iPhone saved per
cycle). Verify still strict-checks → **lossless**. The trie grows via
`applyCommit` so subsequent cycles, prompts, and sessions benefit.

Opt-in two-stage:
1. Launch: `LLM_SUFFIX_DRAFT=1` (loads the trie from Documents).
2. Per-cycle: `MTP_SUFFIX_DRAFT=1` (consults the trie inside MTP cycles).

Cold-start tells: first generation has empty trie → no benefit (one
small per-cycle CPU lookup that misses, microseconds). Trie grows
with accepted commits. Second-and-later generations on similar prompts
get cross-session matches. Within-prompt benefit fires on repetitive
output (lists, code, templated text).

Bench:
```bash
bash scripts/iphone_autobench_sweep.sh suffix_draft code
# sweep script auto-sets LLM_SUFFIX_DRAFT=1 alongside MTP_SUFFIX_DRAFT.
```
For the cleanest measurement, run twice in a row — the first pass
seeds the trie, the second pass reads it. Cool iPhone, 10-min gap
between paired runs.

Expectations:
* Code BST cold: parity-ish (trie is empty)
* Code BST warm (after seed gen): +5-15 % if intra-prompt suffix
  matches fire on `{` / `}` / `;` / `for (i...)` patterns
* Narrative: parity (no repetition — trie produces few hits)

If empirical confirms positive gain → flip `MTP_SUFFIX_DRAFT=1` default
on inside `MtpSpeculativeEngine` (similar precedent: `MTP_L5_ASYNC` is
default-on).

## AutoBench command reference

| variant | command snippet |
|---|---|
| Default (4 prompts + warmup) | `--environment-variables '{"LLM_AUTOBENCH": "1"}'` |
| Single prompt | `... "LLM_AUTOBENCH_PROMPTS": "code" ...` |
| Skip warmup (for cold-bench diagnostic) | `... "LLM_AUTOBENCH_NO_WARMUP": "1" ...` |
| Shorter timed runs | `... "LLM_AUTOBENCH_MAX_TOKENS": "128" ...` (faster, more noisy) |
| Different bundle | `... "LLM_AUTOBENCH_MODEL": "gemma4-e2b-stateful" ...` |
| MTP off (T=1 baseline) | — no env; modify Swift to read `LLM_AUTOBENCH_MTP_OFF` if needed |

Output prefix: `[AutoBench]`. Parse with
`grep -aE '^\[AutoBench\] \w+: tok/s='`.

Thermal indicator: `grep -E 'Thermal.*state=(serious|critical)'` —
if hits land before the first prompt, abort and wait.

## Tactical gotchas (must read)

* **iPhone must be unlocked** before launch — `FBSOpenApplicationErrorDomain
  error 7`. Unlock with Face ID then immediately fire the bench.
* **Never `| head -N`** on `--console` output — SIGPIPE truncates
  devicectl, hangs the on-device app. Use `> log 2>&1 &` then grep.
* **iPhone thermal=serious after 4-5 consecutive 256-token benches.**
  Cool-down: 10 min minimum, 20 min for full recovery after heavy
  sessions. Don't chain sweeps without cool gaps.
* **Drafter cold-start**: first MTP cycle accept rate is unreliable.
  AutoBench's warmup prompt fixes this. Don't disable warmup unless
  testing the cold path itself.
* **App watchdog kill**: iOS kills the app if it doesn't make UI
  progress for ~5 min during launch. Under thermal=serious, chunk
  loads alone can exceed this. Watchdog kill manifests as devicectl
  console hanging silently (no further log output).

## Lever priorities ranked by ROI

| lever | est gain | est cost | risk |
|---|---|---|---|
| K_USE=1 + warmup validation | +20-30 % code on cool | 1 cool bench cycle | low |
| FLy K sweep | +5-15 % code | 4 cool bench cycles | low |
| Fallback threshold tune | +5-15 % narrative | 5 cool bench cycles | low |
| Per-prompt K_USE adapter | +5-10 % code, neutral narrative | 1-2 h Swift + bench | medium |
| Drafter Path B retrain | narrative +30 % (acc 0.25→0.50) | 1 GPU-week | high cost |
| Custom 4-chunk verify topology | iPhone-specific gain TBD | 3-5 h Python + iPhone bench | medium |

The first 3 are env knobs only — no rebuild. Should saturate today's
training-free iPhone ceiling.

## Honest framing for tomorrow

The Mac wins (+30-100 %) are real and shipped. iPhone empirical so
far: code +28 % validated, narrative parity. The "iPhone 1.5×
training-free ceiling = 1.22-1.25 ×" estimate from memory
`project_lever_hunt_ceiling.md` is consistent with today's +28 %
ceiling (= 1.28 ×). To push past it for narrative, drafter retrain
is the structural lever; for code, K_USE=1 + warmup may still
deliver another +20 %.

Don't oversell. Default-on iPhone narrative MTP is *no longer a
regression*; that itself is a real ship today.

## File index (today's session)

| file | purpose |
|---|---|
| `docs/IPHONE_AUTOBENCH_INFRA.md` | AutoBench infra reference |
| `docs/IPHONE_PERF_EMPIRICAL_2026_05_13.md` | iPhone bench results + variability sources |
| `docs/SESSION_2026_05_13_MISTAKES.md` | 12 mistakes from today, what should've happened |
| `docs/SESSION_2026_05_13_SCRATCH_STACK.md` | Mac side scratch + backings stack |
| `Examples/CoreMLLLMChat/CoreMLLLMChat/AutoBench.swift` | iPhone bench Swift |
| `scripts/iphone_autobench_sweep.sh` | env grid runner |
| `scripts/push_gemma4_e2b_bundle.sh` | bundle push (use /tmp/push-bundle deref'd copy) |

Master roadmap: still
`docs/ROADMAP_2026_04_26.md` (older but conceptually current). Today's
session was an unplanned but high-yield iPhone-targeted detour.
