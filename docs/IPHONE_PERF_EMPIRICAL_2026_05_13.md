# iPhone 17 Pro perf — empirical (2026-05-13)

Headless AutoBench runs on iPhone 17 Pro, 4-chunk + MTP + FLy K=16 +
centroid drafter, after the today's commit stack (`fc31660..` HEAD).

## Production iPhone bench config

| component | value |
|---|---|
| Bundle | `output/gemma4-e2b/bundle_diff_logits/` (fp16 K/V verify chunks) |
| Drafter | `mtp_drafter_centroid.mlmodelc` (149 MB) — auto-pushed via `push_gemma4_e2b_bundle.sh` |
| Topology | 4-chunk decode + 4-chunk verify (3-chunk verify dead — `project_3way_mtp_dead_drafter_mismatch`) |
| `MTP_K_USE` | default = K-1 = 2 (K_USE=1 cold-bails — see mistake #11) |
| `MTP_FLY_TOPK` | iOS default 16 (lossy, sweet spot — quality / accept) |
| `MTP_FALLBACK_THRESHOLD` | iOS default 0.30 (narrative auto-bails to T=1) |
| Decode output backings | disabled (iPhone-only crash — see `5b68fb3`) |

## Measured baselines

| prompt class | iPhone tok/s | conditions | iPhone vs Mac | iPhone vs T=1 |
|---|---|---|---|---|
| Narrative essay (warm) | **31.17** | acc 0.25 → bail @ 0.30 → T=1 | 43.1 Mac (-28 %) | parity |
| Code BST (warm, 2nd prompt in sweep) | **50.61** | acc 1.0 plateau, 148 stable cycles, K_USE=1 explicit | 63.9 Mac (-21 %) | **+58 %** |
| Code BST (cold, 1st prompt) | 31.02 | K_USE=1 cold-bail → T=1 | — | parity |
| Code BST (warm, K_USE=2 default) | **40.89** | first valid bench post-decode-backings-fix | 63.9 Mac (-36 %) | **+28 %** |
| List 30 emperors | 16.40 | thermal serious (after multiple back-to-back benches) | 49.4 Mac | -49 % thermal |
| Yes 30 times | 28.99 | thermal serious | — | thermal |

## Variability sources (high)

1. **Drafter warm vs cold**:
   * Cold (1st MTP cycle after model load): drafter makes wrong prediction → accept 0/K → rolling EMA decays past 0.30 → auto-bail → T=1 baseline for the prompt.
   * Warm (post 1 prior prompt): drafter predictions land → 30-100 % accept → MTP runs through.
   * **Mitigation**: AutoBench prepends "Hello." warmup (commit `bc5b04a`). Skip via `LLM_AUTOBENCH_NO_WARMUP=1`.

2. **Thermal state escalation**:
   * Cool (`state=fair`): chunk loads 0.1 s, decode 31-37 tok/s.
   * Serious (`state=serious`, after 3+ consecutive long benches): chunk loads 5-22 s, decode degraded by 30-50 %.
   * Cool-down between serious-state benches: **15+ min** (10 min not always enough on iPhone 17 Pro).
   * ChunkedEngine inserts 1.8 s thermal gaps between chunks on serious; total bench load time goes from 20 s → 100+ s.

3. **K_USE choice**:
   * `K_USE=1`: brittle cold (auto-bails), aggressive warm (+58 % code).
   * `K_USE=2` (default): robust cold (1/2 partial accepts sustain rolling), conservative warm (+28 % code).

4. **App lifecycle**:
   * Fresh install → AutoBench cold-start.
   * Reinstall preserves Documents → bundle stays cached.
   * Watchdog kill if app appears unresponsive during thermal-throttled launch (>~5 min).

## Reproducible iPhone production numbers (cool + warmup)

After commit `bc5b04a` (warmup) + `e9af22d` (K_USE=2 revert):

| prompt | iPhone tok/s | gain vs T=1 |
|---|---|---|
| Narrative essay | 31 (parity, auto-bails) | 0 % |
| Code BST | ~40 (TBD, requires fresh cool bench) | ~+25 % |
| List 30 emperors | TBD | TBD |

These are the numbers users actually see in production. Mac wins
(narrative +35 %, code +100 %) don't fully transfer because iPhone
ANE 18 verify cycle is ~50 % heavier than Mac M ANE.

## What today's commit stack actually delivered iPhone-side

1. **No regression** vs fc31660 production state (centroid drafter,
   FLy K=16 already iOS default since 2026-05-13).
2. **Fixed crashes** on long-form bench: decode-output-backings revert
   (`5b68fb3`) + verify-backings disabled (in `14b5822`) prevent the
   pixel-buffer-lock + kv13_v shape-mismatch failures introduced
   earlier in the day.
3. **Auto-bail floor** (`beb5bef`): iOS=0.30 keeps narrative at T=1
   parity instead of running drafter cycles that don't pay off, so
   long-form narrative chat doesn't regress on iPhone.
4. **Headless bench infrastructure** (`2433734` + `bc5b04a`): future
   iPhone-side iteration unblocked.

## What does NOT transfer Mac → iPhone

* **K_USE=1 default** — Mac sweep showed code +24 %, iPhone needs warm
  drafter to repro and bails cold. Default reverted (`e9af22d`).
* **3-chunk + MTP** — Mac confirmed 25-38 % slower than 4-chunk MTP;
  iPhone not even tested because Mac proved dead end
  (`project_3way_mtp_dead_drafter_mismatch`).
* **Decode output backings** — Mac no-op, iPhone crash
  (`project_verify_backings_bug_fixed` extended).

## What's still credibly on the table for iPhone speedup

| lever | est gain | cost |
|---|---|---|
| Drafter Path B retrain (lift narrative acc 0.25 → 0.50) | +30 % narrative | 1 GPU-week |
| Per-prompt K_USE adapter (sense prompt type, switch K_USE) | +5-10 % code, neutral narrative | 1-2 day |
| ANE-resident drafter (audit 13 ms iPhone drafter cycle) | maybe -3 ms = +5 % | research |
| iPhone-specific verify chunk topology (smaller, ANE-friendly) | TBD | training-equivalent build |

Memory's "training-free iPhone 1.5× ceiling = 1.22–1.25×" remains the
estimated upper bound. Today we sit at iPhone code +28 % (1.28×) — at
the ceiling for the structured / code workload.
