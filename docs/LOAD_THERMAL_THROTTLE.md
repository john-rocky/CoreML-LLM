# Load-time thermal throttle

Cold model load is the hottest moment of the session: ANE compile daemon,
P-cores, and weight paging all peak with no breathing room between phases.
This doc covers the env-tunable knobs added to spread that work out so the
device clocks down between phases.

## Knobs

| Variable | Default | Effect |
|---|---|---|
| `LLM_LOAD_COOL_MS` | `300` | Base cool-down (ms) inserted between heavy load phases. `0` disables (back-to-back, original behaviour). Multiplied by thermal-state factor at runtime. |
| `LLM_LOAD_LITE` | unset | When `1`, skips optional auxiliary prewarms: vision MLModel load + dry forward, EAGLE-3 draft+fusion warm, and the finalPrewarm prefill/verify/transition tail. Pays first-call compile cost on those paths instead. |
| `LLM_LOAD_MAX_PARALLEL` | `1` | Already shipped — caps simultaneous chunk compiles. `1` (sequential) is the existing default; cool-down only fires in this mode. |

## Thermal-state multiplier

`ThermalThrottle.coolDown` reads `ProcessInfo.processInfo.thermalState` at
each call and scales the wait:

| State | Multiplier |
|---|---|
| `.nominal` | 1× |
| `.fair` | 2× |
| `.serious` | 6× |
| `.critical` | 15× |

A cool device pays only the configured base wait. A device already in
`.serious` waits long enough to actually cool, and the scaled wait is
logged so the spike is visible in console output.

## Phase placement

Cool-downs fire at:

1. Between sequential chunk compiles (`ChunkedEngine.load`, only when `cap == 1`).
2. Before the 4-step early prewarm.
3. Before `finalPrewarm` (decode + verify + transition warmup).
4. Before vision GPU MLModel load.
5. Before vision ANE MLModel load (only when `LLM_VISION_FORCE_ANE=1`).
6. Before the EAGLE-3 draft+fusion prewarm (only when EAGLE-3 enabled).

## Workload reductions (always on)

- `finalPrewarm` decode loop: 8 → 4 steps. Combined with the 4-step early
  prewarm this gives the same 8 total dispatches across the load, just
  spread with a cool-down gap. Steady-state perf parity is the assumption;
  if iPhone tok/s regresses, revert that constant.

## Trade-offs

- **Wall-clock load time**: nominal device adds ~3 s
  (7 chunks × 300 ms + 4 phase cool-downs × 300 ms). A `.serious`-state
  device adds ~20 s but only because it needed to.
- **Lite mode first-call cost**: skipping vision load defers ~1 s GPU load
  and ~30 s GPU compile to first image prompt. Skipping EAGLE-3 prewarm
  costs ~100 ms extra on first speculative burst. Skipping finalPrewarm
  prefill/verify/transition costs ~100-300 ms extra on first user prompt.

## How to verify on device

Compare console output of two cold launches:

- baseline: `LLM_LOAD_COOL_MS=0` to disable the new gaps.
- new default: no env vars set.

Also try `LLM_LOAD_LITE=1` for text-only sessions and confirm vision/EAGLE-3
prewarm lines are absent. The `[Thermal] ... cool-down N ms (state=...)`
line only fires when scaled past `.nominal`, so its absence means the chip
stayed cool.
