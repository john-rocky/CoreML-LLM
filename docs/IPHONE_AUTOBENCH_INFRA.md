# iPhone AutoBench infrastructure (2026-05-13)

Headless tok/s bench on iPhone 17 Pro driven from the Mac shell. Unlocks
agent-side iteration (no human typing prompts) for MTP / FLy / chunk
topology sweeps.

## Components

1. **`Examples/CoreMLLLMChat/CoreMLLLMChat/AutoBench.swift`** —
   activated by `LLM_AUTOBENCH=1` env. Runs 4 prompt classes after
   model load:
   - narrative essay (chat free-form)
   - code BST (Python class)
   - list 30 emperors (structured)
   - "Say yes 30 times" (degenerate repetition baseline)

   Knobs:
   - `LLM_AUTOBENCH_MAX_TOKENS` (default 256)
   - `LLM_AUTOBENCH_MODEL` (default `gemma4-e2b`)
   - `LLM_AUTOBENCH_PROMPTS` (comma-separated subset)

   Output: `[AutoBench] <label>: tokens=N wall=Xs tok/s=Y` per prompt,
   then `[AutoBench] done` and `exit(0)`.

2. **`scripts/iphone_autobench_sweep.sh`** — env grid runner over
   AutoBench. Currently supports `k_use`, `fly_topk`, `bail_threshold`,
   `chunk_pipeline`, `l5_async`. Each sweep launches the app once per
   value, collects tok/s into `/tmp/iphone_sweep_<name>.tsv`.

## Standard invocation

```bash
DEVICE=$(xcrun devicectl list devices | grep "iPhone 17 Pro" | grep connected \
  | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}' | head -1)
xcrun devicectl device process launch \
  --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1"}' \
  com.example.CoreMLLLMChat
```

The `--console` flag attaches the app's stdout to the host TTY until
`exit(0)` runs.

## Constraints / gotchas

- **iPhone must be unlocked** before launch. Locked device returns
  `FBSOpenApplicationErrorDomain error 7` and the launch silently
  fails.
- **Don't pipe through `head -N`** in the foreground. SIGPIPE to
  devicectl truncates the console stream and orphans the on-device
  process. Use `... > /tmp/bench.log 2>&1 &` instead and grep the log
  after the background command completes.
- **Thermal throttling dominates after ~3 consecutive prompts.**
  iPhone 17 Pro thermal state escalates `nominal → fair → serious`
  during the run; with `serious`, chunk loads go from 0.1 s to 6-14 s
  and decode tok/s drops 50-60 %. ChunkedEngine inserts 1800 ms
  cool-down gaps automatically. For valid bench numbers, ensure the
  iPhone is **at room temperature and idle for ~5-10 min** before
  starting the bench.
- A previous bench session that ended with thermal=serious will
  contaminate the next run for several minutes. Look for
  `[Thermal] chunk-gap cool-down 1800 ms (state=serious)` lines as
  the throttle indicator.
- Initial cold-start model load is ~20 s; subsequent loads (within
  the same Xcode session) reuse the ANE compile cache and drop to
  0.1 s per chunk if not thermal-throttled.

## Calibrated baselines (iPhone 17 Pro, post-fc31660 stack, 256 tokens)

| prompt | T=1 (no MTP) | MTP K=2 default | MTP K=1 (iOS default) |
|---|---|---|---|
| narrative essay | ~32 | 31.17 (auto-bail) | TBD |
| code BST | ~32 | 40.89 (+28%) | 50.61 (+58%) |
| list 30 emperors | ~32 | TBD | TBD |
| yes 30 times | ~32 | TBD | TBD |

Mac equivalents from the same commits:

| prompt | Mac T=1 | Mac MTP K=16 FLy |
|---|---|---|
| narrative essay | 32.0 | 43.1 (+35%) |
| code BST | 32.0 | 63.9 (+100%) |
| list 30 emperors | 32.0 | 49.4 (+54%) |

iPhone scales ~60-80 % of Mac on MTP-positive prompts because the
verify cycle is structurally heavier on ANE 18.

## Known iPhone-only bugs caught here

1. **`kv13_v` shape mismatch** — iPhone ANE 18 transposes verify
   chunk input internally. Mac doesn't. Don't pre-allocate IOSurface
   backings for chained outputs; let CoreML marshal. Fixed in
   `5b68fb3`.
2. **`hidden_states_out` pixel buffer lock** — same root cause:
   `.dataPointer` read on a backed output locks the IOSurface, next
   cycle errors. Fixed in `5b68fb3`.
3. **MTP_K_USE=2 malloc double-free** — observed once on the K_USE
   sweep, immediately after the previous K_USE=1 run. Looks like a
   carried-over state issue. Default K_USE=2 (pre-`6d51439`) did not
   reproduce in isolation. Watch this if reverting to K_USE=2.

## Workflow

1. Edit Swift code.
2. `xcodebuild` from `Examples/CoreMLLLMChat/`.
3. `xcrun devicectl device install app`.
4. (Optional) `scripts/push_gemma4_e2b_bundle.sh /tmp/push-bundle` to
   refresh model bundle.
5. `xcrun devicectl device process launch --console ... > log 2>&1`.
6. Parse `[AutoBench]` lines from the log.
7. Cool the iPhone if results look thermally throttled.
