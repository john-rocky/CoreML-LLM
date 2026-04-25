# Oracles

Committed reference outputs for the CoreML determinism oracle
(`swift run determinism-oracle`). See `docs/QUALITY_GATE.md` for how to
seed and verify.

One file per model bundle, e.g. `gemma4-e2b-argmax.json`. Seed locally
with `--record` once; commit; verify on subsequent runs.
