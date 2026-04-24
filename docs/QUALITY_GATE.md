# Quality gate — CoreML determinism oracle (PoC)

A Mac-only regression test that runs a small fixed corpus through the
shipping `CoreMLLLM` decode path at argmax (all drafters off) and
compares emitted token IDs against a committed oracle JSON file.

Scope: CoreML→CoreML determinism. It catches silent regressions in the
shipped decode path (re-conversion, runtime-flag flip, tokenizer-config
drift, etc.) without requiring an iPhone trip or a PyTorch reference.
A PyTorch-golden mode can be bolted on later — it would only change the
*source* of the reference tokens; the harness and oracle format stay the
same.

## When to run

- Before any iPhone trip that touches `Sources/CoreMLLLM/` decode code.
- After any `conversion/` change that regenerates a deployed chunk.
- Whenever `lastEmittedTokenIDs`-producing code (sampling, KV writes,
  chunk boundaries) is refactored.

Fits the "Mac Studio first, device only when necessary" rule (memory
`feedback_mac_first_validate.md`). A pass is not a release gate — it's
a pre-flight: iPhone A/B still lives in `docs/HANDOFF.md`.

## Seeding an oracle

Oracles live in `Tests/oracles/<model-bundle>-argmax.json` and are
keyed by the stable prompt IDs in
`Sources/determinism-oracle/main.swift`. The file is **not** auto-generated
in CI because it requires the model bundle locally. Seed once per bundle:

```bash
swift run -c release determinism-oracle --record \
  --model ~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b \
  --oracle Tests/oracles/gemma4-e2b-argmax.json \
  --label "$(git rev-parse --short HEAD)"
```

Commit the resulting JSON. The `label` field records the git SHA that
produced the oracle so a future reader can track down what changed.

## Verifying

```bash
swift run -c release determinism-oracle \
  --model ~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b \
  --oracle Tests/oracles/gemma4-e2b-argmax.json
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | All prompts match |
| 1 | At least one prompt diverged (output lists first-divergence index) |
| 2 | Runtime error (model load failed, oracle missing, etc.) |

## When an oracle breaks

Divergence is **not** automatically a bug — it can be a legitimate
intentional change (re-convert, sampling tweak). The fix path:

1. Look at the first-divergence index the tool prints.
2. Decide: is the new behaviour correct? If yes, re-record with the
   current git SHA as `--label`. If no, the recent change regressed
   decode.
3. Never silently re-record to make a failing test pass — the whole
   value of the oracle is that someone has to look.

## Corpus policy

Corpus lives in `Sources/determinism-oracle/main.swift:Oracle.corpus`.
Keep it:

- Small (≤ ~5 prompts) so Mac runs stay under a minute.
- Short (`--max-tokens` defaults to 16) — we want coverage of the
  first few decode steps, not long-tail completions.
- Stable IDs — renaming an ID breaks every existing oracle.

Adding a new prompt is a breaking change for oracles that pin to the
previous set; bump `schema` in the JSON if/when that happens.

## Future extensions (not in PoC)

- PyTorch-golden mode: pre-compute tokens with HF transformers at
  `do_sample=False`, record into a parallel oracle format, compare.
- Per-chunk hidden-state hashes for finer-grained localisation when a
  divergence happens deep into decode.
- INT4 / pruning perplexity regression (needs a held-out eval corpus).

These are separate work items — do not bolt them onto the PoC without a
scoping decision.
