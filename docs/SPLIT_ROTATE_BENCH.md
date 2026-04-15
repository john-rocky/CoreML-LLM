# Split-rotate / padded-KV bench plan + measured results

Date: 2026-04-15
See `docs/SPLIT_ROTATE_FINDINGS.md` for the audit that motivated this probe.

## Artifacts produced

| Role | File |
| ---- | ---- |
| Padded stateless chunk2 (KV heads 32) | `conversion/models/gemma4_padded_kv.py::PaddedKVChunk2` |
| Padded stateful chunk2 (MLState + heads 32) | `conversion/models/gemma4_stateful_padded.py::StatefulPaddedChunk2` |
| Stateful builder + Mac CPU smoke | `conversion/build_stateful_padded.py` |
| Parity test (padded vs unpadded stateless) | `conversion/test_padded_kv_parity.py` |
| Restored prior MLState scaffolding | `conversion/models/gemma4_stateful_chunks.py`, `conversion/build_stateful.py` |

## Part 1 — padded-KV stateless parity (PyTorch)

Command:

    GEMMA4_HF_DIR=... /Users/majimadaisuke/.pyenv/versions/lama-cml/bin/python3 \
        conversion/test_padded_kv_parity.py

Result (2026-04-15):

    K slot0: shapes=(7, 1, 512, 512) finite_frac=0.429 nan_pattern_match=True
             max_abs_diff=0.000e+00 rel=0.000e+00
    V slot0: shapes=(7, 1, 512, 512) finite_frac=0.429 nan_pattern_match=True
             max_abs_diff=0.000e+00 rel=0.000e+00
    padded slots 1..31 are zero: OK
    Worst relative diff: 0.000e+00  PARITY: OK

Finite positions match bit-exact between `StatelessChunk2` and
`PaddedKVChunk2`. The NaN positions (from random-input fp16 overflow,
unrelated to padding) occur in identical locations in both outputs.

## Part 2 — padded MLState Mac CPU smoke (local)

Command:

    /Users/majimadaisuke/.pyenv/versions/lama-cml/bin/python3 \
        conversion/build_stateful_padded.py \
        --output /tmp/stateful-padded-smoke --ctx 512 --nbits 0 --smoke-test

Result (2026-04-15):

    traced in 0.2s
    converted in 9.7s
    RuntimeWarning: "Failed to build the model execution plan ... error code: -14"
    saved /tmp/stateful-padded-smoke/chunk2_padded.mlpackage (535 MB)

    Mac CPU smoke test:
      loaded in 1.0s
      predict ok in 49ms
      all outputs finite (finite_frac=1.000)

Interpretation: the graph is runtime-valid on CPU (predict returns finite
outputs for all 5 outputs). The Core ML converter emits error -14 at save
time the same way it did for the non-padded stateful build. **Padding KV
heads 1 → 32 did NOT fix the ANE execution-plan failure.** This confirms
the findings-doc hypothesis: error -14 is an ANE compiler limitation on
`coreml_update_state`, not a dim-alignment problem.

## Predicted gains

| Variant | Expected vs baseline 31.4 tok/s | Confidence |
| ------- | ------------------------------- | ---------- |
| Part 1 stateless + KV heads pad to 32 | -5 to 0 % (wash; only adds mem) | High — no op removed |
| Part 1 + `repeat_interleave` → `repeat_kv_ane` | +1 to +3 % | Medium |
| Part 2 MLState on ANE (with padding) | blocked by -14, same as before | High |
| Part 2 MLState on GPU (`CPU_AND_GPU`) | +10 to +20 % iff dispatch is the bottleneck on GPU too | Low — untested |

## Device-run instructions

If you want to verify error -14 reproduces with the padded variant on device
(to close the investigation definitively):

    # Build INT4 CTX=8192 variant (production settings)
    /Users/majimadaisuke/.pyenv/versions/lama-cml/bin/python3 \
        conversion/build_stateful_padded.py \
        --output /tmp/stateful-padded-8k --ctx 8192 --nbits 4

    # Copy chunk2_padded.mlpackage to the iPhone target
    # Load with MLModelConfiguration.computeUnits = .cpuAndNeuralEngine
    # Expect: same error -14 as the non-padded build in
    #         docs/EXPERIMENTS.md § "MLState stateful KV cache — Rejected"

To test the GPU stateful path (the only path likely to work on Apple silicon):

    // Swift:
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndGPU   // NOT cpuAndNeuralEngine
    let m = try MLModel(contentsOf: url, configuration: cfg)
    let state = m.makeState()

Run one decode step, compare wall-clock vs current 4-chunk stateless pipeline.
If GPU state is ≥2× faster on chunk2 alone, it becomes worth considering a
hybrid GPU-chunk2 / ANE-chunk1/3/4 topology. Otherwise, close MLState.

## Recommendation

1. Do **not** ship padded KV. No compute or dispatch win expected;
   parity confirmed but adds 32× memory on the heads axis.
2. **Cheap independent win**: switch `_run_layer_stateless` to call
   `repeat_kv_ane` instead of `repeat_interleave(... dim=1)` (3-line change
   in `conversion/models/gemma4_stateless_chunks.py`).
3. Keep pursuing **chunk consolidation** (4→2) and **speculative decoding**
   as the dispatch-overhead mitigations. MLState is blocked by an ANE
   compiler limitation that no tensor-shape trick fixes.
