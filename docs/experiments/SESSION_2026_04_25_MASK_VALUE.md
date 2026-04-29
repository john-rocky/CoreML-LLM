# Mask-fill value probe — `-Inf` vs `-1e4` (2026-04-25)

**TL;DR — CoreML on Mac CPU_AND_NE produces bitwise-identical softmax
outputs for the two values. Keep `0xFC00` (-Inf) in production runtime.**

## Why this measurement

- `Sources/CoreMLLLM/ChunkedEngine.swift:2122,2136,2236,2282,2518,2539`
  fill the attention mask with `0xFC00` (= fp16 -Inf).
- `ml-ane-transformers/ane_transformers/reference/transformer.py:130`
  explicitly says: *"The recommended float value for preventing
  attention is -1e4. This allows for composition of multiple masks
  while staying in the float16-friendly range."*
- We don't compose masks (each layer takes one mask, full or sliding),
  so the overflow argument doesn't apply directly. But the question
  remained whether ANE's softmax kernel internally treats the two values
  differently anyway.

## Method

`conversion/probe_mask_value.py` builds a minimal `softmax(scores +
mask, dim=-1)` model, converts with `target=iOS18`,
`compute_units=CPU_AND_NE`, and runs it twice per random seed:

1. mask filled with `0xFC00` for masked positions
2. mask filled with `0xF0E2` (≈ -1e4)

Shape mirrors single-token decode attention: scores `(1,8,1,512)`, mask
`(1,1,1,512)`, 200 unmasked / 312 masked. Five seeds; scores drawn from
`U(-10, 10)` to approximate real post-scale Q@K^T magnitudes.

## Result

| seed | max\|Δ\| | argmax agree | NaN |
|-----:|---------:|-------------:|:---:|
| 0    | 0        | 100 %        | no  |
| 1    | 0        | 100 %        | no  |
| 2    | 0        | 100 %        | no  |
| 3    | 0        | 100 %        | no  |
| 4    | 0        | 100 %        | no  |

Bitwise identical across all seeds. No NaN, no probability shift, no
top-1 disagreement.

## Caveats

- Tested on Mac (CPU_AND_NE compute unit). For a 512-element softmax
  the dispatcher may pick CPU; we did not force ANE-only.
- iPhone ANE silicon may differ from Mac ANE (per
  `project_iphone_ane_sparsity` — iPhone is realLen-aware, Mac isn't),
  but mask-value handling is a softmax-internal numeric concern, less
  likely to differ across silicon.
- Production has been running with `0xFC00` for months at 32 tok/s; if
  the value broke ANE softmax we'd already see decode quality issues.

## Implication for the action plan

- ❌ **DROP** "P0-1 swap mask fill to -1e4". No action needed.
- The Apple recommendation stands for codebases that *compose* masks;
  ours doesn't, so the `-1e4` advice is non-binding for us.
