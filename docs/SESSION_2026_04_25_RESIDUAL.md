# Gemma 4 E2B FP16 residual-stream measurement (2026-04-25)

**TL;DR — anemll's α=0.5 FP16 scaling hypothesis is empirically wrong for
Gemma 4 E2B. No scaling needed.**

## Why this measurement

- ANEMLL's `gemma3_converter.py:2238-2322` ships an `_apply_fp16_scaling`
  pass that scales embedding weights and post-norm gains so the residual
  stream stays inside fp16 range.
- The companion `GEMMA3_SCALING_FACTORS` dict (line 2218-2221) lists
  `gemma-3n-E2B: 0.5` and `gemma-3n-E4B: 0.5` with comment
  *"Estimate, needs verification"*.
- Ours is the first verification of those numbers.

## Method

`conversion/probe_residual_overflow.py` loads the HF text decoder twice
(bf16 ground truth + fp16 production proxy), runs N representative prompts
with `output_hidden_states=True`, records per-layer `max(|h|)` and
`mean(|h|)`, and flags any NaN.

Decision rule:

| bf16 max(|h|)              | recommended α  |
|---------------------------|----------------|
| < 30 000                   | none           |
| 30 000–60 000              | 0.7            |
| ≥ 60 000 or fp16 NaN       | 0.5            |

## Result

Three passes total — E2B short, E2B long, E4B — all CPU.

| Model | Run | prompts | max_tokens | bf16 max(|h|) | fp16 max(|h|) | Verdict |
|-------|-----|---------|------------|---------------|---------------|---------|
| E2B   | #1  | 4       | 128        | 106.5         | 105.0         | no scaling |
| E2B   | #2  | 8       | 512        | 135           | 132           | no scaling |
| E4B   | #3  | 4       | 256        | 161           | 161           | no scaling |

Per-layer values from run #2 (selected layers, fp16):

| layer | name      | max(|h|) | mean(|h|) |
|------:|-----------|---------:|----------:|
|  0    | embed_out |   13.9   |  0.83     |
|  4    | after_L3  |   54.8   |  0.53     |
| 14    | after_L13 |   15.5   |  0.62     |
| 24    | after_L23 |   39.9   |  0.97     |
| 29    | after_L28 |   86.9   |  1.27     |
| 35    | after_L34 |  131.9   |  3.09     |

`bf16` and `fp16` agree to within ~1 % at every layer. No NaN observed.
Final-layer max grows roughly with sequence length (`T=21 → 105`,
`T=64 → 132`), implying a soft ceiling well under 1 000 even at
production context lengths. fp16 has 5 orders of magnitude of headroom
to overflow.

## Implication for the action plan

- ❌ **DROP** "P1-2: FP16 residual scaling (α=0.5)". The intervention
  would be a no-op at best and a perplexity-neutral weight rewrite at
  worst.
- ✅ This refutes the anemll team's E2B/E4B hypothesis. If the user wants
  external citation, point to this doc; the anemll source explicitly
  flags those numbers as unverified.

## Open questions

1. ~~Does E4B residual behave the same?~~ Confirmed — peak max(|h|)≈161
   at L13 (still <30k). Slightly larger than E2B but same conclusion.
2. Does long-context production (T≈2048) push residual past the
   threshold? Linear extrapolation says no, but worth one Mac probe at
   real ctx if we ever start seeing decode quality issues.
