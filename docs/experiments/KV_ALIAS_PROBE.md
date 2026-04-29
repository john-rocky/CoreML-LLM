# Within-layer K=V alias probe — dead for Gemma 4 E2B (2026-04-24)

## Question

`docs/ANE_ONLY_LEVERS.md` §D asserts:

> Within-layer K=V alias per Gemma 4 design NOT done — we store kv14_k and
> kv14_v as separate tensors even though the architecture guarantees
> equality in global layers.

If true, aliasing k14 = v14 at the cache level would:

- halve `kv14` buffer size (ctx × 512 × fp16 bytes per layer)
- drop one per-step matmul (v_proj) in the global-attention producer
- shave copyBack traffic in the engine

Probe: is the "architecture guarantees equality" claim correct for
Gemma 4 E2B?

## Method

`conversion/probe_kv_alias.py` loads the HF Gemma 4 E2B weights and
compares `k_proj.weight` vs `v_proj.weight` for every full-attn layer
(L4, L9, L14, L19, L24, L29, L34 — producer L14 plus six consumers).

Also inspects `config.json` for any `k_eq_v`-style flag.

## Result

```
Full-attn layers: [4, 9, 14, 19, 24, 29, 34]

  XX L 4  shape=(512, 1536, 1, 1)  cos=0.001506  max_abs=3.5645e-01
  XX L 9  shape=(512, 1536, 1, 1)  cos=-0.001149  max_abs=3.5034e-01
  XX L14  shape=(512, 1536, 1, 1)  cos=0.000204  max_abs=3.7630e-01
  XX L19  shape=(512, 1536, 1, 1)  cos=0.001572  max_abs=1.1890e-01
  XX L24  shape=(512, 1536, 1, 1)  cos=0.000060  max_abs=1.1914e-01
  XX L29  shape=(512, 1536, 1, 1)  cos=-0.000008  max_abs=1.1646e-01
  XX L34  shape=(512, 1536, 1, 1)  cos=-0.000152  max_abs=1.1914e-01

❌ K ≠ V at the weight level — within-layer alias does NOT apply
```

Every full-attn layer's K and V weights are numerically **orthogonal**
(cos ≈ 0) with max_abs in the 0.12–0.37 range. They are independently-
trained projections, not tied weights.

Confirming at the config level — `config.json` declares:

```json
"text_config": {
    ...
    "attention_k_eq_v": false,
    ...
}
```

Gemma 4 does support a K=V tied mode (the flag exists in the architecture
definition), but **E2B ships with it disabled**. Other variants (E4B,
27B MoE) would need the same probe re-run before claiming alias.

## Verdict — skip for E2B

The §D hypothesis does not apply to the E2B weights we ship. Aliasing
k14 to v14 would force V = K numerically, which is a quality regression
— the self-attention output would degenerate to `softmax(QK) K` instead
of `softmax(QK) V`, losing all learned V-space structure.

Forcing alias without retraining = lossy. Since this session's goal is
strict lossless optimization, closing this lever.

## Recoverable cousin

If a future E4B or larger variant flips `attention_k_eq_v` to `true`,
this probe re-runs cheaply. `conversion/probe_kv_alias.py` is kept in
the tree for that case.

## Next lever

**Blockwise-32 per-block palettization** (`docs/PRIORITY_ROADMAP.md` §5h):
swap the current `granularity="per_grouped_channel", group_size=32` for
`granularity="per_block", block_size=32`. Error bounds uniformly
tightened without changing bit width; a pure reconvert pass.
