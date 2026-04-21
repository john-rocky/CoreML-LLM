# D-1 Wiring Patch Plan: FusedQKV + FusedGateUp

**Branch:** `feat/pre-conv-optimizations`
**Date:** 2026-04-15
**Background:** `docs/IMPLEMENTATION_LOG_2026_04_15.md` item D-1;
`docs/GEMMA4_ANE_REWRITES.md` rewrites #1 and #2.
**Fused modules:** already defined in
`conversion/models/gemma4_fused_modules.py` (`FusedQKV`, `FusedGateUp`,
`fuse_layer_projections()`).

This document is a line-by-line wiring plan. It enumerates every existing
call site of the split q/k/v/gate/up projections across all Gemma 4
chunk-builder files, produces a patch diff per site, recommends a
weight-loader strategy, lays out the validation plan, and closes with a
land-it-in-one-PR checklist. All line numbers were verified against the
tip of `feat/pre-conv-optimizations` via `Grep` + `Read`.

Verified numeric facts (not assumptions):
- `conversion/models/gemma4.py:41-43` — `num_attention_heads = 8`,
  `num_key_value_heads = 1`, `head_dim = 256`. The brief claimed 10 q-heads
  and 1 kv-head; the config says 8. **All patches below use 8 q-heads, 1
  kv-head.** Callers of the fused module that compute shapes from
  `config.num_attention_heads` pick this up automatically.
- `conversion/models/gemma4.py:44` — `global_head_dim = 512` for full
  attention; sliding layers use 256. `FusedQKV` is built per-layer, so its
  `q_dim/kv_dim` naturally tracks `config.get_head_dim(layer_idx)`.
- `conversion/models/gemma4.py:56`, `:95-97` — `num_kv_shared_layers = 20`,
  `is_kv_shared(i) = i >= 15`. So L15-L34 are shared (20 layers), L0-L14
  own their K/V.
- `conversion/models/gemma4.py:378-400` — projections live directly as
  `nn.Conv2d` inside `nn.ModuleDict` (no `.conv` indirection). Matches the
  `fuse_layer_projections()` helper.

---

## 1. Summary table

| File | QKV sites | Gate/Up sites | Est. LOC change | Notes |
|---|---|---|---|---|
| `conversion/models/gemma4_swa_chunks.py` | 6 (3 decode + 3 verify) | 4 (2 decode + 2 verify) | ~35 | Core decode + verify helpers. Covers SWAChunk1-4, SWAVerifyChunk1-4, MergedChunk1/12/34 indirectly. |
| `conversion/models/gemma4_prefill_chunks.py` | 3 (q_raw, k_raw, v_raw) | 2 | ~18 | `_run_layer_prefill`. |
| `conversion/models/gemma4_swa_flash.py` | 3 | 2 | ~18 | Flash SWA variant. |
| `conversion/models/gemma4_swa_wfa.py` | 3 | 2 | ~18 | Windowed full-attn variant. |
| `conversion/models/gemma4_swa_cascading.py` | 3 | 2 | ~18 | Cascading variant. |
| `conversion/models/gemma4_stateless_chunks.py` | 3 | 2 | ~18 | Stateless chunks (monolithic). |
| `conversion/models/gemma4_lite_chunks.py` | 3 | 2 | ~18 | Lite single-model path. |
| `conversion/models/gemma4_lite_wrapper.py` | 3 | 2 | ~18 | Lite wrapper. |
| `conversion/models/gemma4.py` | 0 (structural) | 0 | ~25 | `Gemma4DecoderLayer.__init__` (optional fused ctor) + `load_weights()` post-fuse hook. |
| `conversion/models/gemma4_swa_merged1.py` | 0 (delegates) | 0 | 0 | Calls `_run_layer_swa` — inherits fix. |
| `conversion/models/gemma4_swa_merged2.py` | 0 (delegates) | 0 | 0 | Same. |
| `conversion/models/gemma4_decoder.py` | 3 (`nn.Linear`) | — | 0 | Unrelated legacy path (`nn.Linear`). **Leave untouched** — not on shipping graph (`gemma4_decoder.py` is the Medusa-style reference, not the chunk pipeline). |
| `conversion/models/qwen2.py` | n/a | n/a | 0 | Different architecture. Out of scope. |

**Total:** 24 QKV call sites, 16 Gate/Up call sites, ~186 LOC touched.
One-PR scope is reasonable because the mechanical transformation is
identical at every site.

---

## 2. Per-file patches

The canonical transformation pattern is:

**Before (QKV, single-token decode shape):**
```python
q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
...
k = layer.self_attn["k_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
v = layer.self_attn["v_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
```

**After:**
```python
q_packed, k_packed, v_packed = layer.self_attn["qkv_fused"](x)
q = q_packed.view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
...
k = k_packed.view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
v = v_packed.view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
```

Subtle point: the *shared-KV branch* in every `_run_layer_*` never needs
`k_packed`/`v_packed` (it reads from `kv_store_13/14`). In that branch we
must avoid the wasted k/v compute. There are two ways to handle it:

- **Approach I (cheapest, recommended):** call
  `q_only, _, _ = layer.self_attn["qkv_fused"](x)` is NOT a free optimization
  because the fused Conv2d still writes all `q_dim + 2*kv_dim` channels.
  Instead, for shared layers compute Q from a dedicated split `q_proj`
  that is **preserved** on those layers. See §6.
- **Approach II:** run the fused Conv2d on every layer, throw away K/V for
  shared layers. Simpler code, wastes `2 * kv_dim * hidden` multiplies per
  shared layer per step. With kv_dim=256 (sliding) or 512 (full),
  hidden=1536, this is ~1.6 MFLOPs per shared layer per token. Across 20
  shared layers, ~32 MFLOPs/token. Negligible on ANE vs. the 3-to-1 launch
  reduction win.

**Recommendation: Approach I.** Only own-KV layers (L0-L14, 15 layers) get
`qkv_fused`; shared layers (L15-L34, 20 layers) keep the split `q_proj`
and drop `k_proj`/`v_proj` entirely (matches rewrite #6 goals).

### 2.1 `conversion/models/gemma4_swa_chunks.py`

Six QKV, four Gate/Up sites. Two helper functions: `_run_layer_swa`
(decode) and `_run_layer_verify` (verification). Both helpers are called
by the chunk classes (SWAChunk1-4, SWAVerifyChunk1-4) and indirectly by
`MergedChunk1/12/34` via imports. Patching the two helpers fixes all
eight + three chunk classes in one shot.

#### 2.1.1 `_run_layer_swa` — Q projection (own-KV branch)

Line 70:
```python
    q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
```
Replace with a dispatch that uses the fused path when the layer owns K/V:
```python
    if is_kv_shared:
        q_packed = layer.self_attn["q_proj"](x)
    else:
        q_packed, k_packed, v_packed = layer.self_attn["qkv_fused"](x)
    q = q_packed.view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
```

#### 2.1.2 `_run_layer_swa` — K/V projections (own-KV branch)

Lines 84-85:
```python
        k = layer.self_attn["k_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = layer.self_attn["v_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
```
Replace with (using `k_packed`/`v_packed` already returned above):
```python
        k = k_packed.view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = v_packed.view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
```

#### 2.1.3 `_run_layer_swa` — Gate/Up

Lines 156-157:
```python
    gate = layer.mlp["gate_proj"](x_mlp)
    up = layer.mlp["up_proj"](x_mlp)
```
Replace with:
```python
    gate, up = layer.mlp["gate_up_fused"](x_mlp)
```

#### 2.1.4 `_run_layer_verify` — Q projection

Line 495:
```python
    q = layer.self_attn["q_proj"](x)
```
Replace with:
```python
    if is_kv_shared:
        q = layer.self_attn["q_proj"](x)
    else:
        q, k_raw_v, v_raw_v = layer.self_attn["qkv_fused"](x)
```

#### 2.1.5 `_run_layer_verify` — K/V projections

Lines 516-519:
```python
        k = layer.self_attn["k_proj"](x)
        k = k.view(1, num_kv_heads, hd, seq_len).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = layer.self_attn["v_proj"](x)
        v = v.view(1, num_kv_heads, hd, seq_len).permute(0, 1, 3, 2).to(MODEL_DTYPE)
```
Replace with (consume `k_raw_v`/`v_raw_v` from §2.1.4):
```python
        k = k_raw_v.view(1, num_kv_heads, hd, seq_len).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = v_raw_v.view(1, num_kv_heads, hd, seq_len).permute(0, 1, 3, 2).to(MODEL_DTYPE)
```

#### 2.1.6 `_run_layer_verify` — Gate/Up

Lines 610-611 (same pattern as 2.1.3):
```python
    gate, up = layer.mlp["gate_up_fused"](x_mlp)
```

### 2.2 `conversion/models/gemma4_prefill_chunks.py`

One helper: `_run_layer_prefill`. Lines 63, 84-85, 147-148.

**Line 63 (Q):**
```python
    q_raw = layer.self_attn["q_proj"](x)
```
Replace with:
```python
    if is_kv_shared:
        q_raw = layer.self_attn["q_proj"](x)
    else:
        q_raw, k_raw, v_raw = layer.self_attn["qkv_fused"](x)
```

**Lines 84-85 (K/V):**
```python
        k_raw = layer.self_attn["k_proj"](x)
        v_raw = layer.self_attn["v_proj"](x)
```
Delete both lines — `k_raw`, `v_raw` are already bound from the fused call.

**Lines 147-148 (Gate/Up):**
```python
    gate = layer.mlp["gate_proj"](x_mlp)
    up = layer.mlp["up_proj"](x_mlp)
```
Replace with:
```python
    gate, up = layer.mlp["gate_up_fused"](x_mlp)
```

### 2.3 `conversion/models/gemma4_swa_flash.py`

Lines 108, 121-122, 189-190. Same canonical pattern as §2.1.1-2.1.3. Wrap
the Q line in `if is_kv_shared` vs `qkv_fused`; replace K/V lines with the
unpacked `k_packed`/`v_packed`; collapse gate/up into one `gate_up_fused`
call.

### 2.4 `conversion/models/gemma4_swa_wfa.py`

Lines 64, 78-79, 146-147. Identical pattern to §2.3.

### 2.5 `conversion/models/gemma4_swa_cascading.py`

Lines 58, 70-71, 125-126. Identical pattern to §2.3.

### 2.6 `conversion/models/gemma4_stateless_chunks.py`

Lines 42, 52-53, 107-108. Identical pattern to §2.3.

### 2.7 `conversion/models/gemma4_lite_chunks.py`

Lines 113, 126-127, 184-185. Note: this file nests the loop inside the
`forward` (per-layer loop in-class rather than helper function). Patch the
same lines in place.

### 2.8 `conversion/models/gemma4_lite_wrapper.py`

Lines 116, 130-131, 192-193. Note line 192-193 use `x` directly instead of
`x_mlp` — preserve that:
```python
    gate, up = layer.mlp["gate_up_fused"](x)
```

### 2.9 `conversion/models/gemma4_decoder.py` (SKIP)

Lines 154-156, 190-194, 255-257 reference `self.q_proj = nn.Linear(...)`
— this is a legacy reference module not used by the shipping chunk
pipeline (the shipping graph is built from `Gemma4Model` via
`gemma4_swa_chunks.py`, not `gemma4_decoder.py`). Leaving it untouched
keeps the PR scoped. If/when `gemma4_decoder.py` is retired, it can be
removed separately.

### 2.10 `conversion/models/gemma4_swa_merged1.py` and `merged2.py`

No changes needed. Both delegate to `_run_layer_swa` imported from
`gemma4_swa_chunks.py` (see merged1.py:33, merged2.py:41). Patching
§2.1.1-2.1.3 transparently fixes `MergedChunk1`, `MergedChunk12`, and
`MergedChunk34`.

---

## 3. Weight loading plan

The HF `.safetensors` ships six separate parameters per own-KV layer:
`q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `gate_proj.weight`,
`up_proj.weight`, plus `o_proj.weight`. The current loader
(`conversion/models/gemma4.py:285-354`, `_map_weight_name`) maps each to
the corresponding split `Conv2d`. Shared layers (L15-34) have no
`k_proj`/`v_proj` in the HF file, so those tensors in our model currently
stay at their init-time values — never loaded, never used (the chunk
code gates on `is_kv_shared`).

### Option A — Post-load in-place fuse (RECOMMENDED)

**Mechanism.** Keep `Gemma4DecoderLayer.__init__` unchanged. After
`load_weights()` finishes populating the split Conv2d modules, iterate
over L0-L14 (own-KV layers only) and call `fuse_layer_projections(layer)`
(already defined, `gemma4_fused_modules.py:209-235`). This concatenates
the loaded q/k/v weights into a new `qkv_fused` Conv2d and gate/up into
`gate_up_fused`. Optionally delete the split Conv2d attributes after
fusing to recover memory; the chunk code never reads them for own-KV
layers post-patch.

For L15-L34 (shared), call a variant that only fuses gate/up (no QKV fuse
— they don't have k_proj/v_proj anyway and the q_proj stays split).

**Pros:**
- Minimal code change. `load_weights()` already works; we add ~15 LOC.
- No change to `_map_weight_name` needed.
- Shape mismatches surface immediately at fuse time, not deep in the
  conversion pipeline.

**Cons:**
- Peak memory during fuse: briefly holds both split and fused tensors
  (~2x the projection weight footprint for own-KV layers).
  Quantitative upper bound: 15 layers × (q_dim + 2*kv_dim + 2*inter) *
  hidden * 2 bytes (fp16). For sliding: q_dim=2048, kv_dim=256, so QKV
  fused = 2560 channels × 1536 = ~7.5 MB; Gate/Up fused = 2*6144*1536 =
  ~36 MB. × 15 layers = ~650 MB transient. On 32 GB Mac dev boxes this is
  fine; on a 16 GB M2 Air it is tight but workable (one layer at a time
  is ~45 MB transient with free-after-fuse).

**Patch sketch — add to `conversion/models/gemma4.py` inside
`Gemma4Model.load_weights()` after line 283 (`print(f"Loaded {loaded}
weight tensors")`):**
```python
    # Fuse split QKV and Gate/Up projections for ANE dispatch efficiency
    # (see docs/GEMMA4_ANE_REWRITES.md rewrites #1 and #2).
    if getattr(self.config, "use_fused_projections", True):
        from .gemma4_fused_modules import FusedQKV, FusedGateUp
        fused = 0
        for i, layer in enumerate(self.layers):
            # Gate/Up: every layer has both (even KV-shared layers retain
            # their own MLP — rewrite #6 is only about K/V sharing).
            gate = layer.mlp["gate_proj"]
            up = layer.mlp["up_proj"]
            layer.mlp["gate_up_fused"] = FusedGateUp.from_split(gate, up)
            del layer.mlp["gate_proj"]
            del layer.mlp["up_proj"]

            # QKV: only for own-KV layers (L0-L14). Shared layers keep
            # q_proj split and have no k_proj/v_proj to fuse.
            if not self.config.is_kv_shared(i):
                q = layer.self_attn["q_proj"]
                k = layer.self_attn["k_proj"]
                v = layer.self_attn["v_proj"]
                layer.self_attn["qkv_fused"] = FusedQKV.from_split(q, k, v)
                del layer.self_attn["q_proj"]
                del layer.self_attn["k_proj"]
                del layer.self_attn["v_proj"]
            fused += 1
        print(f"Fused projections on {fused} layers (rewrites #1/#2)")
```

### Option B — Build fused modules in `__init__`, stage HF weights

**Mechanism.** Add `use_fused_projections: bool = True` to
`Gemma4Config`. In `Gemma4DecoderLayer.__init__`, when true, construct
`FusedQKV`/`FusedGateUp` instead of split Conv2d. Extend `_map_weight_name`
to return a sentinel like `("qkv_fused", slice)` and teach
`load_weights()` to stage q/k/v tensors into a per-layer dict, then
concatenate and copy into `qkv_fused.fused.weight.data` once all three
arrive.

**Pros:**
- Lower peak memory (never materializes the split Conv2d).
- `__init__` structure matches final export structure — closer to
  LiteRT-style model definition.

**Cons:**
- ~60-80 LOC of state-machine code in the loader (ordering across
  safetensors shards is not guaranteed; must buffer partial triples).
- `_map_weight_name` becomes stateful; tests get harder.
- Structural change in `Gemma4DecoderLayer` means any existing code that
  reads `layer.self_attn["q_proj"]` elsewhere (scripts, notebooks, tests)
  silently breaks — we'd need a compat shim.

**Recommendation: Option A.** The ~650 MB transient is acceptable on our
conversion boxes (documented to be 32+ GB); the loader simplicity and
debuggability win outweighs the memory saving. If we ever need to convert
on a 16 GB box we can patch Option A to fuse per-layer immediately after
loading that layer's six weights (nested in `_map_weight_name`'s layer
loop), which gives us Option B's memory footprint without the state
machine.

---

## 4. Validation / parity test plan

Order matters — run each step in sequence and only advance on green.

### Step 4.1 Unit-level (pure PyTorch, single layer)

Add `conversion/test_fused_layer_parity.py` (new ~80 LOC):
```python
# Build one Gemma4DecoderLayer.
# Create random fp32 input x of shape (1, 1536, 1, 1).
# Run split forward: q=layer.self_attn["q_proj"](x), etc.
# Call fuse_layer_projections(layer).
# Run fused forward: q2, k2, v2 = layer.self_attn["qkv_fused"](x).
# Assert: torch.allclose(q, q2, atol=1e-5) and same for k, v.
# Also: cosine(q.flatten(), q2.flatten()) >= 0.99999.
# Same for gate_up.
```
Expected: bitwise-equal (weight concat + same kernel), cosine 1.0.

### Step 4.2 Single-chunk re-export with fusion enabled

Re-convert one chunk (recommendation: **SWAChunk1 first** — smallest,
all own-KV layers). Steps:

1. Run `conversion/build_verify_chunks.py --chunk 1 --fused` (add a
   `--fused` flag that threads through to
   `config.use_fused_projections=True`).
2. Produce `swa_chunk1_fused.mlpackage`.
3. Run `conversion/test_merged_parity.py` with the fused chunk1 + stock
   chunk2/3/4. All four values in the reference assertion (token_id,
   normed cosine, logits cosine, final hidden cosine) must match at
   ≥ 0.9999.
4. If green: re-convert chunks 2, 3, 4 individually, retest after each.

### Step 4.3 Full-pipeline + on-device

1. Re-convert all four chunks with `--fused`.
2. Run `conversion/test_merged_parity.py` with the full fused bundle.
3. Run `conversion/test_merged_parity.py --mode one` (1-chunk variant).
4. Ship to iPhone 17 Pro, decode 512 tokens, compare tok/s against
   `docs/BASELINE_SPEED_AUDIT.md` baseline. Expected lift: +13-20% decode
   per item D-1 (item #4 in rollout order, line 131 of
   `IMPLEMENTATION_LOG_2026_04_15.md`).
5. Sample decode on 10 held-out prompts, diff first 128 generated tokens
   against baseline. **All must match exactly** (argmax determinism).

### Step 4.4 Palettization parity (separate follow-up)

Once parity passes in fp16, run the INT4 palettization pipeline:
`OpPalettizerConfig(nbits=4, granularity='per_grouped_channel',
group_size=32)`. Decode 128 tokens and compare PPL against the
pre-fusion palettized build. Rewrite #7 (RMSNorm absorb) is explicitly
deferred *until* this step's numbers are green — see line 75-82 of the
implementation log.

---

## 5. Gotchas

### 5.1 GQA asymmetry

Verified: `num_attention_heads=8`, `num_key_value_heads=1` (not 10/1 as
the brief said). For sliding layers (head_dim=256): q_dim = 2048,
kv_dim = 256, so fused out_channels = 2048 + 2*256 = 2560. For full
layers (head_dim=512): q_dim = 4096, kv_dim = 512, fused = 5120.
`FusedQKV.__init__` and `from_split` both derive these from the split
Conv2d shapes, so no config code path needs to special-case.

### 5.2 Dual head_dim (sliding=256 vs global=512)

`Gemma4Model.__init__` (gemma4.py:132-145) already creates each
`Gemma4DecoderLayer` with `head_dim=config.get_head_dim(i)` — the split
Conv2d shapes are already per-layer-correct. `FusedQKV.from_split`
inherits those shapes. No additional code.

### 5.3 KV-shared layers (L15-L34) — the tricky part

HF safetensors contains no `k_proj`/`v_proj` for these 20 layers, but
`Gemma4DecoderLayer.__init__` currently builds those Conv2d anyway and
leaves them at init values. The chunk code guards with `if not
is_kv_shared`, so they're harmless at runtime — but they **do** bloat
the exported `.mlpackage` (the converter traces shape even for unused
modules, and coremltools emits them as dead weight).

**Post-fuse state for shared layers:**
- `layer.self_attn["q_proj"]` — KEEP (shared layers still need Q).
- `layer.self_attn["k_proj"]`, `["v_proj"]` — DELETE (they hold junk
  anyway; deleting drops ~180 MB of dead weight from chunks 3/4 per
  rewrite #6 §Estimated gain, line 640-646 of GEMMA4_ANE_REWRITES.md).
- `layer.self_attn["qkv_fused"]` — DO NOT CREATE. Shared layers do not
  produce K/V; creating a fused op would be dead weight × 2.
- `layer.mlp["gate_up_fused"]` — CREATE. MLP is not shared; gate/up fuse
  applies uniformly across all 35 layers.

**Dispatch pattern in `_run_layer_swa` (and siblings):** the `if
is_kv_shared` branch calls `q_proj`; the else branch calls `qkv_fused`
and unpacks. The control flow is already present at the outer
`is_kv_shared` switch — we're just moving the projection call into each
arm. This is the shape of §2.1.1 above.

**`fuse_layer_projections` helper needs updating:** the current version
(gemma4_fused_modules.py:209-235) indiscriminately reads `q_proj`,
`k_proj`, `v_proj` — which would fail on L15-34 once we delete their
k_proj/v_proj. Either:
- (a) Update `fuse_layer_projections` to accept an `is_kv_shared` flag
  and skip QKV fuse if true, OR
- (b) Inline the per-layer logic in `load_weights()` (Option A patch
  above already does this — preferred).

### 5.4 Palettization

`OpPalettizerConfig(nbits=4, granularity='per_grouped_channel',
group_size=32)` groups along the **out_channels** dim of a Conv2d. When
we concatenate [Q | K | V] along out_channels:
- sliding layer: [2048 | 256 | 256] channels.
- global layer: [4096 | 512 | 512] channels.

group_size=32 divides all of these evenly (2048/32=64 groups in the
Q-region, 256/32=8 in each of K and V for sliding). Crucially, no group
straddles a Q/K/V boundary (because every sub-block size is a multiple
of 32), so the palette codebook for a group only sees values from one
projection type — preserving the local weight-distribution statistics
the current split-palette code sees.

Gate/Up fuse: 2*6144=12288 channels, 12288/32=384 groups, with no
boundary straddling. Same reasoning.

**Risk level:** low. Validate by running §4.4 on the fused build and
comparing PPL against the pre-fusion palettized baseline (expected Δ ≤
+0.02 PPL, i.e. noise).

### 5.5 Unused `o_proj` (shared-layer cleanup hint)

Shared layers do need `o_proj` (it's the attn output projection, not
K/V related). Do NOT delete it. The above rules apply only to
k_proj/v_proj.

### 5.6 Chunk boundaries and `fuse_layer_projections` timing

`build_verify_chunks.py` instantiates each SWAChunk* after loading the
`Gemma4Model`. If the model was fused in `load_weights()` (Option A),
each SWAChunk's `self.layers = nn.ModuleList([model.layers[i] ...])`
(gemma4_swa_chunks.py:212) picks up the already-fused modules. No
change to chunk constructors needed.

### 5.7 Verify-mode `num_kv_heads=1` with seq_len > 1

`_run_layer_verify` handles `seq_len` tokens at once. The fused Q output
is `(1, q_dim, 1, seq_len)` and reshapes to `(1, num_heads, hd,
seq_len)`. K/V are `(1, kv_dim, 1, seq_len)` reshaping to `(1,
num_kv_heads, hd, seq_len)`. Slicing boundaries in `FusedQKV.forward`
are along the channel dim only — unaffected by `seq_len`. No change to
the module.

### 5.8 RoPE compatibility

RoPE is applied **after** the projection, per-head, and operates on the
last two dims (`(..., seq_len, head_dim)`). Fusing only changes how we
produce q/k — the post-reshape tensors are bitwise-identical to the
split path. RoPE call sites unchanged.

### 5.9 q_norm / k_norm ordering

q_norm and k_norm are `ANERMSNorm(head_dim)` applied per-head after the
projection + reshape. Fusing does not change the per-head tensors, so
norm call sites are unaffected. Confirmed by inspection: all call sites
retain the same `reshape → norm → reshape` sandwich after the patch.

---

## 6. Shared-layer q_proj handling — detail

The shared-layer branch keeps the plain `q_proj` split. After the
patch, for L15-L34 we have:
- `layer.self_attn["q_proj"]` (Conv2d, 1536 → 2048 for sliding shared
  layers; 1536 → 4096 for full shared layers, i.e. L19/24/29/34).
- `layer.self_attn["o_proj"]` (unchanged).
- `layer.self_attn["q_norm"]` (unchanged; k_norm also unchanged since
  HF ships k_norm weights for shared layers — though they're unused post
  patch. **Drop k_norm weights** optionally for extra ~180 KB.).

This matches `GemmaSharedKVAttention` from rewrite #6
(GEMMA4_ANE_REWRITES.md:558-620). We are effectively landing rewrite #6
*as a side effect* of D-1 Option A's `del` logic.

---

## 7. Land-it-in-one-PR checklist

- [ ] **Branch:** already on `feat/pre-conv-optimizations`. Rebase on
      `main` before starting.
- [ ] **Config flag:** add `use_fused_projections: bool = True` to
      `Gemma4Config.__init__` (gemma4.py:38-82). Default true.
- [ ] **Loader patch:** apply the §3 Option A code block to
      `Gemma4Model.load_weights()` just after line 283 of gemma4.py.
      Exercise with `python -c "from models.gemma4 import Gemma4Model;
      Gemma4Model.from_pretrained(os.environ['GEMMA4_HF_DIR'])"` — no
      import-time errors, load succeeds, new "Fused projections on 35
      layers" print appears.
- [ ] **Patch §2.1** (`gemma4_swa_chunks.py`, 6 QKV + 2 Gate/Up sites).
- [ ] **Patch §2.2** (`gemma4_prefill_chunks.py`, 3+1 sites).
- [ ] **Patch §2.3** (`gemma4_swa_flash.py`).
- [ ] **Patch §2.4** (`gemma4_swa_wfa.py`).
- [ ] **Patch §2.5** (`gemma4_swa_cascading.py`).
- [ ] **Patch §2.6** (`gemma4_stateless_chunks.py`).
- [ ] **Patch §2.7** (`gemma4_lite_chunks.py`).
- [ ] **Patch §2.8** (`gemma4_lite_wrapper.py`).
- [ ] **Add** `conversion/test_fused_layer_parity.py` (§4.1).
- [ ] **Run** `python test_fused_layer_parity.py` — must pass.
- [ ] **Run** `python test_merged_parity.py` (on a dev box with HF
      weights) — must pass with use_fused_projections=True.
- [ ] **Re-convert** SWAChunk1 with `--fused`, run parity. Then chunks
      2, 3, 4.
- [ ] **Re-convert** full bundle; on-device tok/s check.
- [ ] **Update** `docs/IMPLEMENTATION_LOG_2026_04_15.md`: move item D-1
      from "Not applied" to "Applied" with the measured tok/s delta.
- [ ] **Do not** land rewrite #7 (RMSNorm absorb) in the same PR — it is
      gated on D-1 being in and palettization parity being validated.
- [ ] **Do not** commit `.mlpackage` artifacts or build outputs (per
      repo-root CLAUDE.md).
- [ ] **Commit:** single commit per file group (models + loader + tests)
      or a single squash-merge PR. Committer/message must not reference
      Claude (per repo-root CLAUDE.md).

---

## 8. Files touched — absolute paths

- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_swa_chunks.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_prefill_chunks.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_swa_flash.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_swa_wfa.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_swa_cascading.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_stateless_chunks.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_lite_chunks.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_lite_wrapper.py`
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/test_fused_layer_parity.py` (new)

Unchanged (delegators or out-of-scope):
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_swa_merged1.py`
  (delegates to `_run_layer_swa`)
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_swa_merged2.py`
  (delegates)
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/gemma4_decoder.py`
  (legacy `nn.Linear`, not on shipping path)
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/models/qwen2.py`
  (different architecture)
