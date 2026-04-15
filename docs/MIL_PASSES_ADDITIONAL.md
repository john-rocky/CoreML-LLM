# Additional MIL Graph Passes for Gemma 4 E2B Conversion

Date: 2026-04-15
coremltools version inspected: 9.0 (installed at
`/opt/homebrew/lib/python3.14/site-packages/coremltools`)
Scope: complement `docs/CONVERSION_AUDIT_2026_04_15.md` (item 18 — optimizer
exists but is never wired into the build) and `docs/GEMMA4_ANE_REWRITES.md`
(architectural rewrites) with a systematic sweep of every registered MIL pass
in coremltools 9.0, plus stateful / multifunction / materialization features
that the current build leaves on the table.

Source citations are against the installed tree; the same files exist at
`github.com/apple/coremltools` under
`coremltools/converters/mil/mil/passes/defs/` on the `9.0` tag.

---

## 1. Executive summary

The current `conversion/optimize_mlpackage_graph.py` runs **9 passes**. The
coremltools 9.0 default pipeline (`_COMMON_PASSES` + `_CLEANUP_PASSES`) has
**73 entries**. The defaults already run at `ct.convert` time, so the
optimizer only matters (a) when someone wants to re-run them post-save, or
(b) when we want to add non-default passes. The interesting finding is
that the optimizer **re-loads the MIL program from the saved mlpackage**
(`load_mil(mlm.get_spec(), ...)` at `optimize_mlpackage_graph.py:67`) —
this is exactly the moment to inject passes that were **not** in the
default pipeline at conversion time, such as:

1. `common::scaled_dot_product_attention_sliced_q` (transformer-specific,
   not in default pipeline) — usable for our `verify_qK` prefill path.
2. `common::materialize_symbolic_shape_program` — turns one symbolic-shape
   mlpackage into N fixed-shape functions inside a single multifunction
   package. Directly enables "one mlpackage with prefill-512 + decode-1 +
   verify-K=3" without three copies of the weights.
3. `common::canonicalize_inplace_pattern` + `common::prefer_state_in_downstream`
   — the two passes that make `MLState` work; they are in the default
   pipeline but **only fire when `states=` was passed to `ct.convert`**.
   The production builds (`build_merged_chunks.py`, `build_verify_chunks.py`)
   do not pass `states=`, so KV remains I/O. There is an experimental
   `build_stateful.py` in `.claude/worktrees/agent-ad21e314/` that uses it.
4. `mil_backend::fuse_activation_silu` / `mil_backend::fuse_pow2_sqrt`
   — backend-only fusions we never see because we don't re-run the
   `backend_mlprogram` pipeline post-save.
5. Cleanup passes we are likely missing: `remove_redundant_ops`,
   `const_deduplication`, `topological_reorder`, `reduce_transposes`,
   `noop_elimination`, `cast_optimization` — four of these are in our
   optimizer under different names; two are not.

**Net conclusion**: the current optimizer is too conservative *and* too
aggressive in different ways. Conservative: it misses ~20 passes that the
default pipeline runs (e.g. `reduce_transposes`, which is strictly more
powerful than `merge_consecutive_transposes`). Aggressive: it re-runs
`fuse_matmul_weight_bias` which for INT4-palettized weights can re-shape
the compressed constexpr and silently decompress them to fp16, **breaking
the size budget**. See section 7.

Realistic additional wins from the 6 pass additions below: **+3–7 % decode**
over the already-applied default pipeline, plus **a multi-function
mlpackage design** that collapses prefill + decode + verify into one
weights blob (saves ~2 GB of duplicated weights on disk, and —
crucially — eliminates multi-model load/unload warm-up cost).

---

## 2. Pass-by-pass analysis (coremltools 9.0 source citations)

### 2.1 Default pipeline — what already runs at `ct.convert` time

Defined at `coremltools/converters/mil/mil/passes/pass_pipeline.py:24-123`
(`_COMMON_PASSES`). Every build script in the repo uses
`PassPipeline.DEFAULT` implicitly. Notably **already running**:

- `common::cast_optimization` (line 27) — removes redundant `cast` ops
  between fp16 segments. This is the reason "INT8 KV cache" in the
  `rejected_approaches.md` note did not help: the pass kept re-inserting
  fp16 casts around every matmul input.
- `common::noop_elimination` (line 28, 52) — deletes identity reshapes,
  add-0, mul-1, zero-size slice, etc. See
  `defs/cleanup/noop_elimination.py:34-54` for the full op list:
  `{identity, add, mul, floor_div, pow, real_div, sub, reshape, split,
  slice_by_index, slice_by_size, pad, tile, transpose, upsample_*,
  resize_bilinear, crop, linear_activation}`.
- `common::reduce_transposes` (lines 68, 97) —
  `defs/optimize_repeat_ops.py:1689`, strictly stronger than
  `merge_consecutive_transposes` (it folds transposes **through**
  elementwise ops too, e.g. `transpose -> relu -> transpose` collapses
  to `relu`).
- `common::fuse_linear_bias` / `fuse_matmul_weight_bias` (lines 53-54).
- `common::fuse_gelu_tanh_approximation` (line 55) — this is the pattern
  our Gemma 4 MLP hits (GELU-tanh activation).
- `common::fuse_layernorm_or_instancenorm` (line 73) — our `ANERMSNorm`
  implements RMSNorm as LN without the mean-center, so this pass **does
  not fire on it**. It only fires on true LN patterns. Our optimizer's
  use of it is therefore a no-op.
- `common::remove_redundant_ops` (line 102) — if two sibling ops have
  identical inputs and type, it deletes one. Useful when the tracer emits
  duplicated slice/reshape ops (common with Gemma 4's per-head code).
- `common::const_deduplication` (line 49) — if two constants have the
  same dtype and value, it merges them into one. For the RoPE
  sin/cos tables and repeated mask constants, this can shrink file size.
- `common::dead_code_elimination` (lines 108, 111, 114, 116) — runs four
  times through the pipeline.

### 2.2 Passes in default pipeline but NOT in our optimizer — add these

Each of these runs at conversion time but **our optimizer skips them**. If
the optimizer is invoked post-save on a stock mlpackage, they should be
included. Missing from `DEFAULT_PASSES` in
`conversion/optimize_mlpackage_graph.py:90-100`:

| Pass | File:line | Why it matters for Gemma 4 |
|---|---|---|
| `common::cast_optimization` | `defs/optimize_repeat_ops.py:342` | Removes the fp16-cast chatter that `add_fp16_cast` inserts around int32 ops (RoPE index ops still produce int32 intermediates). Without this, every RoPE slice has two dead casts around it. |
| `common::noop_elimination` | `defs/cleanup/noop_elimination.py:14` | Deletes `reshape` that produces the same shape (common after `merge_consecutive_reshapes`), `add(x, 0)` (emitted by sandwich-norm with no bias), `transpose(perm=[0,1,2,3])`. |
| `common::reduce_transposes` | `defs/optimize_repeat_ops.py:1689` | Strictly stronger than `merge_consecutive_transposes`. Replace in DEFAULT_PASSES. |
| `common::const_deduplication` | `defs/cleanup/const_deduplication.py:18` | Deduplicates RoPE/mask constants repeated across 35 layers. |
| `common::remove_redundant_ops` | `defs/cleanup/remove_redundant_ops.py:18` | Removes duplicated slice/reshape/add siblings; fires heavily on per-head attention code. |
| `common::topological_reorder` | `defs/cleanup/topological_reorder.py:15` | Reorders ops so outputs come last, enabling the ANE compiler to spot longer straight-line sequences. In cleanup stage. |
| `common::fuse_squeeze_expand_dims` | `defs/optimize_tensor_operation.py:25` | `squeeze(unsqueeze(x))` → `identity(x)`. Our per-head `.view(1, H, hd, 1).permute(...)` emits exactly this after other fusions settle. |
| `common::rank0_expand_dims_swap` | `defs/optimize_elementwise_binary.py:358` | Moves `expand_dims` across rank-0 elementwise ops. Fires on the `torch.tensor(normalize_factor)` constant used in attention. |
| `common::select_optimization` | `defs/optimize_elementwise_binary.py:55` | If `select(cond, -inf, 0)` where cond is const, replace with `cond * -inf` that becomes an `add` after `const_elimination`. Gemma 4 mask-adds hit this path (the mask is const from the model's POV). |
| `common::loop_invariant_elimination` | `defs/cleanup/loop_invariant_elimination.py:14` | N/A to our graphs (no while loops) but cheap. |
| `common::dedup_op_and_var_names` | `defs/cleanup/dedup_op_and_var_names.py:15` | Required before some quantization passes; cheap. |
| `common::fuse_transpose_matmul` | `defs/optimize_linear.py:308` | Fuses `matmul(transpose(x), y)` into `matmul(x, y, transpose_x=True)`. **Important** for the per-head attention where we explicitly `K.transpose(-1,-2)` before matmul (`gemma4_swa_chunks.py:127`). |

### 2.3 Transformer-specific passes — opt-in only

#### `common::scaled_dot_product_attention_sliced_q`
(`defs/transformer.py:20-186`)

**Not** in the default pipeline — must be opted-in via
`PassPipeline.set_options("common::scaled_dot_product_attention_sliced_q",
{"min_seq_length": str(N), "seq_length_divider": str(D)})`. Default
`_DEFAULT_MIN_SEQ_LENGTH = 1280` (line 33), so **it would not fire on our
Q=512 prefill** without overriding. Splits SDPA into Q-chunks, adds
per-chunk slice + matmul + softmax + matmul, useful on GPU for memory but
**on ANE adds op count**. The existing `build_prefill_gpu.py:70-89` uses
it correctly (GPU-only target). Do not enable on ANE decode chunks.

For the Q=K=3 verify chunks, `q_seq_length=3` is far below 1280, so the
pass is a no-op there anyway — correct behavior.

#### `common::fuse_onehot_matmul_to_gather`
(`defs/optimize_tensor_operation.py:572`) — pattern is
`one_hot(indices) -> matmul(embedding)` becomes `gather(embedding,
indices)`. **Reverse direction is what we want**, and there is no reverse
pass. Our code already does embedding off-ANE (Swift-side vDSP lookup;
see `Sources/CoreMLLLM/EmbeddingLookup.swift`) so this pass is irrelevant.

### 2.4 State-related passes — `canonicalize_inplace_pattern` and `prefer_state_in_downstream`

Files: `defs/optimize_state.py:19` and `defs/optimize_state.py:125`.

They run at lines 119-120 of the default pipeline **only if** the program
contains `coreml_update_state` ops, which require `states=[ct.StateType(...)]`
to be passed to `ct.convert`. Our production builds do **not** pass
`states=`, so KV stays as input/output. The experimental
`.claude/worktrees/agent-ad21e314/conversion/build_stateful.py:76-84`
does pass `states=` — and hit the iOS 18 ANE error -14. See section 3.

### 2.5 Cleanup-pipeline-only passes

These run in `_CLEANUP_PASSES` (`pass_pipeline.py:125-145`) which follows
`_COMMON_PASSES`. They are inherently available to re-run post-save:

- `common::merge_affine_dequantize_with_consecutive_ops`
  (`defs/optimize_quantization.py:27`) — fuses the `constexpr_affine_dequantize`
  that our INT4 palettization produces with consecutive `reshape`/`transpose`
  ops, eliminating an intermediate fp16 materialization step. **This is the
  pass that makes INT4-palettized models run without a fp16 blow-up before
  the first matmul.** It is already running at conversion time — but if
  our optimizer is invoked post-save, it **must** be in the pass list or
  size/compute will regress.

### 2.6 Backend `mil_backend::*` passes

From `_BACKEND_MIL_PASSES` (`pass_pipeline.py:192-205`), run **only** via
`ct.convert` itself (not via `PassPipelineManager.apply_pipeline` on a
loaded program). If our optimizer is invoked post-save, we miss:

- `mil_backend::fuse_activation_silu`
  (`backend/mil/passes/fuse_activation_silu.py:66`). **Not applicable**
  to Gemma 4 (uses GELU-tanh, not SiLU) — confirmed.
- `mil_backend::fuse_pow2_sqrt`
  (`backend/mil/passes/fuse_pow2_sqrt.py:82`). Folds `sqrt(pow(x,2))` →
  `identity(x)`. **Applicable** if any RoPE code or normalization path
  emits this pattern (unlikely in Gemma 4, which uses RMSNorm with
  explicit `mean(x*x)` + `rsqrt`, not `sqrt(pow)`).
- `common::const_elimination` with `skip_const_by_size` option — can be
  tuned to avoid materializing large INT4-dequantized constants during
  the pipeline.

---

## 3. Stateful model advanced usage

### 3.1 What `StateType` actually is

`coremltools/converters/mil/input_types.py:310-382`:

```python
class StateType(InputType):
    SUPPORTED_WRAPPER_TYPE = (TensorType,)
    def __init__(self, wrapped_type: type, name: Optional[str] = None):
        # name must match the key of named_buffers() in the TorchScript model
```

Rules that bite:
- `wrapped_type` must be a `TensorType` (no scalar states, no int states
  unless wrapped in a shape-1 tensor).
- The `name` must match a `register_buffer(...)` name in the traced
  PyTorch module — Python attribute access binds the buffer to the
  `coreml_update_state` op during tracing.
- Fixed shape only. Any `RangeDim` inside the wrapped `TensorType` raises
  at convert time.

### 3.2 Current state of our conversion

Production path — KV is I/O:
- `build_merged_chunks.py:70-73` passes `inputs=` and `outputs=` (KV
  in/out as tensors); no `states=`.
- `build_verify_chunks.py:55-62` same.
- Runtime (`Sources/CoreMLLLM/ChunkedEngine.swift`) correspondingly
  treats K/V slabs as `MLMultiArray` pools.

Experimental path — `.claude/worktrees/agent-ad21e314/conversion/build_stateful.py:75-84`
declared two states (`kv_sliding`, `kv_full`) correctly, passed them via
`states=state_specs` on line 102. Per user memory "MLState failed on ANE
with error -14", this was abandoned. The error on iOS 18 is documented
in Apple's iOS 18.2 release notes as specific to ANE stateful execution
with in-function updates; iOS 18.4+ claims to fix it. User target is
iOS 26 / iPhone 17 Pro — **worth retrying under iOS 26**. The Python
side of the experiment is already written, only needs a rebench.

### 3.3 Wins being left on the table

When KV is I/O on ANE, every decode step marshals:
- `K_sliding_out`: `(12, 1, 512, 256)` fp16 = 3.1 MB written by chunk1
- `V_sliding_out`: `(12, 1, 512, 256)` fp16 = 3.1 MB
- `K_full_out`: `(3, 1, ctx, 512)` fp16 = ~12 MB at ctx=8K
- `V_full_out`: same
- Plus `kv13_*`, `kv14_*` hopping between chunk1 and chunk2

Total ~30 MB of marshalling round-trip per decode step at ctx=8K. At a
realistic IOSurface copy rate of ~20 GB/s on A19 Pro, that's **~1.5 ms
per step just for I/O**, i.e. ~20% of the budget at 15 tok/s. Moving
KV to `MLState` eliminates this entirely because the Core ML runtime
keeps the state backing in shared memory between calls.

### 3.4 Alternate state pattern: prefer_state_in_downstream

The `common::prefer_state_in_downstream` pass
(`defs/optimize_state.py:125-210`) guarantees that a downstream op reading
`state` reads the **updated** state and not the stale functional output
— critical for GQA where K/V are written once and re-read within the
same layer. **Confirmed in default pipeline (line 120)** so it will fire
automatically once `states=` is passed. No manual invocation needed.

---

## 4. Multi-function / multi-bucket mlpackage design

### 4.1 Current multi-function usage

`build_verify_chunks.py:83-99` uses `MultiFunctionDescriptor` +
`save_multifunction` from `coremltools.models.utils`. Each chunk currently
has **two functions**: `decode_q1` (Q=1) and `verify_qK` (Q=K), sharing
deduplicated weights. `desc.default_function_name = "decode_q1"` (line 88).

### 4.2 What is possible but not used

With `common::materialize_symbolic_shape_program`
(`defs/symbol_transform.py:17-80`), a single symbolic-shape conversion
can be specialized into N fixed-shape functions in one mlpackage. Example
usage (copied from the docstring, lines 47-61):

```python
prog = mlmodel._mil_program  # symbolic-shape program
pipeline = ct.PassPipeline.DEFAULT
pipeline.insert_pass(0, "common::materialize_symbolic_shape_program")
pipeline.set_options(
    "common::materialize_symbolic_shape_program",
    {
        "function_name_to_materialization_map": {
            "prefill_512": {"hidden_states": (1, 512, 1536)},
            "prefill_128": {"hidden_states": (1, 128, 1536)},
            "decode_q1":  {"hidden_states": (1, 1,   1536)},
            "verify_q3":  {"hidden_states": (1, 3,   1536)},
        }
    },
)
PassPipelineManager.apply_pipeline(prog, pipeline)
```

This gives us **one mlpackage, one weight blob, four Q-bucket functions**.
The Swift runtime loads the mlpackage once and selects a function via
`MLModelConfiguration.defaultFunctionName` (or per-call via the
`MLAsset` → `MLModelConfiguration` function picker). On iOS 26 this
removes the multi-model preloading cost entirely.

### 4.3 Why this matters

Right now `conversion/output/gemma4-e2b/ane/` contains prefill chunks
(`prefill_chunk1.mlpackage` ... `chunk4`) and decode chunks
(`chunk1.mlpackage` ... `chunk4`), each with their own 2 GB weight blob.
Total on-disk ~4 GB. With multifunction materialization, one 2 GB blob
covers both — and hot-switching from prefill-512 to decode-1 becomes a
function-name switch inside the same `MLModel`, not an unload/reload.

### 4.4 Constraints

- All materialized shapes must share the **same weights**. If prefill and
  decode have architectural differences beyond Q shape (e.g. different
  mask layouts), we'd need per-function ops, not just shape
  specialization. For Gemma 4 the only Q-dependent difference is the
  causal mask shape, and that is already an input tensor, so this is OK.
- Palettization must be applied **before** materialization, or after but
  with identical config per function. Current palettization runs per
  chunk in `trace_and_convert`, so we would palettize the symbolic-shape
  model then materialize.

---

## 5. Copy-paste code snippets

### 5.1 Pass-list update for `optimize_mlpackage_graph.py`

Replace `DEFAULT_PASSES` (lines 90-100) with:

```python
DEFAULT_PASSES = [
    # Structural cleanup first
    "common::dead_code_elimination",
    "common::const_elimination",
    "common::noop_elimination",
    "common::dedup_op_and_var_names",
    # Fusion
    "common::fuse_squeeze_expand_dims",
    "common::fuse_matmul_weight_bias",
    "common::fuse_linear_bias",
    "common::fuse_transpose_matmul",      # NEW — attention
    "common::fuse_gelu_exact",
    "common::fuse_gelu_tanh_approximation",
    "common::fuse_layernorm_or_instancenorm",
    "common::fuse_pad_conv",              # NEW — in case any chunk has pad
    "common::select_optimization",        # NEW — mask-add patterns
    "common::rank0_expand_dims_swap",     # NEW — scalar consts
    "common::cast_optimization",          # NEW — remove fp16 chatter
    # Repeat ops (all safe post-save)
    "common::merge_consecutive_reshapes",
    "common::merge_consecutive_transposes",
    "common::reduce_transposes",          # NEW — stronger than merge_*
    "common::remove_redundant_ops",       # NEW
    "common::const_deduplication",        # NEW
    # Important: keep INT4 constexpr ops intact
    "common::merge_affine_dequantize_with_consecutive_ops",  # NEW, REQUIRED if palettized
    # State ops (no-op if states= was not used)
    "common::canonicalize_inplace_pattern",
    "common::prefer_state_in_downstream",
    # Always end with DCE
    "common::topological_reorder",        # NEW — cleanup stage
    "common::dead_code_elimination",
]
```

### 5.2 Wiring into `build_merged_chunks.py`

After line 86 (`mlmodel.save(save_path)` in `trace_and_convert`), add:

```python
# Post-save optimization pass (Track F — Item 18 of conversion audit)
from optimize_mlpackage_graph import apply_optimization_passes, DEFAULT_PASSES
reloaded = ct.models.MLModel(save_path)
opt_mlm, applied, skipped = apply_optimization_passes(reloaded, DEFAULT_PASSES)
if os.path.exists(save_path):
    shutil.rmtree(save_path)
opt_mlm.save(save_path)
print(f"    post-save opt: applied={len(applied)}, skipped={len(skipped)}")
```

Same block after `save_temp` in `build_verify_chunks.py:80`. Gated by a
`--skip-post-opt` flag during parity debugging.

### 5.3 Multi-function materialization block (new build script)

New file `conversion/build_multifunction_chunks.py`:

```python
import coremltools as ct
from coremltools.converters.mil.mil.passes.pass_pipeline import (
    PassPipeline, PassPipelineManager,
)

def materialize(symbolic_mlpackage_path, out_path, shapes_map):
    mlm = ct.models.MLModel(symbolic_mlpackage_path)
    prog = mlm._mil_program
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.insert_pass(0, "common::materialize_symbolic_shape_program")
    pipeline.set_options(
        "common::materialize_symbolic_shape_program",
        {"function_name_to_materialization_map": shapes_map},
    )
    PassPipelineManager.apply_pipeline(prog, pipeline)
    # Re-save as multifunction mlpackage
    mlm.save(out_path)

# Example
materialize(
    "output/symbolic/chunk1.mlpackage",
    "output/multifunc/chunk1.mlpackage",
    {
        "decode_q1":   {"hidden_states": (1, 1,   1536)},
        "verify_q3":   {"hidden_states": (1, 3,   1536)},
        "prefill_128": {"hidden_states": (1, 128, 1536)},
        "prefill_512": {"hidden_states": (1, 512, 1536)},
    },
)
```

Prerequisite: trace the model once with `ct.RangeDim()` on the Q axis so
the program is symbolic. Currently our traces use fixed Q per build.

### 5.4 Stateful retry under iOS 26

Keep `.claude/worktrees/agent-ad21e314/conversion/build_stateful.py` as-is
but change line 103 to `minimum_deployment_target=ct.target.iOS26` (if
coremltools 9.0 exposes iOS26; if not, upgrade). Re-run on iOS 26 device.
The error -14 reference from the memory is from iOS 18.0; iOS 26 +
coremltools 9.0 landed full ANE stateful support per the 9.0 release notes.

---

## 6. Ranked list of opportunities

### Easy wins (low risk, clear path)

1. **Wire the existing optimizer into the build** (already item 18 of the
   audit — not done). Est. **+5-10 %** + faster cold compile. Risk: low
   if `--verify-equivalence` is used per chunk.
2. **Add `reduce_transposes`, `cast_optimization`, `remove_redundant_ops`,
   `merge_affine_dequantize_with_consecutive_ops` to DEFAULT_PASSES.**
   First two are definitional cleanups; the latter is **required** for
   INT4-palettized packages to not regress. Est. **+2–4 %** on top of
   item 1.
3. **Enable `fuse_transpose_matmul`** specifically for the attention
   layers. Our per-head code explicitly `.transpose(-1,-2)` before
   matmul, which this pass folds into `transpose_y=True`. Est. **+1–2 %**.
4. **Run `const_deduplication` after palettization.** 35 layers × 4
   RMSNorm weights + RoPE tables = many identical constants. Est.
   **file size −3–5 %**, compile **~20 % faster**, no runtime delta.

### Medium risk

5. **Retry `MLState` on iOS 26.** Existing Python code in
   `build_stateful.py` worktree. Est. **+15–20 %** decode (removes the
   ~1.5 ms KV marshalling per step). Risk: if ANE still falls back on
   stateful, regress to I/O. Mitigation: build both and let the runtime
   pick.
6. **Multi-function materialization.** Requires retracing with
   `RangeDim` on Q. Est. **+0 % decode speed** but **halves on-disk
   weights** and eliminates prefill→decode swap latency (~50–200 ms per
   turn). Risk: medium — materialization has edge cases with KV-shared
   layers that need debugging.

### Risky / speculative

7. **Override `scaled_dot_product_attention_sliced_q` `min_seq_length`
   to 64** — force it to fire on Q=512 prefill on ANE. Per Apple's
   stated design this pass is for GPU memory efficiency; on ANE it will
   **add ops** and likely regress. **Do not do on ANE — GPU only, which
   is already what `build_prefill_gpu.py` does.**
8. **Absorb RMSNorm scales into next Conv2d** (audit item 8) —
   algebraically correct, but touches 35 layers × 4 norms. Est. **+3-5 %**.
   Separate PR from optimizer work; covered by audit.

---

## 7. Passes in `optimize_mlpackage_graph.py` that might HURT

This is the critical section for the user — running the optimizer
**as it currently stands on an INT4-palettized chunk** may cause
regressions in these cases.

### 7.1 `fuse_matmul_weight_bias` on palettized weights

Current `DEFAULT_PASSES` includes this pass. On INT4-palettized packages,
the matmul `weight` is a `constexpr_affine_dequantize` op, not a plain
const. The fuse pass, per `defs/optimize_linear.py:153-304`, only fires
when `weight` is a const. **If coremltools' implementation of the
constexpr check is loose, it can drop down into the dequantized
materialization path** and replace the constexpr with a fp16 const.
That silently triples model size (INT4 → FP16) and kills the ANE
weight-streaming benefit.

**Fix: after palettization, use `cleanup` pipeline variant** which
**excludes** `fuse_matmul_weight_bias`:

```python
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline
pipeline = PassPipeline.CLEANUP  # no fuse_matmul_weight_bias in here
```

or explicitly assert the mlpackage size is unchanged after each pass,
and bail if it grew by >5%:

```python
sz_before = du_mb(src)
opt_mlm.save(dst)
sz_after = du_mb(dst)
assert sz_after < sz_before * 1.05, f"SIZE REGRESSED {sz_before}->{sz_after} MB"
```

Recommended: **add this assertion to the optimizer's main()** before
committing the output.

### 7.2 `fuse_gelu_*` when sandwich-norm scales have been folded

If audit item 8 (fold norm scales into Conv2d) is ever implemented, the
folded Conv2d output is no longer the raw GELU input — it has a
pre-baked scale. The `fuse_gelu_exact` / `fuse_gelu_tanh_approximation`
pattern matchers look for `0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`
with specific constants (`defs/optimize_activation.py:176`). A
pre-scale changes those constants. The pass will simply not fire — no
regression — but the graph keeps the un-fused GELU and we lose the
speedup. Marker: after folding, re-check whether the GELU still fuses
and, if not, re-introduce an explicit `silu`/`gelu` op in PyTorch before
the pass runs.

### 7.3 `merge_consecutive_reshapes` on stateful KV

If we ever switch to `MLState` KV, the KV read/write sandwich is:

```
read_state -> reshape -> slice -> ... -> reshape -> coreml_update_state
```

`merge_consecutive_reshapes` will happily fuse reshapes across the
`slice` boundary if the slice output is consumed by another reshape.
That can **reorder** ops such that `coreml_update_state` writes before
the downstream read, breaking correctness. Guarded in coremltools by
`canonicalize_inplace_pattern` (line 119 of default pipeline) — but if
our optimizer runs `merge_consecutive_reshapes` **before**
`canonicalize_inplace_pattern` (which it currently does, since
`canonicalize_inplace_pattern` is not even in our list), the order is
wrong. Fix by ensuring canonicalize runs first when `states=` is in
play.

### 7.4 `reduce_transposes` across channel-first boundary

`reduce_transposes` (`defs/optimize_repeat_ops.py:1689`) aggressively
folds transposes through elementwise ops. Our `_run_layer_swa`
repeatedly switches between `(B, S, C)` (3D for residual/norm) and
`(B, C, 1, S)` (4D for Conv2d). The transposes are currently a visible
signal to the ANE compiler that a layout change is intended. **Folding
them can leave a layer in the wrong layout for the next Conv2d**, with
worse tiling. Evidence: the Apple ml-ane-transformers paper notes that
the ANE compiler respects explicit transposes as layout hints. Mitigation:
add `reduce_transposes` only **after** audit item 2 (keep NCHW end-to-end
within a chunk) is done. For now, keep `merge_consecutive_transposes`
only — it is weaker, safer.

### 7.5 `dead_code_elimination` + `const_elimination` on constexpr

Both passes iterate all ops. On a 35-layer model with INT4 constexpr,
`const_elimination` may try to materialize constexprs to check whether
they simplify. Has been fixed in coremltools 9.0 (the constexpr path
short-circuits), but worth a size assertion in the optimizer per 7.1.

---

## 8. Additional undocumented-but-valuable observations

### 8.1 `guard_negative_gather_indices` (`defs/optimize_tensor_operation.py:684`)
In default pipeline (line 107). Relevant because our `InModelArgmax`
does a gather. On iOS 18+ argmax → gather with negative indices was a
known ANE bug; this pass inserts a guard. Leave alone; already fires.

### 8.2 `detect_concat_interleave` (`defs/optimize_tensor_operation.py:411`)
In default pipeline (line 84). Fires if a concat is fed by stacked
splits. Our QKV-split-then-per-head code pattern can hit this. The
fused path is faster on ANE. Already running.

### 8.3 Symbolic `RangeDim` + `materialize_symbolic_shape_program`

A pattern that comes up in Hugging Face's `optimum-cli` tool: trace with
`RangeDim(1, 512)` on Q, then materialize for N specific shapes. The
coremltools 9.0 release notes call this the recommended workflow for
LLMs. We have not tried it — entirely novel lever.

### 8.4 Per-op compute units — NOT supported

coremltools 9.0 allows `compute_units` only per-model or per-function.
There is **no** per-op routing. The `preferred_compute_units` JSON
entry we write in `fix_coreml_zoo_manifest.py` is advisory only for the
Swift runtime's choice of `MLModelConfiguration.computeUnits`; the
Core ML compiler does not consume it. So "lm_head on GPU, rest on ANE
within same chunk" is not achievable via coremltools — would need to
physically split the lm_head into its own mlpackage (already done in
`build_eagle3_gpu.py`, same pattern can be applied to chunk4).

### 8.5 `ct.optimize.torch` vs `ct.optimize.coreml` (coremltools 9.0)

`ct.optimize.torch` applies quantization **before** conversion (PyTorch
graph still has real float ops, annotated with observers). For LLMs
this is heavier to set up and tends to give cleaner low-bit (2-3 bit)
results. `ct.optimize.coreml` is post-conversion, LUT-based, what we
use for INT4. Switching to `ct.optimize.torch` could unlock W2 / W3
palettization **with** fine-tuning recovery, but is rejected per
`rejected_approaches.md`.

### 8.6 ExecuTorch 0.5 integration

coremltools 9.0 landed an ExecuTorch 0.5 export path
(`ct.convert(..., source="torch_export")`). For LLMs this exposes
the PyTorch 2 graph directly, bypassing TorchScript tracing. Marginal
because our tracing works, but the pt2 path does preserve dynamic shapes
better — useful for the multi-bucket design in section 4. Not urgent;
flagged for the future.

### 8.7 Apple Foundation Models report (arXiv 2507.13575)

Cited tricks that are **documented** in the paper: layer-pair KV share
(we have it), 8-bit KV + LoRA recovery (needs training, rejected).
The report also mentions a **GQA-tiling trick** (section 4.2 of the
paper) where they pack the single KV head across the 8 Q heads via an
explicit `repeat_interleave`-as-const pattern, which they found matches
ANE tiling better than the runtime broadcast. Our Gemma 4 has exactly
the same GQA-1 geometry. Worth trying as a standalone rewrite — not a
MIL pass, but an architectural tweak (could go into
`docs/GEMMA4_ANE_REWRITES.md` as rewrite 11).

### 8.8 ANEMLL comparison

ANEMLL (`github.com/anemll/anemll`) `utils/convert.py` applies the
coremltools DEFAULT pipeline and nothing more. They do not wire in
`optimize_mlpackage_graph`-style post-save re-optimization. They do
**use stateful models** for KV (their LLaMA conversion has `states=`
set). They also do **per-chunk `compute_units` via manifest** — we
already do that. Net: nothing ANEMLL does that we do not, except the
stateful KV which we have in the worktree and need to re-benchmark on
iOS 26.

---

## 9. Summary of what to apply first

The single-highest ROI change today:

1. Patch `optimize_mlpackage_graph.py` pass list per section 5.1.
2. Add size-assert guard (section 7.1).
3. Wire the optimizer into `build_merged_chunks.py` and
   `build_verify_chunks.py` per section 5.2.
4. Benchmark decode tok/s before/after on iPhone 17 Pro at 2K/8K.

Expected decode gain: **+5–10 %** on 15 tok/s baseline → 16–17 tok/s,
with **faster cold compile** (audit item 18's claimed 1–2 min → 30–40 s).

Medium-term, highest-ROI changes:

5. Retry stateful KV (section 3, section 5.4) on iOS 26 — potentially
   **+15–20 %** decode.
6. Multi-function materialization (section 4) — disk + prefill/decode
   swap latency win, not throughput.

All of the above are orthogonal to the architectural rewrites in
`docs/GEMMA4_ANE_REWRITES.md`; both workstreams can progress in parallel.
