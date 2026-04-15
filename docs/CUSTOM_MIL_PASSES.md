# Custom MIL Graph Passes and Composite Ops for Gemma 4

Date: 2026-04-15
Scope: evaluate whether `coremltools` lets us register **custom MIL passes**
and/or **custom composite MIL ops** to capture Gemma-4-specific fusion
patterns that stock passes miss (RMSNorm+Conv absorb, logit softcap, QK-norm,
manual softmax, RoPE, residual+pre-norm).
Code references are relative to `coremltools==9.0` installed at
`/opt/homebrew/lib/python3.14/site-packages/coremltools`.

---

## 1. Executive summary

**Custom MIL passes are fully supported and the hook is clean**: a one-line
`@register_pass(namespace="gemma4")` decorator makes any `AbstractGraphPass`
subclass routable from a `ct.PassPipeline`, and our existing
`conversion/optimize_mlpackage_graph.py` already drives the registry via
`PASS_REGISTRY[name](prog)`. Writing Gemma-specific fusion passes is
technically straightforward: each pass is ~50-150 LOC of pattern matching in
the style of `fuse_activation_silu` or `fuse_conv_scale`.

**Custom composite MIL ops are supported but useless for ANE**. The
`@register_op(is_custom_op=True)` path serializes to proto as
`custom_layer`, which is the legacy NN `MLCustomLayer` interface — CPU-only,
Swift-side Obj-C implementation, zero ANE support. A "dialect op" (namespaced,
with a lowering function registered via `LowerComplex`) is the closest thing
to a real composite op, but it must decompose to stock MIL ops before
serialization, so it is just a pretty wrapper around the same scalar ops the
ANE sees today. **No custom op will ever run on ANE.**

**Practical conclusion**: the high-payoff work lives at two tiers:

1. **Python-side PyTorch rewrite** (what we already do with
   `absorb_rmsnorm_scale_into_conv`) for structural changes the tracer should
   never see (weight fusion, module replacement, RoPE packing).
2. **Custom `common::gemma4_*` MIL passes** for peephole patterns that only
   show up after conversion (redundant casts the stock pipeline missed,
   softcap → bounded activation, consolidating the cat/max/sub/exp/sum/div
   softmax back into a single `softmax` op). Each pass is ~100 LOC and runs
   in the same harness we already have.

The third tier — a monolithic "gemma4_attention" op that the ANE backend
magically accelerates — **does not exist as a real capability**, only as a
readability wrapper. Several sections below debunk this in detail so we do
not get tempted again in six months.

Worth pursuing: **yes, but narrowly** — 2-3 passes, ~400-600 LOC total, ~3
days to implement and validate per pass. Expected ANE residency delta:
low (maybe 5-10% op count reduction beyond the stock passes we already run;
the big wins came from `optimize_mlpackage_graph.py` already). Expected wall
clock delta on decode: **1-3%** realistic, not the 10%+ that would justify
heavy investment. Recommend a small bounded experiment on the RMSNorm+Conv
pattern first and let the measurement decide whether to continue.

---

## 2. Custom pass mechanism in coremltools

### 2.1 The registry and base class

`coremltools/converters/mil/mil/passes/pass_registry.py` defines the whole
interface:

```python
# L49-65
PASS_REGISTRY = PassRegistry()

def register_pass(namespace, override=False, name=None):
    def class_wrapper(pass_cls):
        PASS_REGISTRY.add(namespace, pass_cls, override, name)
        return pass_cls
    return class_wrapper
```

A pass id is `namespace::ClassName`. Stock passes use `common`, backend
passes use `mil_backend` / `nn_backend`, frontend passes use `torch` /
`tensorflow`. Nothing prevents us from inventing `gemma4` as our own
namespace — the `PassRegistry.add` only cares about uniqueness.

The base class (`passes/graph_pass.py` L41-75):

```python
class AbstractGraphPass(ABC):
    def __call__(self, prog: Program):
        if not prog.skip_all_passes:
            with mb.scope(ScopeInfo(source=ScopeSource.COREMLTOOLS_GRAPH_PASS,
                                    data=[str(self)])):
                self.apply(prog)

    @abstractmethod
    def apply(self, prog: Program):
        pass

    def set_options(self, pass_options):
        ...  # key/value options the pipeline forwards from set_options()
```

So the shape of every pass is: subclass `AbstractGraphPass`, decorate with
`@register_pass`, implement `apply(prog)` that iterates
`prog.functions.values()` and mutates the block in place. That's it.

### 2.2 Hooking custom passes into a pipeline

`passes/pass_pipeline.py` (L282-320) exposes `PassPipeline` with
`append_pass`, `insert_pass`, `remove_pass`, `set_options`. It validates each
name against `PASS_REGISTRY` at mutation time:

```python
def append_pass(self, pass_name):
    if pass_name not in PASS_REGISTRY:
        raise ValueError(f"The pass {pass_name} is not registered.")
    self._pass_names.append(pass_name)
```

And `ct.convert` accepts the pipeline via the documented `pass_pipeline`
kwarg (`_converters_entry.py` L82 `pass_pipeline: Optional[PassPipeline]`
and L609-658 where it is forwarded to the main conversion driver). So the
canonical recipe is:

```python
import gemma4_passes           # side-effect: registers gemma4::* passes
pipeline = ct.PassPipeline.DEFAULT
pipeline.insert_pass(index=pipeline.passes.index(
    "common::fuse_layernorm_or_instancenorm") + 1,
    pass_name="gemma4::fuse_rmsnorm_conv")
pipeline.append_pass("gemma4::fuse_logit_softcap")
mlmodel = ct.convert(traced, pass_pipeline=pipeline, ...)
```

**However**, our conversion path today is `optimize_mlpackage_graph.py`,
which runs passes **after** conversion by re-loading the MIL program from an
mlpackage, applying passes directly, and re-serializing via
`coremltools.converters.mil.backend.mil.load`. That path is independent of
`ct.convert`'s pipeline — it just indexes `PASS_REGISTRY[name]` directly
(see lines 108-129 of the optimizer). **Any custom pass we register by
importing our module wires up for free**: we just `import gemma4_passes` at
the top of `optimize_mlpackage_graph.py` and add `"gemma4::fuse_..."` to
`DEFAULT_PASSES`.

### 2.3 Existing canonical pattern-matching example

`backend/mil/passes/fuse_activation_silu.py` is the cleanest reference —
50 LOC, detects `sigmoid(x) * x → silu(x)`. Key structural ingredients:

```python
def _match_pattern(op):
    if op.op_type == "sigmoid": ...                # anchor op
    child_ops = list(op.outputs[0].child_ops)      # fan-out
    if mul_candidate.op_type != "mul": return None # shape of follower
    if mul_inputs_actual != mul_inputs_expect: return None
    return mul_candidate

def _try_to_transform(sigmoid_op, mul_op, block):
    x = mb.silu(x=sigmoid_op.x, name=out_name, before_op=sigmoid_op)
    block.replace_uses_of_var_after_op(anchor_op=mul_op,
                                       old_var=mul_op.outputs[0], new_var=x)
    block.remove_ops([sigmoid_op, mul_op])

@block_context_manager
def _fuse_activation_silu_block(block): ...        # walk + recurse into blocks

@register_pass(namespace="mil_backend")
class fuse_activation_silu(AbstractGraphPass):
    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _fuse_activation_silu_block(f)
```

Helpers we get for free from `passes/helper.py`:
`block_context_manager` (avoids per-op `with block:` overhead),
`_check_child_op_type`, `_check_no_output_connection`,
`_check_var_scalar_value`, `_check_var_scalar_value_in_interval`. These are
all we need for the patterns below — pattern matching is sufficient, no
e-graphs or union-find. Traverse order: linear over `block.operations`,
with repeats in a `while block_changed:` loop, with block-context batching.

### 2.4 Risks at this tier (versioning / stability)

`AbstractGraphPass`, `register_pass`, and `PASS_REGISTRY` have been stable
through coremltools 6→7→8→9. The helper functions have been stable since 6.
The main thing that moves between minor versions is the internal set of
stock pass **names and order**, not the pass interface. So a custom Gemma
pass registered at `gemma4::foo` is unlikely to break on a coremltools
upgrade — but our `insert_pass(index=...)` call that inserts after a
specific stock pass is fragile if the stock pass gets renamed. Mitigation:
look up the stock pass name defensively and fall back to `append_pass`.

---

## 3. Candidate custom passes for Gemma 4

All patterns below are **peephole** — we see them in the converted program
(after the stock pipeline has already run) and they represent structure the
stock passes don't catch because they are Gemma-specific.

### 3.1 `gemma4::fuse_rmsnorm_conv` (ANERMSNorm + Conv2d → fused conv)

We already do this **at the PyTorch level** via `absorb_rmsnorm_scale_into_conv`
in `conversion/ane_ops.py` L75-114. A MIL version is still useful because
some traces slip through without the absorb having been called (e.g., merged
1-chunk models where the affine flag got reset). The MIL pass serves as a
safety net.

Pattern (what the tracer emits for `ANERMSNorm(affine=True) -> Conv2d`):

```
concat([x, -x], axis=-1)              # doubled
  -> layer_norm(axes=[-1], gamma=None, beta=None, eps=...)
  -> slice_by_index or split          # drop mirror half
  -> mul(y=const(w_norm))             # affine scale
  -> conv(weight=W_conv, bias=b, strides=(1,1))
```

Pseudocode:

```python
@register_pass(namespace="gemma4")
class fuse_rmsnorm_conv(AbstractGraphPass):
    def apply(self, prog):
        for f in prog.functions.values():
            changed = True
            while changed:
                changed = self._apply_block(f)

    @block_context_manager
    def _apply_block(self, block):
        for op in list(block.operations):
            if op.op_type != "mul":
                continue
            scale_var = op.x if op.y.val is None else op.y
            if scale_var.val is None or scale_var.val.ndim != 1:
                continue
            # Walk backwards: mul <- slice <- layer_norm <- concat([x, -x])
            pred = op.x.op if op.x.val is None else op.y.op
            if pred is None or pred.op_type not in ("slice_by_index", "split"):
                continue
            ln = pred.x.op
            if ln is None or ln.op_type != "layer_norm": continue
            cat = ln.x.op
            if cat is None or cat.op_type != "concat": continue
            if len(cat.values) != 2: continue
            a, b = cat.values
            # b must be -a (const 0 - a, or mul(a, const(-1)))
            if not _is_negation_of(b, a): continue
            # Walk forward: mul -> conv (single consumer, Conv2d weight rank 4)
            child = _single_child(op, "conv")
            if child is None or child.weight.val is None: continue
            W = child.weight.val      # (Cout, Cin, 1, 1)
            s = scale_var.val         # (Cin,)
            if W.shape[1] != s.shape[0]: continue
            # Fold: W' = W * s[None, :, None, None]
            new_W = W * s.reshape(1, -1, 1, 1).astype(W.dtype)
            new_conv = mb.conv(x=op.x_or_y_non_const, weight=new_W,
                               bias=child.bias, strides=child.strides,
                               pad=child.pad, dilations=child.dilations,
                               groups=child.groups, name=child.outputs[0].name,
                               before_op=child)
            block.replace_uses_of_var_after_op(
                anchor_op=child, old_var=child.outputs[0],
                new_var=new_conv.outputs[0])
            block.remove_ops([op, child])
            return True
        return False
```

Wins: eliminates one mul per layer (\~35 ops in Gemma-4 E2B). Stock
`fuse_conv_scale` (`optimize_conv.py` L844) handles scalar scale along Cout
but **rejects per-Cin scales** — see L918-920 where it asserts
`scale.shape[1] == Cout`. We fold per-Cin which is what RMSNorm produces.

### 3.2 `gemma4::fuse_logit_softcap` (tanh(x/S) * S → bounded op)

The Gemma logit softcap is `tanh(logits / softcap) * softcap`, with
`softcap=30.0` for the final head and `softcap=50.0` or `30.0` for
attention logits. Pattern:

```
div(x, const(30))  # or mul(x, const(1/30))
  -> tanh
  -> mul(const(30))
```

We **cannot** fuse this into a single ANE primitive because there is no
`softcap` core op in MIL (iOS15-iOS18 op sets), and a custom op would land
on CPU. What we *can* do is (a) make sure all three ops live on the same
ANE partition by sandwiching them between Conv2d head and the next op so
the compiler keeps them together; (b) convert `div(x, 30) * 30` to
`tanh(x * (1/30)) * 30` using a pre-computed reciprocal const, saving one
div (divs are more expensive than muls on ANE). That's already what the
stock `common::divide_to_multiply` pass (pipeline L45) does for us —
**no custom pass needed here**. Verify by inspecting the prog after stock
passes; if there's still a `real_div(x, 30)` we can add a fallback pass,
otherwise drop this candidate.

### 3.3 `gemma4::fuse_manual_softmax` (cat/max/sub/exp/sum/div → softmax)

When softmax is hand-written in the PyTorch module to control the fp16
numerics, the tracer emits something like:

```
max_vals = reduce_max(x, axis=-1, keep_dims=True)
shifted  = sub(x, max_vals)
exp_x    = exp(shifted)
sum_exp  = reduce_sum(exp_x, axis=-1, keep_dims=True)
out      = real_div(exp_x, sum_exp)
```

MIL has a first-class `softmax` op. Pattern match this five-op chain and
replace with `mb.softmax(x=x, axis=-1)`. Maintenance: low — the pattern is
syntactically rigid. Gain: moderate — fewer ops means less scheduling
overhead, and the ANE softmax kernel is heavily tuned. This is the single
most defensible custom pass; if the audit shows this pattern survives stock
passes (the stock `fuse_layernorm_or_instancenorm` does *not* detect
softmax), write this one first.

### 3.4 `gemma4::fuse_qk_norm_matmul`, `gemma4::fuse_rope`, `gemma4::fuse_residual_prenorm`

All three are **not worth a custom MIL pass** because:

- `fuse_qk_norm_matmul`: the pattern spans an attention sub-block, includes
  dynamic KV access, and is much easier to fuse at the torch.nn.Module level
  by presenting a monolithic `Gemma4Attention.forward` that already has Q/K
  normalized at module construction. We do this today.
- `fuse_rope`: the cos/sin tables are precomputed constants and the rotation
  is already a fixed computation graph. `common::fuse_transpose_matmul` and
  `common::const_deduplication` already dedupe RoPE tables across layers.
  There is no "rope" primitive in MIL, and creating one would be a CPU custom
  layer. Skip.
- `fuse_residual_prenorm`: no layernorm-with-residual primitive exists in
  MIL. The "fusion" would just be a syntactic grouping that the scheduler
  sees identically.

These three go in the "**Python-side rewrite**" section (§5) instead.

---

## 4. Custom composite ops — theoretical capability and practical limits

### 4.1 The three registration paths

`coremltools/converters/mil/mil/ops/registry.py` L17-54 enumerates them:

1. **`core_ops`** (no `namespace` kwarg to `register_op`): ops that have a
   direct mlprogram backend mapping. Apple-internal; new core ops require a
   coremltools release.
2. **`dialect_ops`** (`namespace="..."`, name-prefixed): frontend-scoped ops
   that **must be lowered to core ops** by a pass before backend serialization
   (see `common::lower_complex_dialect_ops`). The only built-in example is
   `complex::*` (FFT etc.), lowered via
   `passes/defs/lower_complex_dialect_ops.py` L44-74 using
   `LowerComplex.register_lower_func`.
3. **`custom_ops`** (`is_custom_op=True`): serialized as `custom_layer`
   (see `backend/mil/load.py` L391-403). This is the **legacy NeuralNetwork
   `MLCustomLayer` path** — the runtime looks up a Swift/Obj-C class by
   `class_name` string and runs it on CPU. Does not exist for
   `mlprogram`/ANE in any meaningful sense.

### 4.2 What this means for a hypothetical `gemma4_rmsnorm` or `gemma4_attention` op

- **Custom op (`is_custom_op=True`)**: falls to CPU on every call. Killswitch.
  Do not use for performance work — ever. The only legitimate use is embedding
  user-provided Swift kernels for ops coremltools genuinely cannot express.
- **Dialect op + lowering**: useful for *readability* of the MIL program and
  for factoring complex decompositions. The lowering fires before backend
  serialization, so the final mlprogram contains the same scalar MIL ops as
  today — no ANE perf delta. A `gemma4_rmsnorm` dialect op would be a nice
  way to centralize the cat/layernorm/slice/mul decomposition, but it
  **does not change what the ANE compiler sees**.
- **Core op**: cannot register without modifying coremltools itself, and
  even then the backend serializer and Espresso compiler on-device need to
  know the op. Not in our reach.

### 4.3 Is there any ANE-level "trusted ops" list we can exploit?

No — not exposed. `backend/mil/load.py` translates MIL ops to proto 1:1;
there is no "ANE whitelist" step in Python. The ANE op coverage is
determined by Espresso at model-compile time on-device, which is
closed-source. What we observe empirically: `layer_norm`, `conv` (1x1 and
3x3), `matmul` (for small contiguous inner dims), `mul`/`add`/`sub`, `silu`,
`gelu`, `softmax`, `reduce_{max,sum,mean}`, `reshape`, `transpose`,
`slice_by_index`, `concat`, `cast`, `linear`, `tanh`, `sigmoid` land on
ANE. Ops that consistently fall off ANE: most dynamic-shape ops (`gather`
with non-const indices), `real_div` with non-broadcastable shapes in some
layouts, `pow`, anything introduced by a custom_layer serialization.

**Implication**: our ceiling is already roughly what ANE supports; inventing
composite ops cannot raise it. The game is to (a) keep everything expressed
in the supported-op subset, and (b) reduce op count / memory-bound
re-layouts, which is exactly what the stock + peephole passes in §3 do.

---

## 5. Python-side rewrite alternative (the practical winner)

This is the tier we already invest in (`conversion/ane_ops.py`,
`conversion/models/gemma4_swa_merged1.py`, etc.). Why it dominates custom
MIL passes for Gemma-specific work:

- **Full control over the graph the tracer sees**. We can replace a whole
  PyTorch module with a ANE-shaped equivalent before `torch.jit.trace` runs.
- **Weight folding**. Absorbing RMSNorm scale into the next Conv weight (as
  `absorb_rmsnorm_scale_into_conv` does) is correct at training-time math
  and only needs to happen once; it leaves the converted MIL graph with
  strictly fewer ops than any post-hoc pass could achieve, because the
  scale multiply never existed in the trace.
- **Monolithic blocks**. A `Gemma4Attention.forward` that internally does
  Q/K-norm + RoPE + matmul in a single `torch.nn.Module.forward` guarantees
  the tracer emits a contiguous MIL region, which is what we want for
  partition quality — not a semantic change in the op graph, but a
  structural hint for the ANE compiler's partitioner.
- **Unit-testable in pure PyTorch**. Custom MIL passes can only be tested
  post-conversion; module rewrites get the full PyTorch testing apparatus
  for free.

The three candidates that the original prompt listed as "custom composite
ops" are all Python-side rewrites in disguise:

- **Fused RMSNormConv module**: a single `nn.Module` that owns both the
  RMSNorm state and the Conv2d weight, applies the absorb at init, and
  exposes a single `forward(x) -> Conv2d(layer_norm(cat([x,-x])))`. We have
  this. Next logical step: make it the default constructor used by
  `gemma4_swa_merged1.py` so new conversions never produce the
  unfused pattern.
- **Fused QKNormAttention**: one module that computes Q/K norm + attention
  score + softmax + V projection, so the trace never spreads attention
  across separate module boundaries. Complements the SWA chunking layout.
- **Packed RoPE**: precompute `cos[None,None,:,:]` and `sin[None,None,:,:]`
  as buffers once, share across layers via `nn.Buffer` deduplication, and
  apply as two elementwise muls plus one add. Current code is already
  roughly this shape.

**Recommendation**: route all new Gemma fusion work through the Python tier
first. Only drop to a custom MIL pass when (a) the pattern is emergent from
conversion (peephole — can't be expressed at PyTorch level) or (b) we need
a safety net for graphs we don't control (merged mlpackages we re-optimize
from disk).

---

## 6. Implementation sketches for the top 2-3 wins

### 6.1 `gemma4::fuse_manual_softmax` (highest payoff, lowest risk)

File: `conversion/mil_passes/fuse_manual_softmax.py` (new). ~100 LOC.

```python
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import (
    _check_child_op_type, block_context_manager,
)
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _match(op):
    # Anchor on the final real_div of the shifted-exp softmax.
    if op.op_type != "real_div":
        return None
    exp_op = op.x.op
    if exp_op is None or exp_op.op_type != "exp":
        return None
    sub_op = exp_op.x.op
    if sub_op is None or sub_op.op_type != "sub":
        return None
    max_op = sub_op.y.op
    if max_op is None or max_op.op_type != "reduce_max":
        return None
    sum_op = op.y.op
    if sum_op is None or sum_op.op_type != "reduce_sum":
        return None
    if sum_op.x != exp_op.outputs[0]:
        return None
    if max_op.x != sub_op.x:
        return None
    if (max_op.axes.val.tolist() != sum_op.axes.val.tolist()
        or not max_op.keep_dims.val or not sum_op.keep_dims.val):
        return None
    axis = int(max_op.axes.val[-1])
    return {"x": sub_op.x, "axis": axis,
            "ops": [max_op, sub_op, exp_op, sum_op, op]}


@register_pass(namespace="gemma4")
class fuse_manual_softmax(AbstractGraphPass):
    def apply(self, prog):
        for f in prog.functions.values():
            changed = True
            while changed:
                changed = self._apply(f)

    @block_context_manager
    def _apply(self, block):
        for op in list(block.operations):
            m = _match(op)
            if m is None:
                continue
            new = mb.softmax(x=m["x"], axis=m["axis"],
                             name=op.outputs[0].name, before_op=m["ops"][0])
            block.replace_uses_of_var_after_op(
                anchor_op=op, old_var=op.outputs[0], new_var=new)
            block.remove_ops(m["ops"])
            return True
        return False
```

Install into the optimizer:

```python
# conversion/optimize_mlpackage_graph.py
from conversion.mil_passes import fuse_manual_softmax  # noqa: F401 register

DEFAULT_PASSES.append("gemma4::fuse_manual_softmax")
```

Validation plan: convert one Gemma-4 chunk with the current
`manual_softmax=True` path in `gemma4_swa_merged1.py`, diff op counts
before/after, then `--verify-equivalence` against a 16-token dummy. Expected
op count delta: –5 per softmax site, –5 × (num_layers × 2 for self+cross if
applicable). For Gemma-4 E2B that's \~50 ops across the model.

### 6.2 `gemma4::fuse_rmsnorm_conv` (safety-net, medium payoff)

See §3.1 for pseudocode. Implementation cost: ~150 LOC because the
"b is the negation of a" check has to handle both `mul(a, const(-1))` and
`sub(0, a)` forms. Guards: only fire if `mul` is a single-consumer
of the rmsnorm output AND the conv has a rank-4 1x1 weight AND the scale
is a rank-1 const of length `Cin`. Test: convert a chunk where the
Python-side absorb was **deliberately skipped** (`affine=True` left on),
verify the MIL pass recovers the same fused conv weights as the Python
absorb.

### 6.3 Optional: `gemma4::fuse_logit_softcap`

Only write this if inspection shows the div→tanh→mul chain surviving after
the stock `divide_to_multiply` pass. Pattern is trivial but the
win is marginal (one div removal per softcap site; Gemma has 2-3 sites).
Do last, or skip.

### 6.4 Integration into the pipeline

Three places touch the pass wiring:

1. `conversion/mil_passes/__init__.py`: imports each pass module so the
   `@register_pass` side effects run on import.
2. `conversion/optimize_mlpackage_graph.py`: `from conversion import mil_passes`
   at module top; extend `DEFAULT_PASSES` with the `gemma4::*` names.
3. Conversion-time hook (if we want the passes to fire during `ct.convert`
   rather than post-hoc): in the frontend conversion script (e.g.,
   `conversion/build_eagle3_chunks.py`), build a `PassPipeline.DEFAULT`,
   call `pipeline.append_pass("gemma4::...")`, and pass to `ct.convert`.
   Prefer the post-hoc path: it's the one we already run in CI and it keeps
   the pass code out of the critical conversion flow.

---

## 7. Maintenance and risk assessment

### 7.1 Per-pass maintenance cost

- `AbstractGraphPass` and `@register_pass` API: stable across
  coremltools 6→9, no breakage. Low risk.
- Stock pass names we depend on for ordering (e.g.,
  `common::fuse_layernorm_or_instancenorm`): stable but Apple could rename
  in a refactor. Mitigation: all our passes are registered but only
  **appended** to the pipeline by our own optimizer script; we don't use
  `insert_pass(index=...)` against stock pass names, so a rename won't
  break us — it would just change where the stock pass runs relative to
  ours, which is tolerable.
- MIL op signatures (e.g., `mb.softmax(x, axis)`, `mb.conv(x, weight,
  bias, strides, ...)`): versioned per opset (iOS13/15/16/17/18). We
  target iOS18 today; adding iOS19 support later means bumping
  `curr_opset_version` and re-validating. Low-medium risk.
- Pattern matching on `op.op_type` string literals: brittle against
  upstream pass reshuffles that introduce new canonical forms. Mitigation:
  each pass is self-contained and idempotent; failures are soft (pattern
  doesn't match → no-op) rather than crashes.

### 7.2 Risks to walk in eyes-open on

- **Silent numerical drift**. Fusing the manual softmax into
  `mb.softmax` changes the fp16 numerics subtly — Apple's softmax may use
  a different max-shift or epsilon. Mitigation: keep
  `--verify-equivalence` in the optimizer CLI and enforce
  `max_abs_diff < 1e-3` in CI. If it fails on a specific chunk, skip the
  pass for that chunk only.
- **Coremltools 10 breakage**. If a future release introduces op
  versioning changes that reject a pattern match, our pass will silently
  no-op (not crash) because we always guard on `.val is None` and
  `op_type` checks. Worst case: we notice a regression in op count
  post-upgrade and re-tune.
- **ANE partition disruption**. A fused op might *prevent* a partition
  that was happening organically. Example: if `mb.softmax` is on a
  different ANE partition than the manual-softmax decomposition (because
  ANE's internal partitioner has different size-based heuristics), the
  fusion could make decode slower, not faster. Mitigation: A/B benchmark
  every pass against the baseline before merging. Do not trust op count
  deltas as a proxy for perf deltas.
- **Maintenance burden scales with pass count**. Each new pass is
  another piece of code to keep tested across ct versions. Cap at 3-5
  passes total; anything more should be consolidated into Python-side
  rewrites.

### 7.3 What Apple's own conversion probably does

Pure speculation since Foundation Models conversion is closed-source, but
the signals we have:

- `ml-ane-transformers` (the public reference): **pure Python-side
  rewrite**, no custom MIL passes at all. They rewrite attention as 4D
  NCHW conv-based kernels at the PyTorch level (see their
  `ReferenceLlamaConv2dFFN` etc.) and let stock ct passes handle the rest.
- Apple's internal pipeline is almost certainly a Python rewrite tier
  plus whatever proprietary ANE-aware passes live inside Espresso
  (on-device compiler). They would not ship a "custom MIL pass" as a
  public API point because it is an unstable surface area for third-party
  developers.

This triangulates with our recommendation: **Python-side rewrites first,
peephole MIL passes as a safety net, never ever custom_layer ops.**

### 7.4 Estimate and recommendation

- `gemma4::fuse_manual_softmax`: 100 LOC, 1 day implement + validate, LOW
  maintenance, 1-2% decode improvement expected. **Build this first.**
- `gemma4::fuse_rmsnorm_conv`: 150 LOC, 2 days implement + validate, LOW
  maintenance, only fires as safety net for graphs where Python absorb
  was skipped. **Build second, bench expected improvement is ~0% on
  properly-absorbed graphs; its value is preventing regressions.**
- `gemma4::fuse_logit_softcap`: skip unless op-count audit shows the
  pattern surviving.
- Any custom op (`is_custom_op=True` or dialect): **do not build**.
  No ANE benefit. Ever.

Total budget to reach the expected ceiling: **~3 days, ~400 LOC, one
unit-test file**. Place code under `conversion/mil_passes/`, wire through
`optimize_mlpackage_graph.py` as additional entries in `DEFAULT_PASSES`.

If after building `fuse_manual_softmax` the measured delta is <0.5% on
decode throughput, stop — the remaining headroom lives in other tiers
(NCHW end-to-end, KV-cache layout, prefill chunk size), not in custom MIL
passes. This is likely the outcome. Budget the investigation accordingly:
1 engineer-day of prototyping before committing to the full 3 days.

---

## Appendix: key source file citations

- `/opt/homebrew/lib/python3.14/site-packages/coremltools/converters/mil/mil/passes/graph_pass.py` L41-75 — `AbstractGraphPass` base class.
- `.../converters/mil/mil/passes/pass_registry.py` L13-65 — `PASS_REGISTRY`, `@register_pass`.
- `.../converters/mil/mil/passes/pass_pipeline.py` L24-297 — pipeline definitions, `append_pass`/`insert_pass`/`remove_pass`.
- `.../converters/_converters_entry.py` L82, L609-658 — `ct.convert(pass_pipeline=...)` plumbing.
- `.../converters/mil/backend/mil/passes/fuse_activation_silu.py` — reference pattern-matching pass, 80 LOC.
- `.../converters/mil/mil/passes/defs/optimize_conv.py` L844-... — `fuse_conv_scale`, the stock pass that refuses per-Cin scales (hence our custom rmsnorm_conv pass).
- `.../converters/mil/mil/passes/defs/lower_complex_dialect_ops.py` L44-74 — `LowerComplex.register_lower_func` pattern for dialect ops.
- `.../converters/mil/mil/ops/registry.py` L17-191 — `SSAOpRegistry`, `register_op`, the three op kinds (core/dialect/custom).
- `.../converters/mil/backend/mil/load.py` L391-403 — custom_op → `custom_layer` serialization (CPU-only sink).
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/optimize_mlpackage_graph.py` — current post-hoc optimizer; the entry point for hooking custom passes.
- `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/ane_ops.py` L25-114 — `ANERMSNorm` + `absorb_rmsnorm_scale_into_conv`, the Python-side rewrite we already use.
