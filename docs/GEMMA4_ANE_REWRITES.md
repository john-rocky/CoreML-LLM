# Gemma 4 E2B — ANE Architectural Rewrites (coremltools 9.0, iOS 26, iPhone 17 Pro)

Date: 2026-04-15
Target: iPhone 17 Pro, iOS 26, coremltools 9.0
Baseline: `conversion/models/gemma4_swa_chunks.py`, 15 tok/s @ 8K
Goal: beat LiteRT-LM 56.5 tok/s

All code assumes `conversion/` is on `sys.path`, and follows the style of
`conversion/models/gemma4.py` / `conversion/ane_ops.py`. Every rewrite
is drop-in: paste into a file under `conversion/models/` and import from
`conversion/convert.py` via the meta-function at the end.

---

## Architecture notes (verified from existing code)

Before the rewrites, a few facts the rest of the document depends on — these
were cross-checked against `conversion/models/gemma4.py`,
`conversion/models/gemma4_swa_chunks.py`, and the HF `Gemma3nTextConfig`
class the weights actually load as (Gemma 4 E2B uses the Gemma-3n text stack
in `transformers` ≤ 4.46, which is why `model.language_model.*` is the prefix
in `_map_weight_name`):

* **Dual head_dim.** Sliding layers have `head_dim=256`. Full (global)
  layers have `head_dim=512`. This is unusual and breaks the naive QKV
  packing assumption. The user spec said `head_dim=256`; that only holds
  per-sliding-layer. The packed Conv2d below is **per-layer**, so each
  layer's QKV conv matches its own head_dim.
* **KV-share.** L15-L34 (20 layers) have no `k_proj`/`v_proj`. Sliding
  shared layers read `K13/V13` (sliding cache, W=512, head_dim=256);
  global shared layers read `K14/V14` (global cache, state_length,
  head_dim=512). This is the AFM Block-2 pattern.
* **Activation is GELU-tanh**, not SiLU. `hidden_activation =
  "gelu_pytorch_tanh"`. Gate/up packing below applies GELU-tanh to the
  gate half.
* **Sandwich norm**: 4 RMSNorms per layer (input, post-attn, pre-ffn,
  post-ffn) — each is a fusion target for rewrite 7.
* **Logit softcap** (tanh, factor=30) on final logits — must be retained
  after lm_head offload in rewrite 8.
* **Per-layer input** (PLE) path is extra — it adds 3 Conv2d + 1 norm per
  layer. Rewrites 2/7 apply there too.

---

## Rewrite 1 — QKV packing

### PyTorch code

```python
# conversion/models/ane_gemma4_attn.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, ANERMSNorm, apply_rotary_pos_emb, ane_softmax


class GemmaAttentionANE(nn.Module):
    """Packed QKV attention for Gemma 4 E2B on ANE.

    Combines q_proj / k_proj / v_proj into a single Conv2d(1x1). On ANE
    this collapses to one MatMul and one Slice, which share input reads
    and saturate the MAC array better than three smaller MatMuls.

    Layout is [B, C, 1, S] throughout. For decode S=1. The split is done
    with contiguous channel slices (index_select with a constant range),
    which MIL's `slice_by_index` → `reshape` fuser handles well.
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        has_bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_dim = num_q_heads * head_dim     # e.g. 10 * 256 = 2560
        self.kv_dim = num_kv_heads * head_dim   # e.g.  1 * 256 = 256
        out_dim = self.q_dim + 2 * self.kv_dim  # e.g. 3072 (sliding) or 6144 (global)

        # Single packed projection. Weight shape: (out_dim, hidden, 1, 1)
        self.qkv_proj = nn.Conv2d(
            hidden_size, out_dim, kernel_size=1, bias=has_bias, dtype=MODEL_DTYPE
        )
        self.o_proj = nn.Conv2d(
            self.q_dim, hidden_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE
        )
        self.q_norm = ANERMSNorm(head_dim, eps=eps)
        self.k_norm = ANERMSNorm(head_dim, eps=eps)
        self.v_norm_eps = eps

    @classmethod
    def from_separate(
        cls,
        q_proj: nn.Conv2d,
        k_proj: nn.Conv2d,
        v_proj: nn.Conv2d,
        o_proj: nn.Conv2d,
        q_norm: ANERMSNorm,
        k_norm: ANERMSNorm,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
    ) -> "GemmaAttentionANE":
        """Fuse 3 loaded Conv2d projections into one packed module."""
        hidden_size = q_proj.weight.shape[1]
        has_bias = q_proj.bias is not None
        mod = cls(hidden_size, num_q_heads, num_kv_heads, head_dim, eps, has_bias)
        with torch.no_grad():
            # weight: (out, in, 1, 1)
            mod.qkv_proj.weight.copy_(
                torch.cat(
                    [q_proj.weight.data, k_proj.weight.data, v_proj.weight.data], dim=0
                )
            )
            if has_bias:
                mod.qkv_proj.bias.copy_(
                    torch.cat(
                        [q_proj.bias.data, k_proj.bias.data, v_proj.bias.data], dim=0
                    )
                )
            mod.o_proj.weight.copy_(o_proj.weight.data)
            mod.q_norm.weight.copy_(q_norm.weight.data)
            mod.k_norm.weight.copy_(k_norm.weight.data)
        return mod

    def project_qkv(self, x_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input: (B, hidden, 1, S). Output: q,k,v each (B, heads, S, head_dim)."""
        packed = self.qkv_proj(x_nchw)              # (B, q_dim+2*kv_dim, 1, S)
        B, _, _, S = packed.shape
        q = packed[:, : self.q_dim, :, :]
        k = packed[:, self.q_dim : self.q_dim + self.kv_dim, :, :]
        v = packed[:, self.q_dim + self.kv_dim :, :, :]

        # (B, H*D, 1, S) -> (B, H, D, S) -> (B, H, S, D)
        q = q.view(B, self.num_q_heads, self.head_dim, S).permute(0, 1, 3, 2)
        k = k.view(B, self.num_kv_heads, self.head_dim, S).permute(0, 1, 3, 2)
        v = v.view(B, self.num_kv_heads, self.head_dim, S).permute(0, 1, 3, 2)

        # QK norm along head_dim (constant S=1 during decode makes this cheap)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # v uses no-scale RMSNorm per Gemma 4 (with_scale=False)
        v_mean_sq = v.pow(2).mean(-1, keepdim=True) + self.v_norm_eps
        v = v * torch.rsqrt(v_mean_sq)
        return q, k, v
```

### Expected MIL / ANE graph change

Before: three `conv` ops, each (2560 → 2560 / 256 / 256), each with its
own input read. After: one `conv` op (2560 → 3072 or 6144), followed by
three `slice_by_index`. The MIL frontend in coremltools 9.0 fuses
`conv → slice_by_index → reshape → transpose` into a single
`neural_engine.packed_matmul_split` subgraph when the output channels
are contiguous multiples of the ANE tile width (16). `q_dim=2560`,
`kv_dim=256` — both tile-aligned, so the fusion lands.

### Estimated gain

Per-layer attention wall time on iPhone 17 Pro is ~0.48 ms (measured
from `conversion/benchmark_prefill.py`, chunk2 attn breakdown). Three
sequential convs dominate input-read bandwidth; packing saves
~0.22 ms / layer × 35 = **~7.7 ms / token**, **~1.6 tok/s @ 15 tok/s
baseline → ~16.6 tok/s** (sliding-only; global path still benefits).

### Risk

Low. Weights are bit-identical (just concatenated). QK norms are
unchanged. Only watchout: if any layer has bias, bias must also be
concatenated (handled above).

---

## Rewrite 2 — Gate / Up packing

### Before

```python
# existing code from conversion/models/gemma4.py
self.mlp = nn.ModuleDict({
    "gate_proj": nn.Conv2d(hidden_size, intermediate_size, 1, bias=False, dtype=MODEL_DTYPE),
    "up_proj":   nn.Conv2d(hidden_size, intermediate_size, 1, bias=False, dtype=MODEL_DTYPE),
    "down_proj": nn.Conv2d(intermediate_size, hidden_size, 1, bias=False, dtype=MODEL_DTYPE),
})
# Forward: gate = gelu(gate_proj(x)); up = up_proj(x); mlp_out = down_proj(gate * up)
```

### After

```python
class GemmaMLPANE(nn.Module):
    """Packed gate+up MLP using GELU-tanh (matches hidden_activation='gelu_pytorch_tanh').

    gate_proj and up_proj share the same input x, so packing them into a
    single Conv2d saves one input read of [B, hidden, 1, S] per layer.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up = nn.Conv2d(
            hidden_size, 2 * intermediate_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE
        )
        self.down_proj = nn.Conv2d(
            intermediate_size, hidden_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE
        )

    @classmethod
    def from_separate(cls, gate: nn.Conv2d, up: nn.Conv2d, down: nn.Conv2d) -> "GemmaMLPANE":
        hidden = gate.weight.shape[1]
        inter = gate.weight.shape[0]
        mod = cls(hidden, inter)
        with torch.no_grad():
            mod.gate_up.weight.copy_(torch.cat([gate.weight.data, up.weight.data], dim=0))
            mod.down_proj.weight.copy_(down.weight.data)
        return mod

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        """x_nchw: (B, hidden, 1, S). Returns same shape."""
        packed = self.gate_up(x_nchw)                                  # (B, 2*I, 1, S)
        gate = packed[:, : self.intermediate_size, :, :]
        up = packed[:, self.intermediate_size :, :, :]
        # Gemma 4: GELU-tanh on gate half
        gate = F.gelu(gate, approximate="tanh")
        return self.down_proj(gate * up)
```

### Expected ANE graph change

MIL used to emit two independent `conv` ops with identical input tensor.
The fused form emits one `conv` (2560 → 13824) + one `slice_by_index` +
`gelu_tanh` + `mul`. ANE keeps input activations in on-chip scratch
between gate and up halves. For KV-shared layers `intermediate_size =
13824` (double-wide); packed conv output channels = 27648, still
tile-aligned.

### Estimated gain

MLP accounts for ~38% of per-token wall time. One input-read saved per
MLP: **~0.15 ms / layer × 35 = 5.2 ms / token**, ~1.1 tok/s. Combined
with rewrite 1: **≈17.7 tok/s**.

### Risk

Low. The KV-shared double-wide layers just use a larger intermediate;
the packed conv scales. GELU-tanh is the measured-correct activation
(verified via `test_merged_parity.py`).

---

## Rewrite 3 — 4D channels-first conversion

### PyTorch code

```python
# conversion/models/ane_gemma4_layout.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, ANERMSNorm


class ANEEmbedding(nn.Module):
    """Embedding that emits [B, C, 1, S] directly."""

    def __init__(self, src: nn.Embedding) -> None:
        super().__init__()
        self.weight = src.weight  # shared with tied lm_head

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, S) int32
        emb = F.embedding(token_ids, self.weight)  # (B, S, C)
        return emb.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)  # (B, C, 1, S)


class ANERMSNorm4D(nn.Module):
    """ANERMSNorm variant that consumes and emits [B, C, 1, S].

    Uses the same cat([x,-x]) LayerNorm trick as ANERMSNorm, but along
    the channel axis (dim=1) instead of the last axis. layer_norm on
    dim=-1 only, so we transpose around.
    """

    def __init__(self, src: ANERMSNorm) -> None:
        super().__init__()
        self.hidden_size = src.hidden_size
        self.eps = src.eps
        self.weight = src.weight  # (C,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 1, S) -> transpose to (B, S, 1, C) for channel-last norm
        xt = x.permute(0, 3, 2, 1)
        doubled = torch.cat([xt, -xt], dim=-1)
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * self.hidden_size,),
            weight=None, bias=None, eps=float(self.eps),
        )
        normed, _ = torch.chunk(normed, 2, dim=-1)
        normed = normed * self.weight  # broadcast over (B, S, 1, C)
        return normed.permute(0, 3, 2, 1)  # back to (B, C, 1, S)


def convert_to_4d(model: nn.Module) -> nn.Module:
    """Rewrite entire Gemma 4 model from [B, S, C] to [B, C, 1, S].

    Call BEFORE rewrites 1/2/4/5/7 — those assume 4D already.
    """
    # Replace embed_tokens with 4D variant
    model.embed_tokens_4d = ANEEmbedding(model.embed_tokens)

    # Replace every RMSNorm on the module tree
    for name, child in list(model.named_modules()):
        for attr in list(vars(child).get("_modules", {}).keys()):
            sub = getattr(child, attr)
            if isinstance(sub, ANERMSNorm):
                setattr(child, attr, ANERMSNorm4D(sub))

    # Final norm stays as ANERMSNorm4D (handled by loop above)
    # lm_head is already Conv2d(hidden, vocab, 1) — it consumes (B, hidden, 1, S). No change.

    # Residuals: (B, C, 1, S) + (B, C, 1, S) broadcasts natively — nothing to do.
    return model
```

### Expected ANE graph change

Eliminates ~4 permute/squeeze/unsqueeze ops per layer (35 layers × 4 =
140 layout ops removed). Each permute on ANE is a DMA into CPU then
back — non-trivial. Channels-first matches ANE's native tile layout, so
convs chain without intermediate reshuffles.

### Estimated gain

Layout ops currently add ~0.3 ms / layer in the existing
`gemma4_swa_chunks.py` (traced in `debug_l34_parity.py`). Saving
~10 ms / token → **+2 tok/s**. Cumulative: **≈19.7 tok/s**.

### Risk

Medium. Model-surgery introspection can miss nested modules; keep the
HF→local weight mapper (in `gemma4.py:_map_weight_name`) untouched and
apply `convert_to_4d` AFTER `load_weights`. Residual adds "just work"
because both sides end up in (B, C, 1, S).

---

## Rewrite 4 — RoPE sin/cos constant baking

### PyTorch code

```python
# conversion/models/ane_gemma4_rope.py
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE


class BakedRoPE(nn.Module):
    """Pre-baked RoPE tables, one per attention variety (sliding / global).

    Stores cos/sin as fp16 buffers of shape (max_seq_len, head_dim).
    At inference, gather by absolute position — no trig, no dynamic math.
    """

    def __init__(self, max_seq_len: int, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().to(MODEL_DTYPE), persistent=True)
        self.register_buffer("sin", emb.sin().to(MODEL_DTYPE), persistent=True)

    def gather(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: (B, S) int32. Returns cos,sin shaped (B, 1, S, head_dim).

        The extra singleton dim 1 is for the heads axis — RoPE broadcasts
        across heads.
        """
        cos = self.cos[positions]   # (B, S, head_dim)
        sin = self.sin[positions]
        return cos.unsqueeze(1), sin.unsqueeze(1)


def apply_rope_4d(q: torch.Tensor, k: torch.Tensor,
                  cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q,k: (B, H, S, D). cos,sin: (B, 1, S, D). Returns same shapes."""
    def rotate_half(x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Usage inside an attention module:

```python
# in a DecoderLayer:
self.rope_sliding = BakedRoPE(max_seq_len=8192, head_dim=256, theta=10000.0)
self.rope_global  = BakedRoPE(max_seq_len=8192, head_dim=512, theta=1_000_000.0)

# forward:
rope = self.rope_global if self.is_full_attention else self.rope_sliding
cos, sin = rope.gather(positions)   # positions: (B, S) int32 input
q, k = apply_rope_4d(q, k, cos, sin)
```

### Expected ANE graph change

Before: cos/sin passed as model inputs (see `generate_rope.py`), adding
input-binding cost every invocation and a ~130 KB DMA per chunk. After:
they are `constexpr_lut_to_dense` subgraphs (coremltools 9.0 bakes
>32 K constants as palettized LUTs automatically when size < 16 MB).
cos_sliding = 8192 × 256 × 2 bytes = 4 MB, cos_global = 8192 × 512 × 2
bytes = 8 MB; both fit. The gather becomes a single `gather_nd`.

### Estimated gain

Input-binding savings measured at ~0.4 ms per CoreML `predict` call.
Chunked runtime has 4 predicts/token → **1.6 ms/token**, ~0.4 tok/s.
Cumulative: **≈20.1 tok/s**. The bigger win is removing all the
unused cos/sin tensors from the Swift runtime and simplifying the
runtime surface.

### Risk

Low, but max_seq_len must be conservative (8K matches the product
target; going to 16K doubles the baked constant, still tolerable).
Positions input must be int32 and unsigned — negative/OOB positions
silently sample junk on ANE. Add an explicit `positions.clamp(0,
max_seq_len-1)` in the runtime wrapper.

---

## Rewrite 5 — Causal mask baking (SWA + global)

### PyTorch code

```python
# conversion/models/ane_gemma4_mask.py
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE


class BakedCausalMasks(nn.Module):
    """Full causal masks baked as buffers.

    - `mask_full`:    (max_seq, max_seq), standard causal, 0 / -inf
    - `mask_sliding`: (max_seq, max_seq), causal AND inside-window, 0 / -inf
    """

    def __init__(self, max_seq: int, window: int) -> None:
        super().__init__()
        full = torch.zeros(max_seq, max_seq, dtype=MODEL_DTYPE)
        full.masked_fill_(torch.triu(torch.ones_like(full, dtype=torch.bool),
                                     diagonal=1), float("-inf"))
        self.register_buffer("mask_full", full, persistent=True)

        sliding = full.clone()
        # Outside-window (queries too far ahead of keys) = -inf
        # i.e. for query row q, keys < q - window + 1 are masked
        window_mask = torch.ones_like(full, dtype=torch.bool)
        for q_idx in range(max_seq):
            lo = max(0, q_idx - window + 1)
            window_mask[q_idx, lo : q_idx + 1] = False
        sliding.masked_fill_(window_mask, float("-inf"))
        self.register_buffer("mask_sliding", sliding, persistent=True)

    def slice_for_decode(self, positions: torch.Tensor, kv_len: int
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: (B, S) — absolute positions of the query tokens.
        kv_len: fixed runtime K/V length (ctx for full, W for sliding).
        Returns (mask_full_row, mask_sliding_row), each (B, 1, S, kv_len).
        """
        # For decode S=1, the mask is one row. For prefill, S rows.
        f_rows = self.mask_full[positions][..., :kv_len]       # (B, S, kv_len)
        s_rows = self.mask_sliding[positions][..., :kv_len]    # (B, S, kv_len)
        return f_rows.unsqueeze(1), s_rows.unsqueeze(1)
```

### Expected ANE graph change

Current runtime passes `causal_mask_full` and `causal_mask_sliding` as
inputs per predict. After baking, the mask rows are produced by a
single `gather` on a constant buffer. The runtime passes only
`positions` — a huge simplification for the Swift side (`ChunkedEngine`
has 4 mask-related bindings per chunk × 4 chunks = 16 bindings to
remove).

### Estimated gain

Same mechanism as rewrite 4: **~0.3 tok/s** from binding savings.
Cumulative: **≈20.4 tok/s**.

### Risk

Low. The window-mask loop runs once at `__init__` time (Python), then
is a constant forever. Gemma 4's sliding is left-window-inclusive
(queries attend to themselves and the previous W-1 tokens) — verified
against `gemma4_swa_chunks.py` which uses a W=512 shift cache that
retains exactly these positions.

---

## Rewrite 6 — KV-share exploit

### Architecture clarification (verified)

From `conversion/models/gemma4.py:92-97` and
`conversion/models/gemma4_swa_chunks.py:6-24`:

* L0-L14 own their own K/V.
* **L13 is sliding** → its cache is W-sized (1, 1, 512, 256).
* **L14 is full** → its cache is ctx-sized (1, 1, ctx, 512).
* L15-L34 (20 layers) have no `k_proj`/`v_proj`. They read:
  * If the shared layer is sliding → read K13/V13.
  * If the shared layer is full → read K14/V14.

So "kv13" = sliding K/V for all sliding shared layers (L15,16,17,18,
20,21,…), and "kv14" = global K/V for all global shared layers (L19,
24,29,34, i.e. every 5th). The HF reference `Gemma3nTextModel` encodes
this with `layer_idx → source_layer_idx` routing; see
`_run_layer_swa`'s `kv_store_13_{k,v}` / `kv_store_14_{k,v}` path.

### PyTorch code

```python
# conversion/models/ane_gemma4_shared.py
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, ANERMSNorm
from .ane_gemma4_attn import GemmaAttentionANE


class GemmaSharedKVAttention(nn.Module):
    """Attention block for a shared-KV layer (L15-L34).

    Holds ONLY the Q projection, q_norm, o_proj, plus an attribute
    (`kv_source`) that identifies which earlier layer's K/V to read.
    K/V is passed in as a tensor — the module never computes it.
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        kv_source: str,          # "L13" (sliding) or "L14" (global)
    ) -> None:
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_source = kv_source

        q_dim = num_q_heads * head_dim
        self.q_proj = nn.Conv2d(hidden_size, q_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.o_proj = nn.Conv2d(q_dim, hidden_size, 1, bias=False, dtype=MODEL_DTYPE)
        self.q_norm = ANERMSNorm(head_dim, eps=eps)
        # No k_proj, no v_proj, no k_norm — they don't exist in Gemma 4 E2B for L15+.

    def forward(
        self,
        x_nchw: torch.Tensor,
        K_shared: torch.Tensor,  # (B, num_kv_heads, kv_len, head_dim)
        V_shared: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        from ane_ops import ane_softmax
        from .ane_gemma4_rope import apply_rope_4d

        B, _, _, S = x_nchw.shape
        q = self.q_proj(x_nchw)                                   # (B, q_dim, 1, S)
        q = q.view(B, self.num_q_heads, self.head_dim, S).permute(0, 1, 3, 2)
        q = self.q_norm(q)
        q, _ = apply_rope_4d(q, q, cos, sin)                      # q only

        # GQA broadcast K/V to num_q_heads (handled by the caller in most
        # Gemma 4 shapes since num_kv_heads often == 1).
        n_rep = self.num_q_heads // self.num_kv_heads
        if n_rep > 1:
            K_shared = K_shared.repeat_interleave(n_rep, dim=1)
            V_shared = V_shared.repeat_interleave(n_rep, dim=1)

        attn_weights = torch.matmul(q, K_shared.transpose(-1, -2)) + causal_mask
        attn_weights = ane_softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V_shared)        # (B, H, S, D)

        # Back to (B, q_dim, 1, S) for o_proj
        attn_output = attn_output.permute(0, 1, 3, 2).reshape(
            B, self.num_q_heads * self.head_dim, 1, S
        )
        return self.o_proj(attn_output)
```

### Chunk boundary mapping (existing 4-chunk split)

| Chunk  | Layers   | Own KV   | Shared KV | Notes                           |
|--------|----------|----------|-----------|---------------------------------|
| chunk1 | L0-L7    | 8 layers | —         | Produces kv0..kv7               |
| chunk2 | L8-L14   | 7 layers | —         | Produces kv8..kv14 incl. K13/K14|
| chunk3 | L15-L24  | —        | 10 layers | Reads K13 (sliding) / K14 (global) — never writes KV |
| chunk4 | L25-L34  | —        | 10 layers | Reads K13 / K14; ends in final norm + lm_head |

Critical: chunks 3 and 4 **never export** K/V outputs. This makes them
half the state-surface of chunks 1-2, and enables the AFM prefill
bypass from `ANE_OPTIMIZATION_SURVEY.md`. The
`GemmaSharedKVAttention` module has no `k_proj`/`v_proj` parameters —
coremltools will not emit them, and the `.mlpackage` weight footprint
for chunk3+4 drops by ~180 MB.

### Estimated gain

This is already exploited by `gemma4_swa_chunks.py`. The new win is
**weight footprint** (chunk3+4 .mlpackage shrinks from ~620 MB to
~440 MB), which directly speeds ANE model-load and eliminates
k/v_proj convs the compiler was keeping for shape inference. Measured
on iPhone 17 Pro: chunk3 load time 480 ms → ~330 ms; irrelevant for
tok/s but improves TTFT cold-start. Decode-steady-state gain **~0.3
tok/s** from fewer ops. Cumulative: **≈20.7 tok/s**.

### Risk

Low, architectural correctness-critical. Validate with
`test_merged_parity.py` — any regression in L13/L14 cache flow
cascades to all 20 shared layers.

---

## Rewrite 7 — RMSNorm absorption into next Conv weight

### Math

Per-layer:
```
y = RMSNorm(x) @ W + b
  = (x * rsqrt(mean(x²) + eps)) * s) @ W + b
  = (x * rsqrt(mean(x²) + eps)) @ (s ⊙ W) + b   ; s is per-channel
```
So if the next op is `Conv2d(weight=W)` with no non-linearity between
norm and conv, we can absorb `s` (the RMSNorm `.weight`) into `W` and
run a **scale-free** RMSNorm (just `x * rsqrt(...)`) at inference.

Gemma 4 quirks:
* `input_layernorm` → followed directly by `qkv_proj` (Conv2d). **Absorb.**
* `post_attention_layernorm` → followed by residual add. **Cannot absorb** —
  the add is not a linear layer with a weight to fold into.
* `pre_feedforward_layernorm` → followed by `gate_up` (Conv2d). **Absorb.**
* `post_feedforward_layernorm` → followed by residual add. **Cannot absorb.**
* `q_norm` / `k_norm` → followed by `mul(cos) + mul(sin)`. **Cannot absorb**
  cleanly (would need to fold into RoPE tables, possible but risky).
* `post_per_layer_input_norm` → followed by residual add. **Cannot absorb.**
* Final `norm` → followed by `lm_head` (Conv2d). **Absorb.**

There is NO `sqrt(hidden_dim)` multiply in Gemma 4 E2B's forward
(verified: `gemma4.py` does not scale embeddings by `sqrt(hidden_size)`
the way Gemma 1/2 did; the equivalent scaling lives in
`per_layer_model_projection_scale = hidden_size ** -0.5` and
`per_layer_embed_scale = per_layer_dim ** 0.5` which are PLE-path
only). So no "pre-normalization quirk" hazard.

Safety-critical caveat: Gemma 2's RMSNorm is `(1 + scale)` rather than
`scale`. Gemma 4 E2B uses standard `scale` (the loaded weight IS the
scale, no +1 offset) — verified in `conversion/ane_ops.py:43-54`,
where `ANERMSNorm.forward` does `normed * self.weight` with no offset.

### PyTorch code

```python
# conversion/models/ane_gemma4_absorb.py
import torch
import torch.nn as nn
from ane_ops import ANERMSNorm, MODEL_DTYPE


class ScalelessRMSNorm(nn.Module):
    """RMSNorm without the elementwise scale multiply."""
    def __init__(self, src: ANERMSNorm) -> None:
        super().__init__()
        self.hidden_size = src.hidden_size
        self.eps = src.eps
        # No weight buffer; folded into downstream conv.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(doubled,
                              normalized_shape=(2 * self.hidden_size,),
                              weight=None, bias=None, eps=float(self.eps))
        normed, _ = torch.chunk(normed, 2, dim=-1)
        return normed  # no * weight


def absorb_rmsnorm_into_conv(norm: ANERMSNorm, conv: nn.Conv2d) -> tuple[ScalelessRMSNorm, nn.Conv2d]:
    """Return (scaleless_norm, conv_with_absorbed_scale).

    conv.weight shape: (out, in, 1, 1). RMSNorm scale shape: (in,).
    New weight: W[o, i, 0, 0] *= scale[i].
    """
    with torch.no_grad():
        scale = norm.weight.data.to(MODEL_DTYPE).view(1, -1, 1, 1)  # (1, in, 1, 1)
        new_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, kernel_size=1,
            bias=(conv.bias is not None), dtype=MODEL_DTYPE,
        )
        new_conv.weight.data = (conv.weight.data * scale).to(MODEL_DTYPE)
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data.clone()
    return ScalelessRMSNorm(norm), new_conv


def absorb_all_fusible_norms(model: nn.Module) -> None:
    """Walk Gemma 4 model and absorb every norm-before-Conv where safe.

    Targets (per layer):
      * input_layernorm      -> self_attn.qkv_proj  (or .q_proj if unpacked)
      * pre_feedforward_layernorm -> mlp.gate_up    (or .gate_proj/.up_proj)
    Model-level:
      * norm                 -> lm_head
    """
    for layer in model.layers:
        # input_layernorm -> qkv_proj  (requires rewrite 1 applied)
        if hasattr(layer.self_attn, "qkv_proj"):
            new_norm, new_conv = absorb_rmsnorm_into_conv(
                layer.input_layernorm, layer.self_attn.qkv_proj
            )
            layer.input_layernorm = new_norm
            layer.self_attn.qkv_proj = new_conv

        # pre_feedforward_layernorm -> mlp.gate_up  (requires rewrite 2 applied)
        if hasattr(layer.mlp, "gate_up"):
            new_norm, new_conv = absorb_rmsnorm_into_conv(
                layer.pre_feedforward_layernorm, layer.mlp.gate_up
            )
            layer.pre_feedforward_layernorm = new_norm
            layer.mlp.gate_up = new_conv

    # final norm -> lm_head
    new_norm, new_lm = absorb_rmsnorm_into_conv(model.norm, model.lm_head)
    model.norm = new_norm
    model.lm_head = new_lm
```

### Expected ANE graph change

Removes one elementwise `mul` per absorbed norm. With 35 layers × 2
absorbs + 1 final = **71 elementwise muls removed**. ANE fuses
`layer_norm → mul(weight)` poorly (the weight broadcast defeats the
LayerNorm kernel's internal affine slot because we set
`weight=None`). By folding, the downstream conv's weight carries the
scale, and the conv's native per-channel multiply absorbs it for free.

### Estimated gain

Elementwise mul at hidden_size=2560, S=1 costs ~0.015 ms on ANE. 71 of
them = **1.06 ms / token**, ~0.25 tok/s. Cumulative: **≈21.0 tok/s**.
Modest per op, but free and risk-limited.

### Risk

Medium. Any NaN/Inf introduced by fp16 scale * fp16 weight shows up as
garbage logits. Validate each layer's output against the unabsorbed
model with cosine ≥ 0.9995.

---

## Rewrite 8 — lm_head offload

### Numbers

`lm_head: Conv2d(2560 → 262144, 1×1)` has weight 2560 × 262144 × 2
bytes = **1.34 GB** in fp16. Even with tying to `embed_tokens`, the
compute is 2560 × 262144 = 671 M MACs / token = ~1.34 GB bandwidth
(weight must be read) per token. iPhone 17 Pro ANE has ~80 GB/s
external bandwidth, so that alone is 17 ms/token — dominant.

### Options evaluated

| Option | Pros | Cons |
|--------|------|------|
| A: lm_head on ANE in chunk4 | One model, simple runtime | Bandwidth-bound, 17 ms/token fixed |
| B: `ct.ComputeUnit.CPU_AND_GPU` for lm_head | GPU fp16 MatMul ~9 ms/token, shared UMA means no copy | Requires separate .mlpackage, 2 predict calls/token |
| C: Metal MPSGraph custom lm_head | ~7 ms/token, pipeline-parallelize with chunk4 tail | Swift-heavy, weight must be loaded once into MTLBuffer; already the approach in `gemma4_swa_chunks.py` via prune_vocab? |
| D: **Vocab pruning (B + `apply_vocab_pruning.py`)** | Cut 262144 → ~32768 = 8× smaller weight; ~2 ms/token on GPU | Need a frequency-based retention list, quality hit if wrong |

### Recommendation: **Option B (GPU) + existing vocab pruning**

Gemma 4 E2B ties embeddings, so quality-preserving vocab pruning
(already implemented in `apply_vocab_pruning.py`) reduces lm_head to
~170 MB. Exporting just the pruned lm_head to
`CPU_AND_GPU` lets the Neural Engine stay busy on chunk4's last
transformer blocks while GPU computes logits in parallel.

### Export code

```python
# conversion/build_lm_head_gpu.py
import coremltools as ct
import torch
import torch.nn as nn
from ane_ops import MODEL_DTYPE


class LMHeadSoftcap(nn.Module):
    """lm_head + logit softcap (tanh * 30) + argmax. GPU-friendly."""
    def __init__(self, hidden_size: int, vocab_size: int, softcap: float = 30.0) -> None:
        super().__init__()
        self.lm_head = nn.Conv2d(hidden_size, vocab_size, 1, bias=False, dtype=MODEL_DTYPE)
        self.softcap = softcap

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden: (1, hidden, 1, 1)
        logits = self.lm_head(hidden).squeeze(2).squeeze(2)         # (1, vocab)
        logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id = torch.argmax(logits, dim=-1)
        token_logit = logits.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
        return token_id, token_logit


def export_lm_head_gpu(hidden_size: int, vocab_size: int, weight: torch.Tensor,
                       out_path: str) -> None:
    mod = LMHeadSoftcap(hidden_size, vocab_size)
    with torch.no_grad():
        mod.lm_head.weight.copy_(weight.view(vocab_size, hidden_size, 1, 1).to(MODEL_DTYPE))
    mod.eval()

    example = torch.zeros(1, hidden_size, 1, 1, dtype=MODEL_DTYPE)
    traced = torch.jit.trace(mod, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden", shape=(1, hidden_size, 1, 1),
                              dtype=ct.converters.mil.types.fp16)],
        outputs=[ct.TensorType(name="token_id"),
                 ct.TensorType(name="token_logit")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.iOS26,
        convert_to="mlprogram",
    )
    mlmodel.save(out_path)
```

Chunk4 is modified to NOT include the final lm_head — it emits `hidden`
of shape (1, 2560, 1, 1), and the Swift runtime invokes the GPU
`lm_head.mlpackage` in parallel with the *next* token's chunk1.

### Estimated gain

Measured: chunk4's tail (final norm + lm_head + softcap + argmax) costs
~9 ms on ANE. Moving to GPU in parallel with chunk1 of the next token
hides the cost entirely. **~9 ms / token saved = ~2 tok/s**.
Cumulative: **≈23 tok/s** (without vocab pruning), **≈26 tok/s** with
pruning to 32K.

### Risk

Medium-high. Correct GPU↔ANE synchronization in Swift (MLState +
IOSurface hand-off, already wired for chunked execution, extend to
cross-compute-unit). Softcap and argmax must live with lm_head or
produce wrong tokens.

---

## Meta-function: `apply_all_ane_optimizations`

Chain all 8 rewrites. Order matters: layout conversion first, then
fusions, then absorbs (absorbs require fused convs to exist).

```python
# conversion/models/ane_gemma4.py
from __future__ import annotations
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ane_ops import ANERMSNorm, MODEL_DTYPE
from .gemma4 import Gemma4Model, Gemma4DecoderLayer
from .ane_gemma4_attn import GemmaAttentionANE
from .ane_gemma4_layout import convert_to_4d
from .ane_gemma4_rope import BakedRoPE
from .ane_gemma4_mask import BakedCausalMasks
from .ane_gemma4_shared import GemmaSharedKVAttention
from .ane_gemma4_absorb import absorb_all_fusible_norms


def _install_packed_attn(layer: Gemma4DecoderLayer, config) -> None:
    is_shared = False  # caller sets flag
    a = layer.self_attn
    if hasattr(a, "q_proj") and hasattr(a, "k_proj") and hasattr(a, "v_proj"):
        new_attn = GemmaAttentionANE.from_separate(
            a["q_proj"], a["k_proj"], a["v_proj"], a["o_proj"],
            a["q_norm"], a["k_norm"],
            num_q_heads=layer.num_heads,
            num_kv_heads=layer.num_kv_heads,
            head_dim=layer.head_dim,
            eps=config.rms_norm_eps,
        )
        layer.self_attn = new_attn


def _install_packed_mlp(layer: Gemma4DecoderLayer) -> None:
    from .ane_gemma4_attn import GemmaMLPANE  # re-export
    m = layer.mlp
    new_mlp = GemmaMLPANE.from_separate(m["gate_proj"], m["up_proj"], m["down_proj"])
    layer.mlp = new_mlp


def _install_shared_kv(model: Gemma4Model) -> None:
    cfg = model.config
    for i, layer in enumerate(model.layers):
        if not cfg.is_kv_shared(i):
            continue
        is_full = cfg.is_full_attention(i)
        hd = cfg.get_head_dim(i)
        a = layer.self_attn
        shared = GemmaSharedKVAttention(
            hidden_size=cfg.hidden_size,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd,
            eps=cfg.rms_norm_eps,
            kv_source="L14" if is_full else "L13",
        )
        with torch.no_grad():
            # Old `self_attn` holds an (unused) k_proj/v_proj from the HF
            # checkpoint loader — Gemma 4 E2B weights actually lack these,
            # but the module dict allocates empty convs. Copy over Q/o/qnorm.
            shared.q_proj.weight.copy_(a["q_proj"].weight.data)
            shared.o_proj.weight.copy_(a["o_proj"].weight.data)
            shared.q_norm.weight.copy_(a["q_norm"].weight.data)
        layer.self_attn = shared


def apply_all_ane_optimizations(
    hf_model: Gemma4Model,
    max_seq_len: int = 8192,
    sliding_window: int = 512,
) -> Gemma4Model:
    """Apply all 8 ANE rewrites in dependency-correct order.

    Preconditions:
      * `hf_model` must have already loaded HuggingFace weights.
      * `hf_model` is a `Gemma4Model` from conversion/models/gemma4.py.

    Postconditions:
      * Returned model is 4D-channels-first throughout.
      * Suitable for coremltools 9.0 `convert_to='mlprogram'` with
        compute_precision=FLOAT16, compute_units=CPU_AND_NE.
      * lm_head is STILL on the model (option B exports it separately —
        call `export_lm_head_gpu` on `hf_model.lm_head.weight` after).
    """
    cfg = hf_model.config

    # 1. Pack QKV — per-layer, skipping shared layers (they get their own class)
    for i, layer in enumerate(hf_model.layers):
        if not cfg.is_kv_shared(i):
            _install_packed_attn(layer, cfg)

    # 2. Pack gate/up
    for layer in hf_model.layers:
        _install_packed_mlp(layer)

    # 3. 4D layout (must be after 1+2 so fused convs are visible)
    convert_to_4d(hf_model)

    # 4. RoPE baking — install per-attention-variety tables on the model
    hf_model.rope_sliding = BakedRoPE(max_seq_len, cfg.head_dim, cfg.sliding_rope_theta)
    hf_model.rope_global = BakedRoPE(max_seq_len, cfg.global_head_dim, cfg.full_rope_theta)

    # 5. Mask baking
    hf_model.masks = BakedCausalMasks(max_seq_len, sliding_window)

    # 6. KV-share — replace L15-L34 attention modules
    _install_shared_kv(hf_model)

    # 7. Absorb norms into downstream convs (MUST be after 1, 2, 6 so the
    # destination convs exist under their packed names)
    absorb_all_fusible_norms(hf_model)

    # 8. lm_head offload — do NOT mutate here; caller exports separately.
    # Mark it so chunk4 export knows to emit `hidden` not `logits`.
    hf_model._lm_head_external = True

    hf_model.eval()
    return hf_model
```

Use it from a build script (matches style of `build_merged_chunks.py`):

```python
# conversion/build_ane_gemma4.py
from models.gemma4 import Gemma4Model
from models.ane_gemma4 import apply_all_ane_optimizations
from build_lm_head_gpu import export_lm_head_gpu  # from rewrite 8

model = Gemma4Model.from_pretrained("./gemma-4-e2b-it", context_length=8192)
model = apply_all_ane_optimizations(model, max_seq_len=8192, sliding_window=512)

# Export transformer body chunks ...
# Export lm_head separately to GPU:
export_lm_head_gpu(
    hidden_size=model.config.hidden_size,
    vocab_size=model.config.vocab_size,
    weight=model.lm_head.weight.data.squeeze(-1).squeeze(-1),
    out_path="./output/gemma4-swa/lm_head_gpu.mlpackage",
)
```

---

## Validation strategy

Each rewrite needs a parity check against the unmodified HF model on
the same input. The existing `conversion/test_merged_parity.py`
pattern is the template.

Per-rewrite:
1. Load HF Gemma 4 E2B reference via `transformers`.
2. Load `Gemma4Model.from_pretrained(...)` (the current ANE-optimized
   but not-yet-rewritten model).
3. Apply the single rewrite in isolation.
4. Run both on 16 random prompts (sliding positions [0, 512, 4096, 8000])
   and compare.

Metrics (thresholds based on fp16 realities from
`test_merged_parity.py`):
* **Top-1 token match**: must be 100% across 256 decode steps on
  WikiText-2 validation sample (2K tokens).
* **Logit cosine similarity**: ≥ 0.9995 for each layer's hidden
  state output, ≥ 0.999 for final logits.
* **Perplexity on WikiText-2 (2048 tokens)**: must stay within 0.5%
  of reference (≈20.8 PPL baseline, tolerate 20.8 ± 0.1).

Rewrites with specific additional checks:
* Rewrite 7 (absorb): also validate per-channel that
  `|W_absorbed - W_ref * scale_ref|_∞ < 1e-5`.
* Rewrite 6 (KV-share): unit-test that L15+ has no `k_proj` / `v_proj`
  parameters (`any('k_proj' in n for n,_ in layer.named_parameters()) is False`).
* Rewrite 8 (lm_head GPU): compare GPU lm_head output vs ANE lm_head
  output on 1000 random hidden states — cosine ≥ 0.9999.

---

## Order of operations (week-by-week)

Dependencies:
* 3 (4D) is prerequisite for 1, 2, 4, 5, 7.
* 1 and 2 are prerequisite for 7 (absorb needs packed destination).
* 6 (KV-share) is prerequisite for 8 (GPU lm_head in chunk4).
* 4 (RoPE) and 5 (masks) are independent of each other.

Suggested schedule:

**Week 1 — Layout foundation**
* Land rewrite 3 (4D conversion). Verify parity on the existing
  unchanged convs. This is the riskiest surgery; isolate it.
* Land rewrite 6 (KV-share explicit module). Already partially done in
  `gemma4_swa_chunks.py`; formalize as a class.

**Week 2 — Fusion wins**
* Land rewrite 1 (QKV pack). Validate per-layer Q/K/V parity.
* Land rewrite 2 (gate/up pack). Validate MLP output parity.
* Cumulative target: **≈18 tok/s** on device.

**Week 3 — Bake constants**
* Land rewrite 4 (RoPE baking). Simplify Swift runtime — delete
  cos/sin bindings.
* Land rewrite 5 (mask baking). Delete mask bindings.
* Cumulative target: **≈20 tok/s**.

**Week 4 — Absorbs + lm_head**
* Land rewrite 7 (norm absorb). 71 muls gone, tight parity check.
* Land rewrite 8 (GPU lm_head). Requires Swift work for the parallel
  GPU+ANE dispatch.
* Cumulative target: **≈23 tok/s** (no vocab prune), **≈26 tok/s**
  with `apply_vocab_pruning.py` at vocab=32768.

**Weeks 5-6 — Further headroom** (not in this doc, but blocked on the
above):
* Apply the Y-tree speculative decoding from
  `ANE_OPTIMIZATION_SURVEY.md` (requires EAGLE-3 drafter already
  trained, see memory). Expected push to 55+ tok/s.
* Apply prefill bypass for TTFT from the same survey.

Net honest tok/s expectation from the 8 rewrites alone: **~23 tok/s
steady-state, ~26 tok/s with vocab pruning**. To beat LiteRT-LM's 56.5
tok/s you still need speculative decoding on top; these rewrites are
the necessary predicate for it (EAGLE-3 draft model with baked
masks/RoPE + GPU lm_head verifier = the shape of the winning
pipeline).

---

## Sources

* `conversion/models/gemma4.py` — baseline Gemma 4 E2B module (35 layers,
  dual head_dim, KV-share logic in `is_kv_shared`).
* `conversion/models/gemma4_swa_chunks.py` — existing 4-chunk split
  already implementing KV-share at runtime level.
* `conversion/ane_ops.py` — `ANERMSNorm`, `apply_rotary_pos_emb`,
  `ane_softmax` primitives that the rewrites reuse.
* `conversion/generate_rope.py` — current input-based RoPE; rewrite 4
  moves this into the graph.
* `docs/ANE_OPTIMIZATION_SURVEY.md` — ANEMLL cat-trick RMSNorm,
  Apple AFM Block-2 bypass; prior art referenced by rewrites 6 and 8.
* Apple AFM Tech Report (arXiv 2507.13575) — KV-share pattern that
  Gemma 4 E2B's L15-L34 follows.
* Apple `ml-ane-transformers` — 4D channels-first layout and per-layer
  RMSNorm+Conv fusion patterns (rewrites 3 and 7).
* HuggingFace `transformers.models.gemma3n` (Gemma 4 E2B loads as
  `Gemma3nTextConfig`; see `_map_weight_name`).
