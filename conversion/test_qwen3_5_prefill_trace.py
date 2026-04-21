"""Phase 2b: try to convert a chunked prefill Gated DeltaNet layer to CoreML.

Wraps the HF torch_chunk_gated_delta_rule algorithm in a pure nn.Module whose
in-place mutations are rewritten for torch.jit.trace compatibility, then:
  (1) verifies parity vs HF at seq=64, 128, 2048
  (2) converts to CoreML at seq=64 (single chunk, no outer loop)
  (3) audits ANE placement

Parity target: cos >= 0.999 vs HF.
Placement target: note which ops stay on ANE vs fall to CPU (cumsum, etc).
"""
from collections import Counter
from pathlib import Path
import math
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, AutoTokenizer

import coremltools as ct


MODEL_ID = "Qwen/Qwen3.5-0.8B"
CHUNK_SIZE = 64


def _linear_as_conv2d(x, weight, bias=None):
    """ANE-friendly Linear equivalent. Apple's ml-ane-transformers layout
    (as used in conversion/ane_ops.py Conv2dLinear): reshape to
    (B, C, 1, T) with sequence on the last dim so the ANE conv kernel
    can process the full tensor efficiently, with fp16 accumulation that
    is more precise than the matmul-backed nn.Linear path.

    x      : (B, T, in)
    weight : (out, in)       — same layout as nn.Linear.weight
    bias   : (out,) or None
    returns: (B, T, out)
    """
    # (B, T, C) -> (B, C, 1, T)
    x4 = x.permute(0, 2, 1).unsqueeze(2)
    # (out, in) -> (out, in, 1, 1)
    w4 = weight.unsqueeze(-1).unsqueeze(-1)
    y4 = F.conv2d(x4, w4, bias=bias)
    # (B, out, 1, T) -> (B, T, out)
    return y4.squeeze(2).permute(0, 2, 1)


def _ane_rmsnorm(x, weight, eps: float):
    """ANE-friendly RMSNorm. Uses the [x, -x] concat + LayerNorm identity
    (cat has zero mean, so LayerNorm ≡ RMSNorm). ANE has a highly-optimized
    LayerNorm kernel; it has no rsqrt kernel. Same trick Gemma 4 uses.

    x      : (..., hidden)
    weight : (hidden,)  — multiplied after the norm
    eps    : RMSNorm epsilon
    """
    hidden = weight.shape[0]
    doubled = torch.cat([x, -x], dim=-1)
    normed = F.layer_norm(
        doubled, normalized_shape=(2 * hidden,),
        weight=None, bias=None, eps=float(eps),
    )
    normed, _ = torch.chunk(normed, 2, dim=-1)
    return normed * weight


def _ane_softmax(x, dim: int = -1):
    """Numerically-stable softmax using only ANE-friendly primitives, with
    explicit fp16 casts to prevent torch.exp auto-upcasting to fp32."""
    dt = x.dtype
    x = x.to(dt)
    x_max = x.max(dim=dim, keepdim=True).values.to(dt)
    exp_x = (x - x_max).exp().to(dt)
    return (exp_x / exp_x.sum(dim=dim, keepdim=True)).to(dt)


class PrefillLinearAttnLayer(nn.Module):
    """One linear_attention layer forward for a fixed seq_len divisible by 64.

    Mirrors the prefill branch of Qwen3_5GatedDeltaNet.forward:
        in_proj (four separate) → conv1d (depthwise + SiLU) → chunked
        Gated-Delta-Rule → RMSNormGated(with z gate) → out_proj.

    The chunked algorithm is inlined here (a trace-friendly rewrite of
    torch_chunk_gated_delta_rule). All Python `for` loops are over
    compile-time-known ranges (chunk_size and num_chunks), so torch.jit.trace
    unrolls them.
    """

    def __init__(self, cfg, hf_layer, seq_len: int):
        super().__init__()
        assert seq_len % CHUNK_SIZE == 0, "seq_len must be divisible by chunk_size=64"
        self.S = seq_len
        self.num_chunks = seq_len // CHUNK_SIZE
        self.CS = CHUNK_SIZE
        self.num_v = cfg.linear_num_value_heads
        self.num_k = cfg.linear_num_key_heads
        self.Dk = cfg.linear_key_head_dim
        self.Dv = cfg.linear_value_head_dim
        self.key_dim = self.Dk * self.num_k
        self.value_dim = self.Dv * self.num_v
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.K = cfg.linear_conv_kernel_dim
        self.eps = cfg.rms_norm_eps
        self.v_per_k = self.num_v // self.num_k  # 1 for 0.8B

        lin = hf_layer.linear_attn
        self.conv_w = nn.Parameter(lin.conv1d.weight.detach().clone(), requires_grad=False)     # (C,1,K)
        self.in_proj_qkv_w = nn.Parameter(lin.in_proj_qkv.weight.detach().clone(), requires_grad=False)
        self.in_proj_z_w   = nn.Parameter(lin.in_proj_z.weight.detach().clone(),   requires_grad=False)
        self.in_proj_b_w   = nn.Parameter(lin.in_proj_b.weight.detach().clone(),   requires_grad=False)
        self.in_proj_a_w   = nn.Parameter(lin.in_proj_a.weight.detach().clone(),   requires_grad=False)
        self.out_proj_w    = nn.Parameter(lin.out_proj.weight.detach().clone(),    requires_grad=False)
        self.dt_bias = nn.Parameter(lin.dt_bias.detach().clone(), requires_grad=False)   # (Hv,)
        self.A_log   = nn.Parameter(lin.A_log.detach().clone(),   requires_grad=False)   # (Hv,)
        self.norm_w  = nn.Parameter(lin.norm.weight.detach().clone(), requires_grad=False)  # (Dv,)

        # Pre-materialize the causal eye used inside the chunk algorithm.
        self.register_buffer(
            "chunk_eye",
            torch.eye(CHUNK_SIZE, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _l2norm(x, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)

    def _rmsnorm_gated(self, x, z):
        # ANE-friendly RMSNorm with z-gate post-multiply. The learned norm_w
        # is NOT a (1 + w) like the decoder-layer norms — it's a plain scale
        # (Qwen3_5RMSNormGated.weight initialised at 1), so pass it straight.
        x = _ane_rmsnorm(x, self.norm_w, self.eps)
        return x * F.silu(z)

    def _chunk_gated_delta_rule(self, q, k, v, g, beta):
        """Trace-friendly rewrite of HF torch_chunk_gated_delta_rule.

        Inputs (already transposed to (B, H, S, D) and fp32):
            q, k: (B, num_v, S, Dk)
            v:    (B, num_v, S, Dv)
            g:    (B, num_v, S)
            beta: (B, num_v, S)

        Returns:
            core_attn_out: (B, num_v, S, Dv) in fp32
            last_state:    (B, num_v, Dk, Dv) in fp32
        """
        B = 1
        H = self.num_v
        S = self.S
        CS = self.CS
        NC = self.num_chunks
        Dk = self.Dk
        Dv = self.Dv

        scale = 1.0 / math.sqrt(Dk)
        q = q * scale

        v_beta = v * beta.unsqueeze(-1)                             # (B,H,S,Dv)
        k_beta = k * beta.unsqueeze(-1)                             # (B,H,S,Dk)

        # Reshape to (B, H, NC, CS, D)
        q = q.reshape(B, H, NC, CS, Dk)
        k = k.reshape(B, H, NC, CS, Dk)
        v = v.reshape(B, H, NC, CS, Dv)
        k_beta = k_beta.reshape(B, H, NC, CS, Dk)
        v_beta = v_beta.reshape(B, H, NC, CS, Dv)
        g = g.reshape(B, H, NC, CS)

        # Per-chunk cumulative g (the only cumsum — over CS=64)
        g = g.cumsum(dim=-1)                                        # (B,H,NC,CS)

        # decay_mask[i,j] = exp(g_i - g_j) for i>=j within a chunk, else 0.
        # Clamp *before* exp to keep the (i<j) upper-triangular portion from
        # producing Inf in fp16 (cumsum over CS=64 can push g_diff to ~+200
        # up there, which overflows exp in fp16 and poisons later bmm with
        # NaN even though tril_mask zeros the visible entries). Clamp value
        # is safely above exp-max for the (i>=j) cells we actually use.
        g_diff = g.unsqueeze(-1) - g.unsqueeze(-2)                  # (B,H,NC,CS,CS)
        g_diff = g_diff.clamp(max=10.0)
        tril_mask = torch.tril(torch.ones(CS, CS, dtype=torch.bool, device=q.device))
        decay_mask = torch.where(tril_mask, g_diff.exp(), torch.zeros_like(g_diff))

        # L = strict-lower part of -(k_beta @ k^T) * decay_mask.
        # HF initializes attn0 like this, then runs a 63-iter in-place loop that
        # implements Gauss elimination to compute (I - L_init)^{-1} - I. We
        # replace that with a numerically stable Neumann iteration:
        # T_{k+1} = I + L @ T_k, converging exactly at k=CS-1 because L is
        # strictly lower and therefore nilpotent (L^CS = 0).
        #
        # All matmul ops here run as 3D torch.bmm on (B*H*NC, CS, D). ANE prefers
        # rank<=4 and bmm maps cleanly; 5D (B,H,NC,CS,CS) matmul tends to CPU.
        BN = B * H * NC
        upper_incl_diag = torch.triu(torch.ones(CS, CS, dtype=torch.bool, device=q.device), diagonal=0)
        k3 = k.reshape(BN, CS, Dk)
        k_beta3 = k_beta.reshape(BN, CS, Dk)
        v_beta3 = v_beta.reshape(BN, CS, Dv)
        L3 = -torch.bmm(k_beta3, k3.transpose(-1, -2)) * decay_mask.reshape(BN, CS, CS)
        L3 = torch.where(upper_incl_diag, torch.zeros_like(L3), L3)

        # Neumann iteration with fp16-safe clamp: T_{k+1} = clamp(I + L @ T_k).
        # L is strictly lower triangular so L^CS = 0 and the series terminates
        # at k=CS-1 with the exact (I - L)^{-1}. Raw Neumann has intermediate
        # T_k peaks of 10^10-10^16 before collapsing back to <20 (the entries
        # of the true inverse), overflowing fp16 and poisoning downstream ops
        # with NaN on CPU / silent drift on ANE. Clamp to ±1e3 bounds the
        # transient excursion to fp16-safe magnitudes; the math converges
        # because the final value is well within the clamp range.
        #
        # Earlier we used row-by-row forward substitution here, which is also
        # mathematically safe, but it produces a cumulative-concat chain of
        # 64 rows × 18 linear_attention layers (~1150 concat ops plus ~2400
        # slice ops). iPhone A18 Pro Core ML CPU runtime mis-handles this
        # pattern on iOS 26.1 and returns garbage logits (cos≈0.3, top-1 0%)
        # even though Mac Core ML on the same mlmodelc gives cos≈1.0. The
        # clamp-Neumann form stays at ~9900 total ops with no cumulative
        # concat and works on both platforms.
        eye = self.chunk_eye                                        # (CS, CS)
        attn3 = eye.expand(BN, CS, CS).contiguous()
        for _ in range(CS):
            attn3 = eye + torch.bmm(L3, attn3)
            attn3 = attn3.clamp(-1e3, 1e3)
        attn = attn3.reshape(B, H, NC, CS, CS)

        # WY products (kept as 3D bmm to stay on ANE)
        value = torch.bmm(attn3, v_beta3).reshape(B, H, NC, CS, Dv)
        k_decay = (k_beta * g.exp().unsqueeze(-1)).reshape(BN, CS, Dk)
        k_cumdecay = torch.bmm(attn3, k_decay).reshape(B, H, NC, CS, Dk)

        # Outer loop over chunks (unrolled)
        last_state = torch.zeros(B, H, Dk, Dv, device=q.device, dtype=q.dtype)
        strict_upper = torch.triu(torch.ones(CS, CS, dtype=torch.bool, device=q.device), diagonal=1)

        core_chunks = []
        for i in range(NC):
            q_i = q[:, :, i]                                        # (B,H,CS,Dk)
            k_i = k[:, :, i]
            v_i = value[:, :, i]                                    # (B,H,CS,Dv)
            attn_i = (q_i @ k_i.transpose(-1, -2)) * decay_mask[:, :, i]  # (B,H,CS,CS)
            attn_i = torch.where(strict_upper, torch.zeros_like(attn_i), attn_i)
            v_prime = k_cumdecay[:, :, i] @ last_state              # (B,H,CS,Dv)
            v_new = v_i - v_prime
            g_i = g[:, :, i]                                        # (B,H,CS)
            attn_inter = (q_i * g_i.unsqueeze(-1).exp()) @ last_state  # (B,H,CS,Dv)
            out_i = attn_inter + attn_i @ v_new                     # (B,H,CS,Dv)
            core_chunks.append(out_i)

            g_last = g_i[..., -1:].unsqueeze(-1)                    # (B,H,1,1)
            decay_k = (g_last.squeeze(-1) - g_i).exp().unsqueeze(-1)  # (B,H,CS,1)
            last_state = last_state * g_last.exp() + (k_i * decay_k).transpose(-1, -2) @ v_new

        core_attn_out = torch.stack(core_chunks, dim=2)             # (B,H,NC,CS,Dv)
        core_attn_out = core_attn_out.reshape(B, H, S, Dv)
        return core_attn_out, last_state

    def forward(self, hidden_in):
        """hidden_in: (1, S, H)  →  hidden_out: (1, S, H)."""
        # 1. Four projections — Conv2d 1x1 for ANE fp16-accumulation precision
        mixed_qkv = _linear_as_conv2d(hidden_in, self.in_proj_qkv_w)
        z = _linear_as_conv2d(hidden_in, self.in_proj_z_w)
        b = _linear_as_conv2d(hidden_in, self.in_proj_b_w)
        a = _linear_as_conv2d(hidden_in, self.in_proj_a_w)

        # 2. depthwise conv1d on (B, C, S) with causal left-padding.
        mixed_qkv_t = mixed_qkv.transpose(1, 2)                      # (1,C,S)
        conv_out = F.conv1d(mixed_qkv_t, self.conv_w, bias=None,
                            groups=self.conv_dim, padding=self.K - 1)
        conv_out = F.silu(conv_out[:, :, :self.S])                    # (1,C,S)
        mixed_qkv = conv_out.transpose(1, 2)                          # (1,S,C)

        # 3. Split
        q, k, v = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        q = q.reshape(1, self.S, self.num_k, self.Dk)
        k = k.reshape(1, self.S, self.num_k, self.Dk)
        v = v.reshape(1, self.S, self.num_v, self.Dv)

        # 4. beta, g (fp32)
        beta = torch.sigmoid(b).float()                               # (1,S,num_v)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # (1,S,num_v)

        # 5. No repeat_interleave (v_per_k=1 for 0.8B)

        # 6. Chunked rule (fp32 internally). Convert layouts to (B,H,S,D).
        q_ = self._l2norm(q.float().transpose(1, 2).contiguous())     # (1,num_v,S,Dk)
        k_ = self._l2norm(k.float().transpose(1, 2).contiguous())
        v_ = v.float().transpose(1, 2).contiguous()                   # (1,num_v,S,Dv)
        beta_ = beta.transpose(1, 2).contiguous()                     # (1,num_v,S)
        g_ = g.transpose(1, 2).contiguous()                           # (1,num_v,S)

        core_out, _ = self._chunk_gated_delta_rule(q_, k_, v_, g_, beta_)
        # core_out (1,num_v,S,Dv) → (1,S,num_v,Dv) → (1,S,V)
        core_out = core_out.transpose(1, 2).contiguous().to(hidden_in.dtype)  # (1,S,num_v,Dv)

        # 7. Gated RMSNorm per Dv vector + out_proj
        core_flat = core_out.reshape(-1, self.Dv)                     # (S*num_v, Dv)
        z_flat = z.reshape(-1, self.Dv)
        out_flat = self._rmsnorm_gated(core_flat, z_flat)
        out = out_flat.reshape(1, self.S, self.value_dim)             # (1,S,V)
        hidden_out = _linear_as_conv2d(out, self.out_proj_w)           # (1,S,H)
        return hidden_out


def extract_hf_prefill_io(hf_model, seq_len: int):
    """Forward the HF model on a fixed prompt of length seq_len (padded if
    shorter), capture layer-0 linear_attn in/out."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # Use a fixed text then pad with eos to hit exact seq_len
    text = "The capital of France is Paris. The capital of Japan is Tokyo. " \
           "Machine learning systems benefit from on-device inference."
    ids = tok(text, return_tensors="pt").input_ids
    if ids.shape[1] < seq_len:
        pad_id = tok.eos_token_id or 0
        pad = torch.full((1, seq_len - ids.shape[1]), pad_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    else:
        ids = ids[:, :seq_len]

    captured = {}
    lin = hf_model.model.layers[0].linear_attn

    def pre_hook(module, args, kwargs):
        h = args[0] if len(args) > 0 else kwargs["hidden_states"]
        captured["input"] = h.detach().clone()

    def post_hook(module, args, kwargs, output):
        out = output if not isinstance(output, tuple) else output[0]
        captured["output"] = out.detach().clone()

    h1 = lin.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = lin.register_forward_hook(post_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            hf_model(input_ids=ids, use_cache=False)
    finally:
        h1.remove(); h2.remove()
    return ids, captured["input"], captured["output"]


def cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def parity_check(hf_model, cfg, seq_lens):
    print("\n=== parity check ===")
    for seq_len in seq_lens:
        ids, hin, hout_ref = extract_hf_prefill_io(hf_model, seq_len)
        mod = PrefillLinearAttnLayer(cfg, hf_model.model.layers[0], seq_len).eval().float()
        with torch.no_grad():
            hout = mod(hin.float())
        c = cos(hout.float(), hout_ref.float())
        print(f"  seq={seq_len:4d}  cos={c:.6f}  "
              f"shapes: hin={tuple(hin.shape)} hout={tuple(hout.shape)}")


def convert_and_audit(hf_model, cfg, seq_len=64):
    print(f"\n=== CoreML conversion (seq={seq_len}) ===")
    mod = PrefillLinearAttnLayer(cfg, hf_model.model.layers[0], seq_len).eval().float()
    example = (torch.randn(1, seq_len, cfg.hidden_size),)
    traced = torch.jit.trace(mod, example, strict=False)
    print("  trace OK")

    ct_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="hidden_in", shape=example[0].shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="hidden_out", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print("  convert OK")

    tmpdir = Path(tempfile.mkdtemp(prefix="qwen35_prefill_"))
    path = tmpdir / f"prefill_seq{seq_len}.mlpackage"
    ct_model.save(str(path))
    size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved {path} ({size_mb:.1f} MB)")

    print("=== placement audit ===")
    reloaded = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    dev_counts: Counter = Counter()
    op_type_by_dev: dict[str, Counter] = {}
    for func_name, func in program.functions.items():
        for op in func.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = a.preferred_compute_device.__class__.__name__ if a else "unknown"
            dev_counts[dev] += 1
            op_type_by_dev.setdefault(dev, Counter())[op.operator_name] += 1
    total = sum(dev_counts.values())
    print(f"  total ops: {total}")
    for dev, n in dev_counts.most_common():
        print(f"    {dev}: {n} ({100*n/total:.1f}%)")
    for dev, c in op_type_by_dev.items():
        if dev == "unknown":
            continue
        print(f"  === {dev} op types (top 20) ===")
        for op_type, n in c.most_common(20):
            print(f"    {op_type}: {n}")


def main():
    print("loading HF model fp32...")
    t0 = time.time()
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    parity_check(model, cfg, seq_lens=[64, 128, 256])

    for seq_len in [64, 128, 256, 512, 2048]:
        convert_and_audit(model, cfg, seq_len=seq_len)


if __name__ == "__main__":
    main()
