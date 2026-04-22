"""INT4-palettize Qwen3.5-0.8B decode.

Shrinks the decode mlpackage bundle from 1.4 GB (fp16) to ~500 MB.
Uses coremltools' post-training weight palettization on a scalar
(per-tensor) 4-bit quantization. Only Linear / Conv2d / MatMul weights
are palettized; activations and states stay fp16.

Risk notes:
- Gated DeltaNet's state recurrence is fp16-sensitive. Adding INT4
  weight quant may worsen argmax-fragility on ANE.
- Qwen's 248K vocab and the linear-attention MLP projections are the
  largest tensors; quantizing them dominates bundle size.
- If parity drops below "oracle top-1 in top-3 = 100%" on the 10-prompt
  oracle, don't ship — fp16 baseline stands.
"""
import argparse, time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)

from test_qwen3_5_full_decode_trace import make_zero_states, cos_sim
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

MODEL_ID = "Qwen/Qwen3.5-0.8B"
ORACLE = Path(__file__).parent / "qwen3_5_reference_logits.pt"
MAX_SEQ = 128


def palettize(input_pkg: Path, output_pkg: Path, nbits: int):
    """Load fp16 decode mlpackage and palettize weights to `nbits` bits."""
    print(f"\n=== palettize {nbits}-bit ===")
    print(f"  loading fp16 source: {input_pkg}")
    m_in = ct.models.MLModel(str(input_pkg))

    print(f"  building palettization config (mode=kmeans, nbits={nbits})...")
    op_cfg = OpPalettizerConfig(mode="kmeans", nbits=nbits, granularity="per_tensor")
    opt_cfg = OptimizationConfig(global_config=op_cfg)

    print(f"  palettizing (this can take several minutes)...")
    t0 = time.time()
    m_out = palettize_weights(m_in, opt_cfg)
    print(f"  done in {time.time()-t0:.1f}s")

    m_out.save(str(output_pkg))
    print(f"  saved: {output_pkg}")

    # Size comparison
    def _dir_size(p):
        return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file()) / 1e6
    src_mb = _dir_size(input_pkg)
    dst_mb = _dir_size(output_pkg)
    print(f"  bundle: {src_mb:.0f} MB (fp16) → {dst_mb:.0f} MB (int{nbits}) "
          f"[{100*dst_mb/src_mb:.1f}%]")


def _layer_state_shapes(cfg, i, max_seq):
    lt = "linear_attention" if i % 4 != 3 else "full_attention"
    if lt == "linear_attention":
        conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 + \
                    cfg.linear_value_head_dim * cfg.linear_num_value_heads
        a = (1, conv_dim, cfg.linear_conv_kernel_dim)
        b = (1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim)
    else:
        a = (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
        b = (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
    return a, b


def parity_check(pkg: Path, oracle, cfg, label):
    """Run 5 prompts on ANE, measure top-1 / top-3 / cos vs oracle."""
    print(f"\n=== parity check ({label}) ===")
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    try:
        mlm = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as e:
        print(f"  load failed: {e}")
        return

    hits_top1 = 0; hits_top3 = 0; hits_top5 = 0
    cos_vals = []
    for pi, rec in enumerate(oracle["records"][:5]):
        ids = rec["input_ids"]; S = ids.shape[1]
        states = make_zero_states(cfg, MAX_SEQ)
        state_d = {f"state_{i}_a": states[2*i].numpy().astype(np.float16)
                   for i in range(cfg.num_hidden_layers)}
        state_d.update({f"state_{i}_b": states[2*i+1].numpy().astype(np.float16)
                        for i in range(cfg.num_hidden_layers)})
        last_logits = None
        for t in range(S):
            pos_ids = torch.tensor([[t]], dtype=torch.long)
            dummy = torch.zeros(1, 1, cfg.hidden_size)
            with torch.no_grad():
                c_t, s_t = rot(dummy, pos_ids)
            inp = {
                "input_token": ids[:, t:t+1].numpy().astype(np.int32),
                "position": np.array([float(t)], dtype=np.float32),
                "cos": c_t.numpy().astype(np.float16),
                "sin": s_t.numpy().astype(np.float16),
                **state_d,
            }
            out = mlm.predict(inp)
            for i in range(cfg.num_hidden_layers):
                state_d[f"state_{i}_a"] = out[f"new_state_{i}_a"]
                state_d[f"state_{i}_b"] = out[f"new_state_{i}_b"]
            if t == S - 1:
                last_logits = out["logits"][0, 0]
        oracle_top1 = int(rec["top10_last_ids"][0].item())
        top10 = np.argsort(-last_logits)[:10]
        if top10[0] == oracle_top1: hits_top1 += 1
        if oracle_top1 in top10[:3]: hits_top3 += 1
        if oracle_top1 in top10[:5]: hits_top5 += 1
        ref = rec["logits_recurrent"][-1].float().numpy()
        c = cos_sim(torch.from_numpy(last_logits.astype(np.float32)),
                    torch.from_numpy(ref))
        cos_vals.append(float(c))
        print(f"  prompt[{pi}] S={S}  top1={top10[0]==oracle_top1}  "
              f"top3={oracle_top1 in top10[:3]}  cos={c:.4f}")
    print(f"  aggregate: top1={hits_top1}/5  top3={hits_top3}/5  "
          f"top5={hits_top5}/5  mean cos={np.mean(cos_vals):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16-pkg", required=True,
                    help="path to existing fp16 decode mlpackage")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nbits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_ID)
    fp16_pkg = Path(args.fp16_pkg)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_pkg = out_dir / f"qwen3_5_0_8b_decode_int{args.nbits}_mseq128.mlpackage"

    palettize(fp16_pkg, out_pkg, args.nbits)

    if args.skip_parity:
        return
    oracle = torch.load(str(ORACLE), map_location="cpu", weights_only=False)
    parity_check(out_pkg, oracle, cfg, label=f"int{args.nbits}")


if __name__ == "__main__":
    main()
