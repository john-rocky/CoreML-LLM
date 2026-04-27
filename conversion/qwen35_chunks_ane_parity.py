"""Mac parity / speed gate for the new ANE-recipe chunked Qwen3.5 build
(qwen3_5_(0_8b|2b)_decode_chunks/). Re-uses the helper logic from
qwen35_2b_chunks_parity.py but doesn't pin to the 2B-specific
build module — caller passes --model-id, hidden size is read from
the bundle.

Usage:
  # 0.8B
  python qwen35_chunks_ane_parity.py \\
      --chunks-dir /tmp/qwen35_0_8b_ane/qwen3_5_0_8b_decode_chunks \\
      --model-id Qwen/Qwen3.5-0.8B
  # 2B
  python qwen35_chunks_ane_parity.py \\
      --chunks-dir /tmp/qwen35_2b_ane/qwen3_5_2b_decode_chunks \\
      --model-id Qwen/Qwen3.5-2B
"""
from pathlib import Path
import argparse
import time

import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoConfig, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from test_qwen3_5_full_decode_trace import make_zero_states
from build_qwen35_decode_chunks_ane import (
    EMBED_BIN_NAME, CHUNK_NAMES, _chunk_boundaries,
)


PROMPTS = [
    "What is the capital of France?",
    "こんにちは、元気ですか?",
    "美味しい餃子のレシピを教えて。",
]


def load_text_config(model_id: str) -> Qwen3_5TextConfig:
    full_cfg = AutoConfig.from_pretrained(model_id)
    text_dict = (full_cfg.text_config.to_dict()
                 if hasattr(full_cfg, "text_config") else full_cfg.to_dict())
    return Qwen3_5TextConfig.from_dict(text_dict)


def run_chunked_decode(
    embed_weight, body_mlms, boundaries, cfg, max_seq, input_ids_list, max_new,
    eos_tokens=(248044, 248045, 248046),
):
    """Greedy chained decode — Swift-equivalent stepPredict loop in Python."""
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    states = make_zero_states(cfg, max_seq)

    chunk_dicts = []
    for (s, e) in boundaries:
        d = {}
        for i in range(s, e):
            d[f"state_{i}_a"] = states[2 * i].numpy().astype(np.float16)
            d[f"state_{i}_b"] = states[2 * i + 1].numpy().astype(np.float16)
        chunk_dicts.append(d)

    def _step(tok_id: int, pos: int):
        hidden = embed_weight[tok_id:tok_id + 1, :].reshape(1, 1, -1).astype(np.float16)
        pos_ids = torch.tensor([[pos]], dtype=torch.long)
        dummy = torch.zeros(1, 1, cfg.hidden_size)
        with torch.no_grad():
            c_t, s_t = rot(dummy, pos_ids)
        cos_np = c_t.numpy().astype(np.float16)
        sin_np = s_t.numpy().astype(np.float16)
        pos_np = np.array([float(pos)], dtype=np.float32)

        last_out = None
        for ci, (mlm, d) in enumerate(zip(body_mlms, chunk_dicts)):
            inp = {
                "hidden_in": hidden,
                "position": pos_np, "cos": cos_np, "sin": sin_np,
                **d,
            }
            out = mlm.predict(inp)
            last_out = out
            s, e = boundaries[ci]
            for i in range(s, e):
                d[f"state_{i}_a"] = out[f"new_state_{i}_a"]
                d[f"state_{i}_b"] = out[f"new_state_{i}_b"]
            if "hidden" in out:
                hidden = out["hidden"].astype(np.float16)
        return last_out["logits"][0, 0]

    last_logits = None
    t_pre = time.time()
    for t, tok_id in enumerate(input_ids_list):
        last_logits = _step(tok_id, t)
    pre_dt = time.time() - t_pre

    generated = []
    next_token = int(np.argmax(last_logits))
    S_prompt = len(input_ids_list)
    t_dec = time.time()
    decode_steps = 0
    for step in range(max_new):
        pos = S_prompt + step
        if pos >= max_seq:
            break
        generated.append(next_token)
        decode_steps += 1
        if next_token in eos_tokens:
            break
        logits = _step(next_token, pos)
        next_token = int(np.argmax(logits))
    dec_dt = time.time() - t_dec
    return generated, pre_dt, dec_dt, decode_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--model-id", required=True,
                    help="Qwen/Qwen3.5-0.8B or Qwen/Qwen3.5-2B")
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--compute-units", default="cpu_and_ne",
                    choices=["cpu_and_ne", "cpu_and_gpu", "cpu_only", "all"])
    args = ap.parse_args()

    cu_map = {
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "all": ct.ComputeUnit.ALL,
    }
    units = cu_map[args.compute_units]

    chunks_dir = Path(args.chunks_dir)
    embed_bin_path = chunks_dir / EMBED_BIN_NAME
    body_paths = [chunks_dir / f"{name}.mlpackage"
                  for name in CHUNK_NAMES[:args.num_chunks]]
    for p in [embed_bin_path, *body_paths]:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    print(f"loading config + tokenizer ({args.model_id})...")
    cfg = load_text_config(args.model_id)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    boundaries = _chunk_boundaries(cfg.num_hidden_layers, args.num_chunks)
    print(f"  hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"vocab={cfg.vocab_size} boundaries={boundaries}")

    embed_size_mb = embed_bin_path.stat().st_size / 1e6
    print(f"loading embed_weight.bin ({embed_size_mb:.0f} MB) "
          f"+ {args.num_chunks} chunks (units={args.compute_units})...")
    t0 = time.time()
    embed_weight = np.frombuffer(embed_bin_path.read_bytes(), dtype=np.float16)
    embed_weight = embed_weight.reshape(cfg.vocab_size, cfg.hidden_size)
    body_mlms = [ct.models.MLModel(str(p), compute_units=units)
                 for p in body_paths]
    print(f"  loaded in {time.time()-t0:.1f}s")

    for pi, prompt in enumerate(PROMPTS):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc.input_ids[0].tolist()
        if len(ids) > args.max_seq - args.max_new - 1:
            ids = ids[:args.max_seq - args.max_new - 1]
        print(f"\n=== prompt[{pi}]: {prompt!r} (len={len(ids)}) ===")
        generated, pre_dt, dec_dt, dec_steps = run_chunked_decode(
            embed_weight, body_mlms, boundaries, cfg, args.max_seq, ids, args.max_new)
        gen_text = tok.decode(generated, skip_special_tokens=False)
        full_text = tok.decode(ids + generated, skip_special_tokens=False)
        pre_tps = len(ids) / pre_dt if pre_dt > 0 else 0.0
        dec_tps = dec_steps / dec_dt if dec_dt > 0 else 0.0
        print(f"  prefill: {len(ids)} tok in {pre_dt:.2f}s ({pre_tps:.1f} tok/s)")
        print(f"  decode:  {dec_steps} tok in {dec_dt:.2f}s ({dec_tps:.1f} tok/s)")
        print(f"  full: {full_text!r}")
        print(f"  gen:  {gen_text!r}")


if __name__ == "__main__":
    main()
