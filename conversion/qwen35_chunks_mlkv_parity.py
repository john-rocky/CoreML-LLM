"""Mac parity / speed bench for the MLKV (KV-only MLState) Qwen3.5
chunked decode build.

KV cache lives in CoreML's MLState (managed via make_state()). SSM
state (conv_state_X / rec_state_X per linear layer in the chunk) flows
through input/output tensors like the original stateless build.

Disk layout: 4 body chunks (chunk_a..chunk_d) + 1 head chunk
(chunk_head). Body chunks emit only `hidden`; head chunk runs
final_norm + lm_head + topk[k=40] and returns top_indices/top_values.
The split lets the head run on a different compute unit (CPU+GPU on
iPhone) for an fp32 lm_head accumulator while bodies stay on ANE.
"""
from pathlib import Path
import argparse
import time

import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoConfig, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_qwen3_5_full_decode_trace import make_zero_states


PROMPTS = [
    "What is the capital of France?",
    "こんにちは、元気ですか?",
    "美味しい餃子のレシピを教えて。",
]

CHUNK_NAMES = ["chunk_a", "chunk_b", "chunk_c", "chunk_d",
               "chunk_e", "chunk_f"]
EMBED_BIN_NAME = "embed_weight.bin"


def load_text_config(model_id: str) -> Qwen3_5TextConfig:
    full_cfg = AutoConfig.from_pretrained(model_id)
    text_dict = (full_cfg.text_config.to_dict()
                 if hasattr(full_cfg, "text_config") else full_cfg.to_dict())
    return Qwen3_5TextConfig.from_dict(text_dict)


def make_causal_mask(pos: int, max_seq: int) -> np.ndarray:
    idx = np.arange(max_seq, dtype=np.float32)
    return np.where(idx <= pos, 0.0, -1e4).astype(np.float16).reshape(1, 1, 1, max_seq)


def chunk_lin_layer_indices(start, end):
    return [i for i in range(start, end) if i % 4 != 3]


def run_decode(
    embed_weight, body_mlms, body_states, boundaries, cfg, max_seq,
    input_ids_list, max_new,
    eos_tokens=(248044, 248045, 248046),
):
    # Precompute RoPE tables once. Avoid the torch graph eval per step
    # (~1-2 ms/step Python overhead).
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    pos_ids_full = torch.arange(max_seq, dtype=torch.long).unsqueeze(0)  # (1, max_seq)
    dummy = torch.zeros(1, 1, cfg.hidden_size)
    with torch.no_grad():
        cos_full, sin_full = rot(dummy, pos_ids_full)  # (1, max_seq, rd)
    cos_table = cos_full.numpy().astype(np.float16)[0]  # (max_seq, rd)
    sin_table = sin_full.numpy().astype(np.float16)[0]

    # Precompute the causal-mask row template — index p gives a (max_seq,)
    # row where positions > p are -1e4 and others 0. Build once.
    idx = np.arange(max_seq, dtype=np.float32)
    pos_arange = np.arange(max_seq, dtype=np.float32)[:, None]  # (max_seq, 1)
    causal_table = np.where(idx[None, :] <= pos_arange, 0.0, -1e4).astype(np.float16)
    # (max_seq, max_seq)

    states = make_zero_states(cfg, max_seq)
    ssm_state = {}
    for i in range(cfg.num_hidden_layers):
        if i % 4 != 3:
            ssm_state[f"conv_state_{i}"] = states[2 * i].numpy().astype(np.float16)
            ssm_state[f"rec_state_{i}"]  = states[2 * i + 1].numpy().astype(np.float16)

    recent = []  # closure list, mutated in-place
    REP_PEN = 1.1
    REP_WIN = 64
    def _step(tok_id: int, pos: int):
        """All chunks except chunk_d emit hidden; chunk_d (tail) emits
        FULL fp16 logits (1, 1, V). We cast to fp32, apply rep_penalty
        over full vocab (not just top-K), and argmax. Same path Swift
        will use on iPhone."""
        hidden = embed_weight[tok_id:tok_id + 1, :].reshape(1, 1, -1).astype(np.float16)
        cos_np = cos_table[pos:pos + 1].reshape(1, 1, -1)
        sin_np = sin_table[pos:pos + 1].reshape(1, 1, -1)
        causal = causal_table[pos].reshape(1, 1, 1, max_seq)
        cur = np.array([pos], dtype=np.int32)

        last_out = None
        for ci, (mlm, st) in enumerate(zip(body_mlms, body_states)):
            start, end = boundaries[ci]
            lin_idxs = chunk_lin_layer_indices(start, end)
            inp = {
                "hidden_in": hidden,
                "cos": cos_np, "sin": sin_np,
                "causal_mask": causal,
                "current_pos": cur,
            }
            for i in lin_idxs:
                inp[f"conv_state_{i}"] = ssm_state[f"conv_state_{i}"]
                inp[f"rec_state_{i}"]  = ssm_state[f"rec_state_{i}"]
            out = mlm.predict(inp, state=st)
            last_out = out
            for i in lin_idxs:
                ssm_state[f"conv_state_{i}"] = out[f"new_conv_state_{i}"]
                ssm_state[f"rec_state_{i}"]  = out[f"new_rec_state_{i}"]
            if "hidden" in out:
                hidden = out["hidden"].astype(np.float16)
        # Tail (chunk_d): full fp16 logits.
        logits = last_out["logits"][0, 0].astype(np.float32)
        for tok in recent:
            if 0 <= tok < logits.shape[0]:
                if logits[tok] > 0: logits[tok] /= REP_PEN
                else:               logits[tok] *= REP_PEN
        nt = int(np.argmax(logits))
        recent.append(nt)
        while len(recent) > REP_WIN:
            recent.pop(0)
        return nt

    t_pre = time.time()
    next_token = -1
    for t, tok_id in enumerate(input_ids_list):
        next_token = _step(tok_id, t)
    pre_dt = time.time() - t_pre

    generated = []
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
        next_token = _step(next_token, pos)
    dec_dt = time.time() - t_dec
    return generated, pre_dt, dec_dt, decode_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=40)
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
    embed_bin = chunks_dir / EMBED_BIN_NAME
    body_paths = [chunks_dir / f"{name}.mlpackage"
                  for name in CHUNK_NAMES[:args.num_chunks]]
    for p in [embed_bin, *body_paths]:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    print(f"loading config + tokenizer ({args.model_id})...")
    cfg = load_text_config(args.model_id)
    tok = AutoTokenizer.from_pretrained(args.model_id)

    boundaries = []
    per = cfg.num_hidden_layers // args.num_chunks
    for i in range(args.num_chunks):
        boundaries.append((i * per, (i + 1) * per))
    print(f"  hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"vocab={cfg.vocab_size} boundaries={boundaries}")

    print(f"loading {args.num_chunks} chunks (units={args.compute_units})...")
    t0 = time.time()
    embed_weight = np.frombuffer(embed_bin.read_bytes(), dtype=np.float16)
    embed_weight = embed_weight.reshape(cfg.vocab_size, cfg.hidden_size)
    body_mlms = [ct.models.MLModel(str(p), compute_units=units)
                 for p in body_paths]
    print(f"  loaded in {time.time()-t0:.1f}s")

    for pi, prompt in enumerate(PROMPTS):
        body_states = [m.make_state() for m in body_mlms]
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc.input_ids[0].tolist()
        if len(ids) > args.max_seq - args.max_new - 1:
            ids = ids[:args.max_seq - args.max_new - 1]
        print(f"\n=== prompt[{pi}]: {prompt!r} (len={len(ids)}) ===")
        generated, pre_dt, dec_dt, dec_steps = run_decode(
            embed_weight, body_mlms, body_states, boundaries, cfg,
            args.max_seq, ids, args.max_new)
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
