#!/usr/bin/env python3
"""Diagnose EAGLE-3 fusion layer choice for Gemma 4 E2B.

EAGLE-3 picks three fusion layers (hidden-state taps) that the draft ingests:
the deployed config uses L8, L17, L34. The official recipe is "low/mid/high";
for Gemma 4 E2B (35 layers, indexed 0..34) that is roughly quartile-spaced.
This diagnostic asks: for *this specific model*, which three layers carry the
most next-token-predictive signal? Are [8, 17, 34] the best, or would
[2, 17, 32] (shifted outward), [11, 17, 23] (tighter), or some other triple
do better?

Method
------
Run the HF fp16 Gemma 4 E2B target with `output_hidden_states=True` on a
small corpus sample. For each layer i in 0..34 and each token position:

  * mean/std L2 norm of the layer hidden
  * cosine similarity to L34 (final layer) — redundancy indicator
  * cosine similarity to the *target's next-token argmax* token embedding
    (via the tied `lm_head` / `embed_tokens` weight, per-row-normalized).
    High cosine means hidden_i already points toward the direction the
    model's output layer will project into at the next step.
  * "logit-lens" probe: project hidden_i through the tied LM head (apply
    softcap, take log-softmax), evaluate at the target's argmax — this
    gives the *log-probability* that an early-exit at layer i would assign
    to the real next-token. It is a much stronger MI proxy than cosine,
    because it uses the whole decoded distribution. Also captures top-K
    agreement (fraction of positions where layer-i's own argmax falls in
    the target's top-K — "agree-top1" is the most direct signal).

Also computes pairwise cosine between every layer's mean-centered hidden
(off-diagonal redundancy), so tightly-coupled triples can be rejected.

Top-3 fusion recommendation: greedily pick the single layer with highest
next-argmax cosine in each of the low (0..11), mid (12..22), high (23..34)
ranges, then check whether the redundancy-aware score changes that choice.

Output
------
- CSV at --csv-out: one row per layer with
    layer_idx, mean_norm, std_norm, cos_to_L34,
    cos_to_next_argmax_embed, logit_lens_logp, agree_top1, agree_top5
- Prints a summary table + the recommended fusion triple.

Inputs
------
The HF weights for `google/gemma-4-E2B-it` must be locally available in
the HF cache (or a custom path). The Gemma 4 repo is gated; this script
does NOT download — it only reads a local snapshot.

Default path: use the first snapshot under
  ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/<sha>/

Usage
-----
    python conversion/diagnose_fusion_layers.py \
        --corpus ~/Downloads/eagle_corpus.jsonl \
        --max-tokens 500 \
        --csv-out docs/FUSION_LAYER_SCORES.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


NUM_LAYERS = 35
HIDDEN = 1536
DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_SNAPSHOT_DIR = (
    "/Users/majimadaisuke/.cache/huggingface/hub/"
    "models--google--gemma-4-E2B-it/snapshots"
)


def find_local_snapshot() -> Path | None:
    """Return path to a local HF snapshot of gemma-4-E2B-it, or None."""
    root = Path(DEFAULT_SNAPSHOT_DIR)
    if not root.exists():
        return None
    for sub in sorted(root.iterdir()):
        if (sub / "model.safetensors").exists() or (sub / "model.safetensors.index.json").exists():
            return sub
    return None


def load_corpus(path: Path, tokenizer, *, max_tokens: int,
                seq_len: int, min_seq: int) -> list[list[int]]:
    """Read JSONL corpus, tokenize, return up to `max_tokens` total across seqs."""
    seqs: list[list[int]] = []
    remaining = max_tokens
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if remaining <= 0:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text, truncation=True, max_length=seq_len,
                                   add_special_tokens=False)
            if len(ids) < min_seq:
                continue
            if len(ids) > remaining:
                ids = ids[:remaining]
            seqs.append(ids)
            remaining -= len(ids)
    return seqs


@torch.no_grad()
def collect_all_hiddens(text_model, lm_head_weight: torch.Tensor,
                         seqs: list[list[int]], device: str,
                         softcap: float = 30.0) -> tuple:
    """Forward each seq through the text model, collect:
       - per-layer hiddens (36 entries: embed + 35 layer outputs), fp32
       - argmax token ids at each position (from the LM head, post-softcap)

    Returns:
      hiddens: list of tensors, each (T, H), one per layer in [0..35]
      argmax:  (N,) int64, concatenated next-token argmax ids, where N =
               sum_i (len(seq_i) - 1)  (last position in each seq has no
               next token; we skip it to align "layer at pos t" with
               "argmax of logits at pos t" which is the model's
               prediction for pos t+1 from pos t's final hidden).
      input_ids: (N,) int64, the input token at each retained position.
    """
    layer_cat: list[list[torch.Tensor]] = [[] for _ in range(NUM_LAYERS + 1)]
    argmax_list: list[torch.Tensor] = []
    input_list: list[torch.Tensor] = []

    lm_head_w = lm_head_weight.to(device=device, dtype=torch.float32)

    for seq_idx, ids in enumerate(seqs):
        if len(ids) < 2:
            continue
        ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = text_model(input_ids=ids_t, output_hidden_states=True,
                         use_cache=False)
        hs = out.hidden_states  # tuple of 36 tensors, each (1, T, H)
        T = ids_t.shape[1]
        # Drop the last position: no next-token target for it.
        for i, h in enumerate(hs):
            # Cast hidden to fp32 for correlation math; keep on CPU to avoid OOM.
            layer_cat[i].append(h[0, :T-1].to(dtype=torch.float32, device="cpu"))

        # Compute next-token argmax from the *final* hidden at each position,
        # using the real LM head (softcap-tanh, per Gemma 4 text config).
        last_h = hs[-1][0, :T-1].to(torch.float32)  # (T-1, H)
        logits = F.linear(last_h, lm_head_w)       # (T-1, V)
        if softcap is not None and softcap > 0:
            logits = torch.tanh(logits / softcap) * softcap
        argmax_tok = logits.argmax(dim=-1).to("cpu")  # (T-1,)
        argmax_list.append(argmax_tok)
        # The corresponding "input at pos t" is ids[t], retained positions = 0..T-2
        input_list.append(torch.tensor(ids[:T-1], dtype=torch.long))

    # Concat each layer's positions into one big (N, H) matrix.
    hiddens: list[torch.Tensor] = []
    for i in range(NUM_LAYERS + 1):
        hiddens.append(torch.cat(layer_cat[i], dim=0))  # (N, H)
    argmax = torch.cat(argmax_list, dim=0)  # (N,)
    input_ids = torch.cat(input_list, dim=0)  # (N,)
    return hiddens, argmax, input_ids


def cosine_rowwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N, H) vs (N, H) -> (N,) row-wise cosine."""
    na = F.normalize(a, dim=-1, eps=1e-8)
    nb = F.normalize(b, dim=-1, eps=1e-8)
    return (na * nb).sum(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=str, default=None,
                    help="Local HF snapshot dir for gemma-4-E2B-it. "
                         "If omitted, auto-detect the newest snapshot under "
                         f"{DEFAULT_SNAPSHOT_DIR}.")
    ap.add_argument("--corpus", type=Path,
                    default=Path.home() / "Downloads" / "eagle_corpus.jsonl",
                    help="JSONL corpus (same format as eagle_corpus.jsonl).")
    ap.add_argument("--max-tokens", type=int, default=500,
                    help="Total token positions to analyse across all seqs.")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="Max tokens per sequence (truncate longer).")
    ap.add_argument("--min-seq", type=int, default=16)
    ap.add_argument("--csv-out", type=Path,
                    default=Path("docs/FUSION_LAYER_SCORES.csv"),
                    help="Output CSV with per-layer scores.")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Torch device for the target forward. 'cpu' is safe "
                         "on Mac; 'mps' may or may not work depending on "
                         "transformers / torch versions.")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    # Resolve model path.
    if args.model_path is None:
        snap = find_local_snapshot()
        if snap is None:
            print(f"[FATAL] No local snapshot of {DEFAULT_MODEL_ID} found under "
                  f"{DEFAULT_SNAPSHOT_DIR}.\n"
                  f"        Gemma 4 is gated — this script will NOT download.\n"
                  f"        Provide --model-path pointing at a local dir with "
                  f"model.safetensors + config.json, or accept the license and "
                  f"run `huggingface-cli download google/gemma-4-E2B-it`.",
                  file=sys.stderr)
            return 2
        args.model_path = str(snap)
    model_path = Path(args.model_path)
    if not (model_path / "config.json").exists():
        print(f"[FATAL] {model_path} missing config.json", file=sys.stderr)
        return 2
    if not (model_path / "model.safetensors").exists() and not (
            model_path / "model.safetensors.index.json").exists():
        print(f"[FATAL] {model_path} missing model.safetensors[.index.json]",
              file=sys.stderr)
        return 2

    # Corpus.
    if not args.corpus.exists():
        print(f"[FATAL] corpus {args.corpus} not found", file=sys.stderr)
        return 2

    # Dtype.
    torch_dtype = {"float16": torch.float16,
                   "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[args.dtype]

    print(f"[Load] Gemma 4 E2B from {model_path}")
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = Gemma4ForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype=torch_dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if args.device != "cpu":
        model.to(args.device)

    # Gemma 4 ties embed_tokens <-> lm_head.
    lm_head_w = model.lm_head.weight.data.detach().clone()
    text_model = model.model.language_model

    softcap = getattr(model.config.text_config, "final_logit_softcapping", None) or 30.0
    print(f"[Config] layers={model.config.text_config.num_hidden_layers}, "
          f"hidden={model.config.text_config.hidden_size}, softcap={softcap}")

    # Corpus.
    print(f"[Data] Tokenizing corpus {args.corpus} (max_tokens={args.max_tokens})")
    seqs = load_corpus(args.corpus, tokenizer,
                       max_tokens=args.max_tokens,
                       seq_len=args.seq_len, min_seq=args.min_seq)
    tot = sum(len(s) for s in seqs)
    print(f"[Data] {len(seqs)} sequences, {tot} total tokens "
          f"(N positions = {tot - len(seqs)} after dropping last-per-seq).")
    if len(seqs) == 0:
        print("[FATAL] no usable sequences", file=sys.stderr)
        return 2

    # Collect all-layer hiddens + next-token argmax.
    print("[Run] forwarding through target...")
    hiddens, argmax, input_ids = collect_all_hiddens(
        text_model, lm_head_w, seqs, device=args.device, softcap=float(softcap))
    N = argmax.shape[0]
    print(f"[Run] N = {N} positions collected")

    # Compute embed-of-argmax: (N, H), from the tied lm_head_w matrix.
    # lm_head_w is (V, H); for tied models this is the same as embed_tokens.
    # We only need rows for the argmax tokens.
    lm_head_w_cpu = lm_head_w.to(dtype=torch.float32, device="cpu")
    argmax_embed = lm_head_w_cpu[argmax]  # (N, H)

    # Per-layer metrics. Index 0 is the embed layer (pre-layer-0); indices 1..35
    # are post-layer outputs. Our "fusion layer i" convention uses the
    # post-layer index (so [8, 17, 34] corresponds to hiddens[9], hiddens[18],
    # hiddens[35]). We report in the post-layer index space directly.
    ref_layer_h = hiddens[NUM_LAYERS]  # (N, H) final (L34)

    # Logit-lens: `lm_head(norm(h_i))` for each layer i. Gemma 4's text
    # model structure is: decoder loop produces raw layer outputs; after
    # the loop, `last_hidden_state = self.norm(raw_L34_out)` is fed to
    # `lm_head`. `out.hidden_states` has 36 entries:
    #   hs[0]     = embed output (raw)
    #   hs[i+1]   = raw output of decoder layer i,  for i in 0..33
    #   hs[35]    = norm(raw output of decoder layer 34) == last_hidden_state
    # So for indices 0..34 of `out.hidden_states` (i.e. hs[0]..hs[34], the
    # 35 raw outputs of embed + layers 0..33) we apply `self.norm` then
    # `lm_head`. For the final one (hs[35]) the norm is already baked in.
    final_norm_w = text_model.norm.weight.data.detach().to(
        dtype=torch.float32, device="cpu")
    final_norm_eps = float(getattr(text_model.norm, "eps", 1e-6))

    def apply_final_norm(x: torch.Tensor) -> torch.Tensor:
        # Gemma 4 RMSNorm (no +1): y = x / sqrt(mean(x^2)+eps) * w.
        # See modeling_gemma4.Gemma4RMSNorm.
        x32 = x.to(torch.float32)
        mean_sq = x32.pow(2).mean(-1, keepdim=True) + final_norm_eps
        return x32 * mean_sq.pow(-0.5) * final_norm_w

    results: list[dict] = []
    for layer_i in range(NUM_LAYERS):
        # `hiddens[layer_i + 1]` is the raw output of decoder layer
        # `layer_i` for layer_i in 0..33, and (for layer_i == 34) it is
        # the already-normed hs[35] == last_hidden_state.
        h = hiddens[layer_i + 1]  # (N, H), fp32
        is_final = (layer_i == NUM_LAYERS - 1)
        norms = h.norm(dim=-1)
        mean_norm = float(norms.mean())
        std_norm = float(norms.std())
        cos_L34 = float(cosine_rowwise(h, ref_layer_h).mean())
        cos_argmax = float(cosine_rowwise(h, argmax_embed).mean())

        # Logit-lens: apply final RMSNorm (unless h is already normed, i.e.
        # the final layer), then lm_head, then softcap, then log-softmax.
        h_for_head = h if is_final else apply_final_norm(h)
        logits = F.linear(h_for_head.to(torch.float32), lm_head_w_cpu)  # (N, V)
        if softcap is not None and float(softcap) > 0:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        logp = F.log_softmax(logits, dim=-1)
        logp_true = logp.gather(1, argmax.unsqueeze(-1).to(torch.long)).squeeze(-1)
        mean_logp = float(logp_true.mean())

        # Top-K agreement: does layer-i's argmax match target's top1? top5?
        layer_argmax = logits.argmax(dim=-1)
        agree_top1 = float((layer_argmax == argmax).float().mean())
        top5 = logits.topk(5, dim=-1).indices  # (N, 5)
        agree_top5 = float((top5 == argmax.unsqueeze(-1)).any(dim=-1).float().mean())

        results.append({
            "layer_idx": layer_i,
            "mean_norm": mean_norm,
            "std_norm": std_norm,
            "cos_to_L34": cos_L34,
            "cos_to_next_argmax_embed": cos_argmax,
            "logit_lens_logp": mean_logp,
            "agree_top1": agree_top1,
            "agree_top5": agree_top5,
        })

    # Pairwise "redundancy" matrix: mean row-wise cosine between each pair.
    # This is expensive if we compute it for every pair, but with N<=500 and
    # 35 layers it's fine (~2 seconds).
    H_stack = torch.stack([hiddens[i + 1] for i in range(NUM_LAYERS)], dim=0)
    # H_stack: (L, N, H). Row-normalize per (l, n) vector.
    H_norm = F.normalize(H_stack, dim=-1, eps=1e-8)  # (L, N, H)
    # Pairwise mean-over-N cosine: einsum on last dim, average over N.
    pair_cos = torch.einsum("lnh,knh->lkn", H_norm, H_norm).mean(dim=-1)
    # pair_cos: (L, L)

    # Write CSV.
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer_idx", "mean_norm", "std_norm",
            "cos_to_L34", "cos_to_next_argmax_embed",
            "logit_lens_logp", "agree_top1", "agree_top5",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[Out] wrote CSV {args.csv_out}")

    # Print compact table (stdout).
    print()
    print(f"{'L':>3}  {'norm':>8}  {'std':>7}  {'cos->L34':>8}  "
          f"{'cos->arg':>9}  {'logp':>8}  {'top1':>6}  {'top5':>6}")
    for r in results:
        print(f"{r['layer_idx']:>3}  "
              f"{r['mean_norm']:>8.2f}  "
              f"{r['std_norm']:>7.2f}  "
              f"{r['cos_to_L34']:>8.4f}  "
              f"{r['cos_to_next_argmax_embed']:>9.4f}  "
              f"{r['logit_lens_logp']:>8.3f}  "
              f"{r['agree_top1']:>6.3f}  "
              f"{r['agree_top5']:>6.3f}")

    # Recommendation: pick best layer in each of three ranges.
    # Official EAGLE-3 selects "low/mid/high" spanning the full depth.
    # For 35-layer Gemma 4, split ~quartile: low=[0..11], mid=[12..22],
    # high=[23..34]. We use the logit-lens log-prob (logp) as the MI proxy:
    # higher = more next-token information decodable from this layer via
    # the tied LM head. agree_top1 is saturated at 0 for shallow layers
    # (not enough of the next-token distribution has been resolved to make
    # the top-1 match), so logp is the strictly finer score here.
    # Start low range at 1 (not 0): L0 is the raw embedding output and
    # would give the draft a trivial copy of the input-token embedding it
    # already has, not a novel shallow feature.
    ranges = {"low":  (1, 12),      # 1..11
              "mid":  (12, 23),     # 12..22
              "high": (23, NUM_LAYERS)}  # 23..34
    score_arr = np.array([r["agree_top1"] for r in results])
    logp_arr = np.array([r["logit_lens_logp"] for r in results])
    best = {}
    for band, (lo, hi) in ranges.items():
        sub_idx = np.arange(lo, hi)
        sub_logp = logp_arr[sub_idx]
        best_local = int(sub_idx[int(np.argmax(sub_logp))])
        best[band] = (best_local, float(score_arr[best_local]),
                      float(logp_arr[best_local]))

    print()
    print("Recommended triple by MI proxy (max logit_lens_logp per band):")
    triple = [best["low"][0], best["mid"][0], best["high"][0]]
    for band, (li, sc, lp) in best.items():
        print(f"  {band}: L{li}  (logp = {lp:.3f}, agree_top1 = {sc:.3f})")
    current = [8, 17, 34]
    print(f"  current deployed: {current}")
    print(f"  recommended    : {triple}")
    def _score_triple(t: list[int]) -> tuple[float, float]:
        return (float(sum(logp_arr[i] for i in t)),
                float(sum(score_arr[i] for i in t)))

    print("  candidate triples (sum-logp, sum-top1):")
    candidates = [
        ("current [8,17,34]",         [8, 17, 34]),
        ("recommended (this script)", triple),
        ("shifted-outward [2,17,32]", [2, 17, 32]),
        ("tighter [11,17,23]",        [11, 17, 23]),
        ("symmetric quartiles [8,17,26]", [8, 17, 26]),
        ("high-dominant [15,26,34]",  [15, 26, 34]),
    ]
    for name, t in candidates:
        slp, stop = _score_triple(t)
        print(f"    {name:34s} sum-logp={slp:8.3f}  sum-top1={stop:.3f}")
    base_logp = _score_triple([8, 17, 34])[0]
    rec_logp = _score_triple(triple)[0]
    delta = rec_logp - base_logp
    print(f"  delta(sum-logp, recommended - current) = {delta:.3f} "
          f"(positive = recommended has more signal)")
    print()
    print(f"  NOTE: in the low band (L1..L11), all layers have logp within "
          f"~{max(logp_arr[1:12]) - min(logp_arr[1:12]):.1f} nats of each "
          f"other; the choice there is largely immaterial. The meaningful "
          f"signal is in mid (L17 >> others) and high (L34/L33).")

    # Redundancy-aware check: if recommended triple has any pair with
    # pair_cos > 0.95, suggest the next-best in that band.
    L = NUM_LAYERS
    redundancy_warn = []
    for a in range(3):
        for b in range(a + 1, 3):
            r = float(pair_cos[triple[a], triple[b]])
            if r > 0.95:
                redundancy_warn.append((triple[a], triple[b], r))
    if redundancy_warn:
        print("  WARNING — pair-cos > 0.95 in recommended triple:")
        for a, b, r in redundancy_warn:
            print(f"    L{a} vs L{b}: {r:.4f}")

    # Also compute a redundancy-adjusted score per band: within each band,
    # score = logp - alpha * mean(pair_cos vs chosen layers in other bands).
    # This is advisory only.
    print()
    print("Diversity-adjusted band picks (penalise redundancy with other bands):")
    adjusted: list[int] = []
    alpha = 5.0  # logp ranges ~-5..-40, pair_cos ~0..1 -> scale appropriately
    for band, (lo, hi) in ranges.items():
        sub_idx = list(range(lo, hi))
        sub_logp = logp_arr[sub_idx]
        # Penalty vs the currently-best layer in the OTHER bands.
        others = [best[b][0] for b in ranges if b != band]
        penalty = np.zeros_like(sub_logp)
        for other in others:
            penalty += np.array([float(pair_cos[i, other]) for i in sub_idx])
        penalty /= max(1, len(others))
        adjusted_logp = sub_logp - alpha * penalty
        pick = sub_idx[int(np.argmax(adjusted_logp))]
        adjusted.append(pick)
        print(f"  {band}: L{pick}  (raw-logp={logp_arr[pick]:.3f}, "
              f"adj={adjusted_logp.max():.3f})")
    print(f"  diversity-adjusted triple: {adjusted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
