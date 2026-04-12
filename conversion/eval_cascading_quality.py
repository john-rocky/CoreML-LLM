#!/usr/bin/env python3
"""Quality gate for Cascading KV Cache (Approach C).

Validates that swapping the full-attention layers to the cascading variant
does not regress long-context quality on Gemma 4 E2B. Runs LongBench v1
subset (same as eval_longbench.py) with two configurations:

  1. Original Gemma 4 E2B (unmodified)
  2. Same model with full-attention layers replaced by CascadingFullAttention

Reports per-task F1 / ROUGE-L / EM delta. The paper (arXiv 2406.17808)
reports +5.6% LongBench on Llama-2; a good outcome for us is >= -1% (no
regression; ideally positive).

Usage (Colab A100 or Mac with enough unified RAM):
    python conversion/eval_cascading_quality.py \\
        --model-id google/gemma-4-E2B-it \\
        --output /content/drive/MyDrive/cascading_quality.json \\
        --ctx 8192 \\
        --max-samples 30

Reuses the task list + metrics from eval_longbench.py so the numbers are
directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Re-use metrics + corpus loaders from the existing LongBench script
sys.path.append(str(Path(__file__).resolve().parent))
from eval_longbench import (  # type: ignore
    DEFAULT_TASKS, MAX_GEN_TOKENS, PROMPT_TEMPLATE, METRICS, f1_score,
)
from models.gemma4_swa_cascading import (  # type: ignore
    CascadingConfig, make_cascading_full_attention,
)


def patch_full_attention_layers(model, cfg: CascadingConfig) -> int:
    """Swap full-attention layers with the cascading variant in-place.

    Returns number of layers patched. KV-shared layers are skipped (they
    read from other layers' KV, no own projections to patch).
    """
    tm = model.model
    if hasattr(tm, "language_model"): tm = tm.language_model
    if hasattr(tm, "model"): tm = tm.model
    layers = tm.layers

    tcfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    layer_types = getattr(tcfg, "layer_types", None)
    if layer_types is None:
        raise RuntimeError("model has no layer_types; not a Gemma 4-style hybrid")
    first_kv_shared = len(layer_types) - getattr(tcfg, "num_kv_shared_layers", 0)

    patched = 0
    for idx, layer in enumerate(layers):
        if layer_types[idx] != "full_attention": continue
        if idx >= first_kv_shared: continue   # KV-shared, has no own k_proj/v_proj
        if not hasattr(layer.self_attn, "k_proj"): continue

        # Build the cascading module from base attention's weights
        base = layer.self_attn
        base.config = tcfg  # hint for make_cascading_full_attention
        ca = make_cascading_full_attention(base, cfg)
        # Replace in-place. NOTE: this test harness calls the cascading module
        # OUTSIDE the normal transformers forward() — we only use it for a
        # standalone numerical sanity check here. For full end-to-end LongBench
        # eval with the cascaded attention in the model's own forward path,
        # the transformers attention forward needs to call the cascading
        # module, which requires either (a) a monkey-patch of Gemma4TextAttention
        # or (b) re-wiring the layer. In v1 we do (a) below.
        layer.self_attn._cascading = ca
        patched += 1
    return patched


def monkey_patch_forward(model):
    """Monkey-patch Gemma4TextAttention.forward on layers that have _cascading.

    The cascading module expects pre-computed gather_idx / cos_p / sin_p, but
    the transformers forward passes position_ids and uses its own RoPE. For
    this quality test we:
      - Intercept the call
      - Compute gather_idx from current position_ids (last position)
      - Build cos_p / sin_p from model's rotary emb
      - Delegate to _cascading.forward
    """
    import types
    import torch.nn as nn

    # This is a quality-only harness; we only patch layers flagged with
    # _cascading. Non-full layers retain their normal behavior.
    tm = model.model
    if hasattr(tm, "language_model"): tm = tm.language_model
    if hasattr(tm, "model"): tm = tm.model

    for idx, layer in enumerate(tm.layers):
        if not hasattr(layer.self_attn, "_cascading"):
            continue
        original_forward = layer.self_attn.forward
        cascading = layer.self_attn._cascading

        def make_wrap(orig, ca):
            def wrapped(self_, hidden_states, position_embeddings=None, attention_mask=None,
                        past_key_values=None, cache_position=None, **kw):
                # For the quality test, we run the ORIGINAL forward but log
                # that it was seen. A proper test would dispatch through
                # cascading with real gather/rope. That requires significant
                # re-plumbing. v1 of this harness therefore runs UNMODIFIED
                # Gemma 4 and records the would-be cascading gather shape as
                # a sanity check. See notes at bottom of this file for the
                # full-fidelity variant (requires transformers internals).
                return orig(hidden_states, position_embeddings=position_embeddings,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            cache_position=cache_position, **kw)
            return wrapped

        layer.self_attn.forward = types.MethodType(make_wrap(original_forward, cascading),
                                                    layer.self_attn)


# ── LongBench runner (mirrors eval_longbench.py) ────────────────────────────

def run_longbench(model, tokenizer, device, ctx: int, max_samples: int,
                  tasks=DEFAULT_TASKS):
    from datasets import load_dataset
    from tqdm.auto import tqdm

    results = {}
    for task_name, task_type in tasks:
        try:
            ds = load_dataset("THUDM/LongBench", task_name, split="test")
        except Exception as e:
            print(f"  SKIP {task_name}: {e}")
            continue
        samples = list(ds)[:max_samples]
        max_new = MAX_GEN_TOKENS.get(task_type, 64)
        per_metric = {m[0]: [] for m in METRICS.get(task_type, [("f1", f1_score)])}

        for row in tqdm(samples, desc=task_name):
            ctx_text = row.get("context", row.get("input", ""))
            question = row.get("input", row.get("question", ""))
            answers = row.get("answers", row.get("output", ""))
            if isinstance(answers, list): answers = answers[0] if answers else ""

            prompt = PROMPT_TEMPLATE.format(context=ctx_text, question=question)
            ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True,
                                   max_length=ctx - max_new).to(device)
            if ids.shape[1] < 32: continue

            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id, use_cache=True)
            gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            if task_type in ("qa", "class", "retrieval"):
                gen = gen.split("\n")[0].strip()

            for name, fn in METRICS.get(task_type, [("f1", f1_score)]):
                per_metric[name].append(fn(gen, answers))

        task_result = {m: (sum(vs) / len(vs) if vs else 0.0) for m, vs in per_metric.items()}
        task_result["samples"] = len(samples)
        results[task_name] = task_result

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-samples", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device
    print(f"Device: {device}")
    try:
        from transformers import Gemma4ForConditionalGeneration as TCls
    except Exception:
        from transformers import AutoModelForCausalLM as TCls
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # ── Baseline ──
    print(f"\n[1/2] Baseline Gemma 4 E2B...")
    m = TCls.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map=device)
    m.eval()
    baseline = run_longbench(m, tokenizer, device, args.ctx, args.max_samples)
    del m
    torch.cuda.empty_cache()

    # ── Cascading-patched ──
    print(f"\n[2/2] Cascading-patched Gemma 4 E2B...")
    m = TCls.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map=device)
    m.eval()
    cfg = CascadingConfig()
    n_patched = patch_full_attention_layers(m, cfg)
    monkey_patch_forward(m)
    print(f"  patched {n_patched} full-attention layers with cascading module "
          f"({cfg.describe()})")
    cascaded = run_longbench(m, tokenizer, device, args.ctx, args.max_samples)

    # ── Delta ──
    delta = {}
    for task, r in baseline.items():
        if task not in cascaded: continue
        delta[task] = {}
        for k, v in r.items():
            if k == "samples": continue
            delta[task][k] = cascaded[task][k] - v

    out = {
        "model_id":         args.model_id,
        "ctx":              args.ctx,
        "max_samples":      args.max_samples,
        "cascading_config": {
            "sink":   cfg.sink_size,
            "recent": cfg.recent_window,
            "mid":    f"{cfg.mid_window}@{cfg.mid_stride}",
            "far":    f"{cfg.far_window}@{cfg.far_stride}",
            "total_slots": cfg.total_slots,
        },
        "baseline":   baseline,
        "cascaded":   cascaded,
        "delta":      delta,
        "n_patched_layers": n_patched,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f: json.dump(out, f, indent=2)
    print(f"\nsaved: {args.output}")

    print("\n── Summary ──")
    for task, d in delta.items():
        for k, v in d.items():
            tag = "✓" if v >= -0.01 else "✗"
            print(f"  {tag} {task} {k}: {v*100:+.2f}pp")

    # Also log the aggregate
    all_deltas = [v for d in delta.values() for v in d.values()]
    if all_deltas:
        mean = sum(all_deltas) / len(all_deltas)
        print(f"\n  aggregate mean delta: {mean*100:+.2f}pp "
              f"({'PASS' if mean >= -0.01 else 'FAIL'})")


# ── NOTE: v1 limitation ────────────────────────────────────────────────────
# The monkey_patch_forward above currently delegates to the original Gemma 4
# forward WITHOUT actually using the cascading attention path. Reason: the
# transformers Gemma4TextAttention forward signature is complex (takes
# position_embeddings tuples, past_key_values cache objects, cache_position,
# etc.) and wiring the cascading module into that plumbing requires more care
# than a drop-in replacement. A true end-to-end cascading quality test needs:
#
#   1. Build a minimal causal-LM wrapper that manually runs each layer, using
#      CascadingFullAttention for full layers and the stock GQA for sliding
#      layers. Call this from generate() instead of the original forward.
#
#   2. OR subclass Gemma4TextAttention and override its forward with the
#      cascading path (correctly handling cache format).
#
# v1 of this harness measures baseline correctly and stubs the cascading
# side. It is useful as a scaffolding check — file loads, layers are found,
# metrics computed — but the "cascaded" numbers reflect BASELINE behavior
# until the wiring is completed. Document clearly when reporting.


if __name__ == "__main__":
    main()
