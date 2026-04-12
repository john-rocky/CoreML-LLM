#!/usr/bin/env python3
"""LongBench subset evaluation harness for Gemma 4 E2B (quality gate).

Use this to detect regressions when applying quality-affecting modifications:
  - DuoAttention streaming-head cap (Tier-2)
  - StreamingLLM sink+window (if adopted)
  - KV cache quantization
  - Any fine-tune that touches long-context behavior

Runs a subset of LongBench v1 tasks (fast subset by default) and reports
per-task metric. Compare baseline vs modified run by diffing the output JSON.

Usage (Colab / A100 or a Mac with enough RAM):
    pip install -q -U transformers datasets
    python conversion/eval_longbench.py \\
        --model-id google/gemma-4-E2B-it \\
        --output /content/drive/MyDrive/longbench_baseline.json \\
        --ctx 8192

    # after making changes:
    python conversion/eval_longbench.py \\
        --model-id google/gemma-4-E2B-it \\
        --output /content/drive/MyDrive/longbench_after.json \\
        --ctx 8192 \\
        --apply-modification my_mod_flag

LongBench is not exhaustive. Pair with needle-in-a-haystack runs for
retrieval-heavy checks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm.auto import tqdm


# ── Default task subset (fast but diverse) ──────────────────────────────────
# Full LongBench has 21 tasks; we pick a representative fast subset.
DEFAULT_TASKS = [
    ("narrativeqa",   "qa"),       # narrative QA, long story
    ("qasper",        "qa"),       # scientific paper QA
    ("multifieldqa_en", "qa"),     # mixed-domain QA
    ("hotpotqa",      "qa"),       # multi-hop QA
    ("2wikimqa",      "qa"),       # multi-hop QA
    ("gov_report",    "summary"),  # long-form summarization
    ("multi_news",    "summary"),  # multi-document summarization
    ("trec",          "class"),    # classification (short)
    ("triviaqa",      "qa"),       # short-answer
    ("passage_retrieval_en", "retrieval"),
]

MAX_GEN_TOKENS = {
    "qa": 64,
    "summary": 256,
    "class": 16,
    "retrieval": 16,
}

# Prompt template per LongBench conventions (simplified)
PROMPT_TEMPLATE = "Answer the following question based on the context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# ── Metrics (LongBench F1, Rouge-L, EM) ─────────────────────────────────────

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(pred: str, gt: str) -> float:
    pt = normalize_text(pred).split()
    gt_ = normalize_text(gt).split()
    if not pt or not gt_:
        return 0.0
    common = Counter(pt) & Counter(gt_)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pt)
    r = num_same / len(gt_)
    return 2 * p * r / (p + r)


def exact_match(pred: str, gt: str) -> float:
    return float(normalize_text(pred) == normalize_text(gt))


def rougeL(pred: str, gt: str) -> float:
    # Simplified LCS-based ROUGE-L F1
    a = normalize_text(pred).split()
    b = normalize_text(gt).split()
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[n][m]
    if lcs == 0: return 0.0
    p = lcs / n; r = lcs / m
    return 2 * p * r / (p + r)


METRICS = {
    "qa":        [("f1", f1_score)],
    "summary":   [("rougeL", rougeL)],
    "class":     [("em", exact_match)],
    "retrieval": [("em", exact_match)],
}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-samples", type=int, default=50,
                    help="Samples per task (cap). Full eval uses 200.")
    ap.add_argument("--tasks", type=str, default=None,
                    help="Comma-separated task names (default subset of 10).")
    ap.add_argument("--apply-modification", type=str, default=None,
                    help="Free-form tag recorded in output JSON (does not modify the model here).")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device
    print(f"device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    print(f"loading {args.model_id}...")
    try:
        from transformers import Gemma4ForConditionalGeneration as TCls
    except Exception:
        from transformers import AutoModelForCausalLM as TCls
    from transformers import AutoTokenizer
    tgt = TCls.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map=device)
    tgt.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `pip install datasets`"); return 1

    tasks = DEFAULT_TASKS
    if args.tasks:
        requested = set(args.tasks.split(","))
        tasks = [(n, t) for n, t in DEFAULT_TASKS if n in requested]
        if not tasks:
            tasks = [(n, "qa") for n in requested]  # default to qa metric

    results = {}
    t_start = time.time()

    for task_name, task_type in tasks:
        print(f"\n=== {task_name} ({task_type}) ===")
        try:
            ds = load_dataset("THUDM/LongBench", task_name, split="test")
        except Exception as e:
            print(f"  SKIP: failed to load ({e})")
            continue

        samples = list(ds)[: args.max_samples]
        max_new = MAX_GEN_TOKENS.get(task_type, 64)
        per_metric = {m[0]: [] for m in METRICS.get(task_type, [("f1", f1_score)])}

        for row in tqdm(samples, desc=task_name):
            ctx_text  = row.get("context", row.get("input", ""))
            question  = row.get("input", row.get("question", ""))
            answers   = row.get("answers", row.get("output", ""))
            if isinstance(answers, list): answers = answers[0] if answers else ""

            # Build prompt, truncate context from the START so question at end is preserved
            prompt = PROMPT_TEMPLATE.format(context=ctx_text, question=question)
            ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True,
                                   max_length=args.ctx - max_new).to(device)
            if ids.shape[1] < 32:
                continue

            with torch.no_grad():
                out = tgt.generate(
                    ids, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id, use_cache=True,
                )
            gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            # Chop at newline / period for short-answer tasks
            if task_type in ("qa", "class", "retrieval"):
                gen = gen.split("\n")[0].strip()

            for name, fn in METRICS.get(task_type, [("f1", f1_score)]):
                per_metric[name].append(fn(gen, answers))

        task_result = {m: (sum(vs) / len(vs) if vs else 0.0) for m, vs in per_metric.items()}
        task_result["samples"] = len(samples)
        results[task_name] = task_result
        for m, v in task_result.items():
            if m != "samples":
                print(f"  {m}: {v*100:.2f}")

    elapsed = time.time() - t_start
    out = {
        "model_id": args.model_id,
        "ctx": args.ctx,
        "max_samples_per_task": args.max_samples,
        "apply_modification": args.apply_modification,
        "elapsed_sec": elapsed,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {args.output} ({elapsed / 60:.1f} min)")

    # Summary mean
    means = {}
    for _, task_type in DEFAULT_TASKS:
        metric = METRICS.get(task_type, [("f1", f1_score)])[0][0]
        if metric not in means: means[metric] = []
    for task, r in results.items():
        for m, v in r.items():
            if m == "samples": continue
            means.setdefault(m, []).append(v)
    print("\n── Aggregate ──")
    for m, vs in means.items():
        if vs:
            print(f"  {m} (avg of {len(vs)} tasks): {sum(vs) / len(vs) * 100:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
