#!/usr/bin/env python3
"""Analyze accept-rate-bench JSON outputs.

Computes E[tok/burst] per drafter per mode, aggregated per category, and
emits plain ASCII tables. Handles the v2/v3/v4 schema evolution:

  v2: only `drafters` (oracle replay).
  v3: adds `draftersArgmax`, `emittedArgmaxLen`, `decodeVerifyAgreePrefix`.
  v4: adds `draftersChain`.

Older files just leave newer fields empty; v4 is the superset.

Usage:
    analyze_bench.py FILE [FILE ...] [--summary-only]

When two or more JSON files are passed, the per-category aggregate table
shows deltas between the first file and each additional file (chain-mode
only, that being the live-equivalent number per PHASE_B_V4 findings).

Stdlib only. No external deps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


MODES = [
    ("drafters", "oracle"),
    ("draftersArgmax", "argmax"),
    ("draftersChain", "chain"),
]


def expected_tokens_per_burst(hist: list[int], total_bursts: int) -> float:
    """1 + sum of conditional chainAccept probabilities.

    Mirrors `DrafterStats.expectedTokensPerBurst` in
    `Sources/accept-rate-bench/Drafters.swift`. histogram index k =
    number of drafter-proposed tokens accepted in that burst.
    """
    if total_bursts <= 0:
        return 0.0
    out = 1.0
    remaining = total_bursts
    for k in range(len(hist) - 1):
        accepted = sum(hist[k + 1:])
        if remaining > 0:
            out += accepted / remaining
        remaining = accepted
    return out


def aggregate(doc: dict[str, Any]) -> dict[str, dict[str, dict[str, tuple[list[int], int]]]]:
    """Return agg[category][drafter][mode] = (summed_hist, total_bursts)."""
    agg: dict[str, dict[str, dict[str, tuple[list[int], int]]]] = {}
    for prompt in doc.get("prompts", []):
        category = prompt.get("category", "?")
        cat_block = agg.setdefault(category, {})
        for mode_key, mode_label in MODES:
            mode_block = prompt.get(mode_key) or {}
            for drafter, stats in mode_block.items():
                hist = list(stats.get("histogram", []))
                total = int(stats.get("totalBursts", 0))
                d_block = cat_block.setdefault(drafter, {})
                existing = d_block.get(mode_label)
                if existing is None:
                    d_block[mode_label] = (hist, total)
                else:
                    old_hist, old_total = existing
                    merged = list(old_hist)
                    if len(hist) > len(merged):
                        merged.extend([0] * (len(hist) - len(merged)))
                    for i, c in enumerate(hist):
                        merged[i] += c
                    d_block[mode_label] = (merged, old_total + total)
    return agg


def _pad(s: str, w: int, right: bool = False) -> str:
    if len(s) >= w:
        return s
    pad = " " * (w - len(s))
    return (s + pad) if not right else (pad + s)


def print_per_prompt(doc: dict[str, Any], label: str) -> None:
    print()
    print(f"== {label}: per-prompt ==")
    for prompt in doc.get("prompts", []):
        pid = prompt.get("id", "?")
        cat = prompt.get("category", "?")
        line = f"- {pid} [{cat}] promptLen={prompt.get('promptLen', '?')}"
        fields = [("emittedLen", prompt.get("emittedLen"))]
        if "emittedArgmaxLen" in prompt:
            fields.append(("emittedArgmaxLen", prompt.get("emittedArgmaxLen")))
        if "decodeVerifyAgreePrefix" in prompt:
            fields.append(("decodeVerifyAgreePrefix", prompt.get("decodeVerifyAgreePrefix")))
        for k, v in fields:
            if v is not None:
                line += f" {k}={v}"
        print(line)
        # Per-drafter per-mode E[tok/burst]
        drafter_names: list[str] = []
        seen: set[str] = set()
        for mode_key, _ in MODES:
            for name in (prompt.get(mode_key) or {}).keys():
                if name not in seen:
                    seen.add(name)
                    drafter_names.append(name)
        if not drafter_names:
            continue
        header = "  drafter".ljust(28) + "oracle   argmax   chain"
        print(header)
        for d in drafter_names:
            row = f"    {_pad(d, 24)}"
            for mode_key, _ in MODES:
                block = (prompt.get(mode_key) or {}).get(d)
                if block is None:
                    row += "   -    "
                else:
                    e = expected_tokens_per_burst(
                        list(block.get("histogram", [])),
                        int(block.get("totalBursts", 0)))
                    row += f"  {e:5.2f}  "
            print(row)


def print_category_aggregate(aggs: list[tuple[str, dict]], summary_only: bool) -> None:
    if not aggs:
        return
    print()
    print("== per-category aggregate: E[tok/burst] ==")
    base_label, base_agg = aggs[0]
    categories: list[str] = []
    seen_cats: set[str] = set()
    for _, a in aggs:
        for cat in a.keys():
            if cat not in seen_cats:
                seen_cats.add(cat)
                categories.append(cat)
    categories.sort()
    # Header: category | drafter | oracle | argmax | chain [| Δchain vs base for each extra file]
    extra_labels = [label for label, _ in aggs[1:]]
    header = (_pad("category", 10) + _pad("drafter", 22)
              + _pad("oracle", 9, right=True) + _pad("argmax", 9, right=True)
              + _pad("chain", 9, right=True))
    for el in extra_labels:
        header += _pad(f"Δchain({_short(el)})", 18, right=True)
    print(header)
    print("-" * len(header))
    for cat in categories:
        drafters: list[str] = []
        seen: set[str] = set()
        for _, a in aggs:
            for d in (a.get(cat) or {}).keys():
                if d not in seen:
                    seen.add(d)
                    drafters.append(d)
        for d in drafters:
            row = _pad(cat, 10) + _pad(d, 22)
            vals: dict[str, float | None] = {}
            for mode_key, mode_label in MODES:
                block = ((base_agg.get(cat) or {}).get(d) or {}).get(mode_label)
                if block is None:
                    vals[mode_label] = None
                    row += _pad("-", 9, right=True)
                else:
                    e = expected_tokens_per_burst(*block)
                    vals[mode_label] = e
                    row += _pad(f"{e:.2f}", 9, right=True)
            base_chain = vals.get("chain")
            for other_label, other_agg in aggs[1:]:
                other_block = ((other_agg.get(cat) or {}).get(d) or {}).get("chain")
                if other_block is None:
                    row += _pad("-", 18, right=True)
                else:
                    e_other = expected_tokens_per_burst(*other_block)
                    if base_chain is None:
                        row += _pad(f"{e_other:.2f} (new)", 18, right=True)
                    else:
                        delta = e_other - base_chain
                        row += _pad(f"{e_other:.2f} ({delta:+.2f})", 18, right=True)
            print(row)


def _short(label: str) -> str:
    base = os.path.basename(label)
    if base.endswith(".json"):
        base = base[:-5]
    if base.startswith("accept-rate-bench-"):
        base = base[len("accept-rate-bench-"):]
    return base[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="accept-rate-bench-*.json paths")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip per-prompt detail; print only the category aggregate.")
    args = parser.parse_args()

    docs: list[tuple[str, dict]] = []
    for path in args.files:
        try:
            with open(path) as f:
                doc = json.load(f)
        except OSError as e:
            print(f"error: {path}: {e}", file=sys.stderr)
            return 1
        docs.append((path, doc))

    for path, doc in docs:
        label = _short(path)
        print(f"# {path}")
        meta = [f"mode={doc.get('mode', '?')}", f"K={doc.get('K', '?')}",
                f"maxTokens={doc.get('maxTokens', '?')}",
                f"prompts={len(doc.get('prompts', []))}",
                f"generatedAt={doc.get('generatedAt', '?')}"]
        print("  " + "  ".join(meta))
        if not args.summary_only:
            print_per_prompt(doc, label)

    aggs = [(path, aggregate(doc)) for path, doc in docs]
    print_category_aggregate(aggs, args.summary_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
