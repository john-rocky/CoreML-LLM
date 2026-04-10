#!/usr/bin/env python3
"""Vocabulary pruning analysis for Gemma 4 E2B.

Analyzes the 262,144-token vocabulary and identifies tokens that can be pruned
to reduce embedding table sizes (~2.6 GB down to ~1.0-1.6 GB).

This is a DRY-RUN analysis tool — it does NOT modify any files.

Usage:
    python prune_vocab.py [--model-path PATH] [--top-n 30]
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import unicodedata

DEFAULT_MODEL_PATH = "./output/gemma4-e2b-final/hf_model"

# ── Unicode block classification ─────────────────────────────────────────

# Ranges: (start, end, label)
UNICODE_RANGES = [
    # Latin / ASCII
    (0x0000, 0x007F, "Basic ASCII"),
    (0x0080, 0x00FF, "Latin-1 Supplement"),
    (0x0100, 0x024F, "Latin Extended-A/B"),
    (0x0250, 0x02AF, "IPA Extensions"),
    (0x1E00, 0x1EFF, "Latin Extended Additional"),
    (0x2C60, 0x2C7F, "Latin Extended-C"),
    (0xA720, 0xA7FF, "Latin Extended-D"),
    (0xAB30, 0xAB6F, "Latin Extended-E"),
    # Cyrillic
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic Supplement"),
    (0x2DE0, 0x2DFF, "Cyrillic Extended-A"),
    (0xA640, 0xA69F, "Cyrillic Extended-B"),
    # Greek
    (0x0370, 0x03FF, "Greek"),
    (0x1F00, 0x1FFF, "Greek Extended"),
    # Arabic
    (0x0600, 0x06FF, "Arabic"),
    (0x0750, 0x077F, "Arabic Supplement"),
    (0x08A0, 0x08FF, "Arabic Extended-A"),
    (0xFB50, 0xFDFF, "Arabic Presentation A"),
    (0xFE70, 0xFEFF, "Arabic Presentation B"),
    # Hebrew
    (0x0590, 0x05FF, "Hebrew"),
    (0xFB1D, 0xFB4F, "Hebrew Presentation"),
    # Devanagari / Indic
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    # Thai / Lao / Myanmar
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x1000, 0x109F, "Myanmar"),
    # Georgian / Armenian
    (0x10A0, 0x10FF, "Georgian"),
    (0x0530, 0x058F, "Armenian"),
    # Ethiopian
    (0x1200, 0x137F, "Ethiopic"),
    (0x1380, 0x139F, "Ethiopic Supplement"),
    # Korean
    (0x1100, 0x11FF, "Hangul Jamo"),
    (0x3130, 0x318F, "Hangul Compatibility Jamo"),
    (0xAC00, 0xD7AF, "Hangul Syllables"),
    (0xD7B0, 0xD7FF, "Hangul Jamo Extended"),
    # Japanese
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x31F0, 0x31FF, "Katakana Phonetic Ext"),
    (0xFF65, 0xFF9F, "Halfwidth Katakana"),
    (0x1B000, 0x1B0FF, "Kana Supplement"),
    # CJK (shared Chinese/Japanese/Korean)
    (0x4E00, 0x9FFF, "CJK Unified Ideographs"),
    (0x3400, 0x4DBF, "CJK Extension A"),
    (0x20000, 0x2A6DF, "CJK Extension B"),
    (0x2A700, 0x2B73F, "CJK Extension C"),
    (0x2B740, 0x2B81F, "CJK Extension D"),
    (0xF900, 0xFAFF, "CJK Compatibility Ideographs"),
    (0x2F800, 0x2FA1F, "CJK Compat Supplement"),
    (0x3000, 0x303F, "CJK Symbols & Punctuation"),
    (0x3100, 0x312F, "Bopomofo"),
    (0x31A0, 0x31BF, "Bopomofo Extended"),
    # Fullwidth / Halfwidth
    (0xFF00, 0xFF64, "Fullwidth Latin/Punctuation"),
    (0xFFA0, 0xFFEF, "Halfwidth/Fullwidth Forms"),
    # General punctuation / symbols / math
    (0x2000, 0x206F, "General Punctuation"),
    (0x2070, 0x209F, "Superscripts/Subscripts"),
    (0x20A0, 0x20CF, "Currency Symbols"),
    (0x2100, 0x214F, "Letterlike Symbols"),
    (0x2150, 0x218F, "Number Forms"),
    (0x2190, 0x21FF, "Arrows"),
    (0x2200, 0x22FF, "Mathematical Operators"),
    (0x2300, 0x23FF, "Misc Technical"),
    (0x2400, 0x243F, "Control Pictures"),
    (0x2500, 0x257F, "Box Drawing"),
    (0x2580, 0x259F, "Block Elements"),
    (0x25A0, 0x25FF, "Geometric Shapes"),
    (0x2600, 0x26FF, "Misc Symbols"),
    (0x2700, 0x27BF, "Dingbats"),
    (0x27C0, 0x27EF, "Misc Math Symbols-A"),
    (0x2980, 0x29FF, "Misc Math Symbols-B"),
    (0x2A00, 0x2AFF, "Supplemental Math Operators"),
    # Emoji
    (0x1F300, 0x1F5FF, "Misc Symbols & Pictographs"),
    (0x1F600, 0x1F64F, "Emoticons"),
    (0x1F680, 0x1F6FF, "Transport & Map Symbols"),
    (0x1F700, 0x1F77F, "Alchemical Symbols"),
    (0x1F900, 0x1F9FF, "Supplemental Symbols"),
    (0x1FA00, 0x1FA6F, "Chess Symbols"),
    (0x1FA70, 0x1FAFF, "Symbols Extended-A"),
]


def classify_codepoint(cp: int) -> str:
    """Return a human-readable script/block label for a Unicode codepoint."""
    for start, end, label in UNICODE_RANGES:
        if start <= cp <= end:
            return label
    # Fallback: use unicodedata
    try:
        name = unicodedata.name(chr(cp), "")
    except ValueError:
        name = ""
    if name:
        # Take first word of the unicode name as a rough category
        cat = unicodedata.category(chr(cp))
        return f"Other ({cat}: {name.split()[0]})"
    return f"Unknown (U+{cp:04X})"


def classify_token(token_str: str) -> str:
    """Classify a token string into a broad script category.

    Returns a category label for the dominant script in the token.
    """
    # Strip the sentencepiece leading space marker
    clean = token_str.replace("▁", "").replace("Ġ", "")
    if not clean:
        return "Whitespace/Control"

    # Count codepoints per broad category
    cats = collections.Counter()
    for ch in clean:
        cp = ord(ch)
        if cp < 0x0080:
            cats["ASCII/Latin"] += 1
        elif 0x0080 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF or 0x2C60 <= cp <= 0x2C7F:
            cats["Latin Extended"] += 1
        elif 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF:
            cats["Greek"] += 1
        elif 0x0400 <= cp <= 0x052F or 0x2DE0 <= cp <= 0x2DFF or 0xA640 <= cp <= 0xA69F:
            cats["Cyrillic"] += 1
        elif 0x0530 <= cp <= 0x058F:
            cats["Armenian"] += 1
        elif 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:
            cats["Hebrew"] += 1
        elif (0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F
              or 0x08A0 <= cp <= 0x08FF or 0xFB50 <= cp <= 0xFDFF
              or 0xFE70 <= cp <= 0xFEFF):
            cats["Arabic"] += 1
        elif 0x0900 <= cp <= 0x097F:
            cats["Devanagari"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            cats["Bengali"] += 1
        elif 0x0A00 <= cp <= 0x0A7F:
            cats["Gurmukhi"] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            cats["Gujarati"] += 1
        elif 0x0B00 <= cp <= 0x0B7F:
            cats["Oriya"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            cats["Tamil"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            cats["Telugu"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            cats["Kannada"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            cats["Malayalam"] += 1
        elif 0x0D80 <= cp <= 0x0DFF:
            cats["Sinhala"] += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            cats["Thai"] += 1
        elif 0x0E80 <= cp <= 0x0EFF:
            cats["Lao"] += 1
        elif 0x1000 <= cp <= 0x109F:
            cats["Myanmar"] += 1
        elif 0x10A0 <= cp <= 0x10FF:
            cats["Georgian"] += 1
        elif 0x1200 <= cp <= 0x139F:
            cats["Ethiopic"] += 1
        elif 0x3040 <= cp <= 0x309F:
            cats["Japanese (Hiragana)"] += 1
        elif (0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF
              or 0xFF65 <= cp <= 0xFF9F):
            cats["Japanese (Katakana)"] += 1
        elif (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
              or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF):
            cats["CJK Ideographs"] += 1
        elif 0x3000 <= cp <= 0x303F:
            cats["CJK Symbols"] += 1
        elif (0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F
              or 0xAC00 <= cp <= 0xD7FF):
            cats["Korean"] += 1
        elif 0xFF00 <= cp <= 0xFF64 or 0xFFA0 <= cp <= 0xFFEF:
            cats["Fullwidth/Halfwidth"] += 1
        elif (0x2000 <= cp <= 0x206F or 0x2100 <= cp <= 0x214F
              or 0x2190 <= cp <= 0x27BF):
            cats["Symbols/Punctuation"] += 1
        elif 0x2200 <= cp <= 0x2AFF:
            cats["Math Symbols"] += 1
        elif 0x1F300 <= cp <= 0x1FAFF:
            cats["Emoji"] += 1
        else:
            cat = unicodedata.category(chr(cp))
            if cat.startswith("C"):
                cats["Control"] += 1
            elif cat.startswith("Z"):
                cats["Whitespace"] += 1
            elif cat.startswith("P"):
                cats["Punctuation"] += 1
            elif cat.startswith("S"):
                cats["Symbols"] += 1
            else:
                cats["Other"] += 1

    if not cats:
        return "Empty"
    return cats.most_common(1)[0][0]


def is_special_token(token_str: str, token_id: int, tokenizer) -> bool:
    """Check if a token is a special/control token."""
    # Check explicit special tokens
    if token_id in tokenizer.all_special_ids:
        return True
    # Check for angle-bracket special tokens like <bos>, <|image|>, <turn|>, etc.
    if re.match(r"^<[^>]+>$", token_str):
        return True
    if re.match(r"^<\|[^>]+>$", token_str):
        return True
    if re.match(r"^<[^>]+\|>$", token_str):
        return True
    return False


def should_keep(token_str: str, token_id: int, category: str, tokenizer) -> tuple[bool, str]:
    """Decide whether a token should be kept.

    Returns (keep: bool, reason: str).
    """
    # 1. Always keep special tokens
    if is_special_token(token_str, token_id, tokenizer):
        return True, "special token"

    # 2. Always keep ASCII/Latin tokens (English)
    if category in ("ASCII/Latin", "Latin Extended"):
        return True, "Latin/ASCII"

    # 3. Always keep Japanese tokens
    if category in ("Japanese (Hiragana)", "Japanese (Katakana)", "CJK Ideographs", "CJK Symbols"):
        return True, "Japanese/CJK"

    # 4. Always keep fullwidth (used in Japanese text)
    if category == "Fullwidth/Halfwidth":
        return True, "Fullwidth (JP)"

    # 5. Keep common symbols, punctuation, math, emoji
    if category in ("Symbols/Punctuation", "Math Symbols", "Emoji",
                     "Punctuation", "Symbols", "Greek"):
        return True, "symbols/punctuation"

    # 6. Keep whitespace/control
    if category in ("Whitespace/Control", "Whitespace", "Control", "Empty"):
        return True, "whitespace/control"

    # 7. Keep tokens that are purely numeric or common programming tokens
    clean = token_str.replace("▁", "").replace("Ġ", "")
    if clean and all(c.isdigit() or c in "._+-eExX" for c in clean):
        return True, "numeric"

    # 8. Prune other scripts
    return False, f"prunable ({category})"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Gemma 4 E2B vocabulary for pruning opportunities"
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to HF model directory with tokenizer files"
    )
    parser.add_argument(
        "--top-n", type=int, default=30,
        help="Show top N sample tokens per category"
    )
    args = parser.parse_args()

    # ── Load tokenizer ────────────────────────────────────────────────
    print(f"Loading tokenizer from {args.model_path}...")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    print(f"Vocabulary size: {vocab_size:,} tokens")
    print()

    # ── Build reverse mapping (id -> token string) ────────────────────
    id_to_token = {v: k for k, v in vocab.items()}

    # ── Classify every token ──────────────────────────────────────────
    print("Classifying tokens...")
    category_counts = collections.Counter()
    keep_counts = collections.Counter()
    prune_counts = collections.Counter()

    keep_ids = []
    prune_ids = []
    token_details = {}  # id -> (token_str, category, keep, reason)

    for token_id in range(vocab_size):
        token_str = id_to_token.get(token_id, f"<MISSING_{token_id}>")
        category = classify_token(token_str)

        if is_special_token(token_str, token_id, tokenizer):
            category = "Special"

        keep, reason = should_keep(token_str, token_id, category, tokenizer)

        category_counts[category] += 1
        if keep:
            keep_counts[category] += 1
            keep_ids.append(token_id)
        else:
            prune_counts[category] += 1
            prune_ids.append(token_id)

        token_details[token_id] = (token_str, category, keep, reason)

    # ── Report ────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("VOCABULARY ANALYSIS REPORT")
    print("=" * 80)
    print()

    # Summary
    n_keep = len(keep_ids)
    n_prune = len(prune_ids)
    pct_prune = 100.0 * n_prune / vocab_size
    print(f"Total tokens:    {vocab_size:>10,}")
    print(f"Tokens to KEEP:  {n_keep:>10,}  ({100 * n_keep / vocab_size:.1f}%)")
    print(f"Tokens to PRUNE: {n_prune:>10,}  ({pct_prune:.1f}%)")
    print()

    # Size estimates
    # embed_tokens: vocab × 1536 INT8
    # embed_tokens_per_layer: vocab × 8960 INT8
    # scales: vocab × 2 × float16 (each)
    embed_dim = 1536
    per_layer_dim = 8960
    bytes_per_token = embed_dim + per_layer_dim + 2 * 2 + 2 * 2  # int8 + scales

    original_mb = vocab_size * bytes_per_token / (1024 * 1024)
    pruned_mb = n_keep * bytes_per_token / (1024 * 1024)
    saved_mb = original_mb - pruned_mb

    print("── Size Estimates ──")
    print(f"Bytes per token (embed + per_layer + scales): {bytes_per_token:,}")
    print()
    print(f"Original embedding size: {original_mb:>10.1f} MB  ({vocab_size:,} tokens)")
    print(f"After pruning:           {pruned_mb:>10.1f} MB  ({n_keep:,} tokens)")
    print(f"Space saved:             {saved_mb:>10.1f} MB  ({100 * saved_mb / original_mb:.1f}%)")
    print()

    # Detailed breakdown by file
    print("── Per-file breakdown ──")
    for name, dim, dtype_size in [
        ("embed_tokens_q8.bin", embed_dim, 1),
        ("embed_tokens_per_layer_q8.bin", per_layer_dim, 1),
        ("embed_tokens_scales.bin", 2, 2),
        ("embed_tokens_per_layer_scales.bin", 2, 2),
    ]:
        orig = vocab_size * dim * dtype_size / (1024 * 1024)
        after = n_keep * dim * dtype_size / (1024 * 1024)
        print(f"  {name:<45s}  {orig:>8.1f} MB -> {after:>8.1f} MB  (save {orig - after:.1f} MB)")
    print()

    # Category breakdown table
    print("── Tokens by Category ──")
    print(f"{'Category':<30s} {'Total':>8s} {'Keep':>8s} {'Prune':>8s}")
    print("-" * 58)
    for cat, count in category_counts.most_common():
        k = keep_counts.get(cat, 0)
        p = prune_counts.get(cat, 0)
        marker = "" if p == 0 else " <<<" if p > 1000 else " <"
        print(f"  {cat:<28s} {count:>8,} {k:>8,} {p:>8,}{marker}")
    print()

    # Sample tokens from each PRUNABLE category
    prunable_cats = [cat for cat, cnt in prune_counts.most_common() if cnt > 0]
    if prunable_cats:
        print("── Sample PRUNABLE Tokens ──")
        for cat in prunable_cats:
            samples = []
            for tid in prune_ids:
                t_str, t_cat, _, _ = token_details[tid]
                if t_cat == cat:
                    samples.append((tid, t_str))
                    if len(samples) >= args.top_n:
                        break
            print(f"\n  {cat} ({prune_counts[cat]:,} tokens):")
            for tid, t_str in samples:
                display = repr(t_str) if len(t_str) > 40 else t_str
                print(f"    [{tid:>6d}] {display}")
        print()

    # Sample KEPT tokens
    print("── Sample KEPT Tokens (by category) ──")
    kept_cats = [cat for cat, cnt in keep_counts.most_common() if cnt > 0]
    for cat in kept_cats[:15]:  # Show top 15 kept categories
        samples = []
        for tid in keep_ids:
            t_str, t_cat, _, reason = token_details[tid]
            if t_cat == cat:
                samples.append((tid, t_str, reason))
                if len(samples) >= 10:
                    break
        print(f"\n  {cat} ({keep_counts[cat]:,} tokens, reason: {samples[0][2] if samples else '?'}):")
        for tid, t_str, _ in samples:
            display = repr(t_str) if len(t_str) > 40 else t_str
            print(f"    [{tid:>6d}] {display}")
    print()

    # Special tokens listing
    print("── Special Tokens ──")
    special_count = 0
    for tid in range(vocab_size):
        t_str, t_cat, _, reason = token_details[tid]
        if t_cat == "Special":
            if special_count < 50:
                print(f"  [{tid:>6d}] {t_str}")
            special_count += 1
    if special_count > 50:
        print(f"  ... and {special_count - 50} more special tokens")
    print(f"  Total special tokens: {special_count}")
    print()

    # Final summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Keeping {n_keep:,} / {vocab_size:,} tokens ({100 * n_keep / vocab_size:.1f}%)")
    print(f"  Pruning {n_prune:,} tokens ({pct_prune:.1f}%)")
    print(f"  Estimated savings: {saved_mb:.0f} MB ({100 * saved_mb / original_mb:.1f}%)")
    print()
    print("  Kept scripts: ASCII/Latin, Japanese (Hiragana/Katakana/CJK), Fullwidth,")
    print("                Greek, Symbols, Punctuation, Math, Emoji, Special, Control")
    print()
    print("  Pruned scripts: Cyrillic, Arabic, Hebrew, Devanagari, Bengali, Tamil,")
    print("                  Telugu, Kannada, Malayalam, Thai, Korean, Georgian,")
    print("                  Armenian, Ethiopic, Myanmar, Lao, Sinhala, etc.")
    print()
    print("  NOTE: This is a dry-run analysis. No files were modified.")
    print("  To proceed with pruning, review the above report and decide which")
    print("  categories to include/exclude, then update the should_keep() function.")
    print()


if __name__ == "__main__":
    main()
