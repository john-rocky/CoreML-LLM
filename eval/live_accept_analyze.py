#!/usr/bin/env python3
"""Parse [SpecProfile union] lines from smoke logs and compute per-source
accept-rate stats + tok/s breakdown."""
import re, sys, pathlib, statistics

BURST_RE = re.compile(
    r"\[SpecProfile union #(\d+) src=(\w+)\] draft_total=([\d.]+)ms "
    r"\(cv=([\d.]+) pl3=([\d.]+) pl2=([\d.]+)\) verify=([\d.]+)ms "
    r"commit=[\d.]+ms accepted=(\d+)/(\d+) emitted=(\d+)"
)
FB_RE   = re.compile(r"\[SpecProfile union #\d+ fallback\] target_step=([\d.]+)ms")
TOKPS_RE = re.compile(r"\[smoke\] tok/s = ([\d.]+)")

def analyze(path):
    text = pathlib.Path(path).read_text()
    bursts, fbs = [], []
    for m in BURST_RE.finditer(text):
        bursts.append(dict(
            cyc=int(m.group(1)), src=m.group(2),
            draft=float(m.group(3)), cv=float(m.group(4)),
            pl3=float(m.group(5)), pl2=float(m.group(6)),
            verify=float(m.group(7)),
            acc=int(m.group(8)), cmp=int(m.group(9)),
            emit=int(m.group(10))))
    for m in FB_RE.finditer(text):
        fbs.append(float(m.group(1)))
    tokps_m = TOKPS_RE.search(text)
    tokps = float(tokps_m.group(1)) if tokps_m else None

    by_src = {}
    for b in bursts:
        s = b["src"]
        by_src.setdefault(s, []).append(b)

    lines = [f"\n=== {pathlib.Path(path).name} ==="]
    lines.append(f"total bursts: {len(bursts)}, fallback steps: {len(fbs)}, tok/s: {tokps}")
    lines.append(f"burst src histogram: "
                 + ", ".join(f"{s}={len(by_src[s])}" for s in sorted(by_src)))
    for s in sorted(by_src):
        bs = by_src[s]
        rates = [b["acc"]/max(b["cmp"],1) for b in bs]
        emits = [b["emit"] for b in bs]
        zeros = sum(1 for r in rates if r == 0)
        lines.append(
            f"  {s}: n={len(bs)} "
            f"mean_acc_rate={statistics.mean(rates):.3f} "
            f"median_emit={statistics.median(emits):.1f} "
            f"zero_acc={zeros}/{len(bs)} "
            f"mean_verify={statistics.mean(b['verify'] for b in bs):.2f}ms "
            f"mean_draft={statistics.mean(b['draft'] for b in bs):.2f}ms")

    if fbs:
        lines.append(f"fallback target_step: mean={statistics.mean(fbs):.2f}ms "
                     f"n={len(fbs)} (≈{1000/statistics.mean(fbs):.1f} tok/s baseline)")
    all_rates = [b["acc"]/max(b["cmp"],1) for b in bursts]
    if all_rates:
        lines.append(f"OVERALL accept rate: {statistics.mean(all_rates):.3f}")
    return "\n".join(lines)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(analyze(p))
