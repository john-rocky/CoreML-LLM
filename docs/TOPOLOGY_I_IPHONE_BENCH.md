# Topology I 3-chunk boundary search — dead on iPhone A19 Pro (2026-04-24)

## Context

The shipped 3-chunk decode (Topology II, PR #131) merges chunk2+chunk3
into a 17-layer middle block:

    Topology II:  (0,8) + (8,25) + (25,35+head)

Topology I tries the other natural split — merge chunk1+chunk2 into a
15-layer first block:

    Topology I:   (0,15) + (15,25) + (25,35+head)

Both are 3 ANE dispatches. The question: does 15-layer first beat
17-layer middle?

## Measurement

iPhone 17 Pro (A19 Pro) / iOS 26 / Gemma 4 E2B / INT4 palettization,
steady-state decode after prewarm, same prompts across runs.

| Mode | steady tok/s | c1 | c2 | c3 | c4 | sum |
|---|---:|---:|---:|---:|---:|---:|
| **Topology II** (shipped) | **34.2** | 5.6 | 12.2 | 0.0 | 10.9 | 28.8 ms |
| Topology I | 31.7 | 12.0 | 8.2 | 0.0 | 10.7 | 30.8 ms |

Delta: **Topology II is +2.5 tok/s (+7.9%) over Topology I**.
Same direction, larger magnitude than Mac (Topology II +1.6 tok/s on
Mac vs +2.5 tok/s on iPhone).

## Per-layer cost breakdown

Topology II chunk2 (MergedChunk23, 17 layers, pure attention):  
  12.2 / 17 = **0.72 ms/layer**

Topology I chunk1 (BigChunk1, 15 layers + PLE projection):  
  12.0 / 15 = **0.80 ms/layer**

Topology I chunk2 (SWAChunk3, 10 layers pure shared):  
  8.2 / 10 = **0.82 ms/layer**

Per-layer cost is **10-15% worse** under Topology I. The difference
tracks the PLE-projection inclusion: BigChunk1 combines per-layer
embedding projection + RMSNorm + 15 attention layers in one dispatch,
while Topology II isolates the PLE as a small chunk1 (just PLE +
L0-7 attention = 8 layers) and keeps the big chunk2 pure attention.
ANE appears to disfavor mixing compute kinds inside a single dispatch
at the 15-17 layer scale — possibly due to SRAM-tiling boundaries
changing when PLE tensors move through the same working set.

## Verdict

- **Topology I rejected.** Slower on both Mac and iPhone.
- **Topology II remains the shipping 3-chunk configuration.**
- 3-chunk boundary search at this layer-count envelope is exhausted —
  no sweep candidate beat the shipped split.

## Bonus: Topology II scaled up on the latest build

Earlier iPhone A19 Pro measurement of Topology II was 33.0 tok/s
(against 31.6 4-chunk baseline → +4.4%). This session's build hit
**34.2 tok/s** — +1.2 tok/s over the PR #131 baseline measurement.
Likely drivers: (a) ComputePlanAudit fixes + audit/cherry-pick clean
up landed since the earlier bench, (b) thermal state coolers between
runs. Either way, the real 3-chunk gain vs 4-chunk baseline on this
build is **+2.6 tok/s / +8.2%**, not +4.4%.

## Reproduce

Topology I bundle and Swift routing shipped opt-in:

    LLM_3CHUNK=1 LLM_3CHUNK_TOPO=I

Shipped bundle needs the three Topology I mlmodelc files
(`chunk{1,2,3}_topoI.mlmodelc`) alongside the Topology II files.
`conversion/build_gemma4_topology_I.py` + `install_topoI_bundle.py`
produces and installs them.
