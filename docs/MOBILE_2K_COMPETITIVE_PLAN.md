# Mobile 2K competitive speedup plan

**Goal:** beat Google's Gemma 4 E2B mobile deployment at ctx=2048 by a
clear, measurable margin on iPhone 17 Pro (A19 Pro).

**Reference points:**

| Platform | Engine | tok/s @ 2K (Gemma 4 E2B) | Source |
|---|---|---|---|
| **iPhone (LiteRT-LM iOS build)** | Google LiteRT-LM | **56** | Google's own iOS benchmark (user-reported 2026-04-14) |
| Pixel 9 Pro (Tensor G4) | LiteRT-LM Android | 25–30 | public micro-benches; `docs/LITERT_RUNTIME_ANALYSIS.md` |
| iPhone 15 Pro (A17 Pro) | CoreML (AFM-2B) | ~30 | Apple AFM arxiv 2507.13575 |
| iPhone 17 Pro (A19 Pro) | **our stack today** | **31.4** | decode profile 2026-04-14 (c1=5.9, c2=6.8, c3=8.1, c4=10.4 ms) |

So **on the same hardware class (iOS)** we're at 31 tok/s versus
Google's LiteRT-LM doing 56. They're roughly 1.8× ahead of us on
iPhone specifically. Target for this plan: **70–110 tok/s** at 2K —
i.e. 1.25–2× Google's iOS build, and 2–4× their Android number.

Google's 56-tok/s iOS number implies they've already extracted a
significant amount of the Apple-silicon hybrid-compute win. Our
Tier 2 / Tier 4 async dispatch infrastructure is how we match and
exceed that; today's stack doesn't schedule GPU alongside ANE at all.

---

## Critical question: is drafter training required?

No, but the training-free path caps lower. Two forking routes:

### Route A — drafter-trained (user is here)
Needs A100-class GPU for ~8–30 hours of distill, depending on drafter
architecture.

- MTP self-trained heads (other session in progress)
- HASS / EAGLE-3 retrain
- Clover-2

**Ceiling: ~110 tok/s** if all tiers land.

### Route B — drafter-free
Zero training. Uses pre-trained drafters or prompt-based heuristics.

- **Cross-vocabulary SD** with Qwen 2.5 0.5B — drafter is a publicly-
  shipped pre-trained model; we only adapt vocab at inference time.
  Already bundled in the repo's ModelDownloader defaults.
- **Prompt Lookup Decoding** — n-gram match against prompt history
  (algorithm already merged in PR #36, wiring pending).
- **SuffixDecoding** — suffix-tree built from session history.
- **Harmony-Decoding** — target's own shallow layers as drafter, no
  training, smart phase gate.

**Ceiling: ~85 tok/s** with Mirror SD + staged pipelining on top.

**Both routes share the same async dispatch infrastructure**, so
investing in it pays off regardless of which drafter lands.

---

## Recommended route, 2K-focused

Assumes MTP training (user's other session) or a drafter-free
equivalent lands with ≥ 50% accept. Every tier composes with the
previous.

| Week | Item | Dependency | 2K tok/s (cumulative) |
|---|---|---|---|
| 0 | Current baseline (measured) | — | **31** |
| 1–2 | **Drafter + Q=K verify + KV direct-write**. If trained: MTP or HASS. If not: Cross-vocab SD with Qwen 0.5B + Prompt Lookup wiring. | Drafter artefact ready | **60–62** |
| 3 | **Mirror SD** (GPU drafter asynchronous with ANE target verify) | Async dispatch infra | **82–85** |
| 4 | **Union-of-drafters** — Prompt Lookup ∪ SuffixDecoding ∪ {trained drafter \| Qwen}, verify once on miss | Drafter + aux wiring | **94–98** |
| 5 | **Staged chunk pipelining** — step N+1's chunk1 runs concurrently with step N's chunk3/4 | Async dispatch infra | **115–125** |

Best case by week 5: **120 tok/s @ 2K**. Conservative: **85 tok/s**.
Either clears Google by > 2×.

Route B (no training anywhere) tops out around week 5 at ~85 tok/s
because Cross-vocab Qwen's accept rate (40–50 %) is lower than a
well-trained MTP's (55–65 %).

---

## Infrastructure line items (shared, load-bearing)

These unlock multiple tiers. Build them once.

### I1. Async ANE dispatch queues — 4–6 days
- Multiple `MLModel.prediction` can be kicked off before earlier ones
  return, giving an ordering-safe pipeline across chunks.
- Needed for both Mirror SD (GPU drafter concurrent with ANE verify)
  and staged pipelining (chunk N+1 / chunk N concurrent).
- Risk: ANE has limited in-flight requests (AsahiLinux RE suggests 8
  per handle). Need back-pressure.
- Reference: `github.com/AsahiLinux/docs/wiki/HW:ANE`, `eiln/ane`.

### I2. GPU drafter execution path — 3–4 days
- Compile drafter artefact with `computeUnits = .cpuAndGPU` (avoid
  ANE to free the ANE for target).
- Expose tensors via `IOSurface` so the drafter's hidden-state output
  can be consumed by ANE verify without CPU copy.
- Unified memory (Apple silicon UMA) makes cross-unit tensor passing
  essentially free in bandwidth — a Google Tensor G4 disadvantage.

### I3. KV direct-write in `commitAccepted` — 1–2 days
- Already scoped as roadmap item 9 (EAGLE-3 Blocker 2).
- Without this, every accepted speculative token costs one extra T=1
  decode call, erasing most of the speedup.
- Swift-only change; writes the accepted KV slice back into the
  sliding/full cache buffers in place.

---

## Per-tier detail

### Tier 1 — Drafter + verify (×2.0)

Pick ONE drafter source. They all plug into the same `verifyCandidates`
code path, so switching cost is low once the verify path works.

- **MTP self-trained** (user's in-flight work) — best expected accept
  rate; tight fit with our target's KV-sharing forward.
- **HASS** (arxiv 2408.15766) — EAGLE variant that trains against
  inference-mode target outputs, directly fixing our Blocker 1 root
  cause. ~1 A6000-day to train. Drop-in against our existing verify
  chunks.
- **Cross-vocab Qwen 2.5 0.5B** — no training; requires a vocab
  translation layer (simple lookup). Accept rate lower.

### Tier 2 — Mirror SD (×1.35)

While the target runs `verifyCandidates` on ANE for burst k, the
drafter runs on GPU preparing burst k+1. Latency of the target verify
step hides the drafter cost entirely.

- Requires I1 + I2.
- Apple's unified memory is the key enabler: ship the drafter output
  directly as an `IOSurface`-backed tensor to the ANE `kv13`/`kv14`
  inputs. No staging copy.
- Caveat: drafter forward + target verify must not contend on the same
  memory region. Use separate scratch pools.

### Tier 3 — Union-of-drafters (×1.15–1.3)

Each drafter source has strong workloads and weak ones. Union takes
the longest-accepted prefix across all sources, paying for one verify
pass when all miss.

- Sources to stack:
  - Prompt Lookup (code / summaries / QA — hits when answer quotes
    prompt).
  - SuffixDecoding (multi-turn sessions — hits on self-repetition).
  - MTP / HASS / Qwen (novel text).
- Workload mix on typical chat ≈ 20–30 % prompt-quoting, 70–80 % novel
  → +15–20 % over best single. Code-heavy ≈ +40 %.

### Tier 4 — Staged chunk pipelining (×1.25)

The current decode runs chunk1 → chunk2 → chunk3 → chunk4 strictly
serial per step. Because chunk3 and chunk4 are KV-read-only (shared
kv13/kv14 from chunk2), step N+1's chunk1 can start the moment step
N's chunk2 finishes. That hides roughly one chunk of latency per step.

- Depends on I1.
- Risk: requires ANE async dispatch with strict ordering guarantees.
  The current 0f audit found our pipeline synchronous-safe; this
  pushes into the asynchronous regime, so ping-pong / ring buffers
  (also 0f) become necessary.

---

## TTFT as a separate axis

Decode tok/s is the main metric, but a poor TTFT ruins UX. Covered
separately (roadmap Phase 5):

- **GPU prefill via MLX-Swift** — crunches prompt on GPU tensor cores,
  ANE handles decode. Cuts TTFT 13s → 1s at 512 prompt tokens. 7–10
  days (MLX-Swift is recent and underdocumented).
- **Per-chunk streaming** — emit first decoded token immediately after
  chunk4's LM head, before chunks 3/4 of the next position start.
- **Prefix / system prompt KV caching** — persist the cache across
  app launches so multi-turn conversations skip prefill entirely.

None of these is on the critical path for tok/s, but they all matter
for user perception.

---

## Items explicitly OUT of scope for 2K

- **MoE retrofit** — the compute savings dominate at 8K+; at 2K the
  FFN cost is already overwhelmed by dispatch. Defer.
- **W2 quantization** — bandwidth-limited only matters when weight
  load is the bottleneck, which it isn't at 2K decode. Quality risk
  not justified.
- **BitNet b1.58** — requires QAT from scratch; disproportionate
  effort for marginal gain at 2K.
- **TEAL / Deja Vu activation sparsity** — quality-lossy, and the FFN
  time we'd shave is already small at 2K.
- **RWKV / Mamba / non-Transformer swap** — 10–14-day conversions +
  distill. Decouple from the "beat Google at 2K" goal.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Async dispatch infrastructure (I1) hits an iOS 18 undocumented quirk | Falls back to synchronous dispatch + pipelining dies quietly; Tier 1 still delivers 2× |
| MTP self-training fails (user's session) | Route B with Cross-vocab Qwen still gets to ~85 tok/s |
| Mirror SD's GPU drafter contends with ANE for memory | Separate scratch regions + profile via Instruments' Metal system trace |
| Staged pipelining exposes the ANE async race ANEMLL warned about | Implement ping-pong / ring buffers as prerequisite (roadmap 0f upgrade) |

---

## How this relates to other roadmap docs

- `PRIORITY_ROADMAP.md` is the comprehensive menu; this doc is the
  ordered execution path for the 2K competitive goal.
- `UNEXPLORED_APPROACHES_V6.md` lists the lossless source candidates
  (V6-4 through V6-11) that back the tiers here.
- `EAGLE3_INTEGRATION_STATE.md` has the verify-chunk I/O contract
  that the drafter tiers consume.
- `MTP_PATH_A_FINDINGS.md` documents why we pivoted to self-trained
  MTP after Google's bundled drafter failed.
