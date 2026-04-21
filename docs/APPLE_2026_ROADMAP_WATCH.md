# Apple 2026 Roadmap Watch — WWDC 2026 Pre-Flight Assessment

Date: 2026-04-15
Horizon: now → WWDC 2026 (June 8, 2026) → iOS 27 GA (Sep 2026)
Purpose: Decide whether the Metal-LLM and CoreML-LLM tracks should wait, hedge, or proceed.

---

## Executive Verdict: DO-NOT-WAIT (with one small hedge)

**Proceed at full throttle on both tracks.** Nothing credibly rumored for WWDC 2026 invalidates work we can ship before June 8. The only rumor that could substantively reshape our ANE strategy — a "Core AI" framework that exposes direct ANE dispatch — is (a) a Gurman-sourced rename rumor with no technical substance attached, and (b) even in the optimistic case, Apple's ANE opening history (private `_ANEClient`/`_ANECompiler` APIs documented by the external Orion paper, March 2026) suggests any new surface will be additive rather than disruptive. The Foundation Models framework has been shipping since iOS 26 (June 2025) and did *not* displace Core ML for custom models; Core AI, if real, is overwhelmingly likely to follow the same "added surface, old API kept" pattern. The single hedge: delay investing more than ~3 engineer-days into a *new* MLState-dependent architecture before WWDC, because Apple's known stateful-model bug surface is the one thing they could cleanly fix at WWDC and thereby obsolete our workarounds. Everything else — MPSGraph kernels, chunking, EAGLE-3 drafter tuning, speculative decoding plumbing, dequant fast paths — is robust to any plausible announcement and should ship.

---

## 1. Confirmed Apple 2026 Deliverables (primary-source verified)

| Item | Source | Date | Relevance |
|---|---|---|---|
| **iOS 26.4 GA** (build 23E246) | Apple Developer Releases page | 2026-03-24 | Point release; no headline Core ML API changes visible in dev release notes |
| **iOS 26.4.1** (build 23E254) | Apple Developer Releases page | 2026-04-08 | Bug-fix dot release |
| **coremltools 9.0** | GitHub apple/coremltools | 2025-11-10 | Python 3.13, int8 I/O, model state read/write, `AllowLowPrecisionAccumulationOnGPU`, deployment targets iOS26/macOS26. **No 9.1 or 10.0 exists as of 2026-04-15.** |
| **WWDC 2026 dates** | Apple Newsroom | keynote **2026-06-08** | Main event |
| **Apple × Google Gemini deal** | Bloomberg (Gurman), CNBC, TechCrunch | announced 2026-01-12 | Server-side Siri; Private Cloud Compute used. Not an on-device Core ML signal. |
| **MLX + M5 Neural Accelerators** | machinelearning.apple.com/research/exploring-llms-mlx-m5 | late 2025 | MLX now uses per-GPU-core matmul accelerators; 19–27% gen-speed gain M4→M5. GPU-side, not ANE. |
| **Apple ML Research: "Parallel Track Transformers"** | machinelearning.apple.com/research | 2026 | GPU inference, reduced synchronization — relevant to our Metal track, not ANE |
| **"Mirror Speculative Decoding"** (Bhendawade et al.) | Apple ML Research | 2025 | Apple is publicly exploring spec-decode — aligned with our MTP/EAGLE bets |

**Implication:** Apple has *already* shipped the stateful-model APIs (iOS 18, extended in iOS 26 via `model state read/write` in coremltools 9.0). The MLX-on-M5 story is GPU-side and does not threaten ANE work. No confirmed 2026 deliverable changes our plan.

---

## 2. Strong Rumors — Bloomberg/Gurman-tier (probability **Medium-High**)

### 2.1 "Core AI" framework to replace/augment Core ML at WWDC 2026
- **Primary source:** Mark Gurman, Bloomberg *Power On* newsletter (2026-03-01)
- **Secondary coverage:** 9to5Mac, AppleInsider, MacRumors (all downstream of Gurman, no independent corroboration)
- **Gurman's actual claim:** It is a rename/modernization — "machine learning" sounds dated, Apple is "modernizing" around the AI term. **Gurman explicitly states the framework's purpose is unchanged: "helping developers integrate outside AI models into their apps."**
- **Probability of shipping at WWDC 2026:** High (Gurman's WWDC leaks have ~80% hit rate historically)
- **Probability of ANE dispatch opening:** LOW — no source makes this claim. Speculation only.
- **Impact on strategy:** LOW if Gurman is right about the rename nature. Core ML `.mlpackage` files will continue to load on iOS 27. Worst case we add a thin adapter layer.
- **What to watch at WWDC:** Does the "What's New in Core AI" session introduce (a) any public ANE-dispatch API, (b) any custom MIL op registration, (c) any multi-state model support?

### 2.2 Siri 2.0 / Apple Intelligence 2.0 (Gemini-backed, server-side)
- **Primary source:** Bloomberg (Gurman), CNBC, TechCrunch 2026-01-12; Gurman follow-up 2026-01-25
- **Claim:** ~$1B/year Gemini license; Gemini weights run in Apple Private Cloud Compute; partial Feb 2026 rollout (iOS 26.4), full chatbot rebuild at WWDC 2026 for iOS 27
- **Probability:** High (multi-sourced, Apple-confirmed the partnership)
- **Impact on our strategy:** **ZERO.** Server-side. Does not compete with on-device LLM workloads — it's a complement. Arguably *helps* our positioning ("for privacy, run on-device" remains a valid story).

### 2.3 M6 Neural Engine in 50+ TOPS range, A20/A20 Pro on TSMC N2 (2nm)
- **Source:** Industry analysts, supply chain reports; Gurman has mentioned N2 timeline
- **Probability:** High for A20 silicon/shipping in iPhone 18 (Sep 2026); medium-high for M6 2026Q4
- **Impact:** Positive but not disruptive. If M6 Neural Engine jumps 38→50+ TOPS, our ANE code gets a free ~30% boost on new hardware. No API change required.

### 2.4 Xcode 18 with expanded ML tooling
- **Source:** Widely rumored, no specific leak
- **Probability:** Medium (Xcode versions follow macOS)
- **Impact:** Low — tooling improvements help us, don't invalidate us.

---

## 3. Weak Rumors — logged, not actionable

| Rumor | Source tier | Probability | Why we ignore |
|---|---|---|---|
| ANE public dispatch API | Hacker News speculation | Low | No primary source. Apple has never telegraphed this. |
| Custom MIL op registration for third parties | Forum chatter | Low | Would contradict Apple's entire Core ML security model |
| Metal 5 announcement | No credible source | Low | Metal 4 only shipped with iOS 18/macOS 15; jump to Metal 5 at WWDC 2026 would be rushed. Expect Metal 4.x point updates. |
| "Persistent command buffers" / instanced execution | No source | Low | Wish-list, not rumor |
| Foundation Models exposing ≥7B variants | Speculation | Medium | Would require adapter retraining but wouldn't touch our custom Gemma stack |
| coremltools 10.0 at WWDC | Historical pattern | Medium-High | coremltools major bumps do track WWDC. A 10.0 is plausible, but prior major bumps (8.0, 9.0) did *not* break backward compatibility for `.mlpackage`. |
| Multi-state models / state versioning API | Our internal wishlist | Low-Medium | Would help our KV-cache work but no Apple signal |

---

## 4. Competitive Landscape 2026 — on-device LLM race

| Platform | Confirmed delivery | Implication for us |
|---|---|---|
| **Google LiteRT-LM GA** (2026-04-07/08) | Production-grade, speculative decoding, KV-cache, iOS via Metal, Android NPU via Qualcomm/MediaTek/Exynos | This is our direct competitor. Gemma 4 2.5B in <1.5GB / 4k tokens in <3s on phones. We need our own chain-mode spec-decode to match. |
| **Qualcomm Snapdragon X2 Elite** | Hexagon NPU 80 TOPS, ~2× prior gen | Sets a performance bar for Android. Our A19 Pro ANE at ~35 TOPS is outgunned on paper, but Apple wins on memory bandwidth and framework maturity. |
| **MediaTek Dimensity 9500** (Feb 2026) | NPU 990 at 100 TOPS, Generative AI Engine 2.0, BitNet 1-bit, CIM | Android flagship. 128K token window claim is noteworthy — our chunked attention work is still valuable here. |
| **External ANE work — Orion paper** (arXiv 2603.06728, 2026-03-06) | Direct ANE dispatch bypassing Core ML via private `_ANEClient`/`_ANECompiler`; 170+ tok/s GPT-2 124M on M4 Max | **Proof that direct ANE dispatch is technically achievable** but uses private APIs. If we wanted to pursue it, we'd be unshipable in the App Store. Confirms Apple has not opened this surface publicly. |

**Takeaway:** LiteRT-LM is the beat-target. Nothing here suggests we should pause; it suggests we should accelerate.

---

## 5. Impact on Each Planned Track

### Metal-LLM (new repo, Weeks 1–12)
| Week | Planned item | Affected by WWDC 2026? | Recommendation |
|---|---|---|---|
| 1–2 | MPSGraph boilerplate, model loader | No | Proceed |
| 3–4 | SDPA / fused attention kernels | No | Proceed — MPSGraph SDPA is stable API since macOS 15 |
| 5–6 | KV cache, paged attention | No | Proceed |
| 7–8 | Quant (W4A16) dequant on GPU | No | Proceed |
| 9–10 | Speculative decoding integration | No | Proceed — Apple's own Mirror Spec Decoding paper aligns with approach |
| 11–12 | Benchmark parity vs LiteRT-LM, ship v0 | **Possible upside if Metal 4.x point update lands at WWDC** | Proceed; re-plan week 13+ *after* June 8 |

**Verdict: zero items to pause.** Metal APIs are the stablest surface Apple has. MLX adopting Neural Accelerators in M5 GPU cores validates the whole GPU direction.

### CoreML-LLM conversion optimization (D-1 … D-6)
| Step | Affected by WWDC 2026? | Recommendation |
|---|---|---|
| D-1 Conversion audit / MIL pass catalog | No | Proceed |
| D-2 Embedding-bypass + chunk consolidation | No | Proceed — already landed, keep iterating |
| D-3 Stateful KV-cache retry | **Yes — one of the few places a WWDC fix could invalidate workarounds** | **HEDGE:** do exploratory spike only; don't refactor >1 file until we see WWDC release notes |
| D-4 Drafter (MTP / EAGLE-3) CoreML integration | No | Proceed — this is where our moat lives |
| D-5 Chat CV residual / rolling-gate closure | No | Proceed |
| D-6 Track-A tolerance-aware chain-mode | No | Proceed — just shipped #73 |

**Verdict:** only D-3 (stateful KV) gets a soft hedge. Everything else ships.

---

## 6. Signal-to-Noise Summary (per-rumor scorecard)

| Claim | Source tier | Probability | Strategic impact | Action |
|---|---|---|---|---|
| Core AI framework rename at WWDC 2026 | Bloomberg/Gurman | High | Low (additive) | Monitor, no change |
| Core AI opens ANE public dispatch | Speculation only | **Low** | Would be High | Ignore until primary source |
| Gemini-powered Siri | Bloomberg/Apple-confirmed | Confirmed | Zero (server-side) | None |
| M6 50+ TOPS NE | Analysts | High | Positive free lunch | None |
| A20/N2 in iPhone 18 | Supply chain | High | Positive free lunch | None |
| coremltools 10.0 at WWDC | Historical pattern | Medium-High | Low (back-compat) | Plan 1-day upgrade sprint in June |
| Stateful model multi-state API | Internal wish | Low-Medium | Would be High | Defer D-3 refactor |
| Custom MIL op exposure | Forum | Low | Would be High | Ignore |
| Metal 5 | None | Low | N/A | Ignore |
| MLState -14 error fix | None | Unknown | Medium | Test again after each iOS beta |

---

## 7. Budgetary Response: How Much Runway Pre-WWDC?

**Policy:** burn runway aggressively from 2026-04-15 → 2026-06-08 (about 8 weeks).

- Every Metal-LLM week is independent of WWDC; spend those weeks now.
- Every CoreML-LLM conversion win (except D-3 stateful) is independent of WWDC; spend those hours now.
- Cap D-3 investment at ≤3 engineer-days until 2026-06-08. If WWDC doesn't deliver a state-API fix, resume full investment 2026-06-09.
- Reserve a 2–3 day buffer in early June to watch the keynote, State of the Union, and specifically:
  - *"What's new in Core ML / Core AI"* session
  - *"What's new in Metal"* session
  - *"Optimize on-device inference"* session (likely title variant)
- Plan a 1-week "WWDC absorption" sprint 2026-06-09 to 2026-06-16 to pick up any coremltools 10.0, new MPSGraph ops, or API deltas.

**Estimated pre-WWDC runway consumed:** ~85% of planned work ships; 15% deliberately held for post-WWDC re-plan.

---

## 8. Watch List — Re-check Items on June 8, 2026 (WWDC Keynote Day)

Checklist to run the morning after the keynote, highest priority first:

1. **Core AI framework:** does it introduce a new file format? Is `.mlpackage` deprecated? What's the migration path? (Expected: no deprecation.)
2. **ANE public dispatch:** search the iOS 27 SDK for any new public header in `CoreML.framework` or `CoreAI.framework` that takes raw MIL programs, bypasses the compiler, or exposes neural engine queue priority. If found, prioritize a prototype within 1 week.
3. **MLState evolution:** any new APIs around multi-state models, state versioning, or fix for the -14 IOAF crash? Unblocks D-3.
4. **coremltools 10.0 release notes:** check GitHub; focus on any new MIL ops or quantization modes.
5. **MPSGraph updates:** new fused ops? Persistent command buffer? Native paged-KV attention op?
6. **Metal 4.x point update** or Metal 5 announcement.
7. **Foundation Models expansion:** bigger adapter rank? Support for larger base models? On-device speculative decoding API exposure?
8. **New hardware announcements:** if M6 Mac Studio announced, we get free bench headroom.
9. **Xcode 18 ML tooling:** new `.mlpackage` inspector? Better profiler for ANE dispatches?
10. **Private Cloud Compute developer access:** any public API to offload Core ML inference to PCC? Would change our product story.

For each item, if the answer is "no change" → continue current plan. If "yes, material change" → run a 72-hour assessment before re-planning Week 12+ and D-7+.

---

## 9. Sources (tiered by credibility)

**Primary / Apple-published:**
- `https://developer.apple.com/documentation/updates/coreml` — Core ML updates page
- `https://developer.apple.com/documentation/ios-ipados-release-notes/ios-ipados-26_4-release-notes` — iOS 26.4 release notes
- `https://github.com/apple/coremltools/releases` — coremltools 9.0 (2025-11-10) confirmed; no 9.1/10.0 yet
- `https://machinelearning.apple.com/research/exploring-llms-mlx-m5` — MLX on M5 Neural Accelerators
- `https://machinelearning.apple.com/research/` — 2025–2026 paper index
- `https://developer.apple.com/apple-intelligence/foundation-models-adapter/` — LoRA adapter training

**Secondary / Bloomberg-tier (Gurman):**
- 9to5Mac 2026-03-01 "Apple replacing Core ML with modernized Core AI framework" (quotes Gurman)
- AppleInsider 2026-03-01 "WWDC 2026 to introduce Core AI as replacement for Core ML"
- Bloomberg Power On 2026-01-25 (Siri/Gemini timeline)
- CNBC 2026-01-12 / TechCrunch 2026-01-12 (Apple–Google partnership)

**Tertiary / MacRumors / aggregator:**
- MacRumors 2026-03-01 (iOS 27 change leak)
- MacRumors 2026-03-18 (iOS 26.4 release notes summary)

**External research (non-Apple):**
- arXiv 2603.06728 "Orion: Characterizing and Programming Apple's Neural Engine" (2026-03-06) — external work, uses private `_ANEClient`/`_ANECompiler` APIs, not shippable in App Store
- Google LiteRT-LM GA announcement 2026-04-07/08

**Explicitly NOT relied upon:**
- Twitter/X speculation
- Anonymous forum rumor threads
- Non-Gurman "analyst" blog posts

---

## 10. Bottom Line

Ship. The single highest-leverage thing we can do between now and June 8 is to have a demonstrable sub-1.5GB, speculative-decoded, iPhone-17-Pro benchmark that beats LiteRT-LM's 56.5 tok/s. Every week spent waiting for WWDC is a week LiteRT-LM gets better. Apple's 2026 roadmap, as currently visible, does not reward caution — it rewards Core ML mastery and Metal/MPSGraph depth, both of which compound.

**Revisit this document: 2026-06-09, post-keynote.**
