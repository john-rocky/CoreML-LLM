# In-flight work

Per `docs/ROADMAP_2026_04_26.md` cross-session protocol. Each parallel
Claude session adds a line when it claims a stage; removes it when the
branch merges to main. Helps avoid two sessions stepping on the same
stage / hot file.

Format: one line per active claim.
`branch | session-id | started UTC | task`

## Special claims

- **`iphone-push`** — the iPhone is a single USB device. Before
  `xcrun devicectl device copy to ...`, append a claim with
  `branch=iphone-push`, do the push, remove the claim. Other sessions
  that need the iPhone busy-wait on the file (e.g. `until ! grep -q
  "^iphone-push" docs/INFLIGHT.md; do sleep 30; done`) before claiming
  it themselves. Push usually 2-3 minutes per ~3.7 GB bundle.
- **`mac-build-heavy`** — Stage 3 (multifunction prefill_bN) peaks
  ~30-40 GB RAM and saturates more cores than the other stages.
  Optional claim if RAM is tight; remove on build done. Other stages
  can ignore unless RAM-constrained.

---

(empty — no active claims)
stage3-prefill-bn | 8A5C2B35 | 2026-04-25T19:27:16Z | multifunction prefill_bN — ready for merge (Mac + iPhone tested; 3-chunk merged dual-state Linear is the ship variant)
stage6-multimodal-stateful | 521DE3E7 | 2026-04-26T21:34:48Z | multimodal stateful (vision/audio/video)
