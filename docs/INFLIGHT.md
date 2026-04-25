# In-flight work

Per `docs/ROADMAP_2026_04_26.md` cross-session protocol. Each parallel
Claude session adds a line when it claims a stage; removes it when the
branch merges to main. Helps avoid two sessions stepping on the same
stage / hot file.

Format: one line per active claim.
`branch | session-id | started UTC | task`

---

stage2-e4b | B7B118BA | 2026-04-25T19:37:20Z | E4B stateful port + Phase 2a
