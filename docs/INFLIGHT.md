# In-flight work

Per `docs/ROADMAP_2026_04_26.md` cross-session protocol. Each parallel
Claude session adds a line when it claims a stage; removes it when the
branch merges to main. Helps avoid two sessions stepping on the same
stage / hot file.

Format: one line per active claim.
`branch | session-id | started UTC | task`

---

(empty — no active claims)
stage4-lmsplit-finalise | 3D633D83 | 2026-04-25T19:38:40Z | iPhone A/B/C measurement
