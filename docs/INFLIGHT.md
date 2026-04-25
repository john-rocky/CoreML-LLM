# In-flight work

Per `docs/ROADMAP_2026_04_26.md` cross-session protocol. Each parallel
Claude session adds a line when it claims a stage; removes it when the
branch merges to main. Helps avoid two sessions stepping on the same
stage / hot file.

Format: one line per active claim.
`branch | session-id | started UTC | task`

---

(empty — no active claims)
