# iPhone Gemma 4 E2B Free-form Speedup — Mobile ANE Constraint Inversion Attempt

Date: 2026-05-13. Goal: lossless 1.5× iPhone free-form. Training EXCLUDED.

## Lever inventory (Mac empirical → iPhone projection)

| Lever | Approach | Mac speedup | iPhone proj | Status |
|---|---|---|---|---|
| FLy top-K=16 + never-bail | loose accept rule | 1.49× | 1.16× | **shipped current** |
| INT2 drafter | 8× weight compression | 0.86× | <1.0× | dead (accept 6%) |
| INT3 drafter | 5× compression | 0.94× | <1.0× | dead (accept 10%) |
| INT4 (baseline) | 4× compression | 1.49× | 1.16× | optimal |
| INT6 drafter | 2× compression | 1.30× | <1.0× | worse than INT4 |
| K=2 chunks | shorter chain | 1.0× | 0.91× | dead |
| K=4 chunks | longer chain | 1.13× | 0.85× | iPhone net loss |
| Compute unit sweep | CPU/GPU/ANE | 1.03-1.06× | similar | marginal |
| Mask offset sweep | mask anchor | 1.05× | similar | dead |
| Tree verify | parallel branches | regression | regression | dead |
| PLD prefetch | n-gram skip drafter | 1.27× | similar | regression (low hit rate) |
| Lookahead-only | PLD+Jacobi (no MTP) | 1.07× | similar | marginal |
| Cross-vocab Qwen | larger drafter | 0.54× | <1.0× | dead (48ms drafter cost) |
| Async drafter L5 | parallel via speculation | 1.27× | 1.27× | data dep blocks > marginal |
| K-step unrolled | single dispatch | NA | NA | blocked by embed lookup |

## Structural ceiling math

iPhone empirical baseline:
- Plain decode cycle: 30ms (32 tok/s)
- MTP cycle with current drafter: 47ms (11ms drafter + 36ms verify + 0.4ms commit)
- Best emit_avg observed on EN free-form: 1.78 (FLy top-K=16)

For iPhone 1.5× lossless requires:
- emit / cycle = 1.5 × 1/0.030 = 50 tok/s
- → cycle ≤ 36ms (verify floor) AND emit ≥ 1.78  
- → drafter cost must hide entirely behind verify (parallel)

Async drafter math (best case full speculation success):
- cycle = max(drafter, verify) = max(11, 36) = 36ms ✓
- emit 1.78 / 0.036 = 49.4 = **1.54× ✓**

But: full speculation requires matching guess of cycle N+1's nextID
- nextID N+1 = targetArgmax[matchCount(N)] (post-verify N)
- Speculative guess: drafter's last token output  
- Match rate (full accept = matchCount=K-1): ~25% on EN free-form

Realistic async with 25% hit:
- avg cycle = 0.25 × 36 + 0.75 × 47 = 44.3ms
- emit 1.78 / 0.044 = 40.5 tok/s = **1.27×**

NOT 1.5×. 1.5× would require ~80% speculation hit rate which requires drafter
quality on par with target — impossible without training.

## Verdict

Without training, iPhone lossless ceiling on free-form English chat is:
- **1.16×** with current ship (FLy top-K=16 + INT4 + never-bail)
- **1.27×** with async drafter (multi-day implementation, marginal gain)
- **1.5×** structurally impossible
