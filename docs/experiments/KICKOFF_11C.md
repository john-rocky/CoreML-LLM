# Kickoff: Item 11C — Verify-Protocol Redesign (Write-After-Accept)

**Paste this into a new session to resume work.**

---

## Context (60 seconds)

Speculative decoding on iPhone accepts ~0% of draft proposals while Mac CPU accepts 42.9% on the same draft. Root cause is confirmed (PR #72, `docs/PHASE_B_DECISION.md`): the current verify chunks write draft K/V into the persistent cache *during* the verify graph execution — before Swift decides which tokens to accept. Subsequent target argmaxes at positions P+1..P+K-1 attend over contaminated KV that includes rejected draft tokens. This is item 11c in `PRIORITY_ROADMAP.md` and Blocker 2 in `docs/EAGLE3_INTEGRATION_STATE.md`. The fix is the same for both: verify chunks return per-T-position K/V *slices* as outputs, and Swift writes only the accepted prefix into the persistent IOSurface cache.

---

## State of Play (Phase 0 audit — 2026-04-17)

- `docs/11C_CURRENT_CONTRACT.md` — full audit of the current I/O contract, the exact call-path where KV writes happen (lines 776–800 in `ChunkedEngine.swift`), and why `commitAccepted` cannot undo them.
- `docs/11C_PROPOSED_CONTRACT.md` — complete spec for the new contract: new output tensors, shapes, Swift write-back algorithm including sliding-window shift, full-attention scatter write, kv13/kv14 shared-layer handling, and bonus-token (K+1) decode.
- `docs/11C_PARITY_PLAN.md` — three validation gates (fp32 cosine, fp16 argmax, iPhone acceptance).

No code has been modified. All four deliverable docs live in `docs/`.

---

## Branch and Worktree

```bash
# Open this session in the dedicated worktree:
cd /path/to/CoreML-LLM
git worktree add ../coreml-llm-11c feat/verify-protocol-redesign 2>/dev/null \
    || git -C ../coreml-llm-11c checkout feat/verify-protocol-redesign
cd ../coreml-llm-11c
source conversion/.venv/bin/activate
```

Do NOT commit to `main` until Gate 2 passes on Mac.

---

## Multi-Week Work Breakdown

### Week 1 (days 1–3): Python — new verify chunk models

**Files to touch:**
- `conversion/models/gemma4_swa_chunks.py` — `SWAVerifyChunk1`, `SWAVerifyChunk2` (lines ~200+). Remove `update_indicator` blend. Return per-T-position K/V slices as new tensors instead of writing back into the cache argument.
- `conversion/build_verify_chunks.py` — update `vout1` / `vout2` output name lists to include `new_K_sliding_c1`, etc. Remove `update_indicator` from `vin1` / `vin2` input specs.

**Contract:**
- `SWAVerifyChunk1` forward: returns `(hidden_states_out, per_layer_combined_out, new_K_sliding_c1 (7,1,K,256), new_V_sliding_c1, new_K_full_c1 (1,1,K,512), new_V_full_c1)`
- `SWAVerifyChunk2` forward: same pattern + `kv13_k_slices`, `kv13_v_slices`, `kv14_k_slices`, `kv14_v_slices`
- Chunks 3 and 4: no change. They already receive kv13/kv14 as explicit inputs and are read-only.

**Go/no-go gate (end of days 1–3):** Gate 1 passes — `test_verify_parity_fp32.py` shows cosine ≥ 0.9999 at all 35 layers for all 5 prompts.

### Week 1 (days 4–5): Mac fp16 parity

Rebuild chunks with INT4 palettization (`build_verify_chunks.py --K 3`). Run Gate 2 (`test_verify_parity_fp16.py`). Expected: argmax[1] and argmax[2] now match serial T=1 fp16 decode.

This is the empirical proof that KV contamination is eliminated. If argmax[1] match rate is still < 95%, re-examine the kv13/kv14 path in chunk2 — verify that chunks 3/4 are receiving the pre-verify window, not new slices.

**Go/no-go gate (end of day 5):** Gate 2 passes.

### Week 2 (days 6–10): Swift rewrite

**Files to touch:**
- `Sources/CoreMLLLM/ChunkedEngine.swift`:
  - `verifyCandidates(tokens:startPosition:)`: remove `copyBack` calls after chunk1/chunk2 predictions (lines 776–800, 796–800). Capture `newKSlidingC1`, etc. from new outputs.
  - New internal method `commitKVSlices(N:newKSlidingC1:...)`: implements the shift-and-append for sliding layers and scatter-write for full layers, as specified in `docs/11C_PROPOSED_CONTRACT.md`.
  - `commitAccepted(_ tokens:)`: call `commitKVSlices` with the accepted count N, then advance `currentPosition`.
  - Add `shiftAndAppendSliding(_ buf: MLMultiArray, slot: Int, newSlices: UnsafePointer<UInt16>, count: Int, hd: Int)` helper using `memmove` + `memcpy` on the IOSurface-backed buffer.

**Key correctness invariant:** after `commitKVSlices(N:...)`, the state of `kSliding1/2`, `kFull1/2` must be identical to what N sequential T=1 `predictStep` calls would have produced. Write a Mac test to verify this (`test_kv_commit_parity.swift` or a Python-side test that dumps the KV state).

**Bonus token:** if N == K+1 (all drafts accepted), run one T=1 `predictStep` for the bonus token at `currentPosition` after writing N slices. This covers the K+1 case without additional verify chunks.

**Go/no-go gate (end of day 10):** Gate 2 still passes after Swift changes (run the same fp16 harness via `ChunkedEngine` directly), and one end-to-end speculative generation on Mac produces coherent text.

### Week 3 (day 11): iPhone bench (user-executed)

- Compile and push `.mlmodelc` to iPhone 17 Pro.
- User runs 20 prompts from `eval/spec_bench_prompts.txt` in `CoreMLLLMChat`.
- Capture `[SpecDbg]` console output, paste into `eval/11c_iphone_acceptance.txt`.
- Target: rolling acceptance ≥ 0.30, tok/s ≥ 35.0.

---

## Track Coordination

**Track C (S0 — monolithic model):** builds a separate model and does NOT touch `conversion/models/gemma4_swa_chunks.py` or `build_verify_chunks.py`. These are safe to modify on `feat/verify-protocol-redesign` without conflicts.

**Track A (EAGLE-3 retrain):** if the retrain completes before Gate 2, re-run Gate 2 with the new draft (`eagle3_draft_best.pt` v2) — same Python harness, zero additional Python work. The retrained draft only changes acceptance rate measurements, not the verify chunk contract.

**MTP drafter:** same dependency as EAGLE-3. MTP acceptance is gated behind 11c closing (77% break-even per `docs/MTP_INVESTIGATION_SUMMARY.md`). Do not revisit MTP until Gate 3 measures ≥ 30% acceptance rate.

---

## iPhone Bench Commands

```bash
# 1. Build mlmodelc (Mac)
python conversion/build_verify_chunks.py \
    --hf-dir output/gemma4-e2b/hf_model \
    --output output/verify-nodraft \
    --K 3

xcrun coremlcompiler compile output/verify-nodraft/chunk1.mlpackage output/verify-nodraft/
# repeat for chunk2, chunk3, chunk4

# 2. Push to device
UDID=$(xcrun devicectl list devices | grep "iPhone 17" | awk '{print $1}')
for f in chunk1 chunk2 chunk3 chunk4; do
  xcrun devicectl device copy to --device "$UDID" \
    output/verify-nodraft/${f}.mlmodelc \
    /var/mobile/Containers/Data/Application/<APP_UUID>/Documents/
done

# 3. Run bench (user: launch CoreMLLLMChat, enable speculative, run 20 prompts)
# Capture Xcode console: filter "[SpecDbg]"
# Expected output line format:
#   [SpecDbg] burst #N: accepted=M/3, rate=0.XX, rolling=0.YY, dt=ZZ.Zms
```

---

## Open Questions for This Session

1. Does removing `update_indicator` from chunk1/2 verify inputs require a mask shape change in `makeVerifyUpdateIndicator` in ChunkedEngine? (Probably yes — that method should be deleted or made no-op.)
2. For sliding-layer write-back: `shiftAndAppend` must match the `torch.cat([K[:, :, 1:, :], k_padded], dim=2)` semantics exactly. Validate with a unit test comparing the buffer contents after N=1 vs N=3 commits.
3. kv13 in chunk2 is sliding (W-sized). The verify graph still needs to compute attention at positions P..P+K-1 against a K-position-expanding window of kv13. Confirm the causal sliding mask `(1,1,K,W)` encodes this correctly for each t in [0..K-1].
