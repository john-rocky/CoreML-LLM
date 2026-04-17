# 11C Parity Plan

**Status:** Spec — 2026-04-17. Consumed by the implementation session.

---

## Stage 0: Reference generation (prerequisite)

Before any verify-chunk modification, generate a clean reference chain using pure serial T=1 decode with no speculative state (oracle baseline).

```bash
cd /path/to/CoreML-LLM
source conversion/.venv/bin/activate
python conversion/test_serial_chain.py \
    --hf-dir output/gemma4-e2b/hf_model \
    --prompt "The quick brown fox" \
    --tokens 64 \
    --out eval/serial_chain_ref.json
```

This script should:
- Run T=1 HF forward at each position, collecting per-layer K/V and the token ID.
- Save token IDs (64 tokens), per-layer K/V for each position, and final logits.

If `test_serial_chain.py` does not exist, write it from scratch (≈60 lines, pure PyTorch, no CoreML dependency). Reference is used in all three parity gates below.

---

## Gate 1: fp32 per-layer cosine parity (Mac)

**Goal:** New verify chunks (no write-through) produce the same per-layer hidden states as a serial T=1 fp32 decode loop, up to cosine ≥ 0.9999 at every layer for every T position.

**Test harness:** `conversion/test_verify_parity_fp32.py` (new file, ≈100 lines)

```
for each prompt in eval/parity_prompts.txt (5 prompts, 50 tokens each):
    run serial fp32 decode for 50 positions, record hidden_states per layer
    run new verify chunks in fp32 (--no-quantize flag) at batch positions [P, P+1, P+2]
    for each T position t in [0, 1, 2]:
        for each layer l in [0..34]:
            cosine = dot(h_verify[l][t], h_serial[l][t]) / (|..| * |..|)
            assert cosine >= 0.9999
        assert argmax(logits_verify[t]) == argmax(logits_serial[t+P])
```

**Pass criterion:** All cosine values ≥ 0.9999, all argmax matches.

**Failure modes to watch:**
- Wrong causal mask for verify positions: attention attends to future tokens.
- Incorrect kv13/kv14 passed to chunk3/4: shared-layer logits diverge.
- Update-indicator removal breaks full-layer K/V shape.

---

## Gate 2: fp16 argmax parity (Mac)

**Goal:** After INT4 palettization, verify chunks with the new contract produce the same argmax as serial fp16 decode on real Gemma-4 token sequences.

**Test harness:** `conversion/test_verify_parity_fp16.py` (≈80 lines)

```
prompts = eval/parity_prompts.txt (same 5 prompts, extend to 100 tokens each)
for each prompt:
    run serial fp16 CoreML T=1 decode for 100 tokens via ChunkedEngine.predictStep
    at positions P=50, P+1, P+2: run new verify_qK function with K=3
    check:
        argmax[0] from verify == T=1 argmax at P     (must match)
        argmax[1] from verify == T=1 argmax at P+1   (must match; this is the key fix)
        argmax[2] from verify == T=1 argmax at P+2   (must match)
        cosine(verify_logits[t], serial_logits[P+t]) >= 0.995 for t in [0,1,2]
```

**Pass criterion:** argmax[0] always matches (guaranteed by causal mask structure). argmax[1] and argmax[2] must match at ≥ 95% of positions across all prompts. Cosine ≥ 0.995.

**Why argmax[1]/[2] would fail under old contract:** old contract wrote draft K/V at P+1/P+2 before computing argmax — so argmax[1] was conditioned on draft context. New contract reads only the verified prefix up to P — so argmax[1] at P+1 should now be identical to the clean T=1 argmax.

**Expected result after fix:** argmax match rate → 100% for argmax[1] on real prompts (not random logits), confirming contamination is eliminated.

---

## Gate 3: iPhone acceptance measurement

**Goal:** Measure live acceptance rate before and after the fix on device using the same 20 prompts as prior bench runs. Baseline is ~0%.

**Preparation (on Mac, by implementer):**
```bash
# Push new .mlmodelc bundles to device
xcrun devicectl device copy to --device <UDID> \
    output/verify-nodraft/chunk1.mlmodelc \
    ...
    /path/on/device/
```

**User runs (on iPhone 17 Pro):**
```
App: CoreMLLLMChat
Settings: speculative ON, K=3, same 20 prompts from eval/spec_bench_prompts.txt
Metric to capture: [SpecDbg] lines from Xcode console:
  burst #N: accepted=M, rate=X.XX, rolling=Y.YY
Expected baseline (old): rolling acceptance decays to <0.05 within 15 bursts
Expected after fix: rolling acceptance ≥ 0.40 steady-state (matching Mac CPU result)
```

**Acceptance target for go/no-go:**
- Mac CPU (PyTorch HF target, current draft): 42.9%
- iPhone target after fix: ≥ 30% rolling (77% break-even sets the speedup threshold)
- tok/s improvement threshold: ≥ 35 tok/s (vs baseline 28.6 tok/s)

**What to record:**
```
Prompt category, burst count, mean accepted/burst, final rolling rate, tok/s
```
Paste raw `[SpecDbg]` output into `eval/11c_iphone_acceptance.txt`.

---

## Stage Summary

| Gate | Where | Automated? | Pass threshold | Blocks |
|------|-------|-----------|----------------|--------|
| 0: serial reference | Mac | yes | — | Gates 1+2 |
| 1: fp32 per-layer cosine | Mac | yes | cosine ≥ 0.9999 all layers | Python merge |
| 2: fp16 argmax | Mac | yes | argmax[1]/[2] match ≥ 95% | Swift work |
| 3: iPhone acceptance | Device (user) | manual | rolling ≥ 0.30, tok/s ≥ 35 | Ship |

Gates 1 and 2 must pass before deploying to iPhone. Gate 3 is user-executed.
