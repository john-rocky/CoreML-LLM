# KICKOFF: Track C — S0 GPU Verify + ANE Decode

**Self-contained prompt for a new session. Paste entire file into a fresh chat.**

---

## 1. Context (why S0)

We are trying to beat LiteRT-LM's 56.5 tok/s on iPhone 17 Pro with Gemma 4 E2B.
Baseline ANE-only decode is ~31 tok/s. Three parallel tracks are running:
Track C (S0) is the highest-ceiling option: **62–88 tok/s at oracle acceptance rates**.

The key insight: every prior speculative-decode attempt put verify on ANE.
ANE has a 4-chunk serial dispatch floor (~104 ms for K=3), making break-even 2.0×
— most drafters fail to clear it. **Moving verify to GPU breaks this:**

- GPU verify (monolithic, 1 dispatch): ~60 ms estimated on iPhone A19
- ANE decode: ~52 ms/token
- Break-even: 60/52 = **1.15** — every drafter we have clears this

KV handoff is zero-copy via Apple Silicon unified memory: ANE writes IOSurface-backed
MLMultiArray, GPU reads the same physical pages. No DMA.

---

## 2. State of play after Phase 0

**Phase 0 RESULT: GO.** Conversion succeeded.

Key facts from Phase 0 (2026-04-17):
- `conversion/build_s0_monolithic.py` was written and run
- 35-layer monolithic `.cpuAndGPU` model compiled in **77.7 seconds**
- 16,243 MIL ops, all 3 MIL pipelines completed (frontend, default 95 passes, backend)
- fp16 package size: **9.4 GB** (needs INT4 for iPhone and for Mac predict)
- Parity test blocked on Mac by size (>1 GB CPU predict limit) — NOT an iPhone issue
- `conversion/output/s0_probe/s0_probe_result.json`: `"go_no_go": "GO"`

**Phase 1 entry condition: already met.** Start with INT4 palettization.

Result JSON:
```bash
cat conversion/output/s0_probe/s0_probe_result.json
```

**Go/no-go gate for Phase 1:**
- Result already `"status": "success"` → proceed to Phase 1 below
- If re-running and seeing `"stage": "coremltools_convert"` failure → see §Fallback
- OOM during trace → model too large for Mac RAM; skip to iPhone direct

**Key files:**
- `conversion/build_s0_monolithic.py` — the monolithic GPU model + conversion
- `conversion/models/gemma4_swa_chunks.py:457` — `_run_layer_verify` reference implementation
- `docs/SURVIVING_HYPOTHESES.md:239` — full S0 spec with math
- `conversion/models/gemma4_wrapper.py:107` — NCHW bug (do NOT fix here; fixed locally in build_s0)

---

## 3. Work breakdown (7–10 days)

### Phase 1: INT4 palettization + parity (Day 1)
The fp16 model compiled (9.4 GB). Now palettize to INT4 for Mac predict + iPhone deploy:

```bash
cd conversion
GEMMA4_HF_DIR=./output/gemma4-e2b-final/hf_model \
python build_s0_monolithic.py \
  --quantize \
  --output ./output/s0_probe
# Output: output/s0_probe/gemma4_monolithic_int4.mlpackage (~600 MB)
```

Then run parity validation (Mac predict works at ~600 MB):
```bash
GEMMA4_HF_DIR=./output/gemma4-e2b-final/hf_model \
python build_s0_monolithic.py \
  --quantize --bench \
  --output ./output/s0_probe
```

Gate: `parity_top1_match: true` AND `parity_max_logit_diff < 5e-3`
If parity fails after INT4: check top-1 only (fp16 drift expected ~2e-3 logit diff).

### Phase 2: Swift `GpuVerifier.swift` (Day 3–4)
Create `Sources/CoreMLLLM/GpuVerifier.swift`:
- Load `gemma4_monolithic.mlpackage` with `.cpuAndGPU`
- Accept KV arrays from `ChunkedDecodeEngine` (IOSurface-backed MLMultiArray)
- Run `predict` with T=3 (K=3 draft tokens)
- Return accepted token count + accepted KV positions

**DO NOT touch:**
- `Sources/CoreMLLLM/ChunkedEngine.swift` (Track A/B territory — no core verify path changes)
- `conversion/models/gemma4_verify_chunks.py` (Track B territory)
- Only add new files; extend via protocol conformance or parallel switch

IOSurface KV contract:
```swift
// Decode produces: MLMultiArray backed by IOSurface
// GPU verify reads same buffer — no copy needed
let kvBuffer: MLMultiArray  // existing from ChunkedDecodeEngine
gpuVerifier.predict(kvBuffer: kvBuffer, draftTokens: draftIds)
```

### Phase 3: KV layout alignment (Day 4–5)
The monolithic model uses flat K_sliding `(12, 1, W, 512)` and K_full `(3, 1, ctx, 512)`.
The existing 4-chunk decode uses per-chunk KV arrays.

Options:
1. **Re-pack**: Swift code assembles flat tensor from chunks before calling GPU verify
2. **Re-export**: Modify `build_s0_monolithic.py` to use identical KV layout as chunks

Check existing chunk KV layout:
```
conversion/models/gemma4_swa_chunks.py:184  # _layer_kv_map
Sources/CoreMLLLM/ChunkedEngine.swift       # see how KV arrays are structured
```

Gate: verify output token matches greedy decode output for same KV state.

### Phase 4: iPhone deploy + bench (Day 6–7)
```bash
# Copy INT4 mlpackage to device via Xcode scheme or:
cp -r conversion/output/s0_probe/gemma4_monolithic_int4.mlpackage \
      Examples/CoreMLLLMChat/CoreMLLLMChat/

# In Xcode: add to target, load alongside existing chunks
```

**User runs benchmark on device.** Expected output format:
```
[GpuVerifier] T=1 latency: XX ms  (decode-equivalent single token)
[GpuVerifier] T=3 latency: XX ms  (verify 3 draft tokens)
[S0 spec] break-even = T3_latency / decode_latency
```

Go/no-go: if T=3 latency < 1.15 × decode_latency → S0 math works, full integration.
If T=3 latency > 1.8 × decode_latency → S0 fails on A19; kill and pivot.

### Phase 5: End-to-end speculative decoding (Day 8–10)
Wire up full ANE decode → MTP drafter → GPU verify → accept/reject loop:
- Drafter: `output/mtp_drafter.mlpackage` (already converted, see `docs/MTP_DRAFTER_CONVERSION.md`)
- Acceptance: greedy chain-compare (same as LiteRT K=3 fixed)
- KV advance: after accept, increment position by `n_accepted + 1`

---

## 4. Fallback if monolithic fails

**2-chunk GPU fallback:**
- Chunk A: L0–17 (produces kv13, kv14)
- Chunk B: L18–34 (reads kv13, kv14)
- Each chunk < 18 layers → below the ANE 15-layer threshold
- Expected latency: ~90 ms (two dispatches) → break-even ~1.73 → still better than ANE verify (2.0×)

Script outline: copy `build_s0_monolithic.py`, add `--two-chunk` flag, split `forward()` at layer 18.

---

## 5. Coordination

| Territory | Owner | Rule |
|---|---|---|
| `conversion/models/gemma4_verify_chunks.py` | Track B | DO NOT modify |
| `Sources/CoreMLLLM/ChunkedEngine.swift` core verify path | Track A | DO NOT modify |
| `Sources/CoreMLLLM/GpuVerifier.swift` | Track C (you) | add new file |
| `conversion/build_s0_monolithic.py` | Track C (you) | extend freely |
| `conversion/output/s0_probe/` | Track C (you) | gitignored, safe |

---

## 6. Quick-start commands

```bash
# Activate venv
source /Users/majimadaisuke/Downloads/CoreML-LLM/conversion/.venv/bin/activate
cd /Users/majimadaisuke/Downloads/CoreML-LLM/conversion

# Phase 0 already done (fp16 model exists at output/s0_probe/gemma4_monolithic.mlpackage)
# Phase 1: INT4 palettize + parity + Mac bench
GEMMA4_HF_DIR=./output/gemma4-e2b-final/hf_model \
python build_s0_monolithic.py \
  --quantize --bench \
  --output ./output/s0_probe
# Output: output/s0_probe/gemma4_monolithic_int4.mlpackage (~600 MB)

# Check result
cat output/s0_probe/s0_probe_result.json

# fp16 model already at: output/s0_probe/gemma4_monolithic.mlpackage (9.4 GB, gitignored)
```

---

## 7. Memory budget check

iPhone 17 Pro has 8 GB RAM. Estimated footprint:
- ANE decode chunks (4×): ~450 MB each INT4 = ~1.8 GB
- GPU verify (monolithic fp16): ~2.8 GB unquantized → target INT4 = ~700 MB
- Total with system: ~5–6 GB → tight but within jetsam limit (~6.5 GB for foreground)
- If OOM: INT4 palettize the GPU model in `build_s0_monolithic.py` (add `quantize=True`)

---

*Phase 0 complete. Check `s0_probe_result.json` and proceed from Phase 1.*
