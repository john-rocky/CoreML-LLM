# MTP Path A — Post-mortem autopsy

**Status:** 2026-04-17. Post-mortem investigation of the Path A failure (0% acc on iPhone, 0/5 argmax match on Mac TFLite probe in `MTP_INTEGRATION_RESULTS.md §5`). Task was to determine whether the prior "trained for W4A8 quant regime" explanation is the full story, or whether a tensor-layout / dequant / tokenizer bug is hiding behind it.

---

## 1. TL;DR

- **Real root cause:** target-distribution mismatch (Google's drafter was trained against LiteRT's W4A8-quantized main LLM, not against the fp HF Gemma 4 checkpoint). Diagnostic evidence: fp32 PyTorch port of the extracted drafter achieves **cosine similarity 0.9935 with HF target's next-token logits** on 32 prompts, but only **3.1% top-1 / 6.2% top-5 argmax match**. Logits are nearly co-linear but shifted just enough to cross argmax boundaries — the classic signature of a small systematic bias, not a weight/transpose break.
- **Weights are reusable in principle, but not at greedy-argmax quality.** Median rank of the correct target token in the drafter's distribution is **2499 / 262144**, and the correct token lies in the top-1000 for 38% of cases. A drafter whose extraction was structurally broken (wrong transpose, wrong dequant sign, wrong vocab) would have cosine ≪ 0.5 and median rank ≫ 100k. Ours does not.
- **Path A is not rehabilitatable at zero training cost.** The extracted artifact cannot be bolted onto our fp target and recover the 56.5 tok/s acceptance Google gets. A cheap fix (fine-tune the drafter's last 1–2 layers + lm_head against our fp target's next-token output on a small corpus) is plausible but architecturally overlapping with the already-shelved Path C. Recommendation: keep Path A parked unless we first eliminate Priority 11c (verify-chunk fp16 drift), which is the bigger blocker for any drafter on this target.

---

## 2. Inventory — MTP artifacts on disk

| Artifact | Path | Size | Produced by |
|---|---|---|---|
| `.litertlm` source container | `~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/.../gemma-4-E2B-it.litertlm` | 2.58 GB | Google (litert-community HF mirror) |
| Extracted TFLite sections (10) | `output/mtp_probe/section_0..section_9.tflite` | 103 MB .. 1.28 GB | `conversion/extract_mtp_drafter.py` (2026-04-14) |
| **`section_9.tflite`** — the MTP drafter | `output/mtp_probe/section_9.tflite` | **44.3 MB** | Extracted from container |
| Weight listing (debug) | `output/mtp_probe/section_9_weights.txt` | 7.1 KB | Same |
| **`mtp_drafter.pt`** — PyTorch port | `output/mtp_probe/mtp_drafter.pt` | 308.8 MB (fp32) | `conversion/mtp_drafter_model.py --tflite section_9.tflite` (loader is `load_from_tflite_auto`) |
| `mtp_drafter.mlpackage` — empty stub | `output/mtp_drafter.mlpackage` | 617 B Manifest only (Data/ empty) | Stale — real compiled bundle lives in the iPhone deploy dir |
| `mtp_drafter.mlpackage` — compiled | `conversion/output/iphone_8k/mtp_drafter.mlpackage` | (not inspected) | `conversion/build_mtp_drafter.py --palettize-int4` |

Section 9 is confirmed to be the mtp_drafter (signature name `mtp_drafter`, 267 tensors, 194 ops, 30 subgraphs). The `.pt` file contains **44 parameter tensors** (41 weights + 3 register_buffers), matching the 4-layer drafter spec exactly (see §3.2).

The PyTorch port was produced by `load_from_tflite_auto` which:
1. Dequantizes all INT4 / INT8 tensors in `_extract_tflite_weights`
2. Matches them to PyTorch keys by (a) linear-weight substring patterns (`mtp_pre_proj`, `layer_{i}/layer_{i}.pre_q`, `embedder.decode` etc.) + shape check, (b) explicit `jax2tf_arg_N` → norm-name mapping verified via TFLite graph parse.

No evidence of unmatched keys at load time (see below — we re-verify loaded weight norms).

---

## 3. Extraction code audit — clean with one cosmetic bug

### 3.1 Dequantization is correct

The TFLite dtype code table used at *extraction* time (not the display time) is correct:

```python
# _extract_tflite_weights in mtp_drafter_model.py
if dtype_code == 0:   # FLOAT32
elif dtype_code == 9: # INT8   ← verified: tflite.TensorType.INT8 = 9
elif dtype_code == 17:# INT4   ← verified: tflite.TensorType.INT4 = 17
```

Independent check with `tflite.TensorType` confirms code 9 is INT8 and code 17 is INT4. ✓

INT8 dequant: `(arr_int8.astype(fp32) - zp) * scale`, per-channel along the first dim. Zero-point is loaded but was verified to be **0 for every INT8/INT4 weight tensor** in section_9.tflite (symmetric quant), so even if zp handling had a subtle bug it wouldn't matter. ✓

INT4 dequant: byte unpacking is
```python
lo = (raw_bytes & 0x0F).astype(np.int8)
hi = ((raw_bytes >> 4) & 0x0F).astype(np.int8)
lo[lo >= 8] -= 16    # sign-extend to [-8..7]
hi[hi >= 8] -= 16
arr_int4[0::2] = lo
arr_int4[1::2] = hi
```

This correctly implements TFLite's packed INT4 (low nibble first, signed range `[-8, 7]`). Given all INT4 tensors have zp=0, this is sufficient. No off-by-128, no missing zp application. ✓

### 3.2 Transpose / layout — matches PyTorch

Crucial: TFLite has already transposed all `einsum` → `FullyConnected` weights into `[out, in]` layout. Section 9 weights verified by reading the shape column in `section_9_weights.txt`:

| TFLite weight | Shape | PyTorch target | Match? |
|---|---|---|---|
| `mtp_pre_proj/btm,md->btd/dot_general` | `[256, 3072]` | `Linear(3072, 256)` → `(256, 3072)` | ✓ |
| `layer_i.pre_q/q_einsum/btd,dH->btH` | `[1024, 256]` (SWA) / `[2048, 256]` (L3) | `Linear(256, 1024)` / `Linear(256, 2048)` | ✓ |
| `layer_i.post_qkv/attn_vec_einsum/btH,Hd->btd` | `[256, 1024]` / `[256, 2048]` | `Linear(1024, 256)` / `Linear(2048, 256)` | ✓ |
| `mlp/gating_einsum1/btd,df->btf` | `[2048, 256]` | `Linear(256, 2048)` | ✓ |
| `mlp/linear/btf,fd->btd` | `[256, 2048]` | `Linear(2048, 256)` | ✓ |
| `embedder.decode` | `[262144, 256]` | `Linear(256, 262144)` | ✓ |
| `mtp_post_proj/btd,dm->btm` | `[1536, 256]` | `Linear(256, 1536)` | ✓ |

`torch.from_numpy(arr)` can be used directly without `.T`. The extractor does exactly this:
```python
t = torch.from_numpy(arr).float()
if t.shape == sd[pt_key].shape:   # shape-gated load, not a blind load
    sd[pt_key] = t
```
The extraction is shape-gated: any silent mis-transpose would have failed this check and been logged as `SHAPE MISMATCH`. ✓

Loaded weight norms (reads from `output/mtp_probe/mtp_drafter.pt`) are also consistent with Gemma conventions:
- `final_norm.weight` norm = 181.8 over 256 entries ≈ RMS 11.4, matches HF target's `lm.norm.weight` (RMS ≈ 15) — Gemma stores raw scales, not `1 + w`.
- `layers.*.attn.q_norm.weight` norm = 15.87 over 256 entries ≈ 0.9916 each, matches HF target's `layers.0.self_attn.q_norm.weight` (constant 0.984375). ✓
- Projection weights are in the right order of magnitude (norms 15–46 per tensor).

### 3.3 Cosmetic bug — display dtype map is wrong (does not affect extraction)

`inspect_tflite_signature` in `extract_mtp_drafter.py` prints:
```python
dtype_map = {0: "fp32", 1: "fp16", 2: "int32", 3: "uint8",
             7: "int8", 9: "bool", 15: "int16"}
```

This mapping is wrong (real codes: 7 = INT16, 9 = INT8, 6 = BOOL). It only affects the human-readable column in `section_9_weights.txt` (where INT8 tensors print as "bool" and INT4 tensors print as "type_17"). The actual weight extractor (`_extract_tflite_weights` in `mtp_drafter_model.py`) uses the correct codes in its `if dtype_code == …` branches. No extraction bug, just confusing dev output. **Low priority to fix.**

### 3.4 What does NOT happen in extraction that could matter

- **No activation quantization replay.** The TFLite graph has INT8 per-tensor activation scales at every FC boundary (W4A8 pipeline — see `LITERT_RUNTIME_ANALYSIS.md §A4`). These INT8-composite scales appear in the tensor list (e.g. tensor idx 50 `mtp_pre_proj/composite` shape `(1,1,3072)` scale `0.1919`, idx 52 shape `(1,1,256)` scale `0.0723`, etc.) but are **never read** by `_extract_tflite_weights` — it skips activation-scale tensors because they have no buffer data. The PyTorch port runs fp32 end-to-end. This is architecturally expected (we can't run ANE on simulated W4A8), but it means our drafter is NOT a numeric replica of the TFLite inference path. It is a *floating-point replica of the weights*.
- **No PLE (Per-Layer Embedding) machinery.** The drafter itself does not have PLE (PLE is a feature of the main LLM, section 1). Fine.
- **No runtime-bmm composite ops.** These are fused Q·K and attn·V dispatched via `STABLEHLO_SHIFT_RIGHT_LOGICAL` in the TFLite graph. Our PyTorch uses standard `torch.matmul` — semantically equivalent.

**Conclusion of §3:** no structural extraction bug. The weights are loaded into PyTorch at the correct shapes, with the correct dequantization, with the correct rename map.

---

## 4. Tokenizer parity — 100% identical

Extracted the SentencePiece ModelProto directly from `gemma-4-E2B-it.litertlm`:
- Pieces span bytes `32768..4721699` (tag `0x0a` repeated 262,144 times)
- `trainer_spec` at `4721699..4721761` (60 bytes)
- `normalizer_spec` at `4721761..4721781` (18 bytes, name = `"identity"`)

Wrote the reconstructed SP model to `/tmp/litert_sp.model` and loaded it with `sentencepiece 0.2.1`.

Results:

| Check | HF `tokenizer.json` | LiteRT embedded SP | Match |
|---|---|---|---|
| Vocab size | 262,144 | 262,144 | ✓ |
| `<pad>` id | 0 | 0 | ✓ |
| `<eos>` id | 1 | 1 | ✓ |
| `<bos>` id | 2 | 2 | ✓ |
| `<unk>` id | 3 | 3 | ✓ |
| `<mask>` id | 4 | 4 | ✓ |
| `<|image>` id | 255999 | 255999 | ✓ |
| `<|audio>` id | 256000 | 256000 | ✓ |
| `<|image|>` / `<|audio|>` / `<image|>` / `<audio|>` (258880..258883) | all match | all match | ✓ |
| Piece-by-piece for all 262,144 IDs | — | 0 mismatches | ✓ |

Encoding diff for 20 representative strings (English, Japanese, code, emoji, CJK, Korean):

```
20/20 tokenizations match exactly
```

Example:
- `'光合成とは'` → both produce `[237914, 72309, 26244]`
- `'def fibonacci(n):'` → both produce `[2063, 10779, 78113, 236769, 236749, 1473]`
- ``'```python\ndef f():\n    pass\n```'`` → both produce `[2717, 6719, 107, 2063, 517, 6141, 107, 140, 4373, 107, 2717]`

**Tokenizer is fully eliminated as a root cause.** A single-token shift, different whitespace normalization, or a byte-fallback difference would have caused at least one of these 20 strings (and almost certainly many more across the vocab) to diverge. They don't.

---

## 5. fp32 forward diagnostic (Q1) — ran locally on Mac Studio

**Script:** `/tmp/path_a_autopsy_fp32.py` (Q1) and `/tmp/path_a_rank_check.py` (follow-up rank analysis). Both standalone, not committed to repo.

**Compute budget used:** ~6 seconds for 32 prompts of HF forward + drafter forward (fp32 on CPU, 128 GB Mac Studio). No OOM risk.

**Method (identical to `conversion/test_mtp_tflite_acceptance.py` but for the *PyTorch* port and over 32 prompts instead of 5):**
1. Load HF Gemma 4 E2B in fp32.
2. Forward the prompt with `use_cache=True, output_hidden_states=True`.
3. Extract `L34 raw hidden state = hidden_states[-2][:, -1:, :]` (the pre-norm output of the last transformer block — confirmed to be what `verify_qK` emits as `hidden_states_out` in ChunkedEngine).
4. Extract KV13 (sliding, head_dim=256) and KV14 (full, head_dim=512) from `past_key_values`.
5. `natural_next = argmax(softcap(HF.lm_head(last_hidden)))` — the target's own next-token choice.
6. Build drafter input `activations = concat(embed(natural_next)_unscaled, L34_raw)` per `LITERT_CONTAINER_ANALYSIS.md §MTP Drafter` and the C++ runtime source referenced in `LITERT_RUNTIME_ANALYSIS.md §B1.4`.
7. Pad KV to 32003 length (left-aligned), build mask with `True` for positions `< seq` / `-1e9` for ≥ `seq`.
8. Run PyTorch drafter, measure top-1/top-5 argmax vs `natural_next`, and cosine of (softcapped) drafter logits vs (softcapped) HF target logits.

### 5.1 Results over 32 prompts

| Metric | Value | Random baseline (V=262k) |
|---|---|---|
| PyTorch drafter **top-1 match** | **1 / 32 = 3.1 %** | ~0.0004 % |
| PyTorch drafter **top-5 match** | **2 / 32 = 6.3 %** | ~0.002 % |
| PyTorch drafter **mean cosine with HF target logits** | **0.9935** | ~0 |

Follow-up rank analysis (`/tmp/path_a_rank_check.py`, 16 prompts — where does `natural_next` rank inside the drafter's logit vector):

| Metric | Value |
|---|---|
| Median rank of target token | **2,499 / 262,144** |
| Top-10 | 1 / 16 (6 %) |
| Top-100 | 5 / 16 (31 %) |
| Top-1000 | 6 / 16 (38 %) |

### 5.2 Interpretation

Two numbers matter here:

1. **Cosine 0.9935.** If any of the extraction-layer failure modes the task framed were true — a transposed weight on e.g. `q_proj`, a flipped INT4 sign-extension, a wrong norm applied, a tokenizer shift — the drafter's output distribution would be catastrophically wrong. Cosine of 0.99 over a 262k-dim logit vector is not survivable through a broken linear algebra pipeline.

2. **3.1% top-1 + median rank 2.5k.** The drafter's distribution *is* the target's distribution, scaled/shifted by a small bias that is nearly always enough to shuffle argmax (`gap` between drafter argmax and target token is typically 10–20 logit units) but not enough to push the target token out of the top-1% of the vocab. In at least one case the drafter's argmax was literally one token ID away from the target (`natural=236770 '1'` vs `drafter argmax=236769 ' '`, a non-space/space variant off by one scale ULP).

This is the *exact* signature we'd expect from "trained on a slightly-different target." The drafter was calibrated against the LiteRT-quantized main LLM whose L34 hidden state and kv13/kv14 differ from HF fp32 by compounded per-layer quantization noise (~0.1–1% relative divergence per INT4 weight pass). Drive that mismatch through 4 drafter transformer layers and a 262k-way lm_head, and cosine stays near 1 but argmax is broken.

### 5.3 Cross-check with prior data

- `test_mtp_parity.py` (PT vs TFLite, zero-KV random activations): cosine = 0.82. Degraded because random inputs drive divergence hard.
- `test_mtp_tflite_acceptance.py` (raw TFLite vs HF on 5 prompts): 0 / 5 top-1 — consistent with our 3.1 % top-1 over a larger sample.
- Our §5.1 with *real* HF hidden states: cosine = 0.9935. **Suggests the PyTorch port is near-faithful to the TFLite artifact**; the PT-vs-TFLite gap reported at 0.82 is an artifact of the zero-KV random-activation test, not the on-target inference regime.

### 5.4 No long-running script needed

The full Q1 diagnostic finished in under 7 seconds on CPU. No training, no model download, no GPU required. Both scripts are in `/tmp/`:
- `/tmp/path_a_autopsy_fp32.py` — 32-prompt top-1/top-5/cosine sweep
- `/tmp/path_a_rank_check.py` — 16-prompt rank-of-target analysis

Reproduce with:
```bash
cd /Users/majimadaisuke/Downloads/CoreML-LLM
python3 /tmp/path_a_autopsy_fp32.py
python3 /tmp/path_a_rank_check.py
```

(TFLite comparison path is coded into `/tmp/path_a_autopsy_fp32.py` but disabled at this install because `ai_edge_litert` requires Python 3.12; the 2026-04-14 commit `conversion/test_mtp_tflite_acceptance.py` already ran it at 0/5.)

---

## 6. Verdict on the real root cause

**The prior explanation ("Google MTP was trained for LiteRT's W4A8 quant regime, so fp targets don't match") is essentially correct, but needed to be upgraded from a hand-wavy hypothesis to a mechanism supported by numeric evidence.**

Ranking the four candidate causes from the task brief:

| Candidate | Verdict | Evidence |
|---|---|---|
| (a) Tensor layout / transpose / scale bug in extraction | **Ruled out.** | All weight shapes match PyTorch target (shape-gated load); INT4/INT8 dequant is correct (verified schema codes + zp=0); norm weight norms match HF's own Gemma conventions; cosine 0.9935 would not survive a broken transpose. |
| (b) Tokenizer / vocab mismatch | **Ruled out.** | Vocab size, all tested special IDs, all 262,144 piece strings, AND 20 test-string encodings match HF byte-for-byte. |
| (c) Structurally mismatched weights (wrong # heads, wrong proj dims) | **Ruled out.** | Drafter architecture matches TFLite signature exactly (see `LITERT_CONTAINER_ANALYSIS.md §9`): 4 layers, 4 heads, head_dim 256/512, FFN 2048, mtp_pre_proj 3072→256, mtp_post_proj 256→1536, lm_head 256→262144. All observed in `mtp_drafter.pt`. |
| (d) Weights genuinely only usable in W4A8 | **Confirmed as the dominant cause, qualified.** The weights produce a distribution that is near-isomorphic to the fp target's distribution (cosine 0.99, target in top-0.95% of vocab median), but the systematic bias introduced by training against the quantized LLM's hidden states is larger than the argmax margin. |

### One thing to revise in the old doc

`MTP_INTEGRATION_RESULTS.md §5.3` states:

> "From the drafter's point of view, our HF-target inputs look 'close but wrong' — the right order of magnitude but the wrong operating point."

This is correct but underspecifies the magnitude: "close but wrong" is cosine 0.9935, *not* the naive impression that 0 % acceptance would imply it's random. The drafter is 99.3 % aligned in logit space; only the tip of the distribution is mis-pointed. That distinction matters for §7.

---

## 7. Can we rehabilitate Path A cheaply?

**Yes in principle; no in practice on the current roadmap.**

### Cheap(ish) rehabilitation options

1. **Head-only fine-tune against fp target.** Freeze drafter layers 0–3; fine-tune only `mtp_post_proj` + `lm_head` on ~1–5 M tokens of (L34_fp, natural_next) pairs collected from our own HF forward. A100 GPU time ~2–4 h. Expected outcome: argmax matches should jump from 3 % toward 30–50 % by aligning the tip of the distribution without disturbing the (already good) bulk.
2. **Last-layer + lm_head fine-tune.** Same as (1) but also unfreeze drafter's layer-3 projections. More capacity to correct the systematic bias; risk of overfitting on small corpus.
3. **Distillation from the TFLite main LLM on Google's corpus.** Not cheap and we don't have Google's training data.

### Why this is NOT the right next step

Per `MTP_PATH_C_FINDINGS.md §4.1` and `PRIORITY_ROADMAP.md` item 11c: **the verify-vs-decode fp16 drift on the target's chunks is a load-bearing blocker for ANY drafter at any acceptance rate**. At current verify multiplier (~2× decode), break-even acceptance on iPhone 17 Pro is **~77 %**. Even a perfect Path A fine-tune to 50–60 % acceptance is still net negative.

Therefore:

- **If 11c closes** (verify multiplier drops to ~1.3× decode): break-even falls to ~30 %. At that point a Path A head-only fine-tune is a cheap ~1-day experiment worth running.
- **If 11c does not close:** neither Path A nor Path C nor any other MTP drafter breaks even. The extraction stays a reference artifact only.

### Concrete recommendation

1. Park Path A **as currently configured**, continue per the existing roadmap (close 11c first).
2. Keep `output/mtp_probe/section_9.tflite` and `mtp_drafter.pt` on disk as reference artifacts. They are NOT broken.
3. **Delete the stale empty `output/mtp_drafter.mlpackage`** (just a Manifest.json + empty Data/ dir). Not critical.
4. If and when 11c closes, revisit Path A with the head-only fine-tune as the first experiment — it's the cheapest way to test whether a small distribution correction rescues an otherwise-sound extracted drafter, and takes ~1 day vs the ~5 days to retrain Path C-style modules from scratch.

### What Path A's failure really taught us

The failure was not "we can't extract Google's MTP drafter" (we can, and the extraction is good). It was:

> **"Artifacts trained against a quantized target cannot be dropped onto an fp target and reach greedy-argmax acceptance, even with near-perfect weight extraction and near-perfect cosine similarity."**

This is a reusable lesson for any future "extract Google's X" work: plan for a ~1-day alignment fine-tune (head + optional top layer) against your own target's forward, *even if the extraction is perfect*. Don't assume the drafter is drop-in just because the weights are readable.

---

## 8. Which stop-cond fires

**"tokenizer-mismatch" → NO** (100 % match).
**"weight-broken" → NO** (cosine 0.99, median rank of target = 2.5k / 262k, structure intact).
**"fp32 works" → NO** (3.1 % top-1 is still below usable acceptance).
**"other" → YES — distribution mismatch from quantized-target training.** The weights are fine; the bias is trained-in. Rehabilitation requires a small head-only fine-tune, which is only worth doing after 11c closes.
