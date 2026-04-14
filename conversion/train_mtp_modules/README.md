# Path C: Self-trained MTP modules (DeepSeek V3 style)

Sequential 2-depth MTP modules trained against our frozen Gemma 4 E2B trunk.
Target: ~2.5× speedup over baseline, lossless (greedy verify).

## Context

MTP Path A (Google's extracted drafter) failed — see
`docs/MTP_INTEGRATION_RESULTS.md`. Root cause: the drafter was trained
against LiteRT's W4A8 quantized target, not HF's fp target. Our target
(HF-based) is in a different distribution. Zero acceptance.

Path C trains OUR OWN MTP modules against OUR trunk. No distribution
mismatch possible. Trunk stays frozen — modules are a small addition.

## Architecture

2 sequential transformer blocks (Gemma-style sandwich norm), each ~40 M
params. Total ~80 M trainable. Trunk (2 B) stays frozen.

```
module_k:
  input  = Linear(concat(hidden_prev, embed(tok_{t+k-1})))  # → H=1536
  x      = RMSNorm + attn(self, own KV cache, W=128) + post_attn_norm + resid
  x      = pre_ffw_norm + GeGLU MLP + post_ffw_norm + resid
  h_out  = final_norm(x)
  logits = lm_head(h_out)  # tied with trunk, frozen
```

- No target KV reuse (v1). Module has own W=128 KV cache.
- L34-only input. Multi-layer fusion (L14+L24+L34) deferred to v1.5.
- Shared embedding + shared LM head with trunk.

## Gate structure (waste-minimization)

| Gate | When | Criterion | If fail |
|---|---|---|---|
| **G1: ANE latency** | before training | <20 ms/module on iPhone ANE | abandon, pivot to EAGLE-3 |
| **G1.5: precompute speed** | after Day 1 | 5 M tokens cached in <3 hr | reduce corpus size |
| **G2: module_1 val acc** | after 1 epoch | >50% top-1 | reconsider arch |
| **G3: end-to-end tok/s** | after CoreML deploy | >22 tok/s (+50%) | v1.5: add multi-layer fusion |

**G1 status (2026-04-14)**: Mac ANE = 0.98 ms (body) / 9.37 ms (with LM head).
iPhone A19 Pro ANE is ~1.1× faster than Mac Studio's ANE (17-core vs 16-core).
Provisional PASS. Real iPhone measurement to follow via
`conversion/train_mtp_modules/ane_bench.md` once app integration is added.

## Pipeline (A100 on Colab Pro+)

### 1. Precompute L34 hiddens from a corpus (~2 hr for 5 M tokens)

```bash
cd /content/CoreML-LLM/conversion
python train_mtp_modules/precompute.py \
  --hf-dir /path/to/gemma-4-E2B-it \
  --dataset lmsys-chat stack-small \
  --samples-per-dataset 500 \
  --seq-len 1024 \
  --output-dir ./output/mtp_train_cache \
  --device cuda \
  --dtype fp16
```

Datasets (preset names resolve inside `precompute.py`):

- `lmsys-chat` — LMSYS-Chat-1M, real chat distribution (English-heavy)
- `oasst-ja` — OpenAssistant JA, bilingual JA-EN
- `stack-small` — The Stack (code)
- `codealpaca` — CodeAlpaca instructions
- `c4-en`, `c4-ja` — general web text

Expected output: `./output/mtp_train_cache/shard_0000.tokens.npy`,
`shard_0000.hidden.npy`, etc. ~30 GB for 5 M tokens at seq_len=1024.

### 2. Train (4-8 hr on A100)

```bash
python train_mtp_modules/train.py \
  --cache-dir ./output/mtp_train_cache \
  --hf-dir /path/to/gemma-4-E2B-it \
  --k-depth 2 \
  --batch-size 8 \
  --lr 5e-4 \
  --num-epochs 3 \
  --loss-weights 1.0 0.8 \
  --save-dir ./output/mtp_train_ckpt \
  --dtype bf16 \
  --early-exit-acc 0.85
```

Monitors EMA loss + top-1 acc per module. Saves checkpoint every 500 steps.
Early-exits if module_1 hits 85% top-1 on val (already excellent).

**Expected milestones:**
- Step 500: module_1 acc 20-40% (still learning)
- Step 2000: module_1 acc 50-70% (converging)
- End of epoch 1: module_1 acc 60-80%, module_2 acc 40-60%

If step 2000 acc < 30%: arch problem or data scale too small. Flag G2 fail.

### 3. CoreML conversion (~5 min)

```bash
python train_mtp_modules/build_mtp_coreml.py \
  --ckpt ./output/mtp_train_ckpt/mtp_final.pt \
  --output ./output/mtp_module_1.mlpackage \
  --palettize-int4
```

(Script currently takes first module out of 2. TODO: wire up `--module-idx
0|1` flag to produce separate mlpackages for module_1 and module_2.)

### 4. Deploy + bench on iPhone

```bash
# Compile
xcrun coremlcompiler compile mtp_module_1.mlpackage /tmp/
xcrun coremlcompiler compile mtp_module_2.mlpackage /tmp/

# Deploy via devicectl
xcrun devicectl device copy to \
  --device <DEVICE_UUID> \
  --source /tmp/mtp_module_1.mlmodelc \
  --destination "Documents/Models/gemma4-e2b/mtp_module_1.mlmodelc" \
  --domain-type appDataContainer \
  --domain-identifier com.example.CoreMLLLMChat
# Repeat for mtp_module_2
```

Then use new `MtpModuleStackEngine.swift` (TODO, Phase 3) that replaces
`MtpSpeculativeEngine.swift` — calls 2 modules sequentially per cycle
instead of the old single-drafter auto-regressive loop.

## Files

- `mtp_modules.py` — PyTorch training arch (MtpStack, MtpModule, RMSNorm, RoPE)
- `build_mtp_coreml.py` — ANE-optimized single module, CoreML conversion
- `precompute.py` — Load HF Gemma 4, cache L34 hiddens + tokens to disk
- `data.py` — Dataset for precomputed shards (mmap-backed)
- `train.py` — Training loop with early-exit gate

## Known TODOs (tracked in MTP_INTEGRATION_RESULTS §8)

1. **Batch LM head across 2 modules** — single Conv2d(262144) call for
   concatenated hiddens. Expected savings: ~30-40% of drafter latency.
2. **Multi-layer feature fusion (L14+L24+L34)** — requires exposing L14
   and L24 hidden states from trunk chunks (chunk re-conversion).
3. **Vocab pruning** — drafter-only lm_head with top-50k most-common tokens
   would shrink LM head by 5× at unknown quality cost.
4. **iPhone real ANE latency measurement** — Mac proxy used for G1. Real
   numbers once we add a bench button to CoreMLLLMChat or a CLI tool.
5. **Swift `MtpModuleStackEngine`** — replaces `MtpSpeculativeEngine.swift`;
   drives 2 parallel/sequential module calls per cycle, uses existing
   verify chunks.

## Reference

- DeepSeek V3 paper: 2-token MTP, 85-90% acc on module_2, 1.8× TPS
- Apple 2025 "Your LLM Knows the Future": MTP heads, gated LoRA
- EAGLE-3: feature fusion (low/mid/high) beats top-only
