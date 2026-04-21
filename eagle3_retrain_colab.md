# EAGLE-3 custom retrain — Colab paste script

**Runtime target:** A100 (40 GB). Total wallclock ~3-5h (corpus 5-10 min + hidden-state collection 2-4h + training 25-30 min).

**Branch:** `claude/eagle3-retrain-custom` (collector + trainer live there; `cf0485b` added fusion layers, `2ae26a4` added the standalone trainer).

**From-scratch because:** local `training_data_custom.pt` (2026-04-13) may be pre-fusion-layers (v1), incompatible with the current trainer which requires `train_fusion_L8/L17/L34` keys.

---

## Cell 1 — Clone + deps (~1 min)

```python
!git clone -q -b claude/eagle3-retrain-custom https://github.com/john-rocky/CoreML-LLM.git
%cd CoreML-LLM/conversion
!pip install -q safetensors transformers datasets tqdm
```

## Cell 2 — Mount Drive (~5 s)

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/eagle3_retrain_20260417
```

## Cell 3 — Rebuild text corpus (~5-10 min)

Gemma-4 chat-formatted JSONL from wikitext / C4 / Alpaca / Dolly / CodeAlpaca / UltraChat.

```python
!python download_eagle_corpus.py \
    --output /content/drive/MyDrive/eagle3_retrain_20260417/eagle_corpus.jsonl \
    --num-samples 30000
```

## Cell 4 — Collect fusion hidden states from custom `Gemma4Model` (~2-4h on A100)

Runs the same 35-layer per-token forward path that CoreML chunks use on-device. Emits `(input, target, fusion_L8, fusion_L17, fusion_L34)` tuples with the exact keys `train_eagle3_standalone.py` expects.

```python
!python collect_eagle_hidden_states_custom.py \
    --corpus /content/drive/MyDrive/eagle3_retrain_20260417/eagle_corpus.jsonl \
    --output /content/drive/MyDrive/eagle3_retrain_20260417/training_data.pt \
    --num-samples 30000 --seq-len 512
```

Notes:
- First run downloads HF Gemma 4 E2B weights (~10 GB, a few minutes). Not gated; anonymous download works.
- If Colab runtime dies mid-collection, `training_data.pt` is only written at the end — you will need to restart the cell. Consider `--num-samples 10000` for a cheaper first pass.

## Cell 5 — Train draft (~25-30 min on A100)

```python
!python train_eagle3_standalone.py \
    --data /content/drive/MyDrive/eagle3_retrain_20260417/training_data.pt \
    --save-dir /content/drive/MyDrive/eagle3_retrain_20260417 \
    --epochs 2
```

## Cell 6 — Sanity check (optional, ~2 min)

```python
!python test_eagle3_infer.py \
    --ckpt /content/drive/MyDrive/eagle3_retrain_20260417/eagle3_draft_best.pt \
    --prompt "The capital of Japan is" \
    --max-new 64 --K 3 --device cuda
```

Expected: `outputs match: True`, accept rate ≥ 50%. If the accept rate on this Colab sanity is below 50%, stop — the retrain did not solve Blocker 1.

---

## Outputs in `/content/drive/MyDrive/eagle3_retrain_20260417/`

| File | Purpose |
|---|---|
| `eagle_corpus.jsonl` | chat-formatted text corpus (reusable for future runs) |
| `training_data.pt` | fusion hidden-state dataset (reusable — skip Cell 4 on re-runs) |
| `eagle3_draft_best.pt` | retrained draft checkpoint — goes to Mac for CoreML convert |
| `eagle3_config.json` | architecture config |
| `eagle3_eval.json` | **acc[0], acc[1], acc[2], expL numbers** — this is the go/no-go signal |

## What to paste back after completion

- `eagle3_eval.json` contents (acc[0..2], expL)
- `training_data.pt` file size (sanity check on sample count)
- Cell 6 output (accept rate on the sanity prompt)

Track B (11c) and Track C (S0) consumers of the new draft wait on the acc[0] number — ≥ 50% is the minimum to proceed with CoreML convert; < 40% triggers a retrain-recipe review.

---

## Resume / re-run notes

- **To retrain with more epochs:** Cell 5 with `--epochs 4` (or whatever). `training_data.pt` is already on Drive, so Cell 4 stays skipped.
- **To try a different corpus mix:** rerun Cell 3 with different `--num-samples` or regenerate the JSONL from a different source list, then Cell 4 → Cell 5.
- **If Colab times out during Cell 4:** reduce `--num-samples` to 10000 for a proof-of-life, then scale up once convinced the pipeline runs.
- **iPhone deploy after:** checkpoint goes through `conversion/build_eagle3.py --ckpt eagle3_draft_best.pt` on Mac (not Colab) — see `docs/EAGLE3_DEPLOY.md` §3.4.
