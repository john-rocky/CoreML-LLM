# EAGLE-3 full TTT retrain — final version (all fixes)

**Branch:** `claude/eagle3-ttt` @ commit `5a7d887` or later. Contains all known fixes:

- Streaming memmap collector (no 165 GB RAM OOM)
- Preload option for trainer (in-RAM for fast indexing)
- Sibling-first data_dir resolution
- cuDNN benchmark + TF32 (full GPU utilization)
- **fp32 LM-head matmul for tok_tgt** (fix for the all-zero labels bug)
- Full TTT training (K=3 autoregressive roll-out, not just step 0)
- Sequence-boundary metadata augmenter

**Resource target:** Colab A100 80 GB + ≥ 130 GB system RAM + ≥ 190 GB local disk.
**Wallclock (end-to-end):** ~2-2.5 h in one session.
**Output:** `eagle3_draft_best.pt` + `eagle3_eval.json` + `eagle3_config.json` on Drive.

Run every cell in order in one Colab session. Do not close the tab between Cell 2 and Cell 6 — `/content/` is wiped on disconnect.

---

## Cell 1 — Setup (~1 min)

```python
!rm -rf /content/CoreML-LLM
!git clone -q -b claude/eagle3-ttt https://github.com/john-rocky/CoreML-LLM.git
!pip install -q safetensors transformers datasets tqdm

from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/eagle3_retrain_20260418_ttt
```

Verify the fp32 fix is on the branch (if this prints nothing, STOP — wrong branch):

```python
!grep -A1 "MUST be fp32" /content/CoreML-LLM/conversion/collect_eagle_hidden_states_custom.py
```

Expected: the comment about "MUST be fp32 — with vocab=262144..." prints.

## Cell 2 — Collect fusion hidden states (~30-45 min)

```python
!python /content/CoreML-LLM/conversion/collect_eagle_hidden_states_custom.py \
    --corpus /content/drive/MyDrive/eagle3_retrain_20260417/eagle_corpus.jsonl \
    --output /content/training_data.pt \
    --num-samples 30000 --seq-len 512 \
    --batch-size 16
```

Reuses the existing `eagle_corpus.jsonl` on Drive (no re-download). Writes `~140-180 GB` of memmap files to `/content/training_data.data/`.

## Cell 3 — Sanity check tok_tgt is NOT all zeros (5 seconds)

```python
import numpy as np
import torch

m = torch.load("/content/training_data.pt", map_location="cpu")
t = np.memmap("/content/training_data.data/tok_tgt.dat",
              dtype=np.int64, mode="r", shape=(int(m["total_pairs"]),))

frac_zero = (t[:100000] == 0).mean()
uniq = len(np.unique(t[:10000]))
print(f"first 20: {t[:20]}")
print(f"unique tokens in first 10k: {uniq}")
print(f"% zero in first 100k: {frac_zero*100:.2f}%")

assert frac_zero < 0.10, "tok_tgt is mostly zero — collector is broken. Stop here."
assert uniq > 100, "too few unique tokens — something wrong."
print("OK: tok_tgt looks healthy")
```

If this assertion fails, DO NOT PROCEED — re-check Cell 1 grep output, re-run Cell 2.

## Cell 4 — Augment manifest with seq_starts (~3-5 min)

```python
!python /content/CoreML-LLM/conversion/augment_seq_metadata.py \
    --data /content/training_data.pt \
    --corpus /content/drive/MyDrive/eagle3_retrain_20260417/eagle_corpus.jsonl \
    --num-samples 30000 --seq-len 512
```

Expected final line: `Manifest updated ... Added seq_starts (int64, len ~28501)`.

## Cell 5 — Dry-run sanity (30 seconds)

Catches fatal bugs in the trainer loop before the 90-min full run.

```python
!python /content/CoreML-LLM/conversion/train_eagle3_ttt.py \
    --data /content/training_data.pt \
    --save-dir /content/eagle3_ttt_dryrun \
    --dry-run --preload
```

Pass criteria:
- no traceback
- `TTT training — K=3, weights=[1.0, 0.7, 0.5]` prints
- at least one progress-bar update with acc0 / acc1 / acc2 percentages (any value)

If Cell 5 fails, STOP. Paste the error.

## Cell 6 — Full TTT training (~60-90 min)

Saves checkpoints directly to Drive so a session drop after this cell keeps progress. `--preload` loads ~138 GB into CPU RAM up front (~60-120 s), then training is fast.

```python
!python /content/CoreML-LLM/conversion/train_eagle3_ttt.py \
    --data /content/training_data.pt \
    --save-dir /content/drive/MyDrive/eagle3_retrain_20260418_ttt \
    --epochs 2 --preload
```

Expected log:
```
Preload done in ~60-120s
TTT training — K=3, weights=[1.0, 0.7, 0.5], feature_w=0.1
  step 2000: train_acc=.../.../...  val_acc=.../.../...  val_expL=...
  step 4000: ...
  ** new best: expL=...
...
Training complete in X.X min. Best expL: X.XXX
```

## Cell 7 — Verify outputs (5 s)

```python
!ls -la /content/drive/MyDrive/eagle3_retrain_20260418_ttt/
!cat /content/drive/MyDrive/eagle3_retrain_20260418_ttt/eagle3_eval.json
```

Expected files: `eagle3_draft_best.pt`, `eagle3_draft_final.pt`, `eagle3_config.json`, `eagle3_eval.json`, `eagle3_ttt_training.log`, plus periodic `eagle3_draft_step*.pt`.

## Cell 8 — On-distribution sanity (60 s)

Loads the draft and verifies it does NOT collapse to predicting token 0. This catches a regression of the earlier bug.

```python
import torch, numpy as np
import sys; sys.path.insert(0, "/content/CoreML-LLM/conversion")
from train_eagle3_standalone import EAGLE3Draft, build_rope_cache, HIDDEN, HEAD_DIM, ROPE_THETA

manifest = torch.load("/content/training_data.pt", map_location="cpu")
ckpt = torch.load("/content/drive/MyDrive/eagle3_retrain_20260418_ttt/eagle3_draft_best.pt", map_location="cpu")
draft = EAGLE3Draft(manifest["lm_head_weight"].float()).cuda()
draft.load_state_dict(ckpt["model"])
draft.eval()

shapes, dtypes = manifest["shapes"], manifest["dtypes"]
def open_mm(k):
    return torch.from_numpy(np.memmap(f"/content/training_data.data/{k}.dat",
                                      dtype=np.dtype(dtypes[k]), mode="r",
                                      shape=shapes[k]))

seq_starts = manifest["seq_starts"]
s, e = int(seq_starts[-2]), int(seq_starts[-1])  # last val sequence
tok_tgt = open_mm("tok_tgt")[s:e].cuda().long()
e_in = open_mm("e_in")[s:e].cuda()
fL8 = open_mm("fusion_L8")[s:e].cuda()
fL17 = open_mm("fusion_L17")[s:e].cuda()
fL34 = open_mm("fusion_L34")[s:e].cuda()

COS, SIN = build_rope_cache(1024, HEAD_DIM, ROPE_THETA, "cuda")
valid = (e - s) - 3

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
    h_prev = draft.fuse_target([fL8[:valid], fL17[:valid], fL34[:valid]]).unsqueeze(0)
    e_next = e_in[:valid].unsqueeze(0)
    d_h, logits = draft.step(h_prev, e_next, COS, SIN, is_sequence=True)

pred = logits.argmax(-1)[0]
label = tok_tgt[:valid]
acc = (pred == label).float().mean().item() * 100
uniq_pred = len(torch.unique(pred))

print(f"val seq step-0 acc: {acc:.1f}%")
print(f"unique predictions: {uniq_pred}")
print(f"first 10 predictions: {pred[:10].tolist()}")
print(f"first 10 labels:      {label[:10].tolist()}")
print(f"% zero in predictions: {(pred == 0).float().mean().item()*100:.2f}%")

assert uniq_pred > 5, "predictions collapsed to one token — broken"
print("OK: draft produces varied predictions")
```

## What to paste back

- Output of Cell 7 (`eagle3_eval.json` contents)
- Output of Cell 8 (val acc + unique predictions count)
- Last 20 lines of `eagle3_ttt_training.log`

---

## Go / no-go

| val_acc[0] | val_acc[1] | val_acc[2] | Cell 8 unique | Verdict |
|---|---|---|---|---|
| ≥ 60% | ≥ 30% | ≥ 15% | > 1000 | **Success**. Move to iPhone bench via `build_eagle3.py`. |
| ≥ 60% | < 15% | ≤ 5% | > 1000 | TTT steps 1/2 under-trained. Try +2 epochs or TTT_WEIGHTS=[1.0,1.0,0.7]. |
| 100% | 100% | 100% | ≤ 10 | Trivial-label regression. STOP — paste results, I debug. |
| < 40% | — | — | — | Data or hyperparam issue. STOP — paste results. |

## Failure recovery

- **Cell 2 dies** → re-run Cell 1 + 2 (corpus is on Drive, safe).
- **Cell 3 assertion fails** → fp32 fix didn't take; re-clone (`rm -rf /content/CoreML-LLM` + Cell 1 again).
- **Cell 6 crashes** → `eagle3_draft_step*.pt` snapshots are on Drive every 4000 opt-steps; re-run Cell 6 with `--init-from <last_step_ckpt>`.
- **System RAM OOM during preload** → re-run Cell 6 without `--preload`. Slower (disk-random), but survives smaller RAM.

## What this fixes compared to prior runs

| Prior symptom | Root cause | Fix |
|---|---|---|
| Collection OOM at 38% | all hiddens in RAM lists | memmap streaming (`1e9170a`) |
| Training 3.9 s/iter on SSD | memmap random-read scatter | `--preload` (`9376473`) |
| `data_dir` resolved to stale Drive path | isdir check only, no sibling preference | sibling-first (`229eeb8`) |
| 7 h collection, GPU 0% util | CPU argmax (fp32 matmul per sample) | GPU argmax (`fd12229`) |
| tok_tgt all zeros, 100% val-acc on [0,0,...] | fp16 matmul overflow on vocab=262144 | **fp32 LM-head (`5a7d887`)** |
| Inference 33.3% accept after "training" | step 0 only, TTT 1/2 skipped | full TTT trainer (`b844850`) |

All fixes are on branch `claude/eagle3-ttt`. Clone = get everything.
