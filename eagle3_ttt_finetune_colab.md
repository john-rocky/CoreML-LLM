# EAGLE-3 TTT — 3-round LR decay fine-tune (Colab)

**Purpose:** plateau を抜けて val_expL 1.84 → ~1.92-1.95 を狙う。lr を段階的に下げて近傍最適化。

**Prereq:**
- `/content/training_data.pt` + `/content/training_data.data/` が既存（前回 collection の成果物）
- `/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus2/eagle3_draft_best.pt` が既存（前回 +2 ep warm-start の best、val_expL=1.843）
- CoreML-LLM が `/content/CoreML-LLM` に clone 済 for branch `claude/eagle3-ttt`
- preload 用 ~130 GB RAM（Colab Pro+ instance）

**Wallclock:** Round 1 ~35 min + Round 2 ~35 min + Round 3 ~35 min ≈ **1h50min**（preload は各 round ~7分含む）

**保証される動作:** 各 round で val_expL が前 round より上がる、もし上がらなくても `--init-from` 経由で前 round の best を引き継ぐので **後退はない**。

---

## Cell 0 — 確認（5 秒、スキップ可）

```python
!ls /content/training_data.pt /content/training_data.data/ 2>&1 | head -3
!ls /content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus2/eagle3_draft_best.pt 2>&1
!cd /content/CoreML-LLM && git log --oneline -1
```

Round 1 の `--init-from` がこの best.pt を指すので存在必須。無ければ前セクションの plus2 訓練からやり直し。

## Cell 1 — Round 1: lr=1e-4（中 lr で広域探査、~35 min）

plateau から一旦高めの lr で揺らして新しい basin を探る。

```python
!python /content/CoreML-LLM/conversion/train_eagle3_ttt.py \
    --data /content/training_data.pt \
    --save-dir /content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus4 \
    --init-from /content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus2/eagle3_draft_best.pt \
    --epochs 2 --preload \
    --lr 1e-4 --warmup 200
```

期待: `val_expL` が **1.87-1.89** あたりに到達。acc0 は微増（+1-2%）、acc1/2 は +2-3% ずつ伸びるはず。

## Cell 2 — Round 1 結果確認（5 秒）

```python
import json
with open("/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus4/eagle3_eval.json") as f:
    r1 = json.load(f)
print(f"Round 1 (lr=1e-4, +2 ep):")
print(f"  val_acc[0..2]: {[f'{a*100:.1f}%' for a in r1['final_val_acc']]}")
print(f"  val_expL    : {r1['final_val_expL']:.3f}")
print(f"  best_expL   : {r1['best_val_expL']:.3f}")
```

`best_expL` が 1.843（前段）より上がってればOK。下がってたら lr=1e-4 は強すぎ → Cell 3 の lr を 5e-5 に下げて進む。

## Cell 3 — Round 2: lr=3e-5（局所最適化、~35 min）

Round 1 後の状態から、さらに lr を 1/3 に下げてじわじわ改善。

```python
!python /content/CoreML-LLM/conversion/train_eagle3_ttt.py \
    --data /content/training_data.pt \
    --save-dir /content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus6 \
    --init-from /content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus4/eagle3_draft_best.pt \
    --epochs 2 --preload \
    --lr 3e-5 --warmup 100
```

期待: `val_expL` が **1.90-1.92**。ここで acc0 の伸びは頭打ちでも、acc1/acc2 はまだ伸びる余地あり。

## Cell 4 — Round 2 結果確認

```python
import json
with open("/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus6/eagle3_eval.json") as f:
    r2 = json.load(f)
print(f"Round 2 (lr=3e-5, +2 ep):")
print(f"  val_acc[0..2]: {[f'{a*100:.1f}%' for a in r2['final_val_acc']]}")
print(f"  val_expL    : {r2['final_val_expL']:.3f}")
print(f"  best_expL   : {r2['best_val_expL']:.3f}")
```

`best_expL` が Round 1 より上がってれば Round 3 へ。同じ or 下がってたら収束している可能性、Cell 5 を skip して終了判断。

## Cell 5 — Round 3: lr=1e-5（final annealing、~35 min）

最終 polish。lr=1e-5 は SGD 的な細かい近傍探索、plateau 抜けの最後の一押し。

```python
!python /content/CoreML-LLM/conversion/train_eagle3_ttt.py \
    --data /content/training_data.pt \
    --save-dir /content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus8 \
    --init-from /content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus6/eagle3_draft_best.pt \
    --epochs 2 --preload \
    --lr 1e-5 --warmup 50
```

期待: `val_expL` が **1.92-1.95**。ここで完全に plateau に到達。

## Cell 6 — 最終結果まとめ（5 秒）

```python
import json

def load_eval(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return None

runs = [
    ("Original (2 ep)",  "/content/drive/MyDrive/eagle3_retrain_20260418_ttt/eagle3_eval.json"),
    ("+2 warm (plus2)", "/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus2/eagle3_eval.json"),
    ("Round 1 (lr=1e-4)", "/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus4/eagle3_eval.json"),
    ("Round 2 (lr=3e-5)", "/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus6/eagle3_eval.json"),
    ("Round 3 (lr=1e-5)", "/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus8/eagle3_eval.json"),
]

print(f"{'stage':<20} {'acc0':>8} {'acc1':>8} {'acc2':>8} {'expL':>8}")
for name, path in runs:
    e = load_eval(path)
    if e is None:
        print(f"{name:<20}  (not yet run)")
        continue
    a = e['final_val_acc']
    print(f"{name:<20} {a[0]*100:>7.1f}% {a[1]*100:>7.1f}% {a[2]*100:>7.1f}% {e['best_val_expL']:>8.3f}")
```

**最良の stage の best.pt を採用**。通常は Round 3 の plus8 が最終、もし途中で drop したら plus6 or plus4 を採用。

## Cell 7 — 最終 best.pt の retrieval 指示

```python
# どこから download するかを判定
import json, os

best_dir = None
best_expL = 0.0
for path in [
    "/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus8",
    "/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus6",
    "/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus4",
    "/content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus2",
]:
    ep = os.path.join(path, "eagle3_eval.json")
    if os.path.exists(ep):
        with open(ep) as f:
            e = json.load(f)
        if e['best_val_expL'] > best_expL:
            best_expL = e['best_val_expL']
            best_dir = path

print(f"Best checkpoint: {best_dir}")
print(f"  val_expL: {best_expL:.3f}")
print(f"  Download: {best_dir}/eagle3_draft_best.pt + eagle3_config.json + eagle3_eval.json")
```

このパスの `eagle3_draft_best.pt` を Mac に転送 → `build_eagle3.py` で mlpackage 化 → iPhone bench。

---

## Fallback / 中止判定

**Round 1 の val_expL が 1.843 (plus2) より下がった場合:**

- lr=1e-4 が現状の weights に対して大きすぎ（既に鞍点から十分降りた状態）
- Round 2 を `--lr 5e-5 --warmup 100` に変更して続行
- Round 3 は予定通り `--lr 1e-5`

**Round 2 の val_expL が Round 1 と同等（±0.005 以内）の場合:**

- 局所最適到達、これ以上 lr 下げても微増しか得られない
- Round 3 skip 推奨、plus6 の best を最終採用

**Round 3 の val_expL が Round 2 と同等の場合:**

- 完全収束。これ以上訓練は意味なし
- plus6 or plus8 どちらも同程度、好きな方選択

## Colab credit 余裕があればさらに

Round 3 完了後、**data 倍増で fresh train** も選択肢:

```python
# Step 1: 古い data 削除 (160 GB 解放)
!rm -rf /content/training_data.pt /content/training_data.data

# Step 2: 60k で recollect (seq_len=384 で ~210 GB、disk ギリギリ)
#         Colab Pro+ の 190 GB disk だと入らない可能性あるので --seq-len 256 で妥協
!python /content/CoreML-LLM/conversion/collect_eagle_hidden_states_custom.py \
    --corpus /content/drive/MyDrive/eagle3_retrain_20260417/eagle_corpus.jsonl \
    --output /content/training_data.pt \
    --num-samples 50000 --seq-len 256 --batch-size 16

# Step 3: augment + TTT fresh train + 4 epochs
!python /content/CoreML-LLM/conversion/augment_seq_metadata.py \
    --data /content/training_data.pt \
    --corpus /content/drive/MyDrive/eagle3_retrain_20260417/eagle_corpus.jsonl \
    --num-samples 50000 --seq-len 256
!python /content/CoreML-LLM/conversion/train_eagle3_ttt.py \
    --data /content/training_data.pt \
    --save-dir /content/drive/MyDrive/eagle3_retrain_20260418_ttt_50k \
    --init-from /content/drive/MyDrive/eagle3_retrain_20260418_ttt_plus8/eagle3_draft_best.pt \
    --epochs 4 --preload --lr 5e-5 --warmup 200
```

ただし disk ギリギリ + collection 45 min + training 1h で合計 +2h。確実性は下がる（disk/data 操作の failure）ので、**Round 1-3 完走 → iPhone bench 優先**が推奨。
