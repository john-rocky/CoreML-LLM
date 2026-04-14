#!/bin/bash
# Colab Pro+ A100 runner for Path C training.
# Run these cells in a Colab notebook (not as a script).

set -e

# ---------- Cell 1: Setup ----------
# !nvidia-smi  # verify A100

# Install deps (Colab usually has torch pre-installed)
pip install -U transformers datasets accelerate

# Clone repo (use your fork)
# !git clone https://github.com/john-rocky/CoreML-LLM.git
# !cd CoreML-LLM && git checkout feature/mtp-speculative-v1

# Mount Drive for checkpoint persistence
# from google.colab import drive
# drive.mount('/content/drive')
# CKPT_DIR="/content/drive/MyDrive/mtp_train_ckpt"
# CACHE_DIR="/content/drive/MyDrive/mtp_train_cache"
# mkdir -p $CKPT_DIR $CACHE_DIR

# ---------- Cell 2: Download HF Gemma 4 ----------
# huggingface-cli login  # needed for gated Gemma model
# !huggingface-cli download google/gemma-4-E2B-it \
#     --local-dir /content/gemma-4-E2B-it --local-dir-use-symlinks False

HF_DIR="/content/gemma-4-E2B-it"  # or wherever downloaded
CODE_DIR="/content/CoreML-LLM"

# ---------- Cell 3: Precompute L34 hiddens (~2hr for 5M tokens) ----------
cd $CODE_DIR/conversion
python train_mtp_modules/precompute.py \
    --hf-dir "$HF_DIR" \
    --dataset lmsys-chat oasst-ja stack-small \
    --samples-per-dataset 400 \
    --seq-len 1024 \
    --output-dir "$CACHE_DIR" \
    --device cuda \
    --dtype fp16

# Gate G1.5: precompute finished in <3hr for 5M tokens
# Check: ls $CACHE_DIR/*.tokens.npy | wc -l   # should be ~20-30 shards

# ---------- Cell 4: Train (4-8hr) ----------
python train_mtp_modules/train.py \
    --cache-dir "$CACHE_DIR" \
    --hf-dir "$HF_DIR" \
    --k-depth 2 \
    --batch-size 8 \
    --lr 5e-4 \
    --num-epochs 3 \
    --warmup-steps 200 \
    --eval-interval 500 \
    --loss-weights 1.0 0.8 \
    --save-dir "$CKPT_DIR" \
    --dtype bf16 \
    --early-exit-acc 0.85

# Watch for Gate G2:
#   After ~500 steps, module_1 top-1 should be 20-40%
#   After ~2000 steps, module_1 top-1 should exceed 50%
#   If <30% after 2000 steps: G2 FAIL, pivot to EAGLE-3

# ---------- Cell 5: Inspect checkpoints ----------
# !ls -la $CKPT_DIR
# Keep best-acc checkpoint (highest module_1 val acc).

# ---------- Cell 6: Download checkpoint back to Mac ----------
# In Colab, the checkpoint is on Drive. On Mac:
# gdown --id <file_id>  OR rclone copy "drive:mtp_train_ckpt/mtp_final.pt" ./output/

echo "Training complete. Proceed to CoreML conversion on Mac."
