#!/bin/bash
# Colab Pro+ A100 runner for Path C training.
# These are COLAB CELL COMMANDS — paste each section into a separate cell.
# Lines starting with `#` are comments / instructions.
# Lines starting with `!` are Colab shell commands.
# Lines without prefix are Python code.

# ============================================================
# Cell 1: Verify GPU
# ============================================================
# !nvidia-smi
# Expect: NVIDIA A100 80GB (Colab Pro+)

# ============================================================
# Cell 2: Install dependencies
# ============================================================
# !pip install -q -U "huggingface_hub[cli]" transformers datasets accelerate
# !pip show torch | grep Version   # should be 2.4+

# ============================================================
# Cell 3: Clone repo
# ============================================================
# !git clone https://github.com/john-rocky/CoreML-LLM.git /content/CoreML-LLM
# !cd /content/CoreML-LLM && git checkout feature/mtp-speculative-v1
# !ls /content/CoreML-LLM/conversion/train_mtp_modules/

# ============================================================
# Cell 4: Local smoke test (NO network, NO HF model) — 30 seconds
# This validates the pipeline before spending A100 time on downloads.
# If this fails, STOP and fix. Don't proceed to real training.
# ============================================================
# !cd /content/CoreML-LLM/conversion && python train_mtp_modules/smoke_test.py
# Expect: "Initial L1=12.8 → Final L1=<0.1 (PASS)" and same for L2

# ============================================================
# Cell 5: HuggingFace login (for Gemma 4 — gated model)
# ============================================================
# Option A: interactive
# from huggingface_hub import login
# login()   # paste your HF token when prompted

# Option B: env var (put your token in Colab Secrets as HF_TOKEN, then:)
# import os
# from huggingface_hub import login
# from google.colab import userdata
# login(token=userdata.get('HF_TOKEN'))

# ============================================================
# Cell 6: Download Gemma 4 (10-15 GB, ~5 min)
# ============================================================
# from huggingface_hub import snapshot_download
# snapshot_download(
#     repo_id="google/gemma-4-E2B-it",
#     local_dir="/content/gemma-4-E2B-it",
#     local_dir_use_symlinks=False,
# )
# !ls /content/gemma-4-E2B-it/   # should contain model.safetensors, tokenizer.json, etc.

# ============================================================
# Cell 7: Mount Drive for checkpoint persistence
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')
# import os
# CACHE_DIR = "/content/drive/MyDrive/mtp_train_cache"
# CKPT_DIR  = "/content/drive/MyDrive/mtp_train_ckpt"
# os.makedirs(CACHE_DIR, exist_ok=True)
# os.makedirs(CKPT_DIR, exist_ok=True)

# ============================================================
# Cell 8: Precompute — START SMALL FIRST
# First attempt: 200 samples × 1 dataset = ~200K tokens, ~15 min.
# Verifies pipeline end-to-end before big run.
# ============================================================
# !cd /content/CoreML-LLM/conversion && python train_mtp_modules/precompute.py \
#     --hf-dir /content/gemma-4-E2B-it \
#     --dataset wikitext \
#     --samples-per-dataset 500 \
#     --seq-len 1024 \
#     --output-dir /content/drive/MyDrive/mtp_train_cache_tiny \
#     --device cuda \
#     --dtype fp16

# ============================================================
# Cell 9: Run training on tiny cache (~10 min) — checks G2 gate
# Early-exit at acc=0.6 so we don't waste time if arch is OK
# ============================================================
# !cd /content/CoreML-LLM/conversion && python train_mtp_modules/train.py \
#     --cache-dir /content/drive/MyDrive/mtp_train_cache_tiny \
#     --hf-dir /content/gemma-4-E2B-it \
#     --k-depth 2 \
#     --batch-size 4 \
#     --lr 5e-4 \
#     --num-epochs 5 \
#     --warmup-steps 50 \
#     --eval-interval 100 \
#     --log-interval 20 \
#     --save-dir /content/drive/MyDrive/mtp_train_ckpt_tiny \
#     --dtype bf16 \
#     --early-exit-acc 0.5

# AT THIS POINT you should see module_1 acc climbing above 20% by step 200.
# If stuck below 5% after 300 steps → arch/data problem, STOP, report back.
# If climbing → proceed to Cell 10 (full precompute + train).

# ============================================================
# Cell 10: BIG precompute (~2 hr, 5M tokens)
# ============================================================
# !cd /content/CoreML-LLM/conversion && python train_mtp_modules/precompute.py \
#     --hf-dir /content/gemma-4-E2B-it \
#     --dataset fineweb-edu stack-small codealpaca \
#     --samples-per-dataset 500 \
#     --seq-len 1024 \
#     --output-dir $CACHE_DIR \
#     --device cuda \
#     --dtype fp16

# ============================================================
# Cell 11: BIG train (~6-8 hr)
# ============================================================
# !cd /content/CoreML-LLM/conversion && python train_mtp_modules/train.py \
#     --cache-dir $CACHE_DIR \
#     --hf-dir /content/gemma-4-E2B-it \
#     --k-depth 2 \
#     --batch-size 8 \
#     --lr 5e-4 \
#     --num-epochs 3 \
#     --warmup-steps 200 \
#     --eval-interval 500 \
#     --loss-weights 1.0 0.8 \
#     --save-dir $CKPT_DIR \
#     --dtype bf16 \
#     --early-exit-acc 0.85

# ============================================================
# Cell 12: Check results + download checkpoint
# ============================================================
# !ls -la $CKPT_DIR
# Best checkpoint is the one with highest eval_acc for module_1.
# Download the `.pt` file from Drive to your Mac for CoreML conversion.

echo "See inline cell instructions above. Run cells in order."
