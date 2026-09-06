#!/bin/bash
# 本格規模のファインチューニング。事前学習済みの重みを引き継ぐ。
set -e
cd "$(dirname "$0")/.."

python train.py \
    --stage finetune \
    --data data/conversations.txt \
    --init_from checkpoints/pretrain.pt \
    --device cuda \
    --max_steps 3000 \
    --lr 1e-4 \
    --eval_interval 200 \
    "$@"
