#!/bin/bash
# 本格規模の事前学習。GPU向けにモデルを大きくしている。
# VRAMが少ないGPUの場合は末尾に上書き引数を足せる(例: bash runpod/pretrain.sh --batch_size 32)
set -e
cd "$(dirname "$0")/.."

python train.py \
    --stage pretrain \
    --data data/pretrain_corpus.txt \
    --vocab_extra data/conversations.txt \
    --device cuda \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 384 \
    --block_size 256 \
    --batch_size 64 \
    --max_steps 20000 \
    --eval_interval 500 \
    "$@"
