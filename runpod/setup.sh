#!/bin/bash
# RunPod上でのセットアップ。GPU認識の確認まで行う。
set -e
cd "$(dirname "$0")/.."

pip install -r requirements.txt

python -c "
import torch
print('CUDA利用可能:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM(GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
"
