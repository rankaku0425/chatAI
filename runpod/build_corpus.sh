#!/bin/bash
# 本格規模の事前学習コーパスを収集する(GPU側はネットワーク/CPU処理なので時間がかかる)。
# 例: TARGET_CHARS=5000000 bash runpod/build_corpus.sh
# 例: SOURCE=aozora bash runpod/build_corpus.sh   (青空文庫のみで学習したい場合)
set -e
cd "$(dirname "$0")/.."

python scripts/build_corpus.py \
    --source "${SOURCE:-both}" \
    --target_chars "${TARGET_CHARS:-20000000}" \
    "$@"
