#!/bin/bash
# セットアップ〜コーパス収集〜事前学習〜ファインチューニングまで一括実行する。
set -e
cd "$(dirname "$0")"

bash setup.sh
bash build_corpus.sh
bash pretrain.sh
bash finetune.sh

echo "完了しました。'python chat.py' で会話できます。"
