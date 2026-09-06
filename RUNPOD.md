# RunPodでの本格規模学習

自宅CPU用の小規模設定(`train.py`のデフォルト値、`README.md`参照)とは別に、
GPU向けの本格規模設定を `runpod/` 配下にまとめています。

## 1. コードをRunPodに送る (GitHub経由)

ローカルで作業した内容をGitHubにpushし、RunPod側で`git clone`します。

```bash
# ローカル側 (まだリモート未設定の場合)
git remote add origin <あなたのGitHubリポジトリURL>
git push -u origin master
```

```bash
# RunPod側 (Podのターミナルで)
git clone <あなたのGitHubリポジトリURL>
cd <リポジトリ名>
```

`checkpoints/`, `data/pretrain_corpus.txt`, `data/.cache/` は`.gitignore`で
除外されているので、RunPod側で改めて生成することになります。

## 2. セットアップ〜学習の実行

```bash
bash runpod/setup.sh          # 依存関係インストール + GPU認識確認
bash runpod/build_corpus.sh   # 大規模コーパス収集 (デフォルト2000万文字、時間がかかる)
bash runpod/pretrain.sh       # 事前学習 (n_layer12, n_embd384, block_size256)
bash runpod/finetune.sh       # ファインチューニング
python chat.py                # 会話確認
```

まとめて実行したい場合:

```bash
bash runpod/run_all.sh
```

## 3. GPUのVRAMに応じて調整する

各スクリプトは末尾に引数を追加すると上書きできます。

```bash
# VRAMが少ない場合、バッチサイズを下げる
bash runpod/pretrain.sh --batch_size 32

# コーパスを小さくしたい場合
TARGET_CHARS=5000000 bash runpod/build_corpus.sh
```

## 4. 学習済みモデルをローカルに持ち帰る

`checkpoints/`はgit管理外なので、`runpodctl`でファイル転送するのが簡単です。

```bash
# RunPod側
runpodctl send checkpoints/model.pt
runpodctl send checkpoints/vocab.json

# ローカル側 (表示されたワンタイムコードを使う)
runpodctl receive <コード>
```

## 自宅(CPU)版との違い

| | 自宅版 (`train.py`のデフォルト) | RunPod版 (`runpod/`) |
|---|---|---|
| device | cpu | cuda |
| n_layer / n_head / n_embd | 6 / 6 / 192 | 12 / 12 / 384 |
| block_size | 96 | 256 |
| batch_size | 32 | 64 |
| 事前学習コーパス目標文字数 | 数十万〜100万 | 2000万(既定) |
| pretrain max_steps | 5000程度 | 20000 |

コマンドの使い分けだけで、コード本体(`chatai/`, `train.py`など)は共通です。
