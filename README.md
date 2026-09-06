# 自作AI - スクラッチ実装のチャットAI

既存のLLMやローカルAIモデルを使わず、Transformer(GPT系decoder-only)を
PyTorchでフルスクラッチ実装し、自分のデータで学習させるプロジェクトです。

## 構成

```
自作AI/
    chatai/
        tokenizer.py   文字レベルトークナイザー
        model.py       GPT本体 (Self-Attention, FeedForward, Blockを自前実装)
    data/
        conversations.txt    学習用の会話データ (サンプル、ファインチューニング用)
        pretrain_corpus.txt  事前学習用の大量テキスト (scripts/build_corpus.pyで生成)
    scripts/
        build_corpus.py  Wikipedia日本語版・青空文庫から事前学習コーパスを収集するスクリプト
    train.py           学習スクリプト (事前学習・ファインチューニング両対応)
    chat.py            学習済みモデルと会話するCLI
    webapp.py          ブラウザで会話・学習操作ができるWebアプリ (Flask)
    templates/         webapp.py用のHTMLテンプレート
    checkpoints/       学習後に生成される (pretrain.pt, model.pt, vocab.json)
```

収集したデータの出典は `data/sources.csv`(生ログ) と `data/SOURCES.md`
(集計・一覧表)に自動記録されます。「どのデータをどこから集めたか」を
あとから確認できます。これらは実行したマシンごとに内容が変わるため
`.gitignore`対象です(PCとRunPodで別々に集計されます)。

## セットアップ

```bash
pip install -r requirements.txt
```

## 学習方法: 2段階学習(事前学習 → ファインチューニング)

`data/conversations.txt` だけで学習すると、データ量が少なすぎて
「単なるパターン暗記」以上のことは学べません。そこで、実際のGPTと同様に
以下の2段階で学習させます。

### 1. 事前学習用コーパスを集める

2つの収集元から選べます。

```bash
# Wikipedia日本語版のランダムな記事本文 (百科事典的な文章)
python scripts/build_corpus.py --source wikipedia --target_chars 1000000

# 青空文庫の著作権切れ作品 (小説・随筆など、公式カタログCSVから著作権フラグ「なし」のみ選別)
python scripts/build_corpus.py --source aozora --target_chars 1000000

# 両方を半分ずつ
python scripts/build_corpus.py --source both --target_chars 1000000
```

`data/pretrain_corpus.txt` に約100万文字のテキストが保存されます
(`--target_chars` で調整可能。CPUで学習するなら数十万〜数百万文字が現実的)。
青空文庫の作品一覧は初回のみ公式カタログCSVをダウンロードし、
`data/.cache/` にキャッシュされます(2回目以降は再ダウンロードしません)。

収集したデータの出典(タイトル・URL・文字数・収集日時)は自動的に
`data/sources.csv` と `data/SOURCES.md` に記録されます。

### 2. 事前学習(pretrain)

```bash
python train.py --stage pretrain --data data/pretrain_corpus.txt --vocab_extra data/conversations.txt --max_steps 5000
```

大量の一般的な日本語テキストで「日本語としての文法・語彙」を学ばせます。
`--vocab_extra` で会話データの文字も語彙に含めておくことで、後段の
ファインチューニングで未知の文字によるエラーが起きないようにします。
`checkpoints/pretrain.pt` と `checkpoints/vocab.json` が生成されます。

### 3. ファインチューニング(finetune)

```bash
python train.py --stage finetune --data data/conversations.txt --init_from checkpoints/pretrain.pt --max_steps 800 --lr 1e-4
```

事前学習済みの重みを読み込み、少量の会話データで「対話の受け答えパターン」を
仕込みます。学習率(`--lr`)を事前学習より小さくすることで、事前学習で得た
日本語の基礎力を大きく崩さずに対話スタイルだけを追加学習します。
結果は `checkpoints/model.pt` に保存され、`chat.py` はこれを読み込みます。

### 事前学習を省略する場合

従来通り、会話データだけでゼロから学習することもできます。

```bash
python train.py --stage finetune --data data/conversations.txt
```

## チャット

```bash
python chat.py
```

## ブラウザGUI

CLIの代わりに、ブラウザから会話・学習操作ができるWebアプリも用意しています。

```bash
python webapp.py
```

`http://127.0.0.1:5000` を開くと会話画面、`/train` で学習操作パネル
(コーパス収集・事前学習・ファインチューニングをブラウザから実行し、
ログをリアルタイムで確認できる)にアクセスできます。3つのジョブは
同時実行できないため、1つずつ順番に実行してください。

## アーキテクチャの仕組み

1. **トークナイザー** (`chatai/tokenizer.py`)
     文字ひとつひとつにIDを振るだけの最も単純な方式。サブワード分割(BPE等)は
     使わないため語彙構築の仕組みが理解しやすい。

2. **埋め込み** (`GPT.forward`)
     各トークンIDを「トークン埋め込みベクトル」に変換し、さらに
     「その位置(何番目のトークンか)を表す位置埋め込みベクトル」を足し合わせる。

3. **Self-Attention** (`CausalSelfAttention`)
     各トークンが文脈中のどこに注目すべきかを、Query・Key・Valueの内積計算で
     求める仕組み。未来のトークンを見ないように因果マスク(下三角行列)をかける。

4. **Feed Forward**
     Attentionで集めた情報をトークンごとに非線形変換する層。

5. **Block**
     LayerNorm → Attention → 残差接続、LayerNorm → FeedForward → 残差接続
     という構成をn_layer回積み重ねる(GPT-2と同じPre-LayerNorm方式)。

6. **学習**
     「次のトークンを予測する」タスク(言語モデリング)としてCross Entropy Lossを
     最小化する。会話データを `User: 質問\nAI: 回答` という形式で学習させることで、
     `User: ...` の後に自然と `AI: ...` らしい文章が続くパターンを覚えさせる。

7. **生成** (`GPT.generate`)
     1トークンずつ予測 → サンプリング → 入力に追加、を繰り返す自己回帰生成。
     `temperature` で多様性、`top_k` で候補の絞り込みを調整できる。

8. **事前学習 → ファインチューニング**
     大量の一般テキストで基礎的な言語能力を学ばせてから、少量の会話データで
     対話スタイルを追加学習する2段階方式。少ない対話データでも、事前学習で
     培った日本語の文法・語彙力を活かせるため、応答がより自然になりやすい。

## スケールアップする場合

- GPUがある場合: `python train.py --device cuda`
- モデルを大きくする: `--n_layer 8 --n_head 8 --n_embd 256 --block_size 128`
    (事前学習・ファインチューニングで同じ値を使う必要がある点に注意。
    `--init_from` 使用時はチェックポイントの設定がそのまま使われる)
- 事前学習コーパスを増やす: `scripts/build_corpus.py --target_chars 5000000` など
- 会話データを増やす: `data/conversations.txt` に `User: ...` / `AI: ...` の
    ペアを追加する
- 文字レベルでは限界があるため、より高品質を目指す場合はサブワード分割
    (例: SentencePiece, BPE)へのトークナイザー差し替えが次のステップになる

## 学習目的での読み方の順番

1. `chatai/tokenizer.py` - データがどうID列になるか
2. `chatai/model.py` の `CausalSelfAttention` - Attentionの核心
3. `chatai/model.py` の `GPT.forward` - 全体の流れ
4. `train.py` - 学習ループ、事前学習とファインチューニングの繋ぎ方
5. `chat.py` - 推論・生成の使い方
