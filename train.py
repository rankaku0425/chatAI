"""
GPTモデルの学習スクリプト。2段階学習(事前学習 → ファインチューニング)に対応。

    事前学習(pretrain): 大量の一般的な日本語テキストで「言語モデルとしての基礎力」を学ばせる
    ファインチューニング(finetune): 事前学習済みの重みから、少量の会話データで「対話らしさ」を仕込む

実行例:
    # 1. 事前学習用コーパスを用意する (scripts/build_corpus.py で作成)
    python scripts/build_corpus.py --target_chars 1000000

    # 2. 事前学習 (--vocab_extra に会話データを渡し、後で使う文字も語彙に含めておく)
    python train.py --stage pretrain --data data/pretrain_corpus.txt --vocab_extra data/conversations.txt --max_steps 5000

    # 3. 会話データでファインチューニング
    python train.py --stage finetune --data data/conversations.txt --init_from checkpoints/pretrain.pt --max_steps 800 --lr 1e-4

    # 事前学習を使わず、従来通り会話データだけでゼロから学習する場合:
    python train.py --stage finetune --data data/conversations.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chatai.model import GPT, GPTConfig
from chatai.tokenizer import CharTokenizer

ROOT = Path(__file__).parent


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    high = len(data) - block_size - 1
    if high <= 0:
        raise ValueError(
            f"データが短すぎます(長さ{len(data)}文字)。block_size({block_size})を減らすか、データを増やしてください。"
        )
    ix = torch.randint(high, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def load_or_build_vocab(args: argparse.Namespace, out_dir: Path) -> CharTokenizer:
    vocab_path = out_dir / "vocab.json"

    # ファインチューニングでは、事前学習時と同じ「文字 <-> ID」対応を使い続ける必要があるため
    # (IDがずれると埋め込み層の意味が壊れる)、既存の語彙があればそれをそのまま読み込む。
    if args.stage == "finetune" and vocab_path.exists():
        return CharTokenizer.load(vocab_path)

    text = Path(args.data).read_text(encoding="utf-8")
    if args.vocab_extra:
        text += Path(args.vocab_extra).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(vocab_path)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--data", default=str(ROOT / "data" / "conversations.txt"))
    parser.add_argument("--vocab_extra", default=None, help="事前学習時に語彙へ含めたい追加テキスト(例: 会話データ)")
    parser.add_argument("--init_from", default=None, help="この重み(例: checkpoints/pretrain.pt)から学習を再開する")
    parser.add_argument("--out_dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--out_name", default=None, help="保存ファイル名。省略時はstageに応じて自動設定")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=192)
    parser.add_argument("--block_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or ("pretrain.pt" if args.stage == "pretrain" else "model.pt")
    save_path = out_dir / out_name

    tokenizer = load_or_build_vocab(args, out_dir)

    text = Path(args.data).read_text(encoding="utf-8")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(len(data) * (1 - args.val_ratio))
    train_data, val_data = data[:n], data[n:]
    if len(val_data) <= args.block_size + 1:
        print("警告: 検証データが少なすぎるため、学習データを検証にも流用します。")
        val_data = train_data

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=args.device, weights_only=False)
        config: GPTConfig = ckpt["config"]
        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "語彙サイズが一致しません。事前学習時に --vocab_extra で会話データの文字も"
                "語彙に含めていたか確認してください。"
            )
        model = GPT(config).to(args.device)
        model.load_state_dict(ckpt["model_state"])
        print(f"{args.init_from} から重みを読み込みました。")
    else:
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
        )
        model = GPT(config).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{args.stage}] パラメータ数: {n_params:,} / 語彙サイズ: {tokenizer.vocab_size} / "
        f"データ長: {len(data):,}文字 / block_size: {config.block_size}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        losses = {}
        for split, d in [("train", train_data), ("val", val_data)]:
            total = 0.0
            for _ in range(20):
                x, y = get_batch(d, config.block_size, args.batch_size, args.device)
                _, loss = model(x, y)
                total += loss.item()
            losses[split] = total / 20
        model.train()
        return losses

    model.train()
    for step in range(1, args.max_steps + 1):
        x, y = get_batch(train_data, config.block_size, args.batch_size, args.device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            torch.save({"model_state": model.state_dict(), "config": config}, save_path)

    print(f"[{args.stage}] 学習完了。チェックポイントを {save_path} に保存しました。")


if __name__ == "__main__":
    main()
