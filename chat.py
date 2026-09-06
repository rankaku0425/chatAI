"""
学習済みモデルと対話するCLIチャット。

使い方:
    python chat.py
「User: 」の後にメッセージを入力すると、モデルが続く応答を生成します。
「exit」と入力すると終了します。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chatai.model import GPT
from chatai.tokenizer import CharTokenizer

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "model.pt"))
    parser.add_argument("--vocab", default=str(ROOT / "checkpoints" / "vocab.json"))
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus sampling。Noneで無効化")
    parser.add_argument("--repetition_penalty", type=float, default=1.3, help="1.0で無効。大きいほど繰り返しを抑制")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"チェックポイントが見つかりません: {args.checkpoint}\n"
            "先に `python train.py` を実行してモデルを学習してください。"
        )

    tokenizer = CharTokenizer.load(args.vocab)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = GPT(ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("チャット開始 (終了するには 'exit' と入力してください)")
    history = ""
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break

        history += f"User: {user_input}\nAI:"
        idx = torch.tensor([tokenizer.encode(history)], dtype=torch.long, device=args.device)

        out = model.generate(
            idx,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        generated = tokenizer.decode(out[0].tolist())

        # 新しく生成された部分だけ取り出し、次の "User:" が出てきたところで打ち切る
        new_text = generated[len(history) :]
        reply = new_text.split("User:")[0].strip()

        print(f"AI:{reply}")
        history += f" {reply}\n"


if __name__ == "__main__":
    main()
