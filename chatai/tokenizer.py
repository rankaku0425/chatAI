"""
文字レベル(character-level)トークナイザー。

仕組み:
    - 学習データに登場する文字を全て集めて「文字 <-> 整数ID」の対応表(語彙)を作る。
    - ニューラルネットは文字を直接理解できないので、
        文字列 -> ID列 に変換(encode)してからモデルに入力し、
        モデルが出力したID列 -> 文字列 に戻す(decode)ために使う。
    - 単語やサブワード分割(BPEなど)を使わない最も単純な方式なので、
        語彙サイズが小さく仕組みが理解しやすい(その分、同じ長さの文章でもトークン数は多くなる)。
"""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, vocab: list[str] | None = None):
        self.vocab = vocab or []
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """学習データ文字列から語彙を構築する。"""
        vocab = sorted(set(text))
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab)
