"""
GPT風 decoder-only Transformer のフルスクラッチ実装。

全体の流れ:
    1. 入力トークンID列を「トークン埋め込み」+「位置埋め込み」でベクトル化する
    2. Transformerブロックを何層も通す
             各ブロック = (LayerNorm -> Self-Attention -> 残差加算)
                                    + (LayerNorm -> FeedForward   -> 残差加算)
    3. 最終LayerNormを通し、Linear層で各位置ごとに「次のトークンの確率分布」を出力する
    4. 学習時はCross Entropy Lossで正解の次トークンと比較する
    5. 生成時は出力された確率分布からサンプリングして1トークンずつ文章を伸ばしていく
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128   # 一度に見ることができる文脈の長さ(トークン数)
    n_layer: int = 4        # Transformerブロックの数
    n_head: int = 4         # Attentionのヘッド数
    n_embd: int = 128       # 埋め込みベクトルの次元数
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """
    Self-Attention(自己注意機構)。

    「いま処理しているトークンは、これまでのどのトークンに注目すべきか」を計算する仕組み。
    Query(問い合わせ)とKey(鍵)の内積で「どれくらい似ているか(注目度)」を求め、
    それをsoftmaxで確率に変換し、Value(値)を重み付き平均する。

    Causal(因果的)マスクをかけることで、未来のトークンを見ないようにする。
    (文章を左から右に生成するため、まだ生成していない未来の情報はカンニングできない)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V をまとめて1回の行列演算で計算する(効率化のため)
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 未来を見ないためのマスク(下三角行列)。学習対象ではないのでbufferとして登録する。
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time(系列長), channel(埋め込み次元)

        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # ヘッドごとに分割: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attentionスコア = Q・K^T / sqrt(head_dim) 「スケーリングされた内積」
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 未来位置のスコアを -inf にして、softmax後の重みを0にする
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_dim) 各位置の「注目に基づく加重平均」
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # ヘッドを結合して元の形に戻す

        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    """
    Attentionで集めた情報を、各トークン位置ごとに非線形変換する層。
    Attentionが「トークン間の関係」を扱うのに対し、こちらは「トークンごとの処理」を担当する。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    Transformerブロック1つ分。Pre-LayerNorm方式(GPT-2と同じ)を採用。
    残差接続(x + f(x))により、層を深くしても勾配が消えにくくなる。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 重み共有: 入力埋め込みと出力層で同じ重みを使うことでパラメータ数を削減する
        # (GPT-2などでも採用されている手法)。
        self.token_emb.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, "入力系列が block_size を超えています"

        pos = torch.arange(0, T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)  # トークンの意味 + 位置の情報
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        自己回帰生成: 1トークンずつ予測しては、それを入力に追加していく。

        temperature: 高いほどランダム(多様)、低いほど確定的(保守的)な出力になる。
        top_k: 確率上位k個のトークンだけに絞ってサンプリングする。
        top_p: 累積確率がtop_p以上になる最小限の上位トークンだけに絞る(nucleus sampling)。
            top_kと併用でき、その場合は先にtop_kで絞ってからtop_pを適用する。
        repetition_penalty: 1.0より大きくすると、すでに生成済みのトークンのスコアを
            下げて選ばれにくくする(単調な繰り返しループを抑制する)。
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    seen_logits = logits[b, seen]
                    logits[b, seen] = torch.where(
                        seen_logits > 0,
                        seen_logits / repetition_penalty,
                        seen_logits * repetition_penalty,
                    )

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 累積確率がtop_pを超えた地点より後ろを削除する(先頭の1個は必ず残す)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
