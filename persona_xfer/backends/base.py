"""バックエンド共通のインターフェース。

- Chat: system / user と、アシスタント応答の書き出し（prefix）。meta は合成モデル用の付加情報で、
  実モデルは無視する。
- Steer: 層 layer の残差ストリームに α·v を加える指定。
- 実モデル（活性化にアクセスできるもの）は has_activations=True。
  テキストしか扱えない API モデルは False で、ステアリング系の条件は実行しない
  （報告書 4.3 の「Frontier 側には活性化を操作する口がない」をそのまま反映する）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chat:
    system: str
    user: str
    prefix: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Steer:
    layer: int
    vector: np.ndarray
    alpha: float = 1.0


class Backend:
    name: str = "backend"
    has_activations: bool = False
    n_layers: int = 0
    d_model: int = 0

    def hidden(self, chats: list[Chat], layers: list[int]) -> dict[int, np.ndarray]:
        """各チャットの最後のトークン位置での残差ストリーム（層ごとに [n, d]）。"""
        raise NotImplementedError

    def choice_logprobs(self, chats: list[Chat], labels: list[str],
                        steer: Steer | None = None) -> np.ndarray:
        """次に来る応答が各ラベル（"A","B" や "1".."5"）である対数確率 [n, len(labels)]。
        ラベル間で正規化済みである必要はない（呼び出し側で正規化する）。"""
        raise NotImplementedError

    def text_nll(self, texts: list[str], steer: Steer | None = None) -> float:
        """中立テキストの1トークンあたり負の対数尤度（能力保持の確認用）。"""
        raise NotImplementedError
