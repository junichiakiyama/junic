"""Claude（Anthropic API）用バックエンド。テキスト条件（base / prompt / retrieval）だけを扱う。

商用 API は活性化にも対数確率にもアクセスできないため、選択は生成した1文字で判定し、
n_samples 回の回答の頻度を確率の代わりにする（既定は1回。選択肢の順番を入れ替えた2通りを
必ず測るので、1項目あたり 0 / 0.5 / 1 の3値になる）。
アシスタント応答の書き出し（prefill）は現行モデルでは使えないので、指示文で1文字回答を求める。
拒否（stop_reason == "refusal"）は欠測として扱う。別モデルへのフォールバックは測定が
別のモデルの回答に置き換わるため、意図的に使わない。
"""
from __future__ import annotations

import re

import numpy as np

from .base import Backend, Chat, Steer

_EPS = 1e-6


class AnthropicBackend(Backend):
    has_activations = False

    def __init__(self, model: str = "claude-opus-5-5", *, name: str | None = None,
                 n_samples: int = 1, effort: str = "low", max_tokens: int = 4000):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = name or model
        self.n_samples = n_samples
        self.effort = effort
        self.max_tokens = max_tokens
        self.n_refusals = 0

    def _ask(self, chat: Chat) -> str | None:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=chat.system,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": chat.user}],
        )
        if r.stop_reason == "refusal":
            self.n_refusals += 1
            return None
        return "".join(b.text for b in r.content if b.type == "text")

    def choice_logprobs(self, chats, labels, steer: Steer | None = None):
        if steer is not None:
            raise ValueError("API models cannot be steered")
        pat = re.compile("|".join(re.escape(l) for l in sorted(labels, key=len, reverse=True)))
        out = np.full((len(chats), len(labels)), np.nan)
        for i, c in enumerate(chats):
            counts = np.zeros(len(labels))
            for _ in range(self.n_samples):
                txt = self._ask(c)
                if txt is None:
                    continue
                m = pat.search(txt.translate(str.maketrans("ＡＢ１２３４５", "AB12345")))
                if m:
                    counts[labels.index(m.group(0))] += 1
            if counts.sum() > 0:
                out[i] = np.log(np.maximum(counts / counts.sum(), _EPS))
        return out

    def hidden(self, chats, layers):
        raise NotImplementedError("API models expose no activations")

    def text_nll(self, texts, steer=None):
        raise NotImplementedError("API models expose no log-probabilities")
