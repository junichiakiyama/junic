"""パイプライン検証用の合成モデル（実験結果として解釈してはならない）。

目的は、実モデルに GPU 時間を使う前に「解析の連鎖が既知の効果を検出し、効果がないところで
は検出しない」ことを確かめること（シミュレーションによる較正）。

構造（モデル m ごとに次元 d_m と乱数行列 W_m が異なる）
  意味空間 s ∈ R^8（価値の8次元）＋ 雑音特徴 ξ ∈ R^K
  隠れ状態 h_ℓ = W_m^(ℓ) [g_ℓ·s ; ξ] ＋ 雑音   … 層 ℓ ごとに意味の載り方 g_ℓ が違う
  判断の logit(B) = c(α)·(prior_m + λ·w_prompt + decode_m(α·v)) · (tags_B − tags_A) ＋ 項目雑音
  c(α) = 1 / (1 + (|decode_m(α·v)| / 定数)^2) … 強すぎるステアリングで判断が崩れる
  w_prompt はシステムプロンプトに人物がいれば規則込みの重み、いなければ 0。
  decode_m は隠れ状態から意味空間への読み出しで、ステアリングは規則（文脈依存）を表せない。
したがって「写像した人物ベクトルは効く」「規則依存項目ではステアリングが弱い」は
構成上の真実であり、解析がそれを再現できるかを見る。
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..items import Item, Persona
from .base import Backend, Chat, Steer

VALUES = ["family", "honesty", "novelty", "security", "care", "achievement", "autonomy", "tradition"]
K_NUIS = 24


def _h(s: str, seed: int, n: int) -> np.ndarray:
    d = hashlib.sha256(f"{seed}:{s}".encode()).digest()
    return np.random.default_rng(int.from_bytes(d[:8], "little")).standard_normal(n)


class ToyBackend(Backend):
    has_activations = True

    def __init__(self, name: str, personas: list[Persona], *, d_model: int = 64, n_layers: int = 8,
                 seed: int = 0, prompt_strength: float = 0.35, noise: float = 0.6,
                 item_noise: float = 0.8, ignores_persona: bool = False, collapse_scale: float = 20.0):
        self.name = name
        self.personas = {p.pid: p for p in personas}
        self.d_model, self.n_layers = d_model, n_layers
        rng = np.random.default_rng(seed)
        self.seed = seed
        k = len(VALUES) + K_NUIS
        self.W = [rng.standard_normal((d_model, k)) / np.sqrt(k) for _ in range(n_layers)]
        # 意味が一番よく載るのは中間層
        self.gain = np.exp(-((np.arange(1, n_layers + 1) - n_layers * 0.55) / (n_layers * 0.3)) ** 2)
        self.prior = rng.standard_normal(len(VALUES)) * 0.4
        self.lam = prompt_strength
        self.noise, self.item_noise = noise, item_noise
        self.collapse_scale = collapse_scale
        self.ignores_persona = ignores_persona  # 陰性対照：人物情報を一切使わないモデル

    # ------------------------------------------------------------ 内部
    def _sem(self, chat: Chat, with_rules: bool) -> np.ndarray:
        s = np.zeros(len(VALUES))
        pid = chat.meta.get("persona")
        if pid and not self.ignores_persona and chat.meta.get("text_condition", "base") != "base":
            p = self.personas[pid]
            it: Item | None = chat.meta.get("item")
            w = p.effective_weights(it.features) if (with_rules and it is not None) else p.weights
            s = np.array([w.get(v, 0.0) for v in VALUES]) / 5.0
        role = chat.meta.get("role")
        if role:
            s = s + _h(role, 7, len(VALUES)) * 0.6
        return s

    def _state(self, chat: Chat, layer: int) -> np.ndarray:
        s = self._sem(chat, with_rules=False) * self.gain[layer - 1]
        xi = _h(chat.system + "|" + chat.user, 11, K_NUIS)
        h = self.W[layer - 1] @ np.concatenate([s, xi])
        return h + _h(chat.user, 13 + layer + self.seed, self.d_model) * self.noise * 0.1

    def _decode(self, v: np.ndarray, layer: int) -> np.ndarray:
        Wl = self.W[layer - 1]
        return (np.linalg.pinv(Wl) @ v)[: len(VALUES)] * self.gain[layer - 1]

    # ------------------------------------------------------------ API
    def hidden(self, chats, layers):
        return {l: np.stack([self._state(c, l) for c in chats]) for l in layers}

    def choice_logprobs(self, chats, labels, steer: Steer | None = None):
        out = np.zeros((len(chats), len(labels)))
        for i, c in enumerate(chats):
            if "ipip" in c.meta:
                out[i] = self._likert(c, labels)
                continue
            it: Item = c.meta["item"]
            tA = np.array([v in it.tags_A for v in VALUES], float)
            tB = np.array([v in it.tags_B for v in VALUES], float)
            if c.meta.get("swapped"):
                tA, tB = tB, tA
            w = self.prior + self.lam * 5.0 * self._sem(c, with_rules=True)
            collapse = 1.0
            if steer is not None:
                d = self._decode(steer.vector * steer.alpha, steer.layer) * 5.0 * self.lam
                w = w + d
                # 強すぎるステアリングで判断が崩れる（実モデルの一貫性の崩壊を模す）
                collapse = 1.0 / (1.0 + (np.linalg.norm(d) / self.collapse_scale) ** 2)
            z = collapse * (w @ (tB - tA)) + _h(it.item_id, 17 + self.seed, 1)[0] * self.item_noise
            out[i] = [-np.logaddexp(0, z), -np.logaddexp(0, -z)]  # log σ(-z), log σ(z)
        return out

    def _likert(self, c: Chat, labels):
        pid = c.meta.get("persona")
        it = c.meta["ipip"]
        target = 3.0
        if pid and not self.ignores_persona and c.meta.get("text_condition", "base") != "base":
            t = self.personas[pid].big5.get(it["trait"], 3)
            target = t if it["key"] > 0 else 6 - t
        target += _h(it["id"], 19 + self.seed, 1)[0] * 0.5
        ks = np.arange(1, len(labels) + 1)
        return -0.8 * (ks - target) ** 2

    def text_nll(self, texts, steer=None):
        base = 2.0
        if steer is None:
            return base
        return base + 0.002 * float(np.linalg.norm(steer.vector * steer.alpha)) ** 2
