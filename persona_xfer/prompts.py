"""条件ごとのプロンプト構成と、判断記録の検索。

条件（テキスト側）
  base       中立のシステムプロンプトのみ
  prompt     SelfModel（深層・中間・表層の自己記述）をシステムプロンプトに入れる
  retrieval  SelfModel ＋ 質問に近い判断記録 k 件（PAP の検索ベースラインに相当）
ステアリング系の条件はテキスト側を base（または retrieval）にして、活性化に加算する。
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .backends.base import Chat
from .items import DecisionRecord, Item, Persona

NEUTRAL_SYSTEM = "あなたは役に立つアシスタントです。質問に簡潔に答えてください。"

DECISION_Q = ("{text}\n\nA：{A}\nB：{B}\n\n"
              "あなたならどちらを選びますか。AかBの記号1文字だけで答えてください。")
DECISION_PREFIX = "答え："

LIKERT_Q = ("次の文は、あなたにどのくらい当てはまりますか。\n「{text}」\n\n"
            "1=まったく当てはまらない、2=あまり当てはまらない、3=どちらともいえない、"
            "4=やや当てはまる、5=よく当てはまる\n数字1文字だけで答えてください。")
LIKERT_PREFIX = "答え："


def persona_system(persona: Persona, records: list[DecisionRecord] | None = None) -> str:
    s = ("あなたは以下の人物です。この人物になりきり、この人物ならどう考え、どう選ぶかで答えてください。\n\n"
         "【自己モデル】\n" + persona.self_model_text())
    if records:
        s += "\n\n【過去の判断記録】\n" + "\n\n".join(r.text() for r in records)
    return s


# ---------------------------------------------------------------- 検索（文字 bigram TF-IDF）
def _grams(s: str) -> list[str]:
    s = re.sub(r"\s+", "", s)
    return [s[i:i + 2] for i in range(len(s) - 1)]


class Retriever:
    def __init__(self, records: list[DecisionRecord]):
        self.records = records
        docs = [Counter(_grams(r.item.text + r.item.A + r.item.B)) for r in records]
        df = Counter(g for d in docs for g in d)
        n = max(len(docs), 1)
        self.idf = {g: np.log((1 + n) / (1 + c)) + 1.0 for g, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts: Counter) -> dict[str, float]:
        v = {g: c * self.idf.get(g, 0.0) for g, c in counts.items()}
        norm = np.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def top_k(self, item: Item, k: int) -> list[DecisionRecord]:
        q = self._vec(Counter(_grams(item.text + item.A + item.B)))
        sims = [sum(q.get(g, 0.0) * x for g, x in v.items()) for v in self.vecs]
        order = np.argsort(sims)[::-1]
        # 同じ場面の記録は除く（学習側と評価側は場面単位で分けているので通常は起きない）
        out = [self.records[i] for i in order if self.records[i].item.scenario_id != item.scenario_id]
        return out[:k]


# ---------------------------------------------------------------- チャットの組み立て
def system_for(condition_text: str, persona: Persona | None, item: Item | None = None,
               retriever: Retriever | None = None, k: int = 4) -> str:
    if condition_text == "base" or persona is None:
        return NEUTRAL_SYSTEM
    if condition_text == "prompt":
        return persona_system(persona)
    if condition_text == "retrieval":
        recs = retriever.top_k(item, k) if (retriever is not None and item is not None) else []
        return persona_system(persona, recs)
    raise ValueError(condition_text)


def decision_chats(item: Item, system: str, meta: dict) -> tuple[Chat, Chat]:
    """選択肢の順番を入れ替えた2通り。位置バイアスを相殺するため両方を必ず測る。"""
    c1 = Chat(system, DECISION_Q.format(text=item.text, A=item.A, B=item.B), DECISION_PREFIX,
              {**meta, "item": item, "swapped": False})
    c2 = Chat(system, DECISION_Q.format(text=item.text, A=item.B, B=item.A), DECISION_PREFIX,
              {**meta, "item": item, "swapped": True})
    return c1, c2


def likert_chat(ipip_item: dict, system: str, meta: dict) -> Chat:
    return Chat(system, LIKERT_Q.format(text=ipip_item["text"]), LIKERT_PREFIX,
                {**meta, "ipip": ipip_item})
