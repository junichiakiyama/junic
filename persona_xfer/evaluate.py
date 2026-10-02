"""判断課題と Big Five 質問紙の採点。結果は1行1観測の dict のリストで返す。"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .backends.base import Backend, Steer
from .items import Item, Persona, truth
from .prompts import decision_chats, likert_chat


def _norm(lp: np.ndarray) -> np.ndarray:
    """ラベル間で正規化した確率。欠測（NaN）はそのまま NaN。"""
    m = np.nanmax(lp, axis=1, keepdims=True)
    e = np.exp(lp - m)
    return e / e.sum(1, keepdims=True)


def score_decisions(backend: Backend, items: list[Item], persona: Persona | None,
                    system_fn: Callable[[Item], str], steer: Steer | None, meta: dict,
                    truth_personas: list[Persona], train_ids: set[str], margin: float = 0.5
                    ) -> list[dict]:
    """各項目の P(B) を、選択肢の順番を入れ替えた2通りの平均で求める。

    persona=None（base 条件）のときは truth_personas の全員分の行を作る
    （同じ出力を各人物の正解と照合する。base は人物に依存しないため）。
    """
    chats = []
    for it in items:
        chats.extend(decision_chats(it, system_fn(it), meta))
    lp = backend.choice_logprobs(chats, ["A", "B"], steer)
    pr = _norm(lp)
    # ラベル（A/B）に乗った確率の合計。小さいモデルが回答形式に従っているかの診断に使う
    # （低いときは人格の失敗ではなく形式の失敗の可能性がある）。API モデルでは意味を持たない。
    mass = np.exp(lp).sum(1)
    rows = []
    targets = [persona] if persona is not None else truth_personas
    for i, it in enumerate(items):
        p1 = pr[2 * i, 1]       # 元の順番で B
        p2 = pr[2 * i + 1, 0]   # 入れ替えた順番で「A」= 元の B
        pB = float(np.nanmean([p1, p2])) if not (np.isnan(p1) and np.isnan(p2)) else np.nan
        for tp in targets:
            t = truth(tp, it, margin)
            if t.answer is None:
                continue
            p_correct = pB if t.answer == "B" else 1 - pB
            rows.append({
                "persona": tp.pid, "item_id": it.item_id, "scenario_id": it.scenario_id,
                "split": "train" if it.item_id in train_ids else "test",
                "truth": t.answer, "rule_dependent": t.rule_dependent,
                "pB": pB, "pB_order1": float(p1), "pB_order2": float(p2),
                "label_mass": float(np.nanmean(mass[2 * i:2 * i + 2])) if backend.has_activations else np.nan,
                "p_correct": p_correct, "correct": (np.nan if np.isnan(pB) else float(p_correct > 0.5)),
            })
    return rows


def score_big5(backend: Backend, ipip: list[dict], system: str, steer: Steer | None, meta: dict
               ) -> dict[str, float]:
    chats = [likert_chat(q, system, meta) for q in ipip]
    labels = ["1", "2", "3", "4", "5"]
    pr = _norm(backend.choice_logprobs(chats, labels, steer))
    ev = pr @ np.arange(1, 6)
    by = {}
    for q, e in zip(ipip, ev):
        by.setdefault(q["trait"], []).append(e if q["key"] > 0 else 6 - e)
    return {k: float(np.nanmean(v)) for k, v in by.items()}
