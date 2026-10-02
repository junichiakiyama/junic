"""項目バンクと人物定義の読み込み、真の回答の計算、学習/評価の分割。

真の回答はモデルに依存しない「人格の定義」から決定的に計算する。
  U(選択肢) = Σ_v w_eff[v]·[v ∈ tags(選択肢)]
  w_eff = 基本の重み + 文脈（features）に合致する規則の加算
効用差が margin 未満の項目はその人物にとって曖昧として評価から除く。

項目ごとに「規則依存（rule_dependent）」かどうかも記録する：
基本の重みだけで決めた答えと、規則込みの答えが食い違う項目。
固定ベクトルの加算では表しにくい条件付き判断（報告書 5 章の論点）を測るために使う。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Item:
    item_id: str          # 例 d01_0
    scenario_id: str      # 例 d01（分割と クラスタ・ブートストラップの単位）
    text: str             # 文脈 + 場面
    A: str
    B: str
    tags_A: tuple[str, ...]
    tags_B: tuple[str, ...]
    features: dict
    n_variants: int


@dataclass
class Persona:
    pid: str
    name: str
    weights: dict[str, float]
    rules: list[dict]
    big5: dict[str, int]
    self_model: dict[str, str]
    meta: dict = field(default_factory=dict)

    def effective_weights(self, features: dict) -> dict[str, float]:
        w = dict(self.weights)
        for r in self.rules:
            if all(features.get(k) == v for k, v in r["if"].items()):
                for k, dv in r["add"].items():
                    w[k] = w.get(k, 0.0) + dv
        return w

    def self_model_text(self, layers=("core", "rules", "surface")) -> str:
        return "\n".join(self.self_model[k] for k in layers if k in self.self_model)


@dataclass(frozen=True)
class Truth:
    answer: str | None       # "A" / "B" / None（曖昧）
    margin: float            # U_B - U_A
    base_answer: str | None  # 規則なしの答え
    rule_dependent: bool
    reason_value: str | None  # 判断に最も効いた価値（判断記録の理由文に使う）


def load_values(path: Path | None = None) -> dict[str, str]:
    return json.loads((path or DATA / "personas.json").read_text(encoding="utf-8"))["values"]


def load_personas(path: Path | None = None) -> list[Persona]:
    raw = json.loads((path or DATA / "personas.json").read_text(encoding="utf-8"))
    return [Persona(pid=p["id"], name=p["name"], weights=p["weights"], rules=p.get("rules", []),
                    big5=p.get("big5", {}), self_model=p["self_model"],
                    meta={k: v for k, v in p.items() if k not in
                          ("id", "name", "weights", "rules", "big5", "self_model")})
            for p in raw["personas"]]


def load_items(path: Path | None = None) -> list[Item]:
    raw = json.loads((path or DATA / "dilemmas.json").read_text(encoding="utf-8"))
    items = []
    for s in raw["scenarios"]:
        n = len(s["variants"])
        for j, v in enumerate(s["variants"]):
            text = (v["context"] + s["scenario"]) if v["context"] else s["scenario"]
            items.append(Item(item_id=f"{s['id']}_{j}", scenario_id=s["id"], text=text,
                              A=s["A"], B=s["B"], tags_A=tuple(s["tags"]["A"]),
                              tags_B=tuple(s["tags"]["B"]), features=v["features"], n_variants=n))
    return items


def _utility(w: dict[str, float], tags) -> float:
    return sum(w.get(t, 0.0) for t in tags)


def truth(persona: Persona, item: Item, margin: float = 0.5) -> Truth:
    w = persona.effective_weights(item.features)
    d = _utility(w, item.tags_B) - _utility(w, item.tags_A)
    d0 = _utility(persona.weights, item.tags_B) - _utility(persona.weights, item.tags_A)
    ans = None if abs(d) < margin else ("B" if d > 0 else "A")
    base = None if abs(d0) < margin else ("B" if d0 > 0 else "A")
    reason = None
    if ans is not None:
        tags = item.tags_B if ans == "B" else item.tags_A
        reason = max(tags, key=lambda t: w.get(t, 0.0))
    return Truth(answer=ans, margin=d, base_answer=base,
                 rule_dependent=(ans is not None and ans != base), reason_value=reason)


def split_scenarios(items: list[Item], train_frac: float = 0.4, seed: int = 0
                    ) -> tuple[list[Item], list[Item]]:
    """場面（scenario）単位で分割する。文脈違いの対が両側に割れないようにするため。
    対をなす場面と単独の場面を別々に層化して、同じ比率で学習側に入れる。"""
    rng = random.Random(seed)
    sids_pair = sorted({it.scenario_id for it in items if it.n_variants > 1})
    sids_single = sorted({it.scenario_id for it in items if it.n_variants == 1})
    train = set()
    for group in (sids_pair, sids_single):
        g = group[:]
        rng.shuffle(g)
        train.update(g[: round(len(g) * train_frac)])
    tr = [it for it in items if it.scenario_id in train]
    te = [it for it in items if it.scenario_id not in train]
    return tr, te


# 判断記録（DecisionRecord）の理由文。PAP の DecisionRecord に相当する。
REASONS = {
    "family": "家族のことを考えると、こちらを選ぶのが自然だからです。",
    "honesty": "ごまかしたり黙っていたりするのは、自分の中で筋が通らないからです。",
    "novelty": "やったことのないことに挑む方が、自分にとって意味があるからです。",
    "security": "ここで危ない橋を渡るべきではないと判断したからです。",
    "care": "困っている相手のことを放っておけないからです。",
    "achievement": "結果を出すことが、今の自分にとって一番大事だからです。",
    "autonomy": "自分のことは自分で決めたいからです。",
    "tradition": "これまでの付き合いやしきたりを大切にしたいからです。",
}


@dataclass(frozen=True)
class DecisionRecord:
    item: Item
    choice: str
    reason: str

    def text(self) -> str:
        chosen = self.item.A if self.choice == "A" else self.item.B
        return f"状況：{self.item.text}\n選んだ行動：{chosen}\n理由：{self.reason}"


def decision_records(persona: Persona, train_items: list[Item], margin: float = 0.5
                     ) -> list[DecisionRecord]:
    recs = []
    for it in train_items:
        t = truth(persona, it, margin)
        if t.answer is not None:
            recs.append(DecisionRecord(it, t.answer, REASONS[t.reason_value]))
    return recs


def load_ipip(path: Path | None = None) -> list[dict]:
    return json.loads((path or DATA / "ipip_ja.json").read_text(encoding="utf-8"))["items"]


def load_questions(path: Path | None = None) -> dict:
    return json.loads((path or DATA / "questions.json").read_text(encoding="utf-8"))
