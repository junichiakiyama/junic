"""report.md の書き出し。数値はすべて tables/ の CSV と同じもの。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import COND_JA


def _f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "○" if x else "×"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, (float, np.floating)):
        return f"{x:.{d}f}"
    return str(x)


def _md(t: pd.DataFrame, cols: list[str], heads: list[str] | None = None) -> str:
    if t is None or t.empty:
        return "（データなし）\n"
    heads = heads or cols
    out = "| " + " | ".join(heads) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in t.iterrows():
        out += "| " + " | ".join(_f(r.get(c)) for c in cols) + " |\n"
    return out


def _edge_warnings(an) -> str:
    """選ばれた層や α が格子の端なら、格子が狭すぎる可能性がある。"""
    if an.tuning.empty:
        return ""
    alphas = sorted(an.cfg.get("alphas", an.tuning.alpha.unique()))
    out = []
    for (m, c), g in an.tuning.groupby(["model", "condition"], sort=False):
        best = g.loc[g.train_p_correct.idxmax()]
        layers = sorted(g.layer.unique())
        if best.alpha in (alphas[0], alphas[-1]):
            out.append(f"- {m} / {c}: α={best.alpha:g} が格子の端（{alphas[0]:g}〜{alphas[-1]:g}）")
        if best.layer in (layers[0], layers[-1]) and len(layers) > 2:
            out.append(f"- {m} / {c}: 層 {int(best.layer)} が候補の端")
    if not out:
        return ""
    return ("\n> **調整の警告：** 選ばれた値が探索格子の端にある。格子を広げて再実行するまで、"
            "その条件の効果は過小評価の可能性がある。\n\n" + "\n".join(out) + "\n")


def write_report(an) -> Path:
    T = an.tables
    acc = T["accuracy"].copy()
    acc["条件"] = acc["condition"].map(lambda c: COND_JA.get(c.split(":")[0], c) + (
        f"（元: {c.split(':')[1]}）" if c.startswith("steer_transfer:") else ""))
    acc["95% CI"] = acc.apply(lambda r: f"[{r.lo:.3f}, {r.hi:.3f}]", axis=1)
    toy = any(m.get("backend") == "toy" for m in an.cfg.get("models", []))
    s = f"# 疑似人格のモデル間保存：結果報告（{an.cfg.get('name', an.out.name)}）\n\n"
    if toy:
        s += ("> **注意：この結果には合成モデル（toy）が含まれる。** 合成モデルは解析パイプラインの"
              "較正用で、効果は構成上与えたものである。科学的な知見として解釈してはならない。\n\n")
    s += (f"- モデル: {', '.join(an.models)}\n- 人物: {', '.join(p.pid for p in an.personas)}\n"
          f"- 評価項目数（test）: {an.test.item_id.nunique()}（場面 {an.test.scenario_id.nunique()}）\n"
          f"- 判定基準: `{an.crit}`\n\n")
    s += "## 1. 判断一致率（評価項目）\n\n![](figures/F1_accuracy.png)\n\n"
    s += _md(acc, ["model", "条件", "acc", "95% CI", "p_correct", "label_mass", "n", "layer", "alpha"],
             ["モデル", "条件", "一致率", "95% CI", "平均P(正解)", "ラベル確率", "n", "層", "α"])
    s += ("\n「ラベル確率」は次のトークンが A か B である確率の合計（回答形式に従っているかの診断）。"
          "0.5 を大きく下回る条件は、人格ではなく形式で失敗している可能性があるため解釈に注意。\n")
    s += "\n## 2. 事前登録した仮説の検定\n\n"
    s += ("H1: テキスト条件は base より一致率が高い／H2a: 写像移植ベクトル > ランダム方向／"
          "H2b: 写像移植は再抽出に対して非劣性（δ）／H3a: 検索＋人物ベクトル > 検索のみ／"
          "H3b: 人物ベクトル vs 検索（両側・探索的）／H4: 規則依存項目でステアリングの不利が大きい。"
          "p は場面単位の符号反転並べ替え（H4 はブートストラップ）、p_holm は確認的検定全体での Holm 補正。\n\n")
    h = T["hypotheses"]
    s += _md(h, ["id", "model", "test", "diff", "lo", "hi", "p", "p_holm", "noninferior", "n"],
             ["仮説", "モデル", "比較", "差", "CI下", "CI上", "p", "p_holm", "非劣性", "n"])
    s += "\n## 3. 条件付き判断（規則依存項目）\n\n![](figures/F2_rule_dependent.png)\n\n"
    s += _md(T["accuracy_by_rule"], ["model", "condition", "rule_dependent", "acc", "lo", "hi", "n"],
             ["モデル", "条件", "規則依存", "一致率", "CI下", "CI上", "n"])
    s += ("\n## 4. モデルをまたいだ人物の同一性\n\n"
          "各モデルで「人物を与えたことによる判断の動き」Δ = P(B|条件) − P(B|base) を求め、"
          "異なるモデルの間で人物どうしの相関をとった。対角（同じ人物）が非対角より高いほど、"
          "人格がモデルをまたいで同じ形で保たれている。\n\n![](figures/F3_cross_model_identity.png)\n\n")
    s += _md(T["identity"], ["group", "models", "diag_mean", "diag_adv", "lo", "hi"],
             ["条件", "モデル", "対角平均", "対角の優位", "CI下", "CI上"])
    idn = T["identification"]
    if len(idn):
        piv = idn.groupby(["model", "condition", "n_items"]).ident_acc.mean().unstack("n_items").reset_index()
        piv.columns = [str(c) for c in piv.columns]
        s += f"\n### 人物識別率（{len(an.personas)}人から最尤で当てる。チャンス = {1 / len(an.personas):.2f}）\n\n"
        s += _md(piv, list(piv.columns), ["モデル", "条件"] + [f"{c}項目" for c in piv.columns[2:]])
    s += "\n## 5. Big Five プロフィール\n\n"
    b5 = T.get("big5")
    if b5 is not None and len(b5):
        g = b5.groupby(["model", "condition"]).agg(profile_r=("profile_r", "mean"), mae=("mae", "mean")).reset_index()
        s += _md(g, ["model", "condition", "profile_r", "mae"], ["モデル", "条件", "目標との相関(平均)", "平均絶対誤差"])
    s += "\n## 6. ステアリングの信頼性と能力保持\n\n"
    s += _md(T.get("steering_reliability"), ["model", "condition", "mean_shift", "frac_toward", "frac_away", "sd_shift"],
             ["モデル", "条件", "正解方向への平均移動", "正解方向に動いた割合", "逆方向に動いた割合", "移動のSD"])
    if len(an.nll):
        s += "\n" + _md(an.nll, ["model", "condition", "nll_base", "nll", "ppl_ratio"],
                        ["モデル", "条件", "NLL(base)", "NLL", "パープレキシティ比"])
    if len(an.diag):
        d = an.diag.groupby(["target", "source"]).agg(cos=("cos_transfer_native", "mean"),
                                                      cos_rand=("cos_random_native", "mean"),
                                                      r2=("map_r2", "mean")).reset_index()
        s += "\n### 写像の診断（移植ベクトルと再抽出ベクトルのコサイン類似度、全層・全人物の平均）\n\n"
        s += _md(d, ["target", "source", "cos", "cos_rand", "r2"], ["移植先", "元", "cos", "cos(ランダム)", "写像CV R²"])
    s += "\n### 調整曲線\n\n![](figures/F4_tuning.png)\n"
    s += _edge_warnings(an)
    s += "\n## 7. 判定（事前登録の基準）\n\n"
    s += _md(T["criteria"], ["model", "condition", "gain", "gain_lo", "ident@20", "big5_r", "ppl_ratio",
                             "cross_model_adv", "preserved"],
             ["モデル", "条件", "上積み", "上積みCI下", "識別@20", "Big5 r", "PPL比", "モデル間優位", "保存"])
    s += ("\n「保存」は、その条件で測れた基準（上積み・識別・Big Five・PPL・モデル間優位）をすべて満たした場合のみ ○。"
          "テキストのみの API モデルでは PPL は測れないので、その項目は判定から除く。\n")
    p = an.out / "report.md"
    p.write_text(s, encoding="utf-8")
    return p
