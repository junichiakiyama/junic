"""結果の集計・検定・図・報告書（report.md）。主要な判定はすべて評価用（test）項目で行う。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import items as I
from .stats import cluster_bootstrap, holm, noninferior, paired_test

DEFAULT_CRITERIA = {
    "min_gain_vs_base": 0.10,     # 判断一致率の base からの上積み（CI 下限 > 0 も要求）
    "min_identification": 0.80,   # 20項目での人物識別率
    "min_big5_r": 0.60,           # Big Five プロフィールと目標の相関
    "max_ppl_ratio": 1.10,        # ステアリングによる中立文パープレキシティの悪化
    "min_cross_model_diag_adv": 0.20,  # 異なるモデル間で同じ人物どうしの相関の優位
    "noninferiority_delta": 0.05,
}

COND_ORDER = ["base", "prompt", "retrieval", "steer_random", "steer_native", "steer_transfer",
              "retrieval+steer_native"]
COND_JA = {"base": "なし（base）", "prompt": "自己記述プロンプト", "retrieval": "自己記述＋判断記録検索",
           "steer_random": "ランダム方向（対照）", "steer_native": "人物ベクトル（再抽出）",
           "steer_transfer": "人物ベクトル（写像移植）", "retrieval+steer_native": "検索＋人物ベクトル"}


def group(c: str) -> str:
    return "steer_transfer" if c.startswith("steer_transfer:") else c


def _read(out: Path, prefix: str) -> pd.DataFrame:
    fs = sorted(out.glob(f"{prefix}__*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


class Analysis:
    def __init__(self, out_dir: str | Path, criteria: dict | None = None, B: int = 2000):
        self.out = Path(out_dir)
        cfg_path = self.out / "config.json"
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.crit = {**DEFAULT_CRITERIA, **self.cfg.get("criteria", {}), **(criteria or {})}
        self.B = B
        dec = pd.concat([_read(self.out, "decisions"), _read(self.out, "decisions_steer")], ignore_index=True)
        dec["group"] = dec["condition"].map(group)
        self.dec = dec
        self.test = dec[dec["split"] == "test"].copy()
        self.big5 = pd.concat([_read(self.out, "big5"), _read(self.out, "big5_steer")], ignore_index=True)
        self.tuning = _read(self.out, "tuning")
        self.nll = _read(self.out, "nll")
        self.diag = _read(self.out, "transfer_diag")
        self.personas = I.load_personas()
        if self.cfg.get("personas"):
            self.personas = [p for p in self.personas if p.pid in self.cfg["personas"]]
        self.items = {it.item_id: it for it in I.load_items()}
        self.models = list(dict.fromkeys(dec["model"]))
        self.tables: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------ 1. 判断一致率
    def accuracy(self) -> pd.DataFrame:
        rows = []
        for (m, c), g in self.test.groupby(["model", "condition"], sort=False):
            g = g.dropna(subset=["correct"])
            est, lo, hi = cluster_bootstrap(g, lambda x: x["correct"].mean(), B=self.B)
            rows.append({"model": m, "condition": c, "group": group(c), "acc": est, "lo": lo, "hi": hi,
                         "p_correct": g["p_correct"].mean(), "n": len(g),
                         "label_mass": g["label_mass"].mean() if "label_mass" in g else np.nan,
                         "layer": g["layer"].iloc[0] if len(g) else np.nan,
                         "alpha": g["alpha"].iloc[0] if len(g) else np.nan})
        t = pd.DataFrame(rows)
        t["order"] = t["group"].map({c: i for i, c in enumerate(COND_ORDER)})
        t = t.sort_values(["model", "order"]).drop(columns="order")
        self.tables["accuracy"] = t
        return t

    def accuracy_by_rule(self) -> pd.DataFrame:
        rows = []
        for (m, c, rd), g in self.test.groupby(["model", "condition", "rule_dependent"], sort=False):
            g = g.dropna(subset=["correct"])
            est, lo, hi = cluster_bootstrap(g, lambda x: x["correct"].mean(), B=self.B)
            rows.append({"model": m, "condition": c, "group": group(c), "rule_dependent": bool(rd),
                         "acc": est, "lo": lo, "hi": hi, "n": len(g)})
        t = pd.DataFrame(rows)
        self.tables["accuracy_by_rule"] = t
        return t

    # ------------------------------------------------------------ 2. 事前登録した仮説
    def _paired(self, m: str, a: str, b: str, value: str = "correct") -> pd.DataFrame:
        x = self.test[(self.test.model == m) & (self.test.condition == a)]
        y = self.test[(self.test.model == m) & (self.test.condition == b)]
        k = ["persona", "item_id", "scenario_id", "rule_dependent"]
        j = x[k + [value]].merge(y[k + [value]], on=k, suffixes=("_a", "_b"))
        j["d"] = j[f"{value}_a"] - j[f"{value}_b"]
        return j

    def hypotheses(self) -> pd.DataFrame:
        rows = []
        conds = lambda m: set(self.test[self.test.model == m].condition)  # noqa: E731
        for m in self.models:
            cs = conds(m)
            for c in ("prompt", "retrieval"):
                if c in cs and "base" in cs:
                    rows.append({"id": "H1", "model": m, "test": f"{c} > base", "kind": "superiority",
                                 **paired_test(self._paired(m, c, "base"))})
            for c in sorted(x for x in cs if x.startswith("steer_transfer:")):
                if "steer_random" in cs:
                    rows.append({"id": "H2a", "model": m, "test": f"{c} > steer_random", "kind": "superiority",
                                 **paired_test(self._paired(m, c, "steer_random"))})
                if "steer_native" in cs:
                    r = paired_test(self._paired(m, c, "steer_native"), alternative="two-sided")
                    rows.append({"id": "H2b", "model": m, "test": f"{c} ≥ steer_native − δ",
                                 "kind": "noninferiority",
                                 "noninferior": noninferior(r, self.crit["noninferiority_delta"]), **r})
            if "retrieval+steer_native" in cs and "retrieval" in cs:
                rows.append({"id": "H3a", "model": m, "test": "retrieval+steer_native > retrieval",
                             "kind": "superiority",
                             **paired_test(self._paired(m, "retrieval+steer_native", "retrieval"))})
            if "steer_native" in cs and "retrieval" in cs:
                rows.append({"id": "H3b", "model": m, "test": "steer_native vs retrieval", "kind": "two-sided",
                             **paired_test(self._paired(m, "steer_native", "retrieval"), alternative="two-sided")})
                rows.append({"id": "H4", "model": m, "kind": "interaction",
                             "test": "(retrieval − steer_native) is larger on rule-dependent items",
                             **self._interaction(m, "retrieval", "steer_native")})
        t = pd.DataFrame(rows)
        if len(t):
            conf = t["kind"].isin(["superiority", "interaction"])
            t["p_holm"] = np.nan
            t.loc[conf, "p_holm"] = holm(t.loc[conf, "p"].tolist())
        self.tables["hypotheses"] = t
        return t

    def _interaction(self, m: str, a: str, b: str) -> dict:
        j = self._paired(m, a, b).dropna(subset=["d"])

        def stat(x):
            rd, ri = x[x.rule_dependent == True]["d"], x[x.rule_dependent == False]["d"]  # noqa: E712
            return rd.mean() - ri.mean() if len(rd) and len(ri) else np.nan

        if j.rule_dependent.nunique() < 2:
            return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": len(j)}
        est, lo, hi = cluster_bootstrap(j, stat, B=self.B)
        # 片側 p：ブートストラップ分布を推定値の位置から 0 へずらして近似
        rng = np.random.default_rng(3)
        codes, uniq = pd.factorize(j["scenario_id"])
        gl = [np.flatnonzero(codes == k) for k in range(len(uniq))]
        bs = [stat(j.iloc[np.concatenate([gl[k] for k in rng.integers(0, len(gl), len(gl))])])
              for _ in range(self.B)]
        bs = np.asarray(bs)
        p = (1 + np.sum(bs - est >= est)) / (len(bs) + 1)
        return {"diff": est, "lo": lo, "hi": hi, "p": float(p), "n": len(j)}

    # ------------------------------------------------------------ 3. モデルをまたいだ人物の同一性
    def _delta(self) -> pd.DataFrame:
        """同じモデルの base からの P(B) の変化。人物ごとの「人格による判断の動き」。"""
        d = self.test.drop_duplicates(["model", "condition", "persona", "item_id"])
        base = d[d.condition == "base"].drop_duplicates(["model", "item_id"])[["model", "item_id", "pB"]]
        x = d[d.condition != "base"].merge(base, on=["model", "item_id"], suffixes=("", "_base"))
        x["delta"] = x["pB"] - x["pB_base"]
        return x

    def identity(self) -> tuple[pd.DataFrame, dict]:
        x = self._delta()
        pids = [p.pid for p in self.personas]
        items = sorted(x.item_id.unique())
        scen = np.array([self.items[i].scenario_id for i in items])
        out_rows, mats = [], {}
        for g, gx in x.groupby("group", sort=False):
            ms = sorted(gx.model.unique())
            if len(ms) < 2:
                continue
            arr = np.full((len(ms), len(pids), len(items)), np.nan)
            piv = gx.pivot_table(index=["model", "persona"], columns="item_id", values="delta")
            for a, m in enumerate(ms):
                for b, p in enumerate(pids):
                    if (m, p) in piv.index:
                        arr[a, b] = piv.loc[(m, p)].reindex(items).to_numpy()

            def mat(ix):
                M = np.zeros((len(pids), len(pids)))
                cnt = 0
                for a in range(len(ms)):
                    for b in range(len(ms)):
                        if a == b:
                            continue
                        A, Bm = arr[a][:, ix], arr[b][:, ix]
                        for i in range(len(pids)):
                            for k in range(len(pids)):
                                ok = ~np.isnan(A[i]) & ~np.isnan(Bm[k])
                                M[i, k] += (np.corrcoef(A[i][ok], Bm[k][ok])[0, 1]
                                            if ok.sum() > 2 and A[i][ok].std() > 0 and Bm[k][ok].std() > 0 else 0)
                        cnt += 1
                return M / cnt

            def adv(M):
                return np.mean(np.diag(M)) - (M.sum() - np.trace(M)) / (M.size - len(M))

            full = np.arange(len(items))
            M = mat(full)
            uniq = np.unique(scen)
            rng = np.random.default_rng(5)
            bs = []
            for _ in range(min(self.B, 500)):
                pick = rng.choice(uniq, len(uniq))
                ix = np.concatenate([np.flatnonzero(scen == s) for s in pick])
                bs.append(adv(mat(ix)))
            lo, hi = np.nanquantile(bs, [0.025, 0.975])
            mats[g] = pd.DataFrame(M, index=pids, columns=pids)
            out_rows.append({"group": g, "models": ",".join(ms), "diag_mean": float(np.mean(np.diag(M))),
                             "diag_adv": adv(M), "lo": lo, "hi": hi})
        t = pd.DataFrame(out_rows)
        self.tables["identity"] = t
        self.identity_mats = mats
        return t, mats

    def identification(self, budgets=(5, 10, 20), draws: int = 500) -> pd.DataFrame:
        """n 項目の回答から、4人物のうち誰の人格かを最尤で当てられる割合（チャンス = 1/人物数）。"""
        pids = [p.pid for p in self.personas]
        tr = {}
        for it in self.items.values():
            tr[it.item_id] = {p.pid: I.truth(p, it, self.cfg.get("margin", 0.5)).answer for p in self.personas}
        rows = []
        d = self.test.drop_duplicates(["model", "condition", "persona", "item_id"])
        rng = np.random.default_rng(7)
        for (m, c, p), g in d.groupby(["model", "condition", "persona"], sort=False):
            if c == "base":
                continue
            g = g.dropna(subset=["pB"])
            pB = np.clip(g["pB"].to_numpy(), 1e-3, 1 - 1e-3)
            ids = g["item_id"].to_numpy()
            ll = np.full((len(pids), len(ids)), np.nan)
            for k, q in enumerate(pids):
                for j, i in enumerate(ids):
                    a = tr[i][q]
                    if a is not None:
                        ll[k, j] = np.log(pB[j] if a == "B" else 1 - pB[j])
            for n in budgets:
                if n > len(ids):
                    continue
                hit = 0
                for _ in range(draws):
                    s = rng.choice(len(ids), n, replace=False)
                    hit += int(np.nanargmax(np.nanmean(ll[:, s], 1)) == pids.index(p))
                rows.append({"model": m, "condition": c, "group": group(c), "persona": p, "n_items": n,
                             "ident_acc": hit / draws})
        t = pd.DataFrame(rows)
        self.tables["identification"] = t
        return t

    # ------------------------------------------------------------ 4. Big Five・信頼性・能力保持
    def big5_fit(self) -> pd.DataFrame:
        if self.big5.empty:
            return pd.DataFrame()
        tgt = {p.pid: p.big5 for p in self.personas}
        traits = ["O", "C", "E", "A", "N"]
        rows = []
        for _, r in self.big5.iterrows():
            x = np.array([r[t] for t in traits], float)
            y = np.array([tgt[r["persona"]][t] for t in traits], float)
            rr = np.corrcoef(x, y)[0, 1] if x.std() > 0 else np.nan
            rows.append({"model": r["model"], "condition": r["condition"], "group": group(r["condition"]),
                         "persona": r["persona"], "profile_r": rr, "mae": float(np.mean(np.abs(x - y))),
                         **{t: r[t] for t in traits}})
        t = pd.DataFrame(rows)
        self.tables["big5"] = t
        return t

    def steering_reliability(self) -> pd.DataFrame:
        x = self._delta()
        x = x[x.group.str.contains("steer")]
        if x.empty:
            return pd.DataFrame()
        x["toward"] = np.where(x["truth"] == "B", 1, -1) * x["delta"]
        t = x.groupby(["model", "condition"], sort=False).agg(
            mean_shift=("toward", "mean"), frac_toward=("toward", lambda s: float((s > 0).mean())),
            frac_away=("toward", lambda s: float((s < -0.05).mean())), sd_shift=("toward", "std"),
            n=("toward", "size")).reset_index()
        self.tables["steering_reliability"] = t
        return t

    def criteria(self) -> pd.DataFrame:
        acc = self.tables.get("accuracy", self.accuracy())
        ident = self.tables.get("identification", self.identification())
        b5 = self.tables.get("big5", self.big5_fit())
        idt = self.tables.get("identity")
        rows = []
        for _, r in acc.iterrows():
            m, c = r["model"], r["condition"]
            if c == "base":
                continue
            j = self._paired(m, c, "base")
            gain = paired_test(j) if len(j) else {"diff": np.nan, "lo": np.nan}
            ia = ident[(ident.model == m) & (ident.condition == c) & (ident.n_items == 20)]["ident_acc"].mean() \
                if len(ident) else np.nan
            br = b5[(b5.model == m) & (b5.condition == c)]["profile_r"].mean() if len(b5) else np.nan
            pr = self.nll[(self.nll.model == m) & (self.nll.condition == c)]["ppl_ratio"].mean() \
                if len(self.nll) else np.nan
            cm = idt[idt.group == group(c)]["diag_adv"].mean() if idt is not None and len(idt) else np.nan
            row = {"model": m, "condition": c, "gain": gain["diff"], "gain_lo": gain["lo"],
                   "ident@20": ia, "big5_r": br, "ppl_ratio": pr, "cross_model_adv": cm}
            row["pass_gain"] = bool(gain["diff"] >= self.crit["min_gain_vs_base"] and gain["lo"] > 0)
            row["pass_ident"] = bool(ia >= self.crit["min_identification"]) if not np.isnan(ia) else None
            row["pass_big5"] = bool(br >= self.crit["min_big5_r"]) if not np.isnan(br) else None
            row["pass_ppl"] = bool(pr <= self.crit["max_ppl_ratio"]) if not np.isnan(pr) else None
            row["pass_cross"] = bool(cm >= self.crit["min_cross_model_diag_adv"]) if not np.isnan(cm) else None
            checks = [v for k, v in row.items() if k.startswith("pass_") and v is not None]
            row["preserved"] = bool(all(checks)) if checks else None
            rows.append(row)
        t = pd.DataFrame(rows)
        self.tables["criteria"] = t
        return t

    # ------------------------------------------------------------ 実行と出力
    def run(self, figures: bool = True) -> Path:
        self.accuracy()
        self.accuracy_by_rule()
        self.hypotheses()
        self.identity()
        self.identification()
        self.big5_fit()
        self.steering_reliability()
        self.criteria()
        tdir = self.out / "tables"
        tdir.mkdir(exist_ok=True)
        for k, t in self.tables.items():
            t.to_csv(tdir / f"{k}.csv", index=False, encoding="utf-8-sig")
        for g, M in getattr(self, "identity_mats", {}).items():
            M.to_csv(tdir / f"identity_matrix__{g}.csv", encoding="utf-8-sig")
        if not self.diag.empty:
            self.diag.groupby(["target", "source", "layer_target"]).agg(
                cos=("cos_transfer_native", "mean"), cos_random=("cos_random_native", "mean"),
                map_r2=("map_r2", "mean")).reset_index().to_csv(tdir / "transfer_diag.csv", index=False)
        if figures:
            from .figures import make_all
            make_all(self)
        from .report import write_report
        return write_report(self)
