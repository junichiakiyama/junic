"""統計：場面（scenario）単位のクラスタ・ブートストラップ、符号反転の並べ替え検定、Holm 補正、
非劣性の判定。

同じ場面の文脈違いの項目と、同じ項目に対する4人物の行は互いに独立でない。
そのため、再標本化と並べ替えの単位はすべて場面にする。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _cluster_index(df: pd.DataFrame, cluster: str) -> tuple[np.ndarray, list[np.ndarray]]:
    codes, uniq = pd.factorize(df[cluster])
    groups = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    return codes, groups


def cluster_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], cluster: str = "scenario_id",
                      B: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    est = float(stat(df))
    if len(df) == 0:
        return np.nan, np.nan, np.nan
    _, groups = _cluster_index(df, cluster)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        vals.append(stat(df.iloc[idx]))
    a = (1 - level) / 2
    lo, hi = np.nanquantile(vals, [a, 1 - a])
    return est, float(lo), float(hi)


def paired_test(diff: pd.DataFrame, value: str = "d", cluster: str = "scenario_id", B: int = 5000,
                seed: int = 0, alternative: str = "greater") -> dict:
    """diff[value] は同じ（人物, 項目）での条件差。平均差の推定・CI・クラスタ符号反転 p 値。"""
    d = diff.dropna(subset=[value])
    if len(d) == 0:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "lo90": np.nan, "p": np.nan, "n": 0}
    est, lo, hi = cluster_bootstrap(d, lambda x: x[value].mean(), cluster, B=2000, seed=seed)
    _, lo90, hi90 = cluster_bootstrap(d, lambda x: x[value].mean(), cluster, B=2000, seed=seed, level=0.90)
    sums = d.groupby(cluster)[value].sum().to_numpy()
    n = len(d)
    obs = sums.sum() / n
    rng = np.random.default_rng(seed + 1)
    signs = rng.choice([-1.0, 1.0], size=(B, len(sums)))
    null = (signs * sums).sum(1) / n
    if alternative == "greater":
        p = (1 + (null >= obs).sum()) / (B + 1)
    elif alternative == "less":
        p = (1 + (null <= obs).sum()) / (B + 1)
    else:
        p = (1 + (np.abs(null) >= abs(obs)).sum()) / (B + 1)
    return {"diff": est, "lo": lo, "hi": hi, "lo90": lo90, "hi90": hi90, "p": float(p), "n": n,
            "n_clusters": len(sums)}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * p[i]))
        out[i] = running
    return out.tolist()


def noninferior(res: dict, delta: float) -> bool:
    """片側 5% の非劣性：差の 90% CI の下限が −δ を上回る。"""
    return bool(res["lo90"] > -delta)
