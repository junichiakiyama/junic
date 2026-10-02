"""報告書の図。色は warpsim.style の固定順（モデルの識別）と、青↔赤の発散色（相関）。"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from warpsim import style as S

from .analysis import COND_JA, COND_ORDER

DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
MARKERS = ["o", "s", "D", "^", "v", "P"]


def _model_colors(models):
    return {m: S.SERIES[i % len(S.SERIES)] for i, m in enumerate(models)}


def fig_accuracy(an):
    import matplotlib.pyplot as plt
    t = an.tables["accuracy"]
    groups = [g for g in COND_ORDER if g in set(t.group)]
    models = an.models
    col = _model_colors(models)
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(groups) + 1.4))
    off = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else [0]
    for k, m in enumerate(models):
        for y, g in enumerate(groups):
            r = t[(t.model == m) & (t.group == g)]
            for _, rr in r.iterrows():
                ax.plot([rr.lo, rr.hi], [y + off[k]] * 2, color=col[m], lw=2, solid_capstyle="round")
                ax.plot(rr.acc, y + off[k], MARKERS[k % len(MARKERS)], color=col[m], ms=6,
                        mec=S.SURFACE, mew=1, label=m if y == groups.index(r.group.iloc[0]) else None)
    ax.axvline(0.5, color=S.MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(groups)), [COND_JA.get(g, g) for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("評価項目での判断一致率（95% CI、場面単位ブートストラップ）")
    ax.set_title("条件別の判断一致率")
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.0), title="モデル")
    S.note(ax, "破線 = チャンス（0.5）", "lower left")
    return S.save(fig, an.out / "figures", "F1_accuracy")


def fig_rule(an):
    import matplotlib.pyplot as plt
    t = an.tables["accuracy_by_rule"]
    groups = [g for g in COND_ORDER if g in set(t.group) and g != "base"]
    models = an.models
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 0.42 * len(groups) + 1.4),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for k, (rd, lab, c) in enumerate([(False, "規則に依存しない項目", S.SERIES[0]),
                                          (True, "規則依存（条件付き）項目", S.SERIES[1])]):
            for y, g in enumerate(groups):
                r = t[(t.model == m) & (t.group == g) & (t.rule_dependent == rd)]
                for _, rr in r.iterrows():
                    yy = y + (-0.15 if k == 0 else 0.15)
                    ax.plot([rr.lo, rr.hi], [yy] * 2, color=c, lw=2, solid_capstyle="round")
                    ax.plot(rr.acc, yy, "o" if k == 0 else "s", color=c, ms=6, mec=S.SURFACE, mew=1,
                            label=lab if y == 0 else None)
        ax.axvline(0.5, color=S.MUTED, lw=1, ls="--")
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(m)
        ax.set_yticks(range(len(groups)), [COND_JA.get(g, g) for g in groups])
    axes[0][0].invert_yaxis()
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.supxlabel("判断一致率（95% CI）", fontsize=9, y=0.02)
    return S.save(fig, an.out / "figures", "F2_rule_dependent")


def fig_identity(an):
    import matplotlib.pyplot as plt
    mats = getattr(an, "identity_mats", {})
    if not mats:
        return []
    gs = [g for g in COND_ORDER if g in mats]
    fig, axes = plt.subplots(1, len(gs), figsize=(2.4 * len(gs) + 1.4, 3.0), squeeze=False,
                             layout="constrained")
    for n, (ax, g) in enumerate(zip(axes[0], gs)):
        M = mats[g]
        im = ax.imshow(M.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(M)), M.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(M)), M.index if n == 0 else [""] * len(M))
        ax.grid(False)
        for i in range(len(M)):
            for j in range(len(M)):
                v = M.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=S.SURFACE if abs(v) > 0.6 else S.TEXT)
        ax.set_title(COND_JA.get(g, g), fontsize=9)
    axes[0][0].set_ylabel("モデル1 の人物")
    fig.supxlabel("モデル2 の人物（判断の動き Δ の相関、異なるモデル対の平均）", fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    return S.save(fig, an.out / "figures", "F3_cross_model_identity")


def fig_tuning(an):
    import matplotlib.pyplot as plt
    t = an.tuning
    if t.empty:
        return []
    keys = list(dict.fromkeys(zip(t.model, t.condition)))
    n = len(keys)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.5 * rows), sharey=True, squeeze=False,
                             layout="constrained")
    first_of_model = set()
    for ax, (m, c) in zip(axes.flat, keys):
        g = t[(t.model == m) & (t.condition == c)]
        for k, (l, gl) in enumerate(g.groupby("layer")):
            ax.plot(gl.alpha, gl.train_p_correct, marker="o", color=S.SERIES[k % len(S.SERIES)],
                    label=f"層 {l}（{gl.layer_rel.iloc[0]:.2f}）")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{m}\n{c}", fontsize=8)
        if m not in first_of_model:  # 層の番号はモデルごとに違うので、モデルごとに凡例を出す
            ax.legend(fontsize=6)
            first_of_model.add(m)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.supxlabel("α（ステアリングの強さ）", fontsize=9)
    fig.supylabel("学習用項目の平均 P(正解)", fontsize=9)
    return S.save(fig, an.out / "figures", "F4_tuning")


def make_all(an):
    S.setup()
    paths = []
    for f in (fig_accuracy, fig_rule, fig_identity, fig_tuning):
        paths += f(an) or []
    return paths
