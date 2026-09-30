"""候補4：層ごとの速度上限モデル（49ⁿ 則など）の数値表と比較図。

代表値（r = 49, 基底層 3, 最上層 9, cosθ* = 1）で実行:
    python scripts/layer_figures.py
パラメータを変えて実行（出力は figures/custom/<tag>/ と results/custom/<tag>/）:
    python scripts/layer_figures.py --ratio 66.2 --n-max 9 --cos-theta 0.9 --t-in 60 --tag r66
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from warpsim import constants as K  # noqa: E402
from warpsim import layers as Lr  # noqa: E402
from warpsim import style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:   # Excel で文字化けしないよう BOM 付き
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- L1 層 × 目的地の所要時間
def fig_l1_table(out, res, a, label):
    layers = list(range(a.n_base, a.n_max + 1))
    names = [d[0] for d in K.DESTINATIONS]
    T = np.array([[Lr.travel_time_s(d, n, a.ratio, a.n_base, a.cos_theta, a.t_in, a.t_out)
                   for _, d in K.DESTINATIONS] for n in layers])
    rows = []
    for n, row in zip(layers, T):
        rec = {"層 n": n, "速度上限 c_n/c": f"{Lr.ceiling(n, a.ratio, a.n_base):.4g}"}
        for nm, s in zip(names, row):
            rec[f"{nm} [s]"] = f"{s:.4g}"
        rows.append(rec)
    write_csv(res / "layer_travel_times.csv", rows)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    norm = LogNorm(vmin=max(T.min(), 1e-12), vmax=T.max())
    ax.imshow(np.log10(T), cmap=S.SEQ_BLUE, aspect="auto",
              vmin=np.log10(norm.vmin), vmax=np.log10(norm.vmax))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            frac = (np.log10(T[i, j]) - np.log10(norm.vmin)) / (np.log10(norm.vmax) - np.log10(norm.vmin))
            ax.text(j, i, Lr.format_duration(T[i, j]), ha="center", va="center", fontsize=7,
                    color=S.SURFACE if frac > 0.55 else S.TEXT)
    wrap = {"プロキシマ・ケンタウリ": "プロキシマ・\nケンタウリ", "アンドロメダ銀河": "アンドロメダ\n銀河",
            "おとめ座銀河団": "おとめ座\n銀河団", "観測可能な宇宙の半径": "観測可能な\n宇宙の半径"}
    ax.set_xticks(range(len(names)), [wrap.get(n, n) for n in names], fontsize=7)
    ax.set_yticks(range(len(layers)),
                  [f"{n}次元　{Lr.fmt_num(float(Lr.ceiling(n, a.ratio, a.n_base)))} c" for n in layers])
    ax.set_ylabel("経由する層と速度上限 cₙ")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"図L1　経由する層ごとの所要時間（r = {a.ratio:g}、cosθ* = {a.cos_theta:g}"
                 f"{'、t_in+t_out = %g 秒' % (a.t_in + a.t_out) if a.t_in + a.t_out else ''}）",
                 loc="left")
    fig.text(0.01, 0.01, f"モデルの仮定：cₙ = r^(n−{a.n_base}) c、T = t_in + d/(cₙ cosθ*) + t_out（{label}）",
             fontsize=7, color=S.TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return S.save(fig, out, "L1_travel_time_table")


# ---------------------------------------------------------------- L2 教学事例と層の上限
def case_speeds():
    out = []
    for lab, d, tmin, tmax, src in K.TEACHING_CASES:
        out.append((lab, Lr.required_speed(d, tmax), Lr.required_speed(d, tmin), src))
    return out


def fig_l2_cases(out, res, a, label, summary):
    cases = case_speeds()
    layers = list(range(a.n_base + 1, a.n_max + 1))
    rows = []
    for lab, vlo, vhi, src in cases:
        nlo = Lr.required_layer(vlo, a.ratio, a.n_base, a.cos_theta)
        nhi = Lr.required_layer(vhi, a.ratio, a.n_base, a.cos_theta)
        top = Lr.ceiling(a.n_max, a.ratio, a.n_base) * a.cos_theta
        rows.append({"事例": lab, "典拠": src, "必要な速さ 下限 [c]": f"{vlo:.3g}",
                     "必要な速さ 上限 [c]": f"{vhi:.3g}",
                     "必要な層（実数）": f"{nlo:.2f}–{nhi:.2f}",
                     "必要な層（整数）": f"{math.ceil(nlo - 1e-9)}–{math.ceil(nhi - 1e-9)}",
                     f"{a.n_max}次元の上限との比": f"{vhi / top:.3g}"})
    write_csv(res / "teaching_cases.csv", rows)
    summary["cases"] = rows

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ys = np.arange(len(cases))[::-1]
    for y, (lab, vlo, vhi, _) in zip(ys, cases):
        if vhi / vlo > 1.01:
            ax.plot([vlo, vhi], [y, y], color=S.SERIES[0], lw=6, solid_capstyle="round")
        ax.plot([vlo, vhi], [y, y], "o", color=S.SERIES[0], ms=7, mec=S.SURFACE, mew=1.5)
        ax.text(vhi * 2.2, y, f"{Lr.fmt_num(vhi, 2)} c" if vhi / vlo < 1.01
                else f"{Lr.fmt_num(vlo, 2)}–{Lr.fmt_num(vhi, 2)} c",
                va="center", fontsize=7, color=S.TEXT)
    # 観測からの下限（P5）
    yq = -1
    ax.annotate("", xy=(K.QUANTUM_CORRELATION_LOWER_BOUND * 30, yq),
                xytext=(K.QUANTUM_CORRELATION_LOWER_BOUND, yq),
                arrowprops=dict(arrowstyle="->", color=S.SERIES[1], lw=1.5))
    ax.plot(K.QUANTUM_CORRELATION_LOWER_BOUND, yq, "o", color=S.SERIES[1], ms=7,
            mec=S.SURFACE, mew=1.5)
    ax.text(K.QUANTUM_CORRELATION_LOWER_BOUND * 40, yq, "量子相関の速さの下限（実験、P5）",
            va="center", fontsize=7, color=S.TEXT)
    ax.set_yticks(list(ys) + [yq], [c[0] for c in cases] + ["量子相関 ≥ 10⁴ c"])
    for n in layers:
        v = Lr.ceiling(n, a.ratio, a.n_base) * a.cos_theta
        ax.axvline(v, color=S.MUTED, lw=0.8, ls="--")
        ax.text(v, len(cases) - 0.35, f"{n}次元", ha="center", fontsize=7, color=S.TEXT_2)
    top = Lr.ceiling(a.n_max, a.ratio, a.n_base) * a.cos_theta
    ax.axvspan(top, 1e13, color=S.SERIES[7], alpha=0.07, lw=0)
    ax.text(top * 1.5, -1.6, f"{a.n_max}次元の上限を超える", fontsize=7, color=S.SERIES[7])
    ax.set_xscale("log")
    ax.set_xlim(1, 1e13)
    ax.set_ylim(-1.9, len(cases) - 0.1)
    ax.set_xlabel("必要な速さ（光速 c に対する比）")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"図L2　教学事例が要求する速さと各層の上限（破線、r = {a.ratio:g}、cosθ* = {a.cos_theta:g}）",
                 loc="left", fontsize=9.5)
    fig.text(0.01, 0.01, f"事例の速さ＝距離／所要時間（第9章）。層の上限はモデルの仮定（{label}）",
             fontsize=7, color=S.TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return S.save(fig, out, "L2_cases_vs_layers")


# ---------------------------------------------------------------- L3 エネルギー比較
def fig_l3_energy(out, res):
    cats = ["既存物理", "経路C", "霊界科学モデル", "下限"]
    col = {c: S.SERIES[i] for i, c in enumerate(cats)}
    rows = [{"手法": e[0], "最小 [J]": e[1], "最大 [J]": e[2], "分類": e[3], "出典": e[4]}
            for e in K.ENERGIES]
    write_csv(res / "energy_comparison.csv", rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ys = np.arange(len(K.ENERGIES))[::-1] - 1.2
    for y, (lab, lo, hi, cat, src) in zip(ys, K.ENERGIES):
        if hi / lo > 1.01:
            ax.plot([lo, hi], [y, y], color=col[cat], lw=6, solid_capstyle="round")
        ax.plot([lo, hi], [y, y], "o", color=col[cat], ms=7, mec=S.SURFACE, mew=1.5)
        txt = Lr.fmt_num(lo, 2) if hi / lo < 1.01 else f"{Lr.fmt_num(lo, 2)}–{Lr.fmt_num(hi, 2)}"
        ax.text(hi * 30, y, txt + " J", va="center", fontsize=7, color=S.TEXT)
    for i, (lab, e) in enumerate(K.ENERGY_REFERENCES):
        ax.axvline(e, color=S.MUTED, lw=0.8, ls=":")
        ax.text(e, len(K.ENERGIES) - 0.55 - i * 0.42, " " + lab, ha="left", va="center",
                fontsize=6.5, color=S.TEXT_2)
    ax.set_yticks(ys, [e[0] for e in K.ENERGIES], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1e84)
    ax.set_ylim(-1.9, len(K.ENERGIES) - 0.2)
    ax.set_xticks([10.0 ** p for p in range(0, 81, 10)])
    ax.minorticks_off()
    ax.set_xlabel("必要なエネルギー [J]（プロキシマまで、70 kg）")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=col[c], lw=6, label=c) for c in cats]
    ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)
    ax.set_title("図L3　ワープ手法の必要エネルギー（研究ノートの評価値を一枚に）", loc="left")
    fig.text(0.01, 0.01, "値は第2章・第9章・8.13–8.14節の評価。点線は目盛りとしての参照値",
             fontsize=7, color=S.TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return S.save(fig, out, "L3_energy_comparison")


# ---------------------------------------------------------------- L4 比 r の掃引
def fig_l4_sweep(out, res, a, summary):
    cases = case_speeds()
    r = np.linspace(10, 200, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    rows = []
    for i, (lab, vlo, vhi, _) in enumerate(cases):
        n_req = a.n_base + np.log(vhi / a.cos_theta) / np.log(r)
        a1.plot(r, n_req, color=S.SERIES[i], lw=1.8, label=lab.split("（")[0])
        rmin = Lr.min_ratio_for(vhi, a.n_max, a.n_base, a.cos_theta)
        rows.append({"事例": lab, f"{a.n_max}次元までに収まる最小の r": f"{rmin:.4g}"})
    q_n = a.n_base + np.log(K.QUANTUM_CORRELATION_LOWER_BOUND) / np.log(r)
    a1.plot(r, q_n, color=S.SERIES[4], lw=1.2, ls="--", label="量子相関の下限 10⁴ c")
    a1.axhline(a.n_max, color=S.TEXT_2, lw=0.8)
    a1.text(198, a.n_max - 0.1, f"最上層 {a.n_max}次元", ha="right", va="top", fontsize=7,
            color=S.TEXT_2)
    a1.axvline(49, color=S.MUTED, lw=0.8, ls=":")
    a1.text(50, 3.2, "r = 49", fontsize=7, color=S.TEXT_2)
    a1.set_xlabel("1段あたりの光速比 r")
    a1.set_ylabel("必要な層 n（実数値）")
    a1.set_ylim(3, 12)
    a1.set_title("(a) 必要な層と r（事例は速さの上限側）", loc="left")
    a1.legend(fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.36))

    # (b) すべての事例を最上層までに収める最小の r（cosθ* と n_max を変える）
    v_top = max(c[2] for c in cases)
    cth = np.linspace(0.3, 1.0, 200)
    for i, nm in enumerate([9, 10, 11]):
        rmin = (v_top / cth) ** (1.0 / (nm - a.n_base))
        a2.plot(cth, rmin, color=S.SERIES[i], lw=1.8, label=f"最上層 {nm}次元")
    a2.axhline(49, color=S.MUTED, lw=0.8, ls=":")
    a2.text(0.31, 50.5, "r = 49", fontsize=7, color=S.TEXT_2)
    a2.set_xlabel("認識力 cosθ*")
    a2.set_ylabel("必要な最小の r")
    a2.set_title(f"(b) 最速の事例（{Lr.fmt_num(v_top, 2)} c）を収める最小の r", loc="left")
    a2.legend(fontsize=7)
    fig.suptitle("図L4　1段あたりの比 r を自由パラメータとしたときの整合性", x=0.01, ha="left")
    fig.tight_layout()
    write_csv(res / "ratio_sweep.csv", rows)
    summary["min_ratio"] = rows
    summary["min_ratio_all_cases_n9_cos1"] = (v_top) ** (1 / 6)
    return S.save(fig, out, "L4_ratio_sweep")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratio", type=float, default=49.0, help="1段あたりの光速比 r（代表値 49）")
    ap.add_argument("--n-base", type=int, default=3, help="基底層の次元（代表値 3）")
    ap.add_argument("--n-max", type=int, default=9, help="最上層の次元（代表値 9）")
    ap.add_argument("--cos-theta", type=float, default=1.0, help="認識力 cosθ*（代表値 1＝上限いっぱい）")
    ap.add_argument("--t-in", type=float, default=0.0, help="符号化にかかる時間 [s]")
    ap.add_argument("--t-out", type=float, default=0.0, help="復号にかかる時間 [s]")
    ap.add_argument("--tag", default=None, help="指定すると figures/custom/<tag>/ に出力")
    a = ap.parse_args(argv)

    S.setup()
    if a.tag:
        fig_dir = ROOT / "figures" / "custom" / a.tag
        res_dir = ROOT / "results" / "custom" / a.tag
        label = f"自由パラメータ：{a.tag}"
    else:
        fig_dir = ROOT / "figures" / "representative"
        res_dir = ROOT / "results" / "representative"
        label = "代表値"
    summary = {"params": vars(a)}
    fig_l1_table(fig_dir, res_dir, a, label)
    fig_l2_cases(fig_dir, res_dir, a, label, summary)
    fig_l3_energy(fig_dir, res_dir)
    fig_l4_sweep(fig_dir, res_dir, a, summary)
    summary["gw170817_lead_years_if_gravity_used_layer4"] = Lr.gw_lead_time_years(
        a.n_base + 1, a.ratio, a.n_base)
    summary["min_layer_for_quantum_correlation"] = Lr.required_layer(
        K.QUANTUM_CORRELATION_LOWER_BOUND, a.ratio, a.n_base)
    (res_dir / "layer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("figures ->", fig_dir)


if __name__ == "__main__":
    main()
