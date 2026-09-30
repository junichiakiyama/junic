"""候補1：Dirac 量子ウォークによる「分解と再構成」と多層ワープの図と数値解析。

代表値で実行:
    python scripts/qw_figures.py
パラメータを変えて実行（出力は figures/custom/<tag>/ と results/custom/<tag>/）:
    python scripts/qw_figures.py --ratio 20 --cos-theta 0.7 --distance 3000 --tag r20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import PowerNorm
from scipy.special import jv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from warpsim import qwalk as q  # noqa: E402
from warpsim import style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------- Q1 時空図
def fig_q1_spacetime(out):
    thetas = [0.2, np.pi / 4, 1.2]
    N, T = 801, 300
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    for ax, th in zip(axes, thetas):
        L = np.zeros(N, complex)
        R = np.zeros(N, complex)
        L[N // 2] = 1 / np.sqrt(2)
        R[N // 2] = 1j / np.sqrt(2)
        _, _, hist = q.evolve(L, R, th, T)
        x = np.arange(N) - N // 2
        hn = hist / hist.max(axis=1, keepdims=True)     # 各時刻の最大値で規格化
        ax.pcolormesh(x, np.arange(T + 1), hn, cmap=S.SEQ_BLUE,
                      norm=PowerNorm(0.5, vmin=0, vmax=1),
                      shading="auto", rasterized=True)
        t = np.array([0, T])
        ax.plot(t, t, ls="--", lw=1, color=S.TEXT_2)
        ax.plot(-t, t, ls="--", lw=1, color=S.TEXT_2)
        vm = np.cos(th)
        ax.plot(vm * t, t, lw=1.2, color=S.SERIES[1])
        ax.plot(-vm * t, t, lw=1.2, color=S.SERIES[1])
        ax.set_xlim(-T - 20, T + 20)
        ax.set_ylim(0, T)
        ax.set_title(f"θ = {th:.2f}（cosθ = {vm:.2f}）")
        ax.set_xlabel("位置 x / a")
        ax.grid(False)
    axes[0].set_ylabel("時刻 t / τ")
    axes[0].text(-T + 5, T * 0.93, "破線：光円錐 |x| = ct", fontsize=7, color=S.TEXT_2)
    axes[0].text(-T + 5, T * 0.85, "実線：群速度の上限 c cosθ", fontsize=7, color=S.SERIES[1])
    axes[-1].text(T + 15, 5, "色：各時刻の最大値で\n規格化した確率密度", fontsize=7,
                  color=S.TEXT_2, ha="right", va="bottom")
    fig.suptitle("図Q1　Dirac 量子ウォークの時空図：再構成の規則が光円錐を決める", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    return S.save(fig, out, "Q1_spacetime")


# ---------------------------------------------------------------- Q2 分散関係と群速度
def measure_velocity(theta, k0, N=4096, steps=800, sigma=40):
    L, R = q.gaussian_packet(N, N // 4, sigma, theta, k0, "+")
    x = np.arange(N)
    m0 = np.sum(x * (np.abs(L) ** 2 + np.abs(R) ** 2))
    for _ in range(steps):
        L, R = q.step(L, R, theta)
    m1 = np.sum(x * (np.abs(L) ** 2 + np.abs(R) ** 2))
    return (m1 - m0) / steps


def fig_q2_dispersion(out, rows):
    thetas = [0.2, np.pi / 4, 1.2]
    k = np.linspace(-np.pi, np.pi, 721)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for i, th in enumerate(thetas):
        col = S.SERIES[i]
        a1.plot(k, q.omega(k, th), color=col, label=f"θ = {th:.2f}")
        a1.plot(k, -q.omega(k, th), color=col, lw=1, alpha=0.5)
        a2.plot(k, q.group_velocity(k, th), color=col, label=f"θ = {th:.2f}")
        ks = np.linspace(-np.pi * 0.9, np.pi * 0.9, 9)
        vs = [measure_velocity(th, kk) for kk in ks]
        a2.plot(ks, vs, "o", ms=5, color=col, mec=S.SURFACE, mew=1.2)
        for kk, vv in zip(ks, vs):
            rows.append({"theta": th, "k": kk, "v_numeric": vv,
                         "v_analytic": float(q.group_velocity(kk, th))})
        a2.axhline(np.cos(th), color=col, lw=0.8, ls=":")
    a1.set_xlabel("運動量 k a")
    a1.set_ylabel("準エネルギー ωτ")
    a1.set_title("(a) 分散関係 cos ω = cos θ cos k", loc="left")
    a1.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"])
    a1.annotate("ω(0) = θ（mc² = ħθ/τ）", xy=(0, np.pi / 4), xytext=(-1.2, -0.35),
                fontsize=7, color=S.TEXT_2,
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.6))
    a1.legend(loc="lower center", ncol=3, fontsize=7)
    a2.set_xlabel("運動量 k a")
    a2.set_ylabel("群速度 v / c")
    a2.set_title("(b) 群速度：線＝解析式、点＝波束の数値計算", loc="left")
    a2.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"])
    a2.text(0.97, 0.05, "点線：上限 $v_{\\max} = c\\cos\\theta$", transform=a2.transAxes,
            fontsize=7, color=S.TEXT_2, va="bottom", ha="right")
    fig.suptitle("図Q2　分散関係と群速度の上限 $v_{\\max} = c\\cos\\theta$", x=0.01, ha="left")
    fig.tight_layout()
    return S.save(fig, out, "Q2_dispersion")


# ---------------------------------------------------------------- Q3 厳密な光円錐
def fig_q3_cone(out, summary):
    T = 200
    N = 2 * T + 201
    L = np.zeros(N, complex)
    R = np.zeros(N, complex)
    L[N // 2] = 1 / np.sqrt(2)
    R[N // 2] = 1j / np.sqrt(2)
    th = 0.3
    for _ in range(T):
        L, R = q.step(L, R, th)
    P_qw = np.abs(L) ** 2 + np.abs(R) ** 2
    x = np.arange(N) - N // 2
    # 連続時間の格子ホッピング（最大群速度 2J = 1 に揃える）
    P_ct = jv(np.abs(x), float(T)) ** 2
    floor = -40
    # 量子ウォークは t と同じ偶奇の格子点だけを占める
    par = (x % 2) == (T % 2)
    lq = np.log10(np.maximum(P_qw, 1e-300))
    lc = np.log10(np.maximum(P_ct, 1e-300))
    outside = np.abs(x) > T
    summary["q3_max_prob_outside_cone_qw"] = float(P_qw[outside].max())
    summary["q3_prob_outside_cone_ctqw"] = float(P_ct[outside].sum())

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(x, np.clip(lc, floor, None), color=S.SERIES[1], lw=1.5,
            label="連続時間の格子ホッピング（最大群速度 c：光円錐の外にも裾が漏れる）")
    ins = (~outside) & par
    ax.plot(x[ins], np.clip(lq[ins], floor, None), color=S.SERIES[0], lw=1.2,
            label=f"量子ウォーク θ = {th}（離散的な再構成：外側は厳密に 0）")
    ax.plot(x[outside], np.full(outside.sum(), floor), color=S.SERIES[0], lw=2.5)
    for s in (-1, 1):
        ax.axvline(s * T, color=S.TEXT_2, lw=0.8, ls="--")
    ax.text(T + 5, -5, "光円錐\nx = ct", fontsize=7, color=S.TEXT_2)
    ax.text(T + 25, floor + 1.5, "厳密に 0（下端に表示）", fontsize=7, color=S.SERIES[0])
    ax.set_ylim(floor - 1, 1)
    ax.set_xlabel("位置 x / a")
    ax.set_ylabel("log₁₀ 確率")
    ax.set_title(f"図Q3　時刻 t = {T}τ の確率分布：光速は再構成規則の局所性から出る", loc="left")
    ax.legend(loc="center", fontsize=7)
    fig.tight_layout()
    return S.save(fig, out, "Q3_light_cone")


# ---------------------------------------------------------------- Q4 Zitterbewegung
def fig_q4_zitter(out, summary):
    N, T = 2048, 200
    thetas = [0.1, 0.2, 0.4]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(N)
    for i, th in enumerate(thetas):
        L, R = q.spinor_packet(N, N // 2, 60, (1.0, 0.0))
        xs = []
        for _ in range(T + 1):
            rho = np.abs(L) ** 2 + np.abs(R) ** 2
            xs.append(np.sum(x * rho) - N // 2)
            L, R = q.step(L, R, th)
        a1.plot(np.arange(T + 1), xs, color=S.SERIES[i], lw=1.5, label=f"θ = {th}")
    a1.set_xlabel("時刻 t / τ")
    a1.set_ylabel(r"平均位置 $\langle x\rangle$ / a")
    a1.set_title("(a) 初期スピノル (1, 0) の震え", loc="left")
    a1.legend(fontsize=7)
    # 周波数の測定
    th_scan = np.linspace(0.05, 1.2, 12)
    meas = []
    for th in th_scan:
        L, R = q.spinor_packet(N, N // 2, 60, (1.0, 0.0))
        xs = []
        for _ in range(1024):
            rho = np.abs(L) ** 2 + np.abs(R) ** 2
            xs.append(np.sum(x * rho))
            L, R = q.step(L, R, th)
        xs = np.array(xs) - np.polyval(np.polyfit(np.arange(1024), xs, 1), np.arange(1024))
        spec = np.abs(np.fft.rfft(xs * np.hanning(1024), 1 << 16))
        f = np.fft.rfftfreq(1 << 16) * 2 * np.pi
        meas.append(f[np.argmax(spec[1:]) + 1])
    th_line = np.linspace(0, 1.25, 100)
    a2.plot(th_line, 2 * q.omega(0, th_line), color=S.SERIES[0], label="解析：2ω(0) = 2θ")
    a2.plot(th_scan, meas, "o", color=S.SERIES[1], mec=S.SURFACE, mew=1.2, label=r"数値：$\langle x\rangle$ の振動数")
    a2.set_xlabel("θ")
    a2.set_ylabel("角振動数 × τ")
    a2.set_title("(b) 震えの振動数 ＝ 2mc²/ħ", loc="left")
    a2.legend(fontsize=7)
    summary["q4_zb_max_rel_err"] = float(np.max(np.abs(np.array(meas) - 2 * th_scan) / (2 * th_scan)))
    fig.suptitle("図Q4　質量＝向きの反転の頻度（Zitterbewegung、「光のランプの点滅」）", x=0.01, ha="left")
    fig.tight_layout()
    return S.save(fig, out, "Q4_zitterbewegung")


# ---------------------------------------------------------------- Q5 2層ワープの時空図
def fig_q5_warp(out, p: q.WarpParams, summary, label):
    res = q.run_warp(p)
    naive = q.run_warp(q.WarpParams(**{**p.__dict__, "matched": False, "N": p.N}))
    r = p.ratio
    x = np.arange(p.N) - p.x0
    t3 = res.t_fine / r                       # 基底層の時間 τ_3 単位
    xmax = p.distance * 1.25
    sel = (x > -0.35 * p.distance) & (x < xmax)
    T_formula = q.travel_time(p.distance, r, p.cos_theta_up)
    T_num = res.t_peak / r

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    a1.pcolormesh(x[sel], t3, res.rho_up[:, sel], cmap=S.SEQ_BLUE,
                  norm=PowerNorm(0.4, vmin=0, vmax=res.rho_up[5:, sel].max()),
                  shading="auto", rasterized=True)
    tt = np.array([0, t3[-1]])
    a1.plot(tt * 1.0, tt, color=S.SERIES[1], lw=1.2)             # 基底層の光円錐 c_3
    a1.plot(tt * r, tt, color=S.TEXT_2, lw=0.8, ls="--")         # 上位層の光円錐 c_n
    a1.plot(p.distance, T_num, "o", color=S.SERIES[2], ms=7, mec=S.SURFACE, mew=1.5)
    a1.plot(0, 0, "o", color=S.SERIES[2], ms=7, mec=S.SURFACE, mew=1.5)
    a1.annotate("符号化（出発）", (0, 0), xytext=(p.distance * 0.08, t3[-1] * 0.05),
                fontsize=7, color=S.TEXT)
    a1.annotate("復号（到着）", (p.distance, T_num), xytext=(p.distance * 1.03, T_num * 0.82),
                fontsize=7, color=S.TEXT)
    a1.text(t3[-1] * 1.6, t3[-1] * 0.80,
            f"基底層の光円錐 x = c₃t\n（光でも目的地まで {p.distance / 1.0:.0f} τ₃）",
            fontsize=7, color=S.SERIES[1])
    a1.text(p.distance * 0.52, t3[-1] * 0.2, f"上位層の光円錐\nx = cₙt = {r}c₃t",
            fontsize=7, color=S.TEXT_2)
    a1.set_xlim(x[sel][0], x[sel][-1])
    a1.set_ylim(0, t3[-1])
    a1.set_xlabel("位置 x / a（出発点 = 0）")
    a1.set_ylabel("時刻 t / τ₃（基底層の単位）")
    a1.set_title("(a) 上位層を伝わる船の確率密度", loc="left")
    a1.grid(False)

    a2.plot(np.arange(res.capture.size) / r, res.capture, color=S.SERIES[0], label="内部状態まで書き換える符号化")
    a2.plot(np.arange(naive.capture.size) / r, naive.capture, color=S.SERIES[1],
            label="位相だけの素朴な符号化")
    a2.axvline(T_formula, color=S.TEXT_2, lw=0.8, ls="--")
    a2.text(T_formula * 1.06, 0.72, f"T = d/(cₙcosθ*)\n= {T_formula:.1f} τ₃",
            fontsize=7, color=S.TEXT_2)
    a2.set_xlim(T_formula * 0.7, T_formula * 1.2)
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("復号する時刻 t / τ₃")
    a2.set_ylabel("目的地で復元される確率")
    a2.set_title("(b) 復号の時刻と復元確率", loc="left")
    a2.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.12))
    S.note(a1, f"r = cₙ/c₃ = {r}、cosθ* = {p.cos_theta_up}\n"
               f"d = {p.distance} a、{label}", "lower right")
    fig.suptitle("図Q5　2層ワープ：基底層で静止 → 上位層へ符号化 → 伝播 → 目的地で復号",
                 x=0.01, ha="left")
    fig.tight_layout()
    paths = S.save(fig, out, "Q5_two_layer_warp")

    light_time_base = p.distance / 1.0
    summary.update({
        "q5_ratio": r, "q5_cos_theta_up": p.cos_theta_up, "q5_distance_a": p.distance,
        "q5_lattice_N": p.N,
        "q5_T_formula_tau3": T_formula, "q5_T_numeric_tau3": T_num,
        "q5_rel_err": abs(T_num - T_formula) / T_formula,
        "q5_speed_over_c3_numeric": p.distance / T_num,
        "q5_speed_over_c3_formula": r * p.cos_theta_up,
        "q5_base_light_time_tau3": light_time_base,
        "q5_capture_matched": res.p_peak,
        "q5_capture_naive": naive.p_peak,
        "q5_forward_weight_naive": naive.forward_weight,
    })
    return paths


# ---------------------------------------------------------------- Q6 到着時刻と復元確率 vs 距離
def fig_q6_scaling(out, base: q.WarpParams, rows):
    cos_list = [0.5, 0.7, 0.9, 0.99]
    dists = [500, 1000, 2000, 4000, 8000]
    r = base.ratio
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    # (a) 到着時刻
    for i, ct in enumerate(cos_list):
        Tn = []
        for d in dists:
            p = q.WarpParams(ratio=r, cos_theta_up=ct, theta_base=base.theta_base,
                             distance=d, sigma=base.sigma, window=3 * base.sigma,
                             record_every=10 ** 9)
            res = q.run_warp(p)
            Tn.append(res.t_peak / r)
            rows.append({"panel": "a", "ratio": r, "cos_theta": ct, "sigma": base.sigma,
                         "distance_a": d, "T_numeric_tau3": res.t_peak / r,
                         "T_formula_tau3": q.travel_time(d, r, ct), "capture": res.p_peak})
        dd = np.linspace(0, dists[-1] * 1.05, 50)
        a1.plot(dd, q.travel_time(dd, r, ct), color=S.SERIES[i], lw=1.2)
        a1.plot(dists, Tn, "o", color=S.SERIES[i], mec=S.SURFACE, mew=1.2,
                label=f"cosθ* = {ct}")
    a1.set_xlabel("距離 d / a")
    a1.set_ylabel("到着時刻 T / τ₃")
    a1.set_title("(a) 到着時刻：線＝T = d/(cₙcosθ*)、点＝数値", loc="left")
    a1.legend(fontsize=7)
    S.note(a1, f"r = {r}、σ = {base.sigma:g} a", "lower right")
    # (b) 船の大きさ（波束の幅 σ）ごとの復元確率
    sigmas = [3, 6, 12]
    for i, sg in enumerate(sigmas):
        for ct, ls, mfc in ((base.cos_theta_up, "-", S.SERIES[i]), (0.5, ":", S.SURFACE)):
            Pn = []
            for d in dists:
                p = q.WarpParams(ratio=r, cos_theta_up=ct, theta_base=base.theta_base,
                                 distance=d, sigma=sg, window=3 * sg, record_every=10 ** 9)
                res = q.run_warp(p)
                Pn.append(res.p_peak)
                rows.append({"panel": "b", "ratio": r, "cos_theta": ct, "sigma": sg,
                             "distance_a": d, "T_numeric_tau3": res.t_peak / r,
                             "T_formula_tau3": q.travel_time(d, r, ct), "capture": res.p_peak})
            a2.plot(dists, Pn, marker="o", ls=ls, color=S.SERIES[i], mfc=mfc,
                    mec=S.SERIES[i], mew=1.2, lw=1.4, label=f"σ = {sg} a、cosθ* = {ct}")
    a2.set_xscale("log")
    a2.set_xticks(dists, [str(d) for d in dists])
    a2.minorticks_off()
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("距離 d / a")
    a2.set_ylabel("目的地で復元される確率")
    a2.set_title("(b) 局在した船ほど上位層で形が崩れる", loc="left")
    a2.legend(fontsize=6.5, loc="lower left", ncol=1)
    S.note(a2, "復号窓 ±3σ", "upper right")
    fig.suptitle("図Q6　到着時刻の距離依存と、上位層での波束の広がり（3次の分散）による復元確率の低下",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    return S.save(fig, out, "Q6_scaling")


# ---------------------------------------------------------------- Q7 目的地指定の精度
def fig_q7_precision(out, p: q.WarpParams, summary):
    pp = q.WarpParams(**{**p.__dict__, "record_every": 10 ** 9, "N": None})
    res = q.run_warp(pp)
    r = pp.ratio
    t_peak = res.t_peak
    dts = np.arange(-6 * pp.sigma, 6 * pp.sigma + 1, 2)
    dxs = np.arange(-6 * pp.sigma, 6 * pp.sigma + 1, 2)
    W = 2 * pp.sigma
    xd = pp.x0 + pp.distance
    rhos = q.densities_between(pp, int(t_peak + dts[0]), int(t_peak + dts[-1]))
    P = np.zeros((dts.size, dxs.size))
    for i, dt in enumerate(dts):
        rho = rhos[int(dt - dts[0])]
        c = np.concatenate([[0.0], np.cumsum(rho)])
        for j, dx in enumerate(dxs):
            lo = int(xd + dx - W)
            hi = int(xd + dx + W) + 1
            P[i, j] = c[hi] - c[lo]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    m = ax.pcolormesh(dxs / pp.sigma, dts / r, P, cmap=S.SEQ_BLUE, vmin=0, vmax=1,
                      shading="auto", rasterized=True)
    cs = ax.contour(dxs / pp.sigma, dts / r, P, levels=[0.5, 0.9, 0.99],
                    colors=[S.TEXT_2], linewidths=0.8)
    ax.clabel(cs, fontsize=7, fmt="%.2f")
    v = float(q.group_velocity(res.k0, q.theta_from_cos(pp.cos_theta_up))) * r   # a/τ₃
    xl = np.array([dxs[0], dxs[-1]]) / pp.sigma
    ax.plot(xl, xl * pp.sigma / v, color=S.SERIES[1], lw=1.2, ls="--")
    ax.text(0.03, 0.93, "破線：Δx = vΔt（誤差が打ち消し合う方向）", transform=ax.transAxes,
            fontsize=7, color=S.SERIES[1], va="top")
    cb = fig.colorbar(m, ax=ax, pad=0.02)
    cb.set_label("復元される確率")
    ax.set_xlabel("目的地の位置の誤差 Δx / σ")
    ax.set_ylabel("復号時刻の誤差 Δt / τ₃")
    ax.set_title("図Q7　目的地の指定の精度と復元確率", loc="left")
    ax.grid(False)
    S.note(ax, f"復号窓 ±2σ、cosθ* = {pp.cos_theta_up}\nr = {r}", "lower right")
    fig.tight_layout()
    # 位置の許容幅（P ≥ 0.9、Δt = 0）
    i0 = np.argmin(np.abs(dts))
    ok = dxs[P[i0] >= 0.9]
    summary["q7_dx_tolerance_sigma_P90"] = float((ok.max() - ok.min()) / 2 / pp.sigma) if ok.size else 0.0
    j0 = np.argmin(np.abs(dxs))
    okt = dts[P[:, j0] >= 0.9]
    summary["q7_dt_tolerance_tau3_P90"] = float((okt.max() - okt.min()) / 2 / r) if okt.size else 0.0
    return S.save(fig, out, "Q7_precision")


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratio", type=int, default=49, help="1段あたりの光速比 r = c_n/c_(n-1)（代表値 49）")
    ap.add_argument("--cos-theta", type=float, default=0.9, help="上位層での cosθ*（認識力、0〜1）")
    ap.add_argument("--theta-base", type=float, default=0.3, help="基底層での θ（船の質量に対応）")
    ap.add_argument("--distance", type=int, default=4000, help="目的地までの距離（格子数）")
    ap.add_argument("--sigma", type=float, default=30.0, help="波束の幅（格子数）")
    ap.add_argument("--window", type=float, default=200.0, help="復号窓の半幅（格子数）")
    ap.add_argument("--only", nargs="*", default=None, help="作る図を限定（例：Q5 Q7）")
    ap.add_argument("--tag", default=None, help="指定すると figures/custom/<tag>/ に出力")
    a = ap.parse_args(argv)

    S.setup()
    if a.tag:
        fig_dir = ROOT / "figures" / "custom" / a.tag
        res_dir = ROOT / "results" / "custom" / a.tag
        label = f"自由パラメータ（{a.tag}）"
    else:
        fig_dir = ROOT / "figures" / "representative"
        res_dir = ROOT / "results" / "representative"
        label = "代表値"
    res_dir.mkdir(parents=True, exist_ok=True)
    p = q.WarpParams(ratio=a.ratio, cos_theta_up=a.cos_theta, theta_base=a.theta_base,
                     distance=a.distance, sigma=a.sigma, window=a.window)
    want = set(a.only) if a.only else {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"}
    summary, vel_rows, scale_rows = {"label": label}, [], []
    if "Q1" in want:
        fig_q1_spacetime(fig_dir)
    if "Q2" in want:
        fig_q2_dispersion(fig_dir, vel_rows)
    if "Q3" in want:
        fig_q3_cone(fig_dir, summary)
    if "Q4" in want:
        fig_q4_zitter(fig_dir, summary)
    if "Q5" in want:
        fig_q5_warp(fig_dir, p, summary, label)
    if "Q6" in want:
        fig_q6_scaling(fig_dir, p, scale_rows)
    if "Q7" in want:
        fig_q7_precision(fig_dir, p, summary)

    sp = res_dir / "qw_summary.json"
    if sp.exists():                       # 一部の図だけ作り直したときは既存の値を残す
        summary = {**json.loads(sp.read_text()), **summary}
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for name, rows in (("qw_group_velocity.csv", vel_rows), ("qw_scaling.csv", scale_rows)):
        if rows:
            keys = list(rows[0])
            lines = [",".join(keys)] + [",".join(f"{r[k]:.10g}" if isinstance(r[k], float) else str(r[k]) for k in keys) for r in rows]
            (res_dir / name).write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("figures ->", fig_dir)


if __name__ == "__main__":
    main()
