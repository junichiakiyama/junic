"""1次元 Dirac 量子ウォーク（研究ノート 第9章「運動＝分解と再構成」）。

格子間隔 a = 1、1 ステップ τ = 1 の単位で書く（c = a/τ = 1）。
状態は2成分スピノル ψ = (L, R)。1 ステップは
    コイン C(θ) = [[cosθ, -i sinθ], [-i sinθ, cosθ]]  （向きの組み替え＝「分解と再構成」）
    シフト       L は左へ 1 格子、R は右へ 1 格子       （光速 c で進む）
運動量表示では U(k) = diag(e^{ik}, e^{-ik}) C(θ) で、分散関係は
    cos ω = cos θ cos k
よって
    静止エネルギー   ω(0) = θ            （mc² = ħθ/τ）
    群速度の上限     v_max = c cos θ      （k = ±π/2 で到達）
    厳密な光円錐     |x| ≤ c t の外で振幅は厳密に 0
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- 解析式
def omega(k, theta):
    """正エネルギー分枝の準エネルギー ω(k) ∈ [0, π]。"""
    return np.arccos(np.clip(np.cos(theta) * np.cos(k), -1.0, 1.0))


def group_velocity(k, theta):
    """v_g = dω/dk = cosθ sin k / sin ω（c = 1 単位）。"""
    ct = np.cos(theta)
    s = np.sqrt(np.maximum(1.0 - (ct * np.cos(k)) ** 2, 1e-300))
    return ct * np.sin(k) / s


def v_max(theta):
    return np.cos(theta)


def theta_from_cos(cos_theta: float) -> float:
    return float(np.arccos(cos_theta))


# ---------------------------------------------------------------- 実空間の時間発展
def step(L: np.ndarray, R: np.ndarray, theta: float):
    c, s = np.cos(theta), np.sin(theta)
    L2 = c * L - 1j * s * R
    R2 = -1j * s * L + c * R
    return np.roll(L2, -1), np.roll(R2, 1)


def evolve(L, R, theta, n_steps, record_every=1):
    """時間発展し、確率密度 |L|²+|R|² の履歴を返す。"""
    hist = [np.abs(L) ** 2 + np.abs(R) ** 2]
    for t in range(1, n_steps + 1):
        L, R = step(L, R, theta)
        if t % record_every == 0:
            hist.append(np.abs(L) ** 2 + np.abs(R) ** 2)
    return L, R, np.array(hist)


# ---------------------------------------------------------------- 運動量表示の固有基底
def kgrid(N: int) -> np.ndarray:
    return 2 * np.pi * np.fft.fftfreq(N)


def eigvecs(k, theta):
    """U(k) の固有ベクトル（正・負エネルギー分枝）。第1成分を実数正に揃える。

    U(k) v = e^{∓iω} v。v = (sinθ, -i(cosθ - λ e^{-ik})) を正規化。
    """
    k = np.asarray(k, dtype=float)
    w = omega(k, theta)
    out = []
    for lam in (np.exp(-1j * w), np.exp(1j * w)):   # + 分枝, − 分枝
        v1 = np.full(k.shape, np.sin(theta), dtype=complex)
        v2 = -1j * (np.cos(theta) - lam * np.exp(-1j * k))
        nrm = np.sqrt(np.abs(v1) ** 2 + np.abs(v2) ** 2)
        out.append(np.stack([v1 / nrm, v2 / nrm]))
    return out[0], out[1]


def forward_k0(theta: float, N: int, k0: float = np.pi / 2) -> tuple[float, int]:
    """+分枝の群速度が正（右向き）になる k0 と、その格子上のインデックスずれ。"""
    shift = int(round(k0 / (2 * np.pi / N)))
    k0 = 2 * np.pi * shift / N
    # 数値微分で向きを判定（規約に依存しないように）
    dk = 1e-5
    vp = (omega(k0 + dk, theta) - omega(k0 - dk, theta)) / (2 * dk)
    # ω の定義は |ω| なので、+分枝の実際の伝播方向を固有ベクトルから確かめる
    vplus, _ = eigvecs(np.array([k0]), theta)
    # R 成分の重みが大きいほど右向き
    right = np.abs(vplus[1, 0]) ** 2 - np.abs(vplus[0, 0]) ** 2
    if right < 0:
        shift = -shift
        k0 = -k0
    return k0, shift


# ---------------------------------------------------------------- 波束
def gaussian_packet(N, x0, sigma, theta, k0=0.0, branch="+"):
    """指定分枝の固有状態で作ったガウス波束（位置 x0、幅 σ、中心運動量 k0）。"""
    x = np.arange(N)
    env = np.exp(-((x - x0) ** 2) / (4 * sigma ** 2)) * np.exp(1j * k0 * x)
    ek = np.fft.fft(env)
    k = kgrid(N)
    vp, vm = eigvecs(k, theta)
    v = vp if branch == "+" else vm
    Lk, Rk = ek * v[0], ek * v[1]
    L, R = np.fft.ifft(Lk), np.fft.ifft(Rk)
    n = np.sqrt(np.sum(np.abs(L) ** 2 + np.abs(R) ** 2))
    return L / n, R / n


def spinor_packet(N, x0, sigma, spinor=(1.0, 0.0), k0=0.0):
    """スピノルを固定した波束（固有状態ではない。Zitterbewegung 用）。"""
    x = np.arange(N)
    env = np.exp(-((x - x0) ** 2) / (4 * sigma ** 2)) * np.exp(1j * k0 * x)
    L, R = spinor[0] * env, spinor[1] * env
    n = np.sqrt(np.sum(np.abs(L) ** 2 + np.abs(R) ** 2))
    return L.astype(complex) / n, R.astype(complex) / n


# ---------------------------------------------------------------- 層間の符号化・復号
def encode(L, R, theta_src, theta_dst, shift, matched=True):
    """層の乗り換え：運動量を shift だけずらし、分枝を対応させて移す。

    matched=True : +分枝→+分枝、−分枝→−分枝（内部状態まで正しく書き換える）
    matched=False: スピノルはそのまま、位相 e^{ik0x} を掛けるだけ（素朴な符号化）
    """
    N = L.size
    if not matched:
        x = np.arange(N)
        ph = np.exp(1j * 2 * np.pi * shift * x / N)
        return L * ph, R * ph
    k = kgrid(N)
    Lk, Rk = np.fft.fft(L), np.fft.fft(R)
    sp, sm = eigvecs(k, theta_src)
    cp = np.conj(sp[0]) * Lk + np.conj(sp[1]) * Rk
    cm = np.conj(sm[0]) * Lk + np.conj(sm[1]) * Rk
    # 係数を運動量方向に shift だけ移動し、行き先の固有基底で組み直す
    cp, cm = np.roll(cp, shift), np.roll(cm, shift)
    dp, dm = eigvecs(k, theta_dst)
    Lk2 = cp * dp[0] + cm * dm[0]
    Rk2 = cp * dp[1] + cm * dm[1]
    return np.fft.ifft(Lk2), np.fft.ifft(Rk2)


def branch_weights(L, R, theta):
    """+分枝・−分枝の重み。"""
    k = kgrid(L.size)
    Lk, Rk = np.fft.fft(L), np.fft.fft(R)
    sp, sm = eigvecs(k, theta)
    cp = np.conj(sp[0]) * Lk + np.conj(sp[1]) * Rk
    cm = np.conj(sm[0]) * Lk + np.conj(sm[1]) * Rk
    tot = np.sum(np.abs(Lk) ** 2 + np.abs(Rk) ** 2)
    return np.sum(np.abs(cp) ** 2) / tot, np.sum(np.abs(cm) ** 2) / tot


# ---------------------------------------------------------------- 2層ワープ
@dataclass
class WarpParams:
    ratio: int = 49            # 上位層と基底層の光速比 c_n / c_3（1段あたりの比）
    cos_theta_up: float = 0.9  # 上位層での cosθ*（認識力）
    theta_base: float = 0.3    # 基底層（3次元）での θ（船の質量に対応）
    N: int | None = None       # 格子点数（None なら周期境界の回り込みが起きない大きさを自動で選ぶ）
    x0: int = 1000             # 出発点
    distance: int = 4000       # 目的地までの距離（格子）
    sigma: float = 30.0        # 波束の幅（格子）
    window: float = 200.0      # 復号窓の半幅（格子）
    matched: bool = True       # 内部状態を正しく書き換えるか
    record_every: int = 10


@dataclass
class WarpResult:
    params: WarpParams
    k0: float
    t_fine: np.ndarray         # 上位層の時間刻み（τ_n 単位）
    rho_up: np.ndarray         # 上位層の確率密度の履歴 [t, x]
    capture: np.ndarray        # 各時刻で復号したときの捕獲確率
    t_peak: int                # 捕獲確率が最大となる時刻（τ_n 単位）
    p_peak: float              # その捕獲確率
    forward_weight: float      # 符号化直後の +分枝（前進成分）の重み
    L3_after: np.ndarray       # 復号後の基底層の状態
    R3_after: np.ndarray


def run_warp(p: WarpParams, n_steps: int | None = None,
             decode_at: int | None = None) -> WarpResult:
    """基底層で静止した船 → 上位層へ符号化 → 上位層を伝播 → 目的地で復号。

    時間の単位は上位層の τ_n（= τ_3 / ratio）。上位層は毎ステップ、
    基底層は ratio ステップに1回だけ更新される（船は符号化中は基底層にいない）。
    """
    theta_up = theta_from_cos(p.cos_theta_up)
    if p.N is None:
        p.N = auto_size(p)
    k0, shift = forward_k0(theta_up, p.N)
    L3, R3 = gaussian_packet(p.N, p.x0, p.sigma, p.theta_base, 0.0, "+")
    Lu, Ru = encode(L3, R3, p.theta_base, theta_up, shift, matched=p.matched)
    fw, _ = branch_weights(Lu, Ru, theta_up)

    x = np.arange(p.N)
    xd = p.x0 + p.distance
    mask = (np.abs(x - xd) <= p.window).astype(float)
    v_expect = float(group_velocity(k0, theta_up))
    if n_steps is None:
        n_steps = int(1.35 * p.distance / max(v_expect, 1e-3)) + 50

    rho_hist, cap, ts = [], np.zeros(n_steps + 1), []
    L_dec = R_dec = None
    for t in range(n_steps + 1):
        rho = np.abs(Lu) ** 2 + np.abs(Ru) ** 2
        cap[t] = np.sum(rho * mask)
        if t % p.record_every == 0:
            rho_hist.append(rho.copy())
            ts.append(t)
        if decode_at is not None and t == decode_at:
            L_dec, R_dec = Lu * mask, Ru * mask
        Lu, Ru = step(Lu, Ru, theta_up)
    t_peak = int(np.argmax(cap))
    if L_dec is None:
        # 最適時刻で復号し直す（再計算を避けるため波束を再伝播）
        L_dec, R_dec = _state_at(p, theta_up, shift, t_peak, mask)
    L3a, R3a = encode(L_dec, R_dec, theta_up, p.theta_base, -shift, matched=p.matched)
    return WarpResult(p, k0, np.array(ts), np.array(rho_hist), cap, t_peak,
                      float(cap[t_peak]), float(fw), L3a, R3a)


def auto_size(p: WarpParams) -> int:
    """後退成分が周期境界を回り込んで復号窓に入らない格子の大きさ（2 の冪）。"""
    v = max(p.cos_theta_up, 1e-3)
    need = p.x0 + p.distance + 1.5 * p.distance / v + 4 * p.window + 8 * p.sigma
    return int(2 ** np.ceil(np.log2(need)))


def _state_at(p, theta_up, shift, t, mask):
    L3, R3 = gaussian_packet(p.N, p.x0, p.sigma, p.theta_base, 0.0, "+")
    Lu, Ru = encode(L3, R3, p.theta_base, theta_up, shift, matched=p.matched)
    for _ in range(t):
        Lu, Ru = step(Lu, Ru, theta_up)
    return Lu * mask, Ru * mask


def densities_between(p: WarpParams, t0: int, t1: int) -> np.ndarray:
    """上位層の確率密度を時刻 t0〜t1（τ_n 単位）だけ記録する。p.N は設定済みであること。"""
    theta_up = theta_from_cos(p.cos_theta_up)
    _, shift = forward_k0(theta_up, p.N)
    L3, R3 = gaussian_packet(p.N, p.x0, p.sigma, p.theta_base, 0.0, "+")
    Lu, Ru = encode(L3, R3, p.theta_base, theta_up, shift, matched=p.matched)
    out = []
    for t in range(t1 + 1):
        if t >= t0:
            out.append(np.abs(Lu) ** 2 + np.abs(Ru) ** 2)
        Lu, Ru = step(Lu, Ru, theta_up)
    return np.array(out)


def travel_time(distance, ratio, cos_theta, t_in=0.0, t_out=0.0):
    """T = t_in + d / (c_n cosθ*) + t_out（基底層の単位：c_3 = 1）。"""
    return t_in + distance / (ratio * cos_theta) + t_out
