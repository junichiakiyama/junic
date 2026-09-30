"""量子ウォークと層モデルの検算（研究ノートに載せた数値の再現を含む）。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warpsim import constants as K  # noqa: E402
from warpsim import layers as Lr  # noqa: E402
from warpsim import qwalk as q  # noqa: E402


# ---------------------------------------------------------------- 量子ウォーク
def test_unitarity():
    L, R = q.gaussian_packet(512, 256, 10, 0.5, 0.3)
    for _ in range(300):
        L, R = q.step(L, R, 0.5)
    assert np.sum(np.abs(L) ** 2 + np.abs(R) ** 2) == pytest.approx(1.0, abs=1e-12)


def test_eigvecs():
    k = q.kgrid(64)
    for th in (0.2, 0.9):
        U = np.array([[np.exp(1j * k) * np.cos(th), -1j * np.exp(1j * k) * np.sin(th)],
                      [-1j * np.exp(-1j * k) * np.sin(th), np.exp(-1j * k) * np.cos(th)]])
        vp, vm = q.eigvecs(k, th)
        w = q.omega(k, th)
        assert np.allclose(np.einsum("ijk,jk->ik", U, vp), np.exp(-1j * w) * vp)
        assert np.allclose(np.einsum("ijk,jk->ik", U, vm), np.exp(1j * w) * vm)


def test_strict_light_cone():
    N, T = 401, 150
    L = np.zeros(N, complex)
    R = np.zeros(N, complex)
    L[N // 2] = 1.0
    for _ in range(T):
        L, R = q.step(L, R, 0.7)
    x = np.arange(N) - N // 2
    assert np.all((np.abs(L) ** 2 + np.abs(R) ** 2)[np.abs(x) > T] == 0.0)


@pytest.mark.parametrize("th", [0.3, 0.8, 1.2])
def test_vmax_is_c_cos_theta(th):
    k0, _ = q.forward_k0(th, 4096)
    L, R = q.gaussian_packet(4096, 1000, 40, th, k0)
    x = np.arange(4096)
    m0 = np.sum(x * (np.abs(L) ** 2 + np.abs(R) ** 2))
    for _ in range(1000):
        L, R = q.step(L, R, th)
    m1 = np.sum(x * (np.abs(L) ** 2 + np.abs(R) ** 2))
    assert (m1 - m0) / 1000 == pytest.approx(np.cos(th), rel=1e-3)


def test_rest_energy_equals_theta():
    assert q.omega(0.0, 0.37) == pytest.approx(0.37)


def test_warp_matches_travel_time_formula():
    p = q.WarpParams(ratio=49, cos_theta_up=0.9, distance=2000, record_every=10 ** 9)
    res = q.run_warp(p)
    assert res.t_peak / 49 == pytest.approx(q.travel_time(2000, 49, 0.9), rel=2e-3)
    assert res.p_peak > 0.999


def test_naive_encoding_loses_half():
    p = q.WarpParams(distance=1000, matched=False, record_every=10 ** 9)
    res = q.run_warp(p)
    assert res.forward_weight == pytest.approx(0.5, abs=1e-6)
    assert res.p_peak == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------- 層モデル（研究ノート第9章の数値）
def test_c9():
    assert Lr.ceiling(9) == pytest.approx(1.38e10, rel=3e-3)


@pytest.mark.parametrize("dist, expect", [
    (4.2465, 9.7e-3),          # プロキシマ 9.7 ms
    (2.5e6, 95 * 60),          # アンドロメダ 95 分
    (4.65e10, 3.4 * K.YEAR),   # 観測可能な宇宙の半径 3.4 年
])
def test_note_travel_times(dist, expect):
    assert Lr.travel_time_s(dist, 9) == pytest.approx(expect, rel=0.02)


def test_case_speeds():
    assert Lr.required_speed(443, 5) == pytest.approx(2.8e9, rel=0.01)
    assert Lr.required_speed(5e6, 3600) == pytest.approx(4.4e10, rel=0.01)
    assert Lr.required_speed(1.6e6, 600) == pytest.approx(8.4e10, rel=0.01)


def test_gw170817_lead():
    assert Lr.gw_lead_time_years(4) == pytest.approx(1.3e8, rel=0.03)


def test_format():
    assert Lr.fmt_num(2401) == "2,401"
    assert Lr.fmt_num(2.5e39, 2) == "2.5×10³⁹"
    assert Lr.format_duration(9.68e-3) == "9.68 ms"
