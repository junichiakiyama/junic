"""霊界の層ごとの速度上限モデル（研究ノート 第9章・第11章・付録E）。

モデルの仮定（代表値）
    τ_n = r · τ_{n+1}  →  c_n = r^{n-n_base} · c    （49ⁿ 則は r = 49, n_base = 3）
    v_warp = c_{n*} · cosθ*                           （n* は悟り、cosθ* は認識力）
    T = t_in + d / v_warp + t_out
r・n_base・n_max・cosθ*・t_in・t_out はすべて引数として変えられる。
"""
from __future__ import annotations

import math

import numpy as np

from . import constants as K


def ceiling(n, ratio=49.0, n_base=3):
    """層 n の速度上限（光速比）。"""
    return np.power(float(ratio), np.asarray(n, dtype=float) - n_base)


def travel_time_s(distance_ly, n, ratio=49.0, n_base=3, cos_theta=1.0,
                  t_in=0.0, t_out=0.0):
    v = ceiling(n, ratio, n_base) * cos_theta
    return t_in + distance_ly * K.YEAR / v + t_out


def required_speed(distance_ly, seconds):
    """距離[光年]を所要時間[s]で移動するのに必要な速さ（光速比）。"""
    return distance_ly * K.YEAR / seconds


def required_layer(speed, ratio=49.0, n_base=3, cos_theta=1.0):
    """速さ speed（光速比）を出すのに必要な最小の層（実数値。切り上げると整数の層）。"""
    return n_base + math.log(speed / cos_theta) / math.log(ratio)


def min_ratio_for(speed, n_max=9, n_base=3, cos_theta=1.0):
    """層 n_max までで speed に届くのに必要な最小の比 r。"""
    return (speed / cos_theta) ** (1.0 / (n_max - n_base))


def gw_lead_time_years(n, ratio=49.0, n_base=3, distance_ly=K.GW170817_DISTANCE_LY):
    """重力が層 n を通った場合に光より何年早く届くか（GW170817 の検証）。"""
    v = ceiling(n, ratio, n_base)
    return distance_ly * (1.0 - 1.0 / v)


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt_num(x: float, sig: int = 3) -> str:
    """1万未満はカンマ区切り、それ以上は a×10ⁿ で表記する。"""
    if x == 0 or not np.isfinite(x):
        return str(x)
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:,.{sig}g}" if abs(x) < 1000 else f"{x:,.0f}"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    ms = f"{m:.{sig}g}"
    if ms == "10":
        ms, e = "1", e + 1
    exp = str(e).translate(_SUP)
    return f"10{exp}" if ms == "1" else f"{ms}×10{exp}"


def format_duration(s: float) -> str:
    """秒を読みやすい日本語の時間表記に。"""
    if not np.isfinite(s):
        return "—"
    units = [(K.YEAR * 1e8, "億年"), (K.YEAR * 1e4, "万年"), (K.YEAR, "年"),
             (86400.0, "日"), (3600.0, "時間"), (60.0, "分"), (1.0, "秒"),
             (1e-3, "ms"), (1e-6, "μs"), (1e-9, "ns"), (1e-12, "ps")]
    for size, name in units:
        if s >= size:
            return f"{fmt_num(s / size)} {name}"
    return f"{s:.2g} 秒"
