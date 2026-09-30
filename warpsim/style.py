"""研究ノート用の図の共通スタイル（印刷前提・ライトモード）。

色は役割ごとに固定する：
- カテゴリ（系列の識別）は SERIES の順番で割り当て、使い回さない
- 大きさ（確率密度など）は青一色の SEQ_BLUE
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# カテゴリ色（固定順）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#ffffff"

# 青の順序ランプ（100→700）。0 付近は背景に溶けてよい
_SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
        "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", _SEQ)


def _find_jp_font() -> str | None:
    for name in ("IPAexGothic", "IPAPGothic", "IPAGothic", "Noto Sans CJK JP",
                 "Hiragino Sans", "Yu Gothic", "MS Gothic"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return None


def setup() -> None:
    font = _find_jp_font()
    plt.rcParams.update({
        "font.family": [font, "DejaVu Sans"] if font else ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save(fig, outdir: str | os.PathLike, name: str) -> list[Path]:
    """PNG（Word 貼り付け用、300 dpi）と PDF（ベクター）の両方で保存。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def note(ax, text: str, loc: str = "lower right") -> None:
    """図の隅に「仮定」「代表値」などの注記を入れる。"""
    pos = {"lower right": (0.98, 0.02, "right", "bottom"),
           "upper left": (0.02, 0.98, "left", "top"),
           "upper right": (0.98, 0.98, "right", "top"),
           "lower left": (0.02, 0.02, "left", "bottom")}[loc]
    ax.text(pos[0], pos[1], text, transform=ax.transAxes, ha=pos[2], va=pos[3],
            fontsize=7, color=TEXT_2,
            bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID, lw=0.6))
