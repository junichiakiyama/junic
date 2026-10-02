"""人物ベクトルの抽出と、モデル間写像。

抽出（Chen et al. 2025 の手順を判断課題向けに簡略化）
  v_p = mean_q h(人物ありのシステム, q) − mean_q h(中立のシステム, q)
  q は抽出用の中立な質問（評価項目とは重複しない）。応答を生成せず、プロンプト末尾の
  トークン位置の活性化を使う（計算が安く決定的。応答トークン平均は拡張として残す）。
  contrast="others" のときは他の人物の平均を引き、「その人物に固有の方向」だけを残す。

写像（モデル A の層 ℓ_A → モデル B の層 ℓ_B）
  人物を含まない対応テキスト（汎用の役割 × 写像用の質問）で両モデルの活性化を取り、
  ridge  : 中心化した H_B ≈ H_A·M をリッジ回帰で解く（双対形で d が大きくても安い）
  procrustes : 両者を PCA で k 次元に落とし、直交変換＋スケールで合わせる
  差分ベクトルを写すので平行移動（切片）は消える。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends.base import Backend, Chat
from .items import Persona
from .prompts import NEUTRAL_SYSTEM, persona_system


def extraction_chats(questions: list[str], persona: Persona | None) -> list[Chat]:
    if persona is None:
        return [Chat(NEUTRAL_SYSTEM, q, "", {"persona": None, "text_condition": "base"}) for q in questions]
    sys = persona_system(persona)
    return [Chat(sys, q, "", {"persona": persona.pid, "text_condition": "prompt"}) for q in questions]


def persona_vectors(backend: Backend, personas: list[Persona], questions: list[str],
                    layers: list[int], contrast: str = "neutral") -> dict[str, dict[int, np.ndarray]]:
    off = backend.hidden(extraction_chats(questions, None), layers)
    raw = {}
    for p in personas:
        on = backend.hidden(extraction_chats(questions, p), layers)
        raw[p.pid] = {l: on[l].mean(0) - off[l].mean(0) for l in layers}
    if contrast == "neutral":
        return raw
    if contrast == "others":
        out = {}
        for p in personas:
            others = [q.pid for q in personas if q.pid != p.pid]
            out[p.pid] = {l: raw[p.pid][l] - np.mean([raw[o][l] for o in others], 0) for l in layers}
        return out
    raise ValueError(contrast)


def mapping_chats(roles: list[str], questions: list[str], neutral_system: str = NEUTRAL_SYSTEM
                  ) -> list[Chat]:
    chats = [Chat(neutral_system, q, "", {"role": None}) for q in questions]
    for r in roles:
        chats += [Chat(r, q, "", {"role": r}) for q in questions]
    return chats


@dataclass
class LinearMap:
    """差分ベクトル用の線形写像 v_B = apply(v_A)。"""
    kind: str
    params: dict
    cv_r2: float = float("nan")

    def apply(self, v: np.ndarray) -> np.ndarray:
        if self.kind == "ridge":
            # v_B = Y^T (XX^T + λI)^{-1} X v
            return self.params["Yt_Kinv"] @ (self.params["X"] @ v)
        if self.kind == "procrustes":
            P_A, P_B, R, s = (self.params[k] for k in ("P_A", "P_B", "R", "s"))
            # 行ベクトルで z_B = s·z_A·R^T なので、列ベクトルでは z_B = s·R·z_A
            return P_B.T @ (s * (R @ (P_A @ v)))
        raise ValueError(self.kind)


def _ridge_dual(X, Y, lam):
    K = X @ X.T
    Kinv = np.linalg.solve(K + lam * np.eye(len(K)), np.eye(len(K)))
    return Y.T @ Kinv


def fit_ridge(HA: np.ndarray, HB: np.ndarray, lambdas=(1e-2, 1e-1, 1, 10, 100, 1000),
              folds: int = 5, seed: int = 0) -> LinearMap:
    X = HA - HA.mean(0)
    Y = HB - HB.mean(0)
    # 尺度をそろえて λ の格子を意味のあるものにする
    sx = np.sqrt((X ** 2).sum(1).mean()) or 1.0
    Xs = X / sx
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(idx, folds)
    best, best_r2 = None, -np.inf
    for lam in lambdas:
        ss_res = ss_tot = 0.0
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            mu_x, mu_y = Xs[tr].mean(0), Y[tr].mean(0)
            A = _ridge_dual(Xs[tr] - mu_x, Y[tr] - mu_y, lam)
            pred = (A @ ((Xs[tr] - mu_x) @ (Xs[te] - mu_x).T)).T + mu_y
            ss_res += ((Y[te] - pred) ** 2).sum()
            ss_tot += ((Y[te] - Y[tr].mean(0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        if r2 > best_r2:
            best, best_r2 = lam, r2
    A = _ridge_dual(Xs, Y, best)
    return LinearMap("ridge", {"Yt_Kinv": A, "X": Xs / sx, "lambda": best}, cv_r2=float(best_r2))


def fit_procrustes(HA: np.ndarray, HB: np.ndarray, k: int = 64) -> LinearMap:
    X = HA - HA.mean(0)
    Y = HB - HB.mean(0)
    k = int(min(k, X.shape[1], Y.shape[1], len(X) - 1))
    P_A = np.linalg.svd(X, full_matrices=False)[2][:k]
    P_B = np.linalg.svd(Y, full_matrices=False)[2][:k]
    ZA, ZB = X @ P_A.T, Y @ P_B.T
    U, S, Vt = np.linalg.svd(ZA.T @ ZB)
    R = (U @ Vt).T  # ZB ≈ s · ZA R^T
    s = S.sum() / (ZA ** 2).sum()
    pred = s * ZA @ R.T
    r2 = 1 - ((ZB - pred) ** 2).sum() / (ZB ** 2).sum()
    return LinearMap("procrustes", {"P_A": P_A, "P_B": P_B, "R": R, "s": s}, cv_r2=float(r2))


def fit_map(HA, HB, kind: str = "ridge", **kw) -> LinearMap:
    return fit_ridge(HA, HB, **kw) if kind == "ridge" else fit_procrustes(HA, HB, **kw)


def relative_layer(l_target: int, n_target: int, n_source: int) -> int:
    """同じ相対深さにある元モデルの層。"""
    return int(np.clip(round(l_target / n_target * n_source), 1, n_source))


def layers_at(n_layers: int, rel: list[float]) -> list[int]:
    """相対深さ → 層番号。最終層は除く（HF の hidden_states の最後は最終正規化の後の値で、
    ブロック出力に加えるステアリングと空間が一致しないため）。"""
    hi = max(1, n_layers - 1)
    return sorted({int(np.clip(round(r * n_layers), 1, hi)) for r in rel})


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def random_like(v: np.ndarray, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed).standard_normal(v.shape)
    return r / np.linalg.norm(r) * np.linalg.norm(v)
