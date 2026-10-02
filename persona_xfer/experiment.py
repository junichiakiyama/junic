"""実験の実行。設定ファイル（JSON）に従い、モデルごとに結果を CSV に書く。

段階1（各モデル）: テキスト条件（base / prompt / retrieval）の判断と Big Five。
                   活性化を扱えるモデルは、全層の人物ベクトルと写像用コーパスの活性化をキャッシュ。
段階2（活性化を扱える各モデルを移植先として）: ステアリング条件。
  steer_native              移植先で抽出した人物ベクトル
  steer_transfer:<元モデル>  元モデルの人物ベクトルを写像したもの
  steer_random              steer_native と同じノルムのランダム方向（陰性対照）
  retrieval+steer_native    検索プロンプト＋人物ベクトル（テキストに上乗せがあるかの検定用）
  層と α はすべての条件で同じ格子・同じ学習用項目で選ぶ（調整の予算をそろえる）。
大きなモデルを同時に2つ載せないよう、段階ごとにモデルを読み直す。
"""
from __future__ import annotations

import gc
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import items as I
from .backends.base import Backend, Steer
from .evaluate import score_big5, score_decisions
from .prompts import Retriever, persona_system, system_for
from .steering import (cosine, fit_map, layers_at, mapping_chats, persona_vectors, random_like,
                       relative_layer)

TEXT_CONDS = ("base", "prompt", "retrieval")


# ---------------------------------------------------------------- 準備
def make_backend(spec: dict, personas) -> Backend:
    kind = spec["backend"]
    if kind == "toy":
        from .backends.toy import ToyBackend
        kw = {k: v for k, v in spec.items() if k not in ("backend", "name")}
        return ToyBackend(spec["name"], personas, **kw)
    if kind == "hf":
        from .backends.hf import HFBackend
        return HFBackend(spec["path"], name=spec["name"], dtype=spec.get("dtype", "auto"),
                         device=spec.get("device"), batch_size=spec.get("batch_size", 8),
                         trust_remote_code=spec.get("trust_remote_code", False))
    if kind == "anthropic":
        from .backends.anthropic_api import AnthropicBackend
        return AnthropicBackend(spec["model"], name=spec["name"], n_samples=spec.get("n_samples", 1),
                                effort=spec.get("effort", "low"))
    raise ValueError(kind)


def _free(backend):
    del backend
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class Experiment:
    def __init__(self, cfg: dict, root: Path | None = None, backend_factory=None):
        for k in ("alphas", "layers_rel"):
            if isinstance(cfg.get(k), str):
                raise ValueError(f"config '{k}' is not finalized: {cfg[k]}")
        self.cfg = cfg
        self.out = Path(cfg["out_dir"]) if root is None else root / cfg["out_dir"]
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "cache").mkdir(exist_ok=True)
        self.personas = I.load_personas()
        if cfg.get("personas"):
            self.personas = [p for p in self.personas if p.pid in cfg["personas"]]
        self.items = I.load_items()
        self.train, self.test = I.split_scenarios(self.items, cfg.get("train_frac", 0.4),
                                                  cfg.get("split_seed", 0))
        if cfg.get("limit_items"):
            n = cfg["limit_items"]
            self.train, self.test = self.train[:n], self.test[:n]
        self.train_ids = {it.item_id for it in self.train}
        self.eval_items = self.train + self.test
        self.ipip = I.load_ipip()
        q = I.load_questions()
        self.q_extract, self.q_map, self.roles = q["extraction"], q["mapping"], q["roles"]
        self.margin = cfg.get("margin", 0.5)
        self.k = cfg.get("retrieval_k", 4)
        self.records = {p.pid: I.decision_records(p, self.train, self.margin) for p in self.personas}
        self.retrievers = {pid: Retriever(r) for pid, r in self.records.items()}
        self.conds = cfg["conditions"]
        self.specs = {m["name"]: m for m in cfg["models"]}
        self.factory = backend_factory or (lambda spec: make_backend(spec, self.personas))
        (self.out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log = open(self.out / "run.log", "a", encoding="utf-8")

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def _path(self, kind: str, model: str) -> Path:
        return self.out / f"{kind}__{model}.csv"

    def _done(self, kind: str, model: str) -> bool:
        return self._path(kind, model).exists() and not self.cfg.get("force", False)

    # ---------------------------------------------------------------- 段階1
    def system_fn(self, text_cond: str, persona):
        return lambda it: system_for(text_cond, persona, it, self.retrievers.get(
            persona.pid) if persona else None, self.k)

    def run_text(self, m: Backend):
        rows, b5 = [], []
        for cond in [c for c in TEXT_CONDS if c in self.conds]:
            plist = [None] if cond == "base" else self.personas
            for p in plist:
                meta = {"persona": p.pid if p else None, "text_condition": cond}
                r = score_decisions(m, self.eval_items, p, self.system_fn(cond, p), None, meta,
                                    self.personas, self.train_ids, self.margin)
                rows += [{"model": m.name, "condition": cond, "layer": np.nan, "alpha": np.nan, **x} for x in r]
                if self.cfg.get("big5", True):
                    sys = persona_system(p) if p else system_for("base", None)
                    s = score_big5(m, self.ipip, sys, None, meta)
                    for tp in ([p] if p else self.personas):
                        b5.append({"model": m.name, "condition": cond, "persona": tp.pid, **s})
                self.log(f"{m.name} {cond} {p.pid if p else '-'} done")
        pd.DataFrame(rows).to_csv(self._path("decisions", m.name), index=False)
        if b5:
            pd.DataFrame(b5).to_csv(self._path("big5", m.name), index=False)

    def cache_activations(self, m: Backend):
        f = self.out / "cache" / f"{m.name}.npz"
        if f.exists() and not self.cfg.get("force", False):
            return
        layers = list(range(1, m.n_layers + 1))
        vecs = persona_vectors(m, self.personas, self.q_extract, layers, self.cfg.get("contrast", "neutral"))
        acts = m.hidden(mapping_chats(self.roles, self.q_map), layers)
        data = {"n_layers": np.array(m.n_layers)}
        for l in layers:
            data[f"map_{l}"] = acts[l].astype(np.float32)
            for p in self.personas:
                data[f"vec_{p.pid}_{l}"] = vecs[p.pid][l].astype(np.float32)
        np.savez_compressed(f, **data)
        self.log(f"{m.name} activations cached ({len(layers)} layers, d={m.d_model})")

    def load_cache(self, name: str) -> dict:
        return dict(np.load(self.out / "cache" / f"{name}.npz"))

    # ---------------------------------------------------------------- 段階2
    def steer_conditions(self, target: str) -> list[tuple[str, str | None]]:
        out = []
        for c in self.conds:
            if c == "steer_transfer":
                for src, tgt in self.cfg.get("transfer_pairs", []):
                    if tgt == target:
                        out.append((f"steer_transfer:{src}", src))
            elif c.startswith("steer_") or "+steer" in c:
                out.append((c, None))
        return out

    def vector_fn(self, cond: str, src: str | None, target_cache: dict, m: Backend, diag: list):
        """(人物, 層) → ベクトル を返す関数。"""
        def native(pid, l):
            return target_cache[f"vec_{pid}_{l}"]

        if cond in ("steer_native", "retrieval+steer_native", "prompt+steer_native"):
            return native
        if cond == "steer_random":
            return lambda pid, l: random_like(native(pid, l), seed=zlib.crc32(f"{pid}:{l}".encode()))
        if cond.startswith("steer_transfer:"):
            sc = self.load_cache(src)
            n_src = int(sc["n_layers"])
            maps = {}
            kind = self.cfg.get("map_kind", "ridge")

            def transferred(pid, l):
                ls = min(relative_layer(l, m.n_layers, n_src), max(1, n_src - 1))
                if l not in maps:
                    HA, HB = sc[f"map_{ls}"], target_cache[f"map_{l}"]
                    mp = fit_map(HA, HB, kind)
                    # 写像の縮小を、人物を含まないコーパスの尺度比で補正する（移植先の人物情報は使わない）
                    scale = np.median(np.linalg.norm(HB - HB.mean(0), axis=1)) / \
                        np.median(np.linalg.norm(HA - HA.mean(0), axis=1))
                    maps[l] = (mp, scale, ls)
                mp, scale, ls = maps[l]
                vA = sc[f"vec_{pid}_{ls}"]
                vB = mp.apply(vA)
                vB = vB / (np.linalg.norm(vB) + 1e-12) * np.linalg.norm(vA) * scale
                nat = native(pid, l)
                diag.append({"target": m.name, "source": src, "layer_target": l, "layer_source": ls,
                             "persona": pid, "map": kind, "map_r2": mp.cv_r2,
                             "cos_transfer_native": cosine(vB, nat),
                             "cos_random_native": cosine(random_like(nat, 0), nat),
                             "norm_ratio": float(np.linalg.norm(vB) / (np.linalg.norm(nat) + 1e-12))})
                return vB
            return transferred
        raise ValueError(cond)

    def _eval_steer(self, m, cond, vec, pid_layer_alpha, items, persona):
        text_cond = cond.split("+")[0] if "+" in cond else "base"
        l, a = pid_layer_alpha
        meta = {"persona": persona.pid, "text_condition": text_cond}
        st = Steer(l, vec(persona.pid, l), a)
        return score_decisions(m, items, persona, self.system_fn(text_cond, persona), st, meta,
                               self.personas, self.train_ids, self.margin), st, text_cond, meta

    def run_steer(self, m: Backend):
        cache = self.load_cache(m.name)
        layers = layers_at(m.n_layers, self.cfg.get("layers_rel", [0.3, 0.45, 0.6, 0.75]))
        alphas = self.cfg.get("alphas", [0.5, 1, 2, 4, 8])
        rows, b5, tune, nll, diag = [], [], [], [], []
        nll_texts = self.q_map
        base_nll = m.text_nll(nll_texts) if self.cfg.get("nll", True) else np.nan
        for cond, src in self.steer_conditions(m.name):
            vec = self.vector_fn(cond, src, cache, m, diag)
            # 層と α の調整（学習用項目、全人物の平均 P(正解)）
            best, best_s = None, -np.inf
            for l in layers:
                for a in alphas:
                    s = []
                    for p in self.personas:
                        r, *_ = self._eval_steer(m, cond, vec, (l, a), self.train, p)
                        s += [x["p_correct"] for x in r]
                    sc = float(np.nanmean(s))
                    tune.append({"model": m.name, "condition": cond, "layer": l,
                                 "layer_rel": l / m.n_layers, "alpha": a, "train_p_correct": sc})
                    if sc > best_s + 1e-9:
                        best, best_s = (l, a), sc
            self.log(f"{m.name} {cond}: layer={best[0]} alpha={best[1]} train={best_s:.3f}")
            nlls = []
            for p in self.personas:
                r, st, text_cond, meta = self._eval_steer(m, cond, vec, best, self.eval_items, p)
                rows += [{"model": m.name, "condition": cond, "layer": best[0], "alpha": best[1], **x} for x in r]
                if self.cfg.get("big5", True):
                    sys = persona_system(p) if text_cond != "base" else system_for("base", None)
                    b5.append({"model": m.name, "condition": cond, "persona": p.pid,
                               **score_big5(m, self.ipip, sys, st, meta)})
                if self.cfg.get("nll", True):
                    nlls.append(m.text_nll(nll_texts, st))
            if nlls:
                nll.append({"model": m.name, "condition": cond, "nll_base": base_nll,
                            "nll": float(np.mean(nlls)), "ppl_ratio": float(np.exp(np.mean(nlls) - base_nll))})
        pd.DataFrame(rows).to_csv(self._path("decisions_steer", m.name), index=False)
        pd.DataFrame(tune).to_csv(self._path("tuning", m.name), index=False)
        if b5:
            pd.DataFrame(b5).to_csv(self._path("big5_steer", m.name), index=False)
        if nll:
            pd.DataFrame(nll).to_csv(self._path("nll", m.name), index=False)
        if diag:
            pd.DataFrame(diag).drop_duplicates(["source", "layer_target", "persona"]).to_csv(
                self._path("transfer_diag", m.name), index=False)

    # ---------------------------------------------------------------- 全体
    def run(self):
        need_steer = any(c.startswith("steer_") or "+steer" in c for c in self.conds)
        for name, spec in self.specs.items():
            if self._done("decisions", name) and (
                    spec["backend"] == "anthropic" or not need_steer
                    or (self.out / "cache" / f"{name}.npz").exists()):
                self.log(f"{name}: stage 1 already done, skipped")
                continue
            m = self.factory(spec)
            if not self._done("decisions", name):
                self.run_text(m)
            if need_steer and m.has_activations:
                self.cache_activations(m)
            _free(m)
        if need_steer:
            for name, spec in self.specs.items():
                if spec["backend"] == "anthropic" or self._done("decisions_steer", name):
                    continue
                m = self.factory(spec)
                if m.has_activations:
                    self.run_steer(m)
                _free(m)
        self.log("all stages done")
