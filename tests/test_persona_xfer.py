"""疑似人格のモデル間保存の実験コードの検証。

1. 項目バンクと真の回答の性質
2. 写像が既知の線形写像を復元すること
3. 統計（並べ替え検定の偽陽性率、Holm）
4. 陰性対照：人物情報を使わない合成モデルでは効果が出ないこと／陽性対照では出ること
5. 実アーキテクチャ（ランダム初期化の小型 Llama / Qwen2）でのフック・採点の動作
6. API バックエンドの回答の解釈（クライアントはモック）
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_xfer import items as I  # noqa: E402
from persona_xfer.stats import holm, paired_test  # noqa: E402
from persona_xfer.steering import cosine, fit_procrustes, fit_ridge, relative_layer  # noqa: E402


# ---------------------------------------------------------------- 1. 項目バンク
def test_item_bank_properties():
    P, items = I.load_personas(), I.load_items()
    assert len(items) == 65 and len({it.scenario_id for it in items}) == 47
    for p in P:
        T = [I.truth(p, it) for it in items]
        assert sum(t.answer is None for t in T) <= 6          # 曖昧な項目は少数
        assert sum(t.rule_dependent for t in T) >= 7          # 条件付き判断の項目が十分ある
        a = sum(t.answer == "A" for t in T)
        assert 0.3 < a / len(T) < 0.7                          # 正解の偏りが小さい


def test_split_keeps_pairs_together():
    items = I.load_items()
    tr, te = I.split_scenarios(items, 0.4, seed=0)
    assert {i.scenario_id for i in tr}.isdisjoint({i.scenario_id for i in te})
    assert len(tr) + len(te) == len(items)
    assert I.split_scenarios(items, 0.4, seed=0)[0] == tr     # 再現性


def test_personas_are_distinguishable():
    P, items = I.load_personas(), I.load_items()
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            a = [I.truth(P[i], it).answer for it in items]
            b = [I.truth(P[j], it).answer for it in items]
            dis = sum(x != y for x, y in zip(a, b) if x and y)
            assert dis >= 10


def test_rules_flip_answers_in_context():
    """文脈だけ違う対で答えが変わる場面が、人物ごとに少なくとも2つある。"""
    P, items = I.load_personas(), I.load_items()
    for p in P:
        flips = 0
        for sid in {it.scenario_id for it in items if it.n_variants > 1}:
            ans = {I.truth(p, it).answer for it in items if it.scenario_id == sid} - {None}
            flips += len(ans) == 2
        assert flips >= 2, p.pid


def test_decision_records_and_retrieval():
    from persona_xfer.prompts import Retriever, persona_system
    P, items = I.load_personas(), I.load_items()
    tr, te = I.split_scenarios(items)
    recs = I.decision_records(P[0], tr)
    r = Retriever(recs)
    top = r.top_k(te[0], 4)
    assert len(top) == 4 and all(x.item.scenario_id != te[0].scenario_id for x in top)
    s = persona_system(P[0], top)
    assert "過去の判断記録" in s and P[0].self_model["core"] in s


# ---------------------------------------------------------------- 2. 写像
@pytest.mark.parametrize("dA,dB", [(40, 60), (80, 50)])
def test_maps_recover_known_linear_map(dA, dB):
    rng = np.random.default_rng(0)
    k = 12                                       # 共有される潜在次元
    Z = rng.standard_normal((600, k))
    WA, WB = rng.standard_normal((k, dA)), rng.standard_normal((k, dB))
    HA = Z @ WA + 0.01 * rng.standard_normal((600, dA))
    HB = Z @ WB + 0.01 * rng.standard_normal((600, dB))
    z = rng.standard_normal(k)
    vA, vB = z @ WA, z @ WB
    assert cosine(fit_ridge(HA, HB).apply(vA), vB) > 0.98
    assert cosine(fit_procrustes(HA, HB, k=k).apply(vA), vB) > 0.6   # 直交制約のぶん粗い
    assert fit_ridge(HA, HB).cv_r2 > 0.95


def test_relative_layer():
    assert relative_layer(14, 28, 32) == 16
    assert relative_layer(1, 28, 32) == 1 and relative_layer(28, 28, 32) == 32


# ---------------------------------------------------------------- 3. 統計
def test_permutation_test_false_positive_rate():
    rng = np.random.default_rng(1)
    hits = 0
    n_sim = 200
    for s in range(n_sim):
        sc = np.repeat(np.arange(30), 4)
        d = rng.choice([-1, 0, 0, 1], size=len(sc))   # 帰無仮説：平均差 0
        r = paired_test(pd.DataFrame({"d": d, "scenario_id": sc}), B=400, seed=s)
        hits += r["p"] < 0.05
    assert hits / n_sim < 0.09


def test_permutation_test_power():
    rng = np.random.default_rng(2)
    sc = np.repeat(np.arange(30), 4)
    d = (rng.random(len(sc)) < 0.35).astype(float)     # 一方的に 0.35 の改善
    assert paired_test(pd.DataFrame({"d": d, "scenario_id": sc}), B=1000)["p"] < 0.01


def test_holm():
    out = holm([0.01, 0.04, 0.03, np.nan])
    assert out[0] == pytest.approx(0.03) and out[2] == pytest.approx(0.06)
    assert out[1] == pytest.approx(0.06) and np.isnan(out[3])


# ---------------------------------------------------------------- 4. 陰性・陽性対照（合成モデル）
def _run_toy(tmp_path, ignores):
    from persona_xfer.analysis import Analysis
    from persona_xfer.experiment import Experiment
    cfg = {"name": "t", "out_dir": str(tmp_path / ("neg" if ignores else "pos")),
           "models": [{"name": "a", "backend": "toy", "d_model": 32, "n_layers": 6, "seed": 1,
                       "ignores_persona": ignores},
                      {"name": "b", "backend": "toy", "d_model": 40, "n_layers": 6, "seed": 2,
                       "ignores_persona": ignores}],
           "conditions": ["base", "prompt", "steer_native", "steer_transfer", "steer_random"],
           "transfer_pairs": [["a", "b"]], "layers_rel": [0.5], "alphas": [2, 8],
           "big5": False, "nll": False}
    Experiment(cfg, root=Path("/")).run()
    an = Analysis(cfg["out_dir"], B=300)
    an.accuracy()
    return an, an.hypotheses()


def test_negative_control_shows_no_persona_effect(tmp_path):
    _, h = _run_toy(tmp_path, ignores=True)
    h1 = h[h.id == "H1"]
    assert (h1["p"] > 0.05).all() and (h1["diff"].abs() < 1e-9).all()


def test_positive_control_detects_effects(tmp_path):
    an, h = _run_toy(tmp_path, ignores=False)
    assert (h[h.id == "H1"]["p"] < 0.01).all()
    assert (h[h.id == "H2a"]["p"] < 0.05).all()
    t, _ = an.identity()
    real = t[t.group != "steer_random"]
    assert len(real) >= 2 and (real["diag_adv"] > 0.3).all()
    assert (t[t.group == "steer_random"]["diag_adv"] < 0.3).all()   # ランダム方向は人物を運ばない


# ---------------------------------------------------------------- 5. 実アーキテクチャ（ランダム初期化）
def _tiny_hf(arch: str, template: bool):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    corpus = [it.text + it.A + it.B for it in I.load_items()] + ["答え：A", "答え：B", "12345"]
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<s>", "</s>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    from tokenizers import decoders
    tk.decoder = decoders.ByteLevel()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>",
                                               unk_token="<unk>")
    if template:
        tok.chat_template = ("{% for m in messages %}<s>{{ m['role'] }}\n{{ m['content'] }}</s>{% endfor %}"
                             "{% if add_generation_prompt %}<s>assistant\n{% endif %}")
    torch.manual_seed(0)
    if arch == "llama":
        cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                       max_position_embeddings=1024)
        model = transformers.LlamaForCausalLM(cfg)
    else:
        cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                       max_position_embeddings=1024)
        model = transformers.Qwen2ForCausalLM(cfg)
    from persona_xfer.backends.hf import HFBackend
    return HFBackend(model=model, tokenizer=tok, name=arch, batch_size=3)


@pytest.mark.parametrize("arch,template", [("llama", True), ("qwen2", False)])
def test_hf_backend_hooks_and_scoring(arch, template):
    from persona_xfer.backends.base import Chat, Steer
    m = _tiny_hf(arch, template)
    assert m.n_layers == 3 and m.d_model == 32
    chats = [Chat("system text", f"質問 {i} です。どちらですか。", "答え：") for i in range(5)]
    h = m.hidden(chats, [1, 3])
    assert h[1].shape == (5, 32) and not np.allclose(h[1], h[3])
    # 左詰めのパディングがあっても、1件ずつ計算した値と一致する
    single = m.hidden(chats[:1], [3])[3]
    assert np.allclose(single[0], h[3][0], atol=1e-4)
    lp = m.choice_logprobs(chats, ["A", "B"])
    assert lp.shape == (5, 2) and np.all(lp < 0)
    v = np.random.default_rng(0).standard_normal(32)
    lp0 = m.choice_logprobs(chats, ["A", "B"], Steer(2, v, 0.0))
    lp1 = m.choice_logprobs(chats, ["A", "B"], Steer(2, v, 4.0))
    assert np.allclose(lp, lp0) and not np.allclose(lp, lp1)
    # ステアリングを加えた層の活性化は、ちょうど α·v だけ動く（フックの後始末も確認）
    assert np.allclose(m.hidden(chats, [3])[3], h[3], atol=1e-5)
    # 複数トークンのラベル（数字の列）の経路
    lp5 = m.choice_logprobs(chats[:2], ["12345", "A"])
    assert lp5.shape == (2, 2) and np.all(np.isfinite(lp5))
    assert m.text_nll(["今日は晴れです。"]) > 0


def test_hf_steering_shifts_residual_by_alpha_v():
    """層1 にステアリングすると、層2 への入力がちょうど hidden_states[1] + α·v になる。"""
    pytest.importorskip("torch")
    from persona_xfer.backends.base import Chat, Steer
    m = _tiny_hf("llama", True)
    chats = [Chat("s", "u", "")]
    v = np.random.default_rng(1).standard_normal(32).astype(np.float32)
    got = {}
    def grab(_m, args):  # 戻り値を返すと入力が置き換わるので、必ず None を返す
        got.setdefault("x", args[0][0, -1].detach().float().numpy().copy())

    h = m.layers[1].register_forward_pre_hook(grab)
    with m._steering(Steer(1, v, 0.5)):
        m.choice_logprobs(chats, ["A", "B"])
    h.remove()
    base = m.hidden(chats, [1])[1][0]
    assert np.allclose(got["x"], base + 0.5 * v, atol=1e-4)


def test_experiment_runs_on_tiny_hf(tmp_path):
    """実アーキテクチャで、段階1・段階2・解析・報告書まで通ることを確認する（数値は無意味）。"""
    from persona_xfer.analysis import Analysis
    from persona_xfer.experiment import Experiment
    built = {"x": _tiny_hf("llama", True), "y": _tiny_hf("qwen2", False)}
    cfg = {"name": "tiny", "out_dir": str(tmp_path / "tiny"),
           "models": [{"name": "x", "backend": "hf"}, {"name": "y", "backend": "hf"}],
           "conditions": ["base", "prompt", "retrieval", "steer_native", "steer_transfer", "steer_random",
                          "retrieval+steer_native"],
           "transfer_pairs": [["x", "y"]], "layers_rel": [0.5], "alphas": [1.0], "limit_items": 4,
           "personas": ["saeki", "morita"]}

    def factory(spec):
        b = built[spec["name"]]
        b.name = spec["name"]
        return b
    Experiment(cfg, root=Path("/"), backend_factory=factory).run()
    rep = Analysis(cfg["out_dir"], B=50).run(figures=False)
    txt = rep.read_text(encoding="utf-8")
    assert "steer_transfer:x" in txt and "判定" in txt
    d = pd.read_csv(Path(cfg["out_dir"]) / "transfer_diag__y.csv")
    assert set(d.persona) == {"saeki", "morita"}


# ---------------------------------------------------------------- 6. API バックエンド
def test_anthropic_backend_parsing(monkeypatch):
    pytest.importorskip("anthropic")
    from persona_xfer.backends import anthropic_api as A
    from persona_xfer.backends.base import Chat

    replies = iter(["B", "答えは Ａ です", None, "わかりません"])

    class FakeMessages:
        def create(self, **kw):
            assert "temperature" not in kw            # 現行モデルでは指定できない
            r = next(replies)
            if r is None:
                return SimpleNamespace(stop_reason="refusal", content=[])
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=r)])

    b = A.AnthropicBackend.__new__(A.AnthropicBackend)
    b.client = SimpleNamespace(messages=FakeMessages())
    b.model, b.name, b.n_samples, b.effort, b.max_tokens, b.n_refusals = "m", "m", 1, "low", 100, 0
    lp = b.choice_logprobs([Chat("s", "u")] * 4, ["A", "B"])
    p = np.exp(lp)
    assert p[0, 1] == pytest.approx(1, abs=1e-5) and p[1, 0] == pytest.approx(1, abs=1e-5)
    assert np.isnan(lp[2]).all() and np.isnan(lp[3]).all() and b.n_refusals == 1
