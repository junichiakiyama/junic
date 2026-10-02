# warpsim — 霊界科学 研究ノート「ワープ・タイムトラベルと多次元構造」の数値解析と図

研究ノート（第10版）のうち、次の二つを数値計算と図にしたものです。

| 候補 | 内容 | 対応する章 |
|---|---|---|
| 1 | Dirac 量子ウォークによる「運動＝分解と再構成」と、層を乗り換えるワープ（経路D）の数値シミュレーション | 第9章（9.1〜9.3） |
| 4 | 層ごとの速度上限（49ⁿ 則）による所要時間、教学事例との照合、必要エネルギーの比較、比 r の掃引 | 第9章・第11章・付録E・8.13–8.14節 |

図の説明・読み取れること・仮定の区別は **[docs/figures_guide.md](docs/figures_guide.md)** にまとめています。

## 使い方

```bash
pip install -r requirements.txt
bash scripts/run_all.sh          # 代表値の図と数値表をすべて作り直す（約5分）
python -m pytest -q tests        # 研究ノートの数値の再現を含む検算
```

出力先

- `figures/representative/` … 代表値の図（PNG 300 dpi と PDF の両方。Word には PNG、組版には PDF）
- `results/representative/` … 数値表（CSV は Excel でそのまま開ける BOM 付き UTF-8）と要約（JSON）
- `figures/custom/<tag>/`、`results/custom/<tag>/` … 自由パラメータで実行した結果

## パラメータ

### 層モデル（`scripts/layer_figures.py`）

| 引数 | 意味 | 代表値 |
|---|---|---|
| `--ratio` | 1段あたりの光速比 r（cₙ = r^(n−3) c） | 49 |
| `--n-base` | 基底層の次元 | 3 |
| `--n-max` | 最上層の次元 | 9 |
| `--cos-theta` | 認識力 cosθ*（層の上限に対する割合） | 1 |
| `--t-in`, `--t-out` | 符号化・復号にかかる時間 [秒] | 0 |
| `--tag` | 出力先の名前（指定すると custom に出力） | なし |

```bash
python scripts/layer_figures.py --ratio 66.2 --tag r66_all_cases_in_9D
python scripts/layer_figures.py --ratio 49 --cos-theta 0.5 --t-in 60 --t-out 60 --tag slow_encode
```

### 量子ウォーク（`scripts/qw_figures.py`）

| 引数 | 意味 | 代表値 |
|---|---|---|
| `--ratio` | 上位層と基底層の光速比 cₙ/c₃（隣り合う層なら r） | 49 |
| `--cos-theta` | 上位層での cosθ*（認識力） | 0.9 |
| `--theta-base` | 基底層での θ（船の質量 mc² = ħθ/τ に対応） | 0.3 |
| `--distance` | 目的地までの距離（格子数） | 4000 |
| `--sigma` | 船（波束）の大きさ（格子数） | 30 |
| `--window` | 復号窓の半幅（格子数） | 200 |
| `--only` | 作る図を限定（例 `--only Q5 Q7`） | 全部 |
| `--tag` | 出力先の名前 | なし |

```bash
python scripts/qw_figures.py --ratio 20 --cos-theta 0.7 --only Q5 Q7 --tag r20_cos07
```

## 構成

```
warpsim/qwalk.py      量子ウォークの時間発展、固有基底、層間の符号化・復号、2層ワープ
warpsim/layers.py     層ごとの速度上限・所要時間・必要な層・最小の r
warpsim/constants.py  物理定数と、研究ノートから引いた数値（出典の節つき）
warpsim/style.py      図の共通スタイル（日本語フォント、固定色順）
scripts/              図と数値表を作るスクリプト
tests/                検算
```

---

# persona_xfer — 疑似人格のモデル間保存の検証

同じ人格定義を与えたとき、異なる LLM の間でその人格が「同じ人物」として保たれるかを測る実験コードです。テキスト（自己記述・判断記録）、移植先で再抽出した人物ベクトル、別モデルから写像で移植した人物ベクトルを、同じ項目・同じ統計で比べます。

研究の問い・仮説・判定基準・段階計画は **[docs/persona_transfer/protocol.md](docs/persona_transfer/protocol.md)** にまとめています。

```bash
python scripts/persona_run.py configs/persona/toy_pilot.json     # 合成モデルでのパイプライン較正（CPU、約2分）
python scripts/persona_run.py configs/persona/pilot_small.json   # Phase 1：小型オープンモデル（GPU）
python -m pytest -q tests/test_persona_xfer.py
```

| 設定 | 内容 |
|---|---|
| `configs/persona/toy_pilot.json` | 合成モデル3つ（効果を構成上与えた較正用。結果は科学的知見ではない） |
| `configs/persona/pilot_small.json` | Qwen2.5-1.5B/3B、Llama-3.2-3B |
| `configs/persona/main_7b.json` | Llama-3.1-8B、Qwen2.5-7B/3B、Gemma-2-9B、Mistral-7B（`alphas` は Phase 1 の後に確定） |
| `configs/persona/frontier_text.json` | Claude Opus 5.5 / Sonnet 5.5 ＋ Qwen 7B（テキスト条件のみ） |

```
persona_xfer/data/        架空の4人物、判断項目（47場面・65項目）、Mini-IPIP 試訳、抽出・写像用の質問
persona_xfer/items.py     人格定義の読み込み、真の回答（価値の重み＋文脈規則）、場面単位の分割、判断記録
persona_xfer/prompts.py   条件ごとのプロンプト、判断記録の検索
persona_xfer/steering.py  人物ベクトルの抽出、モデル間写像（リッジ／Procrustes）
persona_xfer/backends/    hf（オープンウェイト）、anthropic_api（テキストのみ）、toy（較正用の合成モデル）
persona_xfer/experiment.py  実行（段階1：テキスト条件とキャッシュ、段階2：ステアリング条件）
persona_xfer/analysis.py  検定・同一性行列・識別率・判定、figures.py / report.py
```
