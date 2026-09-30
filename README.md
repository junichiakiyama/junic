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
