#!/usr/bin/env bash
# 代表値の図と数値表をすべて作り直す。自由パラメータの例も2つ作る。
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/qw_figures.py
python3 scripts/layer_figures.py
# 自由パラメータの例
python3 scripts/layer_figures.py --ratio 66.2 --tag r66_all_cases_in_9D
python3 scripts/qw_figures.py --ratio 20 --cos-theta 0.7 --only Q5 Q7 --tag r20_cos07
