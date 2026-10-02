"""疑似人格のモデル間保存の実験を、設定ファイルどおりに実行して報告書を作る。

例:
  python scripts/persona_run.py configs/persona/toy_pilot.json
  python scripts/persona_run.py configs/persona/main_7b.json --analyze-only
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persona_xfer.analysis import Analysis  # noqa: E402
from persona_xfer.experiment import Experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--analyze-only", action="store_true", help="実行済みの結果から報告書だけ作り直す")
    ap.add_argument("--force", action="store_true", help="既存の結果を無視して実行し直す")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.force:
        cfg["force"] = True
    if not a.analyze_only:
        Experiment(cfg, root=ROOT).run()
    rep = Analysis(ROOT / cfg["out_dir"], B=a.bootstrap).run(figures=not a.no_figures)
    print(f"report: {rep}")


if __name__ == "__main__":
    main()
