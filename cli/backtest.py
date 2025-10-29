# cli/backtest.py
from __future__ import annotations
import argparse
import os
import sys

# Permet d'exécuter après avoir fait:  $env:PYTHONPATH = "$PWD\src"
from oro.engine.run_backtest import run_backtest


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest baseline (EW) avec exports")
    ap.add_argument("--config", required=True, help="Chemin du YAML de config")
    ap.add_argument("--report-dir", required=True, help="Dossier de sortie des rapports")
    args = ap.parse_args()

    res = run_backtest(config_path=args.config, report_dir=args.report_dir)
    print(
        f"[OK] Backtest terminé. Dossier: {res['report_dir']} | "
        f"Jours: {res['n_days']} | Actifs: {res['n_assets']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
