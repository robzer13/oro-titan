from __future__ import annotations
import argparse, json, copy
from pathlib import Path
import pandas as pd
import yaml
from itertools import product
from oro.engine.run_backtest import run_backtest

def _load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _dump_yaml(p: Path, obj: dict):
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sweep", required=True, help="JSON de l espace: {\"lookback\":[5,10], ...}")
    ap.add_argument("--outdir", default="reports/exp")
    args = ap.parse_args()

    base_cfg = _load_yaml(Path(args.config))
    space = json.loads(Path(args.sweep).read_text(encoding="utf-8"))

    keys = list(space.keys())
    grid = list(product(*[space[k] for k in keys]))

    rows = []
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for i, combo in enumerate(grid, 1):
        cfg = copy.deepcopy(base_cfg)
        # On pose ces hyperparams dans config["params"]
        cfg.setdefault("params", {})
        for k, v in zip(keys, combo):
            cfg["params"][k] = v
        # écrire cfg temp
        cfg_path = outdir / f"cfg_{i:03d}.yaml"
        _dump_yaml(cfg_path, cfg)

        rep = outdir / f"run_{i:03d}"
        res = run_backtest(config_path=str(cfg_path), report_dir=str(rep))
        rows.append({"run": i, **{k: v for k, v in cfg["params"].items()}, **res})

    lb = pd.DataFrame(rows)
    lb.sort_values(by=["equity_last","tot_return"], ascending=[False, False], inplace=True)
    lb.to_csv(outdir / "leaderboard.csv", index=False)

    # best config
    if not lb.empty:
        best_idx = int(lb.iloc[0]["run"])
        best_cfg = outdir / f"cfg_{best_idx:03d}.yaml"
        (outdir / "best_config.yaml").write_text(best_cfg.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[OK] Tuning terminé. Leaderboard -> {outdir/'leaderboard.csv'} ; best_config.yaml généré.")
    else:
        print("[WARN] Leaderboard vide.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main() or 0)
