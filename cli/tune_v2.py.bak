from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List, Tuple

import yaml

# --- Utils YAML --------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

# --- Dot-key helpers ---------------------------------------------------------

def set_by_dotted_key(d: dict, dotted: str, value: Any) -> None:
    """Set d['a']['b']['c'] = value for dotted='a.b.c' (creates intermediate dicts)."""
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def get_by_dotted_key(d: dict, dotted: str, default=None):
    cur = d
    parts = dotted.split(".")
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

# --- Sweep parsing & expansion -----------------------------------------------

def read_sweep_yaml(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Expect a YAML with:
      grid: [ {k1: [..], k2: [..]}, {k3: [..]} , ... ]
      fixed: { optional defaults applied to every run }
    We return:
      - list of param dicts (cartesian product within each grid block, then concatenation)
      - fixed dict (may be empty)
    """
    raw = load_yaml(path)
    if "grid" not in raw:
        raise ValueError("Le sweep doit contenir une clé 'grid'.")
    grid = raw["grid"]
    fixed = raw.get("fixed", {}) or {}

    if not isinstance(grid, list):
        raise ValueError("La clé 'grid' doit être une liste.")
    combos: List[Dict[str, Any]] = []
    for i, block in enumerate(grid, 1):
        if not isinstance(block, dict):
            raise ValueError("Chaque élément de 'grid' doit être un mapping (dict).")
        # Normalize: scalar -> [scalar]
        keys = list(block.keys())
        values_lists: List[List[Any]] = []
        for k in keys:
            v = block[k]
            if isinstance(v, list):
                values_lists.append(v)
            else:
                values_lists.append([v])
        for prod in itertools.product(*values_lists):
            combos.append({k: v for k, v in zip(keys, prod)})
    if not combos:
        raise ValueError("Aucune combinaison générée depuis 'grid' (listes vides ?).")
    return combos, fixed

# --- Prices path absolutizer -------------------------------------------------

def absolutize_prices_path(base_cfg: dict, base_cfg_path: Path) -> str | None:
    rel = get_by_dotted_key(base_cfg, "data.prices.path")
    if not rel:
        return None
    rel = str(rel)
    # Si déjà absolu, renvoie tel quel
    p = Path(rel)
    if p.is_absolute():
        return str(p)
    # Interpréter relatif depuis le dossier du base-config
    abs_p = (base_cfg_path.parent / p).resolve()
    return str(abs_p)

# --- Engine import -----------------------------------------------------------

# NOTE: On importe ici; PYTHONPATH doit pointer sur 'src'
from oro.engine.run_backtest_v2 import run_backtest_v2  # type: ignore

# --- Leaderboard helpers -----------------------------------------------------

def extract_metrics(report_dir: Path) -> dict:
    rpt = report_dir / "report.yaml"
    if not rpt.exists():
        return {}
    data = load_yaml(rpt)
    metrics = data.get("metrics", {}) or {}
    # Champs utiles
    return {
        "cagr": metrics.get("cagr"),
        "vol": metrics.get("vol"),
        "sharpe": metrics.get("sharpe"),
        "mdd": metrics.get("max_drawdown"),
        "report_dir": str(report_dir),
    }

def write_leaderboard(rows: List[dict], out_csv: Path) -> None:
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Colonnes
    fieldnames = [
        "run_id",
        "signals.mom.lookback_days",
        "strategy.topN",
        "rebalance.days",
        "cagr", "vol", "sharpe", "mdd",
        "is_cagr", "is_sharpe", "oos_cagr", "oos_sharpe",
        "report_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

# --- Main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, dest="base_config",
                    help="Chemin vers le YAML de config de base.")
    ap.add_argument("--sweep", required=True, dest="sweep",
                    help="YAML de sweep (grid/fixed).")
    ap.add_argument("--outdir", default="reports/exp_v2", dest="outdir",
                    help="Dossier de sortie (leaderboard, best_config).")
    args = ap.parse_args()

    base_cfg_path = Path(args.base_config).resolve()
    sweep_path = Path(args.sweep).resolve()
    outdir = Path(args.outdir).resolve()
    runs_root = outdir / "_runs"

    base_cfg = load_yaml(base_cfg_path)
    combos, fixed = read_sweep_yaml(sweep_path)

    # Forcer un chemin prix absolu (injecté dans 'fixed' pour tous les runs)
    abs_prices = absolutize_prices_path(base_cfg, base_cfg_path)
    if abs_prices:
        fixed = dict(fixed)  # copie
        fixed["data.prices.path"] = abs_prices

    print(f"[INFO] {len(combos)} combinaisons à tester…")

    runs_root.mkdir(parents=True, exist_ok=True)
    lb_rows: List[dict] = []
    best_row: dict | None = None

    # horodatage stable-ish pour run_id
    import datetime as _dt
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for i, params in enumerate(combos, 1):
        run_id = f"{stamp}_{i:03d}"
        run_dir = runs_root / f"run_{i:03d}_global"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Construire la config de ce run
        cfg_i = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))  # deep copy
        # Appliquer fixed puis params
        for k, v in (fixed or {}).items():
            set_by_dotted_key(cfg_i, k, v)
        for k, v in params.items():
            # ATTENTION: ici on doit passer des scalaires à l'engine (pas de listes!)
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            elif isinstance(v, list):
                # Si le grid n'a pas été aplati, c'est une erreur de logique
                # mais on protège: on prend la 1ère valeur
                v = v[0]
            set_by_dotted_key(cfg_i, k, v)

        # Sauver la config utilisée pour debug
        tmp_cfg_path = run_dir / "config_used.yaml"
        dump_yaml(cfg_i, tmp_cfg_path)

        # Lancer le backtest
        res = run_backtest_v2(config_path=str(tmp_cfg_path), report_dir=str(run_dir))

        # Extraire métriques
        m = extract_metrics(run_dir)
        m.update({
            "run_id": run_id,
            "signals.mom.lookback_days": get_by_dotted_key(cfg_i, "signals.mom.lookback_days"),
            "strategy.topN": get_by_dotted_key(cfg_i, "strategy.topN"),
            "rebalance.days": get_by_dotted_key(cfg_i, "rebalance.days"),
            # placeholders IS/OOS (si tu ajoutes un vrai split plus tard)
            "is_cagr": None,
            "is_sharpe": None,
            "oos_cagr": None,
            "oos_sharpe": None,
        })
        lb_rows.append(m)

        sharpe = m.get("sharpe")
        if sharpe is not None:
            if (best_row is None) or (sharpe > (best_row.get("sharpe") or float("-inf"))):
                best_row = m

        print(f"[{i}/{len(combos)}] {params} -> sharpe={m.get('sharpe')}")

    # Ecrire leaderboard
    outdir.mkdir(parents=True, exist_ok=True)
    write_leaderboard(lb_rows, outdir / "leaderboard.csv")

    # Choisir le best_config
    if best_row is not None:
        # Copier la config du run gagnant
        best_dir = Path(best_row["report_dir"])
        cfg_used = best_dir / "config_used.yaml"
        if cfg_used.exists():
            shutil.copy2(str(cfg_used), str(outdir / "best_config.yaml"))
        else:
            # fallback: reconstruire à partir des params
            best_cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))
            for k, v in (fixed or {}).items():
                set_by_dotted_key(best_cfg, k, v)
            for k in ["signals.mom.lookback_days", "strategy.topN", "rebalance.days"]:
                set_by_dotted_key(best_cfg, k, best_row.get(k))
            dump_yaml(best_cfg, outdir / "best_config.yaml")
    else:
        print("[WARN] Aucune ligne valide pour sélectionner un best_config.yaml")

    print(f"[OK] Sweep terminé. Leaderboard -> {outdir.relative_to(Path.cwd())}\\leaderboard.csv ; best_config.yaml généré.")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
