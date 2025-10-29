# tools/make_oos_from_best.py
import argparse, csv, datetime as dt, yaml, pathlib, sys

def last_date_from_prices(csv_path: str) -> str:
    last = None
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            d = row["date"]
            if (last is None) or (d > last):
                last = d
    if not last:
        raise ValueError("CSV prix vide ou sans colonne 'date'")
    return last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True, help="best_config.yaml (IS)")
    ap.add_argument("--prices", required=True, help="CSV prix (prices_eod.csv)")
    ap.add_argument("--oos-days", type=int, default=15, help="nb de jours OOS (par défaut 15)")
    ap.add_argument("--out", required=True, help="chemin de sortie YAML OOS")
    args = ap.parse_args()

    # Charge la meilleure config IS
    with open(args.best, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Le fichier --best n'est pas un mapping YAML valide.")

    # Calcule la fenêtre OOS depuis le CSV (dernière date = end ; start = end - oos_days)
    end_str = last_date_from_prices(args.prices)
    end_dt = dt.date.fromisoformat(end_str)
    start_dt = end_dt - dt.timedelta(days=args.oos_days)
    start_str = start_dt.isoformat()

    # Assure la section/structure
    cfg.setdefault("backtest", {})
    cfg["backtest"]["start"] = start_str
    cfg["backtest"]["end"] = end_str

    # Remplace le chemin des prix (chemin Windows bien **quoté**)
    prices_abs = str(pathlib.Path(args.prices).resolve())
    cfg.setdefault("data", {}).setdefault("prices", {})
    cfg["data"]["prices"]["path"] = prices_abs

    # Dump YAML propre
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Écrit {args.out}  | OOS: {start_str} -> {end_str}")
    print(f"[INFO] data.prices.path = {prices_abs}")

if __name__ == "__main__":
    sys.exit(main() or 0)
