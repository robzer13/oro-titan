# src/oro/engine/run_backtest_v2.py
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# =========================
# Structures de données
# =========================

@dataclass
class BacktestResult:
    """
    Conteneur du résultat v2, pour standardiser l’écriture des artéfacts.

    Champs minimaux:
      - equity_curve : DataFrame ['date','equity'] (date str/ts ; equity float)
      - trades       : DataFrame ['date','ticker','w_prev','w_new','turnover_piece', ('cost' optionnelle)]
      - metrics      : dict (cagr, vol, sharpe, max_drawdown, …)
      - meta         : dict (date_range, universe_size, costs.bps, …)

    Optionnels:
      - prices       : DataFrame (index date, colonnes tickers) PRIX
      - returns      : DataFrame (index date, colonnes tickers) RENDEMENTS
    """
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    meta: Dict

    prices: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None


# =========================
# Utilitaires fichiers
# =========================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_csv_lf(df: pd.DataFrame, path: Path) -> None:
    """CSV LF et sans index (compat pandas <2.0)."""
    kwargs = dict(index=False)
    try:
        df.to_csv(path, **kwargs, line_terminator="\n")
    except TypeError:
        df.to_csv(path, **kwargs, lineterminator="\n")


def _dump_yaml(d: dict, path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(d or {}, f, sort_keys=True, allow_unicode=True)


# =========================
# Normalisations
# =========================

def _norm_equity_df(df: pd.DataFrame) -> pd.DataFrame:
    need = ["date", "equity"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"equity_curve manque colonnes {miss}")
    out = df.copy()
    out["date"] = out["date"].astype(str)
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out = out.dropna(subset=["equity"])
    return out


def _norm_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    need = ["date", "ticker", "w_prev", "w_new", "turnover_piece"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"trades manque colonnes {miss}")
    out = df.copy()
    out["date"] = out["date"].astype(str)
    for c in ["w_prev", "w_new", "turnover_piece", "cost"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =========================
# Aggrégations quotidiennes
# =========================

def _daily_from_trades(trades: pd.DataFrame, bps: float | None) -> pd.DataFrame:
    """
    Calcule turnover et coût/jour.
    Si 'cost' absente ou N/A/0 -> fallback bps * turnover / 1e4.
    """
    g = trades.groupby("date", dropna=False, as_index=False)
    daily = g.agg(
        turnover=("turnover_piece", "sum"),
        cost=("cost", "sum") if "cost" in trades.columns else ("turnover_piece", "sum"),
    )
    if "cost" not in trades.columns:
        daily["cost"] = np.nan
    if bps is None:
        bps = 0.0
    mask = daily["cost"].isna() | (daily["cost"] == 0)
    daily.loc[mask, "cost"] = daily.loc[mask, "turnover"] * (bps / 10000.0)
    return daily.sort_values("date").reset_index(drop=True)


def _daily_net_vs_gross(
    equity_curve: pd.DataFrame,
    daily_cost: pd.DataFrame
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """
    r_net depuis equity et r_gross ≈ r_net + coût du jour.
    Retourne (df, mult_net, mult_gross, win_rate, turnover_total).
    """
    ec = equity_curve.sort_values("date").reset_index(drop=True).copy()

    rows = []
    for i in range(1, len(ec)):
        prev, curr = ec.loc[i - 1, "equity"], ec.loc[i, "equity"]
        if pd.notna(prev) and pd.notna(curr) and prev != 0:
            rows.append({"date": str(ec.loc[i, "date"]), "rnet": float(curr) / float(prev) - 1.0})
    rnet = pd.DataFrame(rows)

    if not rnet.empty and not daily_cost.empty:
        joined = rnet.merge(daily_cost[["date", "cost"]], on="date", how="left")
    else:
        joined = rnet.copy()
        if "cost" not in joined:
            joined["cost"] = np.nan

    joined["cost"] = joined["cost"].fillna(0.0).astype(float)
    joined["rgross"] = joined["rnet"].astype(float) + joined["cost"].astype(float)

    mult_net = float(np.prod(1.0 + joined["rnet"].values)) if len(joined) else 1.0
    mult_gross = float(np.prod(1.0 + joined["rgross"].values)) if len(joined) else 1.0

    ups = int((joined["rnet"] > 0).sum())
    downs = int((joined["rnet"] < 0).sum())
    tot = ups + downs
    win_rate = (ups / tot) if tot else float("nan")

    turnover_total = float(daily_cost["turnover"].sum()) if "turnover" in daily_cost.columns else 0.0

    return joined, mult_net, mult_gross, win_rate, turnover_total


# =========================
# Ecriture des artéfacts
# =========================

def _write_artifacts(result: BacktestResult, report_dir: Path) -> Dict[str, Path]:
    _ensure_dir(report_dir)
    paths: Dict[str, Path] = {}

    # equity_curve.csv
    ec = _norm_equity_df(result.equity_curve)
    p_ec = report_dir / "equity_curve.csv"
    _to_csv_lf(ec, p_ec)
    paths["equity_curve.csv"] = p_ec

    # trades.csv (ajoute 'cost' vide si absente)
    tr = _norm_trades_df(result.trades)
    if "cost" not in tr.columns:
        tr["cost"] = np.nan
    tr = tr[["date", "ticker", "w_prev", "w_new", "turnover_piece", "cost"]]
    p_tr = report_dir / "trades.csv"
    _to_csv_lf(tr, p_tr)
    paths["trades.csv"] = p_tr

    # prices.csv (optionnel)
    if result.prices is not None and not result.prices.empty:
        p_prices = report_dir / "prices.csv"
        dfp = result.prices.copy()
        dfp = dfp.rename_axis("date").reset_index()
        _to_csv_lf(dfp, p_prices)
        paths["prices.csv"] = p_prices

    # returns.csv (optionnel)
    if result.returns is not None and not result.returns.empty:
        p_rets = report_dir / "returns.csv"
        dfr = result.returns.copy()
        dfr = dfr.rename_axis("date").reset_index()
        _to_csv_lf(dfr, p_rets)
        paths["returns.csv"] = p_rets

    # metrics.yaml
    p_metrics = report_dir / "metrics.yaml"
    _dump_yaml(result.metrics or {}, p_metrics)
    paths["metrics.yaml"] = p_metrics

    # report.yaml
    p_report = report_dir / "report.yaml"
    _dump_yaml(result.meta or {}, p_report)
    paths["report.yaml"] = p_report

    # trades_daily.csv (agrégé; fallback bps)
    bps = None
    costs = (result.meta or {}).get("costs", {})
    if isinstance(costs, dict) and "bps" in costs:
        try:
            bps = float(costs["bps"])
        except Exception:
            bps = None
    daily = _daily_from_trades(tr, bps)
    p_daily = report_dir / "trades_daily.csv"
    _to_csv_lf(daily, p_daily)
    paths["trades_daily.csv"] = p_daily

    # daily_net_vs_gross.csv
    joined, mult_net, mult_gross, win_rate, tot_turn = _daily_net_vs_gross(ec, daily)
    p_ng = report_dir / "daily_net_vs_gross.csv"
    _to_csv_lf(joined, p_ng)
    paths["daily_net_vs_gross.csv"] = p_ng

    # résumé console
    days = len(ec)
    ups = int((joined["rnet"] > 0).sum()) if "rnet" in joined else 0
    downs = int((joined["rnet"] < 0).sum()) if "rnet" in joined else 0
    tot = ups + downs
    winpct_str = f"{100.0 * ups / tot:0.1f} %" if tot else "N/A"
    tot_cost = float(daily["cost"].sum())

    print("")
    print(f"----- RÉSUMÉ ({report_dir.as_posix()}) -----")
    print(f"Jours d'equity (retours calculés) : {days}")
    print(f"Up/Down (net): {ups}/{downs}  |  Win%: {winpct_str}")
    print(f"Turnover total: {tot_turn:0.3f}")
    print(f"Coût total    : {tot_cost:0.3%}")
    print(f"Perf nette    : {mult_net - 1:0.3%}")
    print(f"Perf brute*   : {mult_gross - 1:0.3%}")
    print(f"Écart coûts   : {mult_gross - mult_net:0.3%}")
    print("Exports :")
    print(f" - {p_daily.as_posix()}")
    print(f" - {p_ng.as_posix()}")

    return paths


# =========================
# Moteur réel (adaptateur)
# =========================

def _adapt_engine_output(res: dict) -> BacktestResult:
    """Adapte un dict moteur → BacktestResult."""
    return BacktestResult(
        equity_curve=res["equity_curve"],
        trades=res["trades"],
        metrics=res.get("metrics", {}),
        meta=res.get("meta", {}),
        prices=res.get("prices"),
        returns=res.get("returns"),
    )


# =========================
# Stub (fallback)
# =========================

def _stub_backtest_logic(cfg: dict) -> BacktestResult:
    """
    Stub déterministe pour boucler tout le flux d’artéfacts.
    Ne dépend d’aucune clé de cfg ; produit des variations d’equity pour Win%.
    """
    dates = pd.bdate_range("2024-03-28", periods=12)
    rng = np.random.default_rng(1234)

    # --- equity (varie vraiment) ---
    rets = rng.normal(loc=0.0005, scale=0.004, size=len(dates))
    eq = np.cumprod(1.0 + rets)
    eq = (eq / eq[0]) * 1.00  # base 1.00
    equity_curve = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "equity": np.round(eq, 6)}
    )

    # --- trades (2 rebalances) ---
    trades = pd.DataFrame(
        {
            "date": [dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")],
            "ticker": ["AAA", "BBB"],
            "w_prev": [0.0, 0.10],
            "w_new": [0.10, 0.20],
            "turnover_piece": [1.0, 2.0],
            # pas de 'cost' -> on teste le fallback bps
        }
    )

    # --- prix / rendements (2 tickers) ---
    tickers = ["AAA", "BBB"]
    rets_m = rng.normal(loc=0.0, scale=0.01, size=(len(dates), len(tickers)))
    prices_arr = np.cumprod(1.0 + rets_m, axis=0)  # IMPORTANT: axis=0
    prices = pd.DataFrame(prices_arr, index=dates, columns=tickers)
    returns = pd.DataFrame(rets_m, index=dates, columns=tickers)

    metrics = {
        "cagr": -0.45,
        "vol": 0.25,
        "sharpe": -1.7,
        "max_drawdown": -0.06,
    }
    meta = {
        "date_range": {"start": str(dates.min().date()), "end": str(dates.max().date()), "n_days": int(len(dates))},
        "universe_size": int(cfg.get("universe_size", 10)),
        "costs": {"bps": float(cfg.get("costs", {}).get("bps", 5.0))},
        "strategy": cfg.get("strategy"),
        "signals": cfg.get("signals"),
        "rebalance": cfg.get("rebalance"),
    }

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        meta=meta,
        prices=prices,
        returns=returns,
    )



# =========================
# Runner principal
# =========================

def _load_config(config_path: Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _try_engine_run(cfg: dict) -> BacktestResult:
    """
    Essaie d’appeler le vrai moteur: `oro.engine.backtest_v2.run(cfg)`.
    Replie sur le stub si indisponible.
    """
    try:
        # import local (retarde l'import et facilite PYTHONPATH)
        from oro.engine.backtest_v2 import run as engine_run  # type: ignore
    except Exception as e:
        print(
            "[WARN] Moteur réel indisponible:", str(e),
            "\n       → utilisation du STUB pour terminer le flux d’artéfacts.",
            file=sys.stderr
        )
        return _stub_backtest_logic(cfg)

    # Appel du moteur réel
    res = engine_run(cfg)
    if not isinstance(res, dict):
        raise RuntimeError("Le moteur réel doit retourner un dict résultat.")
    return _adapt_engine_output(res)


def run_backtest_v2(config_path: Path | str, report_dir: Path | str) -> Dict[str, Path]:
    """
    Lance le backtest v2 et écrit tous les artéfacts v2.
    Retourne {nom_fichier: chemin}.
    """
    cfg = _load_config(Path(config_path))
    result = _try_engine_run(cfg)

    # Ajoute meta.minimales si absentes
    meta = dict(result.meta or {})
    if "date_range" not in meta:
        # si equity existe, derive n_days
        n_days = int(len(result.equity_curve)) if result.equity_curve is not None else 0
        start = str(result.equity_curve["date"].iloc[0]) if n_days else None
        end = str(result.equity_curve["date"].iloc[-1]) if n_days else None
        meta["date_range"] = {"start": start, "end": end, "n_days": n_days}
    if "universe_size" not in meta:
        meta["universe_size"] = int(cfg.get("universe_size", 10))
    if "costs" not in meta:
        meta["costs"] = {"bps": float(cfg.get("costs", {}).get("bps", 5.0))}
    result.meta = meta

    paths = _write_artifacts(result, Path(report_dir))

    # message de fin robuste
    n_days = result.meta.get("date_range", {}).get("n_days", "?")
    u_size = result.meta.get("universe_size", "?")
    print(
        f"[OK] Backtest v2 terminé. Dossier: {Path(report_dir).resolve()} | "
        f"Jours: {n_days} | Actifs: {u_size}"
    )
    return paths


# =========================
# CLI (python -m oro.engine.run_backtest_v2)
# =========================

def _cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Runner Backtest V2 (+écriture artéfacts)")
    ap.add_argument("--config", required=True, help="Chemin du YAML de config")
    ap.add_argument("--report-dir", required=True, help="Dossier de sortie des artéfacts")
    args = ap.parse_args(argv)

    run_backtest_v2(args.config, args.report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(_cli() or 0)
