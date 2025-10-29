# oro/engine/run_backtest_v2.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import argparse
import sys
import numpy as np
import pandas as pd
import yaml


# =========================
# Structures de données
# =========================

@dataclass
class BacktestResult:
    """
    Conteneur standard du résultat v2, pour simplifier l’écriture des artéfacts.

    Champs attendus (minimaux) :
      - equity_curve : DataFrame avec colonnes ['date','equity'] (date str/ts ; equity float)
      - trades       : DataFrame avec colonnes ['date','ticker','w_prev','w_new','turnover_piece','cost?']
      - metrics      : dict de métriques (cagr, vol, sharpe, max_drawdown, etc.)
      - meta         : dict (date_range, universe_size, costs.bps, …)

    Champs optionnels :
      - prices       : DataFrame (index date, colonnes tickers) des PRIX
      - returns      : DataFrame (index date, colonnes tickers) des RENDEMENTS (bruts instruments)
    """
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    meta: Dict

    prices: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None


# =========================
# Utilitaires robustes
# =========================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_csv_lf(df: pd.DataFrame, path: Path) -> None:
    """Ecrit un CSV LF sans index (compat pandas <2.0)."""
    kwargs = dict(index=False)
    try:
        df.to_csv(path, **kwargs, line_terminator="\n")  # pandas >= 1.5
    except TypeError:
        df.to_csv(path, **kwargs, lineterminator="\n")  # pandas < 1.5


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


def _daily_from_trades(trades: pd.DataFrame, bps: float | None) -> pd.DataFrame:
    """
    Calcule turnover et coût/jour. Si 'cost' absente ou nulle -> fallback bps*turnover/1e4.
    """
    if trades.empty:
        return pd.DataFrame(columns=["date", "turnover", "cost"])

    g = trades.groupby("date", dropna=False, as_index=False)
    if "cost" in trades.columns:
        daily = g.agg(turnover=("turnover_piece", "sum"), cost=("cost", "sum"))
    else:
        daily = g.agg(turnover=("turnover_piece", "sum"))
        daily["cost"] = np.nan

    if bps is None:
        bps = 0.0
    mask_missing = daily["cost"].isna() | (daily["cost"] == 0)
    daily.loc[mask_missing, "cost"] = daily.loc[mask_missing, "turnover"] * (bps / 10000.0)

    return daily.sort_values("date").reset_index(drop=True)


def _daily_net_vs_gross(
    equity_curve: pd.DataFrame, daily_cost: pd.DataFrame
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """
    Construit r_net (depuis equity) et r_gross ≈ r_net + coût_du_jour.
    Retourne (df, multNet, multGross, winRate, turnover_total).
    """
    ec = equity_curve.sort_values("date").reset_index(drop=True)

    # rendements nets (ignore le 1er jour)
    rows = []
    for i in range(1, len(ec)):
        prev, curr = ec.loc[i - 1, "equity"], ec.loc[i, "equity"]
        if pd.notna(prev) and pd.notna(curr) and prev != 0:
            rows.append({"date": str(ec.loc[i, "date"]), "rnet": float(curr) / float(prev) - 1.0})
    rnet = pd.DataFrame(rows)

    if rnet.empty:
        joined = pd.DataFrame(columns=["date", "rnet", "cost", "rgross"])
        return joined, 1.0, 1.0, float("nan"), 0.0

    # join coût/jour
    if not daily_cost.empty:
        joined = rnet.merge(daily_cost[["date", "cost"]], on="date", how="left")
    else:
        joined = rnet.copy()
        joined["cost"] = np.nan

    joined["cost"] = joined["cost"].fillna(0.0).astype(float)
    joined["rgross"] = joined["rnet"].astype(float) + joined["cost"]

    # cumuls
    mult_net = float(np.prod(1.0 + joined["rnet"].values))
    mult_gross = float(np.prod(1.0 + joined["rgross"].values))

    # win-rate (net)
    ups = int((joined["rnet"] > 0).sum())
    downs = int((joined["rnet"] < 0).sum())
    tot = ups + downs
    win_rate = (ups / tot) if tot else float("nan")

    turnover_total = float(daily_cost["turnover"].sum()) if "turnover" in daily_cost.columns else 0.0
    return joined, mult_net, mult_gross, win_rate, turnover_total


def _dump_yaml(d: dict, path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(d or {}, f, sort_keys=True, allow_unicode=True)


# =========================
# Ecriture des artéfacts
# =========================

def _write_artifacts(result: BacktestResult, report_dir: Path) -> Dict[str, Path]:
    """
    Ecrit tous les artéfacts utiles (prix, rendements, equity, trades, daily).
    Retourne un dict {nom_fichier: chemin}.
    """
    _ensure_dir(report_dir)
    paths: Dict[str, Path] = {}

    # 1) equity_curve.csv
    ec = _norm_equity_df(result.equity_curve)
    p_ec = report_dir / "equity_curve.csv"
    _to_csv_lf(ec, p_ec)
    paths["equity_curve.csv"] = p_ec

    # 2) trades.csv
    tr = _norm_trades_df(result.trades)
    if "cost" not in tr.columns:
        tr["cost"] = np.nan
    tr = tr[["date", "ticker", "w_prev", "w_new", "turnover_piece", "cost"]]
    p_tr = report_dir / "trades.csv"
    _to_csv_lf(tr, p_tr)
    paths["trades.csv"] = p_tr

    # 3) prices.csv (optionnel)
    if result.prices is not None and not result.prices.empty:
        p_prices = report_dir / "prices.csv"
        dfp = result.prices.rename_axis("date").reset_index()
        _to_csv_lf(dfp, p_prices)
        paths["prices.csv"] = p_prices

    # 4) returns.csv (optionnel)
    if result.returns is not None and not result.returns.empty:
        p_rets = report_dir / "returns.csv"
        dfr = result.returns.rename_axis("date").reset_index()
        _to_csv_lf(dfr, p_rets)
        paths["returns.csv"] = p_rets

    # 5) metrics.yaml
    p_metrics = report_dir / "metrics.yaml"
    _dump_yaml(result.metrics, p_metrics)
    paths["metrics.yaml"] = p_metrics

    # 6) report.yaml (meta + costs.bps)
    p_report = report_dir / "report.yaml"
    _dump_yaml(result.meta, p_report)
    paths["report.yaml"] = p_report

    # 7) trades_daily.csv (agrégé + fallback bps)
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

    # 8) daily_net_vs_gross.csv
    daily_net_vs_gross, mult_net, mult_gross, win_rate, tot_turnover = _daily_net_vs_gross(ec, daily)
    p_netgross = report_dir / "daily_net_vs_gross.csv"
    _to_csv_lf(daily_net_vs_gross, p_netgross)
    paths["daily_net_vs_gross.csv"] = p_netgross

    # 9) résumé console
    days = len(ec)
    ups = int((daily_net_vs_gross["rnet"] > 0).sum()) if not daily_net_vs_gross.empty else 0
    downs = int((daily_net_vs_gross["rnet"] < 0).sum()) if not daily_net_vs_gross.empty else 0
    tot = ups + downs
    winpct_str = f"{100.0*ups/tot:0.1f} %" if tot else "N/A"
    tot_cost = float(daily["cost"].sum()) if not daily.empty else 0.0

    print("")
    print(f"----- RÉSUMÉ ({report_dir}) -----")
    print(f"Jours d'equity (retours calculés) : {days}")
    print(f"Up/Down (net): {ups}/{downs}  |  Win%: {winpct_str}")
    print(f"Turnover total: {tot_turnover:0.3f}")
    print(f"Coût total    : {tot_cost:0.3%}")
    print(f"Perf nette    : {mult_net-1:0.3%}")
    print(f"Perf brute*   : {mult_gross-1:0.3%}")
    print(f"Écart coûts   : {mult_gross - mult_net:0.3%}")
    print("Exports :")
    print(f" - {p_daily}")
    print(f" - {p_netgross}")

    return paths


# =========================
# Moteur réel (adaptateur)
# =========================

_ENGINE_IMPORT_OK = True
try:
    # import préférentiel si ton paquet est namespacé
    from oro.engine.backtest_v2 import run as engine_run  # type: ignore
except Exception:
    try:
        # fallback si le module est à la racine du projet
        from backtest_v2 import run as engine_run  # type: ignore
    except Exception:
        _ENGINE_IMPORT_OK = False
        engine_run = None  # type: ignore


def _real_backtest_logic(cfg: dict) -> BacktestResult:
    """
    Adapte la sortie de backtest_v2.run(cfg) vers BacktestResult.
    """
    if not _ENGINE_IMPORT_OK or engine_run is None:
        raise RuntimeError(
            "Impossible d'importer le moteur réel (backtest_v2.run). "
            "Vérifie PYTHONPATH et le module 'backtest_v2'."
        )

    res = engine_run(cfg)  # <-- appelle ton moteur réel

    return BacktestResult(
        equity_curve=res["equity_curve"],
        trades=res["trades"],
        metrics=res.get("metrics", {}),
        meta=res.get("meta", {}),
        prices=res.get("prices"),
        returns=res.get("returns"),
    )


# =========================
# Stub (pour tests locaux)
# =========================

def _stub_backtest_logic(cfg: dict) -> BacktestResult:
    """
    ⚠️ Stub d’illustration. Remplace par ton moteur.
    """
    dates = pd.date_range(cfg["start"], cfg["end"], freq="B")
    # mini equity décroissante pour l’exemple
    eq = (1.0 + np.linspace(0, -0.035, len(dates))).clip(min=0.9)

    equity_curve = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "equity": eq})

    trades = pd.DataFrame(
        {
            "date": [dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")],
            "ticker": ["AAA", "BBB"],
            "w_prev": [0.0, 0.1],
            "w_new": [0.1, 0.2],
            "turnover_piece": [1.0, 2.0],
            # "cost": [np.nan, np.nan],  # volontairement omis pour tester fallback bps
        }
    )

    tickers = ["AAA", "BBB"]
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.normal(0, 0.01, size=(len(dates), len(tickers)))),
        index=dates,
        columns=tickers,
    )
    returns = prices.pct_change().fillna(0.0)

    metrics = {
        "cagr": -0.0553,
        "vol": 0.305,
        "sharpe": -2.48,
        "max_drawdown": -0.052,
    }
    meta = {
        "date_range": {"start": cfg["start"], "end": cfg["end"], "n_days": int(len(dates))},
        "universe_size": int(cfg.get("universe_size", 10)),
        "costs": {"bps": float(cfg.get("costs", {}).get("bps", 5.0))},
        "signals": cfg.get("signals"),
        "strategy": cfg.get("strategy"),
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
# Orchestrateur public + CLI
# =========================

def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_backtest_v2(config_path: Path, report_dir: Path) -> Dict[str, Path]:
    """
    Lance le backtest v2 (réel) et écrit tous les artéfacts.
    Retourne un dict {nom_fichier: chemin}.
    """
    cfg = _load_config(config_path)

    # Par défaut, on tente le moteur réel ; fallback stub si indisponible
    try:
        result = _real_backtest_logic(cfg)
        used_stub = False
    except Exception as e:
        print(f"[WARN] Moteur réel indisponible: {e}\n       → utilisation du STUB pour terminer le flux d’artéfacts.")
        result = _stub_backtest_logic(cfg)
        used_stub = True

    paths = _write_artifacts(result, report_dir)

    # Message final
    n_days = result.meta.get("date_range", {}).get("n_days", len(result.equity_curve))
    u_size = result.meta.get("universe_size", "?")
    print(
        f"[OK] Backtest v2 terminé. Dossier: {report_dir.resolve()} | "
        f"Jours: {n_days} | Actifs: {u_size} | Source: {'STUB' if used_stub else 'ENGINE'}"
    )

    return paths


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run backtest v2 and write full artifacts.")
    p.add_argument("--config", required=True, type=str, help="Chemin du fichier YAML de configuration")
    p.add_argument("--report-dir", required=True, type=str, help="Dossier de sortie des artéfacts")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    report_dir = Path(args.report_dir)
    run_backtest_v2(config_path, report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
