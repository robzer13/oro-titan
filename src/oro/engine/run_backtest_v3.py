# -*- coding: utf-8 -*-
# src/oro/engine/run_backtest_v3.py
from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ========= Structures =========

@dataclass
class BacktestV3Result:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict
    meta: Dict
    prices: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None
    positions: Optional[pd.DataFrame] = None
    signals: Optional[pd.DataFrame] = None


# ========= Utils =========

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _to_csv_lf(df: pd.DataFrame, path: Path) -> None:
    kwargs = dict(index=False)
    try:
        df.to_csv(path, **kwargs, line_terminator="\n")
    except TypeError:
        df.to_csv(path, **kwargs, lineterminator="\n")

def _dump_yaml(d: dict, path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(d or {}, f, sort_keys=True, allow_unicode=True)

def _norm_equity_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or "equity" not in out.columns:
        raise ValueError("equity_curve doit contenir ['date','equity']")
    out["date"] = out["date"].astype(str)
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out = out.dropna(subset=["equity"]).reset_index(drop=True)
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
    g = trades.groupby("date", dropna=False, as_index=False)
    if "cost" in trades.columns:
        daily = g.agg(turnover=("turnover_piece", "sum"), cost=("cost", "sum"))
    else:
        daily = g.agg(turnover=("turnover_piece", "sum"))
        daily["cost"] = np.nan

    if bps is None:
        bps = 0.0
    mask = daily["cost"].isna() | (daily["cost"] == 0)
    daily.loc[mask, "cost"] = daily.loc[mask, "turnover"] * (bps / 10000.0)
    return daily.sort_values("date").reset_index(drop=True)

def _daily_net_vs_gross(equity_curve: pd.DataFrame, daily_cost: pd.DataFrame) -> Tuple[pd.DataFrame, float, float, float, float]:
    ec = equity_curve.sort_values("date").reset_index(drop=True)
    rnet = []
    for i in range(1, len(ec)):
        prev, curr = float(ec.loc[i-1, "equity"]), float(ec.loc[i, "equity"])
        if np.isfinite(prev) and prev != 0 and np.isfinite(curr):
            rnet.append({"date": str(ec.loc[i, "date"]), "rnet": (curr/prev) - 1.0})
    rnet = pd.DataFrame(rnet)

    if not daily_cost.empty:
        j = rnet.merge(daily_cost[["date", "cost"]], on="date", how="left")
    else:
        j = rnet.copy(); j["cost"] = np.nan

    j["cost"] = j["cost"].fillna(0.0).astype(float)
    j["rgross"] = j["rnet"].astype(float) + j["cost"]

    mult_net   = float(np.prod(1.0 + j["rnet"].values))  if len(j) else 1.0
    mult_gross = float(np.prod(1.0 + j["rgross"].values)) if len(j) else 1.0

    ups   = int((j["rnet"] > 0).sum())
    downs = int((j["rnet"] < 0).sum())
    tot   = ups + downs
    win   = (ups / tot) if tot else float("nan")

    turnover_total = float(daily_cost["turnover"].sum()) if "turnover" in daily_cost.columns else 0.0
    return j, mult_net, mult_gross, win, turnover_total


# ========= Ecriture artéfacts v3 =========

def _write_artifacts(res: BacktestV3Result, report_dir: Path) -> Dict[str, Path]:
    _ensure_dir(report_dir)
    paths: Dict[str, Path] = {}

    # equity
    ec = _norm_equity_df(res.equity_curve)
    p_ec = report_dir / "equity_curve.csv"; _to_csv_lf(ec, p_ec); paths["equity_curve.csv"] = p_ec

    # trades
    tr = _norm_trades_df(res.trades)
    if "cost" not in tr.columns:
        tr["cost"] = np.nan
    tr = tr[["date","ticker","w_prev","w_new","turnover_piece","cost"]]
    p_tr = report_dir / "trades.csv"; _to_csv_lf(tr, p_tr); paths["trades.csv"] = p_tr

    # prices / returns (optionnels)
    if res.prices is not None and not res.prices.empty:
        p_prices = report_dir / "prices.csv"
        _to_csv_lf(res.prices.rename_axis("date").reset_index(), p_prices)
        paths["prices.csv"] = p_prices

    if res.returns is not None and not res.returns.empty:
        p_rets = report_dir / "returns.csv"
        _to_csv_lf(res.returns.rename_axis("date").reset_index(), p_rets)
        paths["returns.csv"] = p_rets

    # positions / signals (optionnels)
    if res.positions is not None and not res.positions.empty:
        p_pos = report_dir / "positions.csv"
        _to_csv_lf(res.positions.rename_axis("date").reset_index(), p_pos)
        paths["positions.csv"] = p_pos

    if res.signals is not None and not res.signals.empty:
        p_sig = report_dir / "signals.csv"
        _to_csv_lf(res.signals.rename_axis("date").reset_index(), p_sig)
        paths["signals.csv"] = p_sig

    # metrics & report
    p_metrics = report_dir / "metrics.yaml"; _dump_yaml(res.metrics or {}, p_metrics); paths["metrics.yaml"] = p_metrics
    p_report  = report_dir / "report.yaml";  _dump_yaml(res.meta or {},    p_report);  paths["report.yaml"]  = p_report

    # daily (agrégé) + net_vs_gross
    bps = None
    try:
        costs = (res.meta or {}).get("costs", {})
        if isinstance(costs, dict) and "bps" in costs:
            bps = float(costs["bps"])
    except Exception:
        bps = None
    daily = _daily_from_trades(tr, bps)
    p_daily = report_dir / "trades_daily.csv"; _to_csv_lf(daily, p_daily); paths["trades_daily.csv"] = p_daily

    dng, mult_net, mult_gross, win, tot_turn = _daily_net_vs_gross(ec, daily)
    p_dng = report_dir / "daily_net_vs_gross.csv"; _to_csv_lf(dng, p_dng); paths["daily_net_vs_gross.csv"] = p_dng

    # résumé console
    ups = int((dng["rnet"] > 0).sum()) if not dng.empty else 0
    downs = int((dng["rnet"] < 0).sum()) if not dng.empty else 0
    tot = ups + downs
    winpct_str = f"{100.0*ups/tot:0.1f} %" if tot else "N/A"
    tot_cost = float(daily["cost"].sum())
    print("")
    print(f"----- RÉSUMÉ ({report_dir.as_posix()}) -----")
    print(f"Jours d'equity (retours calculés) : {len(ec)}")
    print(f"Up/Down (net): {ups}/{downs}  |  Win%: {winpct_str}")
    print(f"Turnover total: {tot_turn:0.3f}")
    print(f"Coût total    : {tot_cost:0.3%}")
    print(f"Perf nette    : {mult_net-1:0.3%}")
    print(f"Perf brute*   : {mult_gross-1:0.3%}")
    print(f"Écart coûts   : {mult_gross - mult_net:0.3%}")
    print("Exports :")
    for k in ["trades_daily.csv", "daily_net_vs_gross.csv"]:
        if k in paths: print(f" - {paths[k].as_posix()}")

    return paths


# ========= Intégration moteur =========

def _try_engine_run(cfg: dict) -> BacktestV3Result:
    try:
        from oro.engine.backtest_v3 import run as engine_run  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Impossible d'importer le moteur v3 (oro.engine.backtest_v3:run). "
            "Vérifie PYTHONPATH et la présence des fichiers."
        ) from e

    res = engine_run(cfg)
    return BacktestV3Result(
        equity_curve=res["equity_curve"],
        trades=res["trades"],
        metrics=res.get("metrics", {}),
        meta=res.get("meta", {}),
        prices=res.get("prices"),
        returns=res.get("returns"),
        positions=res.get("positions"),
        signals=res.get("signals"),
    )


# ========= Entrée publique =========

def run_backtest_v3(config_path: Path, report_dir: Path) -> Dict[str, Path]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    result = _try_engine_run(cfg)
    paths = _write_artifacts(result, Path(report_dir))
    n_days = result.meta.get("date_range", {}).get("n_days", "?")
    u_size = result.meta.get("universe_size", "?")
    print(f"[OK] Backtest v3 terminé. Dossier: {Path(report_dir).resolve()} | Jours: {n_days} | Actifs: {u_size}")
    return paths


# ========= CLI =========

def _cli():
    ap = argparse.ArgumentParser(description="Runner Backtest v3")
    ap.add_argument("--config", required=True, type=Path, help="YAML de config (v3)")
    ap.add_argument("--report-dir", required=True, type=Path, help="Dossier de sortie")
    args = ap.parse_args()
    run_backtest_v3(args.config, args.report_dir)

if __name__ == "__main__":
    sys.exit(_cli() or 0)
