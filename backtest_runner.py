#!/usr/bin/env python3
"""
OroTitan — backtest_runner.py

Run a simple cross‑sectional, top‑K, periodic‑rebalance backtest from a
prices Parquet and a scoring YAML.

Input expectations
------------------
• PRICES Parquet with at least the following columns:
    - date: datetime64[ns] (or parseable to datetime)
    - ticker: string / category
    - close: float (adjusted close preferred)
  Optional but useful:
    - any factor columns referenced in CONFIG weights

• CONFIG YAML (example)
    weights:
      rsi_14_z: 0.8
      macd_diff_z: 0.6
      mom_12_1_z: 1.0
      rev_5_z: -0.3
    options:
      standardize_non_z: true   # z‑score per day columns that do not end with "_z"
      long_only: true

CLI
---
python backtest_runner.py \
  --prices prices.parquet \
  --config scoring_config.yml \
  --top_k 8 \
  --rebalance W-FRI \
  --fees_bps 5 \
  --slip_bps 0 \
  --min_score_z -0.5 \
  --curve_out curve.csv \
  --weights_out weights_log.csv \
  --picks_out picks_log.csv

Notes
-----
• "unrecognized arguments --curve_out ..." in your previous run means
  your script did not declare those arguments. This version adds them.
• Rebalance frequency uses pandas offsets (e.g., W-FRI, M, BM, BQS-DEC...).
• Transaction costs are applied on turnover at each rebalance day only.
• Slippage is modeled as an additional per‑trade bps on turnover.

PowerShell runbook (Windows):

    cd C:\\Users\\robin\\Documents\\OroTitan_V1.4\\oro_titan

    # 1) Enrichir les facteurs
    python .\\scripts\\add_factors.py .\\prices.parquet .\\prices_with_factors.parquet

    # 2) Vérification rapide
    python -c "import pandas as pd; df=pd.read_parquet(r'.\\prices_with_factors.parquet'); print(len(df), df['ticker'].nunique(), df['date'].min(), df['date'].max()); print([c for c in df.columns if 'momentum' in c or 'volatility' in c])"

    # 3) Backtest
    New-Item -ItemType Directory -Path .\\out -ErrorAction SilentlyContinue | Out-Null

    python .\\backtest_runner.py `
      --prices .\\prices_with_factors.parquet `
      --config .\\configs\\scoring.yml `
      --top_k 8 `
      --min_score_z -0.5 `
      --rebalance W-FRI `
      --fees_bps 5 `
      --slip_bps 0 `
      --curve_out .\\out\\curve.csv `
      --weights_out .\\out\\weights_log.csv `
      --picks_out .\\out\\picks_log.csv

Additional PowerShell runbook (data repair/refresh): If validation fails, run B2 (yfinance rebuild).

    # 0) (optional) Console UTF-8 to avoid UnicodeEncodeError
    chcp 65001 > $null
    $env:PYTHONIOENCODING = "utf-8"

    cd C:\\Users\\robin\\Documents\\OroTitan_V1.4\\oro_titan

    # A) Validate existing parquet
    python .\\scripts\\validate_prices.py .\\prices.parquet

    # B1) Try to repair inplace to a new file
    python .\\scripts\\fix_prices.py .\\prices.parquet .\\prices_fixed.parquet
    python .\\scripts\\validate_prices.py .\\prices_fixed.parquet

    # B2) If still bad, rebuild from yfinance (needs internet)
    @"
    AI.PA
    DSY.PA
    ENX.PA
    ASML.AS
    "@ | Set-Content -Encoding UTF8 .\\tickers.txt
    python .\\scripts\\build_prices_yf.py .\\tickers.txt .\\prices_built.parquet 2015-01-01
    python .\\scripts\\validate_prices.py .\\prices_built.parquet

    # C) Always-works synthetic fallback
    python .\\scripts\\make_synth_prices.py .\\prices_synth.parquet
    python .\\scripts\\validate_prices.py .\\prices_synth.parquet

    # D) Factors -> use repaired/built/synth file of your choice
    python .\\scripts\\add_factors.py .\\prices_fixed.parquet .\\prices_with_factors.parquet

    # E) Backtest (choose the best input you have)
    New-Item -ItemType Directory -Path .\\out -ErrorAction SilentlyContinue | Out-Null
    python .\\backtest_runner.py `
      --prices .\\prices_with_factors.parquet `
      --config .\\configs\\scoring.yml `
      --top_k 8 `
      --min_score_z -0.5 `
      --rebalance W-FRI `
      --fees_bps 5 `
      --slip_bps 0 `
      --curve_out .\\out\\curve.csv `
      --weights_out .\\out\\weights_log.csv `
      --picks_out .\\out\\picks_log.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import fnmatch

np.seterr(all="ignore")

# ----------------------------- Utilities ---------------------------------

@dataclass
class BacktestConfig:
    prices_path: str
    config_path: str
    top_k: int = 10
    min_score_z: float = -np.inf
    rebalance: str = "W-FRI"
    fees_bps: float = 0.0
    slip_bps: float = 0.0
    start: str | None = None
    end: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    curve_out: str | None = None
    weights_out: str | None = None
    picks_out: str | None = None
    meta_version: str = "OroTitan_Data_v1.0"


# --------------------------- Loading & Prep -------------------------------

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Standardize column names
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for need in ["date", "ticker", "close"]:
        # find a matching column ignoring case
        if need not in df.columns:
            cand = None
            for c in df.columns:
                if c.lower() == need:
                    cand = c
                    break
            if cand is None:
                raise ValueError(f"Missing required column '{need}' in {path}")
            rename[cand] = need
    if rename:
        df = df.rename(columns=rename)

    # Parse date
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # Sort & set dtypes
    df["ticker"] = df["ticker"].astype("category")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Compute simple daily returns if not present
    if "ret" not in df.columns:
        df["ret"] = df.groupby("ticker")["close"].pct_change().fillna(0.0)

    return df


# --------------------- Basic factors and logging helpers --------------------

def _ensure_basic_factors(df: pd.DataFrame, options: dict) -> pd.DataFrame:
    need = {"momentum_63d","volatility_63d"}
    if need.issubset(df.columns):
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=False)
    out.sort_values(["ticker","date"], inplace=True)
    if "ret" not in out.columns:
        out["ret"] = out.groupby("ticker", observed=False)["close"].pct_change(fill_method=None)
    if "momentum_63d" not in out.columns:
        out["momentum_63d"] = out.groupby("ticker", observed=False)["close"] \
                                   .transform(lambda s: (s / s.shift(63)) - 1)
    if "volatility_63d" not in out.columns:
        out["volatility_63d"] = out.groupby("ticker", observed=False)["ret"] \
                                     .transform(lambda s: s.rolling(63, min_periods=21).std())
    return out


def _log_stage(name: str, df: pd.DataFrame):
    if df is None or len(df) == 0:
        print(f"[stage:{name}] rows=0")
        return
    dmin = pd.to_datetime(df["date"]).min()
    dmax = pd.to_datetime(df["date"]).max()
    tcount = df["ticker"].nunique() if "ticker" in df.columns else None
    print(f"[stage:{name}] rows={len(df):,} tickers={tcount} dates=[{dmin.date()} -> {dmax.date()}]")

def load_scoring_config(path: str) -> Tuple[Dict[str, float], Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    weights = cfg.get("weights", {})
    options = cfg.get("options", {})
    if not isinstance(weights, dict) or len(weights) == 0:
        raise ValueError("YAML must define a non‑empty 'weights' mapping")
    # force float weights
    weights = {str(k): float(v) for k, v in weights.items()}
    return weights, options


# --------------------------- Scoring Engine -------------------------------

def _zscore_cross_section(x: pd.Series) -> pd.Series:
    m = x.mean()
    s = x.std(ddof=0)
    if not np.isfinite(s) or s == 0:
        return pd.Series(0.0, index=x.index)
    return (x - m) / s


def compute_scores(
    df: pd.DataFrame,
    weights: Dict[str, float],
    options: Dict[str, object],
) -> pd.DataFrame:
    """Compute per‑day composite score from factor columns.

    • If a factor name ends with '_z', it is treated as already standardized.
    • Otherwise, if options.standardize_non_z is true (default), we z‑score
      cross‑sectionally per day.
    """
    df = df.copy()
    factor_cols: List[str] = list(weights.keys())

    # Verify presence
    missing = [c for c in factor_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing factor columns in PRICES parquet: " + ", ".join(missing)
        )

    standardize_non_z = bool(options.get("standardize_non_z", True))

    # Build standardized matrix per day
    for col in factor_cols:
        if col.endswith("_z"):
            df[f"std__{col}"] = df[col].astype(float)
        else:
            if standardize_non_z:
                df[f"std__{col}"] = (
                    df.groupby("date")[col].transform(_zscore_cross_section)
                )
            else:
                df[f"std__{col}"] = df[col].astype(float)

    # Composite score
    comp = np.zeros(len(df))
    for col, w in weights.items():
        comp += df[f"std__{col}"] * float(w)
    df["score"] = comp

    return df


# --------------------------- Rebalance Logic ------------------------------

def align_rebalance_dates(all_dates: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    # Generate schedule in calendar time, then align to last available trading date <= target
    start, end = all_dates.min(), all_dates.max()
    sched = pd.date_range(start=start, end=end, freq=rule)
    if len(sched) == 0 or sched[-1] != end:
        # ensure last date is a rebalance boundary
        sched = sched.union([end])

    # align each scheduled date to the last available <= date
    a = all_dates.sort_values().unique()
    aligned = []
    arr = a.view("i8")  # ns to int for searchsorted
    for d in sched:
        pos = arr.searchsorted(d.value, side="right") - 1
        if pos >= 0:
            aligned.append(pd.Timestamp(arr[pos]))
    aligned = pd.DatetimeIndex(sorted(set(aligned)))
    return aligned


# ----------------------------- Backtester ---------------------------------

def run_backtest(
    df: pd.DataFrame,
    top_k: int,
    min_score_z: float,
    rebalance_rule: str,
    fees_bps: float,
    slip_bps: float,
    start: str | None,
    end: str | None,
    meta_version: str = "OroTitan_Data_v1.0",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: (curve_df, weights_log, picks_log)
    """
    # 8. Filter date range (CLI dates override)
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    df = df.copy()
    _log_stage("after_date_range", df)

    # universe per day: drop NaN scores
    df = df.dropna(subset=["score"]).copy()
    _log_stage("after_dropna_score", df)

    all_dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    if len(all_dates) == 0:
        raise ValueError("No data after filtering. Last stage: after_dropna_score. Check start/end, universe, or inputs.")

    rebal_dates = align_rebalance_dates(all_dates, rebalance_rule)

    # Precompute daily returns table [date x ticker]
    ret_tbl = df.pivot(index="date", columns="ticker", values="ret").sort_index()
    score_tbl = df.pivot(index="date", columns="ticker", values="score").sort_index()

    # Containers
    equity = 1.0
    curve_rows = []
    weights_rows = []
    picks_rows = []

    prev_w = pd.Series(0.0, index=ret_tbl.columns)

    for i, d in enumerate(all_dates):
        day_ret = 0.0
        gross_ret = 0.0
        cost = 0.0

        if d in rebal_dates:
            # select top_k by score (with threshold)
            s = score_tbl.loc[d].dropna()
            if np.isfinite(min_score_z):
                s = s[s >= min_score_z]
            picks = s.sort_values(ascending=False).head(top_k)

            # equal‑weight portfolio among picks
            if len(picks) == 0:
                new_w = pd.Series(0.0, index=prev_w.index)
            else:
                new_w = pd.Series(0.0, index=prev_w.index)
                new_w.loc[picks.index] = 1.0 / len(picks)

            # transaction costs on turnover
            turnover = (new_w - prev_w).abs().sum()
            tc = (fees_bps + slip_bps) / 1e4 * turnover
            cost = tc

            # save picks log
            for rank, (tic, sc) in enumerate(picks.items(), start=1):
                picks_rows.append({"date": d, "rank": rank, "ticker": tic, "score": sc})

            prev_w = new_w

        # apply daily return
        ret_row = ret_tbl.loc[d].fillna(0.0)
        gross_ret = float((prev_w * ret_row).sum())
        day_ret = gross_ret - cost
        equity *= (1.0 + day_ret)

        # curve
        curve_rows.append({
            "date": d,
            "equity": equity,
            "daily_return": day_ret,
            "gross_return": gross_ret,
            "cost": cost,
        })

        # weights log (end‑of‑day weights, after rebalance if any)
        nonzero = prev_w[prev_w != 0.0]
        for tic, w in nonzero.items():
            weights_rows.append({"date": d, "ticker": tic, "weight": float(w)})

    curve_df = pd.DataFrame(curve_rows).sort_values("date")
    weights_log = pd.DataFrame(weights_rows).sort_values(["date", "ticker"])\
        .reset_index(drop=True)
    picks_log = pd.DataFrame(picks_rows).sort_values(["date", "rank"])\
        .reset_index(drop=True)
    
    # Add Meta_Version as first column
    if len(curve_df) > 0:
        curve_df.insert(0, "Meta_Version", meta_version)
    if len(weights_log) > 0:
        weights_log.insert(0, "Meta_Version", meta_version)
    if len(picks_log) > 0:
        picks_log.insert(0, "Meta_Version", meta_version)

    return curve_df, weights_log, picks_log


# ------------------------------- Main -------------------------------------

def parse_args(argv: List[str]) -> BacktestConfig:
    p = argparse.ArgumentParser(
        description="OroTitan Top‑K periodic rebalance backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prices", dest="prices_path", type=str, required=True,
                   help="Path to prices parquet")
    p.add_argument("--config", dest="config_path", type=str, required=True,
                   help="Path to scoring YAML")
    p.add_argument("--top_k", type=int, default=10, help="Number of assets to hold")
    p.add_argument("--min_score_z", type=float, default=-np.inf,
                   help="Minimum composite score to be eligible on a rebalance day")
    p.add_argument("--rebalance", type=str, default="W-FRI",
                   help="Pandas offset alias for rebalance frequency (e.g. W-FRI, M, BM)")
    p.add_argument("--fees_bps", type=float, default=0.0, help="Transaction fee bps")
    p.add_argument("--slip_bps", type=float, default=0.0, help="Slippage bps per rebalance")
    p.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    # New: explicit overrides for config dates
    p.add_argument("--date_start", type=str, default=None, help="Override config/options date_start")
    p.add_argument("--date_end", type=str, default=None, help="Override config/options date_end")

    # New: output paths
    p.add_argument("--curve_out", type=str, default=None, help="CSV path for equity curve")
    p.add_argument("--weights_out", type=str, default=None, help="CSV path for weights log")
    p.add_argument("--picks_out", type=str, default=None, help="CSV path for picks log")
    p.add_argument("--meta_version", type=str, default="OroTitan_Data_v1.0",
                   help="Metadata version string to include in output CSVs")

    args = p.parse_args(argv)

    return BacktestConfig(
        prices_path=args.prices_path,
        config_path=args.config_path,
        top_k=args.top_k,
        min_score_z=args.min_score_z,
        rebalance=args.rebalance,
        fees_bps=args.fees_bps,
        slip_bps=args.slip_bps,
        start=args.start,
        end=args.end,
        date_start=args.date_start,
        date_end=args.date_end,
        curve_out=args.curve_out,
        weights_out=args.weights_out,
        picks_out=args.picks_out,
        meta_version=args.meta_version,
    )


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    cfg = parse_args(argv)

    # Load
    prices = load_prices(cfg.prices_path)
    weights, options = load_scoring_config(cfg.config_path)

    # 1. Initial stage log
    _log_stage("initial", prices)

    # 2. Universe include patterns (optional) from YAML
    uni = options.get("universe", {}) if isinstance(options, dict) else {}
    include_patterns = uni.get("include") if isinstance(uni, dict) else None
    if include_patterns:
        all_tickers = prices["ticker"].astype(str).unique().tolist()
        keep = set()
        for pat in include_patterns:
            for t in all_tickers:
                if fnmatch.fnmatch(t, pat):
                    keep.add(t)
        prices = prices[prices["ticker"].astype(str).isin(sorted(keep))].copy()
    
    # 3. After universe include stage log
    _log_stage("after_universe_include", prices)

    # 4. QA gates on universe size and coverage
    uniq = prices["ticker"].nunique() if "ticker" in prices.columns else 0
    if uniq < 3:
        print("[runner] ERROR: universe too small (<3 tickers). Check options.universe.include or source data.")
        return 2
    if len(prices) < 500 or (prices["date"].max() - prices["date"].min()).days < 540:
        print("[runner] ERROR: insufficient coverage (<500 rows or <18 months).")
        return 2

    # 5. Ensure basic factors are present
    prices = _ensure_basic_factors(prices, options)
    
    # 6. Loaded+factors stage log
    _log_stage("loaded+factors", prices)

    # 7. Early close availability check
    _log_stage("after_close_check", prices)
    if "close" not in prices.columns or prices["close"].notna().sum() == 0:
        print("[runner] ERROR: no usable 'close' values after loading/factor ensure. See scripts/validate_prices.py and scripts/fix_prices.py.")
        return 2

    # 9. Score (dates will be filtered in run_backtest)
    scored = compute_scores(prices, weights, options)
    _log_stage("after_scoring", scored)

    # Run
    # Resolve date overrides: CLI has priority over config options
    start_override = cfg.date_start or cfg.start or (options.get("date_start") if isinstance(options, dict) else None)
    end_override = cfg.date_end or cfg.end or (options.get("date_end") if isinstance(options, dict) else None)

    curve, wlog, plog = run_backtest(
        scored,
        top_k=cfg.top_k,
        min_score_z=cfg.min_score_z,
        rebalance_rule=cfg.rebalance,
        fees_bps=cfg.fees_bps,
        slip_bps=cfg.slip_bps,
        start=start_override,
        end=end_override,
        meta_version=cfg.meta_version,
    )

    # Outputs
    if cfg.curve_out:
        curve.to_csv(cfg.curve_out, index=False)
    if cfg.weights_out:
        wlog.to_csv(cfg.weights_out, index=False)
    if cfg.picks_out:
        plog.to_csv(cfg.picks_out, index=False)

    # Small console summary
    if len(curve):
        total_ret = curve["equity"].iloc[-1] - 1.0
        n_days = len(curve)
        ann = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 else 0.0
        vol = curve["daily_return"].std(ddof=0) * np.sqrt(252)
        sharpe = ann / vol if vol and vol != 0 else np.nan
        maxdd = (curve["equity"].cummax() - curve["equity"]) / curve["equity"].cummax()
        maxdd = maxdd.max()
        print({
            "TotalReturn_%": round(total_ret * 100, 2),
            "CAGR_%": round(ann * 100, 2),
            "Vol_%": round(vol * 100, 2) if np.isfinite(vol) else None,
            "Sharpe": round(sharpe, 2) if np.isfinite(sharpe) else None,
            "MaxDD_%": round(maxdd * 100, 2) if np.isfinite(maxdd) else None,
        })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
