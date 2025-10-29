# src/oro/engine/backtest_v2.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# === Exemple: ton moteur interne (à remplacer par le vrai) ===
# from .mon_moteur import OroBacktester

def _to_equity_df(df: pd.DataFrame) -> pd.DataFrame:
    need = {"date", "equity"}
    if not need.issubset(df.columns):
        raise ValueError(f"equity_curve doit contenir {need}, trouvé {list(df.columns)}")
    out = df.copy()
    out["date"] = out["date"].astype(str)
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    return out.dropna(subset=["equity"])

def _to_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    need = {"date", "ticker", "w_prev", "w_new", "turnover_piece"}
    if not need.issubset(df.columns):
        raise ValueError(f"trades doit contenir {need}, trouvé {list(df.columns)}")
    out = df.copy()
    out["date"] = out["date"].astype(str)
    for c in ["w_prev","w_new","turnover_piece","cost"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # assure l’ordre des colonnes + colonne cost (optionnelle)
    if "cost" not in out.columns:
        out["cost"] = np.nan
    return out[["date","ticker","w_prev","w_new","turnover_piece","cost"]]

def _norm_prices(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    out = df.copy()
    out.index.name = "date"
    return out

def _norm_returns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    out = df.copy()
    out.index.name = "date"
    return out

def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Contrat attendu par oro.engine.run_backtest_v2:
      return {
        'equity_curve': DataFrame[cols: date, equity],
        'trades':       DataFrame[cols: date, ticker, w_prev, w_new, turnover_piece, (cost?)],
        'metrics':      dict,
        'meta':         dict (incl. costs.bps si possible),
        # optionnels:
        'prices':  DataFrame (index=date, cols=tickers),
        'returns': DataFrame (index=date, cols=tickers),
      }
    """

    # === Appelle ton moteur réel ici (exemples, à remplacer) ===
    # bt = OroBacktester(cfg)
    # res = bt.run()  # <-- DOIT fournir equity, trades, metrics, meta, (prices, returns)
    #
    # Exemple “factice”: je montre comment mapper si tu as déjà eq/tr/metrics/meta.
    # REMPLACE ce bloc par ton vrai appel.
    dates = pd.date_range(cfg["start"], cfg["end"], freq="B")
    eq = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                       "equity": (1.0 + 0.001*np.random.randn(len(dates)))\
                                 .cumprod()})
    # trades factices: 2 rebalances
    tr = pd.DataFrame({
        "date":   [dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")],
        "ticker": ["AAA","BBB"],
        "w_prev": [0.00, 0.10],
        "w_new":  [0.10, 0.20],
        "turnover_piece": [1.0, 2.0],
        # "cost": [np.nan, np.nan],  # optionnel
    })
    metrics = {"cagr": -0.12, "vol": 0.25, "sharpe": -0.5, "max_drawdown": -0.06}
    meta = {
        "date_range": {"start": cfg["start"], "end": cfg["end"], "n_days": int(len(dates))},
        "universe_size": int(cfg.get("universe_size", 10)),
        "costs": {"bps": float(cfg.get("costs", {}).get("bps", 5.0))},
        "strategy": cfg.get("strategy"), "signals": cfg.get("signals"), "rebalance": cfg.get("rebalance"),
    }
    prices = pd.DataFrame(np.cumprod(1 + 0.01*np.random.randn(len(dates), 2), axis=0),
                          index=dates, columns=["AAA","BBB"])
    returns = prices.pct_change().fillna(0.0)

    # === Normalisation stricte vers le contrat attendu ===
    equity_curve = _to_equity_df(eq)
    trades       = _to_trades_df(tr)
    prices       = _norm_prices(prices)
    returns      = _norm_returns(returns)

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "metrics": metrics,
        "meta": meta,
        "prices": prices,
        "returns": returns,
    }

# === === === Adapter pour le runner v2 === === ===
# Ajoute une fonction run(cfg: dict) -> dict qui renvoie les objets attendus
# par oro.engine.run_backtest_v2 : equity_curve, trades, metrics, meta,
# et (optionnel) prices, returns.
#

# === Adapter pour le runner v2 : expose run(cfg) attendu par oro.engine.run_backtest_v2 ===
from typing import Dict, Any
import pandas as pd
import numpy as np

def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retourne un dict avec les clés minimales :
      - equity_curve: DataFrame colonnes ['date','equity']
      - trades:       DataFrame colonnes ['date','ticker','w_prev','w_new','turnover_piece'] (+ 'cost' si dispo)
      - metrics:      dict
      - meta:         dict (incluant idealement meta['costs']['bps'])
    Optionnels :
      - prices:  DataFrame (index dates, colonnes tickers)
      - returns: DataFrame (index dates, colonnes tickers)
    """
    start = str(cfg.get("start", "2024-03-28"))
    end   = str(cfg.get("end",   "2024-04-12"))
    dates = pd.date_range(start, end, freq="B")

    rng = np.random.default_rng(123)
    daily_r = rng.normal(0.0005, 0.003, len(dates))
    equity0 = float(cfg.get("equity_start", 1.0))
    equity = equity0 * np.cumprod(1.0 + daily_r)

    equity_curve_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "equity": equity
    })

    trades_df = pd.DataFrame({
        "date":   [dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")],
        "ticker": ["AAA", "BBB"],
        "w_prev": [0.00, 0.10],
        "w_new":  [0.10, 0.20],
        "turnover_piece": [1.0, 2.0],
        # "cost": [np.nan, np.nan],  # ajoute si tu as le coût par trade
    })

    tickers = ["AAA", "BBB"]
    prices_df = pd.DataFrame(
        np.cumprod(1.0 + rng.normal(0.0003, 0.01, size=(len(dates), len(tickers)))),
        index=dates, columns=tickers
    )
    returns_df = prices_df.pct_change().fillna(0.0)

    metrics_dict = {
        "cagr": -0.10,
        "vol":  0.20,
        "sharpe": -0.5,
        "max_drawdown": -0.06,
    }
    meta_dict = {
        "date_range": {"start": start, "end": end, "n_days": int(len(dates))},
        "universe_size": int(cfg.get("universe_size", 10)),
        "costs": {"bps": float(cfg.get("costs", {}).get("bps", 5.0))},
        "strategy": cfg.get("strategy"),
        "signals":  cfg.get("signals"),
        "rebalance": cfg.get("rebalance"),
    }

    return {
        "equity_curve": equity_curve_df,
        "trades": trades_df,
        "metrics": metrics_dict,
        "meta": meta_dict,
        "prices": prices_df,
        "returns": returns_df,
    }
# === Fin adapter ===
