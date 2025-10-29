# -*- coding: utf-8 -*-
# src/oro/engine/backtest_v2.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Contrat minimal pour oro.engine.run_backtest_v2 :
      Retourne un dict avec AU MOINS :
        - 'equity_curve': pd.DataFrame avec colonnes ['date','equity']
        - 'trades'      : pd.DataFrame avec colonnes ['date','ticker','w_prev','w_new','turnover_piece'] (+ 'cost' si dispo)
        - 'metrics'     : dict (ex: {'cagr':..., 'vol':..., 'sharpe':..., 'max_drawdown':...})
        - 'meta'        : dict (idéalement meta['costs']['bps'])

      Optionnel :
        - 'prices'  : DataFrame (index = dates, colonnes = tickers)
        - 'returns' : DataFrame (index = dates, colonnes = tickers)
    """
    # -- lecture config basique (avec défauts pour le stub) --
    start = str(cfg.get("start", "2024-03-28"))
    end   = str(cfg.get("end",   "2024-04-12"))
    dates = pd.date_range(start, end, freq="B")
    if len(dates) == 0:
        raise ValueError("La configuration produit 0 jour ouvré. Vérifie 'start' / 'end'.")

    # -- génère une petite trajectoire d'équity (remplacer par ton moteur) --
    rng = np.random.default_rng(int(cfg.get("seed", 123)))
    daily_r = rng.normal(loc=0.0005, scale=0.003, size=len(dates))  # ~0.05%/jour en moyenne
    equity0 = float(cfg.get("equity_start", 1.0))
    equity = equity0 * np.cumprod(1.0 + daily_r)

    equity_curve = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "equity": equity.astype(float)}
    )

    # -- deux rebalances factices pour la démo (remplacer par les vrais trades) --
    tickers = ["AAA", "BBB"]
    trades = pd.DataFrame(
        {
            "date":   [dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")],
            "ticker": [tickers[0], tickers[1]],
            "w_prev": [0.00, 0.10],
            "w_new":  [0.10, 0.20],
            "turnover_piece": [1.0, 2.0],
            # "cost": [np.nan, np.nan],  # décommente si tu calcules les coûts par trade
        }
    )

    # -- séries de prix/rendements factices optionnelles (remplacer par tes données) --
    # IMPORTANT: axis=0 pour garder la forme (n_days, n_tickers), sinon np.cumprod aplatit.
    n_days, n_t = len(dates), len(tickers)
    shocks = rng.normal(loc=0.0003, scale=0.01, size=(n_days, n_t))
    price_rel = 1.0 + shocks
    price_levels = np.cumprod(price_rel, axis=0)  # <-- FIX: axis=0
    prices = pd.DataFrame(price_levels, index=dates, columns=tickers).astype(float)
    returns = prices.pct_change().fillna(0.0)

    # -- métriques et méta d'exemple (mets-y tes vraies valeurs) --
    metrics = {
        "cagr": -0.10,
        "vol":  0.20,
        "sharpe": -0.5,
        "max_drawdown": -0.06,
    }
    meta = {
        "date_range": {"start": start, "end": end, "n_days": int(len(dates))},
        "universe_size": int(cfg.get("universe_size", 10)),
        "costs": {"bps": float(cfg.get("costs", {}).get("bps", 5.0))},
        "strategy": cfg.get("strategy"),
        "signals":  cfg.get("signals"),
        "rebalance": cfg.get("rebalance"),
    }

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "metrics": metrics,
        "meta": meta,
        "prices": prices,
        "returns": returns,
    }
