# -*- coding: utf-8 -*-
# src/oro/engine/backtest_v3.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Contrat minimal attendu par run_backtest_v3 :
      Retourne un dict avec AU MOINS :
        - 'equity_curve': DataFrame colonnes ['date','equity']
        - 'trades'      : DataFrame colonnes ['date','ticker','w_prev','w_new','turnover_piece'] (+ 'cost' si dispo)
        - 'metrics'     : dict (ex: {'cagr':..., 'vol':..., 'sharpe':..., 'max_drawdown':...})
        - 'meta'        : dict (idéalement meta['costs']['bps'])

      Optionnels (recommandés):
        - 'prices'   : DataFrame index=date, colonnes=tickers
        - 'returns'  : DataFrame index=date, colonnes=tickers
        - 'positions': DataFrame index=date, colonnes=tickers (poids portés)
        - 'signals'  : DataFrame index=date, colonnes=tickers (scores/signaux)
    Adapte librement cette fonction pour brancher TON moteur réel.
    """
    # -- lecture config basique (avec défauts pour le stub) --
    start = str(cfg.get("start", "2024-03-28"))
    end   = str(cfg.get("end",   "2024-04-12"))
    dates = pd.date_range(start, end, freq="B")
    if len(dates) == 0:
        raise ValueError("La configuration produit 0 jour ouvré. Vérifie 'start' / 'end'.")

    rng = np.random.default_rng(int(cfg.get("seed", 123)))
    tickers = list(cfg.get("universe", ["AAA", "BBB", "CCC"]))[: max(2, int(cfg.get("universe_size", 10))//5) or 2]

    # -- équity simulée (remplacer par ton PnL réel) --
    daily_r = rng.normal(loc=0.0004, scale=0.0035, size=len(dates))
    equity0 = float(cfg.get("equity_start", 1.0))
    equity = equity0 * np.cumprod(1.0 + daily_r)
    equity_curve = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "equity": equity.astype(float)})

    # -- trades factices : 2 rebalances --
    trades = pd.DataFrame(
        {
            "date":   [dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")],
            "ticker": [tickers[0], tickers[-1]],
            "w_prev": [0.00, 0.10],
            "w_new":  [0.10, 0.20],
            "turnover_piece": [1.0, 2.0],
            # "cost": [np.nan, np.nan],  # si tu calcules les coûts par trade
        }
    )

    # -- prix / rendements factices (forme (n_days, n_tickers)) --
    n_days, n_t = len(dates), len(tickers)
    shocks = rng.normal(loc=0.0003, scale=0.01, size=(n_days, n_t))
    price_levels = np.cumprod(1.0 + shocks, axis=0)
    prices = pd.DataFrame(price_levels, index=dates, columns=tickers).astype(float)
    returns = prices.pct_change().fillna(0.0)

    # -- positions factices (poids ~softmax de signaux) --
    raw_signals = rng.normal(loc=0.0, scale=1.0, size=(n_days, n_t))
    signals = pd.DataFrame(raw_signals, index=dates, columns=tickers)
    expw = np.exp(signals.values)
    positions_w = expw / expw.sum(axis=1, keepdims=True)
    positions = pd.DataFrame(positions_w, index=dates, columns=tickers)

    # -- métriques et méta d'exemple --
    metrics = {
        "cagr": -0.10,
        "vol":  0.20,
        "sharpe": -0.5,
        "max_drawdown": -0.06,
        "version": "v3-demo",
    }
    meta = {
        "date_range": {"start": start, "end": end, "n_days": int(len(dates))},
        "universe_size": int(cfg.get("universe_size", len(tickers))),
        "universe": tickers,
        "costs": {"bps": float(cfg.get("costs", {}).get("bps", 5.0))},
        "strategy": cfg.get("strategy"),
        "signals_setup":  cfg.get("signals"),
        "rebalance": cfg.get("rebalance"),
        "engine": {"name": "demo_v3", "git_sha": cfg.get("git_sha", "unknown")},
    }

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "metrics": metrics,
        "meta": meta,
        "prices": prices,
        "returns": returns,
        "positions": positions,
        "signals": signals,
    }
