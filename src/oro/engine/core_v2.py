# src/oro/engine/core_v2.py
from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np

def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Moteur V2 attendu par le runner : retourne un dict avec les clés
    equity_curve, trades, metrics, meta (+ optionnel prices, returns).
    Remplace ce contenu par ton moteur réel si tu l'as déjà.
    """
    # --- DATES robustes (prend 'start'/'end' ou essaie des variantes) ---
    def pick(key, *alts, default=None):
        for k in (key,)+alts:
            if k in cfg: return cfg[k]
        return default
    start = pick("start", "oos_start", "oos.start", default="2024-03-28")
    end   = pick("end",   "oos_end",   "oos.end",   default="2024-04-12")

    dates = pd.date_range(start, end, freq="B")
    # Fake equity décroissante légèrement (à remplacer par ton vrai equity)
    equity = np.cumprod(1 + np.random.normal(0, 0.0, len(dates))) * 0.965
    equity_curve = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "equity": equity})

    # Deux rebalances (exemple)
    trades = pd.DataFrame({
        "date": [dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")],
        "ticker": ["AAA", "BBB"],
        "w_prev": [0.0, 0.1],
        "w_new": [0.1, 0.2],
        "turnover_piece": [1.0, 2.0],
        # "cost": [np.nan, np.nan],  # facultatif; fallback bps sinon
    })

    metrics = {"cagr": -0.0553, "vol": 0.305, "sharpe": -2.48, "max_drawdown": -0.052}
    meta = {
        "date_range": {"start": str(start), "end": str(end), "n_days": int(len(dates))},
        "universe_size": int(cfg.get("universe_size", 10)),
        "costs": {"bps": float(cfg.get("costs", {}).get("bps", 5.0))},
        "signals": cfg.get("signals"),
        "strategy": cfg.get("strategy"),
        "rebalance": cfg.get("rebalance"),
    }

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "metrics": metrics,
        "meta": meta,
        # Optionnel:
        # "prices": prices_df,
        # "returns": returns_df,
    }
