# src/oro/engine/run_backtest.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_datetime64_any_dtype as is_dt

# ---------- utilitaires ----------

def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config introuvable: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _to_jsonable(obj: Any) -> Any:
    """
    Convertit tout ce qui n'est pas sérialisable YAML "safe" (Timestamp, numpy types…).
    """
    if obj is None:
        return None
    # pandas / numpy scalaires
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isfinite(v):
            return v
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # datetime-like
    if isinstance(obj, (pd.Timestamp,)):
        # on force la date ISO (YYYY-MM-DD) si l'heure == 00:00:00,
        # sinon ISO complet
        if obj.tz is not None:
            obj = obj.tz_convert(None).tz_localize(None)
        if obj.hour == 0 and obj.minute == 0 and obj.second == 0 and obj.microsecond == 0:
            return obj.date().isoformat()
        return obj.isoformat()

    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)

    # python std
    import datetime as pydt
    if isinstance(obj, (pydt.datetime,)):
        return obj.isoformat()
    if isinstance(obj, (pydt.date,)):
        return obj.isoformat()
    if isinstance(obj, (pydt.time,)):
        return obj.isoformat()

    # conteneurs
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    return obj  # on tente la sérialisation native


def _safe_yaml_dump(data: Dict[str, Any], path: Union[str, Path]) -> None:
    clean = _to_jsonable(data)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)


def _resolve_prices_path(cfg: Dict[str, Any], cfg_path: Union[str, Path]) -> Path:
    """
    Résout data.prices.path en testant plusieurs bases :
      - valeur absolue telle quelle
      - relative au dossier de la config
      - relative au CWD (repo root si on lance depuis la racine)
      - relative au parent du dossier config (souvent la racine du repo)
    Lève une erreur explicite listant les chemins testés.
    """
    try:
        prices_entry = cfg["data"]["prices"]["path"]
    except Exception:
        raise FileNotFoundError("Chemin des prix introuvable dans la config (attendu: data.prices.path)")

    rel = Path(str(prices_entry))
    tried: List[Path] = []

    # 1) absolu
    if rel.is_absolute():
        tried.append(rel)
        if rel.exists():
            return rel

    cfg_dir = Path(cfg_path).resolve().parent
    bases: List[Path] = [
        cfg_dir,                 # 2) dossier de la config (ex: configs/)
        Path.cwd(),              # 3) répertoire courant (souvent racine du repo si on lance depuis là)
        cfg_dir.parent,          # 4) parent du dossier config (souvent la racine)
    ]
    # dédupli simple en conservant l'ordre
    seen = set()
    uniq_bases = []
    for b in bases:
        if b not in seen:
            uniq_bases.append(b)
            seen.add(b)

    for base in uniq_bases:
        cand = (base / rel).resolve()
        tried.append(cand)
        if cand.exists():
            return cand

    tried_txt = "\n  - " + "\n  - ".join(str(p) for p in tried)
    raise FileNotFoundError(f"Fichier prix introuvable. Chemins testés:{tried_txt}")


def _read_prices_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Lit un CSV long (date,ticker,close[,volume]) -> DataFrame avec colonnes:
      date (datetime64[ns]), ticker (str), close (float), volume (optionnel)
    """
    p = Path(path)
    df = pd.read_csv(p, dtype={"ticker": "string"}, parse_dates=["date"])
    if "close" not in df.columns or "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError("Le CSV de prix doit contenir au minimum: date,ticker,close")
    # cast numerique
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    # ordonner
    df = df.sort_values(["ticker", "date"])
    return df


def _prices_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    wide = df_long.pivot(index="date", columns="ticker", values="close").sort_index()
    # assure datetime index
    if not is_dt(wide.index):
        wide.index = pd.to_datetime(wide.index)
    return wide


def _compute_simple_metrics(equity: pd.Series) -> Dict[str, Any]:
    """
    equity: série indexée par date, base 1.0 en t0
    """
    out: Dict[str, Any] = {}
    if equity.empty:
        return {"cagr": None, "vol": None, "sharpe": None, "max_drawdown": None}

    # périodes annuelles (252 jours)
    daily = equity.pct_change().dropna()
    ann = 252.0

    # CAGR
    n_days = len(equity.index)
    if n_days > 1:
        total_ret = float(equity.iloc[-1] / equity.iloc[0]) - 1.0
        years = max(1e-9, n_days / ann)
        cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0
        out["cagr"] = float(cagr)
    else:
        out["cagr"] = None

    # vol
    vol = float(daily.std() * np.sqrt(ann)) if not daily.empty else None
    out["vol"] = vol

    # sharpe simplifié (rf=0)
    sharpe = float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(ann)) if not daily.empty else None
    out["sharpe"] = sharpe

    # max drawdown
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = float(dd.min()) if not dd.empty else None
    out["max_drawdown"] = mdd

    return out


# ---------- moteur principal (baseline equal-weight buy&hold) ----------

def run_backtest(config_path: Union[str, Path], report_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Baseline déterministe :
      - lit un fichier long date,ticker,close
      - calcule les rendements journaliers par actif
      - portefeuille égal-pondéré (EW) sur tout l'univers dispo à chaque date (moyenne des retours)
      - équity curve base 1.0
      - exporte: returns_wide.csv, equity_curve.csv, metrics.yaml, report.yaml
    """
    cfg_path = Path(config_path).resolve()
    outdir = Path(report_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(cfg_path)

    # Résolution robuste du chemin des prix
    prices_path = _resolve_prices_path(cfg, cfg_path)

    # lecture et pivot
    df_long = _read_prices_csv(prices_path)
    wide = _prices_long_to_wide(df_long)

    # rendements simples
    rets = wide.pct_change().replace([np.inf, -np.inf], np.nan)

    # portefeuille EW: moyenne sur les colonnes dispos chaque jour
    port_rets = rets.mean(axis=1, skipna=True)

    # equity curve (base 1.0 au premier jour observé)
    equity = (1.0 + port_rets.fillna(0.0)).cumprod()
    if not equity.empty:
        # force premier point = 1.0 (si NaN initial)
        first_idx = equity.index[0]
        equity.loc[first_idx] = 1.0
        equity = equity.sort_index()

    # ---------- exports ----------
    # returns_wide.csv
    df_ret = rets.copy()
    df_ret.insert(0, "date", df_ret.index)  # date en première colonne
    df_ret["date"] = df_ret["date"].dt.date  # évite Timestamp -> YAML/CSV exotique
    df_ret.to_csv(outdir / "returns_wide.csv", index=False)

    # equity_curve.csv
    eq = pd.DataFrame({"date": equity.index.date, "equity": equity.values})
    eq.to_csv(outdir / "equity_curve.csv", index=False)

    # metrics.yaml
    metrics = _compute_simple_metrics(equity)
    _safe_yaml_dump(metrics, outdir / "metrics.yaml")

    # report.yaml
    report: Dict[str, Any] = {
        "config_used": str(cfg_path),
        "prices_path": str(prices_path),
        "date_range": {
            "start": eq["date"].min() if not eq.empty else None,
            "end": eq["date"].max() if not eq.empty else None,
            "n_days": int(len(eq)) if not eq.empty else 0,
        },
        "universe": sorted([str(c) for c in rets.columns]),
        "metrics": metrics,
        "artifacts": {
            "returns_wide_csv": str(outdir / "returns_wide.csv"),
            "equity_curve_csv": str(outdir / "equity_curve.csv"),
            "metrics_yaml": str(outdir / "metrics.yaml"),
        },
        "notes": "Baseline égal-pondéré sur l'univers complet; pas de turnover ni coûts.",
    }
    _safe_yaml_dump(report, outdir / "report.yaml")

    return {
        "report_dir": str(outdir),
        "metrics": metrics,
        "n_assets": int(rets.shape[1]),
        "n_days": int(rets.shape[0]),
    }
