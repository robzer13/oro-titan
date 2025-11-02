#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prices_downloader.py
Un fetcher robuste d'OHLCV ajustés (Euronext/PEA) avec backends Yahoo + FMP (via RapidAPI) + cache local.
Usage (ex.) :
  python prices_downloader.py --universe universe_example.csv --start 2015-01-01 --end 2025-10-30 --out prices.parquet --backend hybrid --benchmarks "^FCHI,^STOXX50E"
Dépendances:
  pip install pandas numpy yfinance requests pyarrow python-dateutil tqdm
"""
from __future__ import annotations

import os, argparse, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import requests
except Exception:
    requests = None

from tqdm import tqdm


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def to_datestr(x) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    return pd.to_datetime(x).strftime("%Y-%m-%d")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume",
        "open":"open","high":"high","low":"low","close":"close","adj close":"adj_close","adj_close":"adj_close","volume":"volume"
    }
    out = df.copy()
    out.rename(columns={c: mapping.get(c,c) for c in out.columns}, inplace=True)
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    cols = ["open","high","low","close","adj_close","volume"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]

class YahooBackend:
    def __init__(self, auto_adjust: bool = True, actions: bool = True, timeout: int = 30):
        if yf is None:
            raise RuntimeError("yfinance introuvable. Installe: pip install yfinance")
        self.auto_adjust = auto_adjust
        self.actions = actions
        self.timeout = timeout

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        df = yf.download(
            tickers=ticker, start=start, end=end,
            auto_adjust=self.auto_adjust, actions=self.actions,
            progress=False, timeout=self.timeout
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna(how="all")
        df.index.name = "date"
        return normalize_cols(df)

class FMPBackend:
    """
    FMP via RapidAPI (clé attendue dans $RAPIDAPI_KEY).
    Endpoint: https://financialmodelingprep.p.rapidapi.com/api/v3/historical-price-full/{symbol}
    """
    def __init__(self, rapidapi_key: Optional[str] = None, host: str = "financialmodelingprep.p.rapidapi.com"):
        if requests is None:
            raise RuntimeError("requests introuvable. Installe: pip install requests")
        self.rapidapi_key = rapidapi_key or os.getenv("RAPIDAPI_KEY", None)
        self.host = host
        if not self.rapidapi_key:
            warnings.warn("RAPIDAPI_KEY non défini. FMPBackend ne fonctionnera pas sans clé.")

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if not self.rapidapi_key:
            return pd.DataFrame()
        url = f"https://{self.host}/api/v3/historical-price-full/{ticker}"
        params = {"from": start, "to": end, "serietype": "line"}
        headers = {"x-rapidapi-key": self.rapidapi_key, "x-rapidapi-host": self.host}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            hist = data.get("historical", [])
            if not hist:
                return pd.DataFrame()
            df = pd.DataFrame(hist)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            df.rename(columns={"adjClose":"adj_close"}, inplace=True)
            return normalize_cols(df)
        except Exception as e:
            warnings.warn(f"FMP fetch fail for {ticker}: {e}")
            return pd.DataFrame()

@dataclass
class Asset:
    name: str
    isin: Optional[str]
    exchange: Optional[str]
    yahoo: Optional[str]

def load_universe_csv(path: str) -> List[Asset]:
    df = pd.read_csv(path)
    req = ["name","isin","exchange","yahoo_ticker"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante dans {path}: {c}")
    assets=[]
    for _, r in df.iterrows():
        assets.append(Asset(
            name=str(r["name"]),
            isin=str(r["isin"]) if not pd.isna(r["isin"]) else None,
            exchange=str(r["exchange"]) if not pd.isna(r["exchange"]) else None,
            yahoo=str(r["yahoo_ticker"]) if not pd.isna(r["yahoo_ticker"]) else None
        ))
    return assets

def infer_yahoo_suffix(exchange: Optional[str]) -> str:
    if not exchange: return ""
    ex = exchange.upper()
    if "PARIS" in ex or ex.endswith(":PA") or ex=="EPA" or ex=="EURONEXT PARIS": return ".PA"
    if "AMSTERDAM" in ex or ex.endswith(":AS") or ex=="AMS" or ex=="EURONEXT AMSTERDAM": return ".AS"
    if "MILAN" in ex or ex.endswith(":MI") or ex=="MIL" or ex=="EURONEXT MILAN": return ".MI"
    if "BRUSSELS" in ex or ex.endswith(":BR") or ex=="BRU" or ex=="EURONEXT BRUSSELS": return ".BR"
    if "LISBON" in ex or ex.endswith(":LS") or ex=="LIS" or ex=="EURONEXT LISBON": return ".LS"
    return ""

def build_ticker_list(assets):
    tickers = []
    for a in assets:
        if getattr(a, "yahoo", None):
            t = str(a.yahoo).strip()
            if t:
                tickers.append(t)
    # dédoublonnage
    out, seen = [], set()
    for t in tickers:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


class PriceFetcher:
    def __init__(self, cache_dir: str = "cache/prices", backend: str = "yahoo"):
        self.cache_dir = cache_dir
        ensure_dir(self.cache_dir)
        self.backend_name = backend.lower()
        if self.backend_name=="yahoo":
            self.backend = YahooBackend()
        elif self.backend_name=="fmp":
            self.backend = FMPBackend()
        elif self.backend_name=="hybrid":
            self.backend = None
        else:
            raise ValueError("backend invalide: {yahoo,fmp,hybrid}")

    def _cache_path(self, ticker: str, start: str, end: str) -> str:
        safe = ticker.replace("^","_^").replace("/","_")
        return os.path.join(self.cache_dir, f"{safe}_{start}_{end}.parquet")

    def fetch_one(self, ticker: str, start: str, end: str, use_cache: bool=True) -> pd.DataFrame:
        start, end = to_datestr(start), to_datestr(end)
        cp = self._cache_path(ticker, start, end)
        if use_cache and os.path.exists(cp):
            try:
                return pd.read_parquet(cp)
            except Exception:
                pass
        if self.backend_name=="hybrid":
            for b in [YahooBackend(), FMPBackend()]:
                df = b.fetch(ticker, start, end)
                if not df.empty:
                    df.to_parquet(cp); return df
            return pd.DataFrame()
        else:
            df = self.backend.fetch(ticker, start, end)
            if not df.empty: df.to_parquet(cp)
            return df

    def fetch_many(self, tickers: List[str], start: str, end: str, use_cache: bool=True, min_rows: int=100) -> Dict[str,pd.DataFrame]:
        out={}
        for t in tqdm(tickers, desc=f"Fetching {self.backend_name}"):
            if not isinstance(t,str) or not t.strip(): continue
            df = self.fetch_one(t.strip(), start, end, use_cache=use_cache)
            if df is None or df.empty or len(df)<min_rows: continue
            out[t]=df
        return out

    @staticmethod
    def merge_long(dfs: Dict[str,pd.DataFrame]) -> pd.DataFrame:
        rows=[]
        for t, df in dfs.items():
            x=df.reset_index(); x["ticker"]=t; rows.append(x)
        if not rows:
            return pd.DataFrame(columns=["date","ticker","open","high","low","close","adj_close","volume"])
        longdf = pd.concat(rows, ignore_index=True)
        cols = ["date","ticker","open","high","low","close","adj_close","volume"]
        return longdf[cols].sort_values(["ticker","date"])

    @staticmethod
    def merge_wide(dfs: Dict[str,pd.DataFrame]) -> pd.DataFrame:
        frames=[]
        for t, df in dfs.items():
            x=df.copy()
            x.columns = pd.MultiIndex.from_product([x.columns, [t]])
            frames.append(x)
        if not frames: return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=str, required=True, help="CSV: name,isin,exchange,yahoo_ticker")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--backend", type=str, default="hybrid", choices=["yahoo","fmp","hybrid"])
    ap.add_argument("--out", type=str, default="prices.parquet")
    ap.add_argument("--format", type=str, default="long", choices=["long","wide"])
    ap.add_argument("--benchmarks", type=str, default="^FCHI,^STOXX50E")
    ap.add_argument("--min_rows", type=int, default=100)
    args = ap.parse_args()

    assets = load_universe_csv(args.universe)
    tickers = build_ticker_list(assets)
    if args.benchmarks:
        tickers += [t.strip() for t in args.benchmarks.split(",") if t.strip()]

    pf = PriceFetcher(cache_dir="cache/prices", backend=args.backend)
    dfs = pf.fetch_many(tickers, args.start, args.end, use_cache=True, min_rows=args.min_rows)

    outdf = pf.merge_long(dfs) if args.format=="long" else pf.merge_wide(dfs)
    ensure_dir(os.path.dirname(args.out) or ".")
    if args.out.lower().endswith(".csv"):
        outdf.to_csv(args.out, index=False if args.format=="long" else True)
    else:
        outdf.to_parquet(args.out)
    print(f"Saved {args.out} with shape {outdf.shape}")

if __name__=="__main__":
    main()
