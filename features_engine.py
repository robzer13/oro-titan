# oro_titan/features_engine.py — OroTitan V1.4
import numpy as np
import pandas as pd

REQ_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in prices: {missing}")
    df["date"] = pd.to_datetime(df["date"], utc=False)
    num_cols = ["open","high","low","close","adj_close","volume"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values(["ticker","date"]).drop_duplicates(["ticker","date"])
    return df

def _zscore(x: pd.Series) -> pd.Series:
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0 or np.isnan(s):
        return pd.Series(0.0, index=x.index)
    return (x - m) / s

def _compute_one(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    px = g["adj_close"]
    ret = px.pct_change()

    # MAs & cross
    sma_fast = px.rolling(20, min_periods=20).mean()
    sma_slow = px.rolling(50, min_periods=50).mean()
    cross_raw = sma_fast - sma_slow
    sma_cross = pd.Series(0.0, index=g.index)
    sma_cross[cross_raw > 0] = 1.0
    sma_cross[cross_raw < 0] = -1.0

    sma50 = px.rolling(50, min_periods=50).mean()
    sma50_slope = sma50.diff(5) / 5.0

    # RSI14
    up = ret.clip(lower=0).rolling(14, min_periods=14).mean()
    down = (-ret.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = up / (down.replace(0, 1e-12))
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12,26,9)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_diff = macd - macd_signal

    # Bollinger %B & width (20,2)
    sma20 = px.rolling(20, min_periods=20).mean()
    std20 = px.rolling(20, min_periods=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bbp = (px - lower) / ((upper - lower).replace(0, np.nan))
    bb_width = (upper - lower) / sma20

    # Donchian 20
    donch_h = g["high"].rolling(20, min_periods=20).max()
    donch_l = g["low"].rolling(20, min_periods=20).min()
    donchian_break20 = pd.Series(0.0, index=g.index)
    donchian_break20[g["close"] >= donch_h] = 1.0
    donchian_break20[g["close"] <= donch_l] = -1.0

    # ATR14 normalisé
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atr14_n = atr14 / px

    # Réalisée 30j (annualisée)
    rv_30 = ret.rolling(30, min_periods=30).std() * np.sqrt(252)

    # Momentum 12m–1m & reversal 5j
    mom_12 = px.pct_change(252)
    mom_1 = px.pct_change(21)
    mom_12_1 = mom_12 - mom_1
    rev_5 = -px.pct_change(5)

    # Volume z-score 20j
    vol_mean = g["volume"].rolling(20, min_periods=20).mean()
    vol_std = g["volume"].rolling(20, min_periods=20).std()
    vol_z20 = (g["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    out = g.copy()
    out["sma_cross"] = sma_cross
    out["sma50_slope"] = sma50_slope
    out["rsi_14"] = rsi_14
    out["macd_diff"] = macd_diff
    out["bbp"] = bbp
    out["bb_width"] = bb_width
    out["donchian_break20"] = donchian_break20
    out["atr14_n"] = atr14_n
    out["rv_30"] = rv_30
    out["mom_12_1"] = mom_12_1
    out["rev_5"] = rev_5
    out["vol_z20"] = vol_z20
    return out

def build_features(px_long: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_types(px_long)
    feats = df.groupby("ticker", group_keys=False).apply(_compute_one)

    feature_cols = [
        "sma_cross","sma50_slope","rsi_14","macd_diff","bbp","bb_width",
        "donchian_break20","atr14_n","rv_30","mom_12_1","rev_5","vol_z20"
    ]
    for c in feature_cols:
        feats[f"{c}_z"] = feats.groupby("date", group_keys=False)[c].transform(_zscore)
    return feats

# Alias attendu par le runner existant
def add_features(px_long: pd.DataFrame) -> pd.DataFrame:
    return build_features(px_long)
