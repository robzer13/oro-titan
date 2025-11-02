import sys, pandas as pd

NUMS = ["open","high","low","close","adj_close","volume"]

def main(inp, outp):
    df = pd.read_parquet(inp)
    
    # Normaliser colonnes en minuscules
    df.columns = [str(c).lower() for c in df.columns]
    
    if "date" not in df or "ticker" not in df:
        raise SystemExit("prices needs at least ['date','ticker',...]")
    
    df["date"] = pd.to_datetime(df["date"], utc=False)
    df.sort_values(["ticker","date"], inplace=True)
    
    # Coerce numerics
    for c in NUMS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Salvage close from adj_close if possible
    if "close" in df.columns and "adj_close" in df.columns:
        mask = df["close"].isna() & df["adj_close"].notna()
        if mask.any():
            df.loc[mask, "close"] = df.loc[mask, "adj_close"]
            print(f"[fix] filled close from adj_close on {mask.sum():,} rows")
    
    # Drop rows where all OHLC are NaN
    ohlc = [c for c in ["open","high","low","close"] if c in df.columns]
    keep = ~(df[ohlc].isna().all(axis=1))
    dropped = (~keep).sum()
    if dropped:
        print(f"[fix] dropped rows with all-NaN OHLC: {dropped:,}")
    df = df[keep]
    
    # Check fatal: 0 rows OR close still 100% NaN
    if len(df) == 0:
        print("FATAL: cannot fix prices (close still empty)")
        sys.exit(2)
    
    if "close" in df.columns and df["close"].notna().sum() == 0:
        print("FATAL: cannot fix prices (close still empty)")
        sys.exit(2)
    
    df.to_parquet(outp, index=False)
    print(f"[fix] wrote {outp} rows={len(df):,}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/fix_prices.py <in_parquet> <out_parquet>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
