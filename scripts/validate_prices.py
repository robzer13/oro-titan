import sys, pandas as pd

def main(p):
    df = pd.read_parquet(p)
    
    # Normaliser colonnes en minuscules
    df.columns = [str(c).lower() for c in df.columns]
    
    # Normalisation date
    if "date" not in df.columns:
        raise SystemExit("[validate] ERROR: no 'date' column present.")
    
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df = df.sort_values(["ticker","date"]) if "ticker" in df.columns else df.sort_values("date")
    
    n = len(df)
    tickers = df["ticker"].astype(str).nunique() if "ticker" in df.columns else 0
    dmin = df["date"].min()
    dmax = df["date"].max()
    print(f"[validate] rows={n:,} tickers={tickers} dates=[{dmin} .. {dmax}]")
    
    cols = ["open","high","low","close","adj_close","volume"]
    for c in cols:
        if c in df.columns:
            nn = df[c].notna().sum()
            pct = 100.0 * nn / n if n else 0
            print(f"  - {c:9s}: non-null={nn:,} ({pct:.1f}%)")
    
    # Top 20 tickers par non-null close
    if "ticker" in df.columns and "close" in df.columns:
        s = (df.groupby("ticker", observed=False)["close"]
               .apply(lambda x: x.notna().sum())
               .sort_values(ascending=False).head(20))
        print("\n[top20 tickers by non-null close]\n", s)
    
    all_close_nan = ("close" in df.columns) and (df["close"].notna().sum() == 0)
    if all_close_nan:
        print("\n[validate] ERROR: all `close` values are NaN.")
        sys.exit(2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_prices.py <prices.parquet>")
        sys.exit(1)
    main(sys.argv[1])
