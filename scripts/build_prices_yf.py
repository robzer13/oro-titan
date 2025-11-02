import sys, pandas as pd
try:
    import yfinance as yf
except ImportError:
    raise SystemExit("pip install yfinance")

def fetch_one(tk, start="2015-01-01"):
    df = yf.download(tk, start=start, auto_adjust=False, progress=False, group_by=False)
    if df is None or df.empty:
        return None
    
    # Aplatir multiindex si besoin
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    # Renommer colonnes
    df = df.rename(columns={
        "Open":"open","High":"high","Low":"low","Close":"close",
        "Adj Close":"adj_close","Volume":"volume"
    })
    
    df.reset_index(inplace=True)
    df.rename(columns={"Date":"date"}, inplace=True)
    df["ticker"] = tk
    
    # Colonnes attendues
    standard = ["date","ticker","open","high","low","close","adj_close","volume"]
    for c in standard:
        if c not in df.columns:
            df[c] = pd.NA
    
    return df[standard]

def main(tickers_file, outp, start="2015-01-01"):
    with open(tickers_file, "r", encoding="utf-8-sig") as f:  # utf-8-sig strip BOM automatiquement
        tickers = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    
    if not tickers:
        raise SystemExit("No tickers found in file.")
    
    frames = []
    for tk in tickers:
        print(f"[yf] {tk} ...", end="", flush=True)
        sub = fetch_one(tk, start=start)
        if sub is None or sub.empty:
            print(" empty")
            continue
        print(f" ok {len(sub):,}")
        frames.append(sub)
    
    if not frames:
        raise SystemExit("No data fetched; check tickers or network.")
    
    df = pd.concat(frames, ignore_index=True).sort_values(["ticker","date"])
    
    # Garantir le schéma standard exact
    standard = ["date","ticker","open","high","low","close","adj_close","volume"]
    for c in standard:
        if c not in df.columns:
            df[c] = pd.NA
    
    # Réordonner selon le schéma standard
    df = df[standard].copy()
    
    df.to_parquet(outp, index=False)
    dmin = df["date"].min()
    dmax = df["date"].max()
    print(f"[yf] wrote {outp} rows={len(df):,} tickers={df['ticker'].nunique()} range=[{dmin} -> {dmax}]")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/build_prices_yf.py <tickers.txt> <out_parquet> [start=YYYY-MM-DD]")
        sys.exit(1)
    start = sys.argv[3] if len(sys.argv) >= 4 else "2015-01-01"
    main(sys.argv[1], sys.argv[2], start)
