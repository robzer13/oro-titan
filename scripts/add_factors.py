import sys
import pandas as pd

def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    req = ["date","ticker","close"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=False)
    out.sort_values(["ticker","date"], inplace=True)

    # Drop rows without usable close before computing factors
    if "close" in out.columns:
        out = out[out["close"].notna()].copy()

    # Daily returns without implicit ffill
    out["ret"] = out.groupby("ticker", observed=False)["close"].pct_change(fill_method=None)

    # Factors (use transform to keep alignment). Looser min_periods for earlier availability.
    out["momentum_63d"] = out.groupby("ticker", observed=False)["close"] \
                             .transform(lambda s: (s / s.shift(63)) - 1)
    out["volatility_63d"] = out.groupby("ticker", observed=False)["ret"] \
                               .transform(lambda s: s.rolling(63, min_periods=21).std())

    # Drop rows where both factors are NaN (keep rows if at least one factor exists)
    before = len(out)
    out = out.dropna(subset=["momentum_63d","volatility_63d"], how="all")
    after = len(out)

    # Light summary
    tickers = out["ticker"].nunique()
    dmin = out["date"].min()
    dmax = out["date"].max()
    print(f"[add_factors] tickers={tickers}, rows_before={before}, rows_after={after}, "
          f"date_range=[{dmin} -> {dmax}]")

    return out

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_factors.py <in_parquet> <out_parquet>")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]
    df = pd.read_parquet(inp)
    out = add_factors(df)
    out.to_parquet(outp, index=False)
    print(f"OK -> wrote {outp} (rows: {len(out):,})")
