import sys, pandas as pd, numpy as np

rng = np.random.default_rng(42)

def synth(ticker, n=600, start="2018-01-01"):
    dates = pd.bdate_range(start=start, periods=n)
    ret = rng.normal(0.0004, 0.02, size=n)
    price = 100 * (1 + pd.Series(ret)).cumprod().values
    df = pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": price,
        "high": price * 1.005,
        "low": price * 0.995,
        "close": price,
        "adj_close": price,
        "volume": rng.integers(1e5, 5e5, size=n)
    })
    return df

if __name__ == "__main__":
    outp = sys.argv[1] if len(sys.argv) == 2 else ".\\prices_synth.parquet"
    df = pd.concat([synth("AAA.PA"), synth("BBB.PA"), synth("CCC.AS")], ignore_index=True)
    df.to_parquet(outp, index=False)
    print(f"[synth] wrote {outp} rows={len(df):,}")
