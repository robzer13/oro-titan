#!/usr/bin/env python3
"""
Simple data profiler for OroTitan prices parquet.

Usage:
    python scripts/profile_prices.py .\prices_built_real.parquet .\out\prices_profile.csv

ASCII only.
"""

import sys
import pandas as pd
from pathlib import Path

STANDARD_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # normalize date
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    if "ticker" not in df.columns and "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    return df


def main(inp: str, outp: str) -> None:
    inp_path = Path(inp)
    out_path = Path(outp)

    if not inp_path.exists():
        print(f"[profile] ERROR: input file not found: {inp_path}")
        sys.exit(2)

    df = load_prices(inp_path)
    # keep only expected cols if present
    for c in STANDARD_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df = df.sort_values(["ticker", "date"])
    results = []

    for tk, g in df.groupby("ticker", observed=False):
        n = len(g)
        dmin = g["date"].min()
        dmax = g["date"].max()
        close_nn = g["close"].notna().sum()
        vol_nn = g["volume"].notna().sum()

        close_pct = (close_nn / n * 100.0) if n else 0.0
        vol_pct = (vol_nn / n * 100.0) if n else 0.0

        if n == 0:
            status = "BAD"
        elif close_pct == 0.0:
            status = "BAD"
        elif close_pct < 90.0:
            status = "PARTIAL"
        else:
            status = "OK"

        results.append(
            {
                "ticker": tk,
                "rows": n,
                "date_min": dmin.date() if pd.notna(dmin) else "",
                "date_max": dmax.date() if pd.notna(dmax) else "",
                "close_non_null_pct": round(close_pct, 1),
                "volume_non_null_pct": round(vol_pct, 1),
                "status": status,
            }
        )

    out_df = pd.DataFrame(results).sort_values(["status", "ticker"])
    out_df.to_csv(out_path, index=False)
    print(f"[profile] wrote {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/profile_prices.py <in_parquet> <out_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
