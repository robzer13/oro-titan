# fix_parquet_columns.py
import pandas as pd
import sys

inp = "prices.parquet"
out = "prices_flat.parquet"
if len(sys.argv) > 1: inp = sys.argv[1]
if len(sys.argv) > 2: out = sys.argv[2]

df = pd.read_parquet(inp)

def flat(c):
    if isinstance(c, tuple):
        return c[0] if len(c)==1 or (len(c)>1 and (c[1] is None or str(c[1]).strip()=="")) else "_".join(str(x) for x in c if str(x).strip()!="")
    return c

df.columns = [flat(c) for c in df.columns]
df.columns = [str(c).strip().lower() for c in df.columns]
df = df.rename(columns={"adjclose":"adj_close","adjusted_close":"adj_close","adj close":"adj_close"})
if "adj_close" not in df.columns and "close" in df.columns:
    df["adj_close"] = df["close"]

df.to_parquet(out, index=False)
print(f"Saved {out} with columns: {list(df.columns)}")
