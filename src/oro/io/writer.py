# oro/io/writer.py
def write_equity_curve(df, path):
    out = df.copy()
    out.columns = ["date", "equity"]         # force l’ordre et les noms
    out["date"] = out["date"].astype(str)
    out["equity"] = out["equity"].astype(float)
    out.to_csv(path, index=False, lineterminator="\n")  # évite \r dans l’entête
