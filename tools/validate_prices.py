import pandas as pd
from sys import exit

path = "data_proc/prices_eod.csv"  # ou prices_sample_ext.csv
df = pd.read_csv(path, parse_dates=["date"])

errors = []

# 1) colonnes
expected = {"date","ticker","close","volume"}
if set(df.columns) != expected:
    errors.append(f"Colonnes attendues {expected}, trouvées {set(df.columns)}")

# 2) types
if not pd.api.types.is_datetime64_any_dtype(df["date"]):
    errors.append("date doit être datetime parsable")
if not pd.api.types.is_numeric_dtype(df["close"]):
    errors.append("close doit être numérique")
if not pd.api.types.is_numeric_dtype(df["volume"]):
    errors.append("volume doit être numérique")

# 3) doublons
dups = df.duplicated(subset=["date","ticker"]).sum()
if dups > 0:
    errors.append(f"{dups} doublon(s) (date,ticker)")

# 4) trous/zeros
na_close = df["close"].isna().sum()
zero_close = (df["close"] <= 0).sum()
if na_close > 0: errors.append(f"{na_close} close NA")
if zero_close > 0: errors.append(f"{zero_close} close <= 0")

# Résumé
if errors:
    print("[ERREUR] Qualité des données:")
    for e in errors: print(" -", e)
    exit(1)
else:
    print("[OK] Données valides.")
    print("Période:", df["date"].min().date(), "->", df["date"].max().date())
    print("Tickers:", df["ticker"].nunique(), "| Jours:", df["date"].nunique())
