# tools/make_synth_prices.py
import numpy as np
import pandas as pd

# Paramètres
tickers = ["AAA", "BBB"]
start = "2023-12-01"
end   = "2024-03-31"
np.random.seed(42)

# Jours ouvrés (sans bourse spécifique)
dates = pd.bdate_range(start, end, freq="C")

out = []
for t in tickers:
    # point de départ différent par ticker pour diversifier un peu
    base = 100.0 if t == "AAA" else 50.0
    # random walk doux (drift léger + bruit)
    drift = 0.0008  # ~0.08% par jour
    vol   = 0.01    # ~1% volatilité quotidienne

    rets = drift + vol*np.random.randn(len(dates))
    prices = [base]
    for r in rets[1:]:
        prices.append(prices[-1] * (1.0 + r))
    prices = np.array(prices)

    # volumes plausibles
    vol_base = 12000 if t == "AAA" else 9000
    vols = (vol_base * (1.0 + 0.1*np.random.randn(len(dates)))).clip(1000).astype(int)

    df_t = pd.DataFrame({
        "date": dates,
        "ticker": t,
        "close": np.round(prices, 4),
        "volume": vols
    })
    out.append(df_t)

df = pd.concat(out, ignore_index=True)
df["date"] = df["date"].dt.strftime("%Y-%m-%d")
df = df.sort_values(["date", "ticker"])

# Conserver le schéma attendu par ton moteur
df.to_csv("data_proc/prices_sample_ext.csv", index=False)
print(f"[OK] Ecrit data_proc/prices_sample_ext.csv avec {df['date'].nunique()} jours et {df['ticker'].nunique()} tickers.")
