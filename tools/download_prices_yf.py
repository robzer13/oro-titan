# tools/download_prices_yf.py
from pathlib import Path
import pandas as pd
import yfinance as yf

# ========= PARAMS À ADAPTER =========
# Tes tickers (Yahoo!)
TICKERS_REAL = ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","AMD","NFLX","AVGO"]
# Mapping éventuel vers tes tickers internes (facultatif)
TICKERS_MAP = {t: t for t in TICKERS_REAL}

MAP_TO_SYN = {"AAPL": "AAA", "MSFT": "BBB"} # ou {}
# Fenêtre de téléchargement
START = "2023-11-01"
END   = "2024-04-15"
# Fichier de sortie (format long): date,ticker,close,volume
OUT_PATH = Path("data_proc/prices_eod.csv")
# ====================================

def _ensure_date_column(df_reset: pd.DataFrame) -> pd.DataFrame:
    """
    Après reset_index(), la colonne de date peut s'appeler 'Date' (index nommé)
    ou 'index' (index sans nom). On la standardise en 'date' (YYYY-MM-DD).
    """
    cols = df_reset.columns
    if "Date" in cols:
        df_reset = df_reset.rename(columns={"Date": "date"})
    elif "index" in cols:
        df_reset = df_reset.rename(columns={"index": "date"})
    elif "Datetime" in cols:
        df_reset = df_reset.rename(columns={"Datetime": "date"})
    # sinon: pas de colonne de date => on fabrique à partir de l'index d'origine
    if "date" not in df_reset.columns:
        # cas de figure improbable si reset_index(drop=True) a été fait ailleurs
        raise RuntimeError("Impossible de trouver la colonne de date après reset_index().")
    # normalise format
    df_reset["date"] = pd.to_datetime(df_reset["date"]).dt.strftime("%Y-%m-%d")
    return df_reset

def _normalize_download(df_download: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Normalise en format long: date,ticker,close,volume
    - Supporte multi-tickers (MultiIndex colonnes) et mono-ticker (colonnes simples).
    - Supporte Close/Adj Close (on met auto_adjust=True donc 'Close' est déjà ajusté).
    - Ignore proprement les tickers absents.
    """
    if df_download is None or len(df_download) == 0:
        raise RuntimeError("yfinance a renvoyé un DataFrame vide (tickers/période ?).")

    frames = []

    if isinstance(df_download.columns, pd.MultiIndex):
        # Avec group_by="ticker", on a (ticker, field)
        level0 = df_download.columns.get_level_values(0)
        level1 = df_download.columns.get_level_values(1)
        fields = set(level1)

        close_field = "Close" if "Close" in fields else ("Adj Close" if "Adj Close" in fields else None)
        if close_field is None:
            raise RuntimeError(f"Colonnes manquantes: ni 'Close' ni 'Adj Close' (dispo: {sorted(fields)})")
        if "Volume" not in fields:
            raise RuntimeError(f"Colonne manquante: 'Volume' (dispo: {sorted(fields)})")

        present = sorted(set(level0))
        missing = [t for t in tickers if t not in present]
        if missing:
            print(f"[WARN] Tickers sans données sur la période: {missing}")

        for tk in present:
            sub = df_download[tk][[close_field, "Volume"]].copy()
            sub = sub.rename(columns={close_field: "close", "Volume": "volume"}).reset_index()
            sub = _ensure_date_column(sub)
            sub["ticker"] = MAP_TO_SYN.get(tk, tk)
            frames.append(sub[["date", "ticker", "close", "volume"]])

    else:
        # Mono-ticker: colonnes simples
        cols = set(df_download.columns)
        close_col = "Close" if "Close" in cols else ("Adj Close" if "Adj Close" in cols else None)
        if close_col is None:
            raise RuntimeError(f"Colonnes manquantes: ni 'Close' ni 'Adj Close' (colonnes: {sorted(cols)})")
        if "Volume" not in cols:
            raise RuntimeError(f"Colonne manquante: 'Volume' (colonnes: {sorted(cols)})")

        tk = tickers[0] if tickers else "TICKER"
        sub = df_download[[close_col, "Volume"]].copy().reset_index()
        sub = _ensure_date_column(sub)
        sub = sub.rename(columns={close_col: "close", "Volume": "volume"})
        sub["ticker"] = MAP_TO_SYN.get(tk, tk)
        frames.append(sub[["date", "ticker", "close", "volume"]])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["close"])
    if out.empty:
        raise RuntimeError("Dataset vide après normalisation (dates/tickers ?).")
    if (out["close"] <= 0).any():
        raise RuntimeError("Valeurs 'close' <= 0 détectées (données invalides).")

    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = yf.download(
        tickers=TICKERS_REAL,
        start=START,
        end=END,
        auto_adjust=True,      # Close déjà ajusté dividendes/splits
        progress=False,
        group_by="ticker",     # IMPORTANT: colonnes (ticker, field)
        threads=True
    )

    df = _normalize_download(data, TICKERS_REAL)
    df.to_csv(OUT_PATH, index=False)
    print(f"[OK] Écrit {OUT_PATH} | jours={df['date'].nunique()} | tickers={df['ticker'].nunique()}")
    print(f"Plage: {df['date'].min()} -> {df['date'].max()}")
    print(df.head(6))

if __name__ == "__main__":
    main()
