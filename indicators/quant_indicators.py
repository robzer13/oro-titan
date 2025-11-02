"""
quant_indicators.py

Script modulaire pour extraire et calculer un large set d'indicateurs
techniques, risque/perf, facteurs et (pré-)macro pour une action cotée.
Données marché : yfinance.
Retour : DataFrame indexé par date.

Dépendances :
    pip install yfinance pandas numpy scipy statsmodels ta
"""

import warnings
warnings.filterwarnings("ignore")

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf

# ------------------------------------------------------------
# 1. FETCHERS
# ------------------------------------------------------------

def fetch_price_data(ticker: str,
                     start: str = "2015-01-01",
                     end: str = None,
                     interval: str = "1d") -> pd.DataFrame:
    """
    Télécharge les données historiques depuis Yahoo Finance.
    """
    if end is None:
        end = dt.datetime.today().strftime("%Y-%m-%d")
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if data.empty:
            raise ValueError(f"Aucune donnée trouvée pour {ticker}")
        data = data.rename(columns=str.title)  # Open, High, Low, Close, Adj Close, Volume
        return data
    except Exception as e:
        print(f"[ERREUR] fetch_price_data({ticker}): {e}")
        return pd.DataFrame()


def fetch_benchmark_data(benchmark: str = "^GSPC",
                         start: str = "2015-01-01",
                         end: str = None) -> pd.DataFrame:
    """
    Télécharge un indice de marché pour calculer bêta, alpha, corrélation, etc.
    """
    return fetch_price_data(benchmark, start, end)


def fetch_info_data(ticker: str) -> dict:
    """
    Infos de base (float, short interest, etc.).
    Ces données sont souvent non datées -> on les duplique dans le temps.
    """
    try:
        tk = yf.Ticker(ticker)
        return tk.info
    except Exception as e:
        print(f"[ERREUR] fetch_info_data({ticker}): {e}")
        return {}


# ------------------------------------------------------------
# 2. OUTILS GÉNÉRAUX
# ------------------------------------------------------------

def rolling_autocorr(returns: pd.Series, window: int = 60, lags: int = 5) -> pd.DataFrame:
    """
    Autocorr roulante pour lags 1..lags.
    Retourne un DataFrame avec une colonne par lag.
    """
    out = {}
    for lag in range(1, lags + 1):
        out[f"autocorr_lag_{lag}"] = (
            returns.rolling(window).apply(lambda x, lag=lag: x.autocorr(lag=lag), raw=False)
        )
    return pd.DataFrame(out)


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Estimateur simple du Hurst exponent.
    On le mettra ensuite en roulante.
    """
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(series.values[lag:], series.values[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def rolling_hurst(series: pd.Series, window: int = 200) -> pd.Series:
    """
    Hurst en fenêtre glissante. Si pas assez de points -> NaN.
    """
    def _hurst(x):
        if x.isna().sum() > 0:
            return np.nan
        try:
            return hurst_exponent(pd.Series(x))
        except Exception:
            return np.nan
    return series.rolling(window).apply(_hurst, raw=False)


def max_drawdown_roll(serie: pd.Series, window: int = 252) -> pd.Series:
    """
    Max drawdown roulant sur window.
    """
    roll_max = serie.rolling(window, min_periods=1).max()
    dd = serie / roll_max - 1.0
    return dd.rolling(window, min_periods=1).min()


# ------------------------------------------------------------
# 3. INDICATEURS TECHNIQUES
# ------------------------------------------------------------

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la plupart des indicateurs techniques demandés.
    df doit contenir : Open, High, Low, Close, Adj Close, Volume
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- Moyennes mobiles & cross
    short_ma = close.rolling(20).mean()
    long_ma = close.rolling(50).mean()
    out["ma_20"] = short_ma
    out["ma_50"] = long_ma
    out["ma_cross_20_50"] = (short_ma > long_ma).astype(int)  # 1 si croisement haussier

    # pente de la MM50 (différence jour vs n-1)
    out["slope_ma_50"] = long_ma.diff()

    # --- RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_diff"] = macd - signal  # écart ligne/signal

    # --- Bollinger
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    out["bb_percent_b"] = (close - bb_low) / (bb_up - bb_low)
    out["bb_width"] = (bb_up - bb_low) / bb_mid

    # --- Donchian 20
    donch_high = high.rolling(20).max()
    donch_low = low.rolling(20).min()
    out["donchian_upper_20"] = donch_high
    out["donchian_lower_20"] = donch_low
    out["donchian_breakout_up"] = (close > donch_high.shift(1)).astype(int)
    out["donchian_breakout_down"] = (close < donch_low.shift(1)).astype(int)

    # --- ATR
    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    out["atr_14"] = atr

    # --- Volatilité réalisée 30j
    returns = close.pct_change()
    out["rv_30d"] = returns.rolling(30).std() * np.sqrt(252)

    # --- Autocorr lags 1..5 (fenêtre 60j)
    ac_df = rolling_autocorr(returns, window=60, lags=5)
    out = out.join(ac_df)

    # --- Asymétrie et kurtosis (fenêtre 60j)
    out["skew_60d"] = returns.rolling(60).apply(lambda x: skew(x, bias=False) if x.count() > 10 else np.nan)
    out["kurt_60d"] = returns.rolling(60).apply(lambda x: kurtosis(x, fisher=True, bias=False) if x.count() > 10 else np.nan)

    # --- Hurst (fenêtre 200j)
    out["hurst_200d"] = rolling_hurst(close, window=200)

    # --- Momentum 12m – 1m
    # prix(t-21) / prix(t-252) - 1
    out["mom_12m_1m"] = (close.shift(21) / close.shift(252)) - 1

    # --- Reversal court terme (rendement 5j inversé)
    out["rev_5d"] = -close.pct_change(5)

    # --- OBV
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    out["obv"] = obv

    # --- Z-score volume vs moyenne 20j
    vol_ma20 = volume.rolling(20).mean()
    vol_std20 = volume.rolling(20).std()
    out["vol_zscore_20d"] = (volume - vol_ma20) / vol_std20

    # --- Ratio volume / volatilité (ATR)
    out["vol_to_atr"] = volume / atr

    # --- Fréquence de clôture au-dessus de la MM200
    ma200 = close.rolling(200).mean()
    out["ma_200"] = ma200
    out["freq_above_ma200_200d"] = (close > ma200).rolling(200).mean()

    # --- Écart au plus haut/bas 52 semaines
    high_52w = close.rolling(252).max()
    low_52w = close.rolling(252).min()
    out["dist_to_52w_high"] = (close / high_52w) - 1
    out["dist_to_52w_low"] = (close / low_52w) - 1

    # --- Taille du gap d’ouverture (normalisée)
    gap = (df["Open"] - close.shift(1)) / close.shift(1)
    out["gap_normalized"] = gap

    # --- VWAP / distance au VWAP
    # Ici on le fait en daily (approx) = somme(P*V)/somme(V) cumulés
    pv_cum = (close * volume).cumsum()
    vol_cum = volume.cumsum()
    vwap = pv_cum / vol_cum
    out["vwap"] = vwap
    out["dist_to_vwap"] = (close / vwap) - 1

    # --- Champs non disponibles facilement (put/call, imbalance uptick/downtick, skew options)
    out["put_call_ratio"] = np.nan
    out["tick_imbalance"] = np.nan
    out["option_skew_25d"] = np.nan

    return out


# ------------------------------------------------------------
# 4. RISQUE & PERFORMANCE
# ------------------------------------------------------------

def compute_risk_performance(df: pd.DataFrame,
                             benchmark_df: pd.DataFrame = None,
                             risk_free_rate: float = 0.04) -> pd.DataFrame:
    """
    Calcule ratio de Sharpe roulant, max DD, alpha/bêta CAPM, corrélation.
    risk_free_rate : taux sans risque annuel (ex 4% -> US 10y 2025).
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    ret = close.pct_change()

    # --- Sharpe 63j (~3 mois)
    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_ret = ret - rf_daily
    roll_mean = excess_ret.rolling(63).mean()
    roll_std = excess_ret.rolling(63).std()
    out["sharpe_63d"] = (roll_mean / roll_std) * np.sqrt(252)

    # --- Max Drawdown 252j
    out["max_dd_252d"] = max_drawdown_roll(close, window=252)

    # --- Si benchmark dispo : alpha/bêta
    if benchmark_df is not None and not benchmark_df.empty:
        mkt_ret = benchmark_df["Close"].pct_change()
        # bêta et alpha roulants 63j
        cov = ret.rolling(63).cov(mkt_ret)
        var_mkt = mkt_ret.rolling(63).var()
        beta = cov / var_mkt
        out["beta_63d"] = beta

        # alpha = ret - rf - beta*(mkt - rf)
        out["alpha_63d"] = (ret - rf_daily) - beta * (mkt_ret - rf_daily)

        # corrélation roulante
        corr = ret.rolling(63).corr(mkt_ret)
        out["corr_mkt_63d"] = corr

        # stabilité de la corrélation = std de la corrélation sur 252j
        out["corr_stability_252d"] = corr.rolling(252).std()
    else:
        out["beta_63d"] = np.nan
        out["alpha_63d"] = np.nan
        out["corr_mkt_63d"] = np.nan
        out["corr_stability_252d"] = np.nan

    # --- Variance Risk Premium (placeholder)
    # IV pas dispo gratuitement sur yfinance de manière simple
    out["vrp"] = np.nan

    # --- Coût d’emprunt des titres (placeholder)
    out["borrow_fee"] = np.nan

    return out


# ------------------------------------------------------------
# 5. FACTEURS, FONDAMENTAUX
# ------------------------------------------------------------

def fetch_fundamentals_yf(ticker: str) -> dict:
    """
    Récupère quelques fondamentaux simples via yfinance.
    On renvoie un dict, qu'on retransformera en Series plus tard.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        # Certains champs utiles
        keys = [
            "returnOnEquity", "profitMargins", "grossMargins",
            "operatingMargins", "debtToEquity", "trailingPE",
            "priceToBook", "enterpriseToEbitda", "forwardEps",
            "trailingEps"
        ]
        return {k: info.get(k, np.nan) for k in keys}
    except Exception as e:
        print(f"[ERREUR] fetch_fundamentals_yf({ticker}): {e}")
        return {}


def compute_fundamental_series(index: pd.DatetimeIndex, fund_dict: dict) -> pd.DataFrame:
    """
    Transforme les fondamentaux (souvent statiques) en séries datées (forward-fill).
    """
    df = pd.DataFrame(index=index)
    for k, v in fund_dict.items():
        df[k] = v
    return df


def compute_style_factors_placeholder(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Placeholder pour l'exposition aux facteurs (value, size, quality, momentum).
    À remplacer par un vrai chargement Fama-French + régression.
    """
    df = pd.DataFrame(index=index)
    df["factor_value"] = np.nan
    df["factor_size"] = np.nan
    df["factor_quality"] = np.nan
    df["factor_momentum"] = np.nan
    return df


def compute_short_interest_turnover(index: pd.DatetimeIndex, info: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les séries pour short interest, days-to-cover, turnover.
    """
    out = pd.DataFrame(index=index)
    volume = df["Volume"]

    float_shares = info.get("floatShares", np.nan)
    shares_short = info.get("sharesShort", np.nan)
    short_ratio = info.get("shortRatio", np.nan)  # days to cover

    # turnover = volume / float
    if not np.isnan(float_shares):
        out["turnover"] = volume / float_shares
    else:
        out["turnover"] = np.nan

    out["short_interest"] = shares_short if shares_short is not None else np.nan
    out["days_to_cover"] = short_ratio if short_ratio is not None else np.nan
    return out


# ------------------------------------------------------------
# 6. MACRO & SENTIMENT (PLACEHOLDERS)
# ------------------------------------------------------------

def compute_macro_sentiment_placeholders(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Données difficiles à avoir gratuitement en direct.
    On met des colonnes vides mais bien nommées.
    """
    df = pd.DataFrame(index=index)
    df["seasonality_dow"] = index.dayofweek  # 0 = lundi
    df["seasonality_eom"] = (index.is_month_end).astype(int)
    df["market_breadth"] = np.nan
    df["credit_spread_hy_oas"] = np.nan
    df["yield_curve_2s10s"] = np.nan
    df["risk_free_10y"] = np.nan
    df["macro_surprise_index"] = np.nan
    df["fx_sensitivity_usd"] = np.nan
    df["etf_flows_sector"] = np.nan
    df["cftc_positioning"] = np.nan
    df["news_sentiment"] = np.nan
    df["social_sentiment"] = np.nan
    df["google_trends"] = np.nan
    return df


# ------------------------------------------------------------
# 7. ASSEMBLEUR GLOBAL
# ------------------------------------------------------------

def build_all_indicators(ticker: str,
                         start: str = "2015-01-01",
                         end: str = None,
                         benchmark: str = "^GSPC",
                         risk_free_rate: float = 0.04) -> pd.DataFrame:
    """
    Fonction principale : retourne un DataFrame avec TOUS les indicateurs possibles.
    """
    # 1. prix
    price_df = fetch_price_data(ticker, start, end)
    if price_df.empty:
        raise ValueError("Impossible de récupérer les données de prix")

    # 2. benchmark
    bench_df = fetch_benchmark_data(benchmark, start, end)

    # 3. techniques
    tech_df = compute_technical_indicators(price_df)

    # 4. risque/perf
    risk_df = compute_risk_performance(price_df, bench_df, risk_free_rate)

    # 5. fondamentaux
    fund_info = fetch_fundamentals_yf(ticker)
    fund_df = compute_fundamental_series(price_df.index, fund_info)

    # 6. factors placeholders
    factor_df = compute_style_factors_placeholder(price_df.index)

    # 7. short interest, turnover
    info = fetch_info_data(ticker)
    short_df = compute_short_interest_turnover(price_df.index, info, price_df)

    # 8. macro/sentiment placeholders
    macro_df = compute_macro_sentiment_placeholders(price_df.index)

    # Merge
    all_df = pd.concat([
        price_df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
        tech_df,
        risk_df,
        fund_df,
        factor_df,
        short_df,
        macro_df
    ], axis=1)

    # gestion erreurs : tri des colonnes, ffill si besoin
    all_df = all_df.sort_index()
    return all_df


# ------------------------------------------------------------
# 8. EXEMPLE D’UTILISATION
# ------------------------------------------------------------

if __name__ == "__main__":
    df = build_all_indicators("AAPL", start="2018-01-01")
    # On affiche les 5 dernières lignes
    print(df.tail(5))
    # Tu peux ensuite sauvegarder :
    # df.to_csv("aapl_indicators.csv")
