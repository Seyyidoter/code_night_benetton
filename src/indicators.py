"""
Bu dosya teknik indikatörleri hesaplar.

Bu projede ilk sürümde kullandıklarımız:
- SMA / EMA: trend yönü
- OBV: hacim teyidi
- Bollinger Bands: volatilite ve aşırı uzama filtresi
- Ortalama hacim ve basit volatilite ölçüsü

Neden burada?
- Strateji dosyası sadece karar mantığıyla ilgilensin
- İndikatör hesapları tek yerde toplansın
"""

import pandas as pd

from config import (
    SHORT_MA,
    LONG_MA,
    BB_WINDOW,
    BB_STD,
    VOLUME_AVG_WINDOW,
    VOLATILITY_WINDOW,
)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume hesaplar.

    Mantık:
    - Kapanış önceki kapanıştan yüksekse hacim eklenir
    - Düşükse hacim çıkarılır
    - Aynıysa OBV değişmez
    """
    obv_values = [0]

    for i in range(1, len(df)):
        current_close = df["Close"].iloc[i]
        previous_close = df["Close"].iloc[i - 1]
        current_volume = df["Volume"].iloc[i]

        if current_close > previous_close:
            obv_values.append(obv_values[-1] + current_volume)
        elif current_close < previous_close:
            obv_values.append(obv_values[-1] - current_volume)
        else:
            obv_values.append(obv_values[-1])

    return pd.Series(obv_values, index=df.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gelen OHLCV DataFrame'ine gerekli indikatör kolonlarını ekler.

    cnlib veri kolonları:
    Date, Open, High, Low, Close, Volume
    """
    df = df.copy()

    # --- SMA ---
    df[f"SMA_{SHORT_MA}"] = df["Close"].rolling(SHORT_MA).mean()
    df[f"SMA_{LONG_MA}"] = df["Close"].rolling(LONG_MA).mean()

    # --- EMA ---
    # Şimdilik ana stratejide zorunlu değil ama test etmek için hazır tutuyoruz.
    df[f"EMA_{SHORT_MA}"] = df["Close"].ewm(span=SHORT_MA, adjust=False).mean()
    df[f"EMA_{LONG_MA}"] = df["Close"].ewm(span=LONG_MA, adjust=False).mean()

    # --- OBV ---
    df["OBV"] = calculate_obv(df)

    # --- Bollinger Bands ---
    rolling_mean = df["Close"].rolling(BB_WINDOW).mean()
    rolling_std = df["Close"].rolling(BB_WINDOW).std()

    df["BB_MID"] = rolling_mean
    df["BB_UPPER"] = rolling_mean + BB_STD * rolling_std
    df["BB_LOWER"] = rolling_mean - BB_STD * rolling_std

    # Bant genişliği, piyasa aşırı sıkışmış mı / aşırı açılmış mı anlamaya yardım eder
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]

    # --- Volume filter ---
    df["VOLUME_AVG"] = df["Volume"].rolling(VOLUME_AVG_WINDOW).mean()
    df["VOLUME_CONFIRMED"] = df["Volume"] >= df["VOLUME_AVG"]

    # --- Basit volatilite ölçüsü ---
    df["RETURNS"] = df["Close"].pct_change()
    df["VOLATILITY"] = df["RETURNS"].rolling(VOLATILITY_WINDOW).std()
    df["VOLATILITY_AVG"] = df["VOLATILITY"].rolling(VOLATILITY_WINDOW).mean()

    return df