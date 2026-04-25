"""
Bu dosya ortak indikatörleri hesaplar.

Kullandıklarımız:
- SMA / EMA: trend yönü
- OBV: hacim teyidi
- Bollinger Bands: aşırı uzama filtresi
- Ortalama hacim ve volume spike
- Basit volatilite
"""

import pandas as pd
import numpy as np

from config import (
    SHORT_MA,
    LONG_MA,
    FAST_EMA_SHORT,
    FAST_EMA_LONG,
    BB_WINDOW,
    BB_STD,
    VOLUME_AVG_WINDOW,
    VOLUME_SPIKE_MULTIPLIER,
    VOLATILITY_WINDOW,
)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume hesaplar.
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
    Gelen DataFrame'e ortak indikatör kolonlarını ekler.
    """
    df = df.copy()

    # Safe strateji için SMA
    df[f"SMA_{SHORT_MA}"] = df["Close"].rolling(SHORT_MA).mean()
    df[f"SMA_{LONG_MA}"] = df["Close"].rolling(LONG_MA).mean()

    # Safe strateji için EMA opsiyonu
    df[f"EMA_{SHORT_MA}"] = df["Close"].ewm(span=SHORT_MA, adjust=False).mean()
    df[f"EMA_{LONG_MA}"] = df["Close"].ewm(span=LONG_MA, adjust=False).mean()

    # Fast strateji için EMA
    df[f"EMA_{FAST_EMA_SHORT}"] = df["Close"].ewm(span=FAST_EMA_SHORT, adjust=False).mean()
    df[f"EMA_{FAST_EMA_LONG}"] = df["Close"].ewm(span=FAST_EMA_LONG, adjust=False).mean()

    # OBV
    df["OBV"] = calculate_obv(df)

    # Bollinger Bands
    rolling_mean = df["Close"].rolling(BB_WINDOW).mean()
    rolling_std = df["Close"].rolling(BB_WINDOW).std()

    df["BB_MID"] = rolling_mean
    df["BB_UPPER"] = rolling_mean + BB_STD * rolling_std
    df["BB_LOWER"] = rolling_mean - BB_STD * rolling_std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]

    # Hacim teyidi
    df["VOLUME_AVG"] = df["Volume"].rolling(VOLUME_AVG_WINDOW).mean()
    df["VOLUME_CONFIRMED"] = df["Volume"] >= df["VOLUME_AVG"]
    df["VOLUME_SPIKE"] = df["Volume"] >= (df["VOLUME_AVG"] * VOLUME_SPIKE_MULTIPLIER)

    # Basit volatilite
    df["RETURNS"] = df["Close"].pct_change()
    df["VOLATILITY"] = df["RETURNS"].rolling(VOLATILITY_WINDOW).std()
    df["VOLATILITY_AVG"] = df["VOLATILITY"].rolling(VOLATILITY_WINDOW).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # ATR (14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()

    # ADX Market Regime Detector (14)
    up_move = df["High"] - df["High"].shift()
    down_move = df["Low"].shift() - df["Low"]
    
    pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    pos_di = 100 * (pos_dm.rolling(window=14).mean() / df["ATR"])
    neg_di = 100 * (neg_dm.rolling(window=14).mean() / df["ATR"])
    
    dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di))
    df["ADX"] = dx.rolling(window=14).mean()

    # ML için yardımcı özellikler
    df["MA_DIFF_SAFE"] = df[f"SMA_{SHORT_MA}"] - df[f"SMA_{LONG_MA}"]
    df["MA_DIFF_FAST"] = df[f"EMA_{FAST_EMA_SHORT}"] - df[f"EMA_{FAST_EMA_LONG}"]
    df["OBV_DIFF"] = df["OBV"].diff()
    df["BB_POSITION"] = (df["Close"] - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"])

    # Zaman özellikleri
    if "Date" in df.columns:
        dt_col = pd.to_datetime(df["Date"])
        df["hour"] = dt_col.dt.hour
        df["day_of_week"] = dt_col.dt.dayofweek
    else:
        df["hour"] = 0
        df["day_of_week"] = 0

    return df