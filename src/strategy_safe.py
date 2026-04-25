"""
Bu dosya SAFE v2 stratejisini içerir.

Amaç:
- SAFE v1'e göre daha seçici olmak
- trade sayısını azaltmak
- unseen veride daha dayanıklı kalmak

Yöntem:
- Trend filtresi: SMA kısa / SMA uzun
- Hacim teyidi: OBV + volume confirmation (+ opsiyonel volume spike)
- Bollinger filtresi: üst/alt banda fazla yapışık girişleri elemek
- BB_MID teyidi: yönü biraz daha doğrulamak
- MA gap filtresi: zayıf trendleri elemek
"""

import pandas as pd
from cnlib.base_strategy import BaseStrategy

import config
from config import (
    USE_EMA,
    DEFAULT_ALLOCATION,
    DEFAULT_LEVERAGE,
    UPPER_BAND_BUFFER,
    LOWER_BAND_BUFFER,
    VOLATILITY_GUARD_MULTIPLIER,
    MIN_HISTORY,
    SAFE_MIN_MA_GAP_RATIO,
    SAFE_REQUIRE_BB_MID_CONFIRM,
    SAFE_REQUIRE_VOLUME_SPIKE,
    SAFE_STRONG_TREND_GAP_RATIO,
    SHORT_MA,
    LONG_MA,
)
from indicators import add_indicators
from utils import (
    has_nan,
    choose_ma_columns,
    build_tp_sl,
    _flat_decision,
    _volatility_guard_active
)


class SafeHybridStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def _volatility_guard_active(self, last_row: pd.Series) -> bool:
        return _volatility_guard_active(
            last_row["VOLATILITY"],
            last_row["VOLATILITY_AVG"],
            VOLATILITY_GUARD_MULTIPLIER
        )

    def _ma_gap_ratio(self, last: pd.Series, short_col: str, long_col: str) -> float:
        """
        MA farkını fiyata oranlayarak normalize eder.
        Coin fiyat seviyesinden bağımsız karar vermek için kullanıyoruz.
        """
        return abs(last[short_col] - last[long_col]) / last["Close"]

    def _long_conditions(self, df: pd.DataFrame) -> bool:
        short_col, long_col = choose_ma_columns(USE_EMA, SHORT_MA, LONG_MA)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        required = [
            last[short_col],
            last[long_col],
            prev[short_col],
            prev[long_col],
            last["OBV"],
            prev["OBV"],
            last["BB_UPPER"],
            last["BB_MID"],
            last["Close"],
            last["VOLUME_AVG"],
        ]

        if has_nan(required):
            return False

        trend_up = last[short_col] > last[long_col]
        prev_not_up = prev[short_col] <= prev[long_col]
        fresh_cross_up = prev_not_up and trend_up

        obv_up = last["OBV"] > prev["OBV"]
        volume_confirmed = bool(last["VOLUME_CONFIRMED"])
        volume_spike = bool(last["VOLUME_SPIKE"])
        not_overextended = last["Close"] < (last["BB_UPPER"] * UPPER_BAND_BUFFER)
        volatility_ok = not self._volatility_guard_active(last)

        ma_gap_ratio = self._ma_gap_ratio(last, short_col, long_col)
        enough_gap = ma_gap_ratio >= SAFE_MIN_MA_GAP_RATIO
        strong_trend = ma_gap_ratio >= SAFE_STRONG_TREND_GAP_RATIO

        bb_mid_confirm = True
        if SAFE_REQUIRE_BB_MID_CONFIRM:
            bb_mid_confirm = last["Close"] > last["BB_MID"]

        # Daha seçici long mantığı:
        # 1) Ya yeni crossover olacak
        # 2) Ya da trend zaten güçlü olacak
        # Hacim de teyit edecek
        if SAFE_REQUIRE_VOLUME_SPIKE:
            volume_ok = volume_confirmed and (volume_spike or fresh_cross_up)
        else:
            volume_ok = volume_confirmed

        # RSI filter: Don't buy if heavily overbought
        rsi_ok = last.get("RSI_14", 50) < config.RSI_OVERBOUGHT if "RSI_14" in last else True

        entry_ok = (fresh_cross_up or (strong_trend and volume_confirmed)) and rsi_ok

        return (
                trend_up
                and obv_up
                and volume_ok
                and not_overextended
                and volatility_ok
                and bb_mid_confirm
                and enough_gap
                and entry_ok
        )

    def _short_conditions(self, df: pd.DataFrame) -> bool:
        short_col, long_col = choose_ma_columns(USE_EMA, SHORT_MA, LONG_MA)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        required = [
            last[short_col],
            last[long_col],
            prev[short_col],
            prev[long_col],
            last["OBV"],
            prev["OBV"],
            last["BB_LOWER"],
            last["BB_MID"],
            last["Close"],
            last["VOLUME_AVG"],
        ]

        if has_nan(required):
            return False

        trend_down = last[short_col] < last[long_col]
        prev_not_down = prev[short_col] >= prev[long_col]
        fresh_cross_down = prev_not_down and trend_down

        obv_down = last["OBV"] < prev["OBV"]
        volume_confirmed = bool(last["VOLUME_CONFIRMED"])
        volume_spike = bool(last["VOLUME_SPIKE"])
        not_overextended = last["Close"] > (last["BB_LOWER"] * LOWER_BAND_BUFFER)
        volatility_ok = not self._volatility_guard_active(last)

        ma_gap_ratio = self._ma_gap_ratio(last, short_col, long_col)
        enough_gap = ma_gap_ratio >= SAFE_MIN_MA_GAP_RATIO
        strong_trend = ma_gap_ratio >= SAFE_STRONG_TREND_GAP_RATIO

        bb_mid_confirm = True
        if SAFE_REQUIRE_BB_MID_CONFIRM:
            bb_mid_confirm = last["Close"] < last["BB_MID"]

        if SAFE_REQUIRE_VOLUME_SPIKE:
            volume_ok = volume_confirmed and (volume_spike or fresh_cross_down)
        else:
            volume_ok = volume_confirmed

        # RSI filter: Don't short if heavily oversold
        rsi_ok = last.get("RSI_14", 50) > config.RSI_OVERSOLD if "RSI_14" in last else True

        entry_ok = (fresh_cross_down or (strong_trend and volume_confirmed)) and rsi_ok

        return (
                trend_down
                and obv_down
                and volume_ok
                and not_overextended
                and volatility_ok
                and bb_mid_confirm
                and enough_gap
                and entry_ok
        )

    def _build_long_decision(self, coin: str, entry: float, atr: float, ma_gap_ratio: float) -> dict:
        # Dynamic TP/SL using ATR
        stop_loss = max(1e-8, entry - (atr * config.ATR_SL_MULTIPLIER))
        take_profit = max(1e-8, entry + (atr * config.ATR_TP_MULTIPLIER))
        
        # Dynamic allocation based on trend strength
        allocation = config.STRONG_SIGNAL_ALLOCATION if ma_gap_ratio >= SAFE_STRONG_TREND_GAP_RATIO else DEFAULT_ALLOCATION

        return {
            "coin": coin,
            "signal": 1,
            "allocation": allocation,
            "leverage": DEFAULT_LEVERAGE,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }

    def _build_short_decision(self, coin: str, entry: float, atr: float, ma_gap_ratio: float) -> dict:
        # Dynamic TP/SL using ATR
        stop_loss = max(1e-8, entry + (atr * config.ATR_SL_MULTIPLIER))
        take_profit = max(1e-8, entry - (atr * config.ATR_TP_MULTIPLIER))

        # Dynamic allocation based on trend strength
        allocation = config.STRONG_SIGNAL_ALLOCATION if ma_gap_ratio >= SAFE_STRONG_TREND_GAP_RATIO else DEFAULT_ALLOCATION

        return {
            "coin": coin,
            "signal": -1,
            "allocation": allocation,
            "leverage": DEFAULT_LEVERAGE,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }

    def predict(self, data: dict) -> list[dict]:
        decisions = []

        for coin, df in data.items():
            df = add_indicators(df)

            if len(df) < MIN_HISTORY:
                decisions.append(_flat_decision(coin))
                continue

            last_close = float(df["Close"].iloc[-1])
            last_atr = float(df["ATR_14"].iloc[-1]) if "ATR_14" in df.columns else (last_close * 0.02)
            
            short_col, long_col = choose_ma_columns(USE_EMA, SHORT_MA, LONG_MA)
            ma_gap_ratio = self._ma_gap_ratio(df.iloc[-1], short_col, long_col)

            if self._long_conditions(df):
                decisions.append(self._build_long_decision(coin, last_close, last_atr, ma_gap_ratio))
                continue

            if self._short_conditions(df):
                decisions.append(self._build_short_decision(coin, last_close, last_atr, ma_gap_ratio))
                continue

            decisions.append(_flat_decision(coin))

        return decisions