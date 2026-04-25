"""
Bu dosya daha hızlı ve daha agresif stratejiyi içerir.

Yöntem:
- Trend filtresi: EMA 9/21
- Hacim teyidi: OBV yönü + volume spike
- Volatilite filtresi: Bollinger bandına yapışmış girişleri azaltma
- Safe stratejiye göre daha reaktif davranır
"""

import pandas as pd
from cnlib.base_strategy import BaseStrategy

import config
from config import (
    FAST_EMA_SHORT,
    FAST_EMA_LONG,
    FAST_ALLOCATION,
    DEFAULT_LEVERAGE,
    UPPER_BAND_BUFFER,
    LOWER_BAND_BUFFER,
    VOLATILITY_GUARD_MULTIPLIER,
    MIN_HISTORY,
)
from indicators import add_indicators
from utils import has_nan, build_tp_sl, _flat_decision, _volatility_guard_active


class FastMomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    # Base logic moved to utils
    pass

    def _volatility_guard_active(self, last_row: pd.Series) -> bool:
        return _volatility_guard_active(
            last_row["VOLATILITY"],
            last_row["VOLATILITY_AVG"],
            VOLATILITY_GUARD_MULTIPLIER
        )

    def _long_conditions(self, df: pd.DataFrame) -> bool:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        required = [
            last[f"EMA_{FAST_EMA_SHORT}"],
            last[f"EMA_{FAST_EMA_LONG}"],
            last["OBV"],
            prev["OBV"],
            last["BB_UPPER"],
            last["Close"],
        ]
        if has_nan(required):
            return False

        trend_up = last[f"EMA_{FAST_EMA_SHORT}"] > last[f"EMA_{FAST_EMA_LONG}"]
        obv_up = last["OBV"] > prev["OBV"]
        volume_spike = bool(last["VOLUME_SPIKE"])
        not_overextended = last["Close"] < (last["BB_UPPER"] * UPPER_BAND_BUFFER)
        volatility_ok = not self._volatility_guard_active(last)
        rsi_ok = last.get("RSI_14", 50) < config.RSI_OVERBOUGHT if "RSI_14" in last else True

        return trend_up and obv_up and volume_spike and not_overextended and volatility_ok and rsi_ok

    def _short_conditions(self, df: pd.DataFrame) -> bool:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        required = [
            last[f"EMA_{FAST_EMA_SHORT}"],
            last[f"EMA_{FAST_EMA_LONG}"],
            last["OBV"],
            prev["OBV"],
            last["BB_LOWER"],
            last["Close"],
        ]
        if has_nan(required):
            return False

        trend_down = last[f"EMA_{FAST_EMA_SHORT}"] < last[f"EMA_{FAST_EMA_LONG}"]
        obv_down = last["OBV"] < prev["OBV"]
        volume_spike = bool(last["VOLUME_SPIKE"])
        not_overextended = last["Close"] > (last["BB_LOWER"] * LOWER_BAND_BUFFER)
        volatility_ok = not self._volatility_guard_active(last)
        rsi_ok = last.get("RSI_14", 50) > config.RSI_OVERSOLD if "RSI_14" in last else True

        return trend_down and obv_down and volume_spike and not_overextended and volatility_ok and rsi_ok

    def _build_long_decision(self, coin: str, entry: float, atr: float) -> dict:
        stop_loss = max(1e-8, entry - (atr * config.ATR_SL_MULTIPLIER))
        take_profit = max(1e-8, entry + (atr * config.ATR_TP_MULTIPLIER))
        
        return {
            "coin": coin,
            "signal": 1,
            "allocation": FAST_ALLOCATION,
            "leverage": DEFAULT_LEVERAGE,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }

    def _build_short_decision(self, coin: str, entry: float, atr: float) -> dict:
        stop_loss = max(1e-8, entry + (atr * config.ATR_SL_MULTIPLIER))
        take_profit = max(1e-8, entry - (atr * config.ATR_TP_MULTIPLIER))
        
        return {
            "coin": coin,
            "signal": -1,
            "allocation": FAST_ALLOCATION,
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

            if self._long_conditions(df):
                decisions.append(self._build_long_decision(coin, last_close, last_atr))
                continue

            if self._short_conditions(df):
                decisions.append(self._build_short_decision(coin, last_close, last_atr))
                continue

            decisions.append(_flat_decision(coin))

        return decisions