"""
Bu dosya güvenli hibrit stratejiyi tutar.

Yöntem:
- Trend filtresi: SMA kısa > SMA uzun ise long bias, tersi short bias
- Hacim teyidi: OBV yönü ve mevcut hacmin ortalama üstünde olması
- Volatilite filtresi: Bollinger bandına fazla yapışmışsa giriş yapmama
- Volatility guard: piyasa anormal derecede oynaksa yeni işlem açmama

Bu strateji öğrenen bir AI değildir.
Açıklanabilir, kural tabanlı bir stratejidir.
"""

import pandas as pd
from cnlib.base_strategy import BaseStrategy

from config import (
    SHORT_MA,
    LONG_MA,
    USE_EMA,
    DEFAULT_ALLOCATION,
    DEFAULT_LEVERAGE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    UPPER_BAND_BUFFER,
    LOWER_BAND_BUFFER,
    VOLATILITY_GUARD_MULTIPLIER,
    MIN_HISTORY,
)
from indicators import add_indicators
from utils import has_nan, choose_ma_columns, build_tp_sl


class SafeHybridStrategy(BaseStrategy):
    """
    İlk aday stratejimiz: güvenli hibrit strateji.
    """

    def __init__(self):
        super().__init__()

    def _flat_decision(self, coin: str) -> dict:
        """
        Bu coin için pozisyon açma / açıksa kapat.
        cnlib'de signal=0 bu anlama gelir.
        """
        return {
            "coin": coin,
            "signal": 0,
            "allocation": 0.0,
            "leverage": 1,
        }

    def _volatility_guard_active(self, last_row: pd.Series) -> bool:
        """
        Son volatilite, yakın geçmiş ortalamasına göre çok yüksekse
        yeni işleme girmeyelim.
        """
        vol = last_row["VOLATILITY"]
        vol_avg = last_row["VOLATILITY_AVG"]

        if has_nan([vol, vol_avg]) or vol_avg == 0:
            return False

        return vol > vol_avg * VOLATILITY_GUARD_MULTIPLIER

    def _long_conditions(self, df: pd.DataFrame) -> bool:
        """
        Long açmak için gereken tüm koşulları kontrol eder.
        """
        short_col, long_col = choose_ma_columns(USE_EMA, SHORT_MA, LONG_MA)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        required = [
            last[short_col],
            last[long_col],
            last["OBV"],
            prev["OBV"],
            last["BB_UPPER"],
            last["Close"],
            last["VOLUME_AVG"],
        ]

        if has_nan(required):
            return False

        trend_up = last[short_col] > last[long_col]
        obv_up = last["OBV"] > prev["OBV"]
        volume_confirmed = bool(last["VOLUME_CONFIRMED"])
        not_overextended = last["Close"] < (last["BB_UPPER"] * UPPER_BAND_BUFFER)
        volatility_ok = not self._volatility_guard_active(last)

        return trend_up and obv_up and volume_confirmed and not_overextended and volatility_ok

    def _short_conditions(self, df: pd.DataFrame) -> bool:
        """
        Short açmak için gereken tüm koşulları kontrol eder.
        """
        short_col, long_col = choose_ma_columns(USE_EMA, SHORT_MA, LONG_MA)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        required = [
            last[short_col],
            last[long_col],
            last["OBV"],
            prev["OBV"],
            last["BB_LOWER"],
            last["Close"],
            last["VOLUME_AVG"],
        ]

        if has_nan(required):
            return False

        trend_down = last[short_col] < last[long_col]
        obv_down = last["OBV"] < prev["OBV"]
        volume_confirmed = bool(last["VOLUME_CONFIRMED"])
        not_overextended = last["Close"] > (last["BB_LOWER"] * LOWER_BAND_BUFFER)
        volatility_ok = not self._volatility_guard_active(last)

        return trend_down and obv_down and volume_confirmed and not_overextended and volatility_ok

    def _build_long_decision(self, coin: str, entry: float) -> dict:
        """
        Long pozisyon kararı oluşturur.
        """
        take_profit, stop_loss = build_tp_sl(
            entry=entry,
            direction=1,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
        )

        return {
            "coin": coin,
            "signal": 1,
            "allocation": DEFAULT_ALLOCATION,
            "leverage": DEFAULT_LEVERAGE,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }

    def _build_short_decision(self, coin: str, entry: float) -> dict:
        """
        Short pozisyon kararı oluşturur.
        """
        take_profit, stop_loss = build_tp_sl(
            entry=entry,
            direction=-1,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
        )

        return {
            "coin": coin,
            "signal": -1,
            "allocation": DEFAULT_ALLOCATION,
            "leverage": DEFAULT_LEVERAGE,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }

    def predict(self, data: dict) -> list[dict]:
        """
        cnlib tarafından her candle kapanışında çağrılır.

        data:
            {
                "kapcoin-usd_train":  pd.DataFrame,
                "metucoin-usd_train": pd.DataFrame,
                "tamcoin-usd_train":  pd.DataFrame,
            }

        Her candle için tüm coinler listede olmalıdır.
        """
        decisions = []

        for coin, df in data.items():
            # İndikatörleri ekle
            df = add_indicators(df)

            # Yeterli geçmiş yoksa işleme girme
            if len(df) < MIN_HISTORY:
                decisions.append(self._flat_decision(coin))
                continue

            last_close = float(df["Close"].iloc[-1])

            # Long koşulları uygunsa long aç
            if self._long_conditions(df):
                decisions.append(self._build_long_decision(coin, last_close))
                continue

            # Short koşulları uygunsa short aç
            if self._short_conditions(df):
                decisions.append(self._build_short_decision(coin, last_close))
                continue

            # Hiçbir koşul uygun değilse flat / close
            decisions.append(self._flat_decision(coin))

        return decisions