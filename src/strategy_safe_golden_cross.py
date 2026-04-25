"""
SafeHybridStrategy — Golden Cross Tabanlı Strateji

Mantık:
- Her coin için 7 günlük ve 25 günlük hareketli ortalama hesaplanır.
- ma7 > ma25 ise (Golden Cross): LONG aç
- ma7 < ma25 ise (Death Cross) : Pozisyonu kapat
- Kaldıraç kullanılmaz (leverage=1), güvenli mod.
- Sermaye 3 coinine eşit dağıtılır (%33 + %33 + %33).
"""

from cnlib.base_strategy import BaseStrategy


SHORT_WINDOW = 7
LONG_WINDOW = 25
ALLOCATION_PER_COIN = 0.33  # 3 coin × 0.33 ≈ 1.0


class SafeHybridStrategy(BaseStrategy):

    def predict(self, data: dict) -> list[dict]:
        decisions = []

        for coin, df in data.items():
            close = df["Close"]

            # Yeterli veri yoksa işlem yapma
            if len(close) < LONG_WINDOW:
                decisions.append({
                    "coin": coin,
                    "signal": 0,
                    "allocation": 0.0,
                    "leverage": 1,
                })
                continue

            ma_short = close.iloc[-SHORT_WINDOW:].mean()
            ma_long = close.iloc[-LONG_WINDOW:].mean()

            if ma_short > ma_long:
                # Golden Cross: kısa vade uzun vadeyi geçti → trend yukarı → LONG
                decisions.append({
                    "coin": coin,
                    "signal": 1,
                    "allocation": ALLOCATION_PER_COIN,
                    "leverage": 1,
                })
            else:
                # Death Cross: trend aşağı → pozisyonu kapat / açma
                decisions.append({
                    "coin": coin,
                    "signal": 0,
                    "allocation": 0.0,
                    "leverage": 1,
                })

        return decisions
