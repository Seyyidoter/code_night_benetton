"""
Bu dosya hafif makine öğrenmesi destekli stratejiyi içerir.

Yöntem:
- Ortak feature set üzerinden yön tahmini yapar
- Model: RandomForestClassifier
- Model predict içinde eğitilmez
- Önce get_data(), sonra model eğitimi, sonra backtest.run()

Bu cnlib kılavuzundaki öneriyle uyumludur.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from cnlib.base_strategy import BaseStrategy

from config import (
    ML_ALLOCATION,
    DEFAULT_LEVERAGE,
    ML_STOP_LOSS_PCT,
    ML_TAKE_PROFIT_PCT,
    ML_PROBA_THRESHOLD,
    MIN_HISTORY,
)
from indicators import add_indicators
from utils import has_nan, build_tp_sl


class MLConfirmedStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.models = {}

    def _flat_decision(self, coin: str) -> dict:
        return {
            "coin": coin,
            "signal": 0,
            "allocation": 0.0,
            "leverage": 1,
        }

    def _feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features["ma_diff"] = df["MA_DIFF_SAFE"] # Golden cross sinyali ana data
        features["volatility"] = df["VOLATILITY"]
        features["rsi"] = df["RSI"]
        features["macd_hist"] = df["MACD_HIST"]
        features["atr"] = df["ATR"]
        features["adx"] = df["ADX"]
        
        return features

    def _build_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Makro Hedef: Gelecek 7 Mum sonrasındaki fiyat, şu ana kıyasla en az %1 yukarıda mı?
        Sadece büyük hedefleri ezberletmek için ufak gürültüleri sıfırlıyoruz.
        """
        return (df["Close"].shift(-7) > (df["Close"] * 1.01)).astype(int)

    def egit(self):
        """
        get_data() sonrası çağrılır.
        coin_data içindeki verilerle her coin için model eğitir.
        """
        self.models = {}

        for coin, df in self._full_data.items():
            enriched = add_indicators(df)
            X = self._feature_frame(enriched)
            y = self._build_labels(enriched)

            dataset = X.copy()
            dataset["target"] = y
            dataset = dataset.dropna().reset_index(drop=True)

            if len(dataset) < MIN_HISTORY:
                continue

            X_clean = dataset.drop(columns=["target"])
            y_clean = dataset["target"]

            model = HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=5,                # Dört yıllık veriyi analiz edebilmesi için zeka derinliği 5'e çıkarıldı
                learning_rate=0.03,         # Sindirerek öğrenme
                l2_regularization=0.5,      # Yumuşak ceza
                min_samples_leaf=20,
                random_state=42,
                class_weight="balanced"
            )
            model.fit(X_clean.iloc[:-10], y_clean.iloc[:-10])

            self.models[coin] = model

    def _build_long_decision(self, coin: str, entry: float, atr_val: float, allocation: float = ML_ALLOCATION, leverage: int = DEFAULT_LEVERAGE) -> dict:
        sl_price = max(entry - (2.0 * atr_val), entry * 0.5)
        tp_price = entry + (4.0 * atr_val)
        return {
            "coin": coin,
            "signal": 1,
            "allocation": allocation,
            "leverage": leverage,
            "take_profit": tp_price,
            "stop_loss": sl_price,
        }

    def _build_short_decision(self, coin: str, entry: float, atr_val: float, allocation: float = ML_ALLOCATION, leverage: int = DEFAULT_LEVERAGE) -> dict:
        sl_price = entry + (2.0 * atr_val)
        tp_price = max(entry - (4.0 * atr_val), entry * 0.1)
        return {
            "coin": coin,
            "signal": -1,
            "allocation": allocation,
            "leverage": leverage,
            "take_profit": tp_price,
            "stop_loss": sl_price,
        }

    def predict(self, data: dict) -> list[dict]:
        decisions = []

        for coin, df in data.items():
            if coin not in self.models or len(df) < MIN_HISTORY:
                decisions.append(self._flat_decision(coin))
                continue

            enriched = add_indicators(df)
            last_features = self._feature_frame(enriched).iloc[-1]

            if has_nan(last_features.values):
                decisions.append(self._flat_decision(coin))
                continue

            proba_up = self.models[coin].predict_proba([last_features.values])[0][1]
            entry = float(enriched["Close"].iloc[-1])

            # Makro Trend Analizi için rahatlatılmış kısıtlamalar
            ma_diff_safe = float(enriched["MA_DIFF_SAFE"].iloc[-1])
            adx_val = float(enriched["ADX"].iloc[-1]) if "ADX" in enriched.columns else 25.0
            atr_val = float(enriched["ATR"].iloc[-1])

            is_safe_long = ma_diff_safe > 0
            is_safe_short = ma_diff_safe < 0
            is_trending = adx_val >= 20.0  # Erken yakalamak için 20'ye gevşetildi

            if not hasattr(self, "debug_print_cnt"):
                self.debug_print_cnt = 0
            if self.debug_print_cnt < 20:
                print(f"DEBUG: coin={coin} proba={proba_up:.4f} safe_L={is_safe_long} trend={is_trending} atr={atr_val:.2f}")
                self.debug_print_cnt += 1

            # YZ'ye Güçlü Yetki Devri (Zeki Bakiye)
            if proba_up >= 0.55 and is_safe_long and is_trending:
                allocation = 0.33 if proba_up > 0.80 else 0.25 # Trende güçlü giriş
                leverage = 3 if proba_up > 0.80 else 2
                decisions.append(self._build_long_decision(coin, entry, atr_val, allocation, leverage))
            elif proba_up <= 0.45 and is_safe_short and is_trending:
                p_down = 1 - proba_up
                allocation = 0.33 if p_down > 0.80 else 0.25
                leverage = 3 if p_down > 0.80 else 2
                decisions.append(self._build_short_decision(coin, entry, atr_val, allocation, leverage))
            else:
                decisions.append(self._flat_decision(coin))

        return decisions