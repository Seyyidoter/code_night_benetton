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
from sklearn.ensemble import RandomForestClassifier
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
        """
        ML için kullanılacak özellikleri çıkarır.
        """
        df = add_indicators(df).copy()

        features = pd.DataFrame(index=df.index)
        features["ret_1"] = df["RETURNS"]
        features["ma_diff_safe"] = df["MA_DIFF_SAFE"]
        features["ma_diff_fast"] = df["MA_DIFF_FAST"]
        features["obv_diff"] = df["OBV_DIFF"]
        features["bb_position"] = df["BB_POSITION"]
        features["volatility"] = df["VOLATILITY"]
        features["volume_confirmed"] = df["VOLUME_CONFIRMED"].astype(float)
        features["volume_spike"] = df["VOLUME_SPIKE"].astype(float)

        return features

    def _build_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Bir sonraki kapanış mevcut kapanıştan yüksekse 1, değilse 0.
        """
        return (df["Close"].shift(-1) > df["Close"]).astype(int)

    def prepare_models(self):
        """
        get_data() sonrası çağrılır.
        coin_data içindeki verilerle her coin için model eğitir.
        """
        self.models = {}

        for coin, df in self.coin_data.items():
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

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
            )
            model.fit(X_clean.iloc[:-1], y_clean.iloc[:-1])

            self.models[coin] = model

    def _build_long_decision(self, coin: str, entry: float) -> dict:
        take_profit, stop_loss = build_tp_sl(
            entry=entry,
            direction=1,
            stop_loss_pct=ML_STOP_LOSS_PCT,
            take_profit_pct=ML_TAKE_PROFIT_PCT,
        )
        return {
            "coin": coin,
            "signal": 1,
            "allocation": ML_ALLOCATION,
            "leverage": DEFAULT_LEVERAGE,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }

    def _build_short_decision(self, coin: str, entry: float) -> dict:
        take_profit, stop_loss = build_tp_sl(
            entry=entry,
            direction=-1,
            stop_loss_pct=ML_STOP_LOSS_PCT,
            take_profit_pct=ML_TAKE_PROFIT_PCT,
        )
        return {
            "coin": coin,
            "signal": -1,
            "allocation": ML_ALLOCATION,
            "leverage": DEFAULT_LEVERAGE,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }

    def predict(self, data: dict) -> list[dict]:
        decisions = []

        for coin, df in data.items():
            if coin not in self.models or len(df) < MIN_HISTORY:
                decisions.append(self._flat_decision(coin))
                continue

            enriched = add_indicators(df)
            X = self._feature_frame(enriched).dropna()

            if len(X) == 0:
                decisions.append(self._flat_decision(coin))
                continue

            last_features = X.iloc[-1]

            if has_nan(last_features.values):
                decisions.append(self._flat_decision(coin))
                continue

            proba_up = self.models[coin].predict_proba([last_features.values])[0][1]
            entry = float(enriched["Close"].iloc[-1])

            if proba_up >= ML_PROBA_THRESHOLD:
                decisions.append(self._build_long_decision(coin, entry))
            elif proba_up <= (1 - ML_PROBA_THRESHOLD):
                decisions.append(self._build_short_decision(coin, entry))
            else:
                decisions.append(self._flat_decision(coin))

        return decisions