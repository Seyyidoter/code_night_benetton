"""
Bu dosya seçilmiş olan en iyi makine öğrenmesi algoritmalarını unseen (görülmemiş) veride 
backtest ederek birbiri ile karşılaştırır.
"""

import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import INITIAL_CAPITAL, MIN_HISTORY
from strategy_ml import MLConfirmedStrategy
from data_splits import split_coin_data_by_ratio, attach_fixed_data_loader, set_strategy_coin_data
from pseudo_unseen_compare import run_strategy_on_fixed_data
from package_data_loader import load_packaged_training_data


class MLBenchmarkStrategy(MLConfirmedStrategy):
    def __init__(self, model_class, **model_kwargs):
        super().__init__()
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        
    def prepare_models(self):
        """Override to use the dynamically passed model"""
        from indicators import add_indicators
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

            X_clean = dataset[self.feature_columns]
            y_clean = dataset["target"]

            model = self.model_class(**self.model_kwargs)
            model.fit(X_clean.iloc[:-1], y_clean.iloc[:-1])
            self.models[coin] = model


def main():
    full_data = load_packaged_training_data()
    dev_data, test_data = split_coin_data_by_ratio(full_data)

    models_to_test = {
        "RandomForest (Baseline)": (RandomForestClassifier, {
            "n_estimators": 200, "max_depth": 4, "min_samples_leaf": 5, 
            "class_weight": "balanced_subsample", "random_state": 42
        }),
        "HistGradientBoosting": (HistGradientBoostingClassifier, {
            "max_iter": 200, "max_depth": 5, "learning_rate": 0.05, 
            "random_state": 42
        }),
        "ExtraTrees": (ExtraTreesClassifier, {
            "n_estimators": 200, "max_depth": 5, "min_samples_leaf": 5, 
            "class_weight": "balanced_subsample", "random_state": 42
        }),
        "LogisticRegression": (LogisticRegression, {
            "class_weight": "balanced", "random_state": 42, "max_iter": 1000
        })
    }

    rows = []

    for name, (cls, kwargs) in models_to_test.items():
        print(f"\n{'=' * 60}")
        print(f"BENCHMARKING MODEL: {name.upper()}")
        print(f"{'=' * 60}")

        strategy = MLBenchmarkStrategy(cls, **kwargs)
        
        result, summary = run_strategy_on_fixed_data(
            name=name,
            strategy=strategy,
            backtest_data=test_data,
            train_data_for_ml=dev_data,
        )

        rows.append(summary)

    df = pd.DataFrame(rows).sort_values(by="balanced_score", ascending=False)

    print("\n" + "=" * 60)
    print("OOS (UNSEEN) BENCHMARK COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
