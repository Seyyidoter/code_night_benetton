"""
Bu dosya gerçek training veride walk-forward mantığıyla kıyas yapar.

Adımlar:
1. training veriyi yükle
2. ilk %75'i geliştirme verisi olarak ayır
3. son %25'i pseudo unseen test olarak ayır
4. Safe / Fast / ML stratejilerini unseen test üzerinde kıyasla

ML için:
- model dev_data üzerinde eğitilir
- backtest test_data üzerinde çalıştırılır
"""

import pandas as pd
from cnlib import backtest

from config import INITIAL_CAPITAL
from strategy_safe import SafeHybridStrategy
from strategy_fast import FastMomentumStrategy
from strategy_ml import MLConfirmedStrategy
from data_splits import split_coin_data_by_ratio, attach_fixed_data_loader, set_strategy_coin_data
from package_data_loader import load_packaged_training_data


from utils import compute_max_drawdown

def main():
    full_data = load_packaged_training_data()
    dev_data, test_data = split_coin_data_by_ratio(full_data)

    sample_coin = list(test_data.keys())[0]
    print("\nPseudo unseen test coin sample:", sample_coin)
    print("DEV range :", dev_data[sample_coin]["Date"].iloc[0], "->", dev_data[sample_coin]["Date"].iloc[-1])
    print("TEST range:", test_data[sample_coin]["Date"].iloc[0], "->", test_data[sample_coin]["Date"].iloc[-1])

    strategies = [
        ("safe", SafeHybridStrategy()),
        ("fast", FastMomentumStrategy()),
        ("ml", MLConfirmedStrategy()),
    ]

    rows = []

    for name, strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"RUNNING PSEUDO-UNSEEN TEST: {name.upper()}")
        print(f"{'=' * 60}")

        result, summary = run_strategy_on_fixed_data(
            name=name,
            strategy=strategy,
            backtest_data=test_data,
            train_data_for_ml=dev_data,
        )

        result.print_summary()
        print("Summary:", summary)
        rows.append(summary)

    df = pd.DataFrame(rows).sort_values(by="balanced_score", ascending=False)

    print("\n" + "=" * 60)
    print("PSEUDO-UNSEEN COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print("\nBest pseudo-unseen candidate:", best["strategy"])

def run_strategy_on_fixed_data(name: str, strategy, backtest_data: dict, train_data_for_ml: dict | None = None):
    if isinstance(strategy, MLConfirmedStrategy):
        if train_data_for_ml is None:
            raise ValueError("ML stratejisi için train_data_for_ml gerekli.")
        set_strategy_coin_data(strategy, train_data_for_ml)
        print(f"[ML TRAIN] loaded coins: {list(strategy.__dict__.get('_full_data', {}).keys())}")
        strategy.prepare_models()

    attach_fixed_data_loader(strategy, backtest_data)

    # backtest öncesi fixed loader'ın gerçekten veri yükleyebildiğini test et
    strategy.get_data()
    print(f"[{name.upper()} TEST] loaded coins: {list(strategy.__dict__.get('_full_data', {}).keys())}")

    result = backtest.run(strategy=strategy, initial_capital=INITIAL_CAPITAL)

    summary = {
        "strategy": name,
        "return_pct": float(result.return_pct),
        "trade_count": len(result.trade_history),
        "max_drawdown_pct": round(compute_max_drawdown(result.portfolio_series), 2),
    }
    summary["balanced_score"] = round(summary["return_pct"] - summary["max_drawdown_pct"], 2)

    return result, summary

if __name__ == "__main__":
    main()