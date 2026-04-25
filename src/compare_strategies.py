"""
Bu dosya SAFE, FAST ve ML stratejilerini art arda çalıştırır
ve temel sonuçlarını karşılaştırır.
"""

import pandas as pd
from cnlib import backtest

from config import INITIAL_CAPITAL
from strategy_safe import SafeHybridStrategy
from strategy_fast import FastMomentumStrategy
from strategy_ml import MLConfirmedStrategy


from utils import compute_max_drawdown


def run_single_strategy(name: str, strategy, silent=True):
    result = backtest.run(strategy=strategy, initial_capital=INITIAL_CAPITAL, silent=silent)

    trade_count = len(result.trade_history)
    max_dd = compute_max_drawdown(result.portfolio_series)

    summary = {
        "strategy": name,
        "return_pct": float(result.return_pct),
        "trade_count": trade_count,
        "liquidations": result.total_liquidations,
        "max_drawdown_pct": round(max_dd, 2),
        "balanced_score": round(float(result.return_pct) - max_dd, 2),
        "final_value": round(result.final_portfolio_value, 2),
    }

    return result, summary


def main():
    results = []

    # 1. Safe
    print(f"\n{'=' * 60}")
    print("RUNNING STRATEGY: SAFE")
    print(f"{'=' * 60}")
    s1 = SafeHybridStrategy()
    r1, sum1 = run_single_strategy("safe", s1)
    r1.print_summary()
    results.append(sum1)

    # 2. Fast
    print(f"\n{'=' * 60}")
    print("RUNNING STRATEGY: FAST")
    print(f"{'=' * 60}")
    s2 = FastMomentumStrategy()
    r2, sum2 = run_single_strategy("fast", s2)
    r2.print_summary()
    results.append(sum2)

    # 3. ML
    print(f"\n{'=' * 60}")
    print("RUNNING STRATEGY: ML")
    print(f"{'=' * 60}")
    s3 = MLConfirmedStrategy()
    s3.get_data()
    s3.prepare_models()
    print(f"ML models trained for: {list(s3.models.keys())}")
    r3, sum3 = run_single_strategy("ml", s3)
    r3.print_summary()
    results.append(sum3)

    # Comparison
    df = pd.DataFrame(results).sort_values(by="return_pct", ascending=False)

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest candidate: {best['strategy']} with {best['return_pct']:.2f}% return")


if __name__ == "__main__":
    main()