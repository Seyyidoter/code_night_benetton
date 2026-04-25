"""
Bu dosya üç stratejiyi art arda çalıştırır ve temel sonuçlarını karşılaştırır.

Karşılaştırılan stratejiler:
- SafeHybridStrategy
- FastMomentumStrategy
- MLConfirmedStrategy

Amaç:
- aynı başlangıç sermayesiyle çalıştırmak
- return, trade sayısı ve max drawdown gibi metriklere bakmak
- en stabil / en mantıklı stratejiyi seçmek
"""

import pandas as pd
from cnlib import backtest

from config import INITIAL_CAPITAL
from strategy_safe import SafeHybridStrategy
from strategy_fast import FastMomentumStrategy
from strategy_ml import MLConfirmedStrategy


def compute_max_drawdown(portfolio_series) -> float:
    """
    Portföy serisinden maksimum drawdown yüzdesini hesaplar.
    """
    values = [x["portfolio_value"] for x in portfolio_series]
    series = pd.Series(values, dtype="float64")
    if len(series) == 0:
        return 0.0

    running_max = series.cummax()
    drawdown = (series - running_max) / running_max * 100.0
    return abs(drawdown.min())


def run_single_strategy(name: str, strategy):
    """
    Tek stratejiyi çalıştırır ve özet metrikleri döndürür.
    """
    # ML stratejisinde modeli önceden eğitmek gerekiyor
    if isinstance(strategy, MLConfirmedStrategy):
        strategy.get_data()
        strategy.prepare_models()

    result = backtest.run(strategy=strategy, initial_capital=INITIAL_CAPITAL)

    trade_count = len(result.trade_history)
    max_dd = compute_max_drawdown(result.portfolio_series)

    summary = {
        "strategy": name,
        "return_pct": float(result.return_pct),
        "trade_count": trade_count,
        "max_drawdown_pct": round(max_dd, 2),
        # Basit dengeli skor: getiri - drawdown
        "balanced_score": round(float(result.return_pct) - max_dd, 2),
    }

    return result, summary


def main():
    strategies = [
        ("safe", SafeHybridStrategy()),
        ("fast", FastMomentumStrategy()),
        ("ml", MLConfirmedStrategy()),
    ]

    results = []

    for name, strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"RUNNING STRATEGY: {name.upper()}")
        print(f"{'=' * 60}")

        result, summary = run_single_strategy(name, strategy)
        result.print_summary()
        results.append(summary)

    df = pd.DataFrame(results).sort_values(by="balanced_score", ascending=False)

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print("\nBest balanced candidate:", best["strategy"])


if __name__ == "__main__":
    main()