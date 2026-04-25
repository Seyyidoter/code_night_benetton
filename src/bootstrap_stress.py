"""
Bu dosya block-bootstrap sentetik gelecek senaryoları üretir.

Amaç:
- tamamen rastgele veri üretmek yerine
- tarihsel OHLCV bloklarını yeniden örnekleyerek
- stratejinin farklı ama makul gelecek senaryolarında ne yaptığını görmek

Varsayılan olarak SAFE stratejiyi test ediyoruz.
"""

import pandas as pd
from cnlib import backtest

from package_data_loader import load_packaged_training_data
from config import INITIAL_CAPITAL, BOOTSTRAP_SCENARIOS, BOOTSTRAP_SEED
from strategy_safe import SafeHybridStrategy
from strategy_ml import MLConfirmedStrategy
from data_splits import split_coin_data_by_ratio, attach_fixed_data_loader, block_bootstrap_coin_data


from utils import compute_max_drawdown


def run_safe_on_fixed_data(backtest_data: dict):
    strategy = SafeHybridStrategy()
    attach_fixed_data_loader(strategy, backtest_data)
    result = backtest.run(strategy=strategy, initial_capital=INITIAL_CAPITAL)

    return {
        "strategy": "safe",
        "return_pct": float(result.return_pct),
        "trade_count": len(result.trade_history),
        "max_drawdown_pct": round(compute_max_drawdown(result.portfolio_series), 2),
    }

def run_ml_on_fixed_data(backtest_data: dict, train_data: dict):
    strategy = MLConfirmedStrategy()
    strategy._full_data = train_data
    strategy.prepare_models()
    
    attach_fixed_data_loader(strategy, backtest_data)
    result = backtest.run(strategy=strategy, initial_capital=INITIAL_CAPITAL, silent=True)

    return {
        "strategy": "ml",
        "return_pct": float(result.return_pct),
        "trade_count": len(result.trade_history),
        "max_drawdown_pct": round(compute_max_drawdown(result.portfolio_series), 2),
    }


def main():
    full_data = load_packaged_training_data()
    dev_data, test_data = split_coin_data_by_ratio(full_data)

    target_len = min(len(df) for df in test_data.values())

    rows = []

    for i in range(BOOTSTRAP_SCENARIOS):
        seed = BOOTSTRAP_SEED + i

        synthetic_future = block_bootstrap_coin_data(
            coin_data=dev_data,
            target_len=target_len,
            seed=seed,
        )

        safe_summary = run_safe_on_fixed_data(synthetic_future)
        safe_summary["scenario"] = i + 1
        safe_summary["seed"] = seed
        rows.append(safe_summary)
        
        ml_summary = run_ml_on_fixed_data(synthetic_future, dev_data)
        ml_summary["scenario"] = i + 1
        ml_summary["seed"] = seed
        rows.append(ml_summary)

        print(f"Scenario {i+1} [SAFE]: {safe_summary}")
        print(f"Scenario {i+1} [ML]  : {ml_summary}")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("BOOTSTRAP STRESS SUMMARY (SAFE vs ML)")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n[SAFE ÖZET]")
    safe_df = df[df["strategy"] == "safe"]
    print("Average return:", round(safe_df["return_pct"].mean(), 2))
    print("Worst return  :", round(safe_df["return_pct"].min(), 2))
    print("Best return   :", round(safe_df["return_pct"].max(), 2))
    print("Worst Drawdown:", round(safe_df["max_drawdown_pct"].max(), 2))

    print("\n[ML ÖZET]")
    ml_df = df[df["strategy"] == "ml"]
    print("Average return:", round(ml_df["return_pct"].mean(), 2))
    print("Worst return  :", round(ml_df["return_pct"].min(), 2))
    print("Best return   :", round(ml_df["return_pct"].max(), 2))
    print("Worst Drawdown:", round(ml_df["max_drawdown_pct"].max(), 2))


if __name__ == "__main__":
    main()