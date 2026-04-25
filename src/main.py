from cnlib import backtest

from config import INITIAL_CAPITAL
from strategy_ml import MLConfirmedStrategy

def main():
    strategy = MLConfirmedStrategy()
    strategy.get_data()
    strategy.egit()

    result = backtest.run(
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
    )

    result.print_summary()

    print("\nToplam getiri yüzdesi:", result.return_pct)
    print("\nTrade sayısı:", len(result.trade_history))


if __name__ == "__main__":
    main()