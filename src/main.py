"""
Bu dosya yarışma için nihai stratejiyi çalıştırır.

Ana yarışma konfigürasyonu varsayılan olarak MLConfirmedStrategy (Logistic Regression) + Leverage 2 olarak belirlenmiştir.
Yedek strateji (SAFE) gerektiğinde strateji import'u değiştirilerek kullanılabilir.
"""

from cnlib import backtest

from config import INITIAL_CAPITAL
from strategy_ml import MLConfirmedStrategy


def main():
    strategy = MLConfirmedStrategy()
    
    print("Makine Öğrenimi modelleri eğitiliyor...")
    strategy.get_data()
    strategy.prepare_models()
    print("Modeller hazır. Backtest başlıyor...")

    result = backtest.run(
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
    )

    result.print_summary()
    print("\nToplam getiri yuzdesi:", result.return_pct)
    print("\nTrade sayisi:", len(result.trade_history))


if __name__ == "__main__":
    main()