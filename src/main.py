"""
Bu dosya finalde seçilen stratejiyi çalıştırmak için kullanılacak.

Şimdilik varsayılan olarak Safe stratejiyi çalıştırıyoruz.
Sonra compare_strategies.py sonucuna göre bunu Fast veya ML ile değiştirebiliriz.
"""

from cnlib import backtest

from config import INITIAL_CAPITAL
from strategy_safe import SafeHybridStrategy


def main():
    strategy = SafeHybridStrategy()
    result = backtest.run(
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
    )

    result.print_summary()

    print("\nToplam getiri yüzdesi:", result.return_pct)
    print("\nTrade sayısı:", len(result.trade_history))


if __name__ == "__main__":
    main()