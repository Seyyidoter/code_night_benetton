"""
Bu dosya projenin çalıştırıcı dosyasıdır.

Asıl strateji mantığı burada değil, strategy_cnlib.py içindedir.
Burada sadece:
- stratejiyi oluşturuyoruz
- cnlib backtest motorunu çalıştırıyoruz
- sonucu ekrana yazdırıyoruz
"""

from cnlib import backtest

from config import INITIAL_CAPITAL
from strategy_cnlib import BenettonStrategy


def main():
    # Strateji nesnesini oluştur
    strategy = BenettonStrategy()

    # cnlib backtest motorunu çalıştır
    result = backtest.run(
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
    )

    # Özet sonuçları yazdır
    result.print_summary()

    # Ek bilgi de basalım
    print("\nToplam getiri yüzdesi:", result.return_pct)
    print("\nİlk birkaç trade kaydı:")
    print(result.trade_history[:5])


if __name__ == "__main__":
    main()