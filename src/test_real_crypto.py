import pandas as pd
from pathlib import Path
from cnlib import backtest
from strategy_ml import MLConfirmedStrategy
from strategy_safe import SafeHybridStrategy

def load_real_crypto_data(data_dir: Path) -> dict:
    coins = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
    data = {}
    for c in coins:
        f = data_dir / f"{c}.parquet"
        if f.exists():
            data[c] = pd.read_parquet(f)
    return data

def main():
    data_dir = Path("real_crypto_data")
    if not data_dir.exists():
        print("Lütfen önce python fetch_binance.py komutunu çalıştırın!")
        return

    full_data = load_real_crypto_data(data_dir)
    print(f"Gerçek kripto verisi yüklendi (Son ~1000 Gün).")
    
    # Veriyi Temsili Olarak Böl: %75 Training, %25 Test
    from data_splits import split_coin_data_by_ratio, attach_fixed_data_loader
    dev_data, test_data = split_coin_data_by_ratio(full_data)
    
    print("\n" + "="*50)
    print("GERÇEK BİTCOİN/ETH/BNB VERİSİ - SAFE STRATEJİ")
    print("="*50)
    safe = SafeHybridStrategy()
    attach_fixed_data_loader(safe, test_data)
    backtest.run(strategy=safe, initial_capital=3000.0).print_summary()
    
    print("\n" + "="*50)
    print("GERÇEK BİTCOİN/ETH/BNB VERİSİ - ML (LOGISTIC REGRESSION)")
    print("="*50)
    ml = MLConfirmedStrategy()
    ml._full_data = dev_data
    ml.prepare_models()
    
    attach_fixed_data_loader(ml, test_data)
    backtest.run(strategy=ml, initial_capital=3000.0).print_summary()

if __name__ == "__main__":
    main()
