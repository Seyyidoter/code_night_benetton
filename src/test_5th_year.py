"""
Bu dosya, makine öğrenmesi (ML) algoritmasının ilk 4 yıllık verinin TAMAMI ile eğitilip, 
tamamen kurgusal (block bootstrap yöntemiyle dev_data içinden üretilmiş) 5. yıl (unseen) 
senaryolarında nasıl tepki vereceğini test eder.
"""

import pandas as pd
from cnlib import backtest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from package_data_loader import load_packaged_training_data
from config import INITIAL_CAPITAL
from strategy_ml import MLConfirmedStrategy
from data_splits import block_bootstrap_coin_data, attach_fixed_data_loader
from utils import compute_max_drawdown


def run_ml_on_future(name: str, strategy, backtest_data: dict, train_data_for_ml: dict):
    # 1. Tam 4 yıllık veriyi kullanarak modeli tam donanımlı eğit
    strategy._full_data = train_data_for_ml
    strategy.prepare_models()
    
    # 2. Sentetik olarak "gelecekteki 5. yıl" verilerini (backtest_data) stratejiye bağla
    attach_fixed_data_loader(strategy, backtest_data)
    
    # 3. Modelin hiç görmediği bu 5. yıl verisinde backtest yap
    result = backtest.run(strategy=strategy, initial_capital=INITIAL_CAPITAL, silent=True)

    return {
        "scenario": name,
        "return_pct": float(result.return_pct),
        "trade_count": len(result.trade_history),
        "max_drawdown_pct": round(compute_max_drawdown(result.portfolio_series), 2),
    }


def main():
    # 1. 4 yıllık tam veri
    full_data = load_packaged_training_data()

    # Varsayılan olarak 5. yıl ~ 393 gün kabul edelim
    # cnlib'teki verinin genelde %25'i 1 yıla denk gelmekte.
    target_5th_year_len = int(len(list(full_data.values())[0]) * 0.25)
    
    scenarios = 5 # 5 adet rastgele gelecekteki 5. yıl olasılığı üretelim
    rows = []
    
    strategy = MLConfirmedStrategy()
    
    print("\n[BILGI] Strateji: MLConfirmedStrategy (Logistic Regression)")
    print(f"[BILGI] Model Eğitilen Veri: Tam 4 Yıl ({len(list(full_data.values())[0])} mum)")
    print(f"[BILGI] Test Edilecek Kurgusal 5. Yıllar: {target_5th_year_len} mum uzunluğunda\n")

    for i in range(scenarios):
        seed = 42 + i # Sabit seed'lerle ama birbirinden farklı 5 olasılık üret

        print(f"5. Yıl Sentetik Senaryo {i+1} oluşturuluyor ve ML backtest koşuluyor...")
        
        synthetic_future = block_bootstrap_coin_data(
            coin_data=full_data,
            target_len=target_5th_year_len,
            seed=seed
        )

        summary = run_ml_on_future(f"Senaryo {i+1}", strategy, synthetic_future, train_data_for_ml=full_data)
        rows.append(summary)


    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("5. YIL KURGUSAL TEST SONUÇLARI (ML STRESS TEST)")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n[ÖZET]")
    print(f"Ortalama Getiri   : +{round(df['return_pct'].mean(), 2)}%")
    print(f"En Kötü Senaryo   : +{round(df['return_pct'].min(), 2)}%")
    print(f"En İyi Senaryo    : +{round(df['return_pct'].max(), 2)}%")


if __name__ == "__main__":
    main()
