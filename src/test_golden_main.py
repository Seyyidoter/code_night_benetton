from cnlib import backtest
from pathlib import Path
from strategy_safe_golden_cross import SafeHybridStrategy

def main():
    print("Testing Golden Cross on Unseen Test Set...")
    strategy = SafeHybridStrategy()
    
    test_data_dir = Path(__file__).parent / "test_data"
    
    result = backtest.run(
        strategy=strategy,
        initial_capital=3000.0,
        start_candle=0,
        data_dir=test_data_dir
    )

    result.print_summary()

if __name__ == "__main__":
    main()
