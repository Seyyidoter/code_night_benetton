from cnlib import backtest
from pathlib import Path
from strategy_ml import MLConfirmedStrategy
from config import DEFAULT_ALLOCATION, DEFAULT_LEVERAGE

def main():
    print("Initializing Advanced Quant Strategy on Unseen Test Set...")
    strategy = MLConfirmedStrategy()
    
    # 1. Train the model on the embedded cnlib default directory (4-year data)
    print("Training model on 4-year historical dataset...")
    # BaseStrategy.egit() or strategy.get_data() retrieves default.
    # Actually wait. egit() reads strategy.get_data() implicitly if it hasn't been called.
    # strategy.get_data() reads from cnlib/data by default.
    strategy.get_data()
    strategy.egit()
    
    # 2. Execute on totally unseen new 5th year data
    print("Starting blind execution on 5th-year unseen parameters...")
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
