"""
Bu dosya cnlib ile gelen training datayı özetlemek için kullanılır.

Amaç:
- coin başına tarih aralığını görmek
- satır sayısını görmek
- toplam getiriyi görmek
- volatiliteyi görmek
- trendli mi, choppy mi anlamaya çalışmak
- strateji seçiminde bize yol göstermek
"""

from pathlib import Path
import pandas as pd

from package_data_loader import load_packaged_training_data


def max_drawdown(close: pd.Series) -> float:
    running_max = close.cummax()
    dd = (close - running_max) / running_max * 100
    return abs(dd.min())


def trendiness_score(close: pd.Series) -> float:
    """
    Basit trendlilik skoru:
    net hareket / toplam mutlak hareket

    1'e yakınsa daha trendli,
    0'a yakınsa daha dalgalı/choppy.
    """
    returns = close.pct_change().dropna()
    if len(returns) == 0:
        return 0.0

    net_move = abs(close.iloc[-1] / close.iloc[0] - 1)
    total_move = returns.abs().sum()

    if total_move == 0:
        return 0.0

    return float(net_move / total_move)


def summarize_coin(coin: str, df: pd.DataFrame) -> dict:
    temp = df.copy()
    temp["Date"] = pd.to_datetime(temp["Date"])
    temp = temp.sort_values("Date").reset_index(drop=True)

    close = temp["Close"].astype(float)
    volume = temp["Volume"].astype(float)

    returns = close.pct_change().dropna()

    summary = {
        "coin": coin,
        "rows": len(temp),
        "start_date": temp["Date"].iloc[0],
        "end_date": temp["Date"].iloc[-1],
        "start_close": round(float(close.iloc[0]), 4),
        "end_close": round(float(close.iloc[-1]), 4),
        "total_return_pct": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2),
        "mean_return_pct": round(float(returns.mean() * 100), 4) if len(returns) else 0.0,
        "volatility_pct": round(float(returns.std() * 100), 4) if len(returns) else 0.0,
        "max_drawdown_pct": round(float(max_drawdown(close)), 2),
        "avg_volume": round(float(volume.mean()), 2),
        "volume_std": round(float(volume.std()), 2),
        "trendiness_score": round(float(trendiness_score(close)), 4),
    }

    return summary


def main():
    coin_data = load_packaged_training_data()

    print("Loaded coins:", list(coin_data.keys()))
    print()

    all_rows = []

    output_dir = Path("data/inspection")
    output_dir.mkdir(parents=True, exist_ok=True)

    for coin, df in coin_data.items():
        print("=" * 80)
        print("COIN:", coin)
        print("Columns:", list(df.columns))
        print("Head:")
        print(df.head(3).to_string(index=False))
        print("\nTail:")
        print(df.tail(3).to_string(index=False))
        print()

        summary = summarize_coin(coin, df)
        all_rows.append(summary)

        # Coin bazlı ham veriyi csv olarak da kaydedelim
        coin_csv_name = coin.replace("/", "_").replace("-", "_")
        df.to_csv(output_dir / f"{coin_csv_name}.csv", index=False)

    summary_df = pd.DataFrame(all_rows)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(output_dir / "training_summary.csv", index=False)
    print(f"\nSummary saved to: {output_dir / 'training_summary.csv'}")


if __name__ == "__main__":
    main()