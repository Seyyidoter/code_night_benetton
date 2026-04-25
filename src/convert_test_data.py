import pandas as pd
from pathlib import Path
import os

# Map CSVs to cnlib's expected naming convention
FILE_MAP = {
    "/Users/alikemaltopak/Desktop/code_night benetton/synthetic_kapcoin_test_year.csv": "kapcoin-usd_train.parquet",
    "/Users/alikemaltopak/Desktop/code_night benetton/synthetic_tamcoin_test_year.csv": "tamcoin-usd_train.parquet",
    "/Users/alikemaltopak/Desktop/code_night benetton/synthetic_metucoin_test_year.csv": "metucoin-usd_train.parquet"
}

target_dir = Path("/Users/alikemaltopak/Desktop/code_night benetton/code_night_benetton/src/test_data")
target_dir.mkdir(parents=True, exist_ok=True)

for csv_path, parquet_name in FILE_MAP.items():
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure Date parsing if necessary, though cnlib doesn't strictly break if string, but let's be safe.
    # We will just save it to parquet as is.
    out_path = target_dir / parquet_name
    df.to_parquet(out_path)
    print(f"Saved to {out_path}")

print("All test data converted to Parquet.")
