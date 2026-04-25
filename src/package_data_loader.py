"""
cnlib ile paket halinde gelen training veriyi doğrudan kurulu paket klasöründen bulup yükler.

Neden lazım?
- Bazı sürümlerde strategy.get_data() backtest dışında tam veri yerine eksik/sınırlı veri döndürebiliyor.
- Biz pseudo-unseen, bootstrap ve inspect scriptlerinde gerçek training veriye erişmek istiyoruz.
"""

from pathlib import Path
import pandas as pd
import cnlib


EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        low = str(col).strip().lower()
        if low == "date":
            rename_map[col] = "Date"
        elif low == "open":
            rename_map[col] = "Open"
        elif low == "high":
            rename_map[col] = "High"
        elif low == "low":
            rename_map[col] = "Low"
        elif low == "close":
            rename_map[col] = "Close"
        elif low == "volume":
            rename_map[col] = "Volume"

    out = df.rename(columns=rename_map).copy()
    return out


def _is_ohlcv_frame(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return all(col in cols for col in EXPECTED_COLUMNS)


def _read_file(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _candidate_roots() -> list[Path]:
    pkg_dir = Path(cnlib.__file__).resolve().parent
    return [
        pkg_dir,
        pkg_dir / "data",
        pkg_dir.parent,
        pkg_dir.parent / "data",
    ]


def _candidate_files() -> list[Path]:
    files = []
    seen = set()

    patterns = [
        "**/*_train.parquet",
        "**/*train*.parquet",
        "**/*.parquet",
        "**/*_train.csv",
        "**/*train*.csv",
        "**/*.csv",
    ]

    for root in _candidate_roots():
        if not root.exists():
            continue

        for pattern in patterns:
            for path in root.glob(pattern):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                files.append(path)

    return files


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    temp = _normalize_columns(df)

    if not _is_ohlcv_frame(temp):
        raise ValueError("OHLCV kolonları bulunamadı.")

    temp = temp[EXPECTED_COLUMNS].copy()

    try:
        temp["Date"] = pd.to_datetime(temp["Date"], format="%Y-%m-%d", errors="coerce")
    except Exception:
        temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        temp[col] = pd.to_numeric(temp[col], errors="coerce")

    temp = temp.dropna(subset=EXPECTED_COLUMNS).sort_values("Date").reset_index(drop=True)
    return temp


def load_packaged_training_data() -> dict[str, pd.DataFrame]:
    """
    Kurulu cnlib paketindeki gerçek training veri dosyalarını bulup yükler.
    En uzun geçerli OHLCV dosyaları tutulur.
    """
    datasets: dict[str, pd.DataFrame] = {}

    for path in _candidate_files():
        raw = _read_file(path)
        if raw is None:
            continue

        try:
            df = _clean_ohlcv(raw)
        except Exception:
            continue

        # Çok kısa veri training set değildir
        if len(df) < 100:
            continue

        coin_name = path.stem

        # Aynı isimden birden fazla aday varsa en uzun olanı tut
        if coin_name not in datasets or len(df) > len(datasets[coin_name]):
            datasets[coin_name] = df

    if not datasets:
        raise FileNotFoundError(
            "cnlib paket klasöründe geçerli training OHLCV dosyası bulunamadı. "
            "Gerekirse pip show cnlib ve site-packages/cnlib içini kontrol edin."
        )

    return datasets