"""
Bu dosya tekrar eden küçük yardımcı fonksiyonları tutar.
Strateji dosyasını gereksiz uzatmamak için burada saklıyoruz.
"""

from typing import Iterable


def has_nan(values: Iterable) -> bool:
    """
    İçeride NaN veya bozuk değer var mı kontrol eder.
    Rolling indikatörlerin ilk satırlarında bu çok işe yarar.
    """
    for value in values:
        if value is None or value != value:  # NaN != NaN olduğu için pratik kontrol
            return True
    return False


def choose_ma_columns(use_ema: bool, short_ma: int, long_ma: int) -> tuple[str, str]:
    """
    Trend filtresinde hangi kolonların kullanılacağını döndürür.
    """
    if use_ema:
        return f"EMA_{short_ma}", f"EMA_{long_ma}"
    return f"SMA_{short_ma}", f"SMA_{long_ma}"


def build_tp_sl(entry: float, direction: int, stop_loss_pct: float, take_profit_pct: float) -> tuple[float, float]:
    """
    Giriş fiyatına göre TP ve SL seviyelerini hesaplar.

    direction:
    1  -> long
    -1 -> short
    """
    if direction == 1:
        stop_loss = entry * (1 - stop_loss_pct)
        take_profit = entry * (1 + take_profit_pct)
        return take_profit, stop_loss

    if direction == -1:
        stop_loss = entry * (1 + stop_loss_pct)
        take_profit = entry * (1 - take_profit_pct)
        return take_profit, stop_loss

    raise ValueError("direction sadece 1 veya -1 olabilir.")