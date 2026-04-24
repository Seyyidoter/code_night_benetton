"""
Bu dosya stratejide tekrar kullanılan küçük yardımcı fonksiyonları tutar.

Neden ayrı dosyada?
- strategy_cnlib.py gereksiz büyümesin
- küçük hesaplar ve ortak kontroller tek yerde dursun
"""

from typing import Iterable


def has_nan(values: Iterable) -> bool:
    """
    İçinde NaN / None benzeri bozuk değer var mı kontrol eder.
    Basit güvenlik katmanı.
    """
    for value in values:
        # NaN için pratik kontrol: NaN != NaN
        if value is None or value != value:
            return True
    return False


def choose_ma_columns(use_ema: bool, short_ma: int, long_ma: int) -> tuple[str, str]:
    """
    Kullanılan trend filtresine göre hangi kolonların okunacağını döner.
    """
    if use_ema:
        return f"EMA_{short_ma}", f"EMA_{long_ma}"
    return f"SMA_{short_ma}", f"SMA_{long_ma}"


def build_tp_sl(entry: float, direction: int, stop_loss_pct: float, take_profit_pct: float) -> tuple[float, float]:
    """
    Giriş fiyatına göre TP ve SL seviyelerini üretir.

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