"""
Bu dosya veri bölme ve sentetik test üretme araçlarını tutar.

Amaç:
1. 4 yıllık training veriyi zaman bazlı bölmek
   - ilk %75 = geliştirme
   - son %25 = pseudo unseen test

2. Tarihsel yapıyı koruyan block-bootstrap senaryoları üretmek

3. cnlib backtest.run() için sabit veri enjekte etmek
"""

import random
import pandas as pd

from config import TRAIN_RATIO, BOOTSTRAP_BLOCK_MIN, BOOTSTRAP_BLOCK_MAX, BOOTSTRAP_SEED


def normalize_coin_data(coin_data: dict) -> dict:
    """
    Coin verilerini tarih sırasına göre normalize eder.
    """
    normalized = {}

    for coin, df in coin_data.items():
        temp = df.copy()
        temp["Date"] = pd.to_datetime(temp["Date"])
        temp = temp.sort_values("Date").reset_index(drop=True)
        normalized[coin] = temp

    return normalized


def split_coin_data_by_ratio(coin_data: dict, train_ratio: float = TRAIN_RATIO) -> tuple[dict, dict]:
    """
    Coin verilerini zaman bazlı olarak ikiye böler.
    İlk kısım geliştirme, son kısım pseudo unseen test.
    """
    normalized = normalize_coin_data(coin_data)

    dev_data = {}
    test_data = {}

    for coin, df in normalized.items():
        split_idx = max(1, int(len(df) * train_ratio))
        dev_data[coin] = df.iloc[:split_idx].reset_index(drop=True)
        test_data[coin] = df.iloc[split_idx:].reset_index(drop=True)

    return dev_data, test_data


def set_strategy_coin_data(strategy, coin_data: dict):
    """
    cnlib içinde asıl kullanılan backing field _full_data gibi görünüyor.
    Güvenli olmak için birkaç olası field'i birlikte dolduruyoruz.
    """
    fixed_copy = {coin: df.copy().reset_index(drop=True) for coin, df in coin_data.items()}

    # cnlib backtest'in baktığı ana alan
    strategy.__dict__["_full_data"] = {coin: df.copy() for coin, df in fixed_copy.items()}

    # Bazı strategy/property implementasyonları için ek yedek alanlar
    strategy.__dict__["_coin_data"] = {coin: df.copy() for coin, df in fixed_copy.items()}
    strategy.__dict__["coin_data"] = {coin: df.copy() for coin, df in fixed_copy.items()}

    return strategy


def attach_fixed_data_loader(strategy, fixed_coin_data: dict):
    """
    cnlib backtest.run() strategy.get_data(data_dir) diye çağırıyor.
    O yüzden override edilen fonksiyon argüman kabul etmeli.
    Ayrıca _full_data'yı doldurmalı.
    """
    fixed_copy = {coin: df.copy().reset_index(drop=True) for coin, df in fixed_coin_data.items()}

    def _fixed_get_data(*args, **kwargs):
        set_strategy_coin_data(strategy, fixed_copy)

    strategy.get_data = _fixed_get_data
    return strategy


def _build_common_block_plan(length: int, target_len: int, block_min: int, block_max: int, seed: int) -> list[tuple[int, int]]:
    """
    Tüm coinlerde aynı tarih bloklarını kullanmak için ortak blok planı üretir.
    Böylece coinler arası ilişki kısmen korunur.
    """
    rng = random.Random(seed)
    plan = []
    total = 0

    while total < target_len:
        block_size = rng.randint(block_min, block_max)
        start = rng.randint(0, max(0, length - block_size))
        plan.append((start, block_size))
        total += block_size

    return plan


def block_bootstrap_coin_data(
    coin_data: dict,
    target_len: int | None = None,
    block_min: int = BOOTSTRAP_BLOCK_MIN,
    block_max: int = BOOTSTRAP_BLOCK_MAX,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Tarihsel veriden bloklar seçerek sentetik bir gelecek üretir.

    Tamamen random white-noise üretmez.
    Onun yerine mevcut tarihsel OHLCV bloklarını yeniden örnekler.
    """
    normalized = normalize_coin_data(coin_data)

    min_len = min(len(df) for df in normalized.values())
    if target_len is None:
        target_len = max(block_max, int(min_len * (1 - TRAIN_RATIO)))

    block_plan = _build_common_block_plan(
        length=min_len,
        target_len=target_len,
        block_min=block_min,
        block_max=block_max,
        seed=seed,
    )

    synthetic = {}

    for coin, df in normalized.items():
        chunks = []

        for start, size in block_plan:
            chunk = df.iloc[start:start + size].copy()
            chunks.append(chunk)

        out = pd.concat(chunks, ignore_index=True).iloc[:target_len].copy()

        # Tarihleri düzgün ve artan halde tutmak için limit koruması
        available_dates = min(target_len, len(df))
        out["Date"] = df["Date"].iloc[:available_dates].reset_index(drop=True)
        # Pad with NaT if needed, but in practice handled correctly
        synthetic[coin] = out.reset_index(drop=True)

    return synthetic