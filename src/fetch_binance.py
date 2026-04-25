import urllib.request
import json
import pandas as pd
from pathlib import Path
import os

# Yarışma motoru bu 3 özel ismi arıyor, biz gerçek coinleri bu isimlerin içine "saklayacağız"
# Böylece cnlib motoru hiç kod değiştirmeden bu coinleri trade edecek.
MAPPING = {
    "BTCUSDT": "kapcoin-usd_train",
    "ETHUSDT": "metucoin-usd_train",
    "BNBUSDT": "tamcoin-usd_train"
}

def fetch_binance_klines(symbol, interval="1d", limit=1000):
    """Binance üzerinden kripto para grafiği çeker (örn: son 1000 günlük)"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    
    print(f"[{symbol}] Binance'den indiriliyor...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        
    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume", 
        "Close time", "Quote asset volume", "Number of trades", 
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])
    
    # cnlib formatına dönüştürme
    df["Date"] = pd.to_datetime(df["Open time"], unit='ms')
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
        
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    
    # Eksik verileri ve None durumunu sil
    df = df.dropna()
    return df

def main():
    target_dir = Path("real_crypto_data")
    target_dir.mkdir(exist_ok=True)
    
    for symbol, fake_name in MAPPING.items():
        # Minimum 1500 mum (4 yıl) alabilmek için 2 ayrı limit çekeceğiz ama 
        # API limit=1000 veriyor tek seferde. Basitlik için son 1000 günü çekelim (~3 yıl)
        # 1000 gün = 2.7 Yıl (ML için yeterli eğitim)
        df = fetch_binance_klines(symbol, limit=1000)
        
        # Dosyayı yarışmanın beklediği isimle parquet olarak kaydet
        out_path = target_dir / f"{fake_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"KAYDEDILDI: {symbol} -> {out_path} ({len(df)} mum)")
        
    print("\nTüm gerçek kripto verileri 'real_crypto_data' klasörüne başarıyla kaydedildi!")
    print("Şimdi kodlarını bu klasörü hedef alarak çalıştırabilirsin.")

if __name__ == "__main__":
    main()
