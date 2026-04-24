"""
Bu dosya projedeki tüm temel sabitleri ve ayarları tutar.

Neden ayrı dosyada?
- Strateji içinde sayı gömmemek için
- Parametreleri tek yerden hızlıca değiştirmek için
- Backtest sırasında hangi ayarlarla çalıştığımızı net görmek için
"""

# Başlangıç sermayesi
INITIAL_CAPITAL = 3000.0

# Hareketli ortalama ayarları
SHORT_MA = 20
LONG_MA = 50

# İsterseniz sonra EMA da test edebiliriz.
# Şimdilik ana strateji SMA ile çalışacak.
USE_EMA = False

# Bollinger Band ayarları
BB_WINDOW = 20
BB_STD = 2

# Hacim ve volatilite filtreleri için rolling window
VOLUME_AVG_WINDOW = 20
VOLATILITY_WINDOW = 20

# Çok sert volatilite varsa yeni işleme girmemek için eşik
# son volatilite > vol_avg * bu katsayı ise guard aktif
VOLATILITY_GUARD_MULTIPLIER = 1.8

# İşlem ayarları
DEFAULT_ALLOCATION = 0.05   # İlk sürümde temkinli allocation
DEFAULT_LEVERAGE = 1        # En güvenlisi: 1x başlamak

# Risk / reward ayarları
STOP_LOSS_PCT = 0.025       # %2.5 stop
RR_RATIO = 2.0              # Minimum 1:2 risk-reward
TAKE_PROFIT_PCT = STOP_LOSS_PCT * RR_RATIO  # %5 take profit

# Fiyat banda çok yapışıkken giriş yapmamak için tamponlar
# Long için üst banda çok yakınsa giriş yok
UPPER_BAND_BUFFER = 0.997

# Short için alt banda çok yakınsa giriş yok
LOWER_BAND_BUFFER = 1.003

# Sağlıklı sinyal için en az kaç bar geçmiş lazım?
MIN_HISTORY = 80