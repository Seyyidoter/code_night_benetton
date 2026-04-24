"""
Bu dosya tüm ortak ayarları tutar.

Buradaki amaç:
- parametreleri tek yerden yönetmek
- stratejiler arasında adil kıyas yapmak
- Safe / Fast / ML stratejilerinin aynı omurgayı kullanmasını sağlamak
"""

# Başlangıç sermayesi
INITIAL_CAPITAL = 3000.0

# --- Safe Strategy için MA ayarları ---
SHORT_MA = 20
LONG_MA = 50
USE_EMA = False  # Safe stratejide varsayılan olarak SMA kullanacağız

# --- Fast Strategy için EMA ayarları ---
FAST_EMA_SHORT = 9
FAST_EMA_LONG = 21

# Bollinger Band ayarları
BB_WINDOW = 20
BB_STD = 2

# Hacim filtresi
VOLUME_AVG_WINDOW = 20
VOLUME_SPIKE_MULTIPLIER = 1.5

# Volatilite guard ayarları
VOLATILITY_WINDOW = 20
VOLATILITY_GUARD_MULTIPLIER = 1.8

# İşlem ayarları
DEFAULT_ALLOCATION = 0.05
FAST_ALLOCATION = 0.07
ML_ALLOCATION = 0.05

DEFAULT_LEVERAGE = 1

# Risk / reward ayarları
STOP_LOSS_PCT = 0.025
RR_RATIO = 2.0
TAKE_PROFIT_PCT = STOP_LOSS_PCT * RR_RATIO

FAST_STOP_LOSS_PCT = 0.02
FAST_TAKE_PROFIT_PCT = FAST_STOP_LOSS_PCT * RR_RATIO

ML_STOP_LOSS_PCT = 0.025
ML_TAKE_PROFIT_PCT = ML_STOP_LOSS_PCT * RR_RATIO

# Banda çok yapışıkken giriş yapmamak için tamponlar
UPPER_BAND_BUFFER = 0.997
LOWER_BAND_BUFFER = 1.003

# Gerekli minimum geçmiş veri
MIN_HISTORY = 80

# ML ayarları
ML_PROBA_THRESHOLD = 0.55