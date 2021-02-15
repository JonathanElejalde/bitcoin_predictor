import keys
from binance import client

# Data variables
EMAS = [20, 40, 50, 60, 200]
VOLUME_EMAS = [20, 40, 200]
LAG = 1
INTERVAL = client.Client.KLINE_INTERVAL_5MINUTE
LIMIT = 300

# Crypto pair variables
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "TRXUSDT"]
THRESHOLDS = [0.99, 0.65, 0.65, 0.65]
NAMES = ["btc", "eth", "bnb", "trx"]

# Keys
API_KEY = keys.api_key
SECRET_KEY = keys.secret_key

# Loop managing variables
WAITING_TIME = 4 * 60

# Testing
TEST = True
CASH = 250.0
