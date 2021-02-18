import keys
from binance import client

# Data variables
EMAS = [20, 40, 50, 60, 100, 200]
VOLUME_EMAS = [20, 40, 200]
LAG = 1
INTERVAL = client.Client.KLINE_INTERVAL_15MINUTE
LIMIT = 300

# Crypto pair variables
COIN_INFO = [("btc", "BTCUSDT", 0.85),
             ("eth", "ETHUSDT", 0.85),
             ("bnb", "BNBUSDT", 0.85),
             ("trx", "TRXUSDT", 0.85),
             ("ada", "ADAUSDT", 0.85),
             ("ltc", "LTCUSDT", 0.85),
             ("xrp", "XRPUSDT", 0.85),]

# Keys
API_KEY = keys.api_key
SECRET_KEY = keys.secret_key

# Loop managing variables
WAITING_TIME = 14 * 60

# Testing
TEST = False
CASH = 250.0

# Get all the currency information
def get_currencies(trader, coin_info, interval):
    """
    
    """
    currencies = dict()
    for name, symbol, threshold in coin_info:
        currencies[name] = dict()
        currencies[name]["name"] = name
        currencies[name]["symbol"] = symbol
        currencies[name]["threshold"] = threshold
        currencies[name]["model"] = trader.load_model(
            f"data/{name}/{interval}_{name}.h5"
        )
        currencies[name]["scaler"] = trader.load_scaler(
            f"data/{name}/{interval}_{name}_scaler.pickle"
        )

        print(f"Loaded scaler and model for {name}")

    return currencies


if __name__ == "__main__":
    pass
