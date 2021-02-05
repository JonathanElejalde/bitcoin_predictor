import requests
import hashlib
import hmac
import time
import datetime
import json
import pandas as pd
import numpy as np

from create_features import Features
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from urllib.parse import urljoin, urlencode


class Trader(Features):
    BASE_URL = "https://api.binance.com"

    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.headers = {"X-MBX-APIKEY": self.api_key}

    def _signature(self, params):
        """
        Creates a signature needed for some request to
        binance API
        """
        querystring = urlencode(params)
        signature = hmac.new(
            secret.encode("utf-8"), querystring.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return signature

    def timestamp2date(self, timestamp):
        """
        converts the binance timestamp (miliseconds) to a 
        normal date
        """
        correct_date = datetime.datetime.fromtimestamp(timestamp / 1000)

        return correct_date

    def server_time(self):
        path = "/api/v1/time"
        url = urljoin(self.BASE_URL, path)
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            return data["serverTime"]
        else:
            print("Problems with the status code")

    def date2timestamp(self, date_string, date_format="%d/%m/%Y"):
        date = datetime.datetime.strptime(date_string, date_format)
        timestamp = datetime.datetime.timestamp(date)

        return timestamp

    def get_price(self, symbol="BTCUSDT"):
        """
        Gets the symbol's current priced
        """
        path = "/api/v3/ticker/price"
        params = {"symbol": symbol}
        r = requests.get(self.BASE_URL + path, headers=self.headers, params=params)
        data = r.json()
        price = data["price"]

        return price

    def get_candles(
        self,
        symbol="BTCUSDT",
        limit=1000,
        interval="4h",
        start_time=None,
        end_time=None,
    ):
        """
        Gets the candles from a symbol. The API has a limit of 1000 candles
        for request.
        Args:
            symbol: str. a pair of currencies
            limit: int. max 1000 the binance default is 500
            interval: str. the candle's interval first the number
                then the letter m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
            start_time: int. timestamp in miliseconds
            end_time: int. timestamp in milisencods
        """

        path = "/api/v3/klines"
        params = {
            "symbol": symbol,
            "limit": limit,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
        }

        r = requests.get(self.BASE_URL + path, params=params)

        return r.json()

    def get_historical_data(self, start_time, end_time=None, interval="4h", **kwargs):
        """
        Gest all the candles starting from a specific day until 
        end_time. If end_time is None, it will get
        the candles from the start_time until present
        """
        historical_data = []

        if end_time == None:
            # current timestamp
            end_time = self.server_time()

        # We add 2 min (120000), if the end_time is higher than the stating time
        # It means that we are still retrieving all data, and if it is less it means that we are
        # in the current date and hour.

        while end_time > (start_time + 120000):
            candles = self.get_candles(
                start_time=start_time, interval=interval, **kwargs
            )
            historical_data.append(candles)

            # update the start_time, we add 1 second to the time where the last candle ended.
            start_time = historical_data[-1][-1][6] + 1

        return historical_data

    def _format_df(self, df):
        df_copy = df.copy()
        # These are the column names from binance api data
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]

        df_copy.columns = columns

        # Columns that we are going to keep
        keep = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "number_of_trades",
        ]
        df_copy = df_copy[keep]

        # Change the timestamp values to real dates.
        df_copy["open_time"] = df_copy.open_time.apply(self.timestamp2date)
        df_copy["open_time"] = pd.to_datetime(df_copy["open_time"])

        return df_copy

    def candles2dataframe(self, candles):
        """
        Converts a list of candles to a pandas 
        DataFrame
        """

        df = pd.DataFrame(candles[0])
        for candles in candles[1:]:
            df_candles = pd.DataFrame(candles)
            df = pd.concat([df, df_candles], ignore_index=True)

        df = self._format_df(df)

        return df

    def get_order_book(self, symbol="BTCUSDT", limit=5):
        path = "/api/v1/depth"
        params = {"symbol": symbol, "limit": limit}

        url = urljoin(self.BASE_URL, path)
        r = requests.get(url, headers=self.headers, params=params)
        data = r.json()
        data = json.dumps(data, indent=2)

        return data

    def create_order(
        self,
        symbol,
        side,
        order_type,
        quantity,
        price,
        recv_window=5000,
        time_in_force="GTC",
    ):
        """
        Args:
            symbol: str.
            side: str. It can be BUY or SELL
            order_type: str. options LIMIT, MARKET, STOP_LOSS,
                STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT,
                LIMIT_MARKER
            quantity: int/float
            price: int/float
            recv_window: int. Amount of milliseconds to process the order or be
                rejected by the server
            time_in_force: str. How long an order will be active
                before expiration.
                GTC: Good Til Canceled
                IOC: Immediate Or Cancel 
                FOK: Fill or Kill
                
        returns:
            data: json response from the binance api
        """

        path = "/api/v3/order"
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "timeInForce": time_in_force,
            "quantity": quantity,
            "price": price,
            "recvWindow": 5000,
            "timestamp": timestamp,
        }

        params["signature"] = self._signature(params)
        url = urljoin(self.BASE_URL, path)
        r = requests.post(url, headers=self.headers, params=params)
        data = r.json()

        return data

    def get_order(self, symbol, order_id, recv_window=5000):
        """
        Get the state of a placed order
        
        Args:
            symbol: str.
            order_id: int. 
            recv_window: int. Amount of milliseconds to process the order or be
                rejected by the server
                
        returns:
            data: json response from binance api
        """
        path = "/api/v3/order"
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "orderId": order_id,
            "recvWindow": recv_window,
            "timestamp": timestamp,
        }

        params["signature"] = self._signature(params)

        url = urljoin(self.BASE_URL, path)
        r = requests.get(url, headers=self.headers, params=params)
        data = r.json()

        return data

    def delete_order(self, symbol, order_id, recv_window=5000):
        """
        Deletes a placed order
        
        Args:
            symbol: str.
            orderId: int
            recv_window: int. Amount of milliseconds to process the order or be
                rejected by the server
        
        returns:
            data: json response from binance api
        """
        path = "/api/v3/order"
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "orderId": order_id,
            "recvWindow": recv_window,
            "timestamp": timestamp,
        }

        params["signature"] = self._signature(params)

        url = urljoin(self.BASE_URL, path)
        r = requests.delete(url, headers=self.headers, params=params)
        data = r.json()

        return data

    def add_features(self, df, emas, volume_emas, indicators=True, patterns=True):
        """
        Creates new features using the Feature class.
        We can add just technical indicators or candle 
        patterns or both.
        
        Args:
            df: pandas dataframe (columns: open_time, open, high, 
                low, close, volume, number_of_trades)
            emas: list of int. With the different ema periods
            volume_emas: list of int. With the different ema periods
            indicators: bool
            patterns: bool
            
        returns:
            df_copy: pandas dataframe. Contains the initial columns
                plus the new
        """
        df_copy = df.copy()

        if indicators:
            df_copy = self.handle_dates(df_copy)
            df_copy = self.add_rsi(df_copy)
            df_copy = self.add_macd(df_copy)
            df_copy = self.add_apo(df_copy)
            df_copy = self.add_ema(df_copy, emas)
            df_copy = self.add_volume_ema(df_copy, volume_emas)
            df_copy = self.add_bbands(df_copy)
            df_copy = self.add_psar(df_copy)

        if patterns:

            df_copy = self.three_inside_up_down(df_copy)
            df_copy = self.three_line_strike(df_copy)
            df_copy = self.three_stars_south(df_copy)
            df_copy = self.advancing_white_soldiers(df_copy)
            df_copy = self.belt_hold(df_copy)
            df_copy = self.breakaway(df_copy)
            df_copy = self.closing_marubozu(df_copy)
            df_copy = self.counteratack(df_copy)
            df_copy = self.doji_star(df_copy)
            df_copy = self.dragonfly_doji(df_copy)
            df_copy = self.engulfing(df_copy)
            df_copy = self.gravestone_doji(df_copy)
            df_copy = self.hammer(df_copy)
            df_copy = self.hanging_man(df_copy)
            df_copy = self.inverted_hammer(df_copy)
            df_copy = self.matching_low(df_copy)
            df_copy = self.morning_doji_star(df_copy)
            df_copy = self.separating_lines(df_copy)
            df_copy = self.shooting_star(df_copy)
            df_copy = self.unique3river(df_copy)

        # delete columns with na. Normally for emas calculations
        df_copy = df_copy.dropna()
        df_copy = df_copy.reset_index(drop=True)

        return df_copy

    def save_df(self, df, filename):
        df.to_csv(filename, index=False)

    def create_target(self, df, lag=1):
        """
        Calculates the log returns using the requested lag.
        It also creates a signal column that is used for
        classification.
        
        Args:
            df: pandas dataframe
            lag: int
        
        returns:
            df_copy: pandas dataframe. The initial dataframe
                plus two new columns returns_lag, signal_lag
        """

        df_copy = df.copy()
        df_copy[f"returns_{lag}"] = np.log(df_copy.close / df_copy.close.shift(lag))
        df_copy[f"signal_{lag}"] = np.where(df_copy[f"returns_{lag}"] > 0, 1, 0)

        # Check for dropna
        df_copy.dropna(inplace=True)

        return df_copy

    def create_splits(df, lag=1, pct_split=0.95, scaler=None):
        """
        Creates the training and validation splits for training
        the model.
        
        Args:
            df: pandas dataframe. Dataframe with the final columns
                to train
            lag: int. To retrieve the correct targets
            pct_split: float. Train percent to keep
            scaler: sklearn.preprocessing. A scaler to normalize the data
                helpful for neural nets.
                
        returns:
            train: pandas dataframe. X_train data
            test: pandas dataframe. X_valid data
            train_targets: pandas dataframe. y_train data
            test_targets: pandas dataframe. y_valid data
            scaler: sklearn.preprocessing. Scaler used later for inverse transforme
                the data.
        """
        df_copy = df.copy()

        # Firts separate the target
        # also add open_time for backtesting
        target_columns = [f"returns_{lag}", f"signal_{lag}", "open_time"]
        targets = df_copy[target_columns]
        df_copy.drop(target_columns, axis=1, inplace=True)
        columns = df_copy.columns

        split = int(len(df_copy) * pct_split)
        train = df_copy.iloc[:split]
        train_targets = targets.iloc[:split]
        test = df_copy.iloc[split:]
        test_targets = targets.iloc[split:]

        if scaler:
            train = scaler.fit_transform(train)
            test = scaler.transform(test)
            train = pd.DataFrame(train, columns=columns)
            test = pd.DataFrame(test, columns=columns)

        print(f"train shape: {train.shape}")
        print(f"test shape: {test.shape}")

        return train, test, train_targets, test_targets, scaler


if __name__ == "__main__":
    pass
