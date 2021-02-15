import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import pickle
import math

from create_features import Features
from binance import client


class Trader(client.Client):
    """
    This class adds functionalities to perform trades in Binance.
    It requires the api and secret key from binance.
    """

    def __init__(self, api_key, secret_key):
        super().__init__(api_key, secret_key)

    def load_model(self, path):
        """
        Loads a pre-trained tensorflow model
        
        Args:
            path: str. the model's path
        
        returns: 
            model: tensorflow model
        """
        model = tf.keras.models.load_model(path)

        return model

    def load_scaler(self, path):
        """
        Loads a serialized sklearn scaler object. Normally 
        StandardScaler() or MinMaxScaler()
        
        Args:
            path: str. The scaler's path
        
        returns:
            scaler: sklearn scaler object
        """
        with open(path, "rb") as f:
            scaler = pickle.load(f)

        return scaler

    def save_scaler(self, path, scaler):
        """
        Serializes a sklearn scaler object.
        
        Args:
            path: str. Where to save the scaler
            scaler: the sklearn scaler object used to 
                transform the model's data
        """
        with open(path, "wb") as f:
            pickle.dump(scaler, f)

    def timestamp2date(self, timestamp):
        """
        converts the binance timestamp (miliseconds) to a
        normal date
        """
        correct_date = datetime.datetime.fromtimestamp(timestamp / 1000)

        return correct_date

    def format_df(self, df):
        """
        Takes the raw klines from binance and adds them the correct
        column names. It also keeps the most relevant columns
        and transforms the string values to numeric values.
        
        Args:
            df: pandas DataFrame. It contains the raw klines from binance
        
        returns:
            df_copy: pandas DataFrame. It contains correct column names
                and correct dtypes.
        """
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

        # Change dtypes
        for column in keep[1:]:
            df_copy[column] = pd.to_numeric(df_copy[column])

        return df_copy

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
        # instantiate the features object
        features = Features()

        df_copy = df.copy()

        if indicators:
            df_copy = features.handle_dates(df_copy)
            df_copy = features.add_rsi(df_copy)
            df_copy = features.add_macd(df_copy)
            df_copy = features.add_apo(df_copy)
            df_copy = features.add_ema(df_copy, emas)
            df_copy = features.add_volume_ema(df_copy, volume_emas)
            df_copy = features.add_bbands(df_copy)
            df_copy = features.add_psar(df_copy)

        if patterns:

            df_copy = features.three_inside_up_down(df_copy)
            df_copy = features.three_line_strike(df_copy)
            df_copy = features.three_stars_south(df_copy)
            df_copy = features.advancing_white_soldiers(df_copy)
            df_copy = features.belt_hold(df_copy)
            df_copy = features.breakaway(df_copy)
            df_copy = features.closing_marubozu(df_copy)
            df_copy = features.counteratack(df_copy)
            df_copy = features.doji_star(df_copy)
            df_copy = features.dragonfly_doji(df_copy)
            df_copy = features.engulfing(df_copy)
            df_copy = features.gravestone_doji(df_copy)
            df_copy = features.hammer(df_copy)
            df_copy = features.hanging_man(df_copy)
            df_copy = features.inverted_hammer(df_copy)
            df_copy = features.matching_low(df_copy)
            df_copy = features.morning_doji_star(df_copy)
            df_copy = features.separating_lines(df_copy)
            df_copy = features.shooting_star(df_copy)
            df_copy = features.unique3river(df_copy)

        # delete columns with na. Normally for emas calculations
        df_copy = df_copy.dropna()
        df_copy = df_copy.reset_index(drop=True)

        return df_copy

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

    def create_splits(self, df, lag=1, pct_split=0.95, scaler=None):
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

    def get_one_prediction_data(self, emas, volume_emas, num_candles=1, **kwargs):
        """
        It gets the last LIMIT prices and adds the features. Then we return 
        the resulting dataframe to be scaled.
        
        Args:
            emas: list. a list of int with the periods for the emas
            volume_emas: list. a list of int with the periods for the volume emas
            num_candles: int. the number of candles or rows that we return
            
        returns:
            data: pandas DataFrame. The candles to make predictions. There are not scaled
            date: pandas Series. The corresponding datetime for the data.

        """
        candles = self.get_klines(**kwargs)
        df = pd.DataFrame(candles)
        df = self.format_df(df)
        df = self.add_features(df, emas, volume_emas)

        # Delete the last row because is the firts seconds of information
        # of the candle that we want to predict
        df.drop(df.tail(1).index, inplace=True)

        # We remove open_time because it is not use for prediction
        open_time = df["open_time"]
        df.drop("open_time", axis=1, inplace=True)

        data = df.tail(num_candles)
        date = open_time.tail(num_candles)

        return data, date

    def model_data(
        self, emas, volume_emas, lag, interval, start_date, scaler, symbol="BTCUSDT"
    ):
        """
        It creates the datasets for training a ml model
        
        Args:
            emas: list. a list of int with the periods for the emas
            volume_emas: list. a list of int with the periods for the volume emas
            lag: int. the price to predict. If 1 it means the next candle
            interval: str. klines interval
            start_date: str. 
            scaler: sklearn scaler object. Normally StandardScaler or MinMaxScaler
            symbol: str.
        
        returns:
            X_train: pandas dataframe. Features for training
            X_valid: pandas dataframe. Features for validation
            y_train: pandas dataframe. Targets for training
            y_valid: pandas dataframe. Targets for validation
            scaler: sklean scaler. Ready to transform new data
        """

        candles = self.get_historical_klines(symbol, interval, start_date)

        df = pd.DataFrame(candles)
        df = self.format_df(df)
        df = self.add_features(df, emas, volume_emas)
        df = self.create_target(df, lag=lag)
        X_train, X_valid, y_train, y_test, scaler = self.create_splits(
            df, lag=lag, scaler=scaler
        )

        # Save files to csv
        X_train.to_csv("X_train.csv", index=False)
        X_valid.to_csv("X_valid.csv", index=False)
        y_train.to_csv("y_train.csv", index=False)
        y_test.to_csv("y_valid.csv", index=False)

        return X_train, X_valid, y_train, y_test, scaler

    def make_prediction(
        self, emas, volume_emas, symbol, interval, limit, scaler, model, num_candles=1
    ):
        """
        It makes a prediction using a pre-trained model
        
        Args:
            emas: list. a list of int with the periods for the emas
            volume_emas: list. a list of int with the periods for the volume emas
            symbol: str.
            interval: str. klines interval
            scaler: sklearn scaler object. The one used for scaling the training data
            model: pre-trained model
            num_candles: int. number of candles to get or the number of predictions to make
            
        returns:
            pred: float. the model's prediction
            open_time: datetime of the prediction
        """
        # Get the data to predict
        data, open_time = self.get_one_prediction_data(
            emas,
            volume_emas,
            symbol=symbol,
            interval=interval,
            limit=limit,
            num_candles=num_candles,
        )

        # Scale the data
        scaled_data = scaler.transform(data)

        # predict with the scaled data
        pred = model.predict(scaled_data)
        pred = pred.squeeze()
        open_time = open_time.values

        return pred, open_time

    def check_filters(self, price, quantity, symbol):
        """
        Fixes the lot_size  and price_filter filters when
        needed.
        Args:
            price: float. the buying or selling price
            quantity: float. amount of units to buy or sell
            symbol: str.
            
        returns:
            price: float. fixed buying or selling price
            quantity: float. fixed amount of units
        """

        for filt in self.get_symbol_info(symbol=symbol)["filters"]:
            if filt["filterType"] == "LOT_SIZE":
                ticks = filt["stepSize"].find("1") - 2
                quantity = math.floor(quantity * 10 ** ticks) / float(10 ** ticks)
                continue

            elif filt["filterType"] == "PRICE_FILTER":
                ticks = filt["tickSize"].find("1") - 2
                price = math.floor(price * 10 ** ticks) / float(10 ** ticks)
                continue
            else:
                continue

        return price, quantity

    def sell_buy(self, coin, buy=True, price=None, quantity=None, cash=None):
        """
        This functions is for testing purposes. When buying, we need to pass the cash
        and when selling we need to pass the quantity.
        
        """
        if price == None:
            info = self.get_ticker(symbol=coin["symbol"])
            price = float(info["lastPrice"])

        if buy:
            quantity = cash / price
            print(f"Bought {quantity} of {coin['name']} at {price}")

            return quantity

        else:
            cash = quantity * price
            print(f"Sold {quantity} of {coin['name']} at {price}")

            return cash


if __name__ == "__main__":
    pass
