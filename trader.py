import pandas as pd
import numpy as np
import datetime

from create_features import Features
from binance import client


class Trader(client.Client):
    """
    This class adds functionalities to perform trades in Binance.
    It requires the api and secret key from binance.
    """

    def __init__(self, api_key, secret_key):
        super().__init__(api_key, secret_key)

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

    def get_one_prediction_data(self, emas, volume_emas, **kwargs):
        """
        It gets the last 500 candles and adds the features. Then we return 
        the resulting dataframe to be scaled.
        
        Note: We use several candles because we need to calculate
        several features that need previous candles
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

        return df.tail(1), open_time.tail(1)
    
    


if __name__ == "__main__":
    pass

