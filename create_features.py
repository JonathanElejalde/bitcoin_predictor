import talib
import pandas as pd
import fastai.tabular as ft

"""
The the outputs by the pattern recognition functions in talib:

- +200 bullish pattern with confirmation
- +100 bullish pattern (most cases)
- 0 none
- -100 bearish pattern
- -200 bearish pattern with confirmation
"""


class Features:

    """
    This class creates multiple features such as indicators
    and candle patterns to a dataframe. It has to be open_time,
    open, high, low, close, volume, number_of_trades
    """

    def handle_dates(self, df, date_column="open_time", add_column=True):
        col = df[date_column]
        df = ft.add_datepart(df, date_column)
        # add the column again
        df[date_column] = col

        return df

    # INDICATORS
    def add_rsi(self, df):
        rsi = talib.RSI(df.close)
        df["rsi"] = rsi

        return df

    def add_macd(self, df):
        macd, macdsignal, macdhist = talib.MACD(
            df.close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        df["macd"] = macd
        df["macdsignal"] = macdsignal
        df["macdhist"] = macdhist

        return df

    def add_apo(self, df):
        apo = talib.APO(df.close, fastperiod=12, slowperiod=26, matype=0)
        df["apo"] = apo

        return df

    def add_ema(self, df, emas):

        for period in emas:
            ema = talib.EMA(df.close, timeperiod=period)
            df[f"ema_{period}"] = ema

        return df

    def add_volume_ema(self, df, emas):
        for period in emas:
            ema = talib.EMA(df.volume, timeperiod=period)
            df[f"ema_{period}_volume"] = ema

        return df

    def add_bbands(self, df):
        upperband, middleband, lowerband = talib.BBANDS(
            df.close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0
        )
        df["bb_upper"] = upperband
        df["bb_middleband"] = middleband
        df["bb_lower"] = lowerband

        return df

    def add_psar(self, df):
        psar = talib.SAR(df.high, df.low, acceleration=0, maximum=0)
        df["parabolic_sar"] = psar

        return df

    # PATTERNS
    def three_inside_up_down(self, df):
        up_down = talib.CDL3INSIDE(df.open, df.high, df.low, df.close)
        df["3inside_up_down"] = up_down

        return df

    def three_line_strike(self, df):
        line_strike = talib.CDL3LINESTRIKE(df.open, df.high, df.low, df.close)
        df["3line_strike"] = line_strike

        return df

    def three_stars_south(self, df):
        stars_south = talib.CDL3STARSINSOUTH(df.open, df.high, df.low, df.close)
        df["3stars_south"] = stars_south

        return df

    def advancing_white_soldiers(self, df):
        advancing_soldiers = talib.CDL3WHITESOLDIERS(df.open, df.high, df.low, df.close)
        df["3advancing_white_soldiers"] = advancing_soldiers

        return df

    def belt_hold(self, df):
        belt_hold = talib.CDLBELTHOLD(df.open, df.high, df.low, df.close)
        df["belt_hold"] = belt_hold

        return df

    def breakaway(self, df):
        breakaway = talib.CDLBREAKAWAY(df.open, df.high, df.low, df.close)
        df["breakaway"] = breakaway

        return df

    def closing_marubozu(self, df):
        marubozu = talib.CDLCLOSINGMARUBOZU(df.open, df.high, df.low, df.close)
        df["closing_marubozu"] = marubozu

        return df

    def counteratack(self, df):
        counteratack = talib.CDLCOUNTERATTACK(df.open, df.high, df.low, df.close)
        df["counteratack"] = counteratack

        return df

    def doji_star(self, df):
        doji_star = talib.CDLDOJISTAR(df.open, df.high, df.low, df.close)
        df["doji_star"] = doji_star

        return df

    def dragonfly_doji(self, df):
        dragonfly = talib.CDLDRAGONFLYDOJI(df.open, df.high, df.low, df.close)
        df["dragonfly_doji"] = dragonfly

        return df

    def engulfing(self, df):
        engulfing = talib.CDLENGULFING(df.open, df.high, df.low, df.close)
        df["engulfing"] = engulfing

        return df

    def gravestone_doji(self, df):
        gravestone = talib.CDLGRAVESTONEDOJI(df.open, df.high, df.low, df.close)
        df["gravestone_doji"] = gravestone

        return df

    def hammer(self, df):
        hammer = talib.CDLHAMMER(df.open, df.high, df.low, df.close)
        df["hammer"] = hammer

        return df

    def hanging_man(self, df):
        hanging = talib.CDLHANGINGMAN(df.open, df.high, df.low, df.close)
        df["hanging_man"] = hanging

        return df

    def inverted_hammer(self, df):
        inverted_hammer = talib.CDLINVERTEDHAMMER(df.open, df.high, df.low, df.close)
        df["inverted_hammer"] = inverted_hammer

        return df

    def matching_low(self, df):
        low = talib.CDLMATCHINGLOW(df.open, df.high, df.low, df.close)
        df["matching_low"] = low

        return df

    def morning_doji_star(self, df):
        morning = talib.CDLMORNINGDOJISTAR(
            df.open, df.high, df.low, df.close, penetration=0
        )
        df["morning_doji_star"] = morning

        return df

    def separating_lines(self, df):
        lines = talib.CDLSEPARATINGLINES(df.open, df.high, df.low, df.close)
        df["separating_lines"] = lines

        return df

    def shooting_star(self, df):
        shooting = talib.CDLSHOOTINGSTAR(df.open, df.high, df.low, df.close)
        df["shooting_star"] = shooting

        return df

    def unique3river(self, df):
        river = talib.CDLUNIQUE3RIVER(df.open, df.high, df.low, df.close)
        df["unique3_river"] = river

        return df

