from backtrader.strategy import SignalStrategy
import pandas as pd
import numpy as np
import pyfolio as pf
import backtrader as bt
import matplotlib.pyplot as plt
import seaborn as sns

from backtrader.feeds import PandasData


class BackTester:
    """
    The index of the dataframe must be a datetime type. thus, 
    the columns that it is specting are open_time, open, high, low, close, 
    volume, returns (true values), direction (it could be in probabilities or classification)
    """

    def __init__(self, df, prediction):
        """
        df: Pandas dataframe
        prediction: str. The column name of the predictions
        """
        self.df = df
        self.prediction = prediction

        # instantiate SignalData class
        self.data = SignalData(self.prediction, dataname=self.df)

    def run_backtest(self, symbol, cash, commission=0.001):

        # instantiate Cerebro, add strategy, data, initial cash, commission and pyfolio for performance analysis
        cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
        cerebro.addstrategy(MLStrategy)
        cerebro.adddata(self.data, name=symbol)
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=comission)
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

        # run the backtest
        print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
        backtest_result = cerebro.run()
        print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

        return backtest_result

    def show_performance(self, backtest_result):
        # Extract inputs for pyfolio
        strat = backtest_result[0]
        pyfoliozer = strat.analyzers.getbyname("pyfolio")
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        returns.name = "Strategy"

        # get performance statistics for strategy
        stats = pf.show_perf_stats(returns)

        print(stats)

    def plot_cumulative_returns(self, probs=False, threshold=0.5):

        if probs:
            self.df["position_strategy"] = np.where(
                self.df[self.prediction] > threshold, 1, -1
            )
        else:
            self.df["position_strategy"] = np.where(self.df[self.prediction] > 0, 1, -1)

        self.df["strategy_returns"] = self.df["position_strategy"] * self.df["returns"]

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(14, 6))
        ax.plot(self.df.returns.cumsum().apply(np.exp), label="Buy and Hold")
        ax.plot(
            self.df.strategy_returns.cumsum().apply(np.exp), label="Strategy returns"
        )
        ax.set(title="BTCUSDT Buy and Hold vs. Strategy", ylabel="Cumulative Returns")
        ax.grid(True)
        ax.legend()
        plt.yscale("log")
        plt.show()


# class to define the columns we will provide


class SignalData(PandasData):
    """
    Define pandas DataFrame structure
    """

    def __init__(self, prediction):
        """
        prediction: str. target column name to use in
            the strategy
        """
        OHLCV = ["open", "high", "low", "close", "volume"]
        columns = OHLCV + ['direction']

        # create lines
        lines = tuple(columns)

        # define parameters
        params = {c: -1 for c in columns}
        params.update({"datetime": "open_time"})
        params = tuple(params.items())


# Define backtesting strategy class
class MLStrategy(bt.Strategy):
    """
    Strategy: 
    1.	Buy when the predicted value is +1 and sell (only if stock is in possession) when the predicted value is 0.
    2.	All-in strategy—when creating a buy order, buy as many shares as possible.
    3.	Short selling is not allowed
    """

    params = dict()

    def __init__(self, prediction):
        """
        prediction: str. It is the name of the target column
            it should be 1 for buying and 0 for selling
        """

        # keep track of the data
        self.data_predicted = self.datas[0][prediction]
        self.data_open = self.datas[0].open
        self.data_close = self.datas[0].close

        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

    # logging function
    def log(self, txt):
        """Logging function"""
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f"{dt}, {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # order already submitted/accepted - no action required
            return

        # report executed order
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}"
                )
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(
                    f"SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}"
                )

        # report failed order
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Failed")

        # set no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(
            f"OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}"
        )

    # We have set cheat_on_open = True.This means that we calculated the signals on day t's close price,
    # but calculated the number of shares we wanted to buy based on day t+1's open price.
    def next_open(self):

        if not self.position:
            if self.data_predicted == 1:
                # calculate the max number of shares ('all-in')
                size = int(self.broker.getcash() / self.datas[0].open)

                # buy order
                self.log(
                    f"BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}"
                )
                self.buy()
        else:
            if self.data_predicted == 0:
                # sell order
                self.log(f"SELL CREATED --- Size: {self.position.size}")
                self.sell(size=self.position.size)

