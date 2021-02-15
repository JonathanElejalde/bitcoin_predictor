import time
import config
import pandas as pd

from trader import Trader
import datetime

# Strategy: Calculate predictions for all coins. If there are more than one above the threshold, we will
# buy the one with the highest prediction probability. Also, when we have a current position, we
# sell that position and calculate all predictions again.


def calculate_max_pred(currencies):

    pred_info = dict()
    preds_df = pd.DataFrame(
        columns=["open_time", "name", "symbol", "threshold", "prediction",]
    )
    for name in config.NAMES:
        coin = currencies[name]

        pred, open_time = trader.make_prediction(
            config.EMAS,
            config.VOLUME_EMAS,
            coin["symbol"],
            config.INTERVAL,
            config.LIMIT,
            coin["scaler"],
            coin["model"],
        )

        # Store the pred_info
        pred_info["open_time"] = open_time
        pred_info["name"] = coin["name"]
        pred_info["symbol"] = coin["symbol"]
        pred_info["threshold"] = coin["threshold"]
        pred_info["prediction"] = pred

        df = pd.DataFrame(pred_info, index=[0])

        # Append to the preds_df
        preds_df = preds_df.append(df, ignore_index=True)

    print(preds_df)

    # Get the predictions equal to the maximum predictions and that are above
    # their threshold
    max_pred = preds_df[
        (preds_df.prediction == preds_df.prediction.max())
        & (preds_df.prediction > preds_df.threshold)
    ]

    return max_pred


def buy(trader, coin, test=False, cash=None):

    balance = float(trader.get_asset_balance(asset="USDT")["free"])
    buy_info = trader.get_ticker(symbol=coin["symbol"])
    # We could try with askPrice, lastPrice as well
    buy_price = float(buy_info["lastPrice"])
    quantity = round(float(balance / buy_price) * 0.995, 6)
    buy_price, quantity = trader.check_filters(buy_price, quantity, coin["symbol"])

    print(f"buy_price and quantity after filters:\n{(buy_price, quantity)}")

    if test:
        balance = cash
        quantity = trader.sell_buy(coin, price=buy_price, cash=balance)

        return quantity

    else:
        buy_order = trader.create_test_order(
            symbol=coin["symbol"],
            side="BUY",
            type="LIMIT",
            timeInForce="GTC",
            quantity=quantity,
            price=buy_price,
        )

        print(f"BUY ORDER:\n{buy_order}")

        return quantity


def sell(trader, coin, quantity, test=False):
    sell_info = trader.get_ticker(symbol=coin["symbol"])
    # We could try with bidPrice, lastPrice as well
    sell_price = float(sell_info["lastPrice"])
    sell_price, quantity = trader.check_filters(sell_price, quantity, coin["symbol"])

    print(f"sell_price and quantity after filters {(sell_price, quantity)}")

    if test:
        cash = trader.sell_buy(coin, buy=False, price=sell_price, quantity=quantity)
        print(f"#### CURRENT CASH: {cash} ####")

        return cash

    else:
        sell_order = trader.create_test_order(
            symbol=coin["symbol"],
            side="SELL",
            type="LIMIT",
            timeInForce="GTC",
            quantity=quantity,
            price=sell_price,
        )

        print(f"SELL ORDER: \n{sell_order}")
        print(f"#### CURRENT CASH: {trader.get_asset_balance(asset='USDT')} ####")


# Instantiate trader
trader = Trader(config.API_KEY, config.SECRET_KEY)

# load models and scalers
currencies = dict()
for i, name in enumerate(config.NAMES):
    currencies[name] = dict()
    currencies[name]["symbol"] = config.SYMBOLS[i]
    currencies[name]["threshold"] = config.THRESHOLDS[i]
    currencies[name]["name"] = name
    currencies[name]["model"] = trader.load_model(
        f"data\\{config.NAMES[i]}\\5m_{config.NAMES[i]}.h5"
    )
    currencies[name]["scaler"] = trader.load_scaler(
        f"data\\{config.NAMES[i]}\\{config.NAMES[i]}_scaler.pickle"
    )

print("########## Models and scalers loaded #############")

# Saves the last minute in which the app maked a prediction
last = None

# Checks if we are in a current position and the coin of the position
position = False
buyed_coin = None

# main loop
while True:
    date = datetime.datetime.now()
    minutes = date.minute
    seconds = date.second

    if (minutes % 5 == 0) and (seconds == 1):
        if last == minutes:
            continue
        else:
            # Wait some seconds to get the final candle
            time.sleep(2)

            # If we have a position, we sell
            if position:
                sell(trader, buyed_coin, quantity, test=config.TEST)

                # update position
                position = False

            # Get the predictions
            max_pred = calculate_max_pred(currencies)

            # If we don't have good predictions, we wait for the next candle
            if len(max_pred) == 0:
                print(
                    f"There are not predictions over the threshold {datetime.datetime.now()}"
                )
                time.sleep(config.WAITING_TIME)
                continue

            # Solve the issue for more than one prediction.
            if len(max_pred) > 1:
                max_pred = max_pred.head(1)

            coin = currencies[max_pred["name"].values[0]]

            quantity = buy(trader, coin, test=config.TEST, cash=config.CASH)

            # update position and currency_name
            position = True
            buyed_coin = coin

            # wait four minutes and start looping again
            time.sleep(config.WAITING_TIME)
            continue

