import shift
from time import sleep
from datetime import datetime, timedelta
import datetime as dt
from threading import Thread
import logging



class CirList:
    def __init__(self, length):
        self.size = length
        self._table = [None] * length
        self.idx = 0
        self._counter = 0

    def insertData(self, data):
        self._counter += 1
        self._table[self.idx] = data
        self.idx = (self.idx + 1) % self.size

    def getData(self):
        tail = self._table[0:self.idx]
        head = self._table[self.idx:]
        ret = head + tail
        return ret.copy()

    def isFull(self):
        return self._counter >= self.size

    def __repr__(self):
        return str(self.getData())


def calculate_moving_average(prices):
    if prices.isFull():
        return sum(prices.getData()) / prices.size
    return None

def check_crossover(fast_ma, slow_ma, previous_fast_ma, previous_slow_ma):
    if fast_ma > slow_ma and previous_fast_ma <= previous_slow_ma:
        return "buy"
    elif fast_ma < slow_ma and previous_fast_ma >= previous_slow_ma:
        return "sell"
    return None


def crossover_strategy(trader: shift.Trader, ticker: str, endtime):
    # Strategy parameters
    fast_ma_days = 5
    slow_ma_days = 20


    # Initialize CirList for prices
    fast_prices = CirList(fast_ma_days)
    slow_prices = CirList(slow_ma_days)
    previous_fast_ma = 0
    previous_slow_ma = 0
    previous_mid_price = 0

    while trader.get_last_trade_time() < endtime:
        # Fetch the best bid and ask prices for the ticker
        best_price = trader.get_best_price(ticker)
        best_bid = best_price.get_bid_price()
        best_ask = best_price.get_ask_price()

        # Calculate the mid-price
        midprice = (best_bid + best_ask) / 2


        # Insert the mid-price into both CirLists
        if midprice != previous_mid_price:
            fast_prices.insertData(midprice)
            slow_prices.insertData(midprice)

        previous_mid_price = midprice

        # Calculate moving averages if possible
        fast_ma = calculate_moving_average(fast_prices)
        slow_ma = calculate_moving_average(slow_prices)

        # # Print updates as soon as they're calculated
        # print(f"{ticker} - New Midprice: {midprice}")
        # print(f"{ticker} - Fast MA: {fast_ma}")
        # print(f"{ticker} - Slow MA: {slow_ma}")
        # print("Buying Power\tTotal Shares\tTotal P&L\tTimestamp")
        # print(
        #     "%12.2f\t%12d\t%9.2f\t%26s"
        #     % (
        #         trader.get_portfolio_summary().get_total_bp(),
        #         trader.get_portfolio_summary().get_total_shares(),
        #         trader.get_portfolio_summary().get_total_realized_pl(),
        #         trader.get_portfolio_summary().get_timestamp(),
        #     )
        # )

        if fast_ma and slow_ma:
            # Assuming previous_ma_fast and previous_ma_slow are tracked similarly
            signal = check_crossover(fast_ma, slow_ma, previous_fast_ma, previous_slow_ma)

            item = trader.get_portfolio_item(ticker)
            shares = item.get_shares()

            if signal == "sell":
                # If we have a long position, close it first.
                if shares > 0:
                    print(f"Closing long position for {ticker} by selling {shares} shares.")
                    close_order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, shares)
                    trader.submit_order(close_order)

                # Then enter a short position
                print(f"Entering short position for {ticker} with a limit sell order.")
                enter_short_order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, 1)  # Adjust quantity as needed
                #enter_short_order.set_price(best_ask)
                trader.submit_order(enter_short_order)

            elif signal == "buy":
                # If we have a short position, close it first.
                if shares < 0:
                    print(f"Closing short position for {ticker} by buying to cover {-shares} shares.")
                    close_order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, shares)
                    trader.submit_order(close_order)

                # Then enter a long position
                print(f"Entering long position for {ticker} with a limit buy order.")
                enter_long_order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, 1)  # Adjust quantity as needed
                #enter_long_order.set_price(best_bid)
                trader.submit_order(enter_long_order)

            # Update previous moving averages for the next iteration
            previous_fast_ma = fast_ma
            previous_slow_ma = slow_ma

        sleep(1)  # Adjust based on your data fetching frequency

