import shift
from threading import Thread, Lock
import pandas as pd
import time
import logging
from time import sleep
import random
from data_feed import *
from datetime import datetime

TRAIN = False

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SHIFT_env:
    def __init__(self, trader, ticker, timeInterval=10, lob_levels=5):
        self.trader = trader
        self.ticker = ticker
        self.lob_levels = lob_levels
        self.thread_alive = True
        self.timeInterval = timeInterval
        self.data_lock = Lock()
        self.df = pd.DataFrame()
        self.thread = Thread(target=self.collect_data)
        self.thread.start()
        self.endTime = datetime.strptime('15:55', '%H:%M').time()


    def get_shares(self):
        """
        Retrieves the net position of shares for the ticker.
        Long shares are positive, and short shares are negative.
        """
        item = self.trader.get_portfolio_item(self.ticker)
        if item:
            net_shares = item.get_long_shares() - item.get_short_shares()
        else:
            net_shares = 0  # No positions if no item found
        return net_shares

    def collect_data(self):
        weighted_price = WeightedPrice()
        high_frequency_data_thread = Thread(target=collect_high_frequency_data,
                                            args=(self.trader,
                                                  self.ticker,
                                                  weighted_price,
                                                  self.thread_alive,
                                                  self.timeInterval))
        high_frequency_data_thread.start()

        try:
            while self.thread_alive:
                sleep(self.timeInterval)  # Delay for data accumulation

                current_time = pd.Timestamp.now()
                data = {'shares_held': self.get_shares() / 100, 'weighted_ask': 1, 'weighted_bid': 1,
                        'weighted_spread': 1}
                for i in range(self.lob_levels):
                    data.update({f'Bid {i + 1} Price': 1, f'Bid {i + 1} Size': 1,
                                 f'Ask {i + 1} Price': 1, f'Ask {i + 1} Size': 1})

                bids = self.trader.get_order_book(self.ticker, shift.OrderBookType.LOCAL_BID, self.lob_levels)
                asks = self.trader.get_order_book(self.ticker, shift.OrderBookType.LOCAL_ASK, self.lob_levels)
                if bids and asks:
                    best_bid = bids[0].price
                    best_ask = asks[0].price
                    total_bid_volume = sum(bid.size for bid in bids)
                    total_ask_volume = sum(ask.size for ask in asks)

                    for i, bid in enumerate(bids):
                        data[f'Bid {i + 1} Price'] = bid.price / best_bid
                        data[f'Bid {i + 1} Size'] = bid.size / total_bid_volume if total_bid_volume > 0 else 1

                    for i, ask in enumerate(asks):
                        data[f'Ask {i + 1} Price'] = ask.price / best_ask
                        data[f'Ask {i + 1} Size'] = ask.size / total_ask_volume if total_ask_volume > 0 else 1

                weighted_ask, weighted_bid, weighted_spread = weighted_price.get_prices()
                data['weighted_ask'] = weighted_ask / best_ask
                data['weighted_bid'] = weighted_bid / best_bid
                if best_ask > best_bid:
                    data['weighted_spread'] = weighted_spread / (best_ask - best_bid)
                else:
                    data['weighted_spread'] = 0

                with self.data_lock:
                    self.df = self.df.append(data, ignore_index=True)
                    if len(self.df) > 10:
                        self.df = self.df.tail(10)

        except Exception as e:
            logging.error(f"Error during data collection: {e}")
            self.thread_alive = False

        finally:
            high_frequency_data_thread.join()

    def step(self, action):
        """ Execute the given action in the environment. """

        # close out position when market is going to end
        last_trade_time = self.trader.get_last_trade_time().time()
        if not TRAIN:
            if last_trade_time > self.endTime:
                action = 0
                self.close_positions()
                self.cancel_orders()
                print("Market is about to end, start closing out.")


        item = self.trader.get_portfolio_item(self.ticker)
        initial_shares = int(self.get_shares() / 100)
        initial_pnl = self.trader.get_unrealized_pl(self.ticker) + item.get_realized_pl()

        if action == -1:  # Sell
            if initial_shares >= -1:
                # Cover short position and then sell
                total_shares = initial_shares - action
                # print(f"market sell {total_shares}")
                self.trader.submit_order(shift.Order(shift.Order.Type.MARKET_SELL, self.ticker, total_shares))
            else:
                total_shares = action - initial_shares
                # print(f"market buy {total_shares}")
                self.trader.submit_order(shift.Order(shift.Order.Type.MARKET_BUY, self.ticker, total_shares))

        elif action == 1:  # Buy
            if initial_shares <= 1:
                # Cover long position and then buy
                total_shares = action - initial_shares
                # print(f"market buy {total_shares}")
                self.trader.submit_order(shift.Order(shift.Order.Type.MARKET_BUY, self.ticker, total_shares))
            else:
                total_shares = initial_shares - action
                # print(f"market sell {total_shares}")
                self.trader.submit_order(shift.Order(shift.Order.Type.MARKET_SELL, self.ticker, total_shares))
        else:
            total_shares = action

        sleep(self.timeInterval)  # Simulate passage of time

        current_shares = int(self.get_shares() / 100)
        current_pnl = self.trader.get_unrealized_pl(self.ticker) + item.get_realized_pl()
        reward = current_pnl - initial_pnl - abs(current_shares - initial_shares) * 0.03
        next_state = self.get_observation()  # Assuming you have implemented this method to fetch the latest state

        return next_state, reward

    def get_observation(self):
        """Fetch the current observation state as a list consisting of the latest price and size data,
        current shares, and buying power."""
        with self.data_lock:
            if not self.df.empty:
                # Extracting the last row as the current observation
                latest_data = self.df.iloc[-1]

                # Flatten the DataFrame row into a list and append additional data
                state = latest_data.tolist()
                return state
            return []  # Return an empty list if no data is available

    def reset(self):
        """Reset the environment to start a new episode."""
        self.cancel_orders()
        self.close_positions()
        # self.df = pd.DataFrame()  # Clear existing data frame
        return self.get_observation()

    def cancel_orders(self):
        print("Cancelling orders")
        """Cancel all pending orders."""
        for order in self.trader.get_waiting_list():
            if order.symbol == self.ticker:
                self.trader.submit_cancellation(order)
                time.sleep(self.timeInterval)  # Wait to ensure cancellation goes through

    def close_positions(self):
        # close all positions for given ticker
        print(f"running close positions function for {self.ticker}")

        item = self.trader.get_portfolio_item(self.ticker)

        # close any long positions
        long_shares = item.get_long_shares()
        if long_shares > 0:
            print(f"market selling because {self.ticker} long shares = {long_shares}")
            order = shift.Order(shift.Order.Type.MARKET_SELL,
                                self.ticker,
                                int(long_shares / 100))  # we divide by 100 because orders are placed for lots of 100 shares
            self.trader.submit_order(order)
            sleep(3)  # we sleep to give time for the order to process

        # close any short positions
        short_shares = item.get_short_shares()
        if short_shares > 0:
            print(f"market buying because {self.ticker} short shares = {short_shares}")
            order = shift.Order(shift.Order.Type.MARKET_BUY,
                                self.ticker, int(short_shares / 100))
            self.trader.submit_order(order)
            sleep(3)

    def close(self):
        self.thread_alive = False
        self.thread.join()


if __name__ == '__main__':
    with shift.Trader("algoagent_test001") as trader:
        trader.connect("initiator.cfg", "x6QYVYRT")
        sleep(1)
        trader.sub_all_order_book()
        sleep(1)

        env = SHIFT_env(trader, "AAPL")  # Initialize the trading environment
        sleep(11)
        state = env.reset()
        print(state)
        # Run the agent for a number of steps
        for step in range(10):  # Change this number based on how many steps you want to simulate
            action = random.choice([-1, 0, 1])
            next_state, reward = env.step(action)  # Execute the action in the environment
            #print(f"Step {step}: Action {action}, Reward {reward}")
            state = next_state
            print(state)
            print(len(state))
            sleep(1)  # Wait a second before the next action
        state = env.reset()

        env.close()  # Ensure the environment is properly closed
        trader.disconnect()