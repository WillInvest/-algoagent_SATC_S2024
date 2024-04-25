import shift
from threading import Thread, Lock
import pandas as pd
import time
import logging
from time import sleep
import random
from data_feed import *
from datetime import datetime
from collections import deque
import gymnasium as gym
import numpy as np

TRAIN = False

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SHIFT_env:
    def __init__(self, trader, ticker, step_time=1, lob_levels=5):
        self.trader = trader
        self.ticker = ticker
        self.order_size = 1
        self.step_time = step_time
        self.time_interval = 0.1
        self.n_time_step = 2 * int(self.step_time / self.time_interval)
        self.lob_levels = lob_levels
        self.gamma = 0.9
        self.alpha = 0.01
        self.endTime = datetime.strptime('15:55', '%H:%M').time()

        self.data_thread_alive = True
        self.data_thread = Thread(target=self._data_thread)
        self.midprice_list = deque(maxlen=self.n_time_step)
        self.data_thread.start()

        self.weighted_price = WeightedPrice()
        high_frequency_data_thread = Thread(target=collect_high_frequency_data,
                                            args=(self.trader,
                                                  self.ticker,
                                                  self.weighted_price,
                                                  self.data_thread_alive))
        high_frequency_data_thread.start()

        self.observation_space = gym.spaces.Box(
            np.array(
                ([-np.inf] * self.n_time_step)  # mid price
                + [0, -np.inf]  # spread
                + ([0] * self.lob_levels * 2)  # bid volume & ask volume
                + [0] * 2  # inventory & bp
            ),
            np.array(
                ([np.inf] * self.n_time_step)
                + [np.inf, np.inf]
                + ([np.inf] * self.lob_levels * 2)
                + [np.inf] * 2
            ),
        )

        sleep(5)

        # reward trackers
        initial_bp = self.trader.get_portfolio_summary().get_total_bp()
        self.initial_lp = self.trader.get_last_price(self.ticker)
        initial_inv = self.get_state()[self.n_time_step + 1]
        self.initial_equity = initial_bp + abs(initial_inv) * self.initial_lp * 100

        while self.trader.get_last_price(self.ticker) == 0:
            sleep(1)
            print(f"{self.ticker} is still waiting to start")

    def _data_thread(self):
        # thread constantly collecting midprice data
        print(f"Data thread starts")

        while self.trader.is_connected() and self.data_thread_alive:
            best_price = self.trader.get_best_price(self.ticker)
            best_bid = best_price.get_bid_price()
            best_ask = best_price.get_ask_price()
            if (best_bid == 0) and (best_ask == 0):
                if len(self.midprice_list) > 0:
                    self.midprice_list.append(self.midprice_list[-1])
            elif (best_bid == 0) or (best_ask == 0):
                self.midprice_list.append(max(best_bid, best_ask))
            else:
                self.midprice_list.append((best_bid + best_ask) / 2)

            sleep(self.time_interval)

        print("Data Thread stopped.")

    def get_state(self):
        # return a list with these values: mp, sp, inv, bp, bid_volumes, ask_volumes, order_size, target order proportions, gamma, alpha$

        while True:
            if len(self.midprice_list) == self.n_time_step:
                best_price = self.trader.get_best_price(self.ticker)
                best_bid = best_price.get_bid_price()
                best_ask = best_price.get_ask_price()

                weighted_ask, weighted_bid, weighted_spread = self.weighted_price.get_prices()

                if (best_bid == 0) or (best_ask == 0):
                    weighted_spread = 0
                else:
                    spread = best_ask - best_bid
                    weighted_spread = weighted_spread / spread if spread > 0 else spread

                inv = self.trader.get_portfolio_item(self.ticker).get_shares() // 100

                bid_book = self.trader.get_order_book(
                    self.ticker, shift.OrderBookType.LOCAL_BID, self.lob_levels
                )
                ask_book = self.trader.get_order_book(
                    self.ticker, shift.OrderBookType.LOCAL_ASK, self.lob_levels
                )

                bid_levels = []
                ask_levels = []
                for level in bid_book:
                    bid_levels.append(level.size)
                if len(bid_book) < self.lob_levels:
                    bid_levels += [0] * (self.lob_levels - len(bid_book))
                for level in ask_book:
                    ask_levels.append(level.size)
                if len(ask_book) < self.lob_levels:
                    ask_levels += [0] * (self.lob_levels - len(ask_book))

                # Convert deque to a numpy array
                midprice_array = np.array(self.midprice_list)

                # Normalize the array by the first element
                # Ensure the first element is not zero to avoid division by zero
                if midprice_array[0] != 0:
                    normalized_midprices = midprice_array / midprice_array[0]
                else:
                    # Handle the case where the first element is zero
                    # You might want to adjust this behavior based on your specific needs
                    normalized_midprices = np.zeros_like(midprice_array)

                # Calculate the sum of bid and ask levels to avoid division by zero
                sum_bid_levels = sum(bid_levels)
                sum_ask_levels = sum(ask_levels)

                # Normalize bid_levels and ask_levels by their respective sums
                normalized_bid_levels = [x / sum_bid_levels if sum_bid_levels > 0 else 0 for x in bid_levels]
                normalized_ask_levels = [x / sum_ask_levels if sum_ask_levels > 0 else 0 for x in ask_levels]

                state = np.concatenate([
                    normalized_midprices,
                    [weighted_spread, inv],
                    normalized_bid_levels,
                    normalized_ask_levels
                ])
                return np.array(state)
            else:
                # print("waiting for to collect more data")
                sleep(self.time_interval)

    def step(self, action):
        """ Execute the given action in the environment. """

        # close out position when market is going to end
        last_trade_time = self.trader.get_last_trade_time().time()
        if not TRAIN:
            if last_trade_time > self.endTime:
                action = 0
                self.close_positions()
                self.trader.cancel_all_pending_orders()
                print("Market is about to end, start closing out.")

        curr_position = self.get_state()[self.n_time_step + 1]

        if action == 1:
            # Buy
            if curr_position <= 0:  # Buy if current position is zero or negative
                order_size = 1 if curr_position == 0 else int(action-curr_position)  # Buy more if currently short
                if self.trader.get_portfolio_summary().get_total_bp() > (
                        order_size * 100 * self.trader.get_last_price(self.ticker)):
                    order = shift.Order(shift.Order.Type.MARKET_BUY, self.ticker, order_size)
                    self.trader.submit_order(order)
                else:
                    print(
                        f"{self.ticker} insufficient buying power: {self.trader.get_portfolio_summary().get_total_bp()}")
            # If current_position is positive, do nothing as we are already long

        elif action == 0:
            # Adjust position to neutral
            if curr_position > 0:
                order = shift.Order(shift.Order.Type.MARKET_SELL, self.ticker, int(abs(curr_position)))
                self.trader.submit_order(order)
            elif curr_position < 0:
                order = shift.Order(shift.Order.Type.MARKET_BUY, self.ticker, int(abs(curr_position)))
                self.trader.submit_order(order)
            # If current_position is 0, do nothing

        elif action == -1:
            # Sell
            if curr_position >= 0:  # Sell if current position is zero or positive
                order_size = 1 if curr_position == 0 else int(curr_position-action)  # Sell more if currently long
                if self.trader.get_portfolio_summary().get_total_bp() > (
                        order_size * 100 * self.trader.get_last_price(self.ticker)):
                    order = shift.Order(shift.Order.Type.MARKET_SELL, self.ticker, order_size)
                    self.trader.submit_order(order)
                else:
                    print(
                        f"{self.ticker} insufficient buying power: {self.trader.get_portfolio_summary().get_total_bp()}")
            # If current_position is negative, do nothing as we are already short

        sleep(self.step_time)  # Simulate passage of time

        state = self.get_state()
        offset = self.n_time_step - 1
        total_bp = self.trader.get_portfolio_summary().get_total_bp()
        curr_lp = self.trader.get_last_price(self.ticker)
        curr_inv = state[2 + offset]
        curr_equity = total_bp + abs(curr_inv) * curr_lp * 100
        pnl = curr_equity - self.initial_equity
        self.initial_equity = curr_equity
        self.initial_lp = curr_lp
        reward = pnl - abs(action) * 0.3
        if abs(reward) > 1000:
            reward = 0

        return state, reward, total_bp, curr_inv, curr_equity, curr_lp

    def close_positions(self):
        # close all positions for given ticker
        print("running close positions function for", self.ticker)

        # close any long positions
        item = self.trader.get_portfolio_item(self.ticker)
        long_shares = item.get_long_shares()
        if long_shares > 0:
            print(f"{self.ticker} market selling because long shares = {long_shares}")
            rejections = 0
            while item.get_long_shares() > 0:
                for _ in range(int(long_shares / 100)):
                    order = shift.Order(
                        shift.Order.Type.MARKET_SELL, self.ticker, 1
                    )
                    self.trader.submit_order(order)
                    sleep(0.1)
                sleep(2)
                if (
                        self.trader.get_order(order.id).status
                        == shift.Order.Status.REJECTED
                ):
                    rejections += 1
                else:
                    break
                if rejections == 5:
                    # if orders get rejected 5 times, just give up
                    break

        # close any short positions
        item = self.trader.get_portfolio_item(self.ticker)
        short_shares = item.get_short_shares()
        if short_shares > 0:
            print(f"{self.ticker} market buying because short shares = {short_shares}")
            rejections = 0
            while item.get_short_shares() > 0:
                for _ in range(int(short_shares / 100)):
                    order = shift.Order(
                        shift.Order.Type.MARKET_BUY, self.ticker, 1
                    )
                    self.trader.submit_order(order)
                    sleep(0.1)
                sleep(2)
                if (
                        self.trader.get_order(order.id).status
                        == shift.Order.Status.REJECTED
                ):
                    rejections += 1
                else:
                    break
                if rejections == 5:
                    # if orders get rejected 5 times, just give up
                    print(f"{self.ticker} could not complete close positions")
                    break

    def reset(self):
        self.trader.cancel_all_pending_orders()
        self.close_positions()
        # reward trackers
        initial_bp = self.trader.get_portfolio_summary().get_total_bp()
        self.initial_lp = self.trader.get_last_price(self.ticker)
        initial_inv = self.get_state()[self.n_time_step + 1]
        self.initial_equity = initial_bp + abs(initial_inv) * self.initial_lp * 100

        return self.get_state()

    def kill_thread(self):
        self.data_thread_alive = False

    def __del__(self):
        self.kill_thread()


if __name__ == '__main__':
    with shift.Trader("algoagent_test001") as trader:
        trader.connect("initiator.cfg", "x6QYVYRT")
        sleep(1)
        trader.sub_all_order_book()
        sleep(1)

        env = SHIFT_env(trader, "CS1")  # Initialize the trading environment
        state = env.reset()
        print(state)
        # Run the agent for a number of steps
        for step in range(10):  # Change this number based on how many steps you want to simulate
            action = random.choice([-1, 0, 1])
            next_state, reward = env.step(action)[0:1]  # Execute the action in the environment
            # print(f"Step {step}: Action {action}, Reward {reward}")
            state = next_state
            print(state)
            print(len(state))
            sleep(1)  # Wait a second before the next action
        state = env.reset()

        trader.disconnect()
