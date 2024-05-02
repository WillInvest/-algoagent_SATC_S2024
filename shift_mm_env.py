from threading import Thread
import logging
from data_feed import *
from datetime import datetime
from collections import deque
import numpy as np
from shift_agent import PPOActorCritic
import tensorflow as tf

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SHIFT_env:
    def __init__(self,
                 traders,
                 ticker,
                 step_time=1,
                 lob_levels=5,
                 weight=0.5,
                 gamma=0.5,
                 ):
        self.trader = traders
        self.ticker = ticker
        self.order_size = 1
        self.step_time = step_time
        self.time_interval = 0.1
        self.nTimeStep = int(self.step_time / self.time_interval)
        self.lob_levels = lob_levels
        self.endTime = datetime.strptime('15:55', '%H:%M').time()
        self.w = weight  # weight on pnl
        self.gamma = gamma  # inventory risk aversion
        self.order_list = []
        self.order_size_sum = 0
        self.current_market_share = 0
        # Data collection
        self.mp_list = deque(maxlen=self.nTimeStep)
        self.data_thread_alive = True
        self.data_thread = Thread(target=self._data_thread)

        self.data_thread.start()
        print("Data Thread start")
        self.weighted_price = WeightedPrice()
        high_frequency_data_thread = Thread(target=collect_high_frequency_data,
                                            args=(self.trader,
                                                  self.ticker,
                                                  self.weighted_price,
                                                  self.data_thread_alive))
        high_frequency_data_thread.start()
        # reward trackers
        self.initial_inv = 0
        self.initial_pnl = 0
        self.init_bp = self.trader.get_portfolio_summary().get_total_bp()
        self.init = False
        self.initial_position = 0
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
                if len(self.mp_list) > 0:
                    self.mp_list.append(self.mp_list[-1])
            elif (best_bid == 0) or (best_ask == 0):
                self.mp_list.append(max(best_bid, best_ask))
            else:
                self.mp_list.append((best_bid + best_ask) / 2)

            sleep(self.step_time)

        print("Data Thread stopped.")

    def get_state(self):
        while True:
            if len(self.mp_list) == self.nTimeStep:
                best_price = self.trader.get_best_price(self.ticker)
                best_bid = best_price.get_bid_price()
                best_ask = best_price.get_ask_price()

                weighted_ask, weighted_bid, weighted_spread = self.weighted_price.get_prices()
                bid_book = self.trader.get_order_book(
                    self.ticker, shift.OrderBookType.LOCAL_BID, self.lob_levels
                )
                ask_book = self.trader.get_order_book(
                    self.ticker, shift.OrderBookType.LOCAL_ASK, self.lob_levels
                )

                bid_levels = []
                ask_levels = []
                bid_price = []
                ask_price = []
                for level in bid_book:
                    bid_levels.append(level.size)
                    bid_price.append(level.price)

                if len(bid_book) < self.lob_levels:
                    bid_levels += [0] * (self.lob_levels - len(bid_book))
                    bid_price += [0] * (self.lob_levels - len(bid_book))

                for level in ask_book:
                    ask_levels.append(level.size)
                    ask_price.append(level.price)

                if len(ask_book) < self.lob_levels:
                    ask_levels += [0] * (self.lob_levels - len(ask_book))
                    ask_price += [0] * (self.lob_levels - len(ask_book))

                # Convert deque to a numpy array
                mp_array = np.array(self.mp_list)
                bid_array = np.array(bid_price)
                ask_array = np.array(ask_price)

                # Normalize the array by the first element
                # Ensure the first element is not zero to avoid division by zero
                if mp_array[0] != 0:
                    normalized_mp = mp_array / mp_array[0]
                    normalized_bid_price = bid_array / mp_array[0]
                    normalized_ask_price = ask_array / mp_array[0]


                else:
                    # Handle the case where the first element is zero
                    # You might want to adjust this behavior based on your specific needs
                    normalized_mp = np.zeros_like(mp_array)
                    normalized_bid_price = np.zeros_like(bid_array)
                    normalized_ask_price = np.zeros_like(ask_array)

                # Calculate the sum of bid and ask levels to avoid division by zero
                sum_bid_levels = sum(bid_levels)
                sum_ask_levels = sum(ask_levels)
                self.order_size_sum = sum_bid_levels + sum_ask_levels
                curr_shares = self.trader.get_portfolio_item(self.ticker).get_shares() // 100
                inv = curr_shares / sum_bid_levels if curr_shares > 0 else curr_shares / sum_ask_levels
                bp = self.init_bp = self.trader.get_portfolio_summary().get_total_bp() / self.init_bp

                # Normalize bid_levels and ask_levels by their respective sums
                normalized_bid_levels = [x / sum_bid_levels if sum_bid_levels > 0 else 0 for x in bid_levels]
                normalized_ask_levels = [x / sum_ask_levels if sum_ask_levels > 0 else 0 for x in ask_levels]

                state = np.concatenate([
                    normalized_mp,
                    [weighted_ask / best_ask,
                     weighted_bid / best_bid,
                     weighted_spread],
                    normalized_bid_price,
                    normalized_bid_levels,
                    normalized_ask_price,
                    normalized_ask_levels
                ])
                return np.array(state)
            else:
                print("waiting for to collect more data")
                sleep(self.time_interval)

    def execute_order(self, actions=None):
        bp = self.trader.get_best_price(self.ticker)
        best_bid = bp.get_bid_price()
        best_ask = bp.get_ask_price()
        mid = (best_ask + best_bid) / 2
        spread = best_ask - best_bid if best_ask > best_bid else 0.01
        if actions is None:
            return
        else:
            actions = np.array(actions).flatten()
            order_size = 1
            # print(f"curr_bp: {self.trader.get_portfolio_summary().get_total_bp()} | "
            #       f"premium: {actions * spread}")
            p_ask = best_ask + max(spread * actions, 0.003)
            p_bid = best_bid - max(spread * actions, 0.003)
            limit_buy = shift.Order(shift.Order.Type.LIMIT_BUY, self.ticker, order_size, p_bid)
            limit_sell = shift.Order(shift.Order.Type.LIMIT_SELL, self.ticker, order_size, p_ask)
            # print("size",order_size)
            self.order_list.append(limit_buy.id)
            self.trader.submit_order(limit_buy)
            self.order_list.append(limit_sell.id)
            self.trader.submit_order(limit_sell)

            return actions * spread
            # self.get_waiting_list()

    def step(self, action):
        """ Execute the given action in the environment. """
        # close out position when market is going to end
        last_trade_time = self.trader.get_last_trade_time().time()
        if last_trade_time > self.endTime:
            self.close_positions()
            self.cancel_all()
            print("Market is about to end, start closing out.")
            return

        # Execute actions
        old_item = self.trader.get_portfolio_item(self.ticker)
        old_position = old_item.get_shares()
        premium = self.execute_order(action)
        sleep(self.step_time)
        item = self.trader.get_portfolio_item(self.ticker)
        curr_position = item.get_shares()
        execute_order = curr_position - old_position
        if curr_position != 0:
            if curr_position > 0:
                order = shift.Order(shift.Order.Type.MARKET_SELL,
                                    self.ticker,
                                    abs(int(curr_position / 100)))
                self.trader.submit_order(order)
                sleep(0.5)
            else:
                order = shift.Order(shift.Order.Type.MARKET_BUY,
                                    self.ticker,
                                    abs(int(curr_position / 100)))
                self.trader.submit_order(order)
                sleep(0.5)
        if self.trader.get_waiting_list_size() > 0:
            self.trader.cancel_all_pending_orders()
            sleep(0.3)
        # -------------------- Reward -------------------- #
        total_bp = self.trader.get_portfolio_summary().get_total_bp()

        # lp = self.trader.get_last_price(self.ticker)
        item = self.trader.get_portfolio_item(self.ticker)
        curr_pnl = item.get_realized_pl()
        reward = curr_pnl - self.initial_pnl + abs(execute_order) * 0.002
        # recent_pl_change = curr_pnl - self.initial_pnl
        # curr_inv = self.trader.get_unrealized_pl(self.ticker)
        # curr_inv_pnl = curr_inv - self.initial_inv
        # curr_position = item.get_shares()
        # execute_order = curr_position - self.initial_position
        # reward = recent_pl_change + (self.gamma * curr_inv_pnl) + abs(execute_order/100) * 0.2
        print(f"Reward: {reward} | "
              f"total_bp: {total_bp} | "
              f"premium: {premium}")
        # self.initial_inv = curr_inv
        self.initial_pnl = curr_pnl
        # self.initial_position = curr_position
        state = self.get_state()
        return state, reward

    def close_positions(self):
        # close all positions for given ticker
        print('running close positions function for', self.ticker)

        item = self.trader.get_portfolio_item(self.ticker)

        # close any long positions
        long_shares = item.get_long_shares()
        if long_shares > 0:
            print("market selling because long shares =", long_shares)
            order = shift.Order(shift.Order.Type.MARKET_SELL, self.ticker, int(long_shares / 100))
            self.trader.submit_order(order)
            time.sleep(0.2)
            # make sure order submitted correctly
            counter = 0
            item = self.trader.get_portfolio_item(self.ticker)
            order = self.trader.get_order(order.id)
            while (item.get_long_shares() > 0):
                print(order.status)
                time.sleep(0.2)
                order = self.trader.get_order(order.id)
                item = self.trader.get_portfolio_item(self.ticker)
                counter += 1
                # if order is not executed after 5 seconds, then break
                if counter > 5:
                    break

        # close any short positions
        short_shares = item.get_short_shares()
        if short_shares > 0:
            print("market buying because short shares =", short_shares)
            order = shift.Order(shift.Order.Type.MARKET_BUY, self.ticker, int(short_shares / 100))
            self.trader.submit_order(order)
            time.sleep(0.2)
            # make sure order submitted correctly
            counter = 0
            item = self.trader.get_portfolio_item(self.ticker)
            order = self.trader.get_order(order.id)
            while (item.get_short_shares() > 0):
                print(order.status)
                time.sleep(0.2)
                order = self.trader.get_order(order.id)
                item = self.trader.get_portfolio_item(self.ticker)
                counter += 1
                # if order is not executed after 5 seconds, then break
                if counter > 5:
                    break

    def reset(self):
        self.execute_order()
        self.cancel_all()
        self.close_positions()
        item = self.trader.get_portfolio_item(self.ticker)
        self.initial_pnl = item.get_realized_pl()
        self.initial_position = item.get_shares()
        return self.get_state()

    def cancel_all(self):
        self.execute_order()
        self.trader.cancel_all_pending_orders()
        for _ in range(5):
            for i, order_id in enumerate(self.order_list):
                order = self.trader.get_order(order_id)
                # print(type(order.status)
                self.trader.submit_cancellation(order)
            time.sleep(0.1)
        print("all order cancelled")

    def get_waiting_list(self):
        print(
            "Symbol\t\t\t\tType\t  Price\t\tSize\tExecuted\tID\t\t\t\t\t\t\t\t\t\t\t\t\t\t Status\t\tTimestamp"
        )
        for order in self.trader.get_waiting_list():
            print(
                "%6s\t%16s\t%7.2f\t\t%4d\t\t%4d\t%36s\t%23s\t\t%26s"
                % (
                    order.symbol,
                    order.type,
                    order.price,
                    order.size,
                    order.executed_size,
                    order.id,
                    order.status,
                    order.timestamp,
                )
            )

    def get_submitted_order(self):
        print(
            "Symbol\t\t\t\tType\t  Price\t\tSize\tExecuted\tID\t\t\t\t\t\t\t\t\t\t\t\t\t\t Status\t\tTimestamp"
        )
        for order in self.trader.get_submitted_orders():
            if order.status == shift.Order.Status.FILLED:
                price = order.executed_price
            else:
                price = order.price
            print(
                "%6s\t%16s\t%7.2f\t\t%4d\t\t%4d\t%36s\t%23s\t\t%26s"
                % (
                    order.symbol,
                    order.type,
                    price,
                    order.size,
                    order.executed_size,
                    order.id,
                    order.status,
                    order.timestamp,
                )
            )

    def kill_thread(self):
        self.data_thread_alive = False

    def __del__(self):
        self.kill_thread()


if __name__ == '__main__':
    with shift.Trader("algoagent") as trader:
        trader.connect("initiator.cfg", "x6QYVYRT")
        sleep(1)
        trader.sub_all_order_book()
        sleep(1)
        tickers = 'CS1'

        env = SHIFT_env(trader, tickers)
        agent = PPOActorCritic(env)
        for _ in range(100):
            env.close_positions()
            sleep(0.1)

        trader.disconnect()
