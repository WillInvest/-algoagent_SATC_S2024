import shift
from threading import Thread, Lock, Event
import pandas as pd
import numpy as np
import time
from time import sleep
from tensorflow.keras.utils import Progbar
from shift_agent import *


class CirList:
    def __init__(self, length):
        self.size = length
        self._table = [None] * length
        self.idx = 0
        self._counter = 0
        self.lock = Lock()

    def insertData(self, data):
        with self.lock:
            self._counter += 1
            self._table[self.idx] = data
            self.idx = (self.idx + 1) % self.size

    def getData(self):
        with self.lock:
            tail = self._table[0:self.idx]
            head = self._table[self.idx:]
            return head + tail

    def isFull(self):
        return self._counter >= self.size

    def __repr__(self):
        return str(self.getData())


class OrderBook:
    def __init__(self, AskOrder, BidOrder, last_price):
        self.last_price = last_price
        idx = 0
        tmp = pd.DataFrame(columns=['price', 'size', 'type', 'last_price'])
        ls = AskOrder

        for order in ls[::-1]:
            tmp.loc[idx] = [order.price, order.size, 'Ask', last_price]
            idx += 1

        self.n_ask = idx
        ls = BidOrder

        for order in ls:
            tmp.loc[idx] = [order.price, order.size, 'Bid', last_price]
            idx += 1

        self.n_bid = idx - self.n_ask

        self.df = tmp

    def __repr__(self):
        return str(self.df)


class SHIFT_env:
    def __init__(self,
                 trader,
                 t,
                 nTimeStep,
                 ODBK_range,
                 symbol,
                 commission,
                 time_steps,
                 target_shares):

        self.timeInterval = t
        self.symbol = symbol
        self.nTimeStep = nTimeStep
        self.ODBK_range = ODBK_range
        self.trader = trader
        self.commission = commission
        self.mutex = Lock()

        self.dataThread = Thread(target=self._link)
        self.table = CirList(nTimeStep)
        self.priceTable = CirList(2)

        # print('Waiting for connection', end='')
        # for _ in range(5):
        #     time.sleep(0.1)
        #     print('.', end='')
        # print()

        self.thread_alive = True
        self.dataThread.start()

        self.remained_share = 0
        self.isBuy = None
        self.remained_time = 5
        self.initial_pl = self.trader.get_portfolio_item(self.symbol).get_realized_pl()
        self.time_steps = time_steps
        self.target_shares = target_shares

    def set_objective(self, share, remained_time):
        self.remained_time = remained_time
        self.isBuy = share > 0
        self.remained_share = abs(share)

    @staticmethod
    def action_space():
        return 1

    @staticmethod
    def obs_space():
        return 7

    def get_signal(self):
        mid_price = []
        tab = self.priceTable
        if tab.isFull():
            for ele in tab.getData():
                mid_price.append(ele)
            if mid_price[-1] > mid_price[0]:
                self.set_objective(self.target_shares, self.time_steps)
            if mid_price[-1] < mid_price[0]:
                self.set_objective(-self.target_shares, self.time_steps)

    def _link(self):
        while self.trader.is_connected() and self.thread_alive:
            try:
                bp = self.trader.get_best_price(self.symbol)
                best_bid = bp.get_global_bid_price()
                best_ask = bp.get_global_ask_price()
                last_price = self.trader.get_last_price(self.symbol)
                self.priceTable.insertData((best_ask + best_bid) / 2)
                # Update order book data
                Ask_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.GLOBAL_ASK, self.ODBK_range)
                Bid_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.GLOBAL_BID, self.ODBK_range)
                orders = OrderBook(Ask_ls, Bid_ls, last_price)
                self.table.insertData(orders)
                time.sleep(self.timeInterval)

            except Exception as e:
                print(f"Error getting best price for {self.symbol}: {e}")

    def step(self, action):
        premium = action
        signBuy = 1 if self.isBuy else -1
        if self.remained_share == 0:
            base_price = self.trader.get_best_price(self.symbol).get_bid_price()
        else:
            base_price = self._getClosePrice(self.remained_share)
        obj_price = base_price - signBuy * premium

        if self.remained_time > 0:
            orderType = shift.Order.LIMIT_BUY if self.isBuy else shift.Order.LIMIT_SELL
        else:
            orderType = shift.Order.MARKET_BUY if self.isBuy else shift.Order.MARKET_SELL

        order = shift.Order(orderType, self.symbol, self.remained_share, obj_price)
        # Corrected method name for submitting order
        self.trader.submit_order(order)
        time.sleep(self.timeInterval)

        if self.trader.get_waiting_list_size() > 0:
            if order.type == shift.Order.LIMIT_BUY:
                order.type = shift.Order.CANCEL_BID
            else:
                order.type = shift.Order.CANCEL_ASK
            self.trader.submit_order(order)
            exec_share = self.remained_share - order.size
            self.remained_share = order.size
        else:
            exec_share = self.remained_share
            self.remained_share = 0

        # Define reward
        done = False

        status = self.trader.get_submitted_orders()[-1].type

        if status == shift.Order.Type.MARKET_BUY or status == shift.Order.Type.MARKET_SELL:
            reward = -2.0
        else:
            if exec_share > 0:
                reward = exec_share * premium + 1
            else:
                reward = -1

        if self.remained_share == 0:
            done = True

        self.remained_time -= 1
        next_obs = self._get_obs()
        time.sleep(self.timeInterval)

        return next_obs, reward, done, dict()

    def _get_obs(self):
        return np.concatenate((self.compute(), np.array([self.remained_share, self.remained_time])))

    def _getClosePrice(self, share):
        return self.trader.get_close_price(self.symbol, self.isBuy, abs(share))

    def _getSubOrder(self):
        # Get the last 5 orders, or all orders if fewer than 5
        latest_orders = self.trader.get_submitted_orders()[-5:]
        for order in latest_orders:
            if order.status == shift.Order.Status.FILLED:
                price = order.executed_price
            else:
                price = order.price
            print(
                "%6s\t%16s\t%7.2f\t\t%4d\t\t%4d\t%23s"
                % (
                    order.symbol,
                    order.type,
                    price,
                    order.size,
                    order.executed_size,
                    order.status
                )
            )

    def getSummary(self):
        print("Buying Power\tTotal Shares\tTotal P&L\tTimestamp")
        print(
            "%12.2f\t%12d\t%9.2f\t%26s"
            % (
                self.trader.get_portfolio_summary().get_total_bp(),
                self.trader.get_portfolio_summary().get_total_shares(),
                self.trader.get_portfolio_summary().get_total_realized_pl(),
                self.trader.get_portfolio_summary().get_timestamp(),
            )
        )

    def _getWaitingList(self):
        for order in self.trader.get_waiting_list():
            print(
                "%6s\t%16s\t%7.2f\t\t%4d\t\t%4d\t"
                % (
                    order.symbol,
                    order.type,
                    order.price,
                    order.size,
                    order.executed_size
                )
            )

    def _getPortfolio(self):
        print("Buying Power\tTotal Shares\tTotal P&L\tTimestamp")
        print(
            "%12.2f\t%12d\t%9.2f\t%26s"
            % (
                self.trader.get_portfolio_summary().get_total_bp(),
                self.trader.get_portfolio_summary().get_total_shares(),
                self.trader.get_portfolio_summary().get_total_realized_pl(),
                self.trader.get_portfolio_summary().get_timestamp(),
            )
        )

    def _getItems(self):
        item = self.trader.get_portfolio_item(self.symbol)
        print("Symbol\t\tShares\t\tPrice\t\tP&L\t\tTimestamp")
        print(
            "%6s\t\t%6d\t%9.2f\t%7.2f\t\t%26s"
            % (
                item.get_symbol(),
                item.get_long_shares(),
                item.get_price(),
                item.get_realized_pl(),
                item.get_timestamp(),
            )
        )

    def cancel_orders(self):
        # cancel all the remaining orders
        for order in self.trader.get_waiting_list():
            if (order.symbol == self.symbol):
                self.trader.submit_cancellation(order)
                sleep(3)  # the order cancellation needs a little time to go through

    def close_positions(self):
        # NOTE: The following orders may not go through if:
        # 1. You do not have enough buying power to close your short postions. Your strategy should be formulated to ensure this does not happen.
        # 2. There is not enough liquidity in the market to close your entire position at once. You can avoid this either by formulating your
        #    strategy to maintain a small position, or by modifying this function to close ur positions in batches of smaller orders.

        # close all positions for given ticker
        print(f"running close positions function for {self.symbol}")

        item = self.trader.get_portfolio_item(self.symbol)

        # close any long positions
        long_shares = item.get_long_shares()
        if long_shares > 0:
            print(f"market selling because {self.symbol} long shares = {long_shares}")
            order = shift.Order(shift.Order.Type.MARKET_SELL,
                                self.symbol,
                                int(long_shares / 100))  # we divide by 100 because orders are placed for lots of 100 shares
            self.trader.submit_order(order)
            sleep(self.timeInterval)  # we sleep to give time for the order to process

        # close any short positions
        short_shares = item.get_short_shares()
        if short_shares > 0:
            print(f"market buying because {self.symbol} short shares = {short_shares}")
            order = shift.Order(shift.Order.Type.MARKET_BUY,
                                self.symbol, int(short_shares / 100))
            self.trader.submit_order(order)
            sleep(self.timeInterval)

    def reset(self):
        return self._get_obs()

    def kill_thread(self):
        self.thread_alive = False

    def compute(self):
        tab = self.table
        feature = []
        if tab.isFull():
            for ele in tab.getData():
                try:
                    n_ask = ele.n_ask
                    n_bid = ele.n_bid
                    if n_ask <= 0 or n_bid <= 0:
                        raise ValueError("The number of Ask/Bid orders is 0 or negative.")

                    bas = self._ba_spread(ele.df, n_ask)
                    p = self._price(ele.df)
                    sp = self._smart_price(ele.df, n_ask)
                    li = self._liquid_imbal(ele.df, n_ask, n_bid)
                    act_direction = 'Buy' if self.isBuy else 'Sell'
                    if act_direction:
                        mc, _ = self._market_cost(ele.df, n_ask, n_bid, act_direction, self.remained_share,
                                                  self.commission)
                        feature += [bas, p, sp, li, mc]
                    else:
                        feature += [bas, p, sp, li, np.nan]
                except Exception as e:
                    print(f"Error computing features for {self.symbol}: {e}")
                    feature += [0, 0, 0, 0, 0]
        return np.array(feature)

    @staticmethod
    def _mid_price(df, n_ask):
        mid_price = (df.price[n_ask - 1] + df.price[n_ask]) / 2
        return mid_price

    @staticmethod
    def _ba_spread(df, n_ask):
        spread = df.price[n_ask - 1] - df.price[n_ask]
        return spread

    @staticmethod
    def _price(df):
        return df.last_price[0]

    @staticmethod
    def _smart_price(df, n_ask):
        price = (df['size'][n_ask] * df.price[n_ask - 1] + df['size'][n_ask - 1] * df.price[n_ask]) \
                / (df['size'][n_ask] + df['size'][n_ask - 1])
        return price

    @staticmethod
    def _liquid_imbal(df, n_ask, n_bid):
        if n_ask > n_bid:
            imbal = df['size'][n_ask:].sum() - df['size'][(n_ask - n_bid):n_ask].sum()
        else:
            imbal = df['size'][n_ask:(2 * n_ask)].sum() - df['size'][0:n_ask].sum()
        return imbal

    @staticmethod
    def _market_cost(df, n_ask, n_bid, act_direction, shares, commission):
        if act_direction == 'Buy':
            counter = df['size'][n_ask - 1]
            n_cross = 1
            while counter < shares and n_ask - 1 >= n_cross:
                counter += df['size'][n_ask - 1 - n_cross]
                n_cross += 1
            if n_cross > 1:
                sub_size = np.array(df['size'][(n_ask - n_cross):n_ask])
                sub_price = np.array(df.price[(n_ask - n_cross):n_ask])
                sub_size[0] = shares - sum(sub_size) + sub_size[0]
                market_price = sub_size.dot(sub_price) / shares
                cost = shares * (market_price - df.price[n_ask] + df.price[n_ask - 1] * commission)
            else:
                market_price = df.price[n_ask - 1]
                cost = shares * (market_price * (1 + commission) - df.price[n_ask])
        else:
            counter = df['size'][n_ask]
            n_cross = 1
            while counter < shares and n_cross <= n_bid - 1:
                counter += df['size'][n_ask + n_cross]
                n_cross += 1
            if n_cross > 1:
                sub_size = np.array(df['size'][n_ask:(n_ask + n_cross)])
                sub_price = np.array(df.price[n_ask:(n_ask + n_cross)])
                sub_size[-1] = shares - sum(sub_size) + sub_size[-1]
                market_price = sub_size.dot(sub_price) / shares
                cost = shares * (market_price - df.price[n_ask - 1] + df.price[n_ask] * commission)
            else:
                market_price = df.price[n_ask]
                cost = shares * (market_price * (1 + commission) - df.price[n_ask - 1])
        return cost, market_price

    def __del__(self):
        self.kill_thread()

