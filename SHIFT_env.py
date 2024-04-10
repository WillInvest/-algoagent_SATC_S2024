import shift
from threading import Thread, Lock, Event
import pandas as pd
import numpy as np
import time
from time import sleep
from tensorflow.keras.utils import Progbar
from shift_agent import *
from datetime import datetime
from multiprocessing import Process, Manager, Barrier, Lock


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

        self.thread_alive = True
        self.dataThread.start()

        self.remained_share = 0
        self.isBuy = None
        self.remained_time = 5
        self.time_steps = time_steps
        self.target_shares = target_shares
        self.current_share = self.trader.get_portfolio_item(self.symbol)
        self.current_long = self.current_share.get_long_shares()
        self.current_short = self.current_share.get_short_shares()
        self.total_bp = self.trader.get_portfolio_summary().get_total_bp()

        self.endTime = datetime.strptime('15:55', '%H:%M').time()

    def set_objective(self, share, remained_time):
        self.remained_time = remained_time
        self.isBuy = share > 0
        self.remained_share = abs(share)

    @staticmethod
    def action_space():
        return 1

    @staticmethod
    def obs_space():
        return 10

    def get_signal(self):
        mid_price = []
        item = self.trader.get_portfolio_item(self.symbol)
        long_shares = item.get_long_shares()
        short_shares = item.get_short_shares()
        tab = self.priceTable
        if tab.isFull():
            for ele in tab.getData():
                mid_price.append(ele)
                # print(f"{self.symbol} mid price table {mid_price}")
            if self.trader.get_last_trade_time().time() > self.endTime:
                # close all positions for given ticker
                # close any long positions
                if long_shares > 0 and mid_price[-1] < mid_price[0]:
                    # print(f"limit selling because {self.symbol} long shares = {long_shares}")
                    self.set_objective(-self.target_shares, self.time_steps)
                # close any short positions
                if short_shares > 0 and mid_price[-1] > mid_price[0]:
                    # print(f"limit buying because {self.symbol} short shares = {short_shares}")
                    self.set_objective(self.target_shares, self.time_steps)
            else:
                if mid_price[-1] > mid_price[0]:
                    self.set_objective(self.target_shares, self.time_steps)
                if mid_price[-1] < mid_price[0]:
                    self.set_objective(-self.target_shares, self.time_steps)

    def _link(self):
        while self.trader.is_connected() and self.thread_alive:
            bp = self.trader.get_best_price(self.symbol)
            best_bid = bp.get_bid_price()
            best_ask = bp.get_ask_price()
            last_price = self.trader.get_last_price(self.symbol)
            self.priceTable.insertData((best_ask + best_bid) / 2)
            # Update order book data
            Ask_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.GLOBAL_ASK, self.ODBK_range)
            Bid_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.GLOBAL_BID, self.ODBK_range)
            orders = OrderBook(Ask_ls, Bid_ls, last_price)
            self.table.insertData(orders)
            time.sleep(1)

    def step(self, action):
        premium = action
        trade_time = self.trader.get_last_trade_time().time()
        initial_short = self.trader.get_portfolio_item(self.symbol).get_short_shares()
        initial_long = self.trader.get_portfolio_item(self.symbol).get_long_shares()
        print("\n-------------------------------------------------")
        print(f"{self.symbol} : Initial Short: {initial_short} | Initial Long: {initial_long}")

        signBuy = 1 if self.isBuy else -1

        base_price = self._getClosePrice(self.remained_share)
        obj_price = base_price - signBuy * premium

        if self.remained_time > 0:
            orderType = shift.Order.LIMIT_BUY if self.isBuy else shift.Order.LIMIT_SELL
        else:
            orderType = shift.Order.MARKET_BUY if self.isBuy else shift.Order.MARKET_SELL

        order = shift.Order(orderType, self.symbol, self.remained_share, obj_price)
        # Corrected method name for submitting order
        self.trader.submit_order(order)

        if premium > 0:
            time.sleep(self.timeInterval * 5)
        else:
            time.sleep(self.timeInterval)

        self.trader.cancel_all_pending_orders()
        for order in self.trader.get_waiting_list():
            self.trader.submit_cancellation(order)

        print(f"{self.symbol} : premium: {premium}, signBuy: {signBuy}, obj_price: {obj_price},"
              f"base_price : {base_price}"
              f" orderType: {self.trader.get_submitted_orders()[-1].type}, ")

        post_short = self.trader.get_portfolio_item(self.symbol).get_short_shares()
        post_long = self.trader.get_portfolio_item(self.symbol).get_long_shares()

        if signBuy == 1:
            exec_share = (post_long + initial_short) - (initial_long + post_short)
            done = exec_share == self.remained_share * 100
        elif signBuy == -1:
            exec_share = (initial_long + post_short) - (post_long + initial_short)
            done = exec_share == self.remained_share * 100
        else:
            done = False
            exec_share = 0

        print(
            f"{self.symbol} : Post Short: {post_short} | Post Long: {post_long} | exec_share: {exec_share} | done: {done}")

        status = self.trader.get_submitted_orders()[-1].type

        if self.remained_time > 0:
            if premium > 0:
                if exec_share > 0:
                    reward = exec_share * premium + exec_share * 0.2
                else:
                    reward = -6
            else:
                if exec_share > 0:
                    reward = exec_share * premium - exec_share * 0.3
                else:
                    reward = -6
        else:
            reward = exec_share * 0 - exec_share * 0.3

        print(f"{self.symbol} : done, {done}, remained_share: {self.remained_share}, reward: {reward}")

        if self.remained_share == 0 or self.remained_time == 0:
            done = True
        self.remained_time -= 1
        next_obs = self._get_obs()
        time.sleep(self.timeInterval)

        return next_obs, reward, done, dict(), post_long, post_short, trade_time

    def _get_obs(self):
        return np.concatenate((self.compute(), np.array([self.remained_share,
                                                         self.remained_time,
                                                         self.current_long,
                                                         self.current_short,
                                                         self.total_bp])))

    def _getClosePrice(self, share):
        return self.trader.get_close_price(self.symbol, self.isBuy, int(abs(share)))

    def cancel_orders(self):
        # cancel all the remaining orders
        for order in self.trader.get_waiting_list():
            if (order.symbol == self.symbol):
                self.trader.submit_cancellation(order)
                sleep(3)  # the order cancellation needs a little time to go through

    def close_positions(self):
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
