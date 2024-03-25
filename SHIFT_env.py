import shift
from threading import Thread, Lock, Event
import pandas as pd
import numpy as np
import time
from time import sleep


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
                 commission):

        self.timeInterval = t
        self.symbol = symbol
        self.nTimeStep = nTimeStep
        self.ODBK_range = ODBK_range
        self.trader = trader
        self.commission = commission
        self.mutex = Lock()

        self.dataThread = Thread(target=self._link)
        self.table = CirList(nTimeStep)

        print('Waiting for connection', end='')
        for _ in range(5):
            time.sleep(1)
            print('.', end='')
        print()

        self.thread_alive = True
        self.dataThread.start()

        self.remained_share = None
        self.isBuy = None
        self.remained_time = None
        self.initial_pl = self.trader.get_portfolio_item(self.symbol).get_realized_pl()

    def set_objective(self, share, remained_time):
        self.remained_share = abs(share)

        self.remained_time = remained_time
        self.isBuy = True if share > 0 else False

    @staticmethod
    def action_space():
        return 1

    @staticmethod
    def obs_space():
        return 7

    def _link(self):
        while self.trader.is_connected() and self.thread_alive:
            last_price = self.trader.get_last_price(self.symbol)
            Ask_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.GLOBAL_ASK, self.ODBK_range)
            Bid_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.GLOBAL_BID, self.ODBK_range)
            orders = OrderBook(Ask_ls, Bid_ls, last_price)
            self.table.insertData(orders)
            #print(self.table)
            time.sleep(self.timeInterval)
        print('Data Thread stopped.')

    def step(self, action):
        print("remained_time", self.remained_time)
        print("---------------stepping---------------")
        premium = action[0]
        signBuy = 1 if self.isBuy else -1
        base_price = self._getClosePrice(self.remained_share)
        print("premium: ", premium, "signBuy: ", signBuy, "remained shares :", self.remained_share, "base_price: ", base_price)
        obj_price = base_price - signBuy * premium
        print("obj_price: ", obj_price)

        #print(f'obj price: {obj_price}')

        print("---------------Initial Portfolio---------------")
        print("initial shares holding:", self.trader.get_portfolio_item(self.symbol).get_long_shares())

        if self.remained_time > 0:
            orderType = shift.Order.LIMIT_BUY if self.isBuy else shift.Order.LIMIT_SELL
        else:
            orderType = shift.Order.MARKET_BUY if self.isBuy else shift.Order.MARKET_SELL

        order = shift.Order(orderType, self.symbol, self.remained_share, obj_price)
        # Corrected method name for submitting order
        self.trader.submit_order(order)
        time.sleep(1)

        print("---------------submitted orders---------------")
        print("Submitted orders size: %d" % self.trader.get_submitted_orders_size())
        self._getSubOrder()
        print("post-submit shares holding:", self.trader.get_portfolio_item(self.symbol).get_long_shares())


        print("---------------waiting list---------------")
        # Corrected method name for getting the waiting list size
        print(f'waiting list size : {len(self.trader.get_waiting_list())}')
        self._getWaitingList()

        if self.trader.get_waiting_list_size() > 0:
            print(f'order size: {order.size}')
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

        print("execute_shares: ", exec_share)
        print("post-waiting shares holding:", self.trader.get_portfolio_item(self.symbol).get_long_shares())


        # Define reward
        done = False
        #if self.remained_time > 0:
        if self.remained_share != 0:
            if premium > 0:
                reward = exec_share * premium + exec_share * 0.2
            else:
                reward = exec_share * premium - exec_share * 0.3
        else:
            reward = exec_share * 0 - exec_share * 0.3
            print("reward: ", reward)
            done = True
            print("remained_share: ", self.remained_share)
            print("Episode finished.")
            # cancel unfilled orders and close positions for this ticker
            #self.cancel_orders()
            #self.close_positions()
            self._getWaitingList()
            print("---------------ending portfolio---------------")
            print("ending shares holding:", self.trader.get_portfolio_item(self.symbol).get_long_shares())
            next_obs = self._get_obs()
            return next_obs, reward, done, dict()

        print("reward: ", reward)
        self.remained_time -= 1
        next_obs = self._get_obs()
        time.sleep(self.timeInterval)

        return next_obs, reward, done, dict()

    def _get_obs(self):
        return np.concatenate((self.compute(), np.array([self.remained_share, self.remained_time])))

    def _getClosePrice(self, share):
        return self.trader.get_close_price(self.symbol, self.isBuy, abs(share))

    def _getSubOrder(self):
        for order in self.trader.get_submitted_orders():
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
            sleep(3)  # we sleep to give time for the order to process

        # close any short positions
        short_shares = item.get_short_shares()
        if short_shares > 0:
            print(f"market buying because {self.symbol} short shares = {short_shares}")
            order = shift.Order(shift.Order.Type.MARKET_BUY,
                                self.symbol, int(short_shares / 100))
            self.trader.submit_order(order)
            sleep(3)

    def reset(self):
        return self._get_obs()

    def kill_thread(self):
        self.thread_alive = False

    def compute(self):
        tab = self.table
        feature = []
        if tab.isFull():
            for ele in tab.getData():
                n_ask = ele.n_ask
                assert n_ask > 0, f'The number of Ask order is 0'
                n_bid = ele.n_bid
                assert n_bid > 0, f'The number of Bid order is 0'
                bas = self._ba_spread(ele.df, n_ask)
                p = self._price(ele.df)
                sp = self._smart_price(ele.df, n_ask)
                li = self._liquid_imbal(ele.df, n_ask, n_bid)
                act_direction = 'Buy' if self.isBuy else 'Sell'
                if act_direction:
                    mc, _ = self._market_cost(ele.df,
                                              n_ask,
                                              n_bid,
                                              act_direction,
                                              self.remained_share,
                                              self.commission)
                    feature += [bas, p, sp, li, mc]
                else:
                    feature += [bas, p, sp, li, np.nan]
        return np.array(feature)

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


# if __name__ == '__main__':
#     with shift.Trader("algoagent_test004") as trader:
#         trader.connect("initiator.cfg", "x6QYVYRT")
#         sleep(1)  # Ensure connection is established
#         trader.sub_all_order_book()
#         sleep(1)
#
#         env = SHIFT_env(trader, 1, 1, 5, 'AAPL', 0.003)
#         env.set_objective(10, 5)
#
#         for i in range(100):
#             action = 0.1 * np.random.uniform(0, 0.1, size=1)
#             obs, reward, done, _ = env.step(action)
#             if done:
#                 print('finished')
#                 break
#
#
#         env.kill_thread()
#         trader.disconnect()
