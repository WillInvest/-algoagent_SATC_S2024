import shift
from threading import Thread, Lock, Event
import pandas as pd
import numpy as np
import time


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
        #         trader = shift.Trader("test001")
        #         assert trader.connect("initiator.cfg", "password"), f'Connection error'
        #         assert trader.subAllOrderBook(), f'subscription fail'

        self.timeInterval = t
        self.symbol = symbol
        self.nTimeStep = nTimeStep
        self.ODBK_range = ODBK_range
        self.trader = trader
        self.commission = commission
        self.mutex = Lock()

        self.dataThread = Thread(target=self._link)
        # features = ['symbol', 'orderPrice', 'orderSize', 'OrderTime']
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
            last_price = self.trader.getLastPrice(self.symbol)

            Ask_ls = self.trader.getOrderBook(self.symbol, shift.OrderBookType.GLOBAL_ASK, self.ODBK_range)
            # assert Ask_ls, f'getOrderBook: return empty list: {self.symbol}-ASK-{self.ODBK_range}'

            Bid_ls = self.trader.getOrderBook(self.symbol, shift.OrderBookType.GLOBAL_BID, self.ODBK_range)
            # assert Bid_ls, f'getOrderBook: return empty list: {self.symbol}-BID-{self.ODBK_range}'

            orders = OrderBook(Ask_ls, Bid_ls, last_price)

            self.mutex.acquire()
            # print(88)
            self.table.insertData(orders)
            # print(tmp)
            self.mutex.release()

            time.sleep(self.timeInterval)
        print('Data Thread stopped.')


    def step(self, action):
        premium = action[0]
        signBuy = 1 if self.isBuy else -1
        base_price = self._getClosePrice(self.remained_share)
        obj_price = base_price - signBuy * premium

        print(f'obj price: {obj_price}')

        if self.remained_time > 0:
            orderType = shift.Order.LIMIT_BUY if self.isBuy else shift.Order.LIMIT_SELL
        else:
            orderType = shift.Order.MARKET_BUY if self.isBuy else shift.Order.MARKET_SELL

        order = shift.Order(orderType, self.symbol, self.remained_share, obj_price)
        self.trader.submitOrder(order)

        time.sleep(self.timeInterval)

        print(f'waiting list size : {len(self.trader.getWaitingList())}')
        if self.trader.getWaitingListSize() > 0:
            print(f'ordre size: {order.size}')
            if order.type == shift.Order.LIMIT_BUY:
                order.type = shift.Order.CANCEL_BID
            else:
                order.type = shift.Order.CANCEL_ASK
            self.trader.submitOrder(order)
            exec_share = self.remained_share - order.size
            self.remained_share = order.size
        else:
            exec_share = self.remained_share
            self.remained_share = 0
        done = False
        if self.remained_time > 0:
            if premium > 0:
                reward = exec_share * premium + exec_share * 0.2
            else:
                reward = exec_share * premium - exec_share * 0.3
        else:
            reward = exec_share * 0 - exec_share * 0.3
            done = True

        self.remained_time -= 1
        next_obs = self._get_obs()
        return next_obs, reward, done, dict()


    def _get_obs(self):
        return np.concatenate((self.compute(), np.array([self.remained_share, self.remained_time])))


    def _getClosePrice(self, share):
        return self.trader.getClosePrice(self.symbol, self.isBuy, abs(share))


    def reset(self):
        return self._get_obs()


    def kill_thread(self):
        self.thread_alive = False


    def compute(self):
        tab = self.table
        feature = []
        if tab.isFull():
            for ele in tab.getData():
                # feat_step = pd.DataFrame(columns=['ba_spread', 'price', 'smart_price', 'liquid_imbal', 'market_cost'])
                # n_ask = ele.type.value_counts()['Ask']
                n_ask = ele.n_ask
                assert n_ask > 0, f'The number of Ask order is 0'
                # n_bid = len(ele) - n_ask
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


    #         self.trader.disconnect()


if __name__ == '__main__':
    import numpy as np

    trader = shift.Trader("test001")
    trader.disconnect()
    trader.connect("initiator.cfg", "password")
    #trader.subAllOrderBook()
    # if trader.isConnected() and trader.subAllOrderBook():
    #     print(trader.isConnected())
    #     order = trader.getOrderBook("AAPL", shift.OrderBookType.GLOBAL_ASK, 1)[0]
    #     print([order.price, order.size, 'Ask'])

    # time.sleep(5)
    # for i in range(10000):
    #     order = trader.getOrderBook("AAPL", shift.OrderBookType.GLOBAL_ASK, 1)[0]
    #     print([order.price, order.size, 'Ask'])
    #     del order
    #     time.sleep(1)

    env = SHIFT_env(trader, 2, 1, 5, 'AAPL', 0.003)
    env.set_objective(5, 5)
    # for  i in range(1000):
    #     print(f'time step {i}')
    #     for ele in env.table.getData():
    #         print(ele)
    #     print(env.compute())
    #     time.sleep(1)
    #     print()

    for i in range(100):
        action = 0.1 * np.random.uniform(0, 0.1, size=1)
        print(f'premium: {action[0]}')
        obs, reward, done, _ = env.step(action)
        print(obs)
        print(reward)
        print(done)
        print()
        if done:
            print('finished')
            break

    order = trader.getPortfolioItem('AAPL')
    print(order.getShares())

    print(trader.getPortfolioSummary().getTotalBP())
    # env.dataThread.is_alive()
    #
    env.kill_thread()
    #
    env.dataThread.is_alive()

    trader.disconnect()