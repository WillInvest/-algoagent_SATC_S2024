from threading import Thread, Lock
import pandas as pd
import numpy as np
import time
import os
import shift
import gymnasium as gym
from copy import deepcopy
from datetime import datetime
from collections import deque


class CirList:
    def __init__(self, length):
        self.size = length
        self._table = [None]*length
        self.idx = 0
        self._counter = 0

    def insertData(self, data):
        self._counter += 1
        self._table[self.idx] = data
        self.idx = (self.idx+1) % self.size

    def getData(self):
        tail = self._table[0:self.idx]
        head = self._table[self.idx:]
        ret = head+tail
        return ret.copy()

    def isFull(self):
        return self._counter >= self.size

    def __repr__(self):
        return str(self.getData())


class SHIFT_env(gym.Env):
    orders = (shift.Order.Type.MARKET_SELL, 
              shift.Order.Type.MARKET_BUY, 
              shift.Order.Type.LIMIT_SELL, 
              shift.Order.Type.LIMIT_BUY)


    def __init__(self,
                 trader,
                 rl_t,
                 info_step,
                 nTimeStep,
                 ODBK_range,
                 symbol,
                 action_sym = [-1,1],
                 commission = 0,
                 rebate = 0,
                 save_data = False, 
                 max_order_list_length = 20,
                 natural_stop = None,
                 #identifiers
                 alpha = 0.09,
                 weight = 0.5,
                 gamma = 0.09,
                 tar_m = 0.8,
                 
                 #self generate LOB
                 start_price = 100,
                 init = False):
        

        
        #newly added##############
        #objective: control the mid price
        #self.target_mid_price = target_price

        #bp threshold:
        self.bp_thres = 50000
        self.init_bp = trader.get_portfolio_summary().get_total_bp()

        self.rl_t = rl_t
        self.info_step = info_step
        self.symbol = symbol
        self.nTimeStep = nTimeStep
        self.ODBK_range = ODBK_range
        self.trader = trader
        self.commission = commission
        self.rebate = rebate
        self.mutex = Lock()
        # self.strategy_mutex = Lock()
        self.order_mutex = Lock()
        
        #initialize LOB           #keep this TRUE if simulating mm only#####
        self.init = init
        
        self.last_available_price = 100#record last price if price has a huge jump
        self.last_available_sp = 0.01

        self.data_thread = Thread(target=self._link)
        # features = ['symbol', 'orderPrice', 'orderSize', 'OrderTime']
        self.table = CirList(nTimeStep) #contains: 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp'
        self._cols = ['reward', 'action_size', 'action_sym', 'action_asym', 'recent_pl_change', 'change_inven_pnl', 
                      'diff_market_share', 'curr_mp', 'current_inventory', 'current_market_share', 'total_volume', 'bp', "spread",
                      "bid_5","bid_5_p","bid_4","bid_4_p","bid_3","bid_3_p","bid_2","bid_2_p","bid_1","bid_1_p",
                      "ask_1_p","ask_1","ask_2_p","ask_2","ask_3_p","ask_3","ask_4_p","ask_4","ask_5_p","ask_5", 
                      "Bid_depth","Ask_depth"]#, "time"

        self.df = pd.DataFrame(columns=self._cols)
        self.df_idx = 0
        self.order_cols = ["timestep", "Type", "Status", "Symbol", "Size", "Executed_size", "Price", "Executed Price", "ID"]
        self.order_df = pd.DataFrame(columns = self.order_cols)
        self.spread_deque = deque(maxlen=20)

        print('Waiting for connection', end='')
        for _ in range(5):
            time.sleep(1)
            print('.', end='')
        print()

        self.thread_alive = True
        self.data_thread.start()

        self.remained_share = None
        self.total_share = None
        self.currentPos = None
        self.objPos = None
        self.isBuy = None
        self.remained_time = None
        self.tmp_obs = [None]*7
        self.name = 'exec_one_asset'
        self.isSave = save_data
        
        self.current_action_index = None
        
        self.order_list = []
        self.max_order_list_length = max_order_list_length
        self.order_thread = Thread(target = self._maintain_order_list)
        self.order_thread.start()
        
        self.strategy = self.void_action = lambda *args: None
        

        #Actions: ############
        #self.maxsize = 8                  ##############################important for action size#########################################        
        #now it's % instead of action order size                                #5 to 50 order size
        self.action_space= gym.spaces.Box(low = np.array([0.000005, action_sym[0], -1]), high = np.array([0.00005, action_sym[1], 1]), shape = (3,)) #gym.spaces.Box(low = np.array([1]), high = np.array([5])) #dtype=np.float32, 
        self.observation_space = gym.spaces.Box(np.array([0,0,0,0, 0,0,0,0,0,  -np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                               np.array([0,0,0,0,
                                                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                                         ]), dtype = np.float64)
        

        #new:
        self.current_state_info = self.compute_state_info()
        self.total_pl = 0
        self.alpha = alpha    #to normalize pnl part of the reward
        self.w = weight        #weight on pnl
        self.gamma = gamma    #inventory risk aversion
        self.target_market_share = tar_m  #m*

        self.step_counter = 0
        self.natural_stop = natural_stop
        self.last_market_share = 0 
        self.last_inventory_pnl = 0
    
        
    # @staticmethod
    # def action_space():
    #     return 3

    # @staticmethod
    # def state_space():
    #     return 4

    #thread constantly collecting order book data and Last Price
    def _link(self):
        print(f"Data thread starts")
        while self.trader.is_connected() and self.thread_alive:

            item = self.trader.get_portfolio_item(self.symbol)

            Ask_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.LOCAL_ASK, self.ODBK_range)
            # assert Ask_ls, f'getOrderBook: return empty list: {self.symbol}-ASK-{self.ODBK_range}'

            Bid_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.LOCAL_BID, self.ODBK_range)
            # assert Bid_ls, f'getOrderBook: return empty list: {self.symbol}-BID-{self.ODBK_range}'
            
            #get remaining buying power
            bp = self.trader.get_portfolio_summary().get_total_bp()

            #get best bid and ask prices
            best_p = self.trader.get_best_price(self.symbol)

            info = self.LOB_to_list(Ask_ls, Bid_ls, best_p, bp, item)

            self.mutex.acquire()
            # print(88)
            self.table.insertData(info) #table info: [mid, ask_size, bid_size, bp, inventory]
            # print(tmp)
            self.mutex.release()

            time.sleep(self.info_step)
        print('Data Thread stopped.')
    
    def LOB_to_list(self, ask_book, bid_book, best_p, bp, item): #return a list with these info 'curr_mp', 'ask_book', 'bid_book', 'remained_bp', inventory, spread, 'ask_book_price', 'bid_book_price' 
        # get LOB
        bid_b = []
        ask_b = []
        bid_p = []
        ask_p = []
        for order in ask_book:
            ask_b.append(order.size) 
            ask_p.append(order.price) 
        for order in bid_book:
            bid_b.append(order.size)
            bid_p.append(order.price)
        
        while len(ask_b) < 5:
            ask_b.append(0)
            ask_p.append(0)
        while len(bid_b) < 5:
            bid_b.append(0)
            bid_p.append(0)
            
        #inventory:
        short_shares = int(item.get_short_shares())
        long_shares = int(item.get_long_shares())
        inventory = long_shares - short_shares
        
        # # #initialize lob:
        if self.init:
            #self.init = False
            return [100, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], self.init_bp, 0, 0.02]
        
        #spread
        sp = abs(round((best_p.get_ask_price() - best_p.get_bid_price()),3))
        
        #mid price
        mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),3)
        #check if mid price has a huge jump due to 0 liquidity

        if abs(mid - self.last_available_price) >= 10 or mid == 0 or min(best_p.get_ask_price(),best_p.get_bid_price()) == 0:
            mid = max(self.last_available_price,max(best_p.get_ask_price(),best_p.get_bid_price())) #self.table.getData()[self.nTimeStep-1][0]
            sp = self.last_available_sp#self.table.getData()[self.nTimeStep-1][5]
        self.last_available_price = mid
        self.last_available_sp = sp

        return [mid, ask_b, bid_b, bp, inventory, sp, ask_p, bid_p]

    def compute_state_info(self):
        #return the following items 'curr_mp', 'ask_book', 'bid_book', 'remained_bp', 'current_inventory', spread, "his_prices", 'current_inventory_pnl', total_volume, 'ask_book_price', 'bid_book_price' 
        big_to_small = int(self.rl_t / self.info_step)  #since 5 * 0.4 = 2 which is the time for each step
        tab = self.table
        while True:
            if tab.isFull():
                his_mp = []
                his_inventory = []
                for ele in tab.getData():
                    his_mp.append(ele[0])
                    his_inventory.append(ele[4])
                
                #get the price change at from last timestep
                price_change = his_mp[self.nTimeStep-1]-his_mp[self.nTimeStep - big_to_small]          
                #get the inventory pnl
                current_inventory_pnl = price_change * his_inventory[self.nTimeStep - big_to_small]    #since 5 * 0.4 = 2 which is the time for each step
                
                book = np.asarray(tab.getData()[self.nTimeStep-1][1] + tab.getData()[self.nTimeStep-1][2])
                total_volume = np.sum(book)
                #mean_inven_pnl = np.mean(his_inventory_pnl_np)
                #std_inven_pnl = abs(np.std(his_inventory_pnl_np))
                return (tab.getData()[self.nTimeStep-1][0:6] + [his_mp[5:self.nTimeStep]] + [current_inventory_pnl, total_volume]+tab.getData()[self.nTimeStep-1][6:])
            else:
                print("need to wait for table to fill up")
                time.sleep(1)
    
    
    def load_action_by_index(self, actions=None):
        if actions is None:
            return
        else: 
            parameters = actions #self.action_list[index]
            order_size = int(round(self.trader.get_portfolio_summary().get_total_bp() * actions[0] / (100 * self.last_available_price),0))           #order size#self.trader.get_last_price(self.symbol)
            self.generate_mm_trader(self.order_list, self.trader, self.symbol, order_size, parameters[1], parameters[2])
        self.current_action_index = actions
        #print(f"Action: {self.action_list[index]}, {self.symbol}")

    def generate_mm_trader(self, 
                           order_list: list, 
                           trader,
                           ticker: str,
                           order_size: int,    
                           symmetric_e = float,
                            asymmetric_e = float):
        # get necessary data
        
        mid = self.table.getData()[self.nTimeStep-1][0]
        spread = self.table.getData()[self.nTimeStep-1][5]
        #*********************************************************need to ask if ok**********************************************
        spread = np.mean(self.spread_deque)
        if spread == 0: spread = 0.01
        #elif spread > 1: spread = 0.01
        # get inventory
        item = trader.get_portfolio_item(ticker)
        imbalance = int(item.get_long_shares()/100 - item.get_short_shares()/100)      #inventory imbalance = long - short
        # print("imbalance", imbalance)
        # print("bp",self.trader.get_portfolio_summary().get_total_bp())
        # submit limit orders
        if order_size > 0:
            p_ask = round((mid + (spread*(0.5*(1+symmetric_e) + asymmetric_e))),2)
            p_bid = round((mid - (spread*(0.5*(1+symmetric_e) - asymmetric_e))),2)
            limit_buy = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, order_size, p_bid)
            limit_sell = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, order_size, p_ask)
            #print("size",order_size)
            order_list.append(limit_buy.id)
            trader.submit_order(limit_buy)
            order_list.append(limit_sell.id)
            trader.submit_order(limit_sell)


    
    def get_states(self):
        #returns:  identifiers | his_prices | current_inventory | last market share | ask book | bid book |  bp
        #identifier:
        ident = [self.alpha, self.w, self.gamma, self.target_market_share]
        """
        is_bp_risk = 0
        if self.current_state_info[3] <= self.bp_thres:
            is_bp_risk = 1"""
        #print(np.array(self.current_state_info[6] + [self.current_state_info[4]] + [self.last_market_share] + self.current_state_info[1] + self.current_state_info[2] + diff_hedging_cost))
        return np.array(ident + self.current_state_info[6] + [self.current_state_info[4]] + [self.last_market_share] + self.current_state_info[1] + self.current_state_info[2] + [100*self.current_state_info[3]/self.init_bp])
    
    def step(self, actions):
        if self.init:
            self.init = False
            limit_buy = shift.Order(shift.Order.Type.LIMIT_BUY, self.symbol, 50, 100-0.02)
            limit_sell = shift.Order(shift.Order.Type.LIMIT_SELL, self.symbol, 50, 100+0.02)    
            self.order_list.append(limit_buy.id)
            self.trader.submit_order(limit_buy)
            self.order_list.append(limit_sell.id)
            self.trader.submit_order(limit_sell)
            time.sleep(3)
        actions = actions
        #get current total pl by size:
        self.total_pl = self.trader.get_portfolio_summary().get_total_realized_pl()# + self.trader.get_unrealized_pl(symbol=self.symbol)
        #print(f"ticker{self.symbol}, action {action}")
        #save LOB depth info:
        Ask_book = self.trader.get_order_book(self.symbol, shift.OrderBookType.LOCAL_ASK, 99)
        Bid_book = self.trader.get_order_book(self.symbol, shift.OrderBookType.LOCAL_BID, 99)
        bid_depth = int(len(Bid_book))
        ask_depth = int(len(Ask_book))
        
        #action = [5,1,0,1]
        self.load_action_by_index(actions)
        #print("actions", actions)

        #print(actions)        
        time.sleep(self.rl_t)
        
        #STATES: 7 components: ##################################################################################################################
        #   identifiers | his_prices | current_inventory | last market share | ask book | bid book | different hedge cost 
        state_info = self.compute_state_info() #:'curr_mp', 'ask_book', 'bid_book', 'remained_bp', 'current_inventory', spread, "his_prices", 'current_inventory_pnl', total_volume, 'ask_book_price', 'bid_book_price' 
        #print(state_info)
        self.current_state_info = state_info
        state = self.get_states()  #: identifiers | his_prices | current_inventory | last market share | ask book | bid book | bp
        #print(state)
        #calculate reward: ###########################################################################################################################################
        recent_pl_change = self.trader.get_portfolio_summary().get_total_realized_pl() - self.total_pl# + self.trader.get_unrealized_pl(symbol=self.symbol) 
        #market share
        current_order_sizes = 0
        for order in self.trader.get_waiting_list():
            current_order_sizes += order.size - order.executed_size
        if state_info[8] == 0:
            current_market_share = 0
        else: current_market_share = current_order_sizes / state_info[8] #changed to current agent' shares / market total shares, # actions[0]

        diff_market_share = abs((current_market_share - self.target_market_share))# paper may has a typo：  - (self.last_market_share - self.target_market_share)
        change_inven_pnl = abs(state_info[7] - self.last_inventory_pnl)
        reward =  self.w * self.alpha * (recent_pl_change - (self.gamma * change_inven_pnl)) - (1-self.w) * diff_market_share

        #print(f"Reward: {reward}")

        curr_mp = state_info[0]
        current_inventory = state_info[4]
        total_volume = state_info[8]
        bp = self.trader.get_portfolio_summary().get_total_bp()
        
        self.last_market_share = current_market_share
        self.last_inventory_pnl = state_info[7]
        
        done = False
        
        lob = []
        for i in range(5):
            lob.append(state_info[2][::-1][i])
            lob.append(state_info[10][::-1][i])
        for i in range(5):
            lob.append(state_info[9][i])
            lob.append(state_info[1][i])
            
        
        self.spread_deque.append(state_info[5])
        # if self.step_counter % 100 == 0:
        #     self.save_to_csv(self.name)
        if self.natural_stop != None:
            self.step_counter+=1
            if self.step_counter > self.natural_stop:
                done = True
        #print(reward)
        # print(actions)
        return np.array(state), reward, done, False, dict()
            
            
    def _maintain_order_list(self):
        print(f"Order thread starts")
        self.order_thread_alive = True
        while self.trader.is_connected() and  self.order_thread_alive:
            for i, order_id in enumerate(self.order_list):
                order = self.trader.get_order(order_id)
                # print(type(order.status))
                if order.status == shift.Order.Status.CANCELED or \
                   order.status == shift.Order.Status.REJECTED or \
                   order.status == shift.Order.Status.FILLED:
                    popped_order = self.order_list.pop(i)
                    o = self.trader.get_order(popped_order)
                    #print(f"remove order: {o.type, o.price, o.size, o.status, o.id}, order list length: {len(self.order_list)}")
                    break
            #print(f"{len(self.order_list)}/{self.max_order_list_length}")
            if len(self.order_list) >= self.max_order_list_length:
                for i, order_id in enumerate(self.order_list[:int(self.max_order_list_length/2)]):
                    order = self.trader.get_order(order_id)
                    self.trader.submit_cancellation(order)
                    # print(f"try to cancel: {order_id}")
                    
            time.sleep(0.1)
        print("Order cancel thread done. ")
                    
        
 
    def reset(self):
        # print(f'Holding shares: {self.trader.getPortfolioItem(self.symbol).getShares()}')
        # print(f'Buying Power: {self.trader.getPortfolioSummary().getTotalBP()}')
        
        self.load_action_by_index()
        self.cancel_all()
        self.close_positions()
        self.current_state_info = self.compute_state_info() 
        print(self.get_states())
        return self.get_states(), dict()
    
    def cancel_all(self):
        
        self.load_action_by_index()
        self.trader.cancel_all_pending_orders()
        for _ in range(5):
            for i, order_id in enumerate(self.order_list):
                order = self.trader.get_order(order_id)
                # print(type(order.status)
                self.trader.submit_cancellation(order)
            time.sleep(0.1)
            if len(self.order_list) == 0:
                break
            else:
                print("Tried to cancel existing orders.")
                continue
        print("all order cancelled")

                    

    def save_to_csv(self, filelocation): # TODO replace with logger
        print("saving")
        #saving the orders submitted
        # step = 0
        # for order in self.trader.get_submitted_orders():
        #     order_info = [order.timestamp, order.type, order.status, order.symbol,order.size, order.executed_size, order.price,
        #         order.executed_price, order.id]
        #     self.order_df.loc[step] = order_info
        #     step += 1
        #     print(step)
        #save the states, reward, LOB infos 
        #try:
        self.df.to_csv(filelocation)
        self.df = pd.DataFrame(columns=self._cols)
            #self.order_df.to_csv(f'./iteration_info/Orders_info_{epoch}.csv')
        # except FileNotFoundError:
        #     os.makedirs(f'./iteration_info/', exist_ok= True)
        #     self.df.to_csv(filelocation)
        #     self.df = pd.DataFrame(filelocation)
            #self.order_df.to_csv(f'./iteration_info/Orders_info_{epoch}.csv')


    def kill_thread(self):
        self.thread_alive = False


    def close_positions(self):
        trader = self.trader
        ticker = self.symbol
        # close all positions for given ticker
        print('running close positions function for', ticker)
        
        item = trader.get_portfolio_item(ticker)

        # close any long positions
        long_shares = item.get_long_shares()
        if long_shares > 0:
            print("market selling because long shares =", long_shares)
            order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(long_shares/100))
            trader.submit_order(order)
            time.sleep(0.2)
            # make sure order submitted correctly
            counter = 0
            item = trader.get_portfolio_item(ticker)
            order = trader.get_order(order.id)
            while (item.get_long_shares() > 0):
                print(order.status)
                time.sleep(0.2)
                order = trader.get_order(order.id)
                item = trader.get_portfolio_item(ticker)
                counter += 1
                # if order is not executed after 5 seconds, then break
                if counter > 5:
                    break
                

        # close any short positions
        short_shares = item.get_short_shares()
        if short_shares > 0:
            print("market buying because short shares =", short_shares)
            order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(short_shares/100))
            trader.submit_order(order)
            time.sleep(0.2)
            # make sure order submitted correctly
            counter = 0
            item = trader.get_portfolio_item(ticker)
            order = trader.get_order(order.id)
            while (item.get_short_shares() > 0):
                print(order.status)
                time.sleep(0.2)
                order = trader.get_order(order.id)
                item = trader.get_portfolio_item(ticker)
                counter += 1
                # if order is not executed after 5 seconds, then break
                if counter > 5:
                    break
            
    def __del__(self):
        self.kill_thread()
        
    def __call__(self, *args, **kwds):
        return self
 