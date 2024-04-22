import pandas as pd
import threading
import time
from time import sleep
import shift



class WeightedPrice:
    def __init__(self):
        self.data_lock = threading.Lock()
        self.weighted_ask = 0
        self.weighted_bid = 0
        self.weighted_spread = 0

    def update(self, ask, bid, spread):
        with self.data_lock:
            self.weighted_ask = ask
            self.weighted_bid = bid
            self.weighted_spread = spread

    def get_prices(self):
        with self.data_lock:
            return self.weighted_ask, self.weighted_bid, self.weighted_spread

def collect_high_frequency_data(trader, ticker, weighted_price, thread_alive, time_interval=10):
    interval_data_list = []
    while thread_alive:
        bid_orders = trader.get_order_book(ticker, shift.OrderBookType.LOCAL_BID, 1)
        ask_orders = trader.get_order_book(ticker, shift.OrderBookType.LOCAL_ASK, 1)

        if bid_orders and ask_orders:
            best_bid = bid_orders[0]
            best_ask = ask_orders[0]
            interval_data_list.append({
                'Bid Price': best_bid.price,
                'Bid Size': best_bid.size,
                'Ask Price': best_ask.price,
                'Ask Size': best_ask.size
            })

        # Process data less frequently
        if len(interval_data_list) >= time_interval:  # For example, process every 10 seconds
            df = pd.DataFrame(interval_data_list)
            bid_results_df, ask_results_df = get_weighted(df)
            weighted_bid = calculate_weighted_price(bid_results_df)
            weighted_ask = calculate_weighted_price(ask_results_df)
            weighted_spread = weighted_ask - weighted_bid
            weighted_price.update(weighted_ask, weighted_bid, weighted_spread)
            interval_data_list.clear()  # Clear after processing

        sleep(1)

def get_weighted(df):
    # Split the DataFrame into bids and asks
    bid_df = df[['Bid Price', 'Bid Size']].copy()
    ask_df = df[['Ask Price', 'Ask Size']].copy()

    # Initialize dictionaries to store cumulative sizes and last recorded sizes for each price
    bid_cumulative_sizes = {}
    bid_last_sizes = {}
    ask_cumulative_sizes = {}
    ask_last_sizes = {}

    # Process bids
    for _, row in bid_df.iterrows():
        price, size = row['Bid Price'], row['Bid Size']
        if price not in bid_cumulative_sizes:
            bid_cumulative_sizes[price] = size  # Initialize with the first size
            bid_last_sizes[price] = size  # Also store the first size as the last size
        else:
            # Calculate the difference from the last size and update cumulative size
            size_change = abs(size - bid_last_sizes[price])
            bid_cumulative_sizes[price] += size_change
            bid_last_sizes[price] = size  # Update the last size for this price

    # Process asks
    for _, row in ask_df.iterrows():
        price, size = row['Ask Price'], row['Ask Size']
        if price not in ask_cumulative_sizes:
            ask_cumulative_sizes[price] = size  # Initialize with the first size
            ask_last_sizes[price] = size  # Also store the first size as the last size
        else:
            # Calculate the difference from the last size and update cumulative size
            size_change = abs(size - ask_last_sizes[price])
            ask_cumulative_sizes[price] += size_change
            ask_last_sizes[price] = size  # Update the last size for this price

    # Convert dictionaries to DataFrames
    bid_results_df = pd.DataFrame(list(bid_cumulative_sizes.items()), columns=['Price', 'Cumulative Size'])
    ask_results_df = pd.DataFrame(list(ask_cumulative_sizes.items()), columns=['Price', 'Cumulative Size'])

    return bid_results_df, ask_results_df


def calculate_weighted_price(df):
    df['Price'] = df['Price'].astype(float)
    df['Cumulative Size'] = df['Cumulative Size'].astype(float)

    total_weighted_price = (df['Price'] * df['Cumulative Size']).sum()

    total_cumulative_size = df['Cumulative Size'].sum()

    if total_cumulative_size > 0:
        weighted_price = total_weighted_price / total_cumulative_size
        return weighted_price
    else:
        return 0

if __name__ == '__main__':
    with shift.Trader("algoagent") as trader:
        trader.connect("initiator.cfg", "x6QYVYRT")
        sleep(1)
        trader.sub_all_order_book()
        sleep(1)

        tickers = ['CS1']
        print(f"ticker: {tickers}")

        print("  Price\t\tSize\t  Dest\t\tTime")
        for order in trader.get_order_book("CS1", shift.OrderBookType.LOCAL_BID, 5):
            print(
                "%7.2f\t\t%4d\t%6s\t\t%19s"
                % (order.price, order.size, order.destination, order.time)
            )

        print("  Price\t\tSize\t  Dest\t\tTime")
        for order in trader.get_order_book("CS1", shift.OrderBookType.LOCAL_ASK, 5):
            print(
                "%7.2f\t\t%4d\t%6s\t\t%19s"
                % (order.price, order.size, order.destination, order.time)
            )

        for _ in range(100):
            for ticker in tickers:
                bids = trader.get_order_book(ticker, shift.OrderBookType.LOCAL_BID, 1)
                asks = trader.get_order_book(ticker, shift.OrderBookType.LOCAL_ASK, 1)
                last_price = trader.get_last_price(ticker)
                best_price = trader.get_best_price(ticker)
                print(f"last_price: {last_price}")
                print(f"best_ask: {best_price.get_ask_price()} | {asks[0].price}")
                print(f"best_bid: {best_price.get_bid_price()} | {bids[0].price}")
                sleep(1)

        trader.disconnect()
