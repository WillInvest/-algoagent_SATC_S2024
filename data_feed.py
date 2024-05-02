import pandas as pd
import threading
import time
from time import sleep
import shift
from collections import deque


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


def collect_high_frequency_data(trader, ticker, weighted_price, thread_alive, time_step=0.1, step_time=10):
    # Initialize a deque with a maximum length to store 100 data points
    interval_data_list = deque(maxlen=60)
    count = 0

    while thread_alive:
        count += 1
        # Collect bid and ask orders
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
        # print(f"length: {len(interval_data_list)}")
        # Process data less frequently (every 10 seconds)
        if len(interval_data_list) == 60 and count % 10 == 0:
            df = pd.DataFrame(interval_data_list)

            bid_results_df, ask_results_df = get_weighted(df)
            # print(f"bid_results_df: {bid_results_df}")
            # print(f"ask_results_df: {ask_results_df}")
            weighted_bid = calculate_weighted_price(bid_results_df)
            weighted_ask = calculate_weighted_price(ask_results_df)
            weighted_spread = weighted_ask - weighted_bid

            weighted_price.update(weighted_ask, weighted_bid, weighted_spread)
            # print(f"weighted_price: {weighted_price}")
        # Wait for the next time step before collecting more data
        time.sleep(time_step)

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

        print(trader.get_stock_list())

        trader.disconnect()
