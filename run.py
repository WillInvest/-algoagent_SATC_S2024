from SHIFT_env import *
from shift_agent import *
import threading
from multiprocessing import Barrier
import datetime

TRAINING = True
LOADING_TRAINED_MODEL = True

CLIP_RANGE = 0.2
VF_COEFFICIENT = 1.0
weights_path = 'ppo_network_weights.h5'  # Define a path for the weights
decay_rate = 0.995


def run_episodes(worker_id, env, agent, total_episodes, ticker, lock, gradient_list, barrier, new_weight):
    # When not loading trained model,
    if TRAINING:
        entropy_coefficient = 0.1
    else:
        entropy_coefficient = 0
    # initialize parameters to initial state by saving the initial parameters of the 1st worker.
    # Load weights if they exist

    for episode_cnt in range(total_episodes):

        entropy_coefficient *= decay_rate

        if episode_cnt % 1 == 0:
            if worker_id == 0:
                agent.network.save_weights(weights_path)
                print(f"{episode_cnt} : save weight to {weights_path}")

        with lock:
            agent.network.set_weights(new_weight)

        timeout = False
        loss = None  # Initialize loss as None to handle cases where timeout is True
        current_time = datetime.datetime.now()
        env.isBuy = None
        while env.isBuy is None:
            env.get_signal()
            if (datetime.datetime.now() - current_time).total_seconds() > 1:
                timeout = True
                env.set_objective(0, 10)
                break

        state = agent.reset()

        with tf.GradientTape() as tape:
            if not timeout:
                for time_step in range(env.remained_time + 1, -1, -1):
                    action, state_value = agent.feed_networks(state)
                    action = action * state[0]
                    next_state, reward, done, _, long, short, trade_time = env.step(action)

                    agent.reward_list.append(reward)
                    state = np.copy(next_state)

                    if done or time_step <= 0:
                        agent.episode_return = sum(agent.reward_list)
                        print("Episode: #", episode_cnt, "ticker:", ticker, "Return: ", agent.episode_return,
                              "long_shares: ", long, "short_shares: ", short, "trade_time: ", trade_time)
                        break

                # Calculate the discounted return for each step in the episodic history.
                return_list = []
                discounted_sum = 0
                agent.reward_list.reverse()
                for r in agent.reward_list:
                    discounted_sum = r + GAMMA * discounted_sum
                    return_list.append(discounted_sum)
                return_list.reverse()
                agent.reward_list.reverse()

                # Calculate the loss.
                actor_object_list = agent.prob_ratio_list
                old_action_prob_list = agent.old_action_prob_list
                # TODO: Test more into the effects of GAE.

                history = zip(agent.state_value_list, actor_object_list, return_list, old_action_prob_list)
                actor_losses = []
                critic_losses = []

                for state_value, actor_object, ret, old_action_prob in history:
                    advantage = ret - state_value
                    clipped_actor_object = tf.clip_by_value(actor_object, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
                    actor_loss = -tf.minimum(actor_object * advantage, old_action_prob * advantage)
                    actor_losses.append(actor_loss)
                    state_value = tf.expand_dims(state_value, 0)
                    ret = tf.expand_dims(ret, 0)
                    # Use huber loss rather than MSE loss to improve stability.
                    critic_loss = agent.huber_loss(state_value, ret)
                    agent.critic_loss(critic_loss)
                    critic_losses.append(critic_loss)

                loss = \
                    tf.reduce_mean(actor_losses) \
                    + VF_COEFFICIENT * tf.reduce_mean(critic_losses) \
                    - entropy_coefficient * tf.reduce_mean(agent.policy_entropy_list)

                agent.training_loss(loss)

            else:
                print(f"{ticker} time out, skip this episode")

        # Compute gradients
        if loss is not None:
            grads = tape.gradient(loss, agent.network.trainable_variables)
            with lock:
                gradient_list.append(grads)

        # Ensure this part is executed regardless of timeout
        barrier.wait()
        agent.old_network.set_weights(agent.network.get_weights())
        if worker_id == 0:
            # Check if the gradient list is not empty
            if len(gradient_list) > 0:
                # Filter out None gradients from the list
                valid_gradient_lists = [grads for grads in gradient_list if all(g is not None for g in grads)]
                if len(valid_gradient_lists) > 0 and TRAINING:
                    for grads in valid_gradient_lists:
                        agent.optimizer.apply_gradients(zip(grads, agent.network.trainable_variables))
                    print("all weight update")

                    # Clear the gradients list after applying
                    gradient_list[:] = []

                    # Save parameters with the lock to ensure thread safety
                    with lock:
                        new_weight[:] = agent.network.get_weights()
                else:
                    print("No weights")
        barrier.wait()


def cancel_orders(trader, ticker):
    # cancel all the remaining orders
    for order in trader.get_waiting_list():
        if (order.symbol == ticker):
            trader.submit_cancellation(order)
            sleep(1)  # the order cancellation needs a little time to go through


def close_positions(trader, ticker):
    # close all positions for given ticker
    print(f"running close positions function for {ticker}")

    item = trader.get_portfolio_item(ticker)

    # close any long positions
    long_shares = item.get_long_shares()
    if long_shares > 0:
        print(f"market selling because {ticker} long shares = {long_shares}")
        order = shift.Order(shift.Order.Type.MARKET_SELL,
                            ticker,
                            int(long_shares / 100))  # we divide by 100 because orders are placed for lots of 100 shares
        trader.submit_order(order)
        sleep(3)  # we sleep to give time for the order to process

    # close any short positions
    short_shares = item.get_short_shares()
    if short_shares > 0:
        print(f"market buying because {ticker} short shares = {short_shares}")
        order = shift.Order(shift.Order.Type.MARKET_BUY,
                            ticker, int(short_shares / 100))
        trader.submit_order(order)
        sleep(3)


def strategy(worker_id, trader: shift.Trader, ticker: str, total_episode, gradient_lock, gradient_list, barrier,
             new_weight):
    env = SHIFT_env(trader, 0.2, 1, 5, ticker, 0.003, 5, 1)
    agent = PPOActorCritic(env)
    initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()

    if os.path.exists(weights_path):
        # print("Loading existing weights.")
        # with lock:
        #     agent.network.load_weights(weights_path)
        #     agent.old_network.set_weights(agent.network.get_weights())
        if worker_id == 0:
            agent.network.load_weights(weights_path)
            new_weight[:] = agent.network.get_weights()
            print(f"new weights loaded: {new_weight}")
            print(f"network weight initialized from trained model")
    else:
        if worker_id == 0:
            agent.network.save_weights(weights_path)
            print(f"file {weights_path} does not exist, weight initialized from scratch")

    barrier.wait()

    run_episodes(worker_id, env, agent, total_episode, ticker, gradient_lock, gradient_list, barrier, new_weight)

    print(
        f"total profits/losses for {ticker}: {trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl}")

    env.kill_thread()


def main(trader, episode):
    initial_pl = trader.get_portfolio_summary().get_total_realized_pl()

    while trader.get_last_trade_time() < trader.get_last_trade_time():
        print("still waiting for market open")
        sleep(1)

    # create tickers
    #tickers = ['MSFT']
    #tickers = ["AAPL", "MSFT", "JNJ", "BA", "IBM"]
    tickers = trader.get_stock_list()
    print(f"tickers: {tickers}")

    # extract ticker name
    # all_tickers = trader.get_stock_list()

    # Exclude "DOW" and "RTX" from the list
    # tickers = [ticker for ticker in all_tickers if ticker not in ["DOW", "RTX"]]

    grad_list = []
    new_weight = []

    PROCESS_NUM = len(tickers)

    ba = Barrier(PROCESS_NUM)
    lo = threading.Lock()

    threads = []
    for i, ticker in zip(range(PROCESS_NUM), tickers):
        worker = Thread(target=strategy, args=(i, trader, ticker, episode, lo, grad_list, ba, new_weight))
        threads.append(worker)
        worker.start()

    for thread in threads:
        thread.join()
    # TODO: when remaining time < end_time, stop placing new order and start to close position using limit order

    # make sure all remaining orders have been cancelled and all positions have been closed
    for ticker in tickers:
        cancel_orders(trader, ticker)
        close_positions(trader, ticker)

        print("END")

        print(f"final bp: {trader.get_portfolio_summary().get_total_bp()}")
        print(
            f"final profits/losses: {trader.get_portfolio_summary().get_total_realized_pl() - initial_pl}")


if __name__ == '__main__':
    with shift.Trader("algoagent") as trader:
        trader.connect("initiator.cfg", "x6QYVYRT")

        sleep(1)
        trader.sub_all_order_book()
        sleep(1)

        episode = 10000000000
        main(trader, episode)
        trader.disconnect()
