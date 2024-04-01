from SHIFT_env import *
from shift_agent import *
from shift_agent import PPOActorCritic, GAMMA, LAMBDA, CLIP_RANGE, ENTROPY_COEFFICIENT, VF_COEFFICIENT, LEARNING_RATE
from threading import Lock
from multiprocessing import Manager, Barrier

learning = True


def run_episodes(env, agent, total_episodes, ticker, lock, gradient_list, barrier):
    for episode_cnt in range(total_episodes):

        timeout = False
        current_time = datetime.datetime.now()
        env.isBuy = None
        state = agent.reset()

        while env.isBuy is None:
            env.get_signal()
            if (datetime.datetime.now() - current_time).total_seconds() > 1:
                timeout = True
                env.set_objective(0, 5)
                break

        with tf.GradientTape() as tape:
            if timeout:
                print(f"Timed out waiting for {ticker}, skipping this episode/thread.")
            else:
                lock.acquire()
                agent.network.load_weights(agent.weights_path)
                # print("Ticker: ", ticker, " load weights")
                lock.release()

                for time_step in range(env.remained_time + 1, -1, -1):
                    action, state_value = agent.feed_networks(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.reward_list.append(reward)
                    state = np.copy(next_state)

                    if done or time_step <= 0:
                        agent.episode_return = sum(agent.reward_list)
                        print("Episode: #", episode_cnt, "ticker:", ticker, "Return: ", agent.episode_return)
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
                actor_object_list = agent.neg_action_prob_ratio_list
                # TODO: Test more into the effects of GAE.

                history = zip(agent.state_value_list, actor_object_list, return_list)
                actor_losses = []
                critic_losses = []

                for state_value, actor_object, ret in history:
                    # Each of the return minus the estimated state value gives
                    # the estimated advantage of taking the action in the state.
                    advantage = ret - state_value

                    clipped_actor_object = tf.clip_by_value(actor_object, -1.0 - CLIP_RANGE, -1.0 + CLIP_RANGE)
                    actor_loss = tf.maximum(actor_object * advantage, clipped_actor_object * advantage)
                    agent.actor_loss(actor_loss)
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
                    - ENTROPY_COEFFICIENT * tf.reduce_mean(agent.policy_entropy_list)

                agent.training_loss(loss)
                # Compute gradients
                clipped_gradients = tape.gradient(loss, agent.network.trainable_variables)
                # Clip gradients by value
                #clipped_gradients = [tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0) for grad in grads]

            # Ensure this part is executed regardless of timeout
            barrier.wait()
            print("All tickers finish episode {}".format(episode_cnt), " ready to append weights")

            lock.acquire()
            if not timeout:
                gradient_list.append(clipped_gradients)
            lock.release()
            barrier.wait()

            # Save parameters into old_network before updating network.
            agent.old_network.set_weights(agent.network.get_weights())
            # Apply all gradients shared by other workers.
            # Assuming gradient_list is a list of gradients from all threads
            if ticker == "AAPL":
                # Check if the gradient list is not empty
                if len(gradient_list) > 0:
                    # Filter out None gradients from the list
                    valid_gradient_lists = [grads for grads in gradient_list if all(g is not None for g in grads)]

                    # Proceed only if there are valid gradients
                    if len(valid_gradient_lists) > 0:
                        # Initialize a list to store the sum of gradients for each variable
                        sum_grads = [tf.zeros_like(var, dtype=tf.float32) for var in agent.network.trainable_variables]

                        # Sum up all valid gradients for each variable
                        for grads in valid_gradient_lists:
                            for i, grad in enumerate(grads):
                                sum_grads[i] += grad

                        # Compute the mean of gradients
                        mean_grads = [sum_grad / len(valid_gradient_lists) for sum_grad in sum_grads]

                        # Apply the mean gradients
                        agent.optimizer.apply_gradients(zip(mean_grads, agent.network.trainable_variables))

                        # Clear the gradients list after applying
                        gradient_list[:] = []

                        # Save parameters with the lock to ensure thread safety
                        lock.acquire()
                        agent.network.save_weights(agent.weights_path)
                        print("Shared weight updated.")
                        lock.release()
                    else:
                        print("No valid gradients to apply.")

            # if ticker == "AAPL":
            #     for grads in gradient_list:
            #         agent.optimizer.apply_gradients(zip(grads, agent.network.trainable_variables))
            #         gradient_list[:] = []
            #         # Save parameters.
            #         lock.acquire()
            #         agent.network.save_weights(agent.weights_path)
            #         print("Shared weight updated.")
            #         lock.release()
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
        sleep(1)  # we sleep to give time for the order to process

    # close any short positions
    short_shares = item.get_short_shares()
    if short_shares > 0:
        print(f"market buying because {ticker} short shares = {short_shares}")
        order = shift.Order(shift.Order.Type.MARKET_BUY,
                            ticker, int(short_shares / 100))
        trader.submit_order(order)
        sleep(1)


def strategy(trader: shift.Trader, ticker: str, total_episode, gradient_lock, gradient_list, barrier):
    env = SHIFT_env(trader, 0.5, 1, 5, ticker, 0.003, 5, 1)
    agent = PPOActorCritic(env)
    initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()
    run_episodes(env, agent, total_episode, ticker, gradient_lock, gradient_list, barrier)

    print(
        f"total profits/losses for {ticker}: {trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl}")

    env.kill_thread()


def main(trader, episode):
    initial_pl = trader.get_portfolio_summary().get_total_realized_pl()

    # create tickers
    #tickers = ['AAPL']
    #tickers = ["AAPL", "MSFT", "JNJ", "BA", "IBM"]
    tickers = trader.get_stock_list()

    manager = Manager()
    gradient_list = manager.list()
    gradient_lock = Lock()
    ba = Barrier(len(tickers))
    PROCESS_NUM = len(tickers)

    print(tickers)

    print("START")

    threads = []
    for ticker in tickers:
        worker = Thread(target=strategy, args=(trader, ticker, episode, gradient_lock, gradient_list, ba))
        print(f"{ticker} thread created")
        threads.append(worker)
        worker.start()

        # TODO: when remaining time < end_time, stop placing new order and start to close position using limit order

    # # wait for all threads to finish
    for thread in threads:
        # NOTE: this method can stall your program indefinitely if your strategy does not terminate naturally
        # setting the timeout argument for join() can prevent this
        thread.join()

    # make sure all remaining orders have been cancelled and all positions have been closed
    for ticker in tickers:
        cancel_orders(trader, ticker)
        close_positions(trader, ticker)

        print("END")

        print(f"final bp: {trader.get_portfolio_summary().get_total_bp()}")
        print(
            f"final profits/losses: {trader.get_portfolio_summary().get_total_realized_pl() - initial_pl}")


if __name__ == '__main__':
    with shift.Trader("algoagent_test004") as trader:
        trader.connect("initiator.cfg", "x6QYVYRT")

        sleep(1)
        trader.sub_all_order_book()
        sleep(1)

        episode = 20000

        main(trader, episode)
        trader.disconnect()
