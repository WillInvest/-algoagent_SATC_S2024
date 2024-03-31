
import shift
from SHIFT_env import *
from shift_agent import *
from datetime import timedelta
from GradientSharingWorker import gradient_sharing_worker
from shift_agent import PPOActorCritic, GAMMA, LAMBDA, CLIP_RANGE, ENTROPY_COEFFICIENT, VF_COEFFICIENT, LEARNING_RATE
from multiprocessing import Process, Manager, Barrier, Lock

USING_GAE = False



def run_episodes(env, agent, total_episodes, ticker):
    for episode_cnt in range(total_episodes):

        # # Save weights every 100 episodes
        # if (episode_cnt + 1) % 20 == 0:
        #     print("Saving weights at episode", episode_cnt + 1)
        #     agent.network.save_weights(agent.weights_path)
        #     print("Weights saved.")
        state = agent.reset()
        env.isBuy = None
        while env.isBuy is None:
            env.get_signal()

        agent.network.load_weights(agent.weights_path)
        print("Ticker: ", ticker, " load weights")

        with tf.GradientTape() as tape:
            for time_step in range(env.remained_time + 1, -1, -1):
                action, state_value = agent.feed_networks(state)
                next_state, reward, done, _ = env.step(action)

                # if USING_GAE:
                #     # Calculate TD errors (delta).
                #     if not done:
                #         # TODO: test feeding network only once by saving next_action
                #         _, next_state_value = agent.feed_networks(next_state)
                #     else:
                #         next_state_value = 0
                #     td_error = reward + GAMMA * next_state_value - state_value
                #     agent.td_error_list.append(td_error)
                # print("reward_history:", agent.reward_history)
                # print("sum reward:", sum(agent.reward_history))
                agent.reward_list.append(reward)
                state = np.copy(next_state)
                # print("isbuy: ", env.isBuy)

                if done or time_step <= 0:
                    agent.episode_return = sum(agent.reward_list)
                    print("Episode: #", episode_cnt, " Return: ", agent.episode_return)
                    break

            # Use TD errors to calculate generalized advantage estimators.
            if USING_GAE:
                gae = 0
                gae_list = []
                agent.td_error_list.reverse()
                for delta in agent.td_error_list:
                    gae = delta + GAMMA * LAMBDA * gae
                    gae_list.append(gae)
                agent.td_error_list.reverse()
                gae_list.reverse()

                return_list = []
                for advantage, state_value in zip(gae_list, agent.state_value_list):
                    return_list.append(advantage + state_value)
            else:
                # Calculate the discounted return for each step in the episodic history.
                return_list = []
                discounted_sum = 0
                agent.reward_list.reverse()
                for r in agent.reward_list:
                    discounted_sum = r + GAMMA * discounted_sum
                    return_list.append(discounted_sum)
                return_list.reverse()
                agent.reward_list.reverse()

            # # Convert the discounted returns into standard scores.
            # print("return_list: ", return_list)
            # # Calculate mean and standard deviation
            # mean = tf.reduce_mean(return_list)
            # std = tf.math.reduce_std(return_list) + 1e-10
            #
            # # Normalize
            # return_list = (return_list - mean) / std
            # #return_list = return_list.tolist()

            # Calculate the loss.
            actor_object_list = agent.neg_action_prob_ratio_list
            # TODO: Test more into the effects of GAE.
            if USING_GAE:
                history = zip(agent.state_value_list, actor_object_list, return_list, gae_list)
                actor_losses = []
                critic_losses = []

                for state_value, actor_object, ret, gae in history:
                    advantage = gae

                    clipped_actor_object = tf.clip_by_value(actor_object, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
                    actor_loss = tf.maximum(actor_object * advantage, clipped_actor_object * advantage)
                    agent.actor_loss(actor_loss)
                    actor_losses.append(actor_loss)
                    state_value = tf.expand_dims(state_value, 0)
                    ret = tf.expand_dims(ret, 0)
                    # Use huber loss rather than MSE loss to improve stability.
                    critic_loss = agent.huber_loss(state_value, ret)
                    agent.critic_loss(critic_loss)
                    critic_losses.append(critic_loss)
            else:
                history = zip(agent.state_value_list, actor_object_list, return_list)
                actor_losses = []
                critic_losses = []

                for state_value, actor_object, ret in history:
                    # Each of the return minus the estimated state value gives
                    # the estimated advantage of taking the action in the state.
                    advantage = ret - state_value

                    clipped_actor_object = tf.clip_by_value(actor_object, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
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

            #print("actor loss: ", tf.reduce_mean(actor_losses))
            #print("critic loss: ", VF_COEFFICIENT * tf.reduce_mean(critic_losses))
            #print("Entropy gain: ", ENTROPY_COEFFICIENT * tf.reduce_mean(agent.policy_entropy_list))
            #print("total loss: ", loss)
            # print("loss = ", loss)
            agent.training_loss(loss)

        # Compute gradients
        grads = tape.gradient(loss, agent.network.trainable_variables)
        # for grad, var in zip(grads, agent.network.trainable_variables):
        #     tf.print(f"Gradient for {var.name}: {tf.norm(grad)}")

        # Clip gradients by value
        clipped_gradients = [tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0) for grad in grads]
        # for grad, var in zip(clipped_gradients, agent.network.trainable_variables):
        #     tf.print(f"Clipped Gradient for {var.name}: {tf.norm(grad)}")
        agent.old_network.set_weights(agent.network.get_weights())

        # Apply clipped gradients
        agent.optimizer.apply_gradients(zip(clipped_gradients, agent.network.trainable_variables))

        save_weights_lock = Lock()
        with save_weights_lock:
            agent.network.save_weights(agent.weights_path)
            print("Ticker: ", ticker, " save weights")
        print("-----------------------------------\n")

        # print("\n---------------------------------------------\n")
    env.close_positions()
    env.cancel_orders()
    env.getSummary()
    sleep(3)

def cancel_orders(trader, ticker):
    # cancel all the remaining orders
    for order in trader.get_waiting_list():
        if (order.symbol == ticker):
            trader.submit_cancellation(order)
            sleep(1)  # the order cancellation needs a little time to go through


def close_positions(trader, ticker):
    # NOTE: The following orders may not go through if:
    # 1. You do not have enough buying power to close your short postions. Your strategy should be formulated to ensure this does not happen.
    # 2. There is not enough liquidity in the market to close your entire position at once. You can avoid this either by formulating your
    #    strategy to maintain a small position, or by modifying this function to close ur positions in batches of smaller orders.

    # close all positions for given ticker
    print(f"running close positions function for {ticker}")

    item = trader.get_portfolio_item(ticker)

    # close any long positions
    long_shares = item.get_long_shares()
    if long_shares > 0:
        print(f"market selling because {ticker} long shares = {long_shares}")
        order = shift.Order(shift.Order.Type.MARKET_SELL,
                            ticker, int(long_shares/100))  # we divide by 100 because orders are placed for lots of 100 shares
        trader.submit_order(order)
        sleep(1)  # we sleep to give time for the order to process

    # close any short positions
    short_shares = item.get_short_shares()
    if short_shares > 0:
        print(f"market buying because {ticker} short shares = {short_shares}")
        order = shift.Order(shift.Order.Type.MARKET_BUY,
                            ticker, int(short_shares/100))
        trader.submit_order(order)
        sleep(1)


def strategy(trader: shift.Trader, ticker: str, total_episode):
    env = SHIFT_env(trader, 0.1, 1, 5, ticker, 0.003, 5, 1)
    agent = PPOActorCritic(env)
    initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()
    run_episodes(env, agent, total_episode, ticker)

    print(
        f"total profits/losses for {ticker}: {trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl}")

    env.kill_thread()


def main(trader, episode):
        # we track our overall initial profits/losses value to see how our strategy affects it
    initial_pl = trader.get_portfolio_summary().get_total_realized_pl()

    threads = []

    # create tickers
    #tickers = trader.get_stock_list()
    tickers = ["AAPL", "MSFT", "CAT", "CSCO"]

    print(tickers)

    print("START")

    threads = []
    for ticker in tickers:
        # initializes threads containing the strategy for each ticker
        # threads.append(
        #     Thread(target=strategy, args=(trader, ticker, episode)))
        worker = Thread(target=strategy, args=(trader, ticker, episode))
        #worker = Thread(target=gradient_sharing_worker, args=(trader, ticker, ba, lo, grad_list, PROCESS_NUM, episode))
        threads.append(worker)
        worker.start()

        # for thread in threads:
        #     thread.start()
        #     sleep(1)

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
    with shift.Trader("algoagent") as trader:
        trader.connect("initiator.cfg", "x6QYVYRT")

        sleep(1)
        trader.sub_all_order_book()
        sleep(1)

        # env = SHIFT_env(trader, 0.1, 1, 5, 'AAPL', 0.003, 5, 1)
        # agent = PPOActorCritic(env)
        #
        # TOTAL_EPISODES = 10
        # run_episodes(env, agent, TOTAL_EPISODES)


        #env.kill_thread()
        episode = 10

        main(trader, episode)
        trader.disconnect()
