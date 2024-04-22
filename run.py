from shift_env import SHIFT_env
from shift_agent import *
import os
from time import sleep
import tensorflow as tf
import shift
from threading import Lock, Barrier, Thread
from datetime import datetime

TRAIN = True  # Set to True to enable training
weights_path = 'A2C_network_weights.h5'  # Define a path for the weights
MAX_STEPS_PER_EPISODE = 10
TOTAL_EPISODES = 100000
GAMMA = 0.99  # Discount factor for rewards


def strategy(trader, ticker, gradient_list, barrier, lock, worker_id):
    ENTROPY_COEFFICIENT = 1 if TRAIN else 0
    env = SHIFT_env(trader, ticker)
    print(f"Shift Environment Initialized: {ticker}")
    agent = ActorCritic(env)
    print(f"Agent Initialized: {ticker}")

    # Synchronize neural network parameters between all workers before learning starts.
    if not os.path.exists(weights_path):
        if worker_id == 0:
            agent.network.save_weights(weights_path)
    barrier.wait()

    agent.network.load_weights(weights_path)

    barrier.wait()

    time_interval = 15
    sleep(time_interval)
    for episode_cnt in range(TOTAL_EPISODES):
        ENTROPY_COEFFICIENT -= 0.001
        state = agent.reset()
        print(f"state: {state}")
        print(f"episode: {episode_cnt} | state reset")

        with tf.GradientTape() as tape:
            for time_step in range(1, MAX_STEPS_PER_EPISODE):
                print(f"--------------------------------------\n"
                      f"time step {time_step} : "
                      f"--------------------------------------\n")
                action = agent.policy(state)
                next_state, reward = env.step(action)
                print(f"action: {action} | reward: {reward}")
                agent.reward_history.append(reward)
                state = np.copy(next_state)

            last_trade_time = trader.get_last_trade_time().time()
            if TRAIN and last_trade_time < env.endTime:
                # Calculate the discounted return for each step in the history.
                returns = []
                discounted_sum = 0
                for r in agent.reward_history[::-1]:
                    discounted_sum = r + GAMMA * discounted_sum
                    returns.insert(0, discounted_sum)

                history = zip(agent.action_prob_history, agent.state_value_history, returns)
                actor_losses = []
                critic_losses = []
                policy_entropy = agent.entropy_history

                for log_prob, state_value, ret in history:
                    error = ret - state_value
                    # Use TD-error as the advantage value for each action in the history.
                    actor_losses.append(-log_prob * error)
                    # Use Huber loss instead of MSE loss for the critic.
                    critic_losses.append(agent.huber_loss(tf.expand_dims(state_value, 0), tf.expand_dims(ret, 0)))

                ENTROPY_COEFFICIENT = tf.maximum(ENTROPY_COEFFICIENT, 0.001)
                loss = tf.reduce_sum(actor_losses) + \
                       tf.reduce_sum(critic_losses) - \
                       tf.reduce_sum(policy_entropy) * ENTROPY_COEFFICIENT

                grads = tape.gradient(loss, agent.network.trainable_variables)
                agent.action_prob_history.clear()
                agent.state_value_history.clear()
                agent.reward_history.clear()

                # Every a few episodes, all workers push their gradients into the shared list of gradients.
                # After the last worker finishes pushing the gradients, all workers pull and apply the gradients.
                period = 1
                if episode_cnt % period == 0:
                    # Wait for other workers to arrive.
                    # Barrier.wait() will release all processes together.
                    barrier.wait()

                    # The order of the workers pushing gradients should be random to some extend.
                    lock.acquire()
                    gradient_list.append(grads)
                    lock.release()

                    # Wait for other workers to finish pushing gradients.
                    barrier.wait()
                    # Apply all gradients shared by other workers.
                    for grads in gradient_list:
                        agent.optimizer.apply_gradients(zip(grads, agent.network.trainable_variables))

                    # Wait for all workers to finish applying gradients.
                    # The first worker clears the shared gradient list.
                    barrier.wait()
                    if worker_id == 0:
                        gradient_list[:] = []
                        # Save parameters.
                        agent.network.save_weights(weights_path)
                print("Episode {}: finished training".format(episode_cnt))
            else:
                print(f"TRAIN: {TRAIN} | last_trade_time: {last_trade_time}")


if __name__ == '__main__':
    with shift.Trader("algoagent_test001") as trader:
        trader.connect("initiator.cfg", "x6QYVYRT")
        sleep(1)
        trader.sub_all_order_book()
        sleep(1)

        # all_tickers = trader.get_stock_list()
        tickers = ['CS1'] if TRAIN else ['CS1', 'CS2']
        print(f"Ticker: {tickers}")
        PROCESS_NUM = len(tickers)

        ba = Barrier(PROCESS_NUM)
        lo = Lock()
        gradient_list = []
        threads = []

        for i in range(PROCESS_NUM):
            worker = Thread(target=strategy,
                            args=(trader,
                                  tickers[i],
                                  gradient_list,
                                  ba, lo, i))
            threads.append(worker)
            worker.start()

        for thread in threads:
            thread.join()
