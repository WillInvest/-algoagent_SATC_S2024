from shift_env import SHIFT_env, TRAIN
from shift_agent import *
import os
from time import sleep
import tensorflow as tf
import shift
from threading import Lock, Barrier, Thread
from datetime import datetime

weights_path = 'A2C_network_weights.h5'  # Define a path for the weights
MAX_STEPS_PER_EPISODE = 60
TOTAL_EPISODES = 100000
GAMMA = 0.99  # Discount factor for rewards

widths = {
    "time_step": 2,
    "ticker": 2,
    "account": 20,
    "action": 2,
    "probabilities": 20,
    "reward": 6,
    "total_bp": 10,
    "curr_equity": 10,
    "curr_inv": 6,
    "curr_last_price": 5
}


def strategy(account, ticker, gradient_list, barrier, lock, worker_id):
    trader = shift.Trader(account)
    trader.connect("initiator.cfg", "x6QYVYRT")
    sleep(1)
    trader.sub_all_order_book()
    sleep(1)

    ENTROPY_COEFFICIENT = 0.1 if TRAIN else 0
    env = SHIFT_env(trader, ticker)
    print(f"Shift Environment Initialized: {ticker}")
    agent = ActorCritic(env)
    print(f"Agent Initialized: {ticker}")
    last_trade_time = trader.get_last_trade_time()

    # Synchronize neural network parameters between all workers before learning starts.
    if not os.path.exists(weights_path):
        if worker_id == 0:
            agent.network.save_weights(weights_path)

    barrier.wait()

    agent.network.load_weights(weights_path)

    barrier.wait()  # Another synchronization point

    for episode_cnt in range(TOTAL_EPISODES):
        ENTROPY_COEFFICIENT -= 0.001
        state = agent.reset()

        with tf.GradientTape() as tape:
            for time_step in range(1, MAX_STEPS_PER_EPISODE):
                action, action_prob = agent.policy(state)
                next_state, reward, total_bp, curr_inv, curr_equity, curr_mp = env.step(action)
                action_prob_array = action_prob.numpy()  # Convert EagerTensor to numpy
                action_prob_formatted = " ".join(f"{prob:.4f}" for prob in action_prob_array.flatten())
                print(
                    f"Time Step {time_step:{widths['time_step']}}: {ticker:{widths['ticker']}} | "
                    f"{account:>{widths['account']}} | Action: {action:{widths['action']}} | "
                    f"Prob: {action_prob_formatted:{widths['probabilities']}} | "
                    f"Reward: {reward:>{widths['reward']}.2f} | Total BP: {total_bp:>{widths['total_bp']}.2f} | "
                    f"Inv: {curr_inv:>{widths['curr_inv']}.2f} | "
                    f"Equity: {curr_equity:>{widths['curr_equity']}.2f} | "
                    f"LP {curr_mp:>{widths['curr_last_price']}.2f}"
                )
                agent.reward_history.append(reward)
                state = np.copy(next_state)

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


if __name__ == '__main__':

    if TRAIN:
        accounts = [
            'algoagent',
            'algoagent_test000',
            'algoagent_test001',
            'algoagent_test002',
            'algoagent_test003',
            'algoagent_test004'
        ]

    else:
        accounts = ['algoagent']

    tickers = ['CS1']

    print(f"Ticker: {tickers}")
    PROCESS_NUM = len(tickers)

    ba = Barrier(PROCESS_NUM)
    lo = Lock()
    gradient_list = []
    threads = []
    worker_id = 0

    for account in accounts:
        for i in range(PROCESS_NUM):
            worker = Thread(target=strategy,
                            args=(account,
                                  tickers[i],
                                  gradient_list,
                                  ba, lo,
                                  worker_id))
            threads.append(worker)
            worker.start()
            print(f"Worker {worker_id} starts")
            worker_id += 1

    for thread in threads:
        thread.join()
