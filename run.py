from shift_mm_env import SHIFT_env
from shift_agent import PPOActorCritic
import os
from time import sleep
import tensorflow as tf
import shift
from threading import Lock, Barrier, Thread
import numpy as np

weights_path = 'A2C_network_weights.h5'  # Define a path for the weights
MAX_STEPS_PER_EPISODE = 60
TOTAL_EPISODES = 100000
GAMMA = 0.99  # Discount factor for rewards
LAMBDA = 0.95
CLIP_RANGE = 0.2
VF_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5
TRAIN = True

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

    env = SHIFT_env(trader, ticker)
    print(f"Shift Environment Initialized: {ticker}")
    agent = PPOActorCritic(env)
    print(f"Agent Initialized: {ticker}")

    # Synchronize neural network parameters between all workers before learning starts.
    if not os.path.exists(weights_path):
        if worker_id == 0:
            agent.network.save_weights(weights_path)

    barrier.wait()

    agent.network.load_weights(weights_path)

    barrier.wait()  # Another synchronization point

    # Get the environment ready
    state = agent.reset()

    for episode_cnt in range(TOTAL_EPISODES):
        env.cancel_all()

        with tf.GradientTape() as tape:
            for time_step in range(1, MAX_STEPS_PER_EPISODE):
                state = np.array(state, dtype=np.float32)  # Convert to float32 for consistency
                state = state.reshape(1, 33)  # Reshape directly to a 2D tensor
                action_mean, state_value = agent.network(state)

                distribution = agent.make_distribution(action_mean)
                raw_action = distribution.sample()
                action = tf.nn.sigmoid(raw_action[0, 0])
                next_state, reward = env.step(action)

                agent.state_value_list.append(state_value)
                agent.reward_list.append(reward)
                agent.policy_entropy_list.append(distribution.entropy())
                agent.calc_action_prob_ratio(raw_action, distribution, state)

                # Compute TD error
                next_state = np.array(next_state, dtype=np.float32)  # Convert to float32 for consistency
                next_state = next_state.reshape(1, 33)  # Reshape directly to a 2D tensor
                _, next_state_value = agent.old_network(next_state)
                td_error = reward + GAMMA * next_state_value - state_value
                agent.td_error_list.append(td_error)

                # Compute Entropy and Probability Ratio
                state = np.copy(next_state)

            # ------------------- Loss ------------------- #
            # Use TD errors to calculate the Generalized Advantage Estimations.
            advantage_list = agent.calc_advantage()
            return_list = agent.calculate_value_target()
            actor_object_list = tf.convert_to_tensor(agent.action_prob_ratio_list, dtype=tf.float32)
            history = zip(agent.state_value_list, actor_object_list, return_list, advantage_list)
            actor_losses = []
            critic_losses = []
            for state_value, actor_object, ret, advantage in history:
                clipped_actor_object = tf.clip_by_value(actor_object, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
                actor_loss = -tf.minimum(actor_object * advantage, clipped_actor_object * advantage)
                actor_losses.append(actor_loss)
                state_value = tf.expand_dims(state_value, 0)
                ret = tf.expand_dims(ret, 0)
                # Use huber loss rather than MSE loss to improve stability.
                critic_loss = agent.huber_loss(ret, state_value)
                critic_losses.append(critic_loss)
            loss = \
                tf.reduce_mean(actor_losses) \
                  + VF_COEFFICIENT * tf.reduce_mean(critic_losses) \
                  - ENTROPY_COEFFICIENT * tf.reduce_mean(
                    agent.policy_entropy_list)
            loss = tf.clip_by_norm(loss, clip_norm=0.5)
            tf.debugging.assert_all_finite(loss, "Loss contains NaN or Inf")
            grads = tape.gradient(loss, agent.network.trainable_variables)
        grads = [(tf.clip_by_norm(grad, clip_norm=0.5)) for grad in grads]

        # ------------------- Record/Apply Gradient  ------------------- #
        # Wait for other workers to arrive.
        barrier.wait()
        # The order of the workers pushing gradients should be random to some extend.
        lock.acquire()
        gradient_list.append(grads)
        lock.release()

        # Save parameters into old_network before updating network.
        agent.old_network.set_weights(agent.network.get_weights())
        print("old network update")
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
            'algoagent_test000',
            'algoagent_test001',
            'algoagent_test002',
            'algoagent_test003',
            'algoagent_test004'
        ]

    else:
        accounts = ['algoagent']

    # tickers = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS',
    #            'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO',
    #            'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG',
    #            'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM']
    tickers = ['CS1']

    print(f"Ticker: {tickers}")
    PROCESS_NUM = len(tickers) * len(accounts)

    ba = Barrier(PROCESS_NUM)
    lo = Lock()
    gradient_list = []
    threads = []
    worker_id = 0

    for account in accounts:
        for i in range(len(tickers)):
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
