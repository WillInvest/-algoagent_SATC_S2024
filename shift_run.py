"""
run.py

Run OpenAI Gym CartPole-v0 with Advantage Actor-Critic.

Zheng Xing <zxing@stevens.edu>

Reference:

"""
from agent import *
from SHIFT_env import SHIFT_env
import shift
from time import sleep

RENDER = True
remained_times = 5
TOTAL_EPISODES = 10


with shift.Trader("algoagent_test004") as trader:
    trader.connect("initiator.cfg", "x6QYVYRT")
    sleep(1)  # Ensure connection is established
    trader.sub_all_order_book()
    sleep(1)

    env = SHIFT_env(trader, 1, 1, 5, 'AAPL', 0.003)
    agent = ActorCritic(env)

    for episode_cnt in range(TOTAL_EPISODES):
        print("New Episode")
        env.set_objective(10, remained_times)
        state = agent.reset()
        episode_return = 0

        with tf.GradientTape() as tape:
            for time_step in range(1, remained_times):
                action = agent.policy(state)
                next_state, reward, done, _ = env.step(action)
                agent.reward_history.append(reward)
                state = np.copy(next_state)

                if done:
                    episode_return = sum(agent.reward_history)
                    print("\nEpisode: #", episode_cnt, " Return: ", episode_return)
                    with agent.summary_writer.as_default():
                        tf.summary.scalar('Episode Return', episode_return, step=episode_cnt)
                        agent.summary_writer.flush()
                    break

            # Calculate the discounted return for each step in the history.
            returns = []
            discounted_sum = 0
            for r in agent.reward_history[::-1]:
                discounted_sum = r + GAMMA * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize the discounted returns.
            returns = np.array(returns)
            eps = 1e-10  # Small epsilon value
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            history = zip(agent.action_prob_history, agent.state_value_history, returns)
            actor_losses = []
            critic_losses = []

            for log_prob, state_value, ret in history:
                error = ret - state_value
                # Use TD-error as the advantage value for each action in the history.
                actor_losses.append(-log_prob * error)
                # Use Huber loss instead of MSE loss for the critic.
                critic_losses.append(agent.huber_loss(tf.expand_dims(state_value, 0), tf.expand_dims(ret, 0)))

            loss = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss, agent.network.trainable_variables)
            agent.optimizer.apply_gradients(zip(grads, agent.network.trainable_variables))

            agent.action_prob_history.clear()
            agent.state_value_history.clear()
            agent.reward_history.clear()

    env.close_positions()
    env.cancel_orders()
    env.kill_thread()
    trader.disconnect()

