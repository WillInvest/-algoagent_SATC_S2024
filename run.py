import shift
from SHIFT_env import *
from shift_agent import *


USING_GAE = False


def run_episodes(env, agent, total_episodes):
    for episode_cnt in range(total_episodes):

        # Save weights every 100 episodes
        if (episode_cnt + 1) % 5 == 0:
            print("Saving weights at episode", episode_cnt + 1)
            agent.network.save_weights(agent.weights_path)
            print("Weights saved.")

        state = agent.reset()
        env.isBuy = None
        while env.isBuy is None:
            env.get_signal()
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
                    # with agent.summary_writer.as_default():
                    #     tf.summary.scalar('Episode Return', agent.episode_return, step=episode_cnt)
                    #     tf.summary.scalar('Actor Loss', agent.actor_loss.result(), step=episode_cnt)
                    #     tf.summary.scalar('Critic Loss', agent.critic_loss.result(), step=episode_cnt)
                    #     tf.summary.scalar('PPO Loss', agent.training_loss.result(), step=episode_cnt)
                    #     agent.actor_loss.reset_states()
                    #     agent.critic_loss.reset_states()
                    #     agent.training_loss.reset_states()
                    #     agent.summary_writer.flush()
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

            # loss = \
            #     sum(actor_losses) \
            #     + VF_COEFFICIENT * sum(critic_losses) \
            #     - ENTROPY_COEFFICIENT * sum(agent.policy_entropy_list)
            loss = \
                tf.reduce_mean(actor_losses) \
                + VF_COEFFICIENT * tf.reduce_mean(critic_losses) \
                - ENTROPY_COEFFICIENT * tf.reduce_mean(agent.policy_entropy_list)
            # print("loss = ", loss)
            agent.training_loss(loss)

        # Calculate the gradients but not apply gradients.
        grads = tape.gradient(loss, agent.network.trainable_variables)

        agent.optimizer.apply_gradients(zip(grads, agent.network.trainable_variables))

            # agent.action_prob_history.clear()
            # agent.state_value_history.clear()
            # agent.reward_history.clear()
        # print("\n---------------------------------------------\n")
    env.close_positions()
    env.cancel_orders()
    env.getSummary()
    sleep(3)


if __name__ == '__main__':
    with shift.Trader("algoagent_test004") as trader:
        trader.connect("initiator.cfg", "x6QYVYRT")
        sleep(1)
        trader.sub_all_order_book()
        sleep(1)

        env = SHIFT_env(trader, 0.1, 1, 5, 'AAPL', 0.003, 5, 1)
        agent = PPOActorCritic(env)

        TOTAL_EPISODES = 10000
        run_episodes(env, agent, TOTAL_EPISODES)

        env.kill_thread()
        trader.disconnect()
