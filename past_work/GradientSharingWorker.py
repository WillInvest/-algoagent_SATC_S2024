
import shift
from SHIFT_env import *
from shift_agent import *
from datetime import timedelta


USING_GAE = False
# Trained model checkpoint.
# CHECKPOINT_DIR = "checkpoints"
# CHECKPOINT_PATH = CHECKPOINT_DIR + "/model.ckpt"


def gradient_sharing_worker(trader, ticker, barrier, lock, gradient_list, process_number, total_episodes):
    """
    Without experience replay, each worker interacts with their own environments and calculates their gradients using
    the experiences of their past episode. The first worker collects gradients shared by all workers and applies the
    gradients on its own network. All workers synchronize network parameters with the first worker between each episode.

    If experience replay is needed, use the experience_sharing_worker().

    :param ticker:
    :param barrier:
    :param lock:
    :param gradient_list:
    :param process_number:
    :return:
    """
    import tensorflow as tf
    from shift_agent import PPOActorCritic, GAMMA, LAMBDA, CLIP_RANGE, ENTROPY_COEFFICIENT, VF_COEFFICIENT, LEARNING_RATE

    env = SHIFT_env(trader, 0.1, 1, 5, ticker, 0.003, 5, 1)

    agent = PPOActorCritic(env, learning_rate=LEARNING_RATE/process_number)

    # Synchronize neural network parameters between all workers before learning starts.
    if ticker == "AAPL":
        print("start to save weight")
        agent.network.save_weights(agent.weights_path)
        print("save weight successfully")

    barrier.wait()
    if ticker != "AAPL":
        agent.network.load_weights(agent.weights_path)

    for episode_cnt in range(total_episodes):
        print("Episode {}".format(episode_cnt))

        # # Save weights every 100 episodes
        # if (episode_cnt + 1) % 20 == 0:
        #     print("Saving weights at episode", episode_cnt + 1)
        #     agent.network.save_weights(agent.weights_path)
        #     print("Weights saved.")

        state = agent.reset()
        env.isBuy = None
        while env.isBuy is None:
            env.get_signal()
        print("signal: ", env.isBuy)
        with tf.GradientTape() as tape:
            for time_step in range(env.remained_time + 1, -1, -1):
                action, state_value = agent.feed_networks(state)
                next_state, reward, done, _ = env.step(action)
                print("next_state:", next_state)

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

            # print("actor loss: ", tf.reduce_mean(actor_losses))
            # print("critic loss: ", VF_COEFFICIENT * tf.reduce_mean(critic_losses))
            # print("Entropy gain: ", ENTROPY_COEFFICIENT * tf.reduce_mean(agent.policy_entropy_list))
            # print("total loss: ", loss)
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

        period = 1
        if episode_cnt % period == 0:
            # Wait for other workers to arrive.
            # Barrier.wait() will release all processes together.
            barrier.wait()

            # The order of the workers pushing gradients should be random to some extend.
            lock.acquire()
            gradient_list.append(clipped_gradients)
            lock.release()

            # Wait for other workers to finish pushing gradients.
            barrier.wait()
            # Save parameters into old_network before updating network.
            agent.old_network.set_weights(agent.network.get_weights())
            # Apply all gradients shared by other workers.
            for grads in gradient_list:
                agent.optimizer.apply_gradients(zip(grads, agent.network.trainable_variables))

            # Wait for all workers to finish applying gradients.
            # The first worker clears the shared gradient list.
            barrier.wait()
            if ticker == "AAPL":
                gradient_list[:] = []
                # Save parameters.
                agent.network.save_weights(agent.weights_path)
                print("weights saved")

        # # Apply clipped gradients
        # agent.optimizer.apply_gradients(zip(clipped_gradients, agent.network.trainable_variables))

        # print("\n---------------------------------------------\n")
    # env.close_positions()
    # env.cancel_orders()
    # env.getSummary()
    # sleep(3)

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

