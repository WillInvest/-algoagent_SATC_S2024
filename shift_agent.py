import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 0.001
STATE_SHAPE = (33,)
HIDDEN_SIZE = [64, 64]

LOG_DIR = "tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class PPOActorCritic:
    def __init__(self, env):
        self.env = env
        self.state_shape = STATE_SHAPE
        self.hidden_sizes = HIDDEN_SIZE
        self.network = self._build_network()
        self.old_network = self._build_network()
        self.old_network.set_weights(self.network.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.huber_loss = tf.keras.losses.Huber()

        # Use lists to store local memory of experiences.
        self.action_list = []
        self.reward_list = []
        self.action_prob_ratio_list = []
        self.state_value_list = []
        self.value_target_list = []
        self.td_error_list = []
        self.policy_entropy_list = []

    def reset(self):

        state = self.env.reset()
        self.action_list.clear()
        self.reward_list.clear()
        self.action_prob_ratio_list.clear()
        self.state_value_list.clear()
        self.value_target_list.clear()
        self.td_error_list.clear()
        self.policy_entropy_list.clear()

        return state

    def _build_network(self):
        state_input = Input(shape=self.state_shape)

        # Actor Network Layers
        layer1 = Dense(self.hidden_sizes[0], activation='relu')(state_input)
        layer2 = Dense(self.hidden_sizes[1], activation='relu')(layer1)
        action = Dense(1)(layer2)

        state_value = Dense(1)(layer2)

        network = Model(inputs=state_input, outputs=[action, state_value])
        return network

    @staticmethod
    def make_distribution(actions):
        """
        Creates an independent normal distribution for each action dimension.
        """
        means = actions
        std_devs = tf.ones_like(means)
        distribution = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=means, scale=std_devs),
            reinterpreted_batch_ndims=1,
        )
        return distribution

    def calculate_value_target(self):
        """
        Take the list of rewards in the memory and calculate the discounted return of each step.
        :return:
        """
        # Calculate the discounted return for each step in the episodic history.
        discounted_return_list = []
        discounted_sum = 0
        self.reward_list.reverse()
        for r in self.reward_list:
            discounted_sum = r + GAMMA * discounted_sum
            discounted_return_list.append(discounted_sum)
        discounted_return_list.reverse()
        self.reward_list.reverse()

        # Convert the returns into standard scores.
        # TODO: Test the effects of this conversion.
        value_target_tensor = tf.convert_to_tensor(discounted_return_list)
        mean = tf.reduce_mean(value_target_tensor)
        std = tf.math.reduce_std(value_target_tensor)
        if std != 0:
            value_target_tensor = (value_target_tensor - mean) / std

        return value_target_tensor

    def calc_action_prob_ratio(self, raw_action, distribution, state):
        old_action, _ = self.old_network(state)
        # print(f"old1: {old1}, old2: {old2}, old3: {old3}, old_action_mean: {old_action_mean}")
        old_distribution = self.make_distribution(old_action)
        old_log_prob = old_distribution.log_prob(raw_action)
        new_log_prob = distribution.log_prob(raw_action)
        prob_ratio = tf.math.exp(new_log_prob - old_log_prob)
        prob_ratio = tf.clip_by_value(prob_ratio, clip_value_min=0.01, clip_value_max=100)
        # print(f"old_log_prob: {old_log_prob}, new_log_prob: {new_log_prob}, prob_ratio: {prob_ratio}")
        self.action_prob_ratio_list.append(prob_ratio)

    def calc_advantage(self):
        gae = 0
        gae_list = []
        self.td_error_list.reverse()
        for delta in self.td_error_list:
            gae = delta + GAMMA * LAMBDA * gae
            gae_list.append(gae)
        self.td_error_list.reverse()
        gae_list.reverse()

        # Convert to TensorFlow tensor
        advantage_tensor = tf.convert_to_tensor(gae_list, dtype=tf.float32)

        mean = tf.reduce_mean(advantage_tensor)
        std = tf.math.reduce_std(advantage_tensor)

        if std != 0:
            advantage_tensor = (advantage_tensor - mean) / std

        return advantage_tensor

    def test_network(self):
        model = self._build_network()

        state = np.random.rand(1, 37).astype(np.float32)

        actions, state_value = model(state)

        print(f"Actions: {actions}")
        print(f"State Value: {state_value}")


