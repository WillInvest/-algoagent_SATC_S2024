"""
Agent.py

Hao Fu <hfu11@stevens.edu>

Reference:

Zheng Xing <zxing@stevens.edu>

"""

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import tensorflow_probability as tfp
import os

USING_GAE = False

GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 0.01
STATE_SHAPE = (7,)
PREMIUMS = [-3, -2, -1, 1, 2, 3]
NUM_ACTIONS = len(PREMIUMS)
decay_rate = 0.995

ACTIONS = (10,)

# Use microseconds as the last section of the log directory name to have different directories for multiprocess workers.
LOG_DIR = "tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")


class PPOActorCritic:
    def __init__(self, environment, learning_rate=LEARNING_RATE):
        self.env = environment
        self.actions = PREMIUMS
        self.network = self._build_network()
        # Clone a network for PPO calculating action probability ratio.
        self.old_network = self._build_network()
        self.old_network.set_weights(self.network.get_weights())
        self.learning_rate = learning_rate

        self.episode_return = 0

        # Lists of Memory/History
        self.state_list = []
        self.state_value_list = []
        self.old_action_prob_list = []
        self.action_log_prob_list = []
        self.prob_ratio_list = []
        self.reward_list = []
        self.td_error_list = []
        self.policy_entropy_list = []

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.huber_loss = tf.keras.losses.Huber()
        self.summary_writer = tf.summary.create_file_writer(LOG_DIR)

        self.critic_loss = tf.keras.metrics.Mean('Critic_Loss', dtype=tf.float32)
        self.actor_loss = tf.keras.metrics.Mean('Actor_Loss', dtype=tf.float32)
        self.training_loss = tf.keras.metrics.Mean('Training_Loss', dtype=tf.float32)

    def _build_network(self):
        network_input = Input(shape=STATE_SHAPE)
        common = Dense(128, activation='relu')(network_input)
        action_prob = Dense(len(self.actions), activation='softmax')(common)
        state_value = Dense(1)(common)

        network = tf.keras.Model(inputs=network_input, outputs=[action_prob, state_value])

        return network

    def reset(self):
        """
        Get the learning agent ready for the next episode. Call this function before the start of every episode.
        :return: The next state returned by the reset function of the environment. This can be used as the first
        state of a new episode.
        """
        state = self.env.reset()

        self.state_list.clear()
        self.old_action_prob_list.clear()
        self.action_log_prob_list.clear()
        self.prob_ratio_list.clear()
        self.state_value_list.clear()
        self.reward_list.clear()
        self.td_error_list.clear()
        self.policy_entropy_list.clear()

        return state

    def feed_networks(self, state):
        state = tf.convert_to_tensor(state)
        # Add the first dimension of the batch size.
        state = tf.expand_dims(state, 0)

        action_prob_array, state_value = self.network(state)
        action_prob_array = action_prob_array + 1e-10
        selected_action_index = np.random.choice(len(self.actions), p=np.squeeze(action_prob_array))
        action = self.actions[selected_action_index]
        action_prob = action_prob_array[0][selected_action_index]
        action_log_prob = tf.math.log(action_prob)

        old_action_prob_array, old_state_value = self.old_network(state)
        old_action_prob_array = old_action_prob_array + 1e-10
        old_action_prob = old_action_prob_array[0][selected_action_index]
        old_action_log_prob = tf.math.log(old_action_prob)

        ratio = tf.math.exp(action_log_prob - old_action_log_prob)

        # Calculate policy entropy for the action distribution
        action_log_prob_array = tf.math.log(action_prob_array)
        policy_entropy = - tf.reduce_sum(action_prob_array * action_log_prob_array)

        # Storing necessary values for later loss calculation
        self.prob_ratio_list.append(ratio)
        self.old_action_prob_list.append(old_action_prob)
        self.policy_entropy_list.append(policy_entropy)
        self.state_value_list.append(state_value)

        return action, state_value.numpy()
