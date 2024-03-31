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
CLIP_RANGE = 0.2
VF_COEFFICIENT = 1.0
ENTROPY_COEFFICIENT = 0.01
LEARNING_RATE = 0.01
STATE_SHAPE = (7,)
ACTIONS = (1,)

# Use microseconds as the last section of the log directory name to have different directories for multiprocess workers.
LOG_DIR = "tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")


class PPOActorCritic:
    def __init__(self, environment, learning_rate=LEARNING_RATE):
        self.env = environment
        self.actions = ACTIONS
        self.network = self._build_network()
        # Clone a network for PPO calculating action probability ratio.
        self.old_network = self._build_network()
        self.old_network.set_weights(self.network.get_weights())
        self.learning_rate = learning_rate

        self.episode_return = 0

        # Lists of Memory/History
        self.state_list = []
        self.state_value_list = []
        self.action_list = []
        self.action_neg_log_prob_list = []
        self.neg_action_prob_ratio_list = []
        self.reward_list = []
        self.td_error_list = []
        self.policy_entropy_list = []

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.huber_loss = tf.keras.losses.Huber()
        self.summary_writer = tf.summary.create_file_writer(LOG_DIR)

        self.critic_loss = tf.keras.metrics.Mean('Critic_Loss', dtype=tf.float32)
        self.actor_loss = tf.keras.metrics.Mean('Actor_Loss', dtype=tf.float32)
        self.training_loss = tf.keras.metrics.Mean('Training_Loss', dtype=tf.float32)

        self.weights_path = 'ppo_network_weights.h5'  # Define a path for the weights

        # Load weights if they exist
        if os.path.exists(self.weights_path):
            print("Loading existing weights.")
            self.network.load_weights(self.weights_path)
            self.old_network.set_weights(self.network.get_weights())

    def _build_network(self):
        # Original state input
        state_input = Input(shape=STATE_SHAPE)
        common = Dense(128, activation='relu')(state_input)

        # New: Add a separate input for spread
        spread_input = Input(shape=(1,))
        # Use `spread_input` to compute dynamic clipping ranges
        dynamic_clip_min = -tf.abs(spread_input)
        dynamic_clip_max = tf.abs(spread_input)

        action_mean = Dense(len(self.actions), activation=None)(common)
        # Apply dynamic clipping
        action_mean = tf.clip_by_value(action_mean, dynamic_clip_min, dynamic_clip_max)

        action_std_pre_activation = Dense(len(self.actions))(common)
        action_std = tf.math.softplus(action_std_pre_activation) + 0.1
        action_std = tf.clip_by_value(action_std, clip_value_min=0.1*spread_input, clip_value_max=1.0*spread_input)
        state_value = Dense(1)(common)

        # Include `spread_input` in the model inputs
        network = tf.keras.Model(inputs=[state_input, spread_input], outputs=[action_mean, action_std, state_value])

        return network

    def reset(self):
        """
        Get the learning agent ready for the next episode. Call this function before the start of every episode.
        :return: The next state returned by the reset function of the environment. This can be used as the first
        state of a new episode.
        """
        state = self.env.reset()

        self.state_list.clear()
        self.action_list.clear()
        self.action_neg_log_prob_list.clear()
        self.neg_action_prob_ratio_list.clear()
        self.state_value_list.clear()
        self.reward_list.clear()
        self.td_error_list.clear()
        self.policy_entropy_list.clear()

        return state

    def feed_networks(self, state):
        """
        This takes in a state observation and outputs an action which is what a policy function does.
        In addition, this returns a estimation of the state value mainly because we combined the actor
        and the critic network.
        :param state:
        :return: selected action, estimated state value
        """
        # Assuming `state` is a numpy array or a list with the spread as the first element
        spread = state[0]
        # Convert state to tensor and process as before
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        # Now pass `spread` alongside state to the network, and modify the network to accept and use it.
        action_mean, action_std, state_value = self.network([state, spread])

        #print("action_mean", action_mean, "action_std", action_std)
        #action_std = tf.constant([1] * len(self.actions))  # Example std dev, could be part of your model

        # Create a distribution for the current policy
        normal_dist = tfp.distributions.Normal(action_mean, action_std)
        sampled_action = normal_dist.sample()
        action_log_prob = normal_dist.log_prob(sampled_action) + 1e-10
        new_action_neg_log_prob = - action_log_prob
        # Store the log probability of the sampled action for the new policy
        self.action_neg_log_prob_list.append(-action_log_prob)
        self.state_value_list.append(state_value)

        old_action_mean, old_action_std, old_state_value = self.old_network([state, spread])

        # Create a distribution for the old policy
        old_normal_dist = tfp.distributions.Normal(old_action_mean, old_action_std)
        old_sampled_action = old_normal_dist.sample()
        old_action_log_prob = normal_dist.log_prob(old_sampled_action) + 1e-10
        old_action_neg_log_prob = - old_action_log_prob

        # e^[-ln(p') + ln(p)] = p/p'
        ratio = tf.math.exp(old_action_neg_log_prob - new_action_neg_log_prob)
        # Add negative to convert gradient descent to ascent.
        self.neg_action_prob_ratio_list.append(- ratio)

        # Store policy entropy
        policy_entropy = - tf.reduce_sum(tf.math.exp(action_log_prob) * action_log_prob)
        self.policy_entropy_list.append(policy_entropy)

        return sampled_action, state_value
