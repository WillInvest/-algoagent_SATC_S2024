"""
Agent.py

Actor-Critic using TD-error of state value of the critic as the performance score of the actor.

Zheng Xing <zxing@stevens.edu>

Reference:

TODO: Test separating Actor and Critic networks.

"""

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import tensorflow_probability as tfp  # Import TensorFlow Probability


GAMMA = 0.99
LEARNING_RATE = 0.01
ACTOR_LEARNING_RATE = 0.01
CRITIC_LEARNING_RATE = 0.01
STATE_SHAPE = (7,)
ACTION_DIM = 1

LOG_DIR = "tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class ActorCritic:
    def __init__(self, environment):
        self.env = environment
        self.actions = ACTION_DIM
        self.network = self._build_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.huber_loss = tf.keras.losses.Huber()
        self.summary_writer = tf.summary.create_file_writer(LOG_DIR)
        self.action_prob_history = []
        self.state_value_history = []
        self.reward_history = []

    def _build_network(self):
        network_input = Input(shape=STATE_SHAPE)
        common = Dense(128, activation='relu')(network_input)

        # Output layer for action mean - adjust units according to the dimensionality of the action space
        action_mean = Dense(ACTION_DIM, activation='linear')(common)  # Use linear activation for the mean

        # Output layer for action log standard deviation
        # Using log variance as output for numerical stability
        action_log_std = Dense(ACTION_DIM, activation='linear')(common)  # Use linear or another appropriate activation

        # Output layer for state value estimation
        state_value = Dense(1)(common)

        network = tf.keras.Model(inputs=network_input, outputs=[action_mean, action_log_std, state_value])

        return network

    def reset(self):
        state = self.env.reset()
        return state

    def policy(self, state):
        """
        Generate a continuous action from the policy network, constrained by the spread.

        :param state: The current state of the environment, with the first element being the spread.
        :return: The selected action (premium) and the log probability of that action.
        """
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)  # Convert state to tensor

        # Extract the spread from the state; assuming it's the first element
        spread = state[0]
        # Define the premium range as ±3 times the spread
        premium_range = 3 * spread

        # Get action distribution parameters and state value from the network
        action_mean, action_log_std, state_value = self.network(state_tensor)
        action_std = tf.exp(action_log_std)  # Convert log std to std

        # Create the action distribution
        action_distribution = tfp.distributions.Normal(action_mean, action_std)
        # Sample an action from the distribution
        action = action_distribution.sample()

        # Clamp the action to be within the defined premium range based on the spread
        action = tf.clip_by_value(action, -premium_range, premium_range)

        # Calculate and store the log probability of the sampled action
        log_prob = action_distribution.log_prob(action)
        self.action_prob_history.append(log_prob)
        self.state_value_history.append(state_value)

        # Squeeze the action to ensure it has the correct dimensions and convert to numpy
        return tf.squeeze(action, axis=-1).numpy()

