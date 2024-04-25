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

GAMMA = 0.99
LEARNING_RATE = 0.0001
STATE_SHAPE = (32,)
ACTIONS = [-1, 0, 1]

LOG_DIR = "tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class ActorCritic:
    def __init__(self, environment):
        self.env = environment
        self.actions = ACTIONS
        self.network = self._build_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.huber_loss = tf.keras.losses.Huber()
        self.action_prob_history = []
        self.state_value_history = []
        self.reward_history = []
        self.entropy_history = []

    def _build_network(self):
        network_input = Input(shape=STATE_SHAPE)
        common = Dense(128, activation='relu')(network_input)
        action_prob = Dense(len(self.actions), activation='softmax')(common)
        state_value = Dense(1)(common)

        network = tf.keras.Model(inputs=network_input, outputs=[action_prob, state_value])

        return network

    def reset(self):
        state = self.env.reset()
        return state

    def policy(self, state):
        """
        :param state:
        :return:
        """
        state = tf.convert_to_tensor(state)
        # Add the first dimension of the batch size.
        state = tf.expand_dims(state, 0)

        # Feed the neural network with one state and get one set of action probabilities
        # and one state value estimation.
        action_prob, state_value = self.network(state)

        # Check if action_prob contains NaN values using tf.reduce_any and tf.math.is_nan.
        if tf.reduce_any(tf.math.is_nan(action_prob)):
            print("NaN detected in action probabilities, resetting the network.")
            self.network = self._build_network()  # Reinitialize the network
            action_prob, state_value = self.network(state)  # Re-run the policy with the new network

        # Select action based on the processed probabilities
        action = np.random.choice(np.array(self.actions), p=np.squeeze(action_prob))
        # Calculate and store the logarithm probability of the selected action.
        log_prob = tf.math.log(action_prob[0][action])
        action_log_prob = tf.math.log(action_prob)
        policy_entropy = - tf.reduce_sum(action_prob * action_log_prob)
        self.entropy_history.append(policy_entropy)
        self.action_prob_history.append(log_prob)
        self.state_value_history.append(state_value)

        return action, action_prob

