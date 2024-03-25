from collections import deque
import random

import numpy as np


class Memory:
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
