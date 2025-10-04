from collections import deque
from typing import SupportsFloat
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(
        self,
        state: tuple,
        action: int,
        reward: SupportsFloat,
        next_state: tuple,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def get_batch(self):
        indices = np.random.choice(len(self), size=self.batch_size, replace=False)

        state = np.stack([self.buffer[i][0] for i in indices])
        action = np.stack([self.buffer[i][1] for i in indices]).reshape(-1, 1)
        reward = np.stack([self.buffer[i][2] for i in indices]).reshape(-1, 1)
        next_state = np.stack([self.buffer[i][3] for i in indices])
        done = np.stack([self.buffer[i][4] for i in indices]).reshape(-1, 1)

        return state, action, reward, next_state, done
