import numpy as np
from collections import defaultdict 

class Disa:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        self.Q = defaultdict(lambda: 0)
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        
        qs = [self.Q[state, action] for action in range(self.action_size)]
        return np.argmax(qs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = 0
        else:
            next_qs = [self.Q[next_state, next_action] for next_action in range(self.action_size)]
            next_q = max(next_qs)
        
        target = reward + self.gamma * next_q

        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
    
