import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from network import QNet 

class Disa_Q:
    def __init__(self):
        self.action_size = 4
        self.epsilon = 0.1
        self.gamma = 0.9
        self.lr = 0.01


        self.Q = QNet()
        self.optimizer = optim.SGD(self.Q.parameters(), self.lr)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        qs = self.Q(state)
        return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = np.zeros(1)
        else:
            next_qs = self.Q(next_state)
            next_q_max = next_qs.max(axis=1)
            next_q_max.unchain()
    
        target = reward + self.gamma * next_q_max

        qs = self.Q(state)
        q = qs[:, action]

        loss = F.mse_loss(target, q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.data
        


        
