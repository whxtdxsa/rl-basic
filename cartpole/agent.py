import copy
import numpy as np
import torch
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from network import QNet, PNet
import torch.optim as optim


class Agent:
    def __init__(self):
        self.action_space = (0, 1)

    def get_action(self, state):
        raise NotImplementedError


class RandomAgent(Agent):
    def get_action(self, state):
        return np.random.choice(self.action_space)

    def update(self, state, action, reward, next_state, done):
        pass


class DQNAgent(Agent):
    def __init__(self, buffer_size, batch_size, learning_rate):
        super().__init__()
        self.buffer = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size

        self.qnet = QNet(len(self.action_space))
        self.qnet_target = copy.deepcopy(self.qnet)
        self.qnet_target.eval()

        self.epsilon = 0.1
        self.gamma = 0.9

        self.optimizer = optim.Adam(self.qnet.parameters(), learning_rate)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qs = self.qnet(state_tensor)
        return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        if len(self.buffer) >= self.batch_size:
            state, action, reward, next_state, done = self.buffer.get_batch()
            state = torch.tensor(state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.int64)
            reward = torch.tensor(reward, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)

            with torch.no_grad():
                next_qs = self.qnet_target(next_state)
            target = reward + self.gamma * (1.0 - done) * next_qs.max(1)[0].reshape(
                -1, 1
            )

            qs = self.qnet(state)
            q = qs.gather(1, action)

            loss = F.mse_loss(target, q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class PGAgent(Agent):
    def __init__(self, buffer_size, batch_size, learning_rate):
        super().__init__()
        self.batch_size = batch_size
        self.pi = PNet(len(self.action_space))
        self.memory = []
        self.optimizer = optim.Adam(self.pi.parameters(), learning_rate)
        self.gamma = 0.9

    def add(self, prob, reward):
        self.memory.append((prob, reward))

    def get_action(self, state):
        state = torch.tensor(state).reshape(1, -1)
        prob = self.pi(state)[0]
        action = torch.multinomial(prob, 1).item()

        return action, prob[action]

    def update(self):
        G = 0
        loss = 0

        for prob, reward in reversed(self.memory):
            G = self.gamma * G + reward

        for prob, reward in self.memory:
            loss -= G * torch.log(prob)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


class REINFORCE(Agent):
    def __init__(self, buffer_size, batch_size, learning_rate):
        super().__init__()
        self.pi = PNet(len(self.action_space))
        self.memory = []
        self.optimizer = optim.Adam(self.pi.parameters(), learning_rate)
        self.gamma = 0.9

    def add(self, prob, reward):
        self.memory.append((prob, reward))

    def get_action(self, state):
        state = torch.tensor(state).reshape(1, -1)
        prob = self.pi(state)[0]
        action = torch.multinomial(prob, 1).item()
        return action, prob[action]

    def update(self):
        G = 0
        loss = 0
        for prob, reward in reversed(self.memory):
            G = reward + self.gamma * G
            loss -= torch.log(prob) * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
