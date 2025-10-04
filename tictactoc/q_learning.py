from collections import defaultdict
import numpy as np
from utils import greedy_probs


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 9

        random_actions = {i: 1 / 9 for i in range(9)}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_space = self.available_actions(state)
        action_probs = greedy_probs(self.Q, state, self.epsilon, action_space)

        probs = np.array([action_probs[a] for a in action_space])
        probs /= probs.sum()

        return np.random.choice(action_space, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_actions = self.available_actions(next_state)
            if not next_actions:
                next_q_max = 0
            else:
                next_qs = [self.Q[next_state, a] for a in next_actions]
                next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        action_space = self.available_actions(state)
        self.pi[state] = greedy_probs(
            self.Q, state, epsilon=0, action_space=action_space
        )

    def available_actions(self, state):
        actions = []
        for i, s in enumerate(state):
            if s == 0:
                actions.append(i)

        return actions


class RandomAgent:
    def get_action(self, state):
        action_space = self.available_actions(state)

        return np.random.choice(action_space)

    def update(self, state, action, reward, next_state, done):
        pass

    def available_actions(self, state):
        actions = []
        for i, s in enumerate(state):
            if s == 0:
                actions.append(i)

        return actions
