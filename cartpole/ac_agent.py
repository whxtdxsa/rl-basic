import torch
import torch.nn.functional as F
import torch.optim as optim
from network import PNet, VNet


class ActorCritic:
    def __init__(self, lr_v, lr_pi):
        self.action_space = (0, 1)

        self.v = VNet()
        self.pi = PNet(len(self.action_space))
        self.optimizer_v = optim.Adam(self.v.parameters(), lr_v)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr_pi)

        self.gamma = 0.9

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
        probs = self.pi(state)[0]
        idx = torch.multinomial(probs, 1)

        return self.action_space[idx], probs[idx]

    def update(self, state, reward, next_state, prob, done):
        state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
        next_state = torch.tensor(next_state, dtype=torch.float32).reshape(1, -1)

        target = reward + (1 - done) * self.gamma * self.v(next_state).detach()

        y = self.v(state)

        loss_v = F.mse_loss(target, y)
        loss_pi = -(target - y.detach()) * torch.log(prob)

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()
