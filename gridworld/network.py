import torch.nn as nn

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 100)
        self.l2 = nn.Linear(100, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return x

