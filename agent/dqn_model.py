# dqn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNet(nn.Module):
    """
    Small network for DQN. Input: observation.
    Output: Q-values for each action.
    """

    def __init__(self, n_obs, n_actions):
        super(DQNNet, self).__init__()
        self.layer1 = nn.Linear(n_obs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.out(x)
