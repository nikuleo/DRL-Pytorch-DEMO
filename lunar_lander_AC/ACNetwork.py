import torch
import torch.nn as nn
import torch.nn.functional as F


class ACNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=128):
        super(ACNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.action_layer = nn.Linear(fc1_units, action_size)
        self.value_layer = nn.Linear(fc1_units, 1)

    def forward(self, state):
        state = torch.from_numpy(state).float()
        x = F.relu(self.fc1(state))

        state_value = self.value_layer(x)
        action_probs = F.softmax(self.action_layer(x), dim=-1)

        return state_value, action_probs
