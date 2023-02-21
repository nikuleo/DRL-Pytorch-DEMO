import numpy as np
from ACNetwork import ACNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ACAgent():

    def __init__(self, state_size, action_size, seed, fc1_units, lr, betas):
        self.network = ACNetwork(state_size, action_size, seed, fc1_units)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, betas=betas)
        self.rewards = []
        self.logprobs = []
        self.state_values = []

    def act(self, state):
        state_v, action_probs = self.network(state)

        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_v)

        return action.item()

    def leran(self):

        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:  # ::-1 倒序遍历
            dis_reward = reward + 0.99 * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        rewards_amount = len(rewards)
        rewards.resize_(rewards_amount, 1)

        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def clearMemory(self):  # 只清除 list 里的值
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    def save(self, PATH):
        Agent_Dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    def load(self, PATH):
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
