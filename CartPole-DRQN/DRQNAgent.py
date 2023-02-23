from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import numpy as np
import math

# Hyperparemeters
CAPACITY = 10000
GAMMA = 0.9
LEARNING_RATE = 1e-3
HIDDEN_NUM = 128
TAU = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRQN(nn.Module):
    def __init__(self, action_dim, observation_dim, layer_num=1, hidden_num=HIDDEN_NUM):
        super(DRQN, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = observation_dim
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        self.lstm = nn.LSTM(input_size=self.obs_dim,
                            hidden_size=self.hidden_num, num_layers=self.layer_num, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_num, 128)
        self.fc2 = nn.Linear(128, self.action_dim)

    def forward(self, x, hidden=None):
        if not hidden:
            # 初始化隐层（层数， batch大小， 隐层维度）
            h0 = torch.zeros(self.layer_num, x.size(0),
                             self.hidden_num).to(device)
            c0 = torch.zeros(self.layer_num, x.size(0),
                             self.hidden_num).to(device)
            hidden = (h0, c0)

        h1, new_hidden = self.lstm(x, hidden)
        h2 = F.relu(self.fc1(h1))
        h3 = self.fc2(h2)
        return h3, new_hidden


class ReplayBuffer:
    def __init__(self, action_size, capacity, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=[
                                     "obs", "action", "reward", "next_obs", "done"])

    def add(self, obs, action, reward, next_obs, done):
        # obs = np.expand_dims(obs, 0)
        # next_obs = np.expand_dims(next_obs, 0)
        e = self.experience(obs, action, reward, next_obs, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        obss = torch.from_numpy(
            np.vstack([e.obs for e in experiences if e is not None])).unsqueeze(1).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_obss = torch.from_numpy(
            np.vstack([e.next_obs for e in experiences if e is not None])).unsqueeze(1).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (obss, actions, rewards, next_obss, dones)

    def __len__(self):
        return len(self.memory)


class Agent(object):
    def __init__(self, action_size, observation_size, batch_size, seed):
        self.action_size = action_size
        self.obs_size = observation_size
        self.batch_size = batch_size
        self.seed = seed
        self.buffer = ReplayBuffer(
            action_size, CAPACITY, self.batch_size, seed)
        self.target_network = DRQN(action_size, observation_size).to(device)
        self.local_network = DRQN(action_size, observation_size).to(device)
        self.local_network.load_state_dict(self.target_network.state_dict())
        self.optimizer = torch.optim.Adam(
            self.local_network.parameters(), lr=LEARNING_RATE)

    def step(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)
        if len(self.buffer) > self.batch_size:
            experiences = self.buffer.sample()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, GAMMA):
        obss, actions, rewards, next_obss, dones = experiences
        # print(next_obss.shape[0], next_obss.shape[1]) next obs （32*4）
        # print(next_obss)
        self.local_network.eval()
        # 采用 double DQN 方式更新
        with torch.no_grad():
            Q_preds, _ = self.local_network(next_obss)  # 选动作
            max_actions = Q_preds.max(-1)[1].detach()
        self.local_network.train()
        Q_expecteds, _ = self.local_network(obss)
        Q_nexts, _ = self.target_network(next_obss)
        Q_next = Q_nexts.gather(-1, max_actions.unsqueeze(-1)).squeeze(-1)
        Q_expected = Q_expecteds.gather(-1,
                                        actions.unsqueeze(-1)).squeeze(-1)
        Q_target = rewards + (GAMMA * (1-dones) * Q_next)
        # print("选动作 local next obs： ", Q_preds)
        # print("动作： ", max_actions)
        # print("期望Q值： ", Q_expected, Q_expected[0].requires_grad)
        # print("目标Q值： ", Q_target, Q_target[0].requires_grad)
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_network, self.target_network, TAU)

    def act(self, obs, epsilon, hidden):
        obs = torch.FloatTensor(np.expand_dims(
            np.expand_dims(obs, 0), 0)).to(device)
        # print("obs: ", obs)
        self.local_network.eval()
        with torch.no_grad():
            q_values, new_hidden = self.local_network.forward(obs, hidden)
        self.local_network.train()
        if random.random() > epsilon:
            action = np.argmax(q_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        return action, new_hidden

    def soft_update(self, local_net, target_net, tau):
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)
