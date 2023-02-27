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
CAPACITY = 2000
LEARNING_RATE = 1e-3
HIDDEN_NUM = 128
TAU = 1e-2
UPDATE_PERIOD = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRQN(nn.Module):
    def __init__(self, action_dim, observation_dim, layer_num=1, hidden_num=HIDDEN_NUM):
        super(DRQN, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = observation_dim
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        # TODO: ADRQN : 动作特征与同时输入LSTM中
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
    """ 只抽样一个 episode 的数据 """

    def __init__(self, capacity, batch_size, seed):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        # self.experience = namedtuple("Experience", field_names=[
        #                              "obs", "action", "reward", "next_obs", "done"])

    # def add(self, obs, action, reward, next_obs, done):
    #     # obs = np.expand_dims(obs, 0)
    #     # next_obs = np.expand_dims(next_obs, 0)
    #     e = self.experience(obs, action, reward, next_obs, done)
    #     self.memory.append(e)
    def add(self, episode):
        self.memory.append(episode)

    def sample(self):
        idx = np.random.randint(0, len(self.memory))
        obs, action, reward, next_obs, done = self.memory[idx].sample()
        epi_len = len(self.memory[idx])
        obs = torch.FloatTensor(obs.reshape(
            self.batch_size, epi_len, -1)).to(device)
        action = torch.LongTensor(action.reshape(
            self.batch_size, epi_len, -1)).to(device)
        reward = torch.FloatTensor(reward.reshape(
            self.batch_size, epi_len, -1)).to(device)
        next_obs = torch.FloatTensor(next_obs.reshape(
            self.batch_size, epi_len, -1)).to(device)
        done = torch.FloatTensor(done.reshape(
            self.batch_size, epi_len, -1)).to(device)
        # experiences = random.sample(self.memory, k=self.batch_size)
        # obss = torch.from_numpy(
        #     np.vstack([e.obs for e in experiences if e is not None])).unsqueeze(1).float().to(device)
        # actions = torch.from_numpy(
        #     np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # rewards = torch.from_numpy(
        #     np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_obss = torch.from_numpy(
        #     np.vstack([e.next_obs for e in experiences if e is not None])).unsqueeze(1).float().to(device)
        # dones = torch.from_numpy(
        #     np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (obs, action, reward, next_obs, done)

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    """放一个 episode 的数据"""

    def __init__(self) -> None:
        self.episode = []
        self.trajectory = namedtuple("trajectory", field_names=[
            "obs", "action", "reward", "next_obs", "done"])

    def add(self, obs, action, reward, next_obs, done):
        tr = self.trajectory(obs, action, reward, next_obs, done)
        self.episode.append(tr)

    def sample(self):
        obs = np.array([t.obs for t in self.episode if t is not None])
        action = np.array([t.action for t in self.episode if t is not None])
        reward = np.array([t.reward for t in self.episode if t is not None])
        next_obs = np.array(
            [t.next_obs for t in self.episode if t is not None])
        done = np.array([t.done for t in self.episode if t is not None])
        return (obs, action, reward, next_obs, done)

    def __len__(self):
        return len(self.episode)


class Agent(object):
    def __init__(self, action_size, observation_size, batch_size, seed):
        self.action_size = action_size
        self.obs_size = observation_size
        self.batch_size = batch_size
        self.seed = seed
        self.buffer = ReplayBuffer(CAPACITY, self.batch_size, seed)
        self.local_network = DRQN(action_size, observation_size).to(device)
        self.target_network = DRQN(action_size, observation_size).to(device)
        self.target_network.load_state_dict(self.local_network.state_dict())
        self.optimizer = torch.optim.Adam(
            self.local_network.parameters(), lr=LEARNING_RATE)

    # def step(self, obs, action, reward, next_obs, done, step_cnt):
    #     self.buffer.add(obs, action, reward, next_obs, done)
    #     if len(self.buffer) > self.batch_size:
    #         experiences = self.buffer.sample()
    #         self.learn(experiences, GAMMA, step_cnt)

    def learn(self, experiences, GAMMA, step_cnt):
        obss, actions, rewards, next_obss, dones = experiences
        # 采用 double DQN 方式更新
        self.local_network.eval()
        with torch.no_grad():
            Q_preds, _ = self.local_network(next_obss)  # 选动作
            max_actions = Q_preds.max(-1)[1].detach()
        self.local_network.train()
        Q_expecteds, _ = self.local_network(obss)
        Q_nexts, _ = self.target_network(next_obss)
        Q_next = Q_nexts.gather(-1, max_actions.unsqueeze(-1)).squeeze(-1)
        Q_expected = Q_expecteds.gather(-1, actions).squeeze(-1)
        Q_target = rewards.squeeze(-1) + (GAMMA * (1-dones.squeeze(-1)) * Q_next)
        # print("local next obs: ", Q_preds)
        # print("期望动作： ", max_actions.unsqueeze(-1))
        # print("期望Q值： ", Q_expecteds)
        # print("目标Q值： ", Q_nexts)
        # print("目标动作Q值： ", Q_next)
        # print("回报： ", rewards.squeeze(-1))
        # print("当前动作： ", actions)
        # print("期望Q值： ", Q_expected)
        # print("目标Q值： ", Q_target)
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if step_cnt % UPDATE_PERIOD == 0:
            self.soft_update(self.local_network, self.target_network, TAU)

    def act(self, obs, epsilon, hidden):
        obs = torch.FloatTensor(np.expand_dims(
            np.expand_dims(obs, 0), 0)).to(device)
        self.local_network.eval()
        with torch.no_grad():
            q_values, new_hidden = self.local_network.forward(obs, hidden)
        self.local_network.train()
        # print("obs: ", obs)
        # print("qvalues", q_values)
        # print("action", np.argmax(q_values.cpu().data.numpy()))
        if random.random() > epsilon:
            action = np.argmax(q_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        return action, new_hidden

    def soft_update(self, local_net, target_net, tau):
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)
