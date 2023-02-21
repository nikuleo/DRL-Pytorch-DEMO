import matplotlib.pyplot as plt
import torch
import gym
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque


def fix(env, seed):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)


class PolicyGradientAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)
        # self.optimizer = optim.Adam(self.network.parameters(), lr=config['lr'])

    def forward(self, state):
        return self.network(state)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 返回动作和概率
    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)  # 初始化抽样类
        action = action_dist.sample()  # 抽样
        log_prob = action_dist.log_prob(action)  # 输出该动作的对率概率e/e+e+e
        return action.item(), log_prob

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


def train(agent, n_episoids, max_t=1000, path="./model/model.pt", isreed=False):
    if (isreed):
        agent.load(path)

    agent.network.train()

    total_rewards, final_rewards, avg_rewards = [], [], []
    rewards_deque = deque(maxlen=100)

    for i_episoid in range(1, n_episoids + 1):
        if (i_episoid % 200 == 0):
            agent.save(path)
        state = env.reset()
        total_reward = 0
        log_probs, rewards = [], []
        for i in range(max_t):

            action, log_prob = agent.sample(state)  # at , log(at|st)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            total_reward += reward

            last_reward = reward
            for j in range(len(rewards) - 1):
                last_reward *= 0.99
                rewards[len(rewards) - 2 - j] += last_reward
            if done:
                final_rewards.append(reward)
                break
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 标准化
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))

        total_rewards.append(total_reward)
        rewards_deque.append(total_reward)
        avg_rewards.append(np.mean(rewards_deque))

        print('\rEpisoid : {}, \tAverage Reward : {:.2f}'.format(i_episoid, np.mean(rewards_deque)))
        if i_episoid % 100 == 0:
            print('\rEpisoid : {}, \tAverage Reward : {:.2f}'.format(i_episoid, np.mean(rewards_deque)))
        if np.mean(rewards_deque) >= 100.0:
            print('\n 训练完成, Average Score: {:.2f}'.format(np.mean(rewards_deque)))
            agent.save(path)
            break

    return total_rewards, avg_rewards, final_rewards


def test():
    network_t = PolicyGradientNetwork()
    agent_t = PolicyGradientAgent(network_t)
    agent_t.load(config['path'])
    env = gym.make('LunarLander-v2')
    for i in range(5):
        state = env.reset()
        for j in range(500):
            action, _ = agent_t.sample(state)
            plt.axis('off')
            state, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
    env.close()


config = {
    'n_episodes': 5000,
    'lr': 0.0005,
    'max_t': 1000,
    'path': "./lunar_lander_PG/model/model.pt"
}

if __name__ == '__main__':
    seed = 543
    env = gym.make('LunarLander-v2')
    fix(env, seed)

    network = PolicyGradientNetwork()
    agent = PolicyGradientAgent(network)

    total_r, avg_r, final_r = train(agent, config['n_episodes'], config['max_t'], path=config['path'],
                                    isreed=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(total_r) + 1), total_r, label="Total Reward")
    plt.plot(np.arange(1, len(avg_r) + 1), avg_r, label="Avg on 100 episodes")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.ylabel('Score')
    plt.xlabel('Episodes ')
    plt.tight_layout()
    plt.show()
    fig.savefig('scores.png', bbox_inches='tight')

    plt.plot(final_r)
    plt.title("Final Rewards")
    plt.show()
    test()
