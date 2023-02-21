import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import gym
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(8, 128)

        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))

        state_value = self.value_layer(state)
        action_probs = F.softmax(self.action_layer(state), dim=-1)

        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    def calculateLoss(self, discount=0.99):

        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:  # ::-1 倒序遍历
            dis_reward = reward + discount * dis_reward
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
            # print(reward, value)
            loss += (action_loss + value_loss)
        return loss

    def clearMemory(self):  # 只清除 list 里的值
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


def fix(env, seed):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(agent, n_episoids, max_t, seed, discount):
    render = False
    lr = 0.02
    betas = (0.9, 0.999)

    env = gym.make('LunarLander-v2')
    fix(env, seed)

    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)

    total_rewards, final_rewards, total_loss = [], [], []
    rewards_array, avg_rewards = [], []
    rewards_deque = deque(maxlen=100)

    running_reward = 0
    for i_episoid in range(1, n_episoids + 1):
        state = env.reset()
        total_reward = 0
        for t in range(max_t):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward
            total_reward += reward
            if render and i_episoid > 1000:
                env.render()
            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                break
        rewards_array.append(total_reward)
        rewards_deque.append(total_reward)
        avg_rewards.append(np.mean(rewards_deque))
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(discount)  # 动作概率 loss 和 状态奖励 loss 的和
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        policy.clearMemory()

        # saving the model if episodes > 999 OR avg reward > 200
        # if i_episode > 999:
        #    torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))

        if running_reward > 4000:
            torch.save(policy.state_dict(), './model/AC_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            print("########## Solved! ##########")
            test(name='AC_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            break

        if i_episode % 20 == 0:
            running_reward = running_reward / 20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0
    return total_loss, total_rewards, final_rewards


def test(n_episodes=10, name='LunarLander_TWO.pth'):
    env = gym.make('LunarLander-v2')
    policy = ActorCritic()

    policy.load_state_dict(torch.load('./model/{}'.format(name)))

    render = True
    save_gif = False

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
                if save_gif:
                    img = env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()


config = {
    'discount': 0.99,
    'lr': 0.02,
    'betas': (0.9, 0.999),
    'seed': 543,
    'n_episoids': 10000,
    'max_t': 1000,
}

if __name__ == '__main__':
    agent = ActorCritic()
    tl, tr, fr = train(agent, config['n_episoids'], config['max_t'], config['seed'], config['discount'])
    plt.plot(tr)
    plt.title("Total Rewards")
    plt.show()

    plt.plot(fr)
    plt.title("Final Rewards")
    plt.show()

    plt.plot(tl)
    plt.title("loss")
    plt.show()
