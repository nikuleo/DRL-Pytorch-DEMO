import gym
from ACAgent import ACAgent
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def train(agent, n_episoids, max_t, seed, isload=False):
    env = gym.make("LunarLander-v2")
    fix(env, seed)

    rewards_array, avg_rewards = [], []
    loss_array = []
    rewards_deque = deque(maxlen=100)
    if isload:
        agent.load(config['path'])
    for i_episoid in range(1, n_episoids + 1):
        if (i_episoid % 200 == 0):
            agent.save(config['path'])
        state = env.reset()
        total_reward = 0
        for t in range(max_t):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
            total_reward += reward
            if done:
                break
        loss = agent.leran()
        loss_array.append(loss)
        rewards_array.append(total_reward)
        rewards_deque.append(total_reward)
        avg_rewards.append(np.mean(rewards_deque))
        agent.clearMemory()
        print('\rEpisoid : {}, \tAverage Reward : {:.2f}'.format(i_episoid, np.mean(rewards_deque)))
        if i_episoid % 100 == 0:
            print('\rEpisoid : {}, \tAverage Reward : {:.2f}'.format(i_episoid, np.mean(rewards_deque)))
        if np.mean(rewards_deque) >= 200.0:
            print('\n 训练完成, Average Reward: {:.2f}'.format(np.mean(rewards_deque)))
            agent.save(config['path'])
            break
    return loss_array, rewards_array, avg_rewards


def test():
    agent_t = ACAgent(state_size=8, action_size=4, seed=config['seed'], fc1_units=config['fc1_utils'], lr=config['lr'],
                      betas=config['betas'])
    agent_t.load(config['path'])
    env = gym.make('LunarLander-v2')
    for i in range(5):
        state = env.reset()
        for j in range(500):
            action = agent_t.act(state)
            state, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
    env.close()


def fix(env, seed):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


config = {
    'n_eposoids': 5000,
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'max_t': 1000,
    'seed': 543,
    'fc1_utils': 128,
    'path': "./model/model.pt",
    'isload': False
}

if __name__ == '__main__':
    agent = ACAgent(state_size=8, action_size=4, seed=config['seed'], fc1_units=config['fc1_utils'], lr=config['lr'],
                    betas=config['betas'])
    la, ra, ar = train(agent, config['n_eposoids'], config['max_t'], config['seed'], config['isload'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(ra) + 1), ra, label="reward")
    plt.plot(np.arange(1, len(ar) + 1), ar, label="Avg on 100 episodes")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.ylabel('Rewards')
    plt.xlabel('Episodes ')
    plt.tight_layout()
    plt.show()
    fig.savefig('scores.png', bbox_inches='tight')

    plt.plot(la)
    plt.title("loss")
    plt.show()

    test()
