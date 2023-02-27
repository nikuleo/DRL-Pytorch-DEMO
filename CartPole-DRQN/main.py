import torch
import gym
from DRQNAgent import Agent
from DRQNAgent import EpisodeBuffer
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import math

# Hyperparemeters
BATCH_SIZE = 1
DECAY = 0.998
EPSILON_INIT = 0.9
EPSILON_MIN = 0.05
EXPLORE = 300
GAMMA = 0.9
MAX_EPISODE = 100000
SEQ_LEN = 50
SOFT_UPDATE_FREQ = 100
SEED = 2233
RENDER = False
MAX_T = 2000


def fix(env, seed):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(env, agent):
    epsilon = EPSILON_INIT
    scores_array, scores_avg = [], []
    scores_deque = deque(maxlen=100)

    for i_episode in range(1, MAX_EPISODE+1):
        episode_buffer = EpisodeBuffer()
        obs = env.reset()
        hidden = None
        reward_total = 0
        for i in range(MAX_T):
            action, hidden = agent.act(obs, epsilon, hidden)
            # reward 活着返回1 ，结束返回0
            next_obs, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0
            episode_buffer.add(obs, action, reward, next_obs, done)
            if RENDER:
                env.render()
            # agent.step(obs, action, reward, next_obs, done_mask, i_episode)
            if len(agent.buffer) > BATCH_SIZE:
                experience = agent.buffer.sample()
                agent.learn(experience, GAMMA, i_episode)
            reward_total += reward
            obs = next_obs
            if done:
                break
        agent.buffer.add(episode_buffer)
        scores_array.append(reward_total)
        scores_deque.append(reward_total)
        scores_avg.append(np.mean(scores_deque))
        # epsilon = EPSILON_MIN + (EPSILON_INIT - EPSILON_MIN) * \
        #     math.exp((-1*i_episode)/10)
        epsilon = max(EPSILON_MIN, DECAY*epsilon)
        print('\rEpisoid : {}, \tAverage Score : {:.2f}, \teps : {}'.format(
            i_episode, np.mean(scores_deque), epsilon))
        if i_episode % 100 == 0:
            print('\rEpisoid : {}, \tAverage Score : {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 190:
            print('\n 训练完成, Average Score: {:.2f}'.format(
                np.mean(scores_deque)))
            torch.save(agent.local_network.state_dict(),
                       './CartPole-DRQN/model/model.pth')
            break
    return scores_array, scores_avg


def test():
    env = gym.make("CartPole-v1")
    env.unwrapped
    agent.local_network.load_state_dict(
        torch.load('./CartPole-DRQN/model/model.pth'))
    hidden = None
    for i in range(20):
        obs = env.reset()
        for j in range(MAX_T):
            action, hidden = agent.act(obs, 1, hidden)
            plt.axis('off')
            obs, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
    env.close()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env = env.unwrapped  # 得到原始环境，不受步数限制
    fix(env, SEED)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Agent(action_dim, obs_dim, BATCH_SIZE, SEED)
    scores, scores_avg = train(env, agent)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores, label="Score")
    plt.plot(np.arange(1, len(scores_avg) + 1),
             scores_avg, label="Avg on 100 episodes")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.ylabel('Score')
    plt.xlabel('Episodes ')
    plt.show()
    plt.tight_layout()
    fig.savefig('scores.png', bbox_inches='tight')

    test()
