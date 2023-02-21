import gym
import torch
from DQNAgent import Agent
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


def train(n_episoids=2000, max_t=1000, eps_start=1.00, eps_end=0.01, eps_decay=0.995, seed=543):
    env = gym.make('LunarLander-v2')
    fix(env, seed)
    scores_array, avg_scores = [], []
    scores_deque = deque(maxlen=100)
    eps = eps_start

    for i_episoid in range(1, n_episoids + 1):
        state = env.reset()
        score = 0
        for i in range(max_t):
            action = agent.act(state, eps)  # 使用局部网络选择动作
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
        scores_array.append(score)
        scores_deque.append(score)
        avg_scores.append(np.mean(scores_deque))
        eps = max(eps_end, eps * eps_decay)
        print('\rEpisoid : {}, \tAverage Score : {:.2f}, \teps : {}'.format(i_episoid, np.mean(scores_deque), eps))
        if i_episoid % 100 == 0:
            print('\rEpisoid : {}, \tAverage Score : {:.2f}'.format(i_episoid, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 200.0:
            print('\n 训练完成, Average Score: {:.2f}'.format(np.mean(scores_deque)))
            torch.save(agent.qnetwork_local.state_dict(), 'model.pth')
            break
    return scores_array, avg_scores


def test():
    agent.qnetwork_local.load_state_dict(torch.load('model.pth'))
    env = gym.make('LunarLander-v2')
    for i in range(3):
        state = env.reset()
        for j in range(200):
            action = agent.act(state)
            plt.axis('off')
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


if __name__ == '__main__':
    random_seed = 543
    agent = Agent(state_size=8, action_size=4, seed=random_seed)
    scores, avg_scores = train()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores, label="Score")
    plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label="Avg on 100 episodes")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.ylabel('Score')
    plt.xlabel('Episodes ')
    plt.show()
    plt.tight_layout()
    fig.savefig('scores.png', bbox_inches='tight')

    test()
