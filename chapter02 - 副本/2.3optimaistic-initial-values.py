#######################################################################
# 目标：绘制图2.3                                                     #
# 总共采样2000轮，当然不一定是2000轮，设为2000可以取得比较平稳的结果  #
# 每一轮采样1000个steps，跳出局部最优需要steps足够大，当然这和k有关   #
# 流程：action---reward---estimate更新                                #
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


class bandit():
    def __init__(self, k=10, episilon=0., step_size=0.1, initial_value=5.0):
        self.k = k  # 摇臂数
        self.actions = np.arange(self.k)  # 行为空间
        self.epsilon = episilon  # exploration probility
        self.step_size = step_size
        self.initial_value = initial_value

    def reset(self):
        self.ground_mean = np.random.randn(self.k)  # 均值服从标准分布
        self.count = np.zeros(self.k)
        self.estimate = np.array([self.initial_value] * self.k)  # 不同摇臂的价值估计
        self.best_action = np.argmax(self.ground_mean)  # int

    def action(self):
        if np.random.rand() < self.epsilon:
            act = np.random.choice(self.actions)
        else:
            # act = np.argmax(self.estimate)
            q_beat = np.max(self.estimate)
            act = np.random.choice(np.where(self.estimate == q_beat)[0])

        self.count[act] += 1
        return act

    def get_reward(self, act):
        current_reward = np.random.randn() + self.ground_mean[act]
        return current_reward

    def update_estimate(self, act, current_reward):
        td_error = current_reward - self.estimate[act]
        self.estimate[act] += self.step_size * td_error
        return

def simulate(bans, run=2000, steps=1000):
    rewards = np.zeros((len(bans), run, steps))
    is_best_action = np.zeros((len(bans), run, steps))
    for i, ban in enumerate(bans):
        for j in trange(run):
            ban.reset()
            for k in range(steps):
                act = ban.action()
                reward = ban.get_reward(act)
                ban.update_estimate(act, reward)
                rewards[i, j, k] = reward
                if act == ban.best_action:
                    is_best_action[i, j, k] = 1
    return rewards.mean(1), is_best_action.mean(1)


def graph2_3():
    # 如果episilon为0的话，会直到将一个选择的均
    # 值采样平均到0以下，才会换选择，易局部最优
    episilon = [0, 0.1]
    initial_val = [5.0, 0.0]
    bans = [bandit(episilon=e, initial_value=i) for e, i in zip(episilon, initial_val)]

    # rewards_mean维度：len(episilon), 1000
    plt.subplot(2, 1, 1)
    rewards_mean, best_action_mean = simulate(bans, run=2000, steps=1000)
    plt.plot(rewards_mean[0], label='episilon=0.0, initial=5.0')
    plt.plot(rewards_mean[1], label='episilon=0.1, initial=0.0')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(best_action_mean[0], label='episilon=0.0, initial=5.0')
    plt.plot(best_action_mean[1], label='episilon=0.1, initial=0.0')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.savefig('./2_3.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    graph2_3()
