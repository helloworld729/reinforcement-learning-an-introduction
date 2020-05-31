#######################################################################
# 目标：绘制图2.2                                                     #
# 总共采样2000轮，当然不一定是2000轮，设为2000可以取得比较平稳的结果  #
# 每一轮采样1000个steps，跳出局部最优需要steps足够大，当然这和k有关   #
# 流程：action---reward---estimate更新                                #
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


class bandit():
    def __init__(self, k=10, episilon=0.):
        self.k = k  # 摇臂数
        self.actions = np.arange(self.k)  # 行为空间
        self.epsilon = episilon  # exploration probility

    def reset(self):
        self.ground_mean = np.random.randn(self.k)  # 均值服从标准分布
        self.count = np.zeros(self.k)
        self.estimate = np.zeros(self.k)  # 不同摇臂的价值估计
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
        self.estimate[act] += td_error / self.count[act]
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


def graph2_2():
    # 如果episilon为0的话，会直到将一个选择的均
    # 值采样平均到0以下，才会换选择，易局部最优
    episilon = [0, 0.01, 0.1]
    bans = [bandit(episilon=e) for e in episilon]

    # rewards_mean维度：len(episilon), 1000
    plt.subplot(2, 1, 1)
    rewards_mean, best_action_mean = simulate(bans, run=2000, steps=1000)
    for eps, reward in zip(episilon, rewards_mean):
        # rewards 包含1000个数据
        plt.plot(reward, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(episilon, best_action_mean):
        plt.plot(counts, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.savefig('./2_2.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    graph2_2()

# bug：
# estimate 没有初始化
# np.zeros 数据类型问题
# greedy 用np.choice而不是np.argmax,因为最大值可能不止一个

# 问题：为什么没有将估值初始化，存在episilon的时候仍有好的结果
# 因为对count进行了初始化，容易跳出局部最优解


