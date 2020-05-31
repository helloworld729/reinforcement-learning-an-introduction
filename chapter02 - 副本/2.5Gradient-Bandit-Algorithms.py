import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


class bandit():
    def __init__(self, k=10, step_size=0.0, with_base_line=True):
        self.k = k  # 摇臂数
        self.actions = np.arange(self.k)  # 行为空间
        self.step_size = step_size
        self.with_base_line = with_base_line

    def reset(self):
        self.ground_mean = np.random.randn(self.k)+4  # 均值为4，方差为1
        self.count = np.zeros(self.k)
        self.estimate = np.asarray([0.0] * self.k)  # 不同摇臂的价值估计
        self.probility = np.asarray([1/self.k] * self.k)  # 不同摇臂的概率估计
        self.best_action = np.argmax(self.ground_mean)  # int
        self.t = 0
        self.base_line = 0

    def action(self):
        act = np.random.choice(self.actions, p=self.probility)
        self.t += 1
        return act

    def get_reward(self, act):
        current_reward = np.random.randn() + self.ground_mean[act]
        if self.with_base_line:
            self.base_line = self.base_line + (current_reward - self.base_line)/self.t
        return current_reward

    def update_estimate(self, act, current_reward):
        td_error = current_reward - self.base_line
        for act_cans in self.actions:
            self.estimate[act_cans] = self.estimate[act_cans] + self.step_size * td_error *(1 - self.probility[act_cans]) \
                if act_cans == act else self.estimate[act_cans] - self.step_size * td_error * self.probility[act_cans]
        denominator = np.sum(np.exp(self.estimate))
        self.probility = np.exp(self.estimate) / denominator

        return 0

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


def graph2_5():
    step_size = [0.1, 0.4]
    bans = [bandit(step_size=s, with_base_line=w) for s, w in zip(
        step_size * 2, [True, True, False, False])]

    rewards_mean, best_action_mean = simulate(bans, run=2000, steps=1000)
    plt.plot(best_action_mean[0], label='step_size=0.1, with_baseLine')
    plt.plot(best_action_mean[1], label='step_size=0.4, with_baseLine')
    plt.plot(best_action_mean[2], label='step_size=0.1, without_baseLine')
    plt.plot(best_action_mean[3], label='step_size=0.4, without_baseLine')

    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.savefig('./2_5.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    graph2_5()
