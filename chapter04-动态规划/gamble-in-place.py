import numpy as np
import matplotlib.pyplot as plt


ph = 0.4
sweep_history = []
state_nums = 100  # 状态数目，实际上加上零有51个状态，不要在意细节


def in_place():
    states_values = np.zeros(state_nums + 1)
    states_values[state_nums] = 1
    while True:
        diff = 0
        states_values_old = states_values.copy()
        sweep_history.append(states_values_old)
        for state in range(state_nums):
            value_cans = []
            actions_cans = min(state_nums - state, state)  # 例如20
            for action in range(actions_cans+1):  # 0-20
                value = ph * states_values[state+action] + (1-ph) * states_values[state-action]
                value_cans.append(value)
            diff += abs(states_values[state] - max(value_cans))  # 变化量
            states_values[state] = max(value_cans)
        if diff < 1e-9:
            break

    # for index, data in enumerate(sweep_history):
    #     plt.plot(data, label="Iteration{}".format(index))
    # plt.xlabel("States")
    # plt.ylabel("Value")
    # plt.legend(loc="best")
    # plt.show()
    # print(states_values)

    # 绘制0-50 共51状态的最佳决策
    policy = np.zeros((state_nums+1))  # 0-100
    for state in range(1, state_nums):  # 1-100，例如20
        value_cans = []
        actions_cans = min(state, state_nums-state)
        for action in range(1, actions_cans+1):  # 0-20
            value = ph * states_values[state + action] + (1 - ph) * states_values[state - action]
            value = round(value, 5)
            value_cans.append([value, action])
        value_cans.sort(key=lambda x: x[0], reverse=True)
        policy[state] = value_cans[0][1]  # arg：0-20
        # print("状态{}，最佳决策{}".format(state, value_cans[0][1]))
        policy[state] = value_cans[0][1]  # arg：0-20

    plt.scatter(np.arange(state_nums+1), policy)
    plt.xlabel("States")
    plt.ylabel("Best-Choice")
    plt.show()


def instant_reward():
    states_values = np.zeros(state_nums + 1)
    states_values[-1] = 100
    while True:
        diff = 0
        states_values_old = states_values.copy()
        sweep_history.append(states_values_old)
        for state in range(state_nums, 0, -1):
            value_cans = []
            actions_cans = min(state_nums - state, state)  # 例如20
            for action in range(actions_cans+1):  # 0-20
                value = ph * (action+states_values[state+action]) + (1-ph) * (-action+states_values[state-action])
                value_cans.append(value)
            diff += abs(states_values[state] - max(value_cans))  # 变化量
            states_values[state] = max(value_cans)
        if diff < 1e-1:
            break

    for index, data in enumerate(sweep_history):
        plt.plot(data, label="Iteration{}".format(index))
    plt.xlabel("States")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    in_place()
    # instant_reward()