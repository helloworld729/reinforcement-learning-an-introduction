import numpy as np
import matplotlib.pyplot as plt
import time

ph = 0.4  # 正面概率
state_nums = 50  # 状态个数
states_value = np.zeros((1 + state_nums))  # 0-100的价值判断
money = np.arange(1 + state_nums)  # 金钱区间
states_value[state_nums] = 1.0  # 100yuan的时候，价值固定为1.0
print(states_value)
sweep_history = []
sweep_history.append(states_value)
while True:
    diff = 0
    new_state_value = np.zeros((1 + state_nums))
    for i in range(1, state_nums):  # 遍历需要更新的状态
        return_list = []
        action_range = min(i, state_nums-i)
        for action in range(action_range+1):
            expected_return = ph * states_value[i + action] + (1 - ph) * states_value[i - action]
            return_list.append(expected_return)
        new_state_value[i] = max(return_list)  # 贪心评估
        diff += abs(new_state_value[i] - states_value[i])
    new_state_value[state_nums] = 1
    print(new_state_value)
    print("\n")
    states_value = new_state_value
    sweep_history.append(new_state_value)
    if diff < 1e-3:
        break

for index, data in enumerate(sweep_history):
    plt.plot(data, label="Iteration{}".format(index))
plt.xlabel('States')
plt.ylabel('Value')
plt.legend(loc="best")
plt.show()