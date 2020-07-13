import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1  #  "strike" in the book
ACTIONS = [ACTION_HIT, ACTION_STAND]

# 玩家策略
POLICY_PLAYER = np.zeros(22, dtype=np.int)  # 初始化
# 策略装填
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT  # 12-20一定是hit
POLICY_PLAYER[20] = ACTION_STAND   # >=20就stand
POLICY_PLAYER[21] = ACTION_STAND

# function form of target policy of player
# def target_policy_player(usable_ace_player, player_sum, dealer_card):
#     return POLICY_PLAYER[player_sum]

# function form of behavior policy of player
# def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
#     if np.random.binomial(1, 0.5) == 1:
#         return ACTION_STAND
#     return ACTION_HIT

# 庄家策略
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND

# get a new card
def get_card():
    card = np.random.randint(1, 14)  # 左闭右开
    card = min(card, 10)
    return card

# get the value of a card (11 for ace).
def card_value(card_id):
    return 11 if card_id == 1 else card_id


def useable_A(lst):
    total = sum(lst)
    if 11 not in lst:
        return False, total
    elif sum(lst) < 22:
        return True, total
    else:
        info = {}
        for data in lst:
            if info.get(data, -1) == -1:
                info[data] = 1
            else:
                info[data] += 1
        while total >= 22 and info[11] > 0:
            total -= 10
            info[11] -= 1
        if total > 21 or info[11] <= 0:
            return False, total
        else:
            return True, total
# print(useable_A([10,10,10]))     # (False, 30)
# print(useable_A([11,2,2]))       # (True, 15)
# print(useable_A([11,3,10]))      # (False, 14)
# print(useable_A([11,11,7, 11]))  # (True, 20)
# print(useable_A([11, 11]))  # (True, 20)



def play():
    """ 模拟游戏，返回：是否有可用的A、玩家手牌之和、庄家明牌 """
    #################### 初始化 ##########################
    history = []
    player_card = []
    dealer_card = []
    #################### 每人发2张牌 ##########################
    for _ in range(2):
        num_i = get_card()
        num_j = get_card()
        player_card.append(card_value(num_i))
        dealer_card.append(card_value(num_j))
    open = dealer_card[1]
    open = open if open != 11 else 1
    _, d_score = useable_A(dealer_card)
    p_useable, p_score = useable_A(player_card)
    if p_score > 21:
        print("p", player_card)
    if d_score > 21:
        print("d", dealer_card)
    history.append(p_score)
    assert p_score <= 21
    assert d_score <= 21
    #################### natural/draw ##########################
    if p_score == 21 and d_score == 21:
        draw = True
        return [[p_useable, 21, open, 0]]
    elif p_score == 21:
        natural = True
        return [[p_useable, 21, open, 1]]
    #################### 玩家阶段 ##########################
    while True:
        if p_score >= 20:
            break
        new_card = card_value(get_card())
        player_card.append(new_card)
        p_useable, p_score = useable_A(player_card)
        if p_score <= 21:
            history.append(p_score)
        if p_score > 21:
            bust = True
            return [[False, i, open, -1] for i in history]
    #################### 庄家阶段 ##########################
    while True:
        if d_score >= 17:
            break
        new_card = card_value(get_card())
        dealer_card.append(new_card)
        d_useable, d_score = useable_A(dealer_card)
        if d_score > 21:
            bust = True
            return [[p_useable, i, open, 1] for i in history]

    #################### 判决阶段 ##########################
    if p_score == d_score:
        reward = 0
    elif p_score < d_score:
        reward = -1
    else:
        reward = 1
    # print("奖励值：{}，玩家：{}， 庄家：{}".format(reward, player_card, dealer_card))
    # print([[p_useable, i, open, reward]for i in history])
    return [[p_useable, i, open, reward]for i in history]


# Monte Carlo Sample with On-Policy
def monte_carlo_on_policy(episodes):
    states_usable_ace = np.zeros((10, 10))
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))

    for i in tqdm(range(0, episodes)):
        # 返回奖励值、游戏记录
        # usable_ace, player_sum, open, reward, player_card, dealer_card = play()
        # print("奖励值：{}，玩家：{}， 庄家：{}".format(reward, player_card, dealer_card))
        for usable_ace, player_sum, open, reward in play():

            assert player_sum <= 21
            assert open <= 10
            player_sum -= 12
            open -= 1
            if usable_ace:
                # 最终绘制的热力图上，player_sum不代表终态，而是基于
                # 当前的策略（阈值设为20），该状态下的奖励平均值。
                states_usable_ace_count[player_sum, open] += 1
                states_usable_ace[player_sum, open] += reward
            else:
                states_no_usable_ace_count[player_sum, open] += 1
                states_no_usable_ace[player_sum, open] += reward
    # 返回平均值（某个状态获得的奖励和/状态出现的次数）
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count


def figure_5_1():
    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(1000000)

    states = [states_usable_ace_1,
              states_no_usable_ace_1]

    titles = ['Usable Ace, 500000 Episodes',
              'No Usable Ace, 500000 Episodes',]

    plt.matshow(states[0])
    plt.colorbar()
    plt.title(titles[0])
    plt.xlabel("dealer showing")
    plt.ylabel("palyer sum")
    # plt.xlim([1, 10])
    # plt.ylim([12, 21])
    # plt.xticks(range(2, 12))
    # plt.yticks(range(12, 22))
    # plt.xlabel(range(2, 12))
    # plt.ylabel(range(12, 22))
    plt.show()
    plt.close()

figure_5_1()