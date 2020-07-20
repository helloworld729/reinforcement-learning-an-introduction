from copy import copy
lst = [10, 1, 5, 3, 7, 2, 9, 1]
res = []
num_sum = 0
def dfs(id, temp):
    if len(temp) == 3:
        print(temp)
        global res
        if res and sum(temp) > num_sum:
            res = copy(temp)

        return
    for next_pos in [id+1, id+2]:
        temp.append(lst[next_pos])
        dfs(next_pos, temp)

for index, num in enumerate(lst[:-4]):
    dfs(index, 0)


# print(lst[:-4])