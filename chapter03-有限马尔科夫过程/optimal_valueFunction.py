import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


gamma = 0.9
ROW, COLUMN = 5, 5
A, B = (0, 1), (0, 3)
act_probility = 0.25


def draw_image(image):
    plt.imshow(image)
    plt.colorbar()
    plt.show()

    ####################################################

    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=str(val)[:4],
                    loc='center', facecolor='white')
    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)
    plt.show()


def get_next(pos, action, grid):
    """返回某个action的奖励值"""
    i, j = pos
    if (i, j) == A:
        (x, y), r = (4, 1), 10
    elif (i, j) == B:
        (x, y), r = (2, 3), 5
    else:
        r = 0
        x, y = i + action[0], j + action[1]
        if x<0 or x >= ROW:
            x, r = i, -1
        if y <0 or y >= COLUMN:
            y, r = j, -1

    return r + gamma*grid[x, y]


def m_1(grid):
    num = 0
    while True:
        change = 0
        new_grid = np.zeros((ROW, COLUMN))
        for i in range(ROW):
            for j in range(COLUMN):
                score = []
                for action in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    score.append(get_next((i, j), action, grid))
                change += abs(max(score) - grid[i, j])
                # grid[i, j] = max(score)
                new_grid[i, j] = max(score)

        # 遍历整个网格之后
        num += 1
        grid = new_grid
        print(grid, "\n")
        time.sleep(0.0)
        if change <= 1e-3:
            print(num)
            break
    draw_image(grid)



if __name__ == '__main__':
    # grid = np.random.randn(ROW, COLUMN)
    grid = np.zeros((ROW, COLUMN))
    m_1(grid)


