r"""
Animated: Conway's Game of Life and the Gosper glider gun
=========================================================

Conway's rules are local and tiny: a dead cell with exactly three live
neighbors is born, a live cell with two or three survives. Yet in 1970
Bill Gosper found a finite pattern that grows forever: a "gun" that
fires a new glider every 30 generations. Each glider is five cells that
rebuild themselves one cell further along the diagonal every four
generations. The live-cell count in the title rises by five per period.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.fractals_chaos import GameOfLife
from mathematicskit.fractals_chaos.visualizers import animate_life

# In Jupyter or JupyterLite, show animations as an HTML/JavaScript player
# (no ffmpeg needed).
plt.rcParams["animation.html"] = "jshtml"

# Gosper's glider gun, as (row, column) offsets.
GUN = [
    (5, 1), (5, 2), (6, 1), (6, 2),
    (5, 11), (6, 11), (7, 11), (4, 12), (8, 12), (3, 13), (9, 13), (3, 14), (9, 14), (6, 15), (4, 16), (8, 16), (5, 17), (6, 17), (7, 17), (6, 18),
    (3, 21), (4, 21), (5, 21), (3, 22), (4, 22), (5, 22), (2, 23), (6, 23), (1, 25), (2, 25), (6, 25), (7, 25),
    (3, 35), (4, 35), (3, 36), (4, 36),
]  # fmt: skip

grid = np.zeros((40, 60), dtype=np.int64)
for row, col in GUN:
    grid[row, col] = 1
history = GameOfLife(grid).run(120)
print("live cells every 30 generations:", history.sum(axis=(1, 2))[::30])

# %%
# The gun at work
# ---------------

anim = animate_life(history)
anim

# %%
plt.show()
