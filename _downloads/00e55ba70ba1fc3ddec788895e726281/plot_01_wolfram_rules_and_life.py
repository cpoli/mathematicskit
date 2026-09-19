r"""
Elementary cellular automata and Conway's Game of Life
=============================================================

Rule 30 (chaotic) and rule 90 (a discrete Sierpinski triangle) from a
single seed cell, plus a glider gun-free Game of Life demonstration
using the classic "glider" spaceship.
"""

# %%
import numpy as np

from mathkit.fractals_chaos import ElementaryCA, GameOfLife
from mathkit.fractals_chaos.visualizers.plots import plot_ca_spacetime

# %%
# Rule 30: chaotic, used as a pseudo-random number generator by Wolfram's Mathematica
# ---------------------------------------------------------------------------------------

rule30 = ElementaryCA(rule=30, width=101)
history30 = rule30.run(60)
plot_ca_spacetime(history30)

# %%
# Rule 90: XOR of the two neighbors, a discrete Sierpinski triangle
# ------------------------------------------------------------------------

rule90 = ElementaryCA(rule=90, width=101)
history90 = rule90.run(60)
plot_ca_spacetime(history90)

# %%
# Conway's Game of Life: the "glider" spaceship
# -----------------------------------------------------
# A glider translates by (1, 1) every 4 generations.

grid = np.zeros((20, 20), dtype=np.int64)
glider = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
for i, j in glider:
    grid[i, j] = 1

life = GameOfLife(grid)
history_life = life.run(4)
print("glider translated by one cell diagonally after 4 generations:", bool(np.array_equal(history_life[4, 1:, 1:], history_life[0, :-1, :-1])))
