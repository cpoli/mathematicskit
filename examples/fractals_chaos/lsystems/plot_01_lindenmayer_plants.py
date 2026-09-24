r"""
Lindenmayer systems: from algae to plants
===============================================

Runs Lindenmayer's 1968 algae system, whose string lengths are Fibonacci
numbers, and draws a Koch curve and a branching plant with turtle
graphics.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.fractals_chaos import lsystem, turtle_path

# %%
# Lindenmayer's algae
# -----------------------------------------------------

for n in range(7):
    word = lsystem("A", {"A": "AB", "B": "A"}, n)
    print(f"n = {n}: {word} (length {len(word)})")

# %%
# A Koch curve and a fractal plant
# -----------------------------------------------------

koch = turtle_path(lsystem("F", {"F": "F+F--F+F"}, 4), 60, heading=0)
plant = turtle_path(lsystem("X", {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"}, 5), 25, heading=65)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, lines, title in zip(axes, (koch, plant), ("Koch curve: F -> F+F--F+F", "plant: X -> F+[[X]-X]-F[-FX]+X")):
    for line in lines:
        ax.plot(*line.T, color="darkgreen" if ax is axes[1] else "tab:blue", lw=0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
