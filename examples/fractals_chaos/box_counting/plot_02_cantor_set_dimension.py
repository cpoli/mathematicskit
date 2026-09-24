r"""
Cantor's middle-thirds set and its dimension
==================================================

Starts from :math:`[0, 1]` and repeatedly removes the open middle third
of every remaining interval. After :math:`n` steps there are
:math:`2^n` intervals of total length :math:`(2/3)^n \to 0`, yet the
limit set is uncountable. Box counting recovers its fractional
dimension :math:`\log 2/\log 3 \approx 0.631`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.fractals_chaos import box_counting_dimension, chaos_game
from mathematicskit.fractals_chaos.visualizers.plots import plot_box_counting


def cantor_intervals(n):
    """Left endpoints and common length of the 2**n stage-n intervals."""
    lefts = np.array([0.0])
    for k in range(1, n + 1):
        lefts = np.concatenate([lefts, lefts + 2.0 / 3**k])
    return np.sort(lefts), 3.0**-n


# %%
# Removing middle thirds
# ----------------------

fig, ax = plt.subplots(figsize=(8, 3.5))
for n in range(7):
    lefts, width = cantor_intervals(n)
    ax.broken_barh([(x, width) for x in lefts], (-n - 0.3, 0.6), color="black")
    print(f"stage {n}: {len(lefts):3d} intervals, total length {len(lefts) * width:.4f}")
ax.set_yticks(-np.arange(7), [f"n = {n}" for n in range(7)])
ax.set_xlim(0, 1)
ax.set_title("Cantor set construction")

# %%
# Box-counting dimension
# ----------------------
#
# The two maps :math:`x \mapsto x/3` and :math:`x \mapsto x/3 + 2/3` send
# the Cantor set onto its two halves, so the chaos game samples it.
# Boxes of size :math:`3^{-k}` meet :math:`2^k` pieces of the set, so the
# slope of :math:`\log N` against :math:`\log(1/\text{size})` tends to
# :math:`\log 2/\log 3`.

third = np.diag([1 / 3, 1 / 3])
points = chaos_game([(third, np.array([0.0, 0.0]), 0.5), (third, np.array([2 / 3, 0.0]), 0.5)], 100000, seed=0)
result = box_counting_dimension(points)
print(f"estimated dimension: {result.dimension:.4f}")
print(f"exact value log(2)/log(3): {np.log(2.0) / np.log(3.0):.4f}")
plot_box_counting(result)
