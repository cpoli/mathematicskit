r"""
Lagrange's four-square theorem
===================================================

Every non-negative integer is a sum of four squares. This script finds
a representation for every :math:`n < 10{,}000` and plots the fewest
squares each :math:`n` actually needs: one for perfect squares, two by
Fermat's criterion, four exactly for :math:`n = 4^k(8m+7)` (Legendre's
three-square theorem), and three otherwise.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import sum_of_four_squares, sum_of_two_squares

# %%
# Four squares always suffice
# -----------------------------------------------------

N = 10000
reps = [sum_of_four_squares(n) for n in range(N)]
print("all representations correct:", all(sum(x * x for x in r) == n for n, r in enumerate(reps)))
for n in (7, 31, 310, 9999):
    a, b, c, d = sum_of_four_squares(n)
    print(f"  {n} = {a}^2 + {b}^2 + {c}^2 + {d}^2")


# %%
# How many squares does each n need?
# -----------------------------------------------------


def squares_needed(n):
    if math.isqrt(n) ** 2 == n:
        return 1
    if sum_of_two_squares(n) is not None:
        return 2
    while n % 4 == 0:
        n //= 4
    return 4 if n % 8 == 7 else 3


needed = np.array([squares_needed(n) for n in range(1, N)])
for k in (1, 2, 3, 4):
    print(f"n < {N} needing exactly {k} squares: {(needed == k).sum()}")

fig, ax = plt.subplots()
ax.bar([1, 2, 3, 4], [(needed == k).sum() for k in (1, 2, 3, 4)], color="tab:blue")
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("fewest squares needed")
ax.set_ylabel(f"count of n < {N}")
ax.set_title("No integer needs more than four squares")
plt.show()
