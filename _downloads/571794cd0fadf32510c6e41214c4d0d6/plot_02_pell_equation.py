r"""
Pell's equation x^2 - Dy^2 = 1
===================================================

Fermat's 1657 challenge: solve :math:`x^2 - Dy^2 = 1` in integers. The
continued fraction of :math:`\sqrt D` always yields the fundamental
solution (Lagrange, 1768), but its size is wildly erratic -- for
:math:`D = 61` it is :math:`x = 1766319049`, while :math:`D = 60` needs
only :math:`x = 31`. This script reproduces Fermat's :math:`D = 61`
case, generates further solutions from the fundamental one, and plots
the fundamental :math:`x` for every non-square :math:`D \le 120`.
"""

# %%
import math

import matplotlib.pyplot as plt

from mathematicskit.number_theory import solve_pell_equation

# %%
# Fermat's challenge: D = 61
# -----------------------------------------------------

pell = solve_pell_equation(61)
print(f"fundamental solution of x^2 - 61y^2 = 1: x={pell.x}, y={pell.y}")
print(f"check: x^2 - 61y^2 = {pell.x**2 - 61 * pell.y**2}")

# %%
# Every solution is a power of the fundamental one
# -----------------------------------------------------
# :math:`x_k + y_k\sqrt D = (x_1 + y_1\sqrt D)^k`.

x, y = pell.x, pell.y
for k in range(2, 4):
    x, y = x * pell.x + 61 * y * pell.y, x * pell.y + y * pell.x
    print(f"  k={k}: x has {len(str(x))} digits, x^2 - 61y^2 = {x * x - 61 * y * y}")

# %%
# The fundamental solution's size jumps around with D
# -----------------------------------------------------

ds = [d for d in range(2, 121) if math.isqrt(d) ** 2 != d]
xs = [solve_pell_equation(d).x for d in ds]

fig, ax = plt.subplots()
ax.semilogy(ds, xs, "o", ms=4)
ax.semilogy([61], [pell.x], "o", ms=9, mfc="none", mec="red", label="D = 61 (Fermat, 1657)")
ax.set_xlabel("D")
ax.set_ylabel("fundamental x")
ax.set_title("Smallest solution of Pell's equation x^2 - Dy^2 = 1")
ax.legend()
plt.show()
