r"""
Fermat's adequality: maxima from a difference quotient
============================================================

Fermat found the maximum of a function by setting a difference quotient
(f(x+h) - f(x))/h equal to zero and then letting h vanish. This example
locates the maximum of Fermat's own problem, the largest product
x(a - x), and shows the O(h) error of the forward difference.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from mathematicskit.calculus import forward_difference

# %%
# Fermat's problem: split a segment of length a to maximize x(a - x)
# ---------------------------------------------------------------------

a = 10.0


def area(x):
    return x * (a - x)


for h in (1.0, 0.1, 0.01):
    x_star = brentq(lambda x: forward_difference(area, x, h=h), 0.0, a)
    print(f"h = {h:5.2f}: difference quotient vanishes at x = {x_star:.4f}")
print(f"letting h -> 0 gives x = a/2 = {a / 2}")

# %%
# The forward difference is first-order accurate
# -----------------------------------------------------

hs = np.logspace(-8, -1, 30)
errors = [abs(forward_difference(np.sin, 1.0, h=h) - np.cos(1.0)) for h in hs]
fig, ax = plt.subplots()
ax.loglog(hs, errors, "o-", label="forward difference")
ax.loglog(hs, 0.5 * np.sin(1.0) * hs, "--", label=r"$\frac{h}{2}|f''|$ truncation")
ax.set_xlabel("step h")
ax.set_ylabel("error in f'(1) for f = sin")
ax.legend()
ax.set_title("Truncation error falls as h, until rounding takes over")
