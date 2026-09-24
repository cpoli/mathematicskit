r"""
Aitken's delta-squared process and Steffensen's method
=========================================================

Aitken's :math:`\Delta^2` process (1926) extrapolates a linearly
convergent sequence from three consecutive terms. Applied to the slowly
converging Leibniz series for :math:`\pi`, one pass gains several digits.
Steffensen's method (1933) applies the same extrapolation *inside* a
fixed-point iteration, turning the linear convergence of
:math:`x_{n+1} = \cos x_n` into quadratic convergence without any
derivative.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import FixedPointIteration, Steffensen, aitken_delta_squared

# %%
# Accelerating the Leibniz series
# ---------------------------------

k = np.arange(30)
partial = 4.0 * np.cumsum((-1.0) ** k / (2 * k + 1))
once = aitken_delta_squared(partial)
twice = aitken_delta_squared(once)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
ax1.semilogy(k, np.abs(partial - np.pi), "o-", ms=3, label="partial sums")
ax1.semilogy(k[2:], np.abs(once - np.pi), "s-", ms=3, label="Aitken once")
ax1.semilogy(k[4:], np.abs(twice - np.pi), "^-", ms=3, label="Aitken twice")
ax1.set_xlabel("$n$")
ax1.set_ylabel(r"error in $\pi$")
ax1.set_title("Leibniz series for $\\pi$")
ax1.legend(fontsize=8)

# %%
# Steffensen vs. plain fixed-point iteration for :math:`x = \cos x`
# --------------------------------------------------------------------

plain = FixedPointIteration(np.cos, x0=1.0, tol=1e-14).solve()
steff = Steffensen(np.cos, x0=1.0, tol=1e-14).solve()
x_star = steff.root

for result, label in ((plain, "fixed-point (linear)"), (steff, "Steffensen (quadratic)")):
    err = np.maximum(np.abs(result.history - x_star), 1e-17)
    ax2.semilogy(err, "o-", ms=3, label=label)
ax2.set_xlabel("iteration $n$")
ax2.set_ylabel(r"$|x_n - x^*|$")
ax2.set_title(r"Fixed point of $\cos x$")
ax2.legend(fontsize=8)
fig.tight_layout()

print(f"30 Leibniz terms: error {abs(partial[-1] - np.pi):.1e}; Aitken twice: {abs(twice[-1] - np.pi):.1e}")
print(f"fixed-point iterations: {plain.iterations}, Steffensen iterations: {steff.iterations}")

plt.show()
