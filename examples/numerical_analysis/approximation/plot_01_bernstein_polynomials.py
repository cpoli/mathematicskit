r"""
Bernstein polynomials and the Weierstrass theorem
=====================================================

Weierstrass proved in 1885 that every continuous function on a closed
interval is a uniform limit of polynomials. Sergei Bernstein's 1912
proof builds the polynomials explicitly: average :math:`f(k/n)` with
binomial weights. This script approximates a function with a kink,
:math:`|x - 1/2|`, and shows the uniform error shrinking, slowly, as the
degree grows.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import bernstein_polynomial

# %%
# Bernstein approximants of increasing degree
# ----------------------------------------------

f = lambda t: np.abs(t - 0.5)
x = np.linspace(0.0, 1.0, 501)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
ax1.plot(x, f(x), "k", lw=2, label="$|x - 1/2|$")
for n in (4, 16, 64, 256):
    ax1.plot(x, bernstein_polynomial(f, n, x), label=f"$B_{{{n}}}f$")
ax1.set_title("Bernstein polynomials")
ax1.legend(fontsize=8)

# %%
# Uniform error against degree
# ------------------------------

degrees = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
errors = np.array([np.max(np.abs(bernstein_polynomial(f, n, x) - f(x))) for n in degrees])
ax2.loglog(degrees, errors, "o-", label="max error")
ax2.loglog(degrees, 0.4 / np.sqrt(degrees), "--", color="gray", label=r"$\propto n^{-1/2}$")
ax2.set_xlabel("degree $n$")
ax2.set_title("Convergence is uniform but slow")
ax2.legend(fontsize=8)
fig.tight_layout()

for n, e in zip(degrees, errors):
    print(f"n = {n:4d}: max |B_n f - f| = {e:.4f}")

plt.show()
