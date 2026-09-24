r"""
Remez's algorithm and best uniform approximation
===================================================

Chebyshev showed that the polynomial of degree :math:`n` closest to
:math:`f` in the maximum norm is characterized by an error curve that
reaches its peak magnitude :math:`n + 2` times with alternating sign.
Evgeny Remez's 1934 exchange algorithm finds it by repeatedly levelling
the error on a set of reference points. This script computes the best
degree-6 approximation to :math:`e^x \sin(3x)` on :math:`[-1, 1]` and
compares its error with Chebyshev interpolation of the same degree.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import ChebyshevInterpolant, remez_minimax

# %%
# The equioscillating error curve
# ----------------------------------

f = lambda x: np.exp(x) * np.sin(3.0 * x)
n = 6
best = remez_minimax(f, n)
cheb = ChebyshevInterpolant(f, n + 1)  # degree n through n + 1 Chebyshev points

x = np.linspace(-1.0, 1.0, 1000)
err_best = f(x) - best.evaluate(x)
err_cheb = f(x) - cheb(x)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, err_cheb, color="steelblue", label="Chebyshev interpolant")
ax.plot(x, err_best, color="firebrick", label="Remez minimax")
ax.scatter(best.reference, f(best.reference) - best.evaluate(best.reference), color="firebrick", zorder=3, label="reference points")
ax.axhline(best.max_error, color="gray", ls="--", lw=0.8)
ax.axhline(-best.max_error, color="gray", ls="--", lw=0.8)
ax.set_title(f"Degree {n}: the minimax error equioscillates at {n + 2} points")
ax.legend(fontsize=8)
fig.tight_layout()

print(f"Remez: {best.iterations} exchanges, max error {best.max_error:.3e}")
print(f"Chebyshev interpolation: max error {np.max(np.abs(err_cheb)):.3e}")

plt.show()
