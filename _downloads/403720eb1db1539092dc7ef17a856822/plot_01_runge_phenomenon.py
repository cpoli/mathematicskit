r"""
The Runge phenomenon: why node placement matters
====================================================

Interpolating Runge's classic example, :math:`f(x) = 1/(1+25x^2)`, at
equally spaced nodes on :math:`[-1, 1]` makes the interpolation error
*diverge* near the endpoints as the polynomial degree grows -- a famous
1901 counterexample to the intuition that more nodes always means a
better fit. Clustering nodes near the endpoints instead, at the Chebyshev
points, eliminates the divergence entirely.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathkit.numerical_analysis import ChebyshevInterpolant, LagrangeInterpolant, chebyshev_nodes, runge_function
from mathkit.numerical_analysis.systems.chebyshev import runge_phenomenon_errors
from mathkit.numerical_analysis.utils.error_analysis import lebesgue_constant

# %%
# Degree-20 interpolant: equally spaced vs. Chebyshev nodes
# --------------------------------------------------------------

n = 21
x_equal = np.linspace(-1.0, 1.0, n)
p_equal = LagrangeInterpolant(x_equal, runge_function(x_equal))
p_cheb = ChebyshevInterpolant(runge_function, n=n)

x_fine = np.linspace(-1.0, 1.0, 400)

fig1, ax1 = plt.subplots(figsize=(6.5, 4.5))
ax1.plot(x_fine, runge_function(x_fine), color="black", lw=1.5, label="f(x) = 1/(1+25x^2)")
ax1.plot(x_fine, p_equal.evaluate(x_fine), color="firebrick", label="equally spaced (n=21)")
ax1.plot(x_fine, p_cheb.evaluate(x_fine), color="steelblue", label="Chebyshev (n=21)")
ax1.set_ylim(-2, 2)
ax1.set_title("Equally spaced nodes overshoot wildly near the edges")
ax1.legend(fontsize=8)
fig1.tight_layout()

# %%
# Error vs. degree: divergence vs. convergence
# -----------------------------------------------

degrees = [5, 10, 15, 20, 25, 30]
equal_errors, chebyshev_errors = runge_phenomenon_errors(degrees)
for d, ee, ce in zip(degrees, equal_errors, chebyshev_errors):
    print(f"degree={d:2d}  equally-spaced max err={ee:.4f}  Chebyshev max err={ce:.6f}")

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.semilogy(degrees, equal_errors, "o-", color="firebrick", label="equally spaced")
ax2.semilogy(degrees, chebyshev_errors, "o-", color="steelblue", label="Chebyshev")
ax2.set_xlabel("polynomial degree")
ax2.set_ylabel("max |error|")
ax2.set_title("Equally spaced error diverges; Chebyshev error shrinks")
ax2.legend()
fig2.tight_layout()

# %%
# The mechanism: the Lebesgue constant
# ----------------------------------------
# The Lebesgue constant bounds interpolation error relative to the best
# possible polynomial approximation; it grows exponentially for equally
# spaced nodes but only logarithmically for Chebyshev nodes.

n_vals = [5, 10, 20, 40]
for nv in n_vals:
    lam_equal = lebesgue_constant(np.linspace(-1.0, 1.0, nv))
    lam_cheb = lebesgue_constant(chebyshev_nodes(nv))
    print(f"n={nv:3d}  Lebesgue(equal)={lam_equal:.2e}  Lebesgue(Chebyshev)={lam_cheb:.3f}")

plt.show()
