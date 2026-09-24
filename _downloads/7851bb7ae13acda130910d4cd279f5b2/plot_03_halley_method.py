r"""
Halley's method: cubic convergence
=====================================

Edmond Halley's 1694 iteration uses the second derivative as well as the
first, stepping to the root of an osculating hyperbola instead of a
tangent line. Near a simple root it *triples* the number of correct
digits per step, against Newton's doubling. This script solves
:math:`x^3 = 2` with both methods from the same starting guess and
compares their error histories and empirical convergence orders.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import Halley, NewtonRaphson, estimate_convergence_order

# %%
# Newton vs. Halley on :math:`f(x) = x^3 - 2`
# ----------------------------------------------

f = lambda x: x**3 - 2.0
fp = lambda x: 3.0 * x**2
fpp = lambda x: 6.0 * x
root = 2.0 ** (1.0 / 3.0)

newton = NewtonRaphson(f, fp, x0=3.0, tol=1e-15).solve()
halley = Halley(f, fp, fpp, x0=3.0, tol=1e-15).solve()

fig, ax = plt.subplots(figsize=(7, 4.5))
for result, color, label in ((newton, "steelblue", "Newton (order 2)"), (halley, "firebrick", "Halley (order 3)")):
    err = np.abs(result.history - root)
    err = np.where(err == 0.0, np.finfo(float).eps * root, err)
    ax.semilogy(err, "o-", color=color, label=label)
ax.set_xlabel("iteration $n$")
ax.set_ylabel(r"$|x_n - 2^{1/3}|$")
ax.set_title("Halley's method converges in fewer steps")
ax.legend()
fig.tight_layout()

# %%
# Empirical convergence orders
# -------------------------------

print(f"Newton: {newton.iterations} iterations, order ~ {estimate_convergence_order(newton.history[:-1], root):.2f}")
print(f"Halley: {halley.iterations} iterations, order ~ {estimate_convergence_order(halley.history[:-1], root):.2f}")

plt.show()
