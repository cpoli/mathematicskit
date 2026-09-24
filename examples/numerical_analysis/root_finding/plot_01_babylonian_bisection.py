r"""
The Babylonian square root and bisection
========================================

Old Babylonian tablets give :math:`\sqrt{2}` to about six decimal
places. The rule Heron of Alexandria later wrote down averages a guess
with :math:`a` divided by the guess, :math:`x_{n+1} = \tfrac12(x_n + a/x_n)`.
Bisection needs even less: it halves a bracket :math:`[a, b]` on which
:math:`f` changes sign, relying only on the intermediate value theorem.
This script computes :math:`\sqrt{2}` both ways and shows the bracket
shrinking by exactly half per step, against the Babylonian rule's
doubling of correct digits.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import Bisection, FixedPointIteration, estimate_convergence_order

root_exact = np.sqrt(2.0)

# %%
# The Babylonian (Heron) iteration
# --------------------------------
# :math:`g(x) = \tfrac12(x + 2/x)` has :math:`\sqrt{2}` as its fixed point.
# Starting from the tablet-friendly guess 1.5, six-place accuracy takes
# only three averaging steps.

babylonian = FixedPointIteration(lambda x: 0.5 * (x + 2.0 / x), x0=1.5, tol=1e-15).solve()
for n, x in enumerate(babylonian.history):
    print(f"Babylonian step {n}: x = {x:.15f}   error = {abs(x - root_exact):.1e}")

# %%
# Bisection on :math:`f(x) = x^2 - 2` over :math:`[1, 2]`
# -------------------------------------------------------
# Each step keeps the half of the bracket where :math:`f` changes sign, so
# the bracket width after :math:`n` steps is exactly :math:`2^{-n}`.

bisection = Bisection(lambda x: x**2 - 2.0, 1.0, 2.0, tol=1e-13).solve()
print(f"bisection: root = {bisection.root:.13f} after {bisection.iterations} halvings")
print(f"Babylonian order ~ {estimate_convergence_order(babylonian.history[:-1], root_exact):.2f}")

# %%
# Bracket width vs. error
# -----------------------

fig, ax = plt.subplots(figsize=(7, 4.5))
steps = np.arange(len(bisection.history))
bis_err = np.abs(bisection.history - root_exact)
bab_err = np.abs(babylonian.history - root_exact)
ax.semilogy(steps, 2.0 ** (-steps.astype(float)), "--", color="gray", label=r"bracket width $2^{-n}$")
ax.semilogy(steps, np.where(bis_err == 0, np.nan, bis_err), "o-", ms=3, color="darkorange", label="bisection midpoint")
ax.semilogy(np.arange(len(bab_err)), np.where(bab_err == 0, np.nan, bab_err), "s-", color="seagreen", label="Babylonian average")
ax.set_xlabel("step $n$")
ax.set_ylabel(r"$|x_n - \sqrt{2}|$")
ax.set_title(r"Computing $\sqrt{2}$: halving a bracket vs. Babylonian averaging")
ax.legend()
fig.tight_layout()

plt.show()
