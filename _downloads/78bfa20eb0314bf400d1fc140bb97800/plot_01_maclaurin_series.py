r"""
Taylor's theorem: polynomials, remainder bound, and convergence
=================================================================

Taylor's theorem expands a function in powers of :math:`x - x_0` built
from its derivatives at one point, with a Lagrange remainder
:math:`|R_n(x)| \le M|x-x_0|^{n+1}/(n+1)!`. This example draws the
Taylor (Maclaurin, :math:`x_0 = 0`) polynomials of :math:`\sin` closing
in on the function, checks the true error against
:func:`~mathematicskit.calculus.taylor_remainder_bound`, and shows what
happens outside a series' radius of convergence, where adding terms no
longer helps.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.calculus.systems.taylor_series import estimate_radius_of_convergence, evaluate_series, maclaurin_coefficients, taylor_remainder_bound
from mathematicskit.calculus.utils.series_utils import truncation_error

# %%
# Taylor polynomials of sin approach the function
# -----------------------------------------------

sin_coeffs = maclaurin_coefficients("sin", order=15)
grid = np.linspace(-2 * np.pi, 2 * np.pi, 400)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, np.sin(grid), "k", lw=2, label="sin x")
for degree in (1, 3, 5, 9, 15):
    ax.plot(grid, evaluate_series(sin_coeffs[: degree + 1], grid), "--", label=f"degree {degree}")
ax.set_ylim(-2, 2)
ax.set_title("Taylor polynomials of sin about x = 0")
ax.legend(fontsize=8)
fig.tight_layout()

# %%
# The Lagrange remainder bounds the true error
# --------------------------------------------
#
# Every derivative of :math:`\sin` is bounded by :math:`M = 1`, so
# Taylor's theorem guarantees :math:`|R_n(x)| \le |x|^{n+1}/(n+1)!`.

x = 1.5
for degree in (1, 3, 5, 7, 9):
    actual = abs(np.sin(x) - evaluate_series(sin_coeffs[: degree + 1], x))
    bound = taylor_remainder_bound(1.0, degree, x)
    print(f"degree {degree}: |error|={actual:.3e} <= bound {bound:.3e}: {actual <= bound}")

# %%
# Convergence inside the radius: log(1+x) at x=0.5 (R=1)
# ------------------------------------------------------------

coeffs = maclaurin_coefficients("log1p", order=40)
errors = truncation_error(np.log1p, coeffs, 0.5)

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(np.arange(len(errors)), np.maximum(errors, 1e-16), "o-")
ax.set_xlabel("truncation degree")
ax.set_ylabel("|error|")
ax.set_title("log(1+x) series error at x=0.5 (inside R=1)")
fig.tight_layout()

# %%
# Failure to converge outside the radius: log(1+x) at x=1.5 (R=1)
# ------------------------------------------------------------------------

errors_outside = truncation_error(np.log1p, coeffs, 1.5)
print("error at degree 10 (x=1.5, outside R=1):", errors_outside[10])
print("error at degree 39 (x=1.5, outside R=1):", errors_outside[39])
print("(error should be growing, not shrinking, since |x| > R)")

# %%
# Radius-of-convergence estimates for each series
# ------------------------------------------------------

for name in ("exp", "sin", "cos", "log1p", "geometric", "arctan"):
    c = maclaurin_coefficients(name, order=30)
    print(f"{name:>10s}: estimated R ~ {estimate_radius_of_convergence(c):.4f}")

print("value check exp(1) via series:", evaluate_series(maclaurin_coefficients("exp", 20), 1.0), "vs.", np.e)

plt.show()
