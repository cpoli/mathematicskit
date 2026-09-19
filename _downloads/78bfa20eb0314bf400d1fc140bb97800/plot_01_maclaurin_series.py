r"""
Maclaurin series and their radius of convergence
=====================================================

Truncated Maclaurin series for exp, sin, cos, log(1+x), and arctan, and
how well they approximate the true function -- including outside their
radius of convergence, where the partial sums stop improving (or
diverge) no matter how many terms are added.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathkit.calculus.systems.taylor_series import estimate_radius_of_convergence, evaluate_series, maclaurin_coefficients
from mathkit.calculus.utils.series_utils import truncation_error

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
