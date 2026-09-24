r"""
Stirling's formula and its asymptotic series
===============================================

Shows that sqrt(2 pi n) (n/e)^n approximates n! with relative error
about 1/(12n), and that Stirling's full series for log Gamma(x) is
asymptotic: at small x, adding terms first helps and then hurts.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import log_gamma_function, stirling_factorial, stirling_log_gamma

# %%
# The leading term: n! ~ sqrt(2 pi n) (n/e)^n
# -----------------------------------------------------

for n in (1, 5, 10, 50, 100):
    ratio = math.factorial(n) / stirling_factorial(float(n))
    print(f"n = {n:3d}: n!/Stirling = {ratio:.6f}, 1 + 1/(12n) = {1 + 1 / (12 * n):.6f}")

# %%
# Error of the truncated series versus number of terms
# -----------------------------------------------------

terms = np.arange(0, 16)
fig, ax = plt.subplots()
for x in (1.0, 2.0, 5.0):
    errors = [abs(stirling_log_gamma(x, terms=int(k)) - log_gamma_function(x)) for k in terms]
    ax.semilogy(terms, errors, "o-", label=f"x = {x:g}")
    print(f"x = {x:g}: best with {int(np.argmin(errors))} terms, error {min(errors):.2e}")
ax.set_xlabel("number of Bernoulli correction terms")
ax.set_ylabel(r"$|\,\mathrm{series} - \ln\Gamma(x)\,|$")
ax.set_title("Stirling's series is asymptotic, not convergent")
ax.legend()
