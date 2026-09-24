r"""
Chebyshev's inequality: one bound for every distribution
========================================================

:math:`P(|X-\mu| \geq k\sigma) \leq 1/k^2` holds for any distribution
with finite variance. Comparing it with exact tails shows how
conservative a universal bound has to be.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.probability import Exponential, Normal, Poisson, Uniform, chebyshev_tail

# %%
# Exact tails against the bound
# -----------------------------------------------------

k = np.linspace(1.0, 5.0, 200)
fig, ax = plt.subplots()
ax.semilogy(k, 1.0 / k**2, "k--", lw=2, label=r"Chebyshev $1/k^2$")
for name, dist in [("Normal", Normal(0.0, 1.0)), ("Exponential", Exponential(rate=1.0)), ("Uniform", Uniform(0.0, 1.0)), ("Poisson(4)", Poisson(mu=4.0))]:
    result = chebyshev_tail(dist, k)
    ax.semilogy(k, np.maximum(result.exact, 1e-8), label=name)
ax.set_ylim(1e-7, 1.5)
ax.set_xlabel(r"$k$ (standard deviations)")
ax.set_ylabel(r"$P(|X-\mu| \geq k\sigma)$")
ax.legend()
ax.set_title("Bienaymé-Chebyshev inequality")

print(chebyshev_tail(Normal(0.0, 1.0), [2.0, 3.0]))
