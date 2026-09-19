r"""
Binomial, Poisson, and geometric distributions
====================================================

The Poisson distribution as the ``n -> infinity``, ``p -> 0`` limit of
the binomial with ``np`` held fixed, plus the geometric distribution's
memoryless property.
"""

# %%
import numpy as np

from mathematicskit.probability import Binomial, Geometric, Poisson
from mathematicskit.probability.visualizers.plots import plot_distribution

# %%
# Binomial approaches Poisson for large n, small p, fixed np
# ------------------------------------------------------------------

mu = 4.0
binomial_large_n = Binomial(n=2000, p=mu / 2000.0)
poisson = Poisson(mu=mu)
ks = np.arange(0, 15)
print("max |binomial - poisson| pmf difference:", np.max(np.abs(binomial_large_n.pmf(ks) - poisson.pmf(ks))))

plot_distribution(poisson)

# %%
# Geometric distribution: mean = 1/p
# ------------------------------------------------------------------

geometric = Geometric(p=0.3)
print(f"geometric(p=0.3) mean = {geometric.mean}, expected 1/p = {1.0 / 0.3:.4f}")
plot_distribution(geometric)
