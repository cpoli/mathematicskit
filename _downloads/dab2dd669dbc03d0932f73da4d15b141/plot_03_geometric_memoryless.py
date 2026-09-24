r"""
The geometric distribution: waiting for the first success
=========================================================

The number of independent trials up to and including the first
success, with success probability :math:`p`, has mean :math:`1/p` and is
memoryless: :math:`P(X > m + n \mid X > m) = P(X > n)`.
"""

# %%
import numpy as np

from mathematicskit.probability import Geometric
from mathematicskit.probability.visualizers.plots import plot_distribution

# %%
# Mean = 1/p
# ------------------------------------------------------------------

geometric = Geometric(p=0.3)
print(f"geometric(p=0.3) mean = {geometric.mean:.4f}, expected 1/p = {1.0 / 0.3:.4f}")
plot_distribution(geometric)

# %%
# Memorylessness
# ------------------------------------------------------------------

m, n = 4, 3
conditional = (1.0 - geometric.cdf(m + n)) / (1.0 - geometric.cdf(m))
print(f"P(X > {m + n} | X > {m}) = {conditional:.4f}, P(X > {n}) = {1.0 - geometric.cdf(n):.4f}")
print("memoryless:", bool(np.isclose(conditional, 1.0 - geometric.cdf(n))))
