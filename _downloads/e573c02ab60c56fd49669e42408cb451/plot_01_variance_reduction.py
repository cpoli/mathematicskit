r"""
Variance reduction: plain vs. control-variate Monte Carlo
================================================================

Estimates :math:`\int_0^1 e^x\,dx = e - 1` three ways, comparing each
method's standard error at the same sample size.
"""

# %%
import numpy as np

from mathematicskit.probability.systems.monte_carlo import control_variates_integrate, monte_carlo_integrate

# %%
# Plain Monte Carlo
# ------------------------------------------------------------------

plain = monte_carlo_integrate(np.exp, 0.0, 1.0, n=20000, seed=0)
print(f"plain:           estimate={plain.estimate:.5f}, std_error={plain.std_error:.5f}")

# %%
# Control variates: g(x) = x, E[X] = 0.5 under Uniform(0, 1)
# ------------------------------------------------------------------

cv = control_variates_integrate(np.exp, lambda x: x, control_mean=0.5, a=0.0, b=1.0, n=20000, seed=0)
print(f"control variate: estimate={cv.estimate:.5f}, std_error={cv.std_error:.5f}")

print(f"\nexact value e - 1 = {np.e - 1.0:.5f}")
print(f"variance reduction factor: {(plain.std_error / cv.std_error) ** 2:.2f}x")
