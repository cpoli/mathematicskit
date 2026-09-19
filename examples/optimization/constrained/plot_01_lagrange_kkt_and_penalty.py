r"""
Lagrange multipliers, KKT verification, and the penalty method
=====================================================================

Minimizes :math:`x^2+y^2` subject to :math:`x+y=1`: the closed-form
solution is :math:`(0.5, 0.5)` with multiplier :math:`\lambda=-1`. Solved
three ways -- directly via the Lagrange stationarity system, verified
via the KKT conditions, and approached indirectly via the quadratic
penalty method.
"""

# %%
import numpy as np

from mathkit.optimization import PenaltyMethod, lagrange_stationary_point, verify_kkt

f = lambda z: z[0] ** 2 + z[1] ** 2
grad_f = lambda z: np.array([2.0 * z[0], 2.0 * z[1]])
h = lambda z: np.array([z[0] + z[1] - 1.0])
grad_h = lambda z: np.array([[1.0, 1.0]])

# %%
# Solve the Lagrange stationarity system directly
# -----------------------------------------------------

lagrange_result = lagrange_stationary_point(grad_f, h, grad_h, x0=np.array([0.0, 0.0]))
print(f"x* = {lagrange_result.x}, lambda* = {lagrange_result.multipliers}")

# %%
# Verify the KKT conditions at that point
# -----------------------------------------------------

kkt = verify_kkt(lagrange_result.x, grad_f, h=h, grad_h=grad_h, eq_multipliers=lagrange_result.multipliers)
print(f"KKT satisfied: {kkt.satisfied}, stationarity residual: {kkt.stationarity_residual:.2e}")

# %%
# Approach the same point via the penalty method
# -----------------------------------------------------

penalty_result = PenaltyMethod(mu0=1.0, mu_factor=10.0, n_outer=8).minimize(f, grad_f, np.array([0.0, 0.0]), h=h, grad_h=grad_h)
print(f"penalty method: x = {penalty_result.x}, mu schedule = {penalty_result.extra['mu_history']}")
