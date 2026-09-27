r"""
Dahlquist's A-stability: backward Euler vs RK4
==============================================

Applied to the test equation :math:`y' = \lambda y`, a one-step method
multiplies :math:`y` by a stability function :math:`R(z)`, with
:math:`z = \lambda h`. Dahlquist (1963) called a method *A-stable* if
:math:`|R(z)| \le 1` on the whole left half-plane, so that no decaying
mode can ever be amplified, whatever the step size. Explicit methods
never qualify, since their :math:`R` is a polynomial and grows without
bound. RK4's region reaches only :math:`z \approx -2.785` on the real
axis. Backward Euler's :math:`R(z) = 1/(1 - z)` is A-stable. Dahlquist
also proved that no A-stable linear multistep method can exceed order 2
(his "second barrier").
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from mathematicskit.integrators import implicit_euler_integrate, rk4_integrate

# %%
# Stability regions in the complex z-plane
# ----------------------------------------

x, y = np.meshgrid(np.linspace(-5, 3, 400), np.linspace(-4, 4, 400))
z = x + 1j * y
regions = {
    "RK4": np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24),
    "backward Euler": np.abs(1 / (1 - z)),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (name, amp) in zip(axes[:2], regions.items(), strict=False):
    ax.contourf(x, y, amp <= 1, levels=[0.5, 1.5], colors=["tab:blue"], alpha=0.35)
    ax.contour(x, y, amp, levels=[1.0], colors="tab:blue")
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    ax.set_title(f"{name}: shaded where |R(z)| <= 1")


# %%
# Consequence on a stiff equation, lambda * h = 30
# ------------------------------------------------
#
# The Prothero-Robinson equation :math:`y' = -\lambda(y - \cos t) - \sin t`
# has the smooth solution :math:`y = \cos t`, but with :math:`\lambda = 3000`
# and :math:`h = 0.01`, :math:`z = -30` lies far outside RK4's region.


@njit
def prothero_robinson(state, t, params):
    return -params[0] * (state - np.cos(t)) - np.sin(t)


params = np.array([3000.0])
ts, ys_ie = implicit_euler_integrate(prothero_robinson, np.array([1.0]), 0.0, 1e-2, 40, params)
_, ys_rk4 = rk4_integrate(prothero_robinson, np.array([1.0]), 0.0, 1e-2, 40, params)
axes[2].semilogy(ts, np.abs(ys_rk4[:, 0] - np.cos(ts)) + 1e-16, "o-", ms=3, label="RK4")
axes[2].semilogy(ts, np.abs(ys_ie[:, 0] - np.cos(ts)) + 1e-16, "s-", ms=3, label="backward Euler")
axes[2].set_xlabel("t")
axes[2].set_ylabel("|error|")
axes[2].set_title("z = -30: only the A-stable method survives")
axes[2].legend()
fig.tight_layout()

print(f"final error: RK4 {abs(ys_rk4[-1, 0] - np.cos(ts[-1])):.2e}, backward Euler {abs(ys_ie[-1, 0] - np.cos(ts[-1])):.2e}")

plt.show()
