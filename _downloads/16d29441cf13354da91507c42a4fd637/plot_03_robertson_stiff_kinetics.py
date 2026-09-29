r"""
Robertson's stiff chemical kinetics
===================================

Robertson (1966) proposed three-species autocatalytic reactions

.. math::

    y_1' = -0.04\,y_1 + 10^4\,y_2 y_3,\quad
    y_2' = 0.04\,y_1 - 10^4\,y_2 y_3 - 3\times10^7\,y_2^2,\quad
    y_3' = 3\times10^7\,y_2^2

whose rate constants span nine orders of magnitude. It became the
standard stiff test problem. The intermediate :math:`y_2` jumps to a
quasi-steady value of about :math:`3.6\times10^{-5}` within about
:math:`10^{-4}` time units, then evolves slowly for over :math:`10^{5}`
units. Mass is conserved, :math:`y_1 + y_2 + y_3 = 1`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.integrators import dopri5_integrate, njit, stiff_integrate


@njit
def robertson(state, t, params):
    k1, k2, k3 = params
    y1, y2, y3 = state
    out = np.empty(3)
    out[0] = -k1 * y1 + k3 * y2 * y3
    out[1] = k1 * y1 - k3 * y2 * y3 - k2 * y2**2
    out[2] = k2 * y2**2
    return out


params = np.array([0.04, 3e7, 1e4])
state0 = np.array([1.0, 0.0, 0.0])

# %%
# Eleven decades of time with an implicit solver
# ----------------------------------------------

ts, ys = stiff_integrate(robertson, state0, 0.0, 1e5, params, rtol=1e-6, atol=1e-10)
print(f"implicit solver: {len(ts) - 1} steps to t = 1e5, max |y1+y2+y3-1| = {np.max(np.abs(ys.sum(axis=1) - 1)):.1e}")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
for i, label in enumerate(["$y_1$", r"$10^4\,y_2$", "$y_3$"]):
    ax0.semilogx(ts[1:], ys[1:, i] * (1e4 if i == 1 else 1.0), label=label)
ax0.set_xlabel("t")
ax0.set_title("Robertson kinetics, t up to 1e5")
ax0.legend()

# %%
# An explicit solver's step is pinned by the fast reaction
# --------------------------------------------------------

ts_dp, _ = dopri5_integrate(robertson, state0, 0.0, 40.0, 1e-6, params, rtol=1e-6, atol=1e-10)
ts_im, _ = stiff_integrate(robertson, state0, 0.0, 40.0, params, rtol=1e-6, atol=1e-10)
print(f"to t = 40: explicit Dormand-Prince {len(ts_dp) - 1} steps, implicit {len(ts_im) - 1} steps")
ax1.loglog(ts_dp[1:], np.diff(ts_dp), label=f"Dormand-Prince ({len(ts_dp) - 1} steps)")
ax1.loglog(ts_im[1:], np.diff(ts_im), label=f"implicit ({len(ts_im) - 1} steps)")
ax1.set_xlabel("t")
ax1.set_ylabel(r"accepted $\Delta t$")
ax1.set_title("Step size up to t = 40")
ax1.legend()
fig.tight_layout()

plt.show()
