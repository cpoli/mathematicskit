r"""
Runge, Heun and Kutta: Runge-Kutta methods
==========================================

Runge (1895), Heun (1900) and Kutta (1901) matched more terms of the
solution's Taylor series without computing any derivatives of :math:`f`.
They did it by sampling the slope at several points inside each step and
combining the samples. Heun's two-stage method is second order, and
Kutta's classical four-stage method

.. math::

    y_{n+1} = y_n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)

is fourth order: halving :math:`h` divides the error by 16.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.integrators import njit, rk4_integrate


@njit
def oscillator(state, t, params):
    return np.array([state[1], -state[0]])


def euler_final(h, n):
    y = np.array([1.0, 0.0])
    for i in range(n):
        y = y + h * oscillator(y, i * h, None)
    return y


def heun_final(h, n):
    y = np.array([1.0, 0.0])
    for i in range(n):
        k1 = oscillator(y, i * h, None)
        k2 = oscillator(y + h * k1, (i + 1) * h, None)
        y = y + 0.5 * h * (k1 + k2)
    return y


# %%
# Error at t = 2 pi for the harmonic oscillator
# ---------------------------------------------

t_end = 2 * np.pi
ns = 2 ** np.arange(4, 12)
hs = t_end / ns
exact = np.array([1.0, 0.0])
errors = {
    "Euler (1 stage)": [np.linalg.norm(euler_final(h, n) - exact) for h, n in zip(hs, ns, strict=False)],
    "Heun (2 stages)": [np.linalg.norm(heun_final(h, n) - exact) for h, n in zip(hs, ns, strict=False)],
    "Kutta RK4 (4 stages)": [
        np.linalg.norm(rk4_integrate(oscillator, np.array([1.0, 0.0]), 0.0, h, n, np.zeros(1))[1][-1] - exact) for h, n in zip(hs, ns, strict=False)
    ],
}

fig, ax = plt.subplots(figsize=(7, 4.5))
for (label, err), order in zip(errors.items(), (1, 2, 4), strict=False):
    ax.loglog(hs, err, "o-", label=label)
    print(f"{label}: error ratio for halved h = {err[-2] / err[-1]:.2f} (expected {2**order})")
ax.set_xlabel("h")
ax.set_ylabel("|error| after one period")
ax.set_title("Runge-Kutta: more stages, higher order")
ax.legend()
fig.tight_layout()

plt.show()
