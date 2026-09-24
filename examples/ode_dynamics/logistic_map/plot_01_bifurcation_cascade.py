r"""
Feigenbaum's universal constant in the period-doubling cascade
==============================================================

As the parameter :math:`r` grows, the logistic map
:math:`x \mapsto rx(1-x)` doubles its period again and again -- 1, 2, 4,
8, ... -- at parameter values whose spacing shrinks by a constant
factor, Feigenbaum's :math:`\delta \approx 4.6692`. Universality means
the *same* :math:`\delta` appears for any smooth one-humped map. This
script measures :math:`\delta` for the logistic map and for the
unrelated sine map :math:`x \mapsto r\sin(\pi x)` and shows both
sequences of ratios converging to the one constant.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, newton

from mathematicskit.ode_dynamics.systems.logistic_map import bifurcation_diagram, estimate_feigenbaum_delta
from mathematicskit.ode_dynamics.visualizers.plots import plot_bifurcation_diagram

FEIGENBAUM_DELTA = 4.669201609


def logistic(x, r):
    return r * x * (1.0 - x)


def sine_map(x, r):
    return r * np.sin(np.pi * x)


# %%
# Library estimate from the first bifurcations
# --------------------------------------------
# :func:`estimate_feigenbaum_delta` locates the first three period
# doublings of the logistic map by scanning :math:`r` -- a rough first
# look at the constant.

print(f"estimate_feigenbaum_delta() -> {estimate_feigenbaum_delta():.4f}  (true delta = {FEIGENBAUM_DELTA:.4f})")

# %%
# Precise ratios from superstable cycles
# --------------------------------------
# Between consecutive doublings sits a *superstable* parameter
# :math:`R_n` where the map's maximum :math:`x_c = 1/2` lies on the
# period-:math:`2^n` cycle, i.e. :math:`f_R^{2^n}(x_c) = x_c`. These
# roots are easy to find accurately, and their spacings shrink by the
# same ratio: :math:`\delta_n = (R_{n} - R_{n-1}) / (R_{n+1} - R_n) \to
# \delta`.


def superstable_parameters(f, R0, R1_bracket, n_max=8):
    def g(R, n):
        x = 0.5
        for _ in range(2**n):
            x = f(x, R)
        return x - 0.5

    R = [R0, brentq(g, *R1_bracket, args=(1,), xtol=1e-14)]
    for n in range(2, n_max + 1):
        delta_guess = 4.67 if n == 2 else (R[-2] - R[-3]) / (R[-1] - R[-2])
        guess = R[-1] + (R[-1] - R[-2]) / delta_guess
        R.append(newton(g, guess, args=(n,), tol=1e-14, maxiter=200))
    R = np.array(R)
    return R, (R[1:-1] - R[:-2]) / (R[2:] - R[1:-1])


R_log, delta_log = superstable_parameters(logistic, 2.0, (3.0, 3.44))
R_sin, delta_sin = superstable_parameters(sine_map, 0.5, (0.72, 0.83))
print(" n   delta_n (logistic)   delta_n (sine map)")
for n, (dl, ds) in enumerate(zip(delta_log, delta_sin), start=1):
    print(f"{n:2d}   {dl:17.5f}   {ds:17.5f}")

# %%
# Two different maps, one cascade, one constant
# ---------------------------------------------

r_values = np.linspace(2.8, 4.0, 1500)
r_plot, x_plot = bifurcation_diagram(r_values, n_transient=500, n_keep=100)

s_values = np.linspace(0.65, 1.0, 1500)
x = np.full_like(s_values, 0.5)
for _ in range(500):
    x = sine_map(x, s_values)
s_orbit = np.empty((100, s_values.size))
for k in range(100):
    x = sine_map(x, s_values)
    s_orbit[k] = x

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
plot_bifurcation_diagram(r_plot, x_plot, ax=axes[0])
axes[0].set_title(r"Logistic map $rx(1-x)$")
axes[1].plot(np.tile(s_values, 100), s_orbit.ravel(), ",", color="black", alpha=0.5)
axes[1].set_xlabel("r")
axes[1].set_ylabel("x (after transient)")
axes[1].set_title(r"Sine map $r\sin(\pi x)$")
for ax, R in ((axes[0], R_log), (axes[1], R_sin)):
    for i, Rn in enumerate(R[1:5]):
        ax.axvline(Rn, color="tab:red", lw=0.6, alpha=0.7, label=r"superstable $R_n$" if i == 0 else None)
    ax.legend(loc="lower left", fontsize=8)

n = np.arange(1, delta_log.size + 1)
axes[2].plot(n, delta_log, "o-", label="logistic map")
axes[2].plot(n, delta_sin, "s--", label="sine map")
axes[2].axhline(FEIGENBAUM_DELTA, color="k", lw=1, ls=":", label=r"Feigenbaum $\delta = 4.6692$")
axes[2].set_xlabel("n")
axes[2].set_ylabel(r"$\delta_n = (R_n - R_{n-1}) / (R_{n+1} - R_n)$")
axes[2].set_title(r"Universality: both ratios $\to \delta$")
axes[2].legend(fontsize=8)
fig.tight_layout()

plt.show()
