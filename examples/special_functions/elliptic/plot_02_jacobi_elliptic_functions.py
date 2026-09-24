r"""
Jacobi elliptic functions and the pendulum
================================================

Plots sn, cn, dn for m = 0.9, showing their real period 4K(m), and uses
them to write the exact motion of a large-amplitude pendulum.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import complete_elliptic_integral_first_kind, jacobi_elliptic_functions

# %%
# sn, cn, dn over two periods
# -----------------------------------------------------

m = 0.9
K = complete_elliptic_integral_first_kind(m)
u = np.linspace(0, 8 * K, 800)
r = jacobi_elliptic_functions(u, m)

fig, ax = plt.subplots()
ax.plot(u / K, r.sn, label="sn")
ax.plot(u / K, r.cn, label="cn")
ax.plot(u / K, r.dn, label="dn")
ax.set_xlabel("u / K(m)")
ax.set_title(f"Jacobi elliptic functions, m = {m}")
ax.legend()
print(f"K({m}) = {K:.6f}; sn(K) = {jacobi_elliptic_functions(K, m).sn:.12f}")
print(f"max |sn^2 + cn^2 - 1| = {np.max(np.abs(r.sn**2 + r.cn**2 - 1)):.1e}")

# %%
# Exact pendulum motion: sin(theta/2) = k sn(K - t, k^2) with omega_0 = 1
# ------------------------------------------------------------------------

theta0 = np.radians(150.0)
k = np.sin(theta0 / 2)
K_pend = complete_elliptic_integral_first_kind(k**2)
t = np.linspace(0, 4 * K_pend, 400)
theta = 2 * np.arcsin(k * jacobi_elliptic_functions(K_pend - t, k**2).sn)
print(f"\n150 degree pendulum: period 4K = {4 * K_pend:.4f}, small-angle period 2 pi = {2 * np.pi:.4f}")

fig, ax = plt.subplots()
ax.plot(t, np.degrees(theta), label="exact (Jacobi sn)")
ax.plot(t, np.degrees(theta0 * np.cos(t)), "--", label="small-angle approximation")
ax.set_xlabel(r"$\omega_0 t$")
ax.set_ylabel("angle (degrees)")
ax.legend()
