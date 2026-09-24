r"""
Bessel functions of the first and second kind
====================================================

Plots J_0, J_1 (regular at the origin) and Y_0 (singular at the
origin) over a range of x, and finds J_0's first zero.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import bessel_first_kind, bessel_second_kind

# %%
# Evaluate over a range
# -----------------------------------------------------

x = np.linspace(0.1, 10.0, 200)
j0 = bessel_first_kind(0.0, x)
j1 = bessel_first_kind(1.0, x)
y0 = bessel_second_kind(0.0, x)

fig, ax = plt.subplots()
ax.plot(x, j0, label=r"$J_0(x)$")
ax.plot(x, j1, label=r"$J_1(x)$")
ax.plot(x, y0, label=r"$Y_0(x)$")
ax.axhline(0, color="gray", lw=0.5)
ax.set_ylim(-1.5, 1.1)
ax.set_xlabel("x")
ax.set_title("Bessel functions of the first and second kind")
ax.legend()

print(f"J_0(0.1) = {j0[0]:.4f}, Y_0(0.1) = {y0[0]:.4f} (Y_0 diverges toward x=0)")

# %%
# The first zero of J_0 (approximately 2.4048)
# -----------------------------------------------------

sign_changes = np.where(np.diff(np.sign(j0)) != 0)[0]
first_zero = x[sign_changes[0]]
print(f"\nfirst zero of J_0 near x = {first_zero:.4f} (known value: 2.4048)")
