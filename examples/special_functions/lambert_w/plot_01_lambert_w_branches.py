r"""
The two real branches of the Lambert W function
====================================================

Plots W_0 and W_{-1}, the two real inverses of w e^w, which meet at the
branch point z = -1/e, and uses W to solve x^x = 27.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import lambert_w

# %%
# Both real branches
# -----------------------------------------------------

z0 = np.linspace(-1 / math.e + 1e-9, 6.0, 400)
zm1 = np.linspace(-1 / math.e + 1e-9, -1e-4, 400)

fig, ax = plt.subplots()
ax.plot(z0, lambert_w(z0), label=r"$W_0$ (principal)")
ax.plot(zm1, lambert_w(zm1, branch=-1), label=r"$W_{-1}$")
ax.plot(-1 / math.e, -1, "ko", label=r"branch point $(-1/e, -1)$")
ax.set_ylim(-6, 2)
ax.set_xlabel("z")
ax.set_ylabel("W(z)")
ax.set_title(r"Solutions of $we^w = z$")
ax.legend()

# %%
# The omega constant and a transcendental equation
# -----------------------------------------------------

omega = lambert_w(1.0)
print(f"Omega = W(1) = {omega:.12f}, exp(-Omega) = {math.exp(-omega):.12f}")

y = 27.0
x = math.log(y) / lambert_w(math.log(y))
print(f"x^x = {y:g} is solved by x = ln y / W(ln y) = {x:.12f}")
