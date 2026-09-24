r"""
Kummer's confluent hypergeometric function
================================================

Checks Kummer's transformation M(a, b, z) = e^z M(b - a, b, -z) and
shows 2F1(a, c; b; z/c) merging into 1F1(a; b; z) as c grows.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import confluent_hypergeometric_1f1, hypergeometric_2f1

# %%
# Kummer's transformation
# -----------------------------------------------------

a, b = 0.6, 2.3
z = np.linspace(-4, 4, 9)
lhs = confluent_hypergeometric_1f1(a, b, z)
rhs = np.exp(z) * confluent_hypergeometric_1f1(b - a, b, -z)
print(f"max |M(a,b,z) - e^z M(b-a,b,-z)| = {np.max(np.abs(lhs - rhs)):.1e}")

# %%
# The confluence of two singular points
# -----------------------------------------------------

a, b = 0.5, 1.5
z = np.linspace(-3, 3, 300)
fig, ax = plt.subplots()
for c in (2.0, 5.0, 20.0):
    ax.plot(z, hypergeometric_2f1(a, c, b, z / c), "--", label=f"${{}}_2F_1(a, {c:g}; b; z/{c:g})$")
ax.plot(z, confluent_hypergeometric_1f1(a, b, z), "k", lw=2, label=r"${}_1F_1(a; b; z)$")
ax.set_xlabel("z")
ax.set_title(r"$a = 1/2$, $b = 3/2$: the confluent limit")
ax.legend()
