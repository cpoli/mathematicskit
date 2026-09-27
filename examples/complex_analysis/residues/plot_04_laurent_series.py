r"""
Laurent series: expansions with negative powers in an annulus
=============================================================

Pierre Alphonse Laurent (1843, after Weierstrass in 1841) expanded a
function holomorphic in an annulus :math:`r < |z - a| < R` as
:math:`\sum_{n=-\infty}^{\infty} a_n (z-a)^n`, with every coefficient a
contour integral :math:`a_n = \frac{1}{2\pi i}\oint f(z)(z-a)^{-n-1}dz`.
For :math:`f(z) = 1/((z-1)(z-2))` the expansion depends on the annulus:
in :math:`1 < |z| < 2` it is :math:`a_n = -2^{-n-1}` for :math:`n \ge 0`
and :math:`a_n = -1` for :math:`n < 0`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import circle_contour, contour_integral


def f(z):
    return 1.0 / ((z - 1.0) * (z - 2.0))


def laurent_coefficient(n, radius):
    return contour_integral(lambda z: f(z) * z ** (-n - 1), circle_contour(0.0, radius)) / (2j * np.pi)


# %%
# Coefficients by contour integration in three annuli
# ---------------------------------------------------

orders = np.arange(-5, 6)
closed_form = np.where(orders >= 0, -(2.0 ** (-orders - 1.0)), -1.0)
fig, ax = plt.subplots()
for radius, label in ((0.5, "|z| < 1 (Taylor)"), (1.5, "1 < |z| < 2"), (3.0, "|z| > 2")):
    coefficients = np.array([laurent_coefficient(n, radius) for n in orders]).real
    ax.plot(orders, coefficients, "o-", label=label)
    if radius == 1.5:
        print(f"max |numeric - closed form| in 1 < |z| < 2: {np.max(np.abs(coefficients - closed_form)):.1e}")
ax.set_xlabel("n")
ax.set_ylabel("$a_n$")
ax.set_title("Laurent coefficients of 1/((z-1)(z-2)) depend on the annulus")
ax.legend()

# %%
# The truncated series converges only inside its annulus
# ------------------------------------------------------

z = 1.5 * np.exp(0.7j)
for terms in (5, 10, 20, 40):
    n = np.arange(-terms, terms + 1)
    a = np.where(n >= 0, -(2.0 ** (-n - 1.0)), -1.0)
    print(f"{terms:>2} terms each side: |partial sum - f(z)| = {abs(np.sum(a * z**n) - f(z)):.2e}")

plt.show()
