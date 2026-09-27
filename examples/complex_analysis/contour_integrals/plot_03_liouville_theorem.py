r"""
Liouville's theorem: a bounded entire function is constant
==========================================================

Cauchy's integral formula on the circle :math:`|z| = R` gives the
estimate :math:`|f'(0)| \le M(R)/R`, where :math:`M(R)` is the maximum
of :math:`|f|` on the circle. If :math:`f` is entire and bounded, letting
:math:`R \to \infty` forces :math:`f' \equiv 0`. So every nonconstant
entire function, such as :math:`\sin z`, is unbounded. Applied to
:math:`1/p(z)`, the theorem proves the fundamental theorem of algebra.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import cauchy_integral_formula, circle_contour

# %%
# Cauchy's estimate :math:`|f'(0)| \le M(R)/R`
# ---------------------------------------------

radii = np.geomspace(0.5, 20, 12)
functions = {r"$\sin z$": np.sin, r"$z^3 - z$": lambda z: z**3 - z, r"$e^{-z^2}$": lambda z: np.exp(-(z**2))}
fig, ax = plt.subplots()
for name, f in functions.items():
    max_modulus = np.array([np.max(np.abs(f(circle_contour(0, R).points(2000)))) for R in radii])
    ax.loglog(radii, max_modulus, "o-", label=f"M(R) for {name}")
    derivative = abs(cauchy_integral_formula(f, circle_contour(0, 1.0), 0.0, n=1))
    print(f"{name:12s} |f'(0)| = {derivative:.4f},  min over R of M(R)/R = {np.min(max_modulus / radii):.4f}")
ax.set_xlabel("R")
ax.set_ylabel("max |f| on |z| = R")
ax.set_title("Nonconstant entire functions are unbounded")
ax.legend()

# %%
# The fundamental theorem of algebra
# ----------------------------------
# If p had no roots, 1/p would be entire; it also tends to 0 as :math:`|z|` grows,
# so it would be bounded, hence constant -- impossible for a nonconstant p.

p = lambda z: z**4 + z + 1
for R in (1, 10, 100):
    print(f"R = {R:3d}: max |1/p| on |z| = R: {np.max(np.abs(1 / p(circle_contour(0, R).points(2000)))):.2e}")
print("roots of p (where 1/p fails to be entire):", np.round(np.roots([1, 0, 0, 1, 1]), 4))
