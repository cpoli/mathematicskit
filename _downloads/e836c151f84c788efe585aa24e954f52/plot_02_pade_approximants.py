r"""
Padé approximants beyond the radius of convergence
=====================================================

Henri Padé's 1892 thesis organized rational approximations
:math:`p(x)/q(x)` that match a function's Taylor series as far as
possible. The Taylor series of :math:`\log(1 + x)` diverges for
:math:`x > 1`, but the Padé approximants built from *the same*
coefficients keep converging well beyond it.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import PadeApproximant

# %%
# Taylor polynomial vs. Padé approximant of :math:`\log(1 + x)`
# ----------------------------------------------------------------

N = 8
c = np.array([0.0] + [(-1.0) ** (k + 1) / k for k in range(1, N + 1)])
taylor = np.polynomial.Polynomial(c)
pade = PadeApproximant(c, N // 2, N // 2)

x = np.linspace(0.0, 4.0, 400)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
ax1.plot(x, np.log1p(x), "k", lw=2, label=r"$\log(1 + x)$")
ax1.plot(x, taylor(x), "--", color="steelblue", label=f"Taylor, degree {N}")
ax1.plot(x, pade(x), ":", color="firebrick", lw=2, label=f"Padé [{N // 2}/{N // 2}]")
ax1.axvline(1.0, color="gray", lw=0.8)
ax1.set_ylim(-1, 3)
ax1.set_title("Same 9 Taylor coefficients")
ax1.legend(fontsize=8)

ax2.semilogy(x, np.abs(taylor(x) - np.log1p(x)) + 1e-17, color="steelblue", label="Taylor")
ax2.semilogy(x, np.abs(pade(x) - np.log1p(x)) + 1e-17, color="firebrick", label="Padé")
ax2.axvline(1.0, color="gray", lw=0.8)
ax2.set_title("Absolute error (radius of convergence at $x = 1$)")
ax2.legend(fontsize=8)
fig.tight_layout()

print(f"at x = 3: Taylor error {abs(taylor(3.0) - np.log(4.0)):.2e}, Padé error {abs(pade(3.0) - np.log(4.0)):.2e}")

plt.show()
