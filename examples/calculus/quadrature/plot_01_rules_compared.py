r"""
Quadrature rules compared: cost vs. accuracy
=================================================

Trapezoidal (O(h^2)), Simpson's (O(h^4)), and Gauss-Legendre (exact up to
degree 2n-1) converge at very different rates as the number of function
evaluations grows; adaptive Simpson instead spends more evaluations only
where the integrand needs them.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.calculus.systems.quadrature import AdaptiveQuadrature, GaussianQuadrature, SimpsonsRule, TrapezoidalRule

# %%
# Convergence vs. number of evaluations
# --------------------------------------------

f = np.sin
a, b = 0.0, np.pi
exact = 2.0

ns_even = [2, 4, 8, 16, 32, 64]
err_trap = [abs(TrapezoidalRule(n=n).integrate(f, a, b).value - exact) for n in ns_even]
err_simp = [abs(SimpsonsRule(n=n).integrate(f, a, b).value - exact) for n in ns_even]
err_gauss = [abs(GaussianQuadrature(n=n).integrate(f, a, b).value - exact) for n in range(1, 8)]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(ns_even, np.maximum(err_trap, 1e-16), "o-", label="trapezoidal")
ax.loglog(ns_even, np.maximum(err_simp, 1e-16), "s-", label="simpson")
ax.loglog(range(1, 8), np.maximum(err_gauss, 1e-16), "^-", label="gauss-legendre")
ax.set_xlabel("n (evaluations)")
ax.set_ylabel("|error|")
ax.set_title("Gauss-Legendre needs far fewer evaluations for smooth integrands")
ax.legend()
fig.tight_layout()

# %%
# Adaptive quadrature on a sharply peaked function
# --------------------------------------------------------

peaked = lambda x: 1.0 / (1.0 + 1000.0 * (x - 0.5) ** 2)
result = AdaptiveQuadrature(tol=1e-8).integrate(peaked, 0.0, 1.0)
exact_peaked = (2.0 * np.arctan(np.sqrt(1000.0) * 0.5)) / np.sqrt(1000.0)
print(f"adaptive: value={result.value:.10f}, evaluations={result.n_evaluations}, |error|={abs(result.value - exact_peaked):.2e}")

plt.show()
