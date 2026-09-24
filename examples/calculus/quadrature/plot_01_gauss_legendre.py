r"""
Gaussian quadrature: optimal nodes are exact to degree 2n-1
=============================================================

Gauss let a quadrature rule choose its sample points as well as its
weights. The optimal :math:`n` nodes turn out to be the roots of the
Legendre polynomial :math:`P_n`, and the resulting rule integrates every
polynomial of degree up to :math:`2n-1` exactly -- about twice the degree
an equally spaced rule with the same number of points manages. This
example shows the nodes and weights from
:func:`~mathematicskit.calculus.legendre_nodes_and_weights`, checks the
degree of exactness of
:class:`~mathematicskit.calculus.GaussianQuadrature`, and compares its
convergence with the equally spaced trapezoidal rule.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import legendre

from mathematicskit.calculus import GaussianQuadrature, TrapezoidalRule, legendre_nodes_and_weights

# %%
# Nodes are the roots of the Legendre polynomial
# ----------------------------------------------

n = 5
nodes, weights = legendre_nodes_and_weights(n)
grid = np.linspace(-1, 1, 400)
p_n = legendre.Legendre.basis(n)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, p_n(grid), "k", label=f"Legendre $P_{n}$")
ax.axhline(0, color="gray", lw=0.5)
ax.stem(nodes, weights, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="Gauss nodes (height = weight)")
ax.set_title(f"{n}-point Gauss-Legendre rule on [-1, 1]")
ax.legend()
fig.tight_layout()

print("nodes:  ", np.round(nodes, 6))
print("weights:", np.round(weights, 6))
print("max |P_n(node)|:", float(np.max(np.abs(p_n(nodes)))))

# %%
# Exact for every polynomial of degree up to 2n - 1
# -------------------------------------------------
#
# With :math:`n = 5` nodes, :math:`x^k` over :math:`[0, 1]` is integrated
# to round-off for :math:`k \le 9` and first fails at :math:`k = 10`.

rule = GaussianQuadrature(n=n)
for k in range(2 * n + 2):
    err = abs(rule.integrate(lambda x, k=k: x**k, 0.0, 1.0).value - 1.0 / (k + 1))
    print(f"degree {k:2d}: error = {err:.2e}{'  <- exactness lost' if k == 2 * n else ''}")

# %%
# Far fewer evaluations than an equally spaced rule
# -------------------------------------------------

f, a, b, exact = np.exp, 0.0, 1.0, np.e - 1.0
ns = np.arange(1, 11)
err_gauss = [abs(GaussianQuadrature(n=int(k)).integrate(f, a, b).value - exact) for k in ns]
err_trap = [abs(TrapezoidalRule(n=int(k) - 1).integrate(f, a, b).value - exact) if k > 1 else np.nan for k in ns]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.semilogy(ns, np.maximum(err_gauss, 1e-17), "o-", label="Gauss-Legendre")
ax.semilogy(ns, err_trap, "s-", label="trapezoidal (equally spaced)")
ax.set_xlabel("function evaluations")
ax.set_ylabel("|error|")
ax.set_title("Integral of exp over [0, 1]")
ax.legend()
fig.tight_layout()

plt.show()
