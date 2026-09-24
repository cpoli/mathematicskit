r"""
Golden-section search: shrinking the bracket
==================================================

Locates the minimum of a unimodal function on an interval. Each
iteration costs one new function evaluation and shrinks the bracket by
the factor :math:`1/\varphi \approx 0.618`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.optimization import golden_section_search

# %%
# Minimize f(x) = x^2 - sin(4x) on [-1, 2]
# -----------------------------------------------------


def f(x):
    return x**2 - np.sin(4.0 * x)


result = golden_section_search(f, -1.0, 2.0, tol=1e-8)
print(f"minimizer x = {result.x:.8f}, f(x) = {result.fun:.8f}")
print(f"{result.iterations} iterations, {result.nfev} function evaluations")

widths = result.brackets[:, 1] - result.brackets[:, 0]
print(f"successive width ratio: {widths[1] / widths[0]:.6f} (1/phi = {(np.sqrt(5) - 1) / 2:.6f})")

# %%
# The first few brackets, and the width decay
# -----------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
xs = np.linspace(-1.0, 2.0, 400)
ax1.plot(xs, f(xs), color="0.3")
for i, (a, b) in enumerate(result.brackets[:8]):
    ax1.plot([a, b], [-1.2 - 0.12 * i] * 2, "|-", color="tab:blue")
ax1.plot(result.x, result.fun, "o", color="tab:red")
ax1.set_title("Brackets (top to bottom: iterations 0-7)")
ax2.semilogy(widths, "o-", ms=3)
ax2.set_xlabel("iteration")
ax2.set_ylabel("bracket width")
ax2.set_title("Width shrinks by 1/phi per step (Kiefer, 1953)")
fig.tight_layout()
