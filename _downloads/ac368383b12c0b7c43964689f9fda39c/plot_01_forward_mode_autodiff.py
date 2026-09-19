r"""
Forward-mode automatic differentiation via dual numbers
============================================================

Dual numbers compute a function's value and its exact derivative
together, with no finite-difference step-size tradeoff between
truncation error (large ``h``) and floating-point cancellation (small
``h``).
"""

# %%
import math

import numpy as np

from mathkit.calculus.systems.dual_numbers import derivative
from mathkit.calculus.systems.finite_differences import central_difference

# %%
# Exact derivative, compared against central differences at various h
# ----------------------------------------------------------------------------

f_dual = lambda x: (x * x).sin()  # d/dx sin(x^2) = 2x cos(x^2)
f_plain = lambda x: math.sin(x * x)
x0 = 1.3
exact = 2.0 * x0 * math.cos(x0**2)

dual_result = derivative(f_dual, x0)
print(f"dual-number derivative: {dual_result:.15f} (exact: {exact:.15f}, error {abs(dual_result - exact):.2e})")

for h in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11):
    cd = central_difference(f_plain, x0, h=h)
    print(f"central diff h={h:.0e}: {cd:.15f}, error {abs(cd - exact):.2e}")

# %%
# Composite functions and the chain rule fall out automatically
# ----------------------------------------------------------------------

g = lambda x: (x.exp() / (x + 1.0)).log()
x1 = 2.0
grad = derivative(g, x1)
print(f"d/dx log(exp(x)/(x+1)) at x={x1}: {grad:.10f}")

# %%
# A NumPy sanity check against a nearby closed-form derivative
# ----------------------------------------------------------------------

numpy_gradient_estimate = np.gradient([f_plain(x0 - 1e-6), f_plain(x0), f_plain(x0 + 1e-6)], 1e-6)[1]
print("close to numpy finite-difference estimate:", bool(np.isclose(dual_result, numpy_gradient_estimate, atol=1e-3)))
