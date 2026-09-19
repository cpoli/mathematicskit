r"""
Approximating pi by continued fractions
==============================================

Expands pi as a continued fraction and shows how quickly its
convergents (355/113 famously accurate to 7 digits) approach the true
value.
"""

# %%
import numpy as np

from mathkit.number_theory import best_rational_approximation, continued_fraction_expansion
from mathkit.number_theory.visualizers.plots import plot_convergent_errors

# %%
# Expand pi and inspect its convergents
# -----------------------------------------------------

result = continued_fraction_expansion(np.pi, max_terms=8)
print("terms:", result.terms)
for p, q in result.convergents:
    print(f"  {p}/{q} = {p / q:.10f}  (error {abs(p / q - np.pi):.2e})")

# %%
# Best approximation under a denominator bound
# -----------------------------------------------------

p, q = best_rational_approximation(np.pi, max_denominator=1000)
print(f"\nbest approximation with denominator <= 1000: {p}/{q}")

# %%
# Plot the convergent errors
# -----------------------------------------------------

plot_convergent_errors(result, x=np.pi)
