r"""
The complex-step derivative: no cancellation
==================================================

Compares the central difference with the complex-step derivative
Im f(x + ih)/h as the step shrinks. The central difference is ruined by
subtractive cancellation, while the complex step stays accurate to
machine precision at any tiny step.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.calculus import central_difference, complex_step_derivative

# %%
# Error against step size
# -----------------------------------------------------


def f(x):
    return np.exp(x) / np.sqrt(np.sin(x) ** 3 + np.cos(x) ** 3)  # Squire and Trapp's test function


x0 = 1.5
reference = complex_step_derivative(f, x0, h=1e-200)
hs = np.logspace(-20, -1, 60)
central = [max(abs(central_difference(f, x0, h=h) - reference), 1e-17) for h in hs]
complex_step = [max(abs(complex_step_derivative(f, x0, h=h) - reference), 1e-17) for h in hs]
print(f"f'({x0}) = {reference:.15f}")

fig, ax = plt.subplots()
ax.loglog(hs, central, "o-", ms=3, label="central difference")
ax.loglog(hs, complex_step, "s-", ms=3, label="complex step")
ax.set_xlabel("step h")
ax.set_ylabel("|error| (floored at 1e-17)")
ax.legend()
ax.set_title("Complex-step differentiation (Lyness and Moler, 1967)")
