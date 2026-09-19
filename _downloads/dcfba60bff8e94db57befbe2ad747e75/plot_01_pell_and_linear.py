r"""
Pell's equation and a linear Diophantine equation
========================================================

Finds the fundamental solution of :math:`x^2 - 61y^2 = 1` (famous for
its surprisingly large fundamental solution) and solves a linear
Diophantine equation with its full solution family.
"""

# %%
from mathkit.number_theory import solve_linear_diophantine, solve_pell_equation

# %%
# Pell's equation for D = 61
# -----------------------------------------------------

pell = solve_pell_equation(61)
print(f"fundamental solution of x^2 - 61y^2 = 1: x={pell.x}, y={pell.y}")
print(f"check: {pell.x}^2 - 61*{pell.y}^2 = {pell.x**2 - 61 * pell.y**2}")

# %%
# A linear Diophantine equation: 12x + 18y = 30
# -----------------------------------------------------

linear = solve_linear_diophantine(12, 18, 30)
print(f"\nparticular solution: x={linear.x0}, y={linear.y0} (gcd={linear.gcd})")
for k in range(-2, 3):
    x = linear.x0 + k * linear.x_step
    y = linear.y0 - k * linear.y_step
    print(f"  k={k}: (x,y)=({x},{y}), 12x+18y={12 * x + 18 * y}")
