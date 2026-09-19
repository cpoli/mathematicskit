r"""
A small production-planning linear program
=================================================

A classic LP: maximize profit from two products given shared resource
constraints. ``scipy.optimize.linprog`` minimizes by convention, so the
objective is negated.
"""

# %%
import numpy as np

from mathkit.optimization import linear_program

# %%
# Problem setup
# -----------------------------------------------------
# Two products A, B. Profit per unit: $3 (A), $5 (B). Each unit of A
# needs 1 hour of labor and 2 units of material; each unit of B needs 2
# hours of labor and 1 unit of material. 40 labor-hours and 30 material
# units available per day.

c = np.array([-3.0, -5.0])  # negate: linprog minimizes
a_ub = np.array([[1.0, 2.0], [2.0, 1.0]])
b_ub = np.array([40.0, 30.0])

result = linear_program(c, a_ub=a_ub, b_ub=b_ub)
print(f"produce {result.x[0]:.2f} units of A and {result.x[1]:.2f} units of B")
print(f"maximum daily profit: ${-result.fun:.2f}")
