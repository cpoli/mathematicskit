r"""
Least squares: normal equations vs. QR
===========================================

Solving the normal equations ``A^T A x = A^T b`` is simple but squares
the condition number of ``A``; solving via ``A = Q R`` instead avoids
ever forming ``A^T A``, so the effective conditioning stays at
``kappa(A)`` rather than ``kappa(A)^2``.
"""

# %%
import numpy as np

from mathkit.linalg import condition_number_2norm, least_squares_normal_equations, least_squares_qr

# %%
# Same fit, two condition numbers
# -------------------------------------

rng = np.random.default_rng(6)
A = rng.uniform(-1, 1, size=(30, 5))
x_true = np.array([1.0, -2.0, 0.5, 3.0, -1.5])
b = A @ x_true + rng.normal(0.0, 1e-6, 30)

result_normal = least_squares_normal_equations(A, b)
result_qr = least_squares_qr(A, b)

print("normal-equations coefficients:", np.round(result_normal.coefficients, 6))
print("QR coefficients:              ", np.round(result_qr.coefficients, 6))
print(f"kappa(A) via QR path:          {result_qr.condition_number:.4e}")
print(f"kappa(A^T A) via normal path:  {result_normal.condition_number:.4e}")
print(f"ratio (should be ~kappa(A)):    {result_normal.condition_number / result_qr.condition_number:.4e}")
print(f"direct kappa_2(A):              {condition_number_2norm(A):.4e}")
