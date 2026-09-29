r"""
Forsythe's quadratic formula: catastrophic cancellation
=======================================================

George Forsythe asked in 1966, "How do you solve a quadratic equation?"
The school formula :math:`x = (-b \pm \sqrt{b^2 - 4ac})/(2a)` fails in
floating point when :math:`b^2 \gg |4ac|`: for the root with the
smaller magnitude, :math:`-b` and :math:`\sqrt{b^2 - 4ac}` nearly cancel.
The subtraction itself is exact, but it exposes the rounding error that
:math:`\sqrt{b^2 - 4ac}` already carries. By the loss-of-precision
theorem, about :math:`-\log_2|1 - y/x|` of the 53 significant bits are
lost when computing :math:`x - y`. The cure is to compute only the root
that involves no cancellation and get the other from :math:`x_1 x_2 = c/a`.
"""

# %%
from decimal import Decimal, getcontext

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import cancellation_bits_lost, quadratic_roots

# %%
# The small root of x^2 + b x + 1
# -------------------------------
#
# For large :math:`b` the small root is :math:`\approx -1/b`. The
# reference value is computed with 50 significant digits by
# :mod:`decimal`, and the stable formula reproduces it to machine
# precision.

getcontext().prec = 50
bs = np.logspace(1, 9, 33)
naive_err, stable_err, predicted = [], [], []
for b in bs:
    exact = float((-Decimal(b) + (Decimal(b) ** 2 - 4).sqrt()) / 2)
    naive_err.append(max(abs(quadratic_roots(1.0, b, 1.0, stable=False)[1] - exact) / abs(exact), 1e-17))
    stable_err.append(max(abs(quadratic_roots(1.0, b, 1.0)[1] - exact) / abs(exact), 1e-17))
    lost = cancellation_bits_lost(-b, -np.sqrt(b * b - 4.0))
    predicted.append(min(2.0 ** (lost - 53), 1.0))  # all 53 bits gone once b^2 - 4 rounds to b^2

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
ax0.loglog(bs, naive_err, "o-", label="textbook formula")
ax0.loglog(bs, stable_err, "s-", label="stable formula (x2 = c / q)")
ax0.loglog(bs, predicted, "k--", label=r"$u \cdot 2^{\mathrm{bits\ lost}}$")
ax0.set_xlabel("b")
ax0.set_ylabel("relative error of the small root")
ax0.set_title(r"$x^2 + bx + 1 = 0$")
ax0.legend(fontsize=8)

for b in (1e4, 1e8):
    print(f"b = {b:.0e}: textbook {quadratic_roots(1.0, b, 1.0, stable=False)[1]:.16e}, stable {quadratic_roots(1.0, b, 1.0)[1]:.16e}")

# %%
# Rewriting removes the subtraction: 1 - cos x
# --------------------------------------------
#
# The same cure applies to any formula that subtracts nearly equal
# numbers: :math:`1 - \cos x = 2\sin^2(x/2)` has no subtraction at all.

x = np.logspace(-9, -1, 400)
naive = (1.0 - np.cos(x)) / x**2
stable = 2.0 * np.sin(x / 2.0) ** 2 / x**2
exact = 0.5 - x**2 / 24.0 + x**4 / 720.0 - x**6 / 40320.0 + x**8 / 3628800.0  # Taylor series
ax1.loglog(x, np.abs(naive - exact) / exact + 1e-17, label=r"$(1 - \cos x)/x^2$")
ax1.loglog(x, np.abs(stable - exact) / exact + 1e-17, label=r"$2\sin^2(x/2)/x^2$")
ax1.set_xlabel("x")
ax1.set_ylabel("relative error")
ax1.set_title("Every digit lost once x < 1e-8")
ax1.legend(fontsize=8)
fig.tight_layout()

plt.show()
