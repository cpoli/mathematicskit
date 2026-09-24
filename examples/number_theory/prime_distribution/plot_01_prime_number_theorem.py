r"""
The prime number theorem
===================================================

Compares the prime-counting function :math:`\pi(x)` with Gauss's
logarithmic integral :math:`\operatorname{li}(x)` and Legendre-style
:math:`x/\ln x` up to ten million. Both ratios tend to 1, as Hadamard
and de la Vallée Poussin proved in 1896, but :math:`\operatorname{li}(x)`
is far more accurate.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import logarithmic_integral, prime_counting

# %%
# pi(x) against its two approximations
# -----------------------------------------------------

xs = np.unique(np.logspace(2, 7, 60).astype(np.int64))
pi_x = prime_counting(xs)
li_x = logarithmic_integral(xs.astype(float))
x_log_x = xs / np.log(xs)
for x in (10**3, 10**5, 10**7):
    i = np.searchsorted(xs, x)
    print(f"x = {xs[i]:>9}: pi(x) = {pi_x[i]:>7}, li(x) = {li_x[i]:>10.1f}, x/ln x = {x_log_x[i]:>10.1f}")

fig, ax = plt.subplots()
ax.semilogx(xs, li_x / pi_x, label="li(x) / pi(x)")
ax.semilogx(xs, x_log_x / pi_x, label="(x / ln x) / pi(x)")
ax.axhline(1.0, color="0.5", ls=":")
ax.set_xlabel("x")
ax.set_ylabel("ratio")
ax.set_title("Both approximations are asymptotic to pi(x)")
ax.legend()
plt.show()
