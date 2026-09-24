r"""
The error function and the normal distribution
===================================================

Recovers the 68-95-99.7 rule from erf, and shows why erfc is needed in
the far tail, where 1 - erf(x) rounds to zero.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import complementary_error_function, error_function

# %%
# The 68-95-99.7 rule
# -----------------------------------------------------

for k in (1, 2, 3):
    print(f"P(|X - mu| < {k} sigma) = erf({k}/sqrt 2) = {error_function(k / math.sqrt(2)):.6f}")

# %%
# erf and erfc
# -----------------------------------------------------

x = np.linspace(-3, 3, 300)
fig, ax = plt.subplots()
ax.plot(x, error_function(x), label="erf(x)")
ax.plot(x, complementary_error_function(x), label="erfc(x)")
ax.set_xlabel("x")
ax.legend()
ax.set_title("The error function")

# %%
# The far tail
# -----------------------------------------------------

for x0 in (3.0, 6.0, 9.0):
    print(f"x = {x0:g}: 1 - erf(x) = {1 - error_function(x0):.3e}, erfc(x) = {complementary_error_function(x0):.3e}")
