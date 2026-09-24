r"""
The Airy functions Ai and Bi
=================================

Plots Ai and Bi across the turning point x = 0 where y'' = xy switches
from oscillation to exponential behaviour, and checks the Wronskian
Ai Bi' - Ai' Bi = 1/pi.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import airy_functions

# %%
# Oscillation on the left, growth or decay on the right
# ------------------------------------------------------

x = np.linspace(-15, 3, 18001)
r = airy_functions(x)
fig, ax = plt.subplots()
ax.plot(x, r.ai, label="Ai(x)")
ax.plot(x, r.bi, label="Bi(x)")
ax.axvline(0, color="gray", ls=":")
ax.set_ylim(-0.6, 1.2)
ax.set_xlabel("x")
ax.set_title(r"Solutions of $y'' = xy$")
ax.legend()

# %%
# Wronskian and the first zeros of Ai
# -----------------------------------------------------

wronskian = r.ai * r.bi_prime - r.ai_prime * r.bi
print(f"Wronskian range: [{wronskian.min():.12f}, {wronskian.max():.12f}], 1/pi = {1 / np.pi:.12f}")
zeros = x[:-1][np.diff(np.sign(r.ai)) != 0][::-1][:3]
print(f"first zeros of Ai near {np.round(zeros, 3)} (known: -2.338, -4.088, -5.521)")
