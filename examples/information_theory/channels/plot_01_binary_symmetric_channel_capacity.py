r"""
Capacity of the binary symmetric channel (1948)
===============================================

A binary symmetric channel flips each bit with probability :math:`p`.
Shannon showed its capacity is :math:`C = 1 - h(p)` bits per use, the
largest mutual information over input distributions, reached by a fair
coin. At :math:`p = 0.11` half of every bit survives; at
:math:`p = 1/2` the output is independent of the input.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import bsc_capacity, bsc_transmit

p = np.linspace(0, 1, 401)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(p, bsc_capacity(p))
for pi in (0.11, 0.5):
    ax1.plot(pi, bsc_capacity(pi), "ko")
    ax1.annotate(f"C({pi}) = {bsc_capacity(pi):.2f}", (pi, bsc_capacity(pi)), xytext=(5, 8), textcoords="offset points")
ax1.set_xlabel("crossover probability p")
ax1.set_ylabel("capacity (bits per use)")
ax1.set_title("C = 1 - h(p)")

# %%
# What the channel does to a picture
# ----------------------------------

yy, xx = np.mgrid[-1:1:64j, -1:1:64j]
image = ((xx**2 + yy**2 < 0.6) ^ (np.abs(xx) < 0.15)).astype(np.uint8)
noisy = bsc_transmit(image, 0.11, seed=0)
ax2.imshow(np.hstack([image, np.ones((64, 4)), noisy]), cmap="Greys", interpolation="nearest")
ax2.set_title(f"sent | received through BSC(0.11), C = {bsc_capacity(0.11):.2f}")
ax2.axis("off")
