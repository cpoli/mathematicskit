r"""
Convolutional codes (1955)
==========================

Elias proposed encoding a bit stream continuously instead of in blocks:
each output pair is the parity of a sliding window of input bits. The
rate-1/2 code with generators :math:`(7, 5)_8` and constraint length 3
spreads every input bit over six output bits, and its *free distance*
-- the lightest nonzero output -- is 5.
"""

# %%
import itertools

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import convolutional_encode

impulse = convolutional_encode([1])
print("impulse response:", impulse.reshape(-1, 2).tolist())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
message = np.array([1, 0, 1, 1, 0, 0, 1, 0])
encoded = convolutional_encode(message).reshape(-1, 2)
ax1.imshow(np.vstack([np.r_[message, 0, 0], encoded[:, 0], encoded[:, 1]]), cmap="Greys", aspect="auto")
ax1.set_yticks([0, 1, 2], ["input", "output 1 (111)", "output 2 (101)"])
ax1.set_xlabel("time step")
ax1.set_title("Encoder: two parities of a 3-bit window")

# %%
# The weight spectrum and the free distance
# -----------------------------------------
# Output weights of every input starting with 1, up to length 8. The
# minimum, 5, is the free distance: two channel errors can always be
# corrected.

weights = [convolutional_encode(m).sum() for L in range(1, 9) for m in itertools.product((0, 1), repeat=L) if m[0]]
values, counts = np.unique(weights, return_counts=True)
ax2.bar(values, counts)
ax2.set_xlabel("output weight")
ax2.set_ylabel("number of inputs")
ax2.set_title(f"Free distance = {min(weights)}")
