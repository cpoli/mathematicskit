r"""
The noisy-channel coding theorem (1948)
=======================================

Shannon proved that every rate below capacity can be sent with error
probability as small as desired, and no rate above it can. His proof
used *random* codes: draw :math:`2^{nR}` codewords at random and decode
to the nearest one. Here the rate is fixed at :math:`R = 1/4` and the
block length grows: below capacity the error falls toward zero, above
capacity it rises toward one.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import bsc_capacity, random_code_error_rate

rate = 0.25
ns = np.arange(4, 41, 4)
fig, ax = plt.subplots(figsize=(7, 4.5))
for p in (0.03, 0.08, 0.3, 0.4):
    errors = [random_code_error_rate(int(n), rate, p, trials=300, seed=int(n)) for n in ns]
    side = "R < C" if rate < bsc_capacity(p) else "R > C"
    ax.plot(ns, errors, "o-", label=f"p = {p}, C = {bsc_capacity(p):.2f} ({side})")
ax.set_xlabel("block length n")
ax.set_ylabel("block error probability")
ax.set_title("Random codes of rate 1/4 with nearest-codeword decoding")
ax.legend()
