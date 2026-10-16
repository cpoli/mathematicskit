r"""
Gallager's low-density parity-check codes (1962-1963)
=====================================================

Gallager's thesis proposed codes defined by a *sparse* random
parity-check matrix, decoded iteratively: flip the bits that sit in
the most failed checks, and repeat. Too costly for the computers of
the time, LDPC codes were forgotten until MacKay and Neal rediscovered
them in 1996; they now operate close to capacity in Wi-Fi, 5G, and
satellite television.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import bit_flip_decode, bsc_capacity, bsc_transmit, gallager_ldpc_matrix

H = gallager_ldpc_matrix(96, column_weight=3, row_weight=6, seed=0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.spy(H, markersize=2)
ax1.set_title(f"(3,6)-regular Gallager matrix, {H.shape[0]}x{H.shape[1]}, rate 1/2")

# %%
# Block error rate against block length
# -------------------------------------
# The all-zero word is a codeword of every linear code, so it can stand
# in for any message.

rng = np.random.default_rng(1)
ps = np.linspace(0.005, 0.07, 9)
for n in (96, 288, 768):
    H = gallager_ldpc_matrix(n, seed=n)
    trials = 80
    fails = [sum(bit_flip_decode(H, bsc_transmit(np.zeros(n), p, seed=rng)).codeword.any() for _ in range(trials)) / trials for p in ps]
    ax2.plot(ps, fails, "o-", label=f"n = {n}")
ax2.axvline(0.11, color="k", ls="--", lw=1)
ax2.text(0.107, 0.5, f"rate 1/2 = capacity at p = 0.11 (C = {bsc_capacity(0.11):.2f})", rotation=90, ha="right", va="center", fontsize=8)
ax2.set_xlim(0, 0.12)
ax2.set_xlabel("crossover probability p")
ax2.set_ylabel("block error rate")
ax2.set_title("Hard-decision bit flipping fails well before capacity")
ax2.legend()
