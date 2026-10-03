r"""
Hamming codes (1950)
====================

Tired of weekend jobs on the Bell Labs relay computers stopping at
every error, Hamming designed codes that correct them. The (7,4) code
adds three parity bits so that the *syndrome* -- which checks fail --
spells out the position of a single flipped bit in binary. Over a
binary symmetric channel it trades rate 4/7 for a much lower bit error
rate when errors are rare.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import bsc_transmit, hamming_decode, hamming_encode, hamming_parity_check_matrix

H = hamming_parity_check_matrix(3)
word = hamming_encode([1, 0, 1, 1])
corrupted = word.copy()
corrupted[4] ^= 1
syndrome = (H.astype(int) @ corrupted) % 2
print("codeword ", word, "\nreceived ", corrupted)
print(f"syndrome {syndrome[::-1]} = position {int(syndrome @ [1, 2, 4])} -> decoded {hamming_decode(corrupted)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.imshow(H, cmap="Greys", vmin=0, vmax=1.4)
for (i, j), v in np.ndenumerate(H):
    ax1.text(j, i, v, ha="center", va="center", color="w" if v else "k")
ax1.set_xticks(range(7), [str(j + 1) for j in range(7)])
ax1.set_yticks(range(3), ["bit 0", "bit 1", "bit 2"])
ax1.set_xlabel("position (column = position in binary)")
ax1.set_title("Parity-check matrix of the (7,4) code")

# %%
# Bit error rate on a binary symmetric channel
# --------------------------------------------

rng = np.random.default_rng(0)
ps = np.logspace(-3, -0.7, 12)
for r in (3, 4, 5):
    k = 2**r - r - 1
    message = rng.integers(0, 2, k * 4000)
    ber = [np.mean(hamming_decode(bsc_transmit(hamming_encode(message, r), p, seed=rng), r) != message) for p in ps]
    ax2.loglog(ps, ber, "o-", label=f"Hamming ({2**r - 1},{k})")
ax2.loglog(ps, ps, "k--", label="uncoded")
ax2.set_xlabel("crossover probability p")
ax2.set_ylabel("decoded bit error rate")
ax2.legend()
