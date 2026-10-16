r"""
Elias' universal code for the integers (1975)
=============================================

To send a positive integer of unknown size, Elias' gamma code writes
:math:`\lfloor\log_2 n\rfloor` zeros, then :math:`n` in binary: the
zeros announce the length. It is a prefix code for all integers at
once, at most twice the optimal length, and universal: on any
decreasing distribution it codes within a constant factor of the
entropy.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import elias_gamma_decode, elias_gamma_encode, entropy

for n in (1, 2, 3, 4, 7, 8, 100):
    print(f"{n:>4} -> {elias_gamma_encode(n)}")
stream = "".join(elias_gamma_encode(n) for n in (5, 1, 17, 2))
print(f"{stream} -> {elias_gamma_decode(stream)}")

# %%
# Codeword length, and universality
# ----------------------------------

n = np.arange(1, 1025)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(n, [len(elias_gamma_encode(int(k))) for k in n], drawstyle="steps-post", label="Elias gamma")
ax1.plot(n, np.log2(n), label=r"$\log_2 n$")
ax1.set_xscale("log", base=2)
ax1.set_xlabel("n")
ax1.set_ylabel("bits")
ax1.legend()

ratios = []
qs = np.linspace(0.05, 0.95, 19)
for q in qs:  # geometric source P(n) = (1-q) q^(n-1)
    pmf = (1 - q) * q ** (n - 1)
    ratios.append(sum(pmf * [len(elias_gamma_encode(int(k))) for k in n]) / entropy(pmf))
ax2.plot(qs, ratios, "o-")
ax2.set_xlabel("geometric source parameter q")
ax2.set_ylabel("average length / entropy")
ax2.set_title("Bounded overhead without knowing the source")
