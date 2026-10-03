r"""
Shannon's source coding theorem: the typical set (1948)
=======================================================

For long i.i.d. sequences, :math:`-\tfrac1n\log_2 p(x^n)` concentrates at
the entropy :math:`H` (the asymptotic equipartition property). Almost
all the probability then sits on about :math:`2^{nH}` "typical"
sequences out of :math:`2^n`, so :math:`nH` bits suffice to describe the
source, and no lossless code can do better on average.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import entropy, typical_set

p, eps = [0.85, 0.15], 0.05
H = entropy(p)
ns = np.arange(25, 1001, 25)
results = [typical_set(p, int(n), eps) for n in ns]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(ns, [r.probability for r in results], "o-", ms=3)
ax1.set_xlabel("sequence length n")
ax1.set_ylabel("P(typical set)")
ax1.set_title(f"Typical set probability -> 1 (epsilon = {eps})")

ax2.plot(ns, [r.log2_size / r.n for r in results], "o-", ms=3, label=r"$\frac{1}{n}\log_2 |A_\epsilon^{(n)}|$")
ax2.axhline(H, color="k", ls="--", label=f"H = {H:.3f}")
ax2.axhline(1.0, color="0.6", ls=":", label="all $2^n$ sequences")
ax2.set_xlabel("sequence length n")
ax2.set_ylabel("bits per symbol")
ax2.set_title("About 2^(nH) typical sequences")
ax2.legend()

# %%
# The concentration itself, from samples
# --------------------------------------

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(6, 3.5))
for n in (20, 100, 1000):
    x = rng.random((5000, n)) < p[1]
    surprise = -(x.sum(1) * np.log2(p[1]) + (n - x.sum(1)) * np.log2(p[0])) / n
    ax.hist(surprise, bins=60, density=True, histtype="step", label=f"n = {n}")
ax.axvline(H, color="k", ls="--")
ax.set_xlabel(r"$-\frac{1}{n}\log_2 p(X^n)$")
ax.legend()
