r"""
Jacob Bernoulli's law of large numbers: frequencies settle down
===============================================================

Bernoulli's "golden theorem" (1713): in :math:`n` independent trials of
an event with probability :math:`p`, the relative frequency of the event
converges to :math:`p` as :math:`n` grows. Here a biased coin with
:math:`p = 0.3` is tossed a million times, and the running fraction of
heads homes in on 0.3 while its typical error shrinks like
:math:`\sqrt{p(1-p)/n}`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.probability import Binomial
from mathematicskit.probability.systems.limit_theorems import law_of_large_numbers_trace

p = 0.3
coin = Binomial(n=1, p=p)  # one Bernoulli trial: 1 = heads, 0 = tails

# %%
# The relative frequency of heads converges to p
# ------------------------------------------------------------------

n_values = np.unique(np.logspace(0, 6, 400).astype(int))
fig, ax = plt.subplots()
for seed in range(5):
    ax.semilogx(n_values, law_of_large_numbers_trace(coin, n_values, seed=seed), lw=1)
band = 2.0 * np.sqrt(p * (1 - p) / n_values)
ax.semilogx(n_values, p + band, "k--", n_values, p - band, "k--", lw=1)
ax.axhline(p, color="k", lw=1.5)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("number of tosses $n$")
ax.set_ylabel("relative frequency of heads")
ax.set_title(r"Law of large numbers: frequency $\to p = 0.3$ (dashed: $\pm 2$ s.d.)")

# %%
# Bernoulli's statement: :math:`P(|\bar{X}_n - p| > \varepsilon) \to 0`
# ---------------------------------------------------------------------

checkpoints = [10, 100, 1000, 10000, 1000000]
for n, freq in zip(checkpoints, law_of_large_numbers_trace(coin, checkpoints, seed=0)):
    print(f"n={n:>8}: relative frequency = {freq:.4f} (error {abs(freq - p):.4f})")

eps = 0.02
for n in (100, 1000, 10000):
    freqs = Binomial(n=n, p=p).sample(size=20000, seed=1) / n
    print(f"n={n:>6}: P(|frequency - p| > {eps}) ~ {np.mean(np.abs(freqs - p) > eps):.4f}")
