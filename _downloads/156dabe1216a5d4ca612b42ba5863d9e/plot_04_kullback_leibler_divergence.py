r"""
The Kullback-Leibler divergence (1951)
======================================

Kullback and Leibler defined :math:`D(p\,\|\,q) = \sum p\log(p/q)` as
the mean information per observation for telling :math:`p` from
:math:`q`. It is never negative (Gibbs' inequality) and vanishes only
at :math:`p = q`, but it is not symmetric. It is also the extra bits
per symbol paid for coding a :math:`p`-source with a code built for
:math:`q`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import entropy, huffman_code, kl_divergence

q = np.linspace(0.001, 0.999, 400)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for p in (0.1, 0.3, 0.5):
    ax1.plot(q, [kl_divergence([p, 1 - p], [qi, 1 - qi]) for qi in q], label=f"D(p={p} || q)")
ax1.plot(q, [kl_divergence([qi, 1 - qi], [0.3, 0.7]) for qi in q], "k--", label="D(q || 0.3): reversed")
ax1.set_ylim(0, 3)
ax1.set_xlabel("q")
ax1.set_ylabel("bits")
ax1.set_title("Non-negative, zero at p = q, asymmetric")
ax1.legend(fontsize=8)

# %%
# The cost of a wrong model
# -------------------------
# Code a source :math:`p` with ideal lengths :math:`-\log_2 q`: the
# average length is :math:`H(p) + D(p\,\|\,q)`.

rng = np.random.default_rng(0)
p = rng.dirichlet(np.ones(8))
models = [rng.dirichlet(np.ones(8)) for _ in range(40)]
excess = [-(p @ np.log2(m)) - entropy(p) for m in models]
divergence = [kl_divergence(p, m) for m in models]
ax2.plot(divergence, excess, "o")
ax2.plot([0, max(divergence)], [0, max(divergence)], "k--", lw=1)
ax2.set_xlabel("D(p || q)")
ax2.set_ylabel("excess bits per symbol")
ax2.set_title("Mismatched code costs exactly D(p || q)")
print(f"Huffman code for p itself: {huffman_code(p).average_length - entropy(p):.3f} excess bits")
