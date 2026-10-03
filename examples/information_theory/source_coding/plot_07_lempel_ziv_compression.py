r"""
Lempel-Ziv compression (1977-1978)
==================================

Ziv and Lempel's LZ78 parses a sequence into phrases, each the longest
earlier phrase plus one new symbol, and sends (phrase index, symbol)
pairs. It knows nothing about the source, yet for any stationary
ergodic source its rate approaches the entropy rate: a universal code,
the ancestor of LZW, GIF and ``zip``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import binary_entropy, lz78_compressed_bits, lz78_decode, lz78_encode

text = "ABBABBABBBAABABAA"
pairs = lz78_encode(text)
print(pairs)
print("".join(lz78_decode(pairs)) == text)

# %%
# Rate against sequence length for Bernoulli sources
# --------------------------------------------------

rng = np.random.default_rng(0)
lengths = np.logspace(2, 5.3, 12).astype(int)
fig, ax = plt.subplots(figsize=(7, 4))
for p, color in ((0.05, "C0"), (0.2, "C1"), (0.5, "C2")):
    x = (rng.random(lengths[-1]) < p).astype(int).tolist()
    rates = [lz78_compressed_bits(lz78_encode(x[:n]), 2) / n for n in lengths]
    ax.plot(lengths, rates, "o-", color=color, label=f"p = {p}")
    ax.axhline(binary_entropy(p), color=color, ls="--")
ax.set_xscale("log")
ax.set_xlabel("sequence length")
ax.set_ylabel("LZ78 bits per symbol")
ax.set_title("LZ78 approaches the entropy (dashed) without a model")
ax.legend()
