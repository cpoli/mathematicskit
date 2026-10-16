r"""
Shannon-Fano coding (1948-1949)
===============================

Shannon gave each symbol :math:`\lceil -\log_2 p\rceil` bits, read off
the binary expansion of the cumulative probability; Fano split the
sorted symbols into two halves of nearly equal probability, recursively.
Both land within one bit of the entropy -- the first practical codes
built on Shannon's theory -- but neither is always optimal.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import shannon_fano_code
from mathematicskit.information_theory.visualizers import plot_code_tree

source = {"A": 0.35, "B": 0.17, "C": 0.17, "D": 0.16, "E": 0.15}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, method in zip(axes, ("shannon", "fano"), strict=True):
    code = shannon_fano_code(source, method=method)
    plot_code_tree(code, ax=ax)
    ax.set_title(f"{method.capitalize()}: {code.average_length:.2f} bits (H = {code.entropy:.3f})")
    print(method, code.codewords)

# %%
# Within one bit of the entropy, on random sources
# ------------------------------------------------

rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(6, 4))
for method, marker in (("shannon", "o"), ("fano", "s")):
    codes = [shannon_fano_code(rng.dirichlet(np.ones(8)), method=method) for _ in range(100)]
    ax.plot([c.entropy for c in codes], [c.average_length for c in codes], marker, ms=4, alpha=0.6, label=method)
h = np.linspace(1.5, 3, 2)
ax.plot(h, h, "k-", lw=1)
ax.plot(h, h + 1, "k--", lw=1)
ax.set_xlabel("entropy H")
ax.set_ylabel("average length")
ax.legend()
