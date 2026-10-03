r"""
Huffman coding (1952)
=====================

Huffman, then a student in Fano's class, found the optimal prefix code
by building the tree bottom-up: repeatedly merge the two least likely
symbols. The resulting code has the smallest possible average length,
between :math:`H` and :math:`H+1` bits, and is never worse than
Shannon-Fano.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import huffman_code, shannon_fano_code
from mathematicskit.information_theory.visualizers import plot_code_tree

source = {"A": 0.35, "B": 0.17, "C": 0.17, "D": 0.16, "E": 0.15}
code = huffman_code(source)
print(code.codewords)
print(f"Huffman {code.average_length:.2f} bits vs Fano {shannon_fano_code(source).average_length:.2f} bits, H = {code.entropy:.3f}")
message = "ABACADAEAB"
bits = code.encode(message)
print(f"{message!r} -> {bits} ({len(bits)} bits) -> {''.join(code.decode(bits))!r}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
plot_code_tree(code, ax=ax1)

# %%
# Redundancy of Huffman vs. Shannon-Fano
# --------------------------------------

rng = np.random.default_rng(0)
sources = [rng.dirichlet(np.full(10, 0.5)) for _ in range(300)]
huff = [huffman_code(p) for p in sources]
fano = [shannon_fano_code(p) for p in sources]
ax2.plot([h.average_length - h.entropy for h in huff], [f.average_length - h.average_length for f, h in zip(fano, huff, strict=True)], "o", ms=3)
ax2.axhline(0, color="k", lw=1)
ax2.set_xlabel("Huffman redundancy: average length - entropy (bits)")
ax2.set_ylabel("Fano length - Huffman length (bits)")
ax2.set_title("300 random sources: Huffman is never longer")
