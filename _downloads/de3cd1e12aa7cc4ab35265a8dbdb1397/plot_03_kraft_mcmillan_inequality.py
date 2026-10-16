r"""
The Kraft-McMillan inequality (1949, 1956)
==========================================

A binary prefix code with codeword lengths :math:`\ell_i` exists if and
only if :math:`\sum_i 2^{-\ell_i} \le 1` (Kraft 1949); McMillan (1956)
showed every uniquely decodable code obeys the same bound. Each
codeword of length :math:`\ell` claims a :math:`2^{-\ell}` share of the
unit interval, and prefix-freeness means the shares cannot overlap.
"""

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mathematicskit.information_theory import canonical_code, kraft_sum

examples = {"complete": [1, 2, 3, 3], "slack": [2, 2, 3, 4], "too short": [1, 2, 2, 3]}
fig, axes = plt.subplots(len(examples), 1, figsize=(9, 4.5), sharex=True)
for ax, (name, lengths) in zip(axes, examples.items(), strict=True):
    total = kraft_sum(lengths)
    ordered = sorted(lengths)
    labels = canonical_code(ordered) if total <= 1 else [f"len {l}" for l in ordered]
    start = 0.0
    for i, (l, label) in enumerate(zip(ordered, labels, strict=True)):
        width = 2.0**-l
        ax.add_patch(Rectangle((start, 0), width, 1, ec="k", fc=f"C{i}", alpha=0.6))
        ax.text(start + width / 2, 0.5, label, ha="center", va="center", fontsize=9)
        start += width
    ax.axvline(1.0, color="k", lw=2)
    ax.set_xlim(0, 1.3)
    ax.set_yticks([])
    verdict = "prefix code exists" if total <= 1 else "no prefix code"
    ax.set_ylabel(name)
    ax.set_title(f"lengths {lengths}: Kraft sum {total:.3f} -> {verdict}", fontsize=10)
axes[-1].set_xlabel("share of the unit interval, 2^-length")
fig.tight_layout()
