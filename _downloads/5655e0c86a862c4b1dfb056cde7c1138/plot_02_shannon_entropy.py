r"""
Shannon entropy (1948)
======================

Shannon replaced Hartley's count of messages with an average over their
probabilities, :math:`H = -\sum p\log_2 p`: the expected surprise of the
next symbol. The binary entropy function peaks at one bit for a fair
coin, and the entropy of English letter frequencies (about 4.2 bits)
falls short of Hartley's :math:`\log_2 26 = 4.7`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import binary_entropy, entropy, hartley_information

# English letter frequencies (percent), A-Z.
english = np.array([8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.15, 0.77, 4.0, 2.4, 6.7, 7.5, 1.9, 0.095, 6.0, 6.3, 9.1, 2.8, 0.98, 2.4, 0.15, 2.0, 0.074])
print(f"English letters: H = {entropy(english):.3f} bits; uniform: {hartley_information(26):.3f} bits")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
p = np.linspace(0, 1, 401)
ax1.plot(p, binary_entropy(p))
ax1.set_xlabel("p")
ax1.set_ylabel("h(p) (bits)")
ax1.set_title("Binary entropy function")

order = np.argsort(english)[::-1]
ax2.bar([chr(65 + i) for i in order], english[order] / english.sum())
ax2.axhline(1 / 26, color="k", ls="--", lw=1, label="uniform")
ax2.set_title(f"English letters: H = {entropy(english):.2f} bits < log2 26 = 4.70")
ax2.legend()
