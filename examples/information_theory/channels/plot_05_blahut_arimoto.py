r"""
The Blahut-Arimoto algorithm (1972)
===================================

Most channels have no closed-form capacity. Blahut and Arimoto
independently found an alternating maximization that converges to it,
with upper and lower bounds that squeeze together at every step. Here
it recovers :math:`1 - h(p)` for the binary symmetric channel and
finds the capacity and skewed optimal input of the Z channel, and
:math:`\log_2 3` for the noisy typewriter (send every other letter).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import blahut_arimoto, bsc_capacity

channels = {
    "BSC(0.1)": [[0.9, 0.1], [0.1, 0.9]],
    "Z channel (1->0 w.p. 0.4)": [[1.0, 0.0], [0.4, 0.6]],
    "noisy typewriter, 6 letters": 0.5 * np.eye(6) + 0.5 * np.roll(np.eye(6), 1, axis=1),
}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (name, W) in zip(axes, channels.items(), strict=True):
    r = blahut_arimoto(W, tol=1e-9)
    k = np.arange(1, r.iterations + 1)
    ax.plot(k, r.upper_bounds, label="upper bound")
    ax.plot(k, r.lower_bounds, label="lower bound")
    ax.set_xscale("log")
    ax.set_title(f"{name}\nC = {r.capacity:.4f}, input {np.round(r.input_distribution, 3)}", fontsize=9)
    ax.set_xlabel("iteration")
axes[0].axhline(bsc_capacity(0.1), color="k", ls=":", label="1 - h(0.1)")
axes[0].legend(fontsize=8)
axes[0].set_ylabel("bits per use")
fig.tight_layout()
