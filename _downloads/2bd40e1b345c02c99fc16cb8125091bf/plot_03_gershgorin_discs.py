r"""
Gershgorin discs
=====================

Without computing anything but row sums, Gershgorin's theorem traps
every eigenvalue in a union of discs centered on the diagonal entries.
Shrinking the off-diagonal part shrinks the discs onto the eigenvalues.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.linalg import gershgorin_discs

# %%
# Discs and eigenvalues
# ---------------------------

rng = np.random.default_rng(3)
D = np.diag([-4.0, 1.0, 3.0 + 2.0j, 3.0 - 2.0j, 7.0])
E = rng.normal(size=(5, 5)) * 0.6

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for ax, eps in zip(axes, (1.0, 0.3)):
    A = D + eps * E
    discs = gershgorin_discs(A)
    eigs = np.linalg.eigvals(A)
    for c, r in zip(discs.centers, discs.radii):
        ax.add_patch(plt.Circle((c.real, c.imag), r, alpha=0.2))
        ax.add_patch(plt.Circle((c.real, c.imag), r, fill=False))
    ax.plot(eigs.real, eigs.imag, "k*", ms=9, label="eigenvalues")
    ax.plot(discs.centers.real, discs.centers.imag, "r+", ms=9, label="a_kk")
    ax.set_title(f"A = D + {eps} E")
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    print(f"eps = {eps}: all eigenvalues inside the discs? {all(discs.contains(z) for z in eigs)}; max radius {discs.radii.max():.3f}")
axes[0].set_ylabel("Im")
axes[0].legend(loc="lower left")
axes[0].autoscale_view()
fig.tight_layout()

plt.show()
