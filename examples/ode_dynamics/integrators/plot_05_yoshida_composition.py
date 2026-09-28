r"""
Yoshida's symplectic composition: fourth order from leapfrog
============================================================

Yoshida (1990) raised the order of a symmetric method by *composing* it
with itself. Three leapfrog substeps of sizes :math:`w_1 h, w_0 h, w_1 h`
with

.. math::

    w_1 = \frac{1}{2 - 2^{1/3}},\qquad w_0 = -\frac{2^{1/3}}{2 - 2^{1/3}}

cancel the :math:`h^3` error term. The result is fourth order and, being
built from symplectic steps, still symplectic. The middle substep runs
*backwards* in time (:math:`w_0 < 0`).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.integrators import leapfrog_integrate, njit, yoshida4_integrate


@njit
def spring(q, t, params):
    return -q


t_end = 10.0
ns = 2 ** np.arange(5, 12)
hs = t_end / ns
errors = {"leapfrog": [], "Yoshida 4th order": []}
for h, n in zip(hs, ns, strict=False):
    for label, method in (("leapfrog", leapfrog_integrate), ("Yoshida 4th order", yoshida4_integrate)):
        _, qs, _ = method(spring, np.array([1.0]), np.array([0.0]), 0.0, h, n, np.zeros(1))
        errors[label].append(abs(qs[-1, 0] - np.cos(t_end)))

# %%
# Error vs step size
# ------------------

fig, ax = plt.subplots(figsize=(7, 4.5))
for (label, err), cost in zip(errors.items(), (1, 3), strict=False):
    ax.loglog(hs, err, "o-", label=f"{label} ({cost} force evaluation(s) per step)")
    print(f"{label}: error ratio for halved h = {err[-2] / err[-1]:.1f}")
ax.loglog(hs, 0.05 * hs**2, "k--", lw=0.8, label=r"$O(h^2)$")
ax.loglog(hs, 0.01 * hs**4, "k:", lw=0.8, label=r"$O(h^4)$")
ax.set_xlabel("h")
ax.set_ylabel("|q(10) - cos 10|")
ax.set_title("Yoshida composition: slope 4 instead of 2")
ax.legend()
fig.tight_layout()

plt.show()
