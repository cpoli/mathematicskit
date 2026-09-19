r"""
Richardson extrapolation dramatically improves accuracy
============================================================

Forward and backward differences are :math:`O(h)`, central differences
:math:`O(h^2)`; Richardson extrapolation combines central differences at
successively halved step sizes to cancel error terms one order at a
time, reaching much higher accuracy without ever needing an
analytically smaller ``h`` (which would eventually be swamped by
floating-point cancellation error).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathkit.calculus.systems.finite_differences import backward_difference, central_difference, forward_difference, richardson_extrapolation

# %%
# Error vs. step size for the three basic differences
# ---------------------------------------------------------

f = np.exp
x0 = 1.0
exact = np.exp(1.0)

hs = np.logspace(-1, -6, 20)
err_fwd = [abs(forward_difference(f, x0, h) - exact) for h in hs]
err_bwd = [abs(backward_difference(f, x0, h) - exact) for h in hs]
err_ctr = [abs(central_difference(f, x0, h) - exact) for h in hs]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(hs, err_fwd, "o-", label="forward, O(h)")
ax.loglog(hs, err_bwd, "s-", label="backward, O(h)")
ax.loglog(hs, err_ctr, "^-", label="central, O(h^2)")
ax.set_xlabel("h")
ax.set_ylabel("|error|")
ax.set_title("Basic finite-difference error vs. step size")
ax.legend()
fig.tight_layout()

# %%
# Richardson extrapolation: much better accuracy at a fixed coarse h
# -------------------------------------------------------------------------

for levels in (1, 2, 3, 4, 5):
    result = richardson_extrapolation(f, x0, h=0.2, levels=levels)
    print(f"levels={levels}: value={result.value:.14f}, error={abs(result.value - exact):.3e}")

plt.show()
