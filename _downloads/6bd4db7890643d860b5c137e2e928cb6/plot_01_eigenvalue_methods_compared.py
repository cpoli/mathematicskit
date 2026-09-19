r"""
Three routes to a symmetric matrix's eigenvalues
====================================================

:func:`numpy.linalg.eigh` finds the *full* spectrum directly; power
iteration finds only the dominant eigenvalue; inverse iteration finds
whichever eigenvalue is nearest a chosen shift. All three are compared
here on the same matrix.
"""

# %%
import numpy as np

from mathkit.linalg import eigen_symmetric, inverse_iteration, power_iteration

# %%
# Full spectrum: numpy.linalg.eigh
# --------------------------------------------

rng = np.random.default_rng(2)
M = rng.uniform(-2, 2, size=(4, 4))
A = M + M.T  # symmetric

eig_full = eigen_symmetric(A)
print("eigh eigenvalues:", np.sort(eig_full.eigenvalues))

# %%
# Dominant eigenvalue: power iteration
# ------------------------------------------

result_power = power_iteration(A)
print(f"power iteration: dominant eigenvalue ~ {result_power.eigenvalues[0]:.8f} (in {result_power.iterations} iterations)")

# %%
# Eigenvalue nearest a shift: inverse iteration
# ---------------------------------------------------
# Shifting near the second-largest eigenvalue in magnitude picks it out
# directly, without computing the full spectrum.

target = np.sort(eig_full.eigenvalues)[-2]
result_inverse = inverse_iteration(A, mu=target + 0.1)
print(f"inverse iteration near {target + 0.1:.4f}: eigenvalue ~ {result_inverse.eigenvalues[0]:.8f} (target {target:.8f})")
