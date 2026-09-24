r"""
Wilkinson's perfidious polynomial
====================================

:math:`w(x) = (x - 1)(x - 2) \cdots (x - 20)` has well separated integer
roots, yet James Wilkinson found that subtracting :math:`2^{-23}` from
its :math:`x^{19}` coefficient, :math:`-210`, sends ten of the roots far
into the complex plane. The root condition numbers explain why: the
roots near 16 amplify a relative change in that coefficient by a
factor of about :math:`3 \times 10^{10}`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.numerical_analysis import root_condition_numbers, wilkinson_polynomial

# %%
# Roots before and after a tiny perturbation
# ---------------------------------------------

w = wilkinson_polynomial(20)
perturbed = w.copy()
perturbed[1] -= 2.0**-23
roots = np.roots(perturbed)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
ax1.scatter(np.arange(1, 21), np.zeros(20), color="black", label="roots of $w$")
ax1.scatter(roots.real, roots.imag, color="firebrick", marker="x", label=r"$a_{19} \to a_{19} - 2^{-23}$")
ax1.set_xlabel("real part")
ax1.set_ylabel("imaginary part")
ax1.set_title("Wilkinson's perturbation experiment")
ax1.legend(fontsize=8)

# %%
# Condition number of each root with respect to :math:`a_{19}`
# ---------------------------------------------------------------

kappa = [root_condition_numbers(w, float(r))[1] for r in range(1, 21)]
ax2.semilogy(np.arange(1, 21), kappa, "o-")
ax2.set_xlabel("root $r$")
ax2.set_ylabel(r"$\kappa_{19}(r) = |a_{19}|\, r^{18} / |w'(r)|$")
ax2.set_title("Relative sensitivity of each root")
fig.tight_layout()

print("roots with |imag| > 0.5:", np.sort_complex(roots[np.abs(roots.imag) > 0.5]).round(3))
print(f"largest condition number: {max(kappa):.2e} at r = {1 + int(np.argmax(kappa))}")

plt.show()
